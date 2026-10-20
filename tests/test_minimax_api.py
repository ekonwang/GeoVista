"""Offline transport tests for the evaluator's MiniMax chat helper."""

import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import httpx
import pytest


@pytest.fixture
def api(monkeypatch):
    # These imports are unrelated to chat requests and require the training stack.
    # Keep the real SDK and its serialization; only the HTTP transport is mocked.
    optional = {name: ModuleType(name) for name in ("numpy", "torch", "requests", "tqdm", "PIL", "utils")}
    optional["tqdm"].tqdm = lambda value: value
    optional["PIL"].Image = object()
    optional["utils"].print_hl = print
    optional["utils"].print_error = print
    for name, module in optional.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-test")
    monkeypatch.delenv("MINIMAX_REGION", raising=False)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    requests = []
    modules = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Test answer"},
                         "finish_reason": "stop"}],
        })

    transport = httpx.MockTransport(respond)

    def load(region=None, base_url=None):
        if region is not None:
            monkeypatch.setenv("MINIMAX_REGION", region)
        if base_url is not None:
            monkeypatch.setenv("MINIMAX_API_BASE", base_url)
        spec = importlib.util.spec_from_file_location("evaluator_api", Path(__file__).parents[1] / "eval/utils_api.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(multiprocessing, "set_start_method"):
            spec.loader.exec_module(module)
        modules.append(module)
        return module, requests

    # The real module constructs the real SDK clients, including regional URLs.
    with patch.object(httpx.HTTPTransport, "handle_request", transport.handle_request):
        yield load
    for module in modules:
        module.client.close()
        module.minimax_client.close()


@pytest.mark.parametrize(("region", "base_url", "expected_url"), [
    (None, None, "https://api.minimax.io/v1/chat/completions"),
    ("global_en", None, "https://api.minimax.io/v1/chat/completions"),
    ("cn_zh", None, "https://api.minimaxi.com/v1/chat/completions"),
    ("unknown", None, "https://api.minimax.io/v1/chat/completions"),
    ("cn_zh", "https://custom.example/v1", "https://custom.example/v1/chat/completions"),
])
def test_default_serialized_request(api, region, base_url, expected_url):
    module, requests = api(region, base_url)
    messages = [{"role": "user", "content": "Locate this landmark."}]
    assert module.chat_minimax(messages) == "Test answer"
    assert len(requests) == 1
    assert str(requests[0].url) == expected_url
    assert requests[0].method == "POST"
    # Omitting thinking/effort uses adaptive thinking and max effort for Flash.
    # No unsupported service_tier or reasoning_split=false is injected.
    assert json.loads(requests[0].content) == {
        "model": "MiniMax-M3.1-Flash", "messages": messages, "max_completion_tokens": 10240,
    }


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
def test_explicit_models_and_multimodal_history_are_preserved(api, model):
    module, requests = api()
    messages = [
        {"role": "assistant", "content": "Previous answer", "reasoning_content": "Previous reasoning"},
        {"role": "user", "content": [
            {"type": "text", "text": "Locate this landmark."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
            {"type": "video_url", "video_url": {"url": "mm_file://test-video", "fps": 1}},
        ]},
    ]
    assert module.chat_minimax(messages, model_name=model, max_completion_tokens=2048) == "Test answer"
    assert json.loads(requests[0].content) == {
        "model": model, "messages": messages, "max_completion_tokens": 2048,
    }


def test_curated_models(api):
    module, _ = api()
    assert list(module.MINIMAX_MODELS) == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    flash = module.MINIMAX_MODELS[module.MINIMAX_MODEL_NAME]
    assert flash["context_window"] == 1_000_000
    assert flash["max_output_tokens"] == 524_288
    assert flash["thinking"] == ["adaptive"]
    assert flash["reasoning_effort"] == ["low", "medium", "high", "xhigh", "max"]
    assert flash["default_reasoning_effort"] == "max"
    assert flash["pricing_usd_per_million_tokens"] == {
        "input": 0.15, "output": 0.6, "cache_read": 0.015, "cache_write": None,
    }
    m3 = module.MINIMAX_MODELS["MiniMax-M3"]
    assert m3["thinking"] == ["adaptive", "disabled"]
    assert m3["pricing_usd_per_million_tokens"] == {
        "input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": None,
    }
    assert m3["pricing_above_512k_usd_per_million_tokens"] == {
        "input": 0.6, "output": 2.4, "cache_read": 0.12, "cache_write": None,
    }


def test_existing_chat_helper_is_unchanged(api):
    module, requests = api()
    messages = [{"role": "user", "content": "Locate this landmark."}]
    assert module.chat_gpt5_nano(messages) == "Test answer"
    assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
    assert json.loads(requests[0].content) == {
        "model": "gpt-5-nano", "messages": messages, "max_completion_tokens": 10240,
    }
