#!/usr/bin/env python3
"""Minimal example runner for GeoVista agent-mode inference with colored prints.

Usage:
  python examples/infer_example.py --multimodal_input examples/geobench-example.png \
      --question "Please analyze where is the place."
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Allow importing sibling modules from repo root and eval helpers directly
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval.inference_agent_tool_mode import (  # type: ignore
    MAX_CROP_CALLS,
    MAX_ROUNDS,
    MAX_SEARCH_CALLS,
    SYSTEM_PROMPT,
    TOOL_RESPONSE_PREFIX,
    TOOL_RESPONSE_SUFFIX,
)
from eval.utils_agent_tool import (  # type: ignore
    ToolCallManager,
    _log_kv,
    _summarize_mm_content,
    dump_tool_call,
    encode_image_to_base64,
    extract_tool_calls,
    get_image_resolution,
    reformat_response,
)
from eval.utils_vllm import chat_vllm  # type: ignore



from utils import print_hl  # type: ignore

# Simple ANSI colors
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"
TOOL_LINE = "# ----------- #"

TOOL_RESPONSE_RE = re.compile(
    f"{re.escape(TOOL_RESPONSE_PREFIX)}(.*?){re.escape(TOOL_RESPONSE_SUFFIX)}",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _print_think(text: str) -> None:
    if text.strip():
        print(f"{BLUE}{text.strip()}{RESET}")


def _print_tool_response(text: str) -> None:
    body = text.strip()
    print(TOOL_LINE)
    print(f"{ORANGE}{body}{RESET}")
    print(TOOL_LINE)


def _print_plain(text: str) -> None:
    if text.strip():
        print(text.strip())


def _render_text_chunks(text: str) -> None:
    # First handle tool responses, then handle think blocks inside remaining text.
    cursor = 0
    for match in TOOL_RESPONSE_RE.finditer(text or ""):
        prefix = text[cursor: match.start()]
        if prefix:
            _render_think_and_plain(prefix)
        _print_tool_response(match.group(1))
        cursor = match.end()
    if cursor < len(text):
        _render_think_and_plain(text[cursor:])


def _render_think_and_plain(chunk: str) -> None:
    pos = 0
    for m in THINK_RE.finditer(chunk or ""):
        pre = chunk[pos: m.start()]
        if pre:
            _print_plain(pre)
        _print_think(m.group(1))
        pos = m.end()
    if pos < len(chunk):
        _print_plain(chunk[pos:])


def _render_message(role: str, content: Any) -> None:
    print_hl(f"{role.capitalize()} >")
    if isinstance(content, str):
        _render_text_chunks(content)
        return

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                _print_plain(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                _render_text_chunks(str(item.get("text", "")))
            elif item_type == "image_url":
                img_url = item.get("image_url", {}).get("url")
                _print_plain(f"[image] {img_url}")
            else:
                _print_plain(str(item))
    else:
        _print_plain(str(content))


def pretty_print_conversation(messages: List[Dict[str, Any]]) -> None:
    prev_role = None
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if prev_role and role in {"user", "assistant"} and role != prev_role:
            print("\n" * 3)
        _render_message(role or "unknown", msg.get("content"))
        prev_role = role


def run_single_image(
    image_path: str,
    question: str,
    temp_dir: str,
    host: str,
    port: int,
    api_key: str,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    manager = ToolCallManager(image_path, temp_dir)

    width, height = get_image_resolution(image_path)
    image_b64 = encode_image_to_base64(image_path)

    # Compute adaptive scaling between original and the resized image that the model sees
    try:
        decoded = base64.b64decode(image_b64)
        from PIL import Image

        with Image.open(io.BytesIO(decoded)) as img_resized:
            resized_w, resized_h = img_resized.size
    except Exception:
        resized_w, resized_h = width, height

    adaptive_scaling = (width / resized_w) if resized_w else 1.0

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": question},
            ],
        },
    ]
    printable_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": question},
            ],
        },
    ]

    if debug:
        _log_kv(
            True,
            "Init",
            {
                "image_path": image_path,
                "resolution": [width, height],
                "resized_resolution": [resized_w, resized_h],
                "adaptive_scaling": adaptive_scaling,
                "temp_dir": manager.temp_dir,
                "MAX_ROUNDS": MAX_ROUNDS,
                "MAX_CROP_CALLS": MAX_CROP_CALLS,
                "MAX_SEARCH_CALLS": MAX_SEARCH_CALLS,
            },
        )
        print_hl("User >")
        for line in _summarize_mm_content(printable_messages[-1]["content"]):
            print(line)

    def _wrap_tool_response_text(text: str) -> str:
        return f"{TOOL_RESPONSE_PREFIX}{text}{TOOL_RESPONSE_SUFFIX}"

    while manager.round_count < MAX_ROUNDS:
        try:
            print_hl(f"Round {manager.round_count + 1} | calling model")
            _log_kv(
                debug,
                "Manager state before call",
                {
                    "round_count": manager.round_count,
                    "crop_calls": manager.crop_call_count,
                    "search_calls": manager.search_call_count,
                },
            )

            response = chat_vllm(
                messages,
                host=host,
                port=port,
                api_key=api_key or None,
            )

            print_hl("Assistant >")
            print(response if isinstance(response, str) else str(response))

            messages.append({"role": "assistant", "content": reformat_response(response)})
            printable_messages.append({"role": "assistant", "content": response})
            manager.increment_round()

            tool_calls = extract_tool_calls(response)
            user_content: List[Dict[str, Any]] = []
            print_user_content: List[Dict[str, Any]] = []

            if tool_calls:
                tool_calls = tool_calls[:1]
                for call in tool_calls:
                    name = call.get("name")
                    arguments = call.get("arguments", {})

                    try:
                        if name == "image_zoom_in_tool":
                            bbox_2d = arguments.get("bbox_2d")
                            label = arguments.get("label")
                            result = manager.execute_crop_tool(
                                bbox_2d,
                                label,
                                debug=False,
                                abs_scaling=adaptive_scaling,
                            )

                            base_text = (
                                "For the image, You have zoomed in on the following area: "
                                f"{dump_tool_call(call)}, and the cropped image is as follows:"
                            )
                            user_content.append({"type": "text", "text": f"{TOOL_RESPONSE_PREFIX}{base_text}"})
                            user_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{result['crop_b64']}"},
                                }
                            )
                            user_content.append({"type": "text", "text": TOOL_RESPONSE_SUFFIX})

                            print_user_content.append({"type": "text", "text": f"{TOOL_RESPONSE_PREFIX}{base_text}"})
                            print_user_content.append({"type": "image_url", "image_url": {"url": result["crop_path"]}})
                            print_user_content.append({"type": "text", "text": TOOL_RESPONSE_SUFFIX})

                        elif name == "search_web":
                            query = arguments.get("query")
                            result = manager.execute_search_tool(query, debug=False)

                            search_result_json = json.dumps(
                                {
                                    "name": "search_web",
                                    "query": result["query"],
                                    "result": result["results"],
                                },
                                ensure_ascii=False,
                                indent=2,
                            )

                            result_text = f"For {dump_tool_call(call)}, the search results are as follows:\n{search_result_json}"
                            user_content.append({"type": "text", "text": _wrap_tool_response_text(result_text)})
                            print_user_content.append({"type": "text", "text": _wrap_tool_response_text(result_text)})

                    except Exception as exc:
                        tb_lines = traceback.format_exc().strip().split("\n")
                        key_info = "\n".join(tb_lines[-6:]) if len(tb_lines) > 6 else traceback.format_exc()
                        err_msg = {
                            "type": "text",
                            "text": _wrap_tool_response_text(
                                f"Tool call error: {str(exc)}\n\nError details:\n{key_info}"
                            ),
                        }
                        user_content.append(err_msg)
                        print_user_content.append(err_msg)
                        _log_kv(debug, "Tool call error", {"error": str(exc)})

            if manager.should_force_final_answer():
                msg = {
                    "type": "text",
                    "text": "Now you must try to identify the place where the original image is located, without more tool uses.",
                }
                user_content.append(msg)
                print_user_content.append(msg)
            elif manager.should_prompt_search():
                msg = {
                    "type": "text",
                    "text": "You should consider using web search tool to find more information about the location.",
                }
                user_content.append(msg)
                print_user_content.append(msg)

            if user_content:
                if debug:
                    print_hl("User >")
                    for line in _summarize_mm_content(print_user_content):
                        print(line)
                messages.append({"role": "user", "content": user_content})
                printable_messages.append({"role": "user", "content": print_user_content})
            else:
                if debug:
                    _log_kv(True, "End", {"reason": "No tool calls produced"})
                break

        except Exception as exc:
            tb = traceback.format_exc()
            err_text = f"error: {str(exc)}"
            messages.append({"role": "user", "content": [{"type": "text", "text": err_text + "\n" + tb}]})
            printable_messages.append({"role": "user", "content": [{"type": "text", "text": err_text + "\n" + tb}]})
            _log_kv(debug, "Caught exception in round", {"error": str(exc) + "\n" + tb})
            break

    _log_kv(
        debug,
        "Conversation finished",
        {
            "total_rounds": manager.round_count,
            "crop_calls": manager.crop_call_count,
            "search_calls": manager.search_call_count,
        },
    )
    return printable_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single GeoVista example inference")
    parser.add_argument("--multimodal_input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="User question for the model")
    parser.add_argument("--temp_dir", type=str, default=".temp/outputs/examples", help="Directory for temp crops")
    parser.add_argument("--host", type=str, default=os.getenv("VLLM_HOST", "localhost"), help="vLLM host")
    parser.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", 8000)), help="vLLM port")
    parser.add_argument("--api_key", type=str, default=os.getenv("VLLM_API_KEY"), help="Optional API key")
    parser.add_argument("--debug", action="store_true", help="Print verbose debug info")
    args = parser.parse_args()

    messages = run_single_image(
        image_path=args.multimodal_input,
        question=args.question,
        temp_dir=args.temp_dir,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        debug=args.debug,
    )

    pretty_print_conversation(messages)


if __name__ == "__main__":
    main()
