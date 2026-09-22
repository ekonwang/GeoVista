import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
import uuid
from pathlib import Path
from openai import OpenAI
import requests

from utils import print_hl, print_error


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
)
MODEL_NAME = 'gpt-5-nano'
MAX_COMPLETION_TOKENS = 10240

def chat_gpt5_nano(messages, model_name=MODEL_NAME, max_completion_tokens=MAX_COMPLETION_TOKENS):
    params = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


# MiniMax provider configuration.
#
# The evaluator's chat client is OpenAI-compatible, so MiniMax models are reached
# through the MiniMax OpenAI-compatible chat completions endpoint. Two regional
# endpoints are available; the global endpoint is used by default and can be
# overridden with MINIMAX_REGION or, directly, with MINIMAX_API_BASE.
MINIMAX_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
MINIMAX_DEFAULT_REGION = os.environ.get("MINIMAX_REGION", "global_en")
MINIMAX_API_BASE = os.environ.get(
    "MINIMAX_API_BASE",
    MINIMAX_ENDPOINTS.get(MINIMAX_DEFAULT_REGION, MINIMAX_ENDPOINTS["global_en"]),
)

# Default MiniMax text model and the models this evaluator knows how to configure.
MINIMAX_MODEL_NAME = "MiniMax-M3"
MINIMAX_MODELS = {
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
        "pricing_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
    },
    "MiniMax-M2.7": {
        "context_window": 204_800,
        "input_modalities": ["text"],
        "thinking": ["always_on"],
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
    },
}

minimax_client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
    base_url=MINIMAX_API_BASE,
)


def chat_minimax(messages, model_name=MINIMAX_MODEL_NAME, max_completion_tokens=MAX_COMPLETION_TOKENS):
    params = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    response = minimax_client.chat.completions.create(**params)
    return response.choices[0].message.content


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = chat_gpt5_nano(messages)
    print_hl(response)
