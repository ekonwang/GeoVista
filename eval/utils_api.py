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


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = chat_gpt5_nano(messages)
    print_hl(response)
