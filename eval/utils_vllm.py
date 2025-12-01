import os
import io
import base64
import imghdr
import requests
from PIL import Image
from typing import Any, Dict, List, Optional
import json

def _base64_encode_image(image_file: str):
    """
    Supports local paths or http(s) URLs.
    Returns (base64_str, image_format); image_format e.g., 'jpeg', 'png'
    """
    # Open via PIL and transcode to JPEG to keep MIME/bytes consistent and reduce size
    try:
        if image_file.startswith('http://') or image_file.startswith('https://'):
            resp = requests.get(image_file, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        else:
            img = Image.open(image_file)

        img = img.convert('RGB')
        w, h = img.size
        max_size = 2048
        if w * h > max_size * max_size:
            scale = (max_size * max_size / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        image_bytes = buf.getvalue()
        image_format = 'jpeg'
    except Exception:
        # Fallback: if PIL fails, still try sending raw bytes; MIME may not match data, avoid this path when possible
        if image_file.startswith('http://') or image_file.startswith('https://'):
            resp = requests.get(image_file, timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            image_format = imghdr.what(None, h=image_bytes) or 'jpeg'
        else:
            with open(image_file, 'rb') as f:
                image_bytes = f.read()
            image_format = imghdr.what(image_file) or 'jpeg'

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_b64, image_format

def _convert_to_openai_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert eval_mat-style input to OpenAI API format
    """
    converted_messages = []
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        
        if role == 'system':
            if isinstance(content, str):
                converted_messages.append({"role": "system", "content": content})
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', '') or item.get('value', '') or '')
                if text_parts:
                    converted_messages.append({"role": "system", "content": " ".join(text_parts)})
        
        elif role in ['user', 'assistant']:
            if isinstance(content, str):
                converted_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                openai_content = []
                
                for item in content:
                    item_type = item.get('type')
                    
                    if item_type == 'text':
                        text_val = item.get('text') or item.get('value') or ''
                        if text_val:
                            openai_content.append({"type": "text", "text": text_val})
                    
                    elif item_type in ['image_url', 'image']:
                        # Support multiple shapes (image_url/url/value/b64_json/detail)
                        url = None
                        detail = item.get('detail')
                        b64_json = None

                        if isinstance(item.get('image_url'), dict):
                            url = item['image_url'].get('url') or item['image_url'].get('value')
                            b64_json = item['image_url'].get('b64_json')
                            detail = item['image_url'].get('detail', detail)

                        url = url or item.get('url') or item.get('value')

                        if not url and not b64_json:
                            continue
                        
                        if b64_json:
                            data_url = f"data:image/jpeg;base64,{b64_json}"
                        elif url.startswith('data:image/'):
                            data_url = url
                        else:
                            b64, fmt = _base64_encode_image(url)
                            data_url = f"data:image/{fmt.lower()};base64,{b64}"
                        
                        image_payload = {"type": "image_url", "image_url": {"url": data_url}}
                        if detail:
                            image_payload["image_url"]["detail"] = detail

                        openai_content.append(image_payload)
                
                if openai_content:
                    converted_messages.append({"role": role, "content": openai_content})
                elif role == 'user':
                    converted_messages.append({"role": role, "content": [{"type": "text", "text": ""}]})
    
    return converted_messages

def chat_gemini(
    messages: List[Dict[str, Any]],
    model_name: str = "qwen2.5-vl",
    host: str = None,
    port: int = None,
    api_key: Optional[str] = None,
    timeout: int = 300,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs
) -> str:
    """
    Use vLLM service for chat inference.
    
    Parameters:
    - messages: Interleaved text-image message list in eval_mat style
    - model_name: Model name (should match served-model-name in vLLM)
    - api_url: vLLM service API URL
    - api_key: API key (optional; vLLM typically does not require)
    - timeout: Request timeout
    - temperature: Sampling temperature
    - max_tokens: Maximum generation tokens
    
    Returns:
    - str: Model answer
    """

    port = port or int(os.getenv("VLLM_PORT", 8000))
    host = host or os.getenv("VLLM_HOST", "localhost")
    # print(host, port)
    api_url = f"http://{host}:{port}/v1/chat/completions"
    
    # Convert message format
    openai_messages = _convert_to_openai_format(messages)
    
    # Build request payload
    payload = {
        "model": model_name,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # "stop": ["<|im_end|>"]
        **kwargs
    }
    
    # Set request headers
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Send request
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout,
            proxies={"http": None, "https": None}
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        # If unable to parse a normal response, return full response for debugging
        raise ValueError(f"Unexpected response format: {data}")
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

def chat_vllm(
    messages: List[Dict[str, Any]], 
    **kwargs
) -> str:
    """
    Alias of chat_gemini to keep interface consistency
    """
    return chat_gemini(messages, **kwargs)

if __name__ == '__main__':
    # Test code
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please compare the difference between these two images."},
                # {"type": "image_url", "image_url": {"url": "/path/to/your/image1.jpg", "detail": "low"}},
                # {"type": "image_url", "image_url": {"url": "/path/to/your/image2.png", "detail": "low"}},
            ],
        },
    ]
    
    try:
        # or set os.environ["VLLM_PORT"] = "8000"
        # response = chat_vllm(messages, port=9001, host="29.210.133.11")
        response = chat_vllm(messages, port=os.environ["VLLM_PORT"])
        print(response)
    except Exception as e:
        print(f"Error: {e}")
