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
    支持本地路径或 http(s) URL。
    返回 (base64_str, image_format)；image_format 例如 'jpeg', 'png'
    """
    # 统一用 PIL 打开并转码为 JPEG，保证 MIME 与字节一致，减少体积
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
        # 回落：如果 PIL 打开失败，仍尝试按原始字节发送，但 MIME 与数据可能不匹配，尽量避免走到这里
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
    将 eval_mat 风格的输入转换为 OpenAI API 格式
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
                        # 兼容多种格式（image_url/url/value/b64_json/detail）
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
    使用 vLLM 服务进行聊天推理
    
    参数:
    - messages: 按 eval_mat 风格的图文交错 message 列表
    - model_name: 模型名称（应与 vLLM 服务中的 served-model-name 一致）
    - api_url: vLLM 服务的 API URL
    - api_key: API 密钥（可选，vLLM 默认不需要）
    - timeout: 请求超时时间
    - temperature: 采样温度
    - max_tokens: 最大生成 token 数
    
    返回:
    - str: 模型回答
    """

    port = port or int(os.getenv("VLLM_PORT", 8000))
    host = host or os.getenv("VLLM_HOST", "localhost")
    # print(host, port)
    api_url = f"http://{host}:{port}/v1/chat/completions"
    
    # 转换消息格式
    openai_messages = _convert_to_openai_format(messages)
    
    # 构建请求载荷
    payload = {
        "model": model_name,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # "stop": ["<|im_end|>"]
        **kwargs
    }
    
    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # 发送请求
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout,
            proxies={"http": None, "https": None}
        )
        response.raise_for_status()
        
        # 解析响应
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        # 如果无法解析正常响应，返回完整响应用于调试
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
    chat_gemini 的别名，保持接口一致性
    """
    return chat_gemini(messages, **kwargs)

if __name__ == '__main__':
    # 测试代码
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
