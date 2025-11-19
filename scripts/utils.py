import json
import os, sys
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track, Progress
from time import sleep
# from utils_execution import SandboxCodeExecutor
from PIL import Image
import re, json, ast

# llm_config={"cache_seed": None, "config_list": [{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "temperature": 0.0, "api_key": "sk-gjezftinzvhzoogekwilcnydixgooycpezqemudmnttqbycj", "base_url": "https://api.siliconflow.cn/v1"}]}
llm_config={"cache_seed": None, "config_list": [{"model": "Pro/deepseek-ai/DeepSeek-V3", "temperature": 0.0, "api_key": "sk-gjezftinzvhzoogekwilcnydixgooycpezqemudmnttqbycj", "base_url": "https://api.siliconflow.cn/v1"}]}

from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
)
from autogen.oai.client import OpenAIWrapper


def get_tool_calls(response_message: str):
    """
    从模型输出中提取多个 <tool_call>...</tool_call>，宽松解析为动作字典列表。
    解析顺序：清洗 → json.loads → ast.literal_eval → 正则兜底字段抽取。
    永不抛异常；解析失败的块会被跳过。
    """
    def _cleanup(s: str) -> str:
        # 去掉代码围栏/语言标签
        s = re.sub(r"```(?:json|python)?", "", s, flags=re.IGNORECASE).replace("```", "")
        # 归一化花括号
        s = s.replace("{{", "{").replace("}}", "}")
        # 智能引号归一化
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        # 去掉行首行尾空白
        s = s.strip()
        # 删除尾随逗号
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _loads_loose(s: str):
        # 尝试 JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # 尝试 Python 字面量（将 true/false/null 转为 Python）
        s2 = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
        s2 = re.sub(r"\bfalse\b", "False", s2, flags=re.IGNORECASE)
        s2 = re.sub(r"\bnull\b", "None", s2, flags=re.IGNORECASE)
        try:
            return ast.literal_eval(s2)
        except Exception:
            return None

    def _fallback_parse(s: str):
        # 兜底正则抽取字段
        def _pick(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            m = re.search(pattern, text, flags)
            return m.group(1).strip() if m else None

        # id
        id_str = _pick(r'"id"\s*:\s*(\d+)|\bid\s*:\s*(\d+)', s)
        if id_str is None:
            # 没 id 就放弃这个块
            return None
        id_val = int(id_str)

        # name
        name = _pick(r'"name"\s*:\s*"([^"]+)"|\bname\s*:\s*\'([^\']+)\'', s) or "image_zoom_in_tool"

        # bbox_2d
        bbox_txt = _pick(r'"bbox_2d"\s*:\s*\[([^\]]+)\]|\bbbox_2d\s*:\s*\[([^\]]+)\]', s)
        query_txt = _pick(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"|\bquery\s*:\s*\'((?:[^\'\\]|\\.)*)\'', s)

        reason = _pick(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"|\breason\s*:\s*\'((?:[^\'\\]|\\.)*)\'', s)

        args = {}
        if bbox_txt:
            nums = [n.strip() for n in bbox_txt.split(",")]
            try:
                args["bbox_2d"] = [int(float(x)) for x in nums[:4]]
            except Exception:
                pass
        if query_txt and ("bbox_2d" not in args):
            # 反转义
            args["query"] = bytes(query_txt, "utf-8").decode("unicode_escape")

        if not args:
            return None

        action = {"id": id_val, "name": name, "arguments": args}
        if reason:
            action["reason"] = bytes(reason, "utf-8").decode("unicode_escape")
        return action

    actions = []
    # 提取所有 tool_call 块
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", response_message, re.IGNORECASE | re.DOTALL):
        raw = m.group(1).strip()
        cleaned = _cleanup(raw)

        # 有些模型会输出一个数组包住多个对象
        candidate = _loads_loose(cleaned)
        if candidate is not None:
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        actions.append(item)
            elif isinstance(candidate, dict):
                actions.append(candidate)
            else:
                # 非 dict/list，继续兜底
                fb = _fallback_parse(cleaned)
                if fb:
                    actions.append(fb)
            continue

        # 结构化失败 → 兜底字段抽取
        fb = _fallback_parse(cleaned)
        if fb:
            actions.append(fb)

    return actions


def chat_vlm(prompt: str, history_messages = None, retry_times: int = 10):
    interval = 1
    for i in range(retry_times):
        try:
            if history_messages is None:
                history_messages = []
            clean_messages = history_messages + [{"role": "user", "content":  prompt}]
            dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
            
            client = OpenAIWrapper(**llm_config)
            response = client.create(
                messages=dirty_messages,
                timeout=600,
            )
            messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
            return response.choices[0].message.content, messages
        except Exception as e:
            if 'limit' in str(e):
                sleep(interval)
                interval = min(interval * 2, 60)
            print_error(e)
            if i >= (retry_times - 1):
                raise e


def mk_pbar(iterable, ncols=80, **kwargs):
    # check if iterable
    if not hasattr(iterable, '__iter__'):
        raise ValueError("Input is not iterable.")
    return tqdm(iterable, ncols=ncols, **kwargs)


# def mk_len_pbar(ncols=80, **kwargs):
def mk_len_pbar(iterable, func, **kwargs):
    # use rich progress bar
    # return track(list(range(total)), **kwargs)
    with Progress(**kwargs) as progress:
        # Create a task with an initial description
        task_id = progress.add_task("[red]Processing...", total=len(iterable))
        for i, elem in enumerate(iterable):
            results = func(elem)
            progress.update(task_id, advance=1)
            yield results


def generate_uuid():
    return str(uuid4())


def _print_with_color(message, color):
    if color == 'red':
        print(f"\033[91m\033[1m{message}\033[0m")
    elif color == 'green':
        print(f"\033[92m\033[1m{message}\033[0m")
    elif color == 'yellow':
        print(f"\033[93m\033[1m{message}\033[0m")
    elif color == 'blue':
        print(f"\033[94m\033[1m{message}\033[0m")
    else:
        raise ValueError(f"Invalid color: {color}")


def print_error(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    _print_with_color(message, 'red')
    sys.stdout.flush()


def print_hl(message):
    """Highlight the message with blue color"""
    _print_with_color(message, 'blue')
    sys.stdout.flush()


def load_jsonl(path):
    data = []
    with open(path, "rt") as f:
        for line in mk_pbar(f):
            data.append(json.loads(line.strip()))
    return data

def load_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def save_jsonl(data, path, mode='w', use_tqdm=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        if use_tqdm:
            data = mk_pbar(data)
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wt") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def tag_join(valid_tags, sep=','):
    return sep.join([str(tag) for tag in valid_tags])


def mean_list(data_list):
    data_list = [float(d) for d in data_list if d is not None]
    return sum(data_list) / len(data_list)


def log_print(*content, **kwargs):
    # set datetime timezone to Shanghai.
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    content = [f'[{datetime.datetime.now()}]'] + list(content)
    print(*content, **kwargs)
    sys.stdout.flush()


def json_print(data):
    print(json.dumps(data, ensure_ascii=False, indent=4))


def multithreading(func, thread_num=8, data=None):
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(func, data)

# def code_to_image(code: str):
#     executor = SandboxCodeExecutor()
#     results = executor.execute(code)
#     if results[0]:
#         raise Exception(f'{results[1]}')

#     results = results[2]
#     image_path = results[0]
#     image = Image.open(image_path)
#     return image

    
if __name__ == '__main__':
    pstr = """
<tool_call>
{"id": 1, "name": "image_zoom_in_tool", "arguments": {"bbox_2d": [1400, 400, 2000, 600]}, "reason": "This bounding box targets the building to the left of "LA NUEVA SEGUROS". Its facade might display signage with address or business name information that could help pinpoint a location. The given coordinates approximate the building's visible front, from roughly the mid-left section of the panorama. The vertical coordinates are chosen to include the building's lower floors, where such signs are commonly placed, considering the distortion towards the top of the panoramic view. The width accounts for the potential spread of any text across the building's face."}
</tool_call>
"""
    actions = get_tool_calls(pstr)
    print(actions)
