#!/usr/bin/env python3
import os
import json
import argparse
import sys
import uuid
import base64
import io
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from pathlib import Path
import traceback
import shutil
import concurrent.futures
import threading

from datasets import load_dataset

# from utils_gpt import chat_gemini
from utils_vllm import chat_vllm as chat_gemini

from gpt_researcher.search_worker import run_search
from utils import print_hl

# Constants
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.MULTILINE)
MAX_CROP_CALLS = 5
MAX_SEARCH_CALLS = 2
MAX_ROUNDS = 6
IMAGE_FACTOR = 28
MIN_PIXELS = 256 * 256
MAX_PIXELS = 2048 * 1024
TOOL_RESPONSE_PREFIX = "<tool_response>"
TOOL_RESPONSE_SUFFIX = "</tool_response>"

SYSTEM_PROMPT = """
You are a helpful assistant.
Answer the user's question based on the image provided.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Tool definition

## Image crop and zoom in tool
<tools>
{
"name": "image_zoom_in_tool",
"description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).",
"parameters": {
    "properties": {
        "bbox_2d": {
            "type": "array",
            "description": "The bounding box as [x1, y1, x2, y2] on the original image."
        }
    },
    "required": ["bbox_2d"]
}
}
</tools>

## Search web tool
<tools>
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Execute a web search and return normalized results containing titles, snippets, and URLs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query string."
        }
      },
      "required": ["query"]
    }
  }
}
</tools>


**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200]}}  
</tool_call>
<tool_call>
{"name": "search_web", "arguments": {"query": "The palace museum"}}
</tool_call>

When you call a tool, think first. Place your internal reasoning inside <think>...</think>, followed by the <tool_call>...</tool_call>. And when you are ready to answer, also think first. Place your internal reasoning inside <think>...</think>, followed by the <answer>...</answer>.
"""


from utils_agent_tool import ToolCallManager 
from utils_agent_tool import dump_tool_call, encode_image_to_base64, get_image_resolution, extract_tool_calls, reformat_response, _log, _log_kv, _summarize_mm_content


def _load_geobench_records(
    dataset_id: str,
    split: str,
    cache_dir: Optional[str],
    temp_dir: Optional[str],
    hf_token: Optional[str],
    limit: Optional[int],
    debug: bool,
) -> List[Dict[str, Any]]:
    """Load GeoBench samples and persist images to disk for downstream inference."""
    try:
        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=cache_dir,
            # use_auth_token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset {dataset_id}:{split} via datasets.load_dataset: {exc}"
        ) from exc

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        if limit is not None and len(records) >= limit:
            break

        metadata_raw = sample.get("metadata")
        metadata_obj: Dict[str, Any] = {}
        if isinstance(metadata_raw, str):
            try:
                metadata_obj = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata_obj = {"raw_metadata": metadata_raw}
        elif isinstance(metadata_raw, dict):
            metadata_obj = metadata_raw

        uid = metadata_obj.get("uid") or sample.get("uid") or f"sample_{idx}"

        image_info = sample.get("raw_image_path")
        if not os.path.exists(image_info):
            image_path = os.path.join(dataset_id, image_info)
        else:
            image_path = image_info
        assert os.path.exists(image_path), f"Image path does not exist: {image_path}, please check the dataset {dataset_id}:{split} sanity, and ensure images are properly downloaded (https://huggingface.co/datasets/LibraTree/GeoVistaBench)."

        records.append({
            "uid": uid,
            "image_path": image_path,
            "metadata": metadata_obj,
        })

    if not records:
        raise RuntimeError(f"No usable samples were retrieved from {dataset_id}:{split}.")

    return records


def run_inference_for_image(image_path: str, temp_dir: str, debug: bool = False) -> List[Dict[str, Any]]:
    """Run inference for a single image."""
    manager = ToolCallManager(image_path, temp_dir)
    
    width, height = get_image_resolution(image_path)
    
    image_b64 = encode_image_to_base64(image_path)

    # Compute adaptive scaling between original and the resized image that the model sees
    try:
        _decoded = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(_decoded)) as _img_resized:
            resized_w, resized_h = _img_resized.size
    except Exception:
        # Fallback to original if decode fails (no scaling)
        resized_w, resized_h = width, height

    adaptive_scaling = (width / resized_w) if resized_w else 1.0
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": f"Please analyze where is the place."}
        ]}
    ]
    print_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": f"Please analyze where is the place."}
        ]}
    ]

    if debug:
        _log_kv(True, "Init", {
            "image_path": image_path,
            "resolution": [width, height],
            "resized_resolution": [resized_w, resized_h],
            "adaptive_scaling": adaptive_scaling,
            "temp_dir": manager.temp_dir,
            "MAX_ROUNDS": MAX_ROUNDS,
            "MAX_CROP_CALLS": MAX_CROP_CALLS,
            "MAX_SEARCH_CALLS": MAX_SEARCH_CALLS
        })
        print_hl("User >")
        for line in _summarize_mm_content(print_messages[-1]["content"]):
            print(line)

    def _wrap_tool_response_text(text: str) -> str:
        return f"{TOOL_RESPONSE_PREFIX}{text}{TOOL_RESPONSE_SUFFIX}"

    while manager.round_count < MAX_ROUNDS:
        try:
            if debug:
                print_hl(f"Round {manager.round_count+1} | calling model")
            _log_kv(debug, "Manager state before call", {
                "round_count": manager.round_count,
                "crop_calls": manager.crop_call_count,
                "search_calls": manager.search_call_count
            })
            response = chat_gemini(messages)
            if debug:
                print_hl("Assistant >")
                print(response if isinstance(response, str) else str(response))
            messages.append({"role": "assistant", "content": reformat_response(response)})
            print_messages.append({"role": "assistant", "content": response})
            manager.increment_round()
            
            tool_calls = extract_tool_calls(response)
            user_content = []
            print_user_content = []
            
            if tool_calls:
                tool_calls = tool_calls[:1]
                for call in tool_calls:
                    try:
                        name = call.get("name")
                        arguments = call.get("arguments", {})
                        
                        if name == "image_zoom_in_tool":
                            bbox_2d = arguments.get("bbox_2d")
                            label = arguments.get("label")
                            # Use adaptive scaling: map bbox to original image space
                            result = manager.execute_crop_tool(bbox_2d, label, debug=False, abs_scaling=adaptive_scaling)
                            
                            base_text = f"For the image, You have zoomed in on the following area: {dump_tool_call(call)}, and the cropped image is as follows:"
                            user_content.append({
                                "type": "text",
                                "text": f"{TOOL_RESPONSE_PREFIX}{base_text}"
                            })
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{result['crop_b64']}"}
                            })
                            user_content.append({
                                "type": "text",
                                "text": TOOL_RESPONSE_SUFFIX
                            })
                            print_user_content.append({
                                "type": "text",
                                "text": f"{TOOL_RESPONSE_PREFIX}{base_text}"
                            })
                            print_user_content.append({
                                "type": "image_url",
                                "image_url": {"url": result["crop_path"]}
                            })
                            print_user_content.append({
                                "type": "text",
                                "text": TOOL_RESPONSE_SUFFIX
                            })
                            
                        elif name == "search_web":
                            query = arguments.get("query")
                            result = manager.execute_search_tool(query, debug=False)
                            
                            search_result_json = json.dumps({
                                "name": "search_web",
                                "query": result["query"],
                                "result": result["results"]
                            }, ensure_ascii=False, indent=2)
                            
                            result_text = f"For {dump_tool_call(call)}, the search results are as follows:\n{search_result_json}"
                            user_content.append({
                                "type": "text",
                                "text": _wrap_tool_response_text(result_text)
                            })
                            print_user_content.append({
                                "type": "text",
                                "text": _wrap_tool_response_text(result_text)
                            })
                            
                    except Exception as e:
                        tb_lines = traceback.format_exc().strip().split('\n')
                        key_info = '\n'.join(tb_lines[-6:]) if len(tb_lines) > 6 else traceback.format_exc()
                        err_msg = {
                            "type": "text",
                            "text": _wrap_tool_response_text(
                                f"Tool call error: {str(e)}\n\nError details:\n{key_info}"
                            )
                        }
                        user_content.append(err_msg)
                        print_user_content.append(err_msg)
                        if debug:
                            _log_kv(True, "Tool call error", {"error": str(e)})
            
            if manager.should_force_final_answer():
                msg = {
                    "type": "text",
                    "text": "Now you must try to identify the place where the original image is located, without more tool uses."
                }
                user_content.append(msg)
                print_user_content.append(msg)

            elif manager.should_prompt_search():
                msg = {
                    "type": "text",
                    "text": "You should consider using web search tool to find more information about the location."
                }
                user_content.append(msg)
                print_user_content.append(msg)
            
            if user_content:
                if debug:
                    print_hl("User >")
                    for line in _summarize_mm_content(print_user_content):
                        print(line)
                messages.append({"role": "user", "content": user_content})
                print_messages.append({"role": "user", "content": print_user_content})
            else:
                if debug:
                    _log(True, "[End] No tool calls or extra prompts produced; breaking the loop.")
                break
                
        except Exception as e:
            tb = traceback.format_exc()
            err_text = f"error: {str(e)}"
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": err_text + '\n' + tb}]
            })
            print_messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": err_text + '\n' + tb}]
            })
            if debug:
                _log_kv(True, "Caught exception in round", {"error": str(e) + '\n' + tb})
            break
    
    if debug:
        _log_kv(True, "Conversation finished", {
            "total_rounds": manager.round_count,
            "crop_calls": manager.crop_call_count,
            "search_calls": manager.search_call_count
        })
    return print_messages


def _process_one_record(i: int, total: int, record: Dict[str, Any], args: argparse.Namespace, out_f, write_lock: threading.Lock, print_lock: threading.Lock) -> None:
    uid = record.get("uid", f"unknown_{i}")
    image_path = record.get("image_path")

    if not image_path or not os.path.exists(image_path):
        with print_lock:
            print(f"[WARN] {uid}: Image path not found: {image_path}")
        return

    with print_lock:
        print(f"[{i}/{total}] Processing {uid}...")

    try:
        messages = run_inference_for_image(image_path, args.temp_dir, debug=args.debug)
        result = {
            "uid": uid,
            "image_path": image_path,
            "pred_output": messages,
            "metadata": record.get("metadata", {})
        }
        line = json.dumps(result, ensure_ascii=False)
        with write_lock:
            out_f.write(line + "\n")
            out_f.flush()

        with print_lock:
            print(f"[{i}/{total}] Done {uid}")

    except Exception as e:
        # Surface vLLM connectivity errors to abort the whole run
        if 'HTTPConnectionPool' in str(e):
            raise
        tb = traceback.format_exc()
        with print_lock:
            print(f"[ERR] {uid}: {str(e)}")
        error_result = {
            "uid": uid,
            "image_path": image_path,
            "error": str(e),
            "traceback": tb,
            "metadata": record.get("metadata", {})
        }
        line = json.dumps(error_result, ensure_ascii=False)
        with write_lock:
            out_f.write(line + "\n")
            out_f.flush()


def main():
    parser = argparse.ArgumentParser(description="Run inference on the GeoBench Hugging Face dataset using agent tool mode")
    parser.add_argument("--dataset_id", type=str, default=".temp/datasets/LibraTree/GeoVistaBench",
                       help="Hugging Face dataset identifier to load (default: GeoBench)")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to load from the Hugging Face repo")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Optional datasets cache directory")
    parser.add_argument("--hf_image_dir", type=str, default=".temp/hf_images",
                       help="Directory to materialize HF dataset images for inference")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                       help="Optional Hugging Face token for private datasets")
    parser.add_argument("--output", type=str, default=".temp/outputs/inference/inference_results_debug.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--temp_dir", type=str, default=".temp/outputs/inference_crops",
                       help="Temporary directory for cropped images")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key for chat model")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug printing")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of worker threads for concurrent processing (default: 1)")
    args = parser.parse_args()

    if args.debug:
        _log_kv(True, "Args", vars(args))
    
    # Load dataset from Hugging Face
    try:
        records = _load_geobench_records(
            dataset_id=args.dataset_id,
            split=args.split,
            cache_dir=args.cache_dir,
            temp_dir=args.hf_image_dir,
            hf_token=args.hf_token,
            limit=args.num_samples,
            debug=args.debug,
        )
    except Exception as e:
        print(f"[ERR] Failed to load dataset {args.dataset_id}:{args.split}: {e}")
        return
    
    total = len(records)
    print(f"Processing {total} records...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # If single worker, preserve original sequential behavior exactly
    if not args.num_workers or args.num_workers == 1:
        with open(args.output, "w", encoding="utf-8") as out_f:
            for i, record in enumerate(records, 1):
                uid = record.get("uid", f"unknown_{i}")
                image_path = record.get("image_path")
                
                if not image_path or not os.path.exists(image_path):
                    print(f"[WARN] {uid}: Image path not found: {image_path}")
                    continue
                
                print(f"[{i}/{total}] Processing {uid}...")
                
                try:
                    messages = run_inference_for_image(image_path, args.temp_dir, debug=args.debug)
                    
                    result = {
                        "uid": uid,
                        "image_path": image_path,
                        "pred_output": messages,
                        "metadata": record.get("metadata", {})
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    
                except Exception as e:
                    # vllm deployment error: preserve original behavior and abort
                    if 'HTTPConnectionPool' in str(e):
                        raise e
                    tb = traceback.format_exc()
                    print(f"[ERR] {uid}: {str(e)}")
                    if args.debug:
                        _log_kv(True, "Record-level error", {"uid": uid, "error": str(e)})
                    error_result = {
                        "uid": uid,
                        "image_path": image_path,
                        "error": str(e),
                        "traceback": tb,
                        "metadata": record.get("metadata", {})
                    }
                    out_f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    out_f.flush()
    else:
        # Concurrent processing
        write_lock = threading.Lock()
        print_lock = threading.Lock()
        with open(args.output, "w", encoding="utf-8") as out_f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = []
                for i, record in enumerate(records, 1):
                    f = executor.submit(_process_one_record, i, total, record, args, out_f, write_lock, print_lock)
                    futures.append(f)
                # Propagate connectivity errors if any
                for f in concurrent.futures.as_completed(futures):
                    _ = f.result()
    
    print(f"Results saved to {args.output}")
    
    # Copy the results to a fixed file name without timestamp in the same directory
    try:
        target_output = str(Path(args.output).with_name("inference.jsonl"))
        shutil.copyfile(args.output, target_output)
        print(f"Copied latest results to {target_output}")
    except Exception as e:
        print(f"[WARN] Failed to copy results to inference.jsonl: {e}")

if __name__ == "__main__":
    main()
