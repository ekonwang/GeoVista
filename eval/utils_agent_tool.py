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
import uuid as uuid_lib

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


def build_system_prompt() -> str:
	"""Construct the detailed system prompt with tool definitions and usage templates."""
	return (
		"You are a geolocation assistant. You will be given a high-resolution panorama or photo. "
		"Use two tools to infer the real-world geographic location of the image.\n\n"
		"- First, call the image-zoom-in tool one or more times to crop and magnify regions of interest (e.g., signage, route displays, address numbers).\n"
		"- After obtaining one or more cropped images, use the web-search tool to query for relevant information extracted from those crops.\n\n"
		"# Tools\n"
		"You may call one or more functions to assist with the user query.\n"
		"Provide function signatures within <tools></tools> XML tags:\n"
		"<tools>\n"
		"{\n"
		"  \"type\": \"function\",\n"
		"  \"function\": {\n"
		"    \"name\": \"image_zoom_in_tool\",\n"
		"    \"description\": \"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).\",\n"
		"    \"parameters\": {\n"
		"      \"type\": \"object\",\n"
		"      \"properties\": {\n"
		"        \"bbox_2d\": {\n"
		"          \"type\": \"array\",\n"
		"          \"items\": {\"type\": \"number\"},\n"
		"          \"minItems\": 4,\n"
		"          \"maxItems\": 4,\n"
		"          \"description\": \"The bounding box as [x1, y1, x2, y2] on the original image.\"\n"
		"        },\n"
		"        \"label\": {\n"
		"          \"type\": \"string\",\n"
		"          \"description\": \"Optional label for the cropped object.\"\n"
		"        }\n"
		"      },\n"
		"      \"required\": [\"bbox_2d\"]\n"
		"    }\n"
		"  }\n"
		"}\n"
		"\n,\n\n"
		"{\n"
		"  \"type\": \"function\",\n"
		"  \"function\": {\n"
		"    \"name\": \"search_web\",\n"
		"    \"description\": \"Execute a web search and return normalized results containing titles, snippets, and URLs.\",\n"
		"    \"parameters\": {\n"
		"      \"type\": \"object\",\n"
		"      \"properties\": {\n"
		"        \"query\": {\n"
		"          \"type\": \"string\",\n"
		"          \"description\": \"The search query string.\"\n"
		"        }\n"
		"      },\n"
		"      \"required\": [\"query\"]\n"
		"    }\n"
		"  }\n"
		"}\n"
		"</tools>\n\n"
		"# How to call a tool\n"
		"Return a JSON object with function name and arguments within <tool_call>...</tool_call> XML tags:\n"
		"<tool_call>\n"
		"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
		"</tool_call>\n\n"
		"Examples:\n"
		"<tool_call>\n"
		"{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [10, 20, 100, 200], \"label\": \"the apple on the desk\"}}\n"
		"</tool_call>\n\n"
		"<tool_call>\n"
		"{\"name\": \"search_web\", \"arguments\": {\"query\": \"The palace museum\"}}\n"
		"</tool_call>\n\n"
		"When you call a tool, think first. Place your internal reasoning inside <think>...</think>, followed by the <tool_call>...</tool_call>."
	)

def encode_image_to_base64(image_path: str, max_pixels: int = MAX_PIXELS) -> str:
	"""Encode image to base64, with optional resizing."""
	img = Image.open(image_path)
	img = img.convert('RGB')
	w, h = img.size
	
	if w * h > max_pixels:
		scale = (max_pixels / (w * h)) ** 0.5
		img = img.resize((int(w * scale), int(h * scale)))
	
	# print(f'image_path: {image_path}, w: {w}, h: {h}, resized image size: {img.size}')
	# import pdb; pdb.set_trace()
	
	buf = io.BytesIO()
	img.save(buf, format='JPEG')
	image_bytes = buf.getvalue()
	image_b64 = base64.b64encode(image_bytes).decode('utf-8')
	return image_b64

def get_image_resolution(image_path: str) -> Tuple[int, int]:
	"""Get original image resolution."""
	with Image.open(image_path) as img:
		return img.size

def save_b64_image(b64_string: str, img_save_dir: str, ext: str = "jpg") -> str:
	file_name = f"{uuid_lib.uuid4().hex}.{ext}"
	file_path = os.path.abspath(os.path.join(img_save_dir, file_name))
	with open(file_path, "wb") as fp:
		fp.write(base64.b64decode(b64_string))
	return file_path

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
	"""Extract tool calls from assistant response."""
	calls: List[Dict[str, Any]] = []
	for m in TOOL_CALL_PATTERN.finditer(text or ""):
		raw = m.group(1)
		try:
			call = json.loads(raw)
			calls.append(call)
		except Exception:
			sanitized = raw.strip().strip("`")
			try:
				call = json.loads(sanitized)
				calls.append(call)
			except Exception:
				continue
	return calls

def round_by_factor(number: int, factor: int) -> int:
	return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
	return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
	return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
	h_bar = max(factor, round_by_factor(height, factor))
	w_bar = max(factor, round_by_factor(width, factor))
	if h_bar * w_bar > max_pixels:
		beta = math.sqrt((height * width) / max_pixels)
		h_bar = floor_by_factor(height / beta, factor)
		w_bar = floor_by_factor(width / beta, factor)
	elif h_bar * w_bar < min_pixels:
		beta = math.sqrt(min_pixels / (height * width))
		h_bar = ceil_by_factor(height * beta, factor)
		w_bar = ceil_by_factor(width * beta, factor)
	return h_bar, w_bar

def _log(debug: bool, msg: str):
	if debug:
		print(msg)

def _log_kv(debug: bool, title: str, obj: Any):
	if debug:
		print_hl(title)
		try:
			print(json.dumps(obj, ensure_ascii=False, indent=2))
		except Exception:
			print(str(obj))

def _summarize_mm_content(content: List[Dict[str, Any]]) -> List[str]:
	lines = []
	for i, item in enumerate(content):
		t = item.get("type")
		if t == "text":
			txt = item.get("text", "")
			lines.append(f"[{i}] type=text len={len(txt)} preview={txt[:160].replace(chr(10),' ')}")
		elif t == "image_url":
			url = item.get("image_url", {}).get("url")
			lines.append(f"[{i}] type=image_url url={url if isinstance(url,str) else str(url)}")
		else:
			lines.append(f"[{i}] type={t}")
	return lines


def dump_tool_call(tool_call) -> str:
	return f"<tool_call>{json.dumps(tool_call, ensure_ascii=False, indent=2)}</tool_call>"


def reformat_response(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
	"""Return the content before the first <tool_call> tag and the first valid tool call (if any). Extra tool calls are omitted."""
	if text is None:
		return "", None
	if not isinstance(text, str):
		text = str(text)
	m = TOOL_CALL_PATTERN.search(text)
	if not m:
		return text
	pre_text = text[:m.start()].strip()
	raw = m.group(1)
	call = None
	for candidate in (raw, raw.strip().strip("`")):
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict) and parsed.get("name"):
				call = parsed
				break
		except Exception:
			continue
	return pre_text + dump_tool_call(call)


# =========================
# Core tool functions (out of class)
# =========================

def crop_tool_core(original_image_path: str, bbox_2d: List[float], label: Optional[str] = None, debug: bool = False, abs_scaling=1., bbox_normalize=False, crop_path = None) -> Dict[str, Any]:
	"""Core logic for image cropping and resizing using the original image path."""
	try:
		left, top, right, bottom = bbox_2d

		if bbox_normalize:
			with Image.open(original_image_path) as _img:
				w, h = _img.size
				left_px = int(round(left / 1000.0 * w))
				top_px = int(round(top / 1000.0 * h))
				right_px = int(round(right / 1000.0 * w))
				bottom_px = int(round(bottom / 1000.0 * h))
				cropped_image = _img.crop((left_px, top_px, right_px, bottom_px))
			w_crop = right_px - left_px
			h_crop = bottom_px - top_px
		else:
			left_px = int(left * abs_scaling)
			top_px = int(top * abs_scaling)
			right_px = int(right * abs_scaling)
			bottom_px = int(bottom * abs_scaling)
			with Image.open(original_image_path) as _img:
				cropped_image = _img.crop((left_px, top_px, right_px, bottom_px))
			w_crop = right_px - left_px
			h_crop = bottom_px - top_px

		new_w, new_h = smart_resize(w_crop, h_crop, factor=IMAGE_FACTOR)
		cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

		if crop_path is None:
			tmpdir = os.path.abspath(".temp/inference")
			os.makedirs(tmpdir, exist_ok=True)
			crop_path = os.path.join(tmpdir, f"crop_{str(uuid.uuid4())}.jpg")
		cropped_image.save(crop_path, format='JPEG')

		buf = io.BytesIO()
		cropped_image.save(buf, format='JPEG')
		crop_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

		if debug:
			_log_kv(True, "Crop saved", {
				"bbox": bbox_2d,
				"label": label,
				"crop_path": crop_path,
				"resized_wxh": [new_w, new_h]
			})

		return {
			"success": True,
			"crop_path": crop_path,
			"crop_b64": crop_b64,
			"label": label,
			"bbox": bbox_2d
		}

	except Exception as e:
		raise Exception(f"Crop tool execution failed: {str(e)}")

def search_tool_core(query: str, debug: bool = False, retriever_name: str = 'tavily') -> Dict[str, Any]:
	"""Core logic for executing a web search via the configured retriever."""
	try:
		if debug:
			_log_kv(True, "Executing web search", {"query": query, "retriever": retriever_name})
		search_results = run_search(query, retriever_name=retriever_name)
		if debug:
			_log_kv(True, "Search results (summary)", {
				"num_results": len(search_results) if isinstance(search_results, list) else "n/a"
			})
		if len(search_results) == 0:
			raise ValueError("Empty search results!! Please Check the availability of the search API key and network connection.")
		return {
			"success": True,
			"query": query,
			"results": search_results
		}
	except Exception as e:
		raise Exception(f"Search tool execution failed: {str(e)}")


class ToolCallManager:
	"""Manager for handling tool calls and conversation flow."""
	
	def __init__(self, original_image_path: str, temp_dir: str = ".temp/outputs/inference"):
		self.original_image_path = original_image_path
		self.temp_dir = os.path.abspath(temp_dir)
		self.crop_call_count = 0
		self.search_call_count = 0
		self.round_count = 0
		self.original_image = Image.open(original_image_path)
		
		# Ensure temp directory exists
		os.makedirs(self.temp_dir, exist_ok=True)
	
	def execute_crop_tool(self, bbox_2d: List[float], label: Optional[str] = None, debug: bool = False, abs_scaling=1., bbox_normalize=False) -> Dict[str, Any]:
		"""Execute image crop tool call."""
		if self.crop_call_count >= MAX_CROP_CALLS:
			raise Exception(f"Maximum crop tool calls ({MAX_CROP_CALLS}) exceeded")
		
		try:
			crop_filename = f"crop_{self.crop_call_count}_{str(uuid.uuid4())}.jpg"
			crop_path = os.path.join(self.temp_dir, crop_filename)

			result = crop_tool_core(
				original_image_path=self.original_image_path,
				bbox_2d=bbox_2d,
				label=label,
				debug=debug,
				abs_scaling=abs_scaling,
				crop_path=crop_path,
				bbox_normalize=bbox_normalize
			)

			self.crop_call_count += 1
			return result
			
		except Exception as e:
			raise Exception(f"Crop tool execution failed: {str(e)}")
	
	def execute_search_tool(self, query: str, debug: bool = False) -> Dict[str, Any]:
		"""Execute web search tool call."""
		if self.search_call_count >= MAX_SEARCH_CALLS:
			raise Exception(f"Maximum search tool calls ({MAX_SEARCH_CALLS}) exceeded")
		
		try:
			result = search_tool_core(query=query, debug=debug, retriever_name='tavily')
			self.search_call_count += 1
			return result
			
		except Exception as e:
			raise Exception(f"Search tool execution failed: {str(e)}")
	
	def should_prompt_search(self) -> bool:
		"""Check if we should prompt for search in the next round."""
		return (self.round_count == MAX_ROUNDS - 2 and 
				self.search_call_count == 0)
	
	def should_force_final_answer(self) -> bool:
		"""Check if we should force final answer."""
		return self.round_count == MAX_ROUNDS - 1
	
	def increment_round(self):
		"""Increment round counter."""
		self.round_count += 1
