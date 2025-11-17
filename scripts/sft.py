import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import pandas as pd
# from qwen_vl_utils import process_vision_info
from utils_vision import process_vision_info
import wandb

from transformers import TrainerCallback
from deepspeed.accelerator import get_accelerator

from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        get_accelerator().empty_cache()


logger = logging.getLogger(__name__)


if 'LAST_ROUND_ONLY' in os.environ:
    print("LAST_ROUND_ONLY is set")
    LAST_ROUND_ONLY = True
else:
    LAST_ROUND_ONLY = False


# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IS_TOKENIZER_GREATER_THAN_0_14 = True

def is_rank0() -> bool:
    """
    Determine if current process is rank 0 using torch.distributed.get_rank().
    Falls back to True when torch.distributed is not initialized (single process).
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True

# wandb.init(mode="online", project="OpenThinkIMG")
# if is_rank0():
#     # wandb.init(mode="disabled")
#     wandb.init(mode="online", project="OpenThinkIMG")
# else:
#     wandb.init(mode="disabled")

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    debug_mask: bool = field(default=False, metadata={"help": "Enable debug printing of masked spans (ids and detokenized strings) on rank 0."})
    min_learning_rate: Optional[float] = field(default=None, metadata={"help": "Minimum learning rate floor to avoid zero LR steps."})

processor = None
DEBUG = False

class MinLRCallback(TrainerCallback):
    def __init__(self, min_lr: float):
        self.min_lr = float(min_lr)

    def on_optimizer_step(self, args, state, control, optimizer, **kwargs):
        updated = False
        for group in optimizer.param_groups:
            if "lr" in group and group["lr"] < self.min_lr:
                group["lr"] = self.min_lr
                updated = True
        if updated and DEBUG and is_rank0():
            try:
                lrs = [group.get("lr", None) for group in optimizer.param_groups]
                print(f"[DEBUG][rank0] MinLR applied. Current LRs: {lrs}")
            except Exception:
                pass


def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    system_prompt = example.get('system') or example.get('system_prompt')
    if system_prompt:
        messages = [{'role': 'system', 'content': system_prompt}]
    else:
        print("no sys prompt!!")

    conversations = example.get('conversations')
    image_paths = example.get('images')
    if image_paths is None:
        image_paths = example.get('image')
    img_idx = 0
    for item in conversations:
        content = []
        if item['from'] == 'human':
            role = 'user'
        else:
            role = 'assistant'

        if '<image>' in item['value']:
            assert item['value'].count('<image>') == 1  # only support one image currently
            if item['value'].startswith('<image>'):
                content.append({'type': 'image', 'image': image_paths[img_idx]})
                content.append({'type': 'text', 'text':item['value']})
            else:
                value = item['value']
                value = value.split('<image>')
                content.append({'type': 'text', 'text': value[0]})
                content.append({'type': 'image', 'image': image_paths[img_idx]})
                content.append({'type': 'text', 'text': value[1]})
            img_idx += 1
            
        else:
            content.append({'type': 'text', 'text': item['value']})
        
        messages.append({
            'role': role,
            'content': content
        })
    example["messages"] = messages

    return example

from PIL import Image
import io

def convert_images(data_dict):
    # 检查字典是否包含必要的键
    if 'image' in data_dict.keys():
        data_dict['images'] = data_dict['image']
    if 'images' not in data_dict.keys():
        return data_dict
    data_dict_new = data_dict.copy()
    del data_dict_new['images']
    pil_list = []
    # 处理每个图像数据
    for img_info in data_dict['images']:
        if 'bytes' not in img_info.keys():
            continue
        if img_info['bytes'] is None:
            img_path = img_info['path']
            pil_image = Image.open(img_path).convert('RGB')
        else:
            # 从字节数据创建PIL图像
            img_bytes = img_info['bytes']
            img_byte_array = io.BytesIO(img_bytes)
            pil_image = Image.open(img_byte_array)
        
        # 确保图像数据被加载（避免延迟加载导致后续错误）
        pil_image.load()
        
        pil_list.append(pil_image)
    data_dict_new['images'] = pil_list
    return data_dict_new


def collate_fn(examples):
    examples = [convert_images(example) for example in examples]
    texts = [processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    # print(labels.size())
    # print('per_device_bsz ', len(texts))
    # labels[labels == processor.tokenizer.pad_token_id] = -100
    # image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    # labels[labels == image_token_id] = -100
    # Mask targets using token markers to avoid image token expansion mismatch
    sep_start = '<|im_start|>assistant\n'
    sep_end = '<|im_end|>\n'
    pad_id = processor.tokenizer.pad_token_id

    def get_marker_ids(s):
        ids = processor.tokenizer(s).input_ids
        if len(ids) > 0 and ids[0] == processor.tokenizer.bos_token_id:
            ids = ids[1:]
        if len(ids) > 0 and ids[-1] == processor.tokenizer.eos_token_id:
            ids = ids[:-1]
        return ids

    start_ids = get_marker_ids(sep_start)
    end_ids = get_marker_ids(sep_end)

    def find_subsequence_positions(sequence, pattern, valid_len):
        positions = []
        plen = len(pattern)
        if plen == 0:
            return positions
        for idx in range(0, valid_len - plen + 1):
            if sequence[idx: idx + plen] == pattern:
                positions.append(idx)
        return positions

    batch_size = labels.size(0)
    for i in range(batch_size):
        input_ids = batch["input_ids"][i]
        valid_len = int(input_ids.ne(pad_id).sum())
        seq = input_ids[:valid_len].tolist()

        labels[i, :valid_len] = IGNORE_INDEX
        labels[i, valid_len:] = IGNORE_INDEX

        start_positions = find_subsequence_positions(seq, start_ids, valid_len)
        end_positions_all = find_subsequence_positions(seq, end_ids, valid_len)

        if len(start_positions) == 0 or len(end_positions_all) == 0:
            continue

        end_list = list(end_positions_all)
        spans = []
        for s in start_positions:
            e = next((e for e in end_list if e > s), None)
            if e is None:
                break
            # 加入 len(end_ids) 保证学习到 eot tokens，在每轮对话中正常终止
            spans.append((s + len(start_ids), min(e + len(end_ids), valid_len)))

        if LAST_ROUND_ONLY and len(spans) > 0:
            spans = [spans[-1]]

        if DEBUG and is_rank0() and len(spans) > 0:
            for s_tok, e_tok in spans:
                if s_tok < e_tok and e_tok <= valid_len:
                    span_ids = batch["input_ids"][i, s_tok:e_tok].tolist()
                    try:
                        span_text = processor.tokenizer.decode(span_ids, skip_special_tokens=False)
                    except Exception:
                        span_text = "<decode_error>"
                    print(f"[DEBUG][rank0] sample={i} span=({s_tok},{e_tok}) len={e_tok - s_tok}")
                    print(f"[DEBUG][rank0] ids={span_ids}")
                    print(f"[DEBUG][rank0] text={span_text}")

        for s_tok, e_tok in spans:
            if s_tok < e_tok and e_tok <= valid_len:
                labels[i, s_tok:e_tok] = batch["input_ids"][i, s_tok:e_tok]

    batch["labels"] = labels

    print(batch['input_ids'].shape)
    if int(batch['input_ids'].shape[1]) > 32768:
        print(f'{examples[0]["id"]} too long!!')

    # 释放中间变量
    batch['input_ids'] = batch['input_ids'][:,:32768]
    batch['labels'] = batch['labels'][:,:32768]
    batch['attention_mask'] = batch['attention_mask'][:,:32768]

    del texts, image_inputs
    return batch

def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # propagate debug flag
    global DEBUG
    DEBUG = bool(getattr(training_args, "debug_mask", False))

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # dataset = load_dataset('json', data_files=script_args.dataset_name)
    # dataset = load_dataset("parquet", data_dir=script_args.dataset_name)
    # import ast
    # dataset_list = ast.literal_eval(script_args.dataset_name)
    # combined_df = pd.concat(
    #     [pd.read_parquet(file) for file in dataset_list],
    #     axis=0,
    #     ignore_index=True
    # )
    # shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df = pd.read_parquet(script_args.dataset_name)
    dataset = datasets.Dataset.from_pandas(shuffled_df)
    ################
    # Load tokenizer
    ################
    min_pixels = 65536
    # max_pixels = 4194304
    max_pixels = 1024 * 2048
    global processor
    # if "vl" in model_args.model_name_or_path.lower():
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=min_pixels, max_pixels=max_pixels
    )
        # if "Qwen2.5-VL" in model_args.model_name_or_path.lower():
        #     processor.image_processor.max_pixels = 4000
        #     # processor.image_processor.min_pixels = 3136
        #     logger.info("Using AutoProcessor for vision-language model.")
    # else:
    #     processor = AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    #     )
    #     logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    print(model_kwargs)
    # training_args.model_init_kwargs = model_kwargs
    # from transformers import Qwen2VLForConditionalGeneration
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path, **model_kwargs
    # )
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=[EmptyCacheCallback()] + ([MinLRCallback(training_args.min_learning_rate)] if training_args.min_learning_rate is not None else [])
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
