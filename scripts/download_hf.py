from datasets import load_dataset
from huggingface_hub import snapshot_download

# change the current directory to the workspace
import os
import sys
from utils import print_error

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--local_model_dir", type=str, default="./checkpoints_gy/pretrained_models")
    parser.add_argument("--local_dataset_dir", type=str, default="./checkpoints_gy/datasets")
    return parser.parse_args()

args = parse_args()

def snap_download_model(args):
    # model_name = "Qwen/Qwen2-VL-2B-Instruct"
    # local_model_name = model_name.replace("/", "_")
    local_model_name = args.model
    if args.model is None:
        return

    while True:
        try:
            model_path = snapshot_download(
                repo_id=args.model,  # The model ID on Hugging Face Hub
                local_dir=args.local_model_dir + "/" + local_model_name  # Specific directory for this model
            )
            print(f"Downloaded {args.model} to {model_path}")
            break 
        except Exception as err:
            print_error(err)


def snap_download_dataset(args):
    local_dataset_name = args.dataset
    if args.dataset is None:
        return

    while True:
        try:
            snapshot_download(
                repo_id=args.dataset, 
                repo_type="dataset", 
                local_dir=args.local_dataset_dir + "/" + local_dataset_name,
            )
            break 
        except Exception as err:
            print_error(err)

if __name__ == "__main__":
    # "Qwen/Qwen2.5-VL-3B-Instruct"
    args = parse_args()
    snap_download_model(args)
    snap_download_dataset(args)
