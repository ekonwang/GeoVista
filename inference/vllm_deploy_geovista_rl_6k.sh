#!/bin/bash

# Default values
MODEL_NAME=.temp/checkpoints/LibraTree/GeoVista-RL-6k-7B

export CUDA_VISIBLE_DEVICES="0,1,2,3"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL=4
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=8
MAX_IMAGES_PER_PROMPT=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --max-images-per-prompt)
            MAX_IMAGES_PER_PROMPT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_NAME                Model name or path (default: Qwen/Qwen2.5-VL-14B-Instruct)"
            echo "  --host HOST                       Host to bind (default: 0.0.0.0)"
            echo "  --port PORT                       Port to bind (default: 8000)"
            echo "  --tensor-parallel-size SIZE       Tensor parallel size (default: 4)"
            echo "  --gpu-memory-utilization RATIO    GPU memory utilization (default: 0.9)"
            echo "  --max-model-len LENGTH            Max model length (default: 32768)"
            echo "  --max-num-seqs SEQS               Max number of sequences (default: 256)"
            echo "  --max-images-per-prompt N         Max images allowed per request (default: 8)"
            echo "  --help, -h                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting vLLM server with the following configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Max Number of Sequences: $MAX_NUM_SEQS"
echo "  Max Images per Prompt: $MAX_IMAGES_PER_PROMPT"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo

# Resolve chat template path relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_PATH="$SCRIPT_DIR/template_qwen.jinja"
EXTRA_TEMPLATE_ARGS=()
if [ -f "$TEMPLATE_PATH" ]; then
    EXTRA_TEMPLATE_ARGS=(--chat-template "$TEMPLATE_PATH")
else
    echo "Warning: chat template not found at $TEMPLATE_PATH; using model's default."
fi

MM_LIMIT_ARGS=(--limit-mm-per-prompt "image=$MAX_IMAGES_PER_PROMPT")

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vllm is not installed or not in PATH"
    echo "Please install vllm first: pip install vllm"
    exit 1
fi

# Start vLLM server
exec vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --trust-remote-code \
    --served-model-name qwen2.5-vl \
    "${EXTRA_TEMPLATE_ARGS[@]}" \
    "${MM_LIMIT_ARGS[@]}"
