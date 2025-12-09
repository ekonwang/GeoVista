set -x
export WANDB_BASE_URL=https://api.wandb.ai

RUN_NAME=GeoVista-SFT-7B
SCRIPT_NAME=scripts/sft.py

COLD_START_DATA_DIR=.temp/datasets/LibraTree/GeoVista-Cold-Start

BASE_MODEL=Qwen2.5-VL-7B-Instruct
MODEL_PATH=.temp/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct
OUTPUT_DIR=.temp/checkpoints/${RUN_NAME}
DATA_PATH=${COLD_START_DATA_DIR}/data/train/data-00000-of-00001.parquet

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_RUN_NAME=${RUN_NAME}
export WANDB_PROJECT="GeoVista"
wandb login $WANDB_API_KEY

rm -rf ${OUTPUT_DIR}
echo 'remove output dir'
cd $(dirname $0)/..
mkdir -p ./logs

nnode=1
nrank=$((INDEX%nnode))
MASTER_ADDR='127.0.0.1'
MASTER_PORT=12566

echo $MASTER_ADDR; echo $nnode; echo $nrank
lsof -ti:${MASTER_PORT} | xargs kill -9

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
torchrun  --nproc_per_node 8 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT} \
    ${SCRIPT_NAME} \
    --deepspeed ./scripts/zero3_new.json \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --seed 42 \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-7 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --save_only_model true | tee ./logs/$RUN_NAME.log 2>&1
