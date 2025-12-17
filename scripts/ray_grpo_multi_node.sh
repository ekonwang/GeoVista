# if ! conda env list 2>/dev/null | grep "deepeyes"; then
#     echo "Error: 'deepeyes' conda environment does not exist."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate deepeyes


set -x

export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_RUN_NAME="ray_grpo_multi_node"
export WANDB_PROJECT="GeoVista"
export SAVE_CHECKPOINT_DIR=/apdcephfs_gy2/share_303214312/rickykwang/deepeyes_logs
export REF_MODEL_PATH=/home/models/1023_sft_search_v4_seedvl

export WORLD_SIZE=8
export VISUAL_DATASET_MAZE=
export VISUAL_DATASET_MAZE_VAL=
# TODO: DEBUG bsz=8
export BATCH_SIZE=64

wandb login $WANDB_API_KEY

## Pre-submit checkpoint setup
# Ensure reference model checkpoint exists; if missing, set it up
if [ -z "${REF_MODEL_PATH:-}" ]; then
  echo "Error: REF_MODEL_PATH is not set." >&2
  exit 1
fi

if [ ! -d "${REF_MODEL_PATH}" ]; then
  echo "REF_MODEL_PATH '${REF_MODEL_PATH}' not found. Running setup_ckpt.sh..."
  bash scripts/search/setup_mp_rsync.sh
fi

cd /mnt/private/agent_workspace/hunyuan-o3


set -a; source /mnt/private/agent_workspace/hunyuan-o3/.env; set +a;
# Ensure TAVILY_API_KEY is set
if [ -z "${TAVILY_API_KEY:-}" ]; then
  echo "Error: TAVILY_API_KEY is not set in the environment." >&2
  exit 1
fi
echo $TAVILY_API_KEY
echo $GPT_CUSTOM_API_KEY_4o_mini

RUNTIME_ENV_JSON=$(
  cat <<EOF
{"excludes": [".git/*", "checkpoints/*", "logs/*"], "env_vars": {"http_proxy": "http://star-proxy.oa.com:3128", "https_proxy": "http://star-proxy.oa.com:3128", "NCCL_IB_GID_INDEX": "3", "NCCL_IB_SL": "3", "NCCL_CHECKS_DISABLE": "1", "NCCL_P2P_DISABLE": "0", "NCCL_IB_DISABLE": "0", "NCCL_LL_THRESHOLD": "16384", "NCCL_IB_CUDA_SUPPORT": "1", "NCCL_SOCKET_IFNAME": "bond1", "UCX_NET_DEVICES": "bond1", "NCCL_COLLNET_ENABLE": "0", "SHARP_COLL_ENABLE_SAT": "0", "NCCL_NET_GDR_LEVEL": "2", "NCCL_IB_QPS_PER_CONNECTION": "4", "NCCL_IB_TC": "160", "NCCL_PXN_DISABLE": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "MKL_SERVICE_FORCE_INTEL": "1", "NCCL_P2P_LEVEL": "NVL", "WANDB_BASE_URL": "https://api.wandb.ai", "WANDB_KEY": "${WANDB_API_KEY}", "WANDB_API_KEY": "${WANDB_API_KEY}", "WANDB_RUN_NAME": "${WANDB_RUN_NAME}", "WANDB_PROJECT": "${WANDB_PROJECT}", "SAVE_CHECKPOINT_DIR": "${SAVE_CHECKPOINT_DIR}", "WORLD_SIZE": "${WORLD_SIZE}", "TAVILY_API_KEY": "${TAVILY_API_KEY}", "GPT_CUSTOM_API_KEY_4o_mini": "${GPT_CUSTOM_API_KEY_4o_mini}"}, "WG_BACKEND": "ray"}
EOF
)

ray job submit --address='http://127.0.0.1:8265' \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- \
  python3 -m verl.trainer.main_ppo \
  data.train_files=[${VISUAL_DATASET_MAZE}] \
  data.val_files=[${VISUAL_DATASET_MAZE_VAL}] \
  data.train_batch_size=${BATCH_SIZE} \
  data.max_prompt_length=16384 \
  data.max_response_length=16384 \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.model.path=${REF_MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.agent.activate_agent=True \
  actor_rollout_ref.rollout.agent.tool_name_key=env_name \
  actor_rollout_ref.rollout.agent.single_response_max_tokens=8192 \
  actor_rollout_ref.rollout.agent.max_turns=6 \
  actor_rollout_ref.rollout.agent.concurrent_workers=8 \
  actor_rollout_ref.rollout.agent.show_tqdm=True \
  trainer.critic_warmup=0 \
  reward_model.reward_manager="native_parallel" \
  trainer.logger=['console','wandb','rl_logging_board'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=${WORLD_SIZE} \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  +trainer.save_before_train=False \
  trainer.resume_mode=auto \
  trainer.project_name=${WANDB_PROJECT} \
  trainer.experiment_name=${WANDB_RUN_NAME} \
  trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_RUN_NAME} \
  +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
  +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
  trainer.total_epochs=1 2>&1 | tee ./logs/${WANDB_RUN_NAME}.log
