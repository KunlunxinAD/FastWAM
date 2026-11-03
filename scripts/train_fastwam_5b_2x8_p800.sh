#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XPU_VISIBLE_DEVICES="${XPU_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the rank-0 host address}"
export MASTER_PORT="${MASTER_PORT:-28510}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export XPYTORCH_RUN_ENHANCE="${XPYTORCH_RUN_ENHANCE:-1}"
export XDNN_USE_FAST_SWISH="${XDNN_USE_FAST_SWISH:-1}"
export XDNN_FAST_DIV_SCALAR="${XDNN_FAST_DIV_SCALAR:-true}"
export XMLIR_ENABLE_LINEAR_FC_FUSION="${XMLIR_ENABLE_LINEAR_FC_FUSION:-1}"
export XMLIR_ENABLE_FAST_FC="${XMLIR_ENABLE_FAST_FC:-1}"
export XDNN_USE_FAST_GELU="${XDNN_USE_FAST_GELU:-1}"
export XPUAPI_SDNN_BF16_ROUND_MODE="${XPUAPI_SDNN_BF16_ROUND_MODE:-3}"

bash scripts/train_zero1.sh 8 \
  task=libero_uncond_2cam224_1e-4 \
  max_steps="${STEPS:-500}" log_every=1 \
  save_every="${SAVE_EVERY:-250}" \
  save_total_limit="${SAVE_TOTAL_LIMIT:-2}" eval_every=0 \
  +data.train.video_backend="${VIDEO_BACKEND:-torchcodec}" \
  +data.train.pretrained_norm_stats="${STATS_PATH:-${REPO_ROOT}/data/libero_mujoco3.3.2/dataset_stats.json}"
