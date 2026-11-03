#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-4}"
TASK="${TASK:-libero_uncond_2cam224_1e-4}"
STEPS="${STEPS-20}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MASTER_PORT="${MASTER_PORT:-29601}"
STATS_PATH="${STATS_PATH:-${REPO_ROOT}/data/libero_mujoco3.3.2/dataset_stats.json}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
NUM_WORKERS="${NUM_WORKERS:-}"
DEVICES="${DEVICES:-$(seq -s, 0 $((GPUS-1)))}"

export MASTER_PORT XPU_VISIBLE_DEVICES="${DEVICES}" CUDA_VISIBLE_DEVICES="${DEVICES}"
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

if [[ ! -f "${STATS_PATH}" ]]; then
  echo "dataset_stats.json 不存在: ${STATS_PATH}" >&2
  exit 1
fi

OVERRIDES=(
  "task=${TASK}" "log_every=${LOG_EVERY}" "save_every=${SAVE_EVERY}"
  "eval_every=${EVAL_EVERY}"
  "+data.train.video_backend=${VIDEO_BACKEND}"
  "+data.train.pretrained_norm_stats=${STATS_PATH}"
)
if [[ -n "${NUM_WORKERS}" ]]; then OVERRIDES+=("num_workers=${NUM_WORKERS}"); fi
if [[ -n "${STEPS}" ]]; then OVERRIDES+=("max_steps=${STEPS}"); fi

if [[ -n "${FASTWAM_LATENT_CACHE_ROOT:-}" && "${LATENT_CACHE_AUTO:-1}" == "1" ]]; then
  LATENT_CACHE_ROOT="${FASTWAM_LATENT_CACHE_ROOT}" DEVICES="${DEVICES}" TASK="${TASK}" \
    bash scripts/ensure_latent_cache.sh
fi

LAUNCHER="scripts/train_zero1.sh"
if [[ "${ZERO_STAGE:-1}" == "2" ]]; then LAUNCHER="scripts/train_zero2.sh"; fi
exec bash "${LAUNCHER}" "${GPUS}" "${OVERRIDES[@]}"
