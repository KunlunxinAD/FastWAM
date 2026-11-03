#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ROOT="${LATENT_CACHE_ROOT:-${REPO_ROOT}/data/latent_cache}"
DEVICES="${DEVICES:-8,9,10,11,12,13,14,15}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<< "${DEVICES}")}"
TASK="${TASK:-libero_uncond_2cam224_1e-4}"
BATCH="${LATENT_BATCH:-16}"
WORKERS="${LATENT_WORKERS:-6}"
BUDGET="${LATENT_TIME_BUDGET:-1380}"
MAX_ROUNDS="${LATENT_MAX_ROUNDS:-20}"
PORT="${LATENT_MASTER_PORT:-29655}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export XPU_VISIBLE_DEVICES="${DEVICES}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"

check() {
  XPU_VISIBLE_DEVICES="${DEVICES%%,*}" CUDA_VISIBLE_DEVICES="${DEVICES%%,*}" \
    python3 scripts/precompute_video_latents.py \
      --root "${ROOT}" --task "${TASK}" --check
}

check
status=$?
if [[ "${status}" == "0" ]]; then exit 0; fi
if [[ "${status}" == "4" ]]; then exit 4; fi

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  python3 -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${PORT}" \
    scripts/precompute_video_latents.py \
    --root "${ROOT}" --task "${TASK}" \
    --batch "${BATCH}" --workers "${WORKERS}" --time-budget "${BUDGET}"
  check
  status=$?
  [[ "${status}" == "0" ]] && exit 0
done

exit 1
