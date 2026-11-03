#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CPU_BASE="${CPU_BASE:-128}"
CPUS_PER_RANK="${CPUS_PER_RANK:-16}"
SIBLING_OFFSET="${SIBLING_OFFSET:-0}"
AFFINITY_LOG="${AFFINITY_LOG:-${REPO_ROOT}/affinity_$(date +%Y%m%d_%H%M%S).log}"

bind_loop() {
  declare -A seen=()
  while :; do
    for pid in $(pgrep -f "scripts/train.py" 2>/dev/null); do
      [[ -n "${seen[$pid]:-}" ]] && continue
      lr="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^LOCAL_RANK=//p' | head -1)"
      [[ -z "${lr}" ]] && continue
      lo=$((CPU_BASE + lr * CPUS_PER_RANK))
      hi=$((lo + CPUS_PER_RANK - 1))
      cpus="${lo}-${hi}"
      if ((SIBLING_OFFSET > 0)); then
        cpus="${cpus},$((lo + SIBLING_OFFSET))-$((hi + SIBLING_OFFSET))"
      fi
      if taskset -apc "${cpus}" "${pid}" >/dev/null 2>&1; then
        echo "rank=${lr} pid=${pid} target=${cpus}" >> "${AFFINITY_LOG}"
        seen[$pid]=1
      fi
    done
    sleep 0.05
  done
}

bind_loop &
BIND_PID=$!
trap 'kill "${BIND_PID}" 2>/dev/null' EXIT

bash scripts/run_libero_smoke.sh
status=$?
kill "${BIND_PID}" 2>/dev/null
exit "${status}"
