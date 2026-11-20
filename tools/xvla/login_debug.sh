#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --allow-login-debug ]] || { echo 'Explicit --allow-login-debug is required' >&2; exit 2; }
shift
[[ "$(hostname)" == 4090node1 ]] || { echo 'This bounded debug wrapper is for 4090node1' >&2; exit 2; }
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'Use the Slurm training entry inside an allocation' >&2; exit 2; }
: "${CUDA_VISIBLE_DEVICES:?Select one idle login-node GPU by its UUID}"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Select exactly one GPU UUID' >&2; exit 2; }
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if git -C "$repo" symbolic-ref -q HEAD >/dev/null || [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo 'Use a clean detached fixed-commit worktree for GPU debug' >&2
    exit 1
fi
processes="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$processes" ]] || { echo 'Selected login GPU has active compute processes' >&2; exit 1; }
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$repo"
exec timeout --signal=TERM --kill-after=15s 600s python -m tools.xvla.gpu_debug --allow-login-debug "$@"
