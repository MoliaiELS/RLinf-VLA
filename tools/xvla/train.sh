#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?A single allocated or explicitly selected GPU is required}"
if [[ "${XVLA_LOGIN_TRAINING_DEBUG:-0}" == 1 ]]; then
    [[ "$(hostname)" == 4090node1 && ! -v SLURM_JOB_ID ]]
    [[ "${XVLA_LOGIN_RUN_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$ ]]
    [[ "${XVLA_LOGIN_CPUS:-}" == 4 ]]
    [[ "${XVLA_LOGIN_SECONDS:-}" =~ ^[1-9][0-9]{1,3}$ && "$XVLA_LOGIN_SECONDS" -ge 30 && "$XVLA_LOGIN_SECONDS" -lt 1500 ]]
    run_id="login_${XVLA_LOGIN_RUN_ID}"
else
    : "${SLURM_JOB_ID:?Run X-VLA training inside an allocated Slurm job}"
    : "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"
    run_id="$SLURM_JOB_ID"
fi
: "${ROBOTWIN_PATH:?Set the pinned native RoboTwin checkout}"
: "${XVLA_MODEL_PATH:?Set the verified pinned X-VLA RoboTwin model directory}"
: "${XVLA_RUN_DIR:?Set a new personal-storage output directory}"
config="${1:?Choose robotwin_ppo_xvla or robotwin_grpo_xvla}"
shift
case "$config" in
    robotwin_ppo_xvla|robotwin_grpo_xvla|robotwin_ppo_xvla_acceptance|robotwin_grpo_xvla_acceptance) ;;
    *) echo "Unsupported X-VLA training config: $config" >&2; exit 2 ;;
esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test "$(git -C "$repo" rev-parse --is-inside-work-tree)" = true
if git -C "$repo" symbolic-ref -q HEAD >/dev/null; then
    echo 'Run training from a detached fixed-commit worktree' >&2
    exit 1
fi
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]]; then
    echo 'Commit the training implementation and config before creating the job worktree' >&2
    exit 1
fi
[[ "$XVLA_RUN_DIR" = /* && "$XVLA_RUN_DIR" != / ]]
mkdir -p "$XVLA_RUN_DIR"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_XVLA_ISOLATED_RAY=1
export RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
if [[ "${XVLA_LOGIN_TRAINING_DEBUG:-0}" == 1 ]]; then
    RAY_TMPDIR="$(python -c 'from tools.xvla.online_submit import private_ray_marker; print(private_ray_marker())')"
    export RAY_TMPDIR
    export FYP_XVLA_RAY_SPILL="$XVLA_RUN_DIR/ray-spill-${XVLA_LOGIN_RUN_ID}"
else
    export RAY_TMPDIR="/tmp/xvla_rl_${USER}_${run_id}"
    export FYP_XVLA_RAY_SPILL="$XVLA_RUN_DIR/ray-spill"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_TORCH=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
python - "$repo" <<'PY'
import importlib.util
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import torch
import rlinf

repo = pathlib.Path(sys.argv[1]).resolve()
origin = pathlib.Path(rlinf.__file__).resolve()
if not origin.is_relative_to(repo):
    raise RuntimeError(f"Expected RLinf from {repo}, got {origin}")
modules = {name: getattr(importlib.util.find_spec(name), 'origin', None)
           for name in ('rlinf', 'transformers', 'xvla_rlinf')}
receipt = dict(commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
               python=sys.executable, imports=modules, torch=torch.__version__,
               cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
               slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
               login_training_debug=os.environ.get('XVLA_LOGIN_TRAINING_DEBUG') == '1',
               login_run_id=os.environ.get('XVLA_LOGIN_RUN_ID'),
               login_cpus=os.environ.get('XVLA_LOGIN_CPUS'))
if torch.cuda.device_count() != 1:
    raise RuntimeError('The initial X-VLA training configuration requires exactly one allocated GPU')
properties = torch.cuda.get_device_properties(0)
uuid = str(properties.uuid).lower().removeprefix('gpu-')
rows = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'], text=True
)
matches = [row[1].strip() for row in csv.reader(rows.splitlines())
           if row[0].strip().lower().removeprefix('gpu-') == uuid]
if len(matches) != 1:
    raise RuntimeError('Cannot uniquely map the allocated CUDA UUID to its Vulkan device')
pci = re.fullmatch(r'[\da-fA-F]+:([\da-fA-F]{2}):([\da-fA-F]{2})\.([\da-fA-F])', matches[0])
if not pci:
    raise RuntimeError(f'Invalid allocated GPU PCI address: {matches[0]}')
vulkan = f'pci:{pci[1]}:{pci[2]}.{pci[3]}'
receipt.update(gpu_uuid=uuid, gpu_model=properties.name, vulkan_device=vulkan)
(pathlib.Path(os.environ['XVLA_RUN_DIR']) / 'vulkan-device.txt').write_text(vulkan)
(pathlib.Path(os.environ['XVLA_RUN_DIR']) / 'training-runtime.json').write_text(json.dumps(receipt, indent=2))
PY
export FYP_ROBOTWIN_VULKAN_DEVICE="$(cat "$XVLA_RUN_DIR/vulkan-device.txt")"
cd "$repo"
if [[ "${XVLA_SAMPLER_DIAGNOSTIC:-0}" == 1 ]]; then
    exec python -m tools.xvla.sampler_diagnostic --config-name "$config" "$@"
fi
if [[ "${XVLA_ONLINE_ACCEPTANCE:-0}" == 1 ]]; then
    exec python -m tools.xvla.online_acceptance --config-name "$config" "$@"
fi
exec python examples/embodiment/train_embodied_agent.py --config-name "$config" "$@"
