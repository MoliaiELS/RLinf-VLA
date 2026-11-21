#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit the online acceptance test through sbatch}"
: "${SLURM_CPUS_PER_TASK:?Slurm must assign the CPU budget}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the Slurm assigned GPU mask}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$(hostname)" != 4090node1 ]] || { echo 'Run online training on a compute node' >&2; exit 2; }
repo="$(realpath "${1:?fixed repository worktree}")"
prefix="$(realpath "${2:?isolated environment prefix}")"
output_root="${3:?personal-storage output root}"
robotwin_root="${4:?pinned native RoboTwin checkout}"
model_path="${5:?verified model directory}"
algorithm="${6:?ppo or grpo}"
minutes="${7:?Slurm wall-time minutes}"
commit="${8:?fixed complete Git commit}"
default_config="robotwin_${algorithm}_xvla_acceptance"
if [[ "${XVLA_SAMPLER_DIAGNOSTIC:-0}" == 1 ]]; then default_config="robotwin_${algorithm}_xvla"; fi
config="${9:-$default_config}"
[[ "$algorithm" == ppo || "$algorithm" == grpo ]]
[[ "$config" == "robotwin_${algorithm}_xvla" || "$config" == "robotwin_${algorithm}_xvla_acceptance" ]]
[[ "$minutes" =~ ^[0-9]+$ && "$minutes" -ge 1 && "$minutes" -le 30 ]]
[[ "$commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$output_root" = /* && "$output_root" != / ]]
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -r "$prefix/bin/activate"
test -x "$prefix/bin/python"
source "$prefix/bin/activate"
# Compute nodes may lack the submitter's passwd entry.
export USER="${SLURM_JOB_USER:-${LOGNAME:-${USER:?Slurm submitter name is required}}}"
export LOGNAME="$USER"
export XVLA_RUN_DIR="$output_root/$SLURM_JOB_ID/$algorithm"
export ROBOTWIN_PATH="$robotwin_root" XVLA_MODEL_PATH="$model_path"
export XVLA_ONLINE_ACCEPTANCE=1 USE_TF=0 USE_TORCH=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export WARP_CACHE_PATH="$XVLA_RUN_DIR/warp-cache"
ray_temp="/tmp/xvla_rl_${USER}_${SLURM_JOB_ID}"
mkdir -p "$XVLA_RUN_DIR" "$WARP_CACHE_PATH"
test ! -e "$XVLA_RUN_DIR/online-job.json"
python - "$repo" "$commit" "$minutes" <<'PY'
import json
import os
import pathlib
import sys
import time

record = {
    "code_root": sys.argv[1], "code_commit": sys.argv[2],
    "time_limit_minutes": int(sys.argv[3]), "started_epoch": time.time(),
    "job_id": os.environ["SLURM_JOB_ID"], "hostname": os.uname().nodename,
    "python": sys.executable, "prefix": sys.prefix,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "sampler_diagnostic": os.environ.get("XVLA_SAMPLER_DIAGNOSTIC", "0") == "1",
    "slurm_cpus": os.environ["SLURM_CPUS_PER_TASK"],
    "physical_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    "memory_isolation": "No Slurm memory request; physical memory is not reliably isolated",
}
(pathlib.Path(os.environ["XVLA_RUN_DIR"]) / "online-job.json").write_text(json.dumps(record, indent=2) + "\n")
PY
cleanup() {
    result=$?
    trap - EXIT TERM INT
    cleanup_result=0
    python "$repo/tools/xvla/online_submit.py" --cleanup-ray-temp "$ray_temp" > "$XVLA_RUN_DIR/cleanup.json" 2> "$XVLA_RUN_DIR/cleanup.err" || cleanup_result=$?
    if [[ "$result" == 0 && "$cleanup_result" != 0 ]]; then result="$cleanup_result"; fi
    python - "$result" "$cleanup_result" <<'PY'
import json
import os
import pathlib
import sys
import time

path = pathlib.Path(os.environ["XVLA_RUN_DIR"]) / "online-job.json"
record = json.loads(path.read_text())
record.update(exit_code=int(sys.argv[1]), cleanup_exit_code=int(sys.argv[2]), finished_epoch=time.time())
path.write_text(json.dumps(record, indent=2) + "\n")
PY
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Leave one minute for scoped Ray cleanup before Slurm's hard wall-time limit.
seconds=$((minutes * 60 - 60))
if (( seconds < 30 )); then seconds=30; fi
cd "$repo"
timeout --signal=TERM --kill-after=15s "${seconds}s" bash tools/xvla/train.sh "$config" "runner.logger.log_path=$XVLA_RUN_DIR"
