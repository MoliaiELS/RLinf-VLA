#!/usr/bin/env bash
set -euo pipefail
parse_login_training_arguments() {
    [[ "$#" -ge 8 ]] || { echo 'Usage: login_training.sh FIXED_REPO ENV_PREFIX NEW_OUTPUT ROBOTWIN_ROOT MODEL_PATH GPU_UUID ALGORITHM RUN_ID [CONFIG_NAME [HYDRA_OVERRIDE...]]' >&2; return 2; }
    repo_arg="$1" prefix_arg="$2" output_arg="$3" robotwin_arg="$4" model_arg="$5"
    gpu_uuid="$6" algorithm="$7" debug_id="$8"
    config="${9-robotwin_${algorithm}_xvla_acceptance}"
    [[ "$algorithm" == ppo || "$algorithm" == grpo ]] || return 2
    [[ "$config" == "robotwin_${algorithm}_xvla" || "$config" == "robotwin_${algorithm}_xvla_acceptance" ]] || return 2
    if (( $# >= 9 )); then shift 9; else shift 8; fi
    hydra_overrides=("$@")
}
parse_login_training_arguments "$@"
[[ "$(hostname)" == 4090node1 && ! -v SLURM_JOB_ID ]] || { echo 'Login training debug requires 4090node1 outside Slurm' >&2; exit 2; }
repo="$(realpath -e "$repo_arg")"
prefix="$(realpath -e "$prefix_arg")"
[[ "$output_arg" = /* && "$output_arg" != / ]]
output="$(realpath -m "$output_arg")"
robotwin_root="$(realpath -e "$robotwin_arg")"
model_path="$(realpath -e "$model_arg")"
[[ "$debug_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$ ]]
[[ "$gpu_uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]
export XVLA_LOGIN_SECONDS="${XVLA_LOGIN_SECONDS:-1200}"
[[ "$XVLA_LOGIN_SECONDS" =~ ^[1-9][0-9]{1,3}$ && "$XVLA_LOGIN_SECONDS" -ge 30 && "$XVLA_LOGIN_SECONDS" -lt 1500 ]]
commit="$(git -C "$repo" rev-parse HEAD)"
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$(basename "$repo")" == "$commit" ]]
if git -C "$repo" symbolic-ref -q HEAD >/dev/null; then
    echo 'Use a detached fixed-commit worktree' >&2
    exit 2
fi
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]]
for protected in "$repo" "$prefix" "$robotwin_root" "$model_path"; do
    [[ "$output" != "$protected" && "$output" != "$protected/"* ]]
done
[[ ! -e "$output" ]]
test -r "$prefix/bin/activate"
test -x "$prefix/bin/python"
test -r "$model_path/verified.json"
test -r "$repo/examples/embodiment/config/$config.yaml"
source "$prefix/bin/activate"
export USER="$(id -un)" LOGNAME="$(id -un)"
export XVLA_LOGIN_TRAINING_DEBUG=1 XVLA_LOGIN_RUN_ID="$debug_id" XVLA_LOGIN_CPUS=4
export XVLA_ONLINE_ACCEPTANCE=1 XVLA_RUN_DIR="$output"
export ROBOTWIN_PATH="$robotwin_root" XVLA_MODEL_PATH="$model_path"
export CUDA_VISIBLE_DEVICES="$gpu_uuid" PYTHONPATH="$repo"
export RLINF_XVLA_RAY_CPUS=4 RLINF_XVLA_RAY_OBJECT_STORE_BYTES=2147483648
RAY_TMPDIR="$(python -c 'from tools.xvla.online_submit import private_ray_marker; print(private_ray_marker())')"
export RAY_TMPDIR
export FYP_XVLA_RAY_SPILL="$output/ray-spill-${debug_id}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_TORCH=1 NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
export WARP_CACHE_PATH="$output/warp-cache"
unset XVLA_LOGIN_DIAGNOSTIC XVLA_SAMPLER_DIAGNOSTIC RAY_ADDRESS
[[ ! -e "$RAY_TMPDIR" ]]
mkdir -p "$output" "$WARP_CACHE_PATH"
cpus="$(python -c 'import os; a=sorted(os.sched_getaffinity(0))[:4]; assert len(a)==4; print(",".join(map(str,a)))')"

bounded_training() {
    set -euo pipefail
    export -n -f bounded_training
    repo="$1"
    config="$2"
    shift 2
    watcher_pid=
    cleanup() {
        result=$?
        trap - EXIT TERM INT
        if [[ -n "$watcher_pid" ]]; then kill "$watcher_pid" 2>/dev/null || true; fi
        cleanup_result=0
        python "$repo/tools/xvla/online_submit.py" --cleanup-ray-temp "$RAY_TMPDIR" > "$XVLA_RUN_DIR/cleanup.json" 2> "$XVLA_RUN_DIR/cleanup.err" || cleanup_result=$?
        if [[ "$result" == 0 && "$cleanup_result" != 0 ]]; then result="$cleanup_result"; fi
        if [[ -e "$XVLA_RUN_DIR/resource-stop.json" && "$result" == 0 ]]; then result=75; fi
        python - "$result" "$cleanup_result" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(os.environ["XVLA_RUN_DIR"]) / "login-training.json"
record = json.loads(path.read_text()) if path.exists() else {}
record.update(exit_code=int(sys.argv[1]), cleanup_exit_code=int(sys.argv[2]), finished_epoch=time.time())
path.write_text(json.dumps(record, indent=2) + "\n")
PY
        exit "$result"
    }
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    python - "$repo" "$config" "$@" <<'PY'
import csv, json, os, pathlib, subprocess, sys, time

def idle():
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    memory = int(subprocess.check_output(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5).strip())
    rows = csv.reader(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines())
    pids = [int(row[1]) for row in rows if row and row[0].strip().lower() == uuid.lower()]
    if memory >= 256 or pids:
        raise RuntimeError(f"Select an idle login GPU: memory={memory} MiB, compute_pids={pids}")
    return {"checked_epoch": time.time(), "memory_used_mib": memory, "compute_pids": pids}

checks = [idle(), idle()]
record = {
    "scope": "Real RLinf online acceptance in bounded login training debug",
    "code_root": sys.argv[1], "config_name": sys.argv[2], "hydra_overrides": sys.argv[3:],
    "code_commit": subprocess.check_output(["git", "-C", sys.argv[1], "rev-parse", "HEAD"], text=True).strip(),
    "started_epoch": time.time(), "run_id": os.environ["XVLA_LOGIN_RUN_ID"],
    "hostname": os.uname().nodename, "slurm_job_id": None,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_idle_checks": checks,
    "cpu_affinity": sorted(os.sched_getaffinity(0)), "ray_cpus": 4,
    "ray_temp": os.environ["RAY_TMPDIR"], "ray_spill": os.environ["FYP_XVLA_RAY_SPILL"],
    "ray_object_store_bytes": 2 * 1024**3, "spill_limit_bytes": 2 * 1024**3,
    "time_limit_seconds": int(os.environ["XVLA_LOGIN_SECONDS"]),
    "nice": os.getpriority(os.PRIO_PROCESS, 0), "process_group": os.getpgrp(),
    "memory_isolation": "Login RAM is not scheduler-isolated",
}
(pathlib.Path(os.environ["XVLA_RUN_DIR"]) / "login-training.json").write_text(json.dumps(record, indent=2) + "\n")
PY
    python - <<'PY' &
import csv, json, os, pathlib, signal, subprocess, time

root = pathlib.Path(os.environ["XVLA_RUN_DIR"])
spill = pathlib.Path(os.environ["FYP_XVLA_RAY_SPILL"])
uuid = os.environ["CUDA_VISIBLE_DEVICES"]
run_tag = ("XVLA_LOGIN_RUN_ID=" + os.environ["XVLA_LOGIN_RUN_ID"]).encode()
while True:
    time.sleep(1)
    reason = None
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=5)
        foreign = []
        for row in csv.reader(text.splitlines()):
            if not row or row[0].strip().lower() != uuid.lower():
                continue
            pid = int(row[1])
            try:
                folder = pathlib.Path("/proc") / str(pid)
                own = folder.stat().st_uid == os.getuid()
                if not own or run_tag not in (folder / "environ").read_bytes().split(b"\0"):
                    foreign.append(pid)
            except FileNotFoundError:
                pass
        size = 0
        for path in spill.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    size += path.stat().st_size
            except FileNotFoundError:
                pass
        if foreign:
            reason = {"foreign_compute_pids": foreign}
        elif size > 2 * 1024**3:
            reason = {"spill_limit_exceeded_bytes": size}
    except Exception as error:
        reason = {"resource_watch_failed": str(error)}
    if reason is not None:
        (root / "resource-stop.json").write_text(json.dumps({"reason": reason, "time": time.time()}, indent=2) + "\n")
        os.killpg(os.getpgrp(), signal.SIGTERM)
        break
PY
    watcher_pid=$!
    bash "$repo/tools/xvla/train.sh" "$config" "runner.logger.log_path=$XVLA_RUN_DIR" "$@"
}
export -f bounded_training
cd "$repo"
# Reserve 30 seconds for TERM, KILL and scoped Ray cleanup inside the budget.
term_seconds=$((XVLA_LOGIN_SECONDS - 30))
if (( term_seconds < 1 )); then term_seconds=1; fi
result=0
taskset -c "$cpus" setsid --wait timeout --signal=TERM --kill-after=15s "${term_seconds}s" nice -n 10 \
    bash -c 'bounded_training "$@"' -- "$repo" "$config" "${hydra_overrides[@]}" > "$output/login-training.log" 2>&1 || result=$?
# Ray daemons may have detached from the private foreground process group.
python "$repo/tools/xvla/online_submit.py" --cleanup-ray-temp "$RAY_TMPDIR" > "$output/cleanup-final.json" 2> "$output/cleanup-final.err" || { if [[ "$result" == 0 ]]; then result=1; fi; }
if [[ -e "$output/resource-stop.json" && "$result" == 0 ]]; then result=75; fi
python - "$result" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(os.environ["XVLA_RUN_DIR"]) / "login-training.json"
record = json.loads(path.read_text()) if path.exists() else {}
record.update(launcher_exit_code=int(sys.argv[1]), launcher_finished_epoch=time.time())
path.write_text(json.dumps(record, indent=2) + "\n")
PY
exit "$result"
