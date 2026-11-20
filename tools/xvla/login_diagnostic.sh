#!/usr/bin/env bash
set -euo pipefail
[[ "$#" == 6 ]] || { echo 'Usage: login_diagnostic.sh FIXED_REPO ENV_PREFIX NEW_OUTPUT ROBOTWIN_ROOT MODEL_PATH GPU_UUID' >&2; exit 2; }
[[ "$(hostname)" == 4090node1 ]] || { echo 'This diagnostic is restricted to 4090node1' >&2; exit 2; }
[[ ! -v SLURM_JOB_ID ]] || { echo 'Use the Slurm diagnostic launcher inside an allocation' >&2; exit 2; }
repo="$(realpath -e "$1")"
prefix="$(realpath -e "$2")"
[[ "$3" = /* && "$3" != / ]] || { echo 'Choose an absolute personal-storage output directory' >&2; exit 2; }
output="$(realpath -m "$3")"
robotwin_root="$(realpath -e "$4")"
model_path="$(realpath -e "$5")"
gpu_uuid="$6"
[[ "$gpu_uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]
commit="$(git -C "$repo" rev-parse HEAD)"
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$(basename "$repo")" == "$commit" ]]
if git -C "$repo" symbolic-ref -q HEAD >/dev/null; then
    echo 'Use a detached fixed-commit worktree' >&2
    exit 2
fi
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]] || { echo 'The fixed worktree must be clean' >&2; exit 2; }
for protected in "$repo" "$prefix" "$robotwin_root" "$model_path"; do
    [[ "$output" != "$protected" && "$output" != "$protected/"* ]] || { echo 'Keep diagnostic output outside code, environment and model directories' >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo 'Choose a new diagnostic output directory' >&2; exit 2; }
test -r "$prefix/bin/activate"
test -x "$prefix/bin/python"
test -r "$repo/tools/xvla/sampler_diagnostic.py"
test -r "$model_path/verified.json"
source "$prefix/bin/activate"
export USER="$(id -un)" LOGNAME="$(id -un)"
export XVLA_LOGIN_DIAGNOSTIC=1 XVLA_RUN_DIR="$output"
export XVLA_LOGIN_SHARE_FREE_CAPACITY="${XVLA_LOGIN_SHARE_FREE_CAPACITY:-0}"
[[ "$XVLA_LOGIN_SHARE_FREE_CAPACITY" == 0 || "$XVLA_LOGIN_SHARE_FREE_CAPACITY" == 1 ]]
export XVLA_LOGIN_MAX_TORCH_GIB=10
export XVLA_LOGIN_SECONDS="${XVLA_LOGIN_SECONDS:-1200}"
[[ "$XVLA_LOGIN_SECONDS" =~ ^[1-9][0-9]{1,3}$ && "$XVLA_LOGIN_SECONDS" -ge 30 && "$XVLA_LOGIN_SECONDS" -le 1200 ]] || { echo 'XVLA_LOGIN_SECONDS must be between 30 and 1200' >&2; exit 2; }
term_seconds=$((XVLA_LOGIN_SECONDS - 15))
export ROBOTWIN_PATH="$robotwin_root" XVLA_MODEL_PATH="$model_path"
export CUDA_VISIBLE_DEVICES="$gpu_uuid" PYTHONPATH="$repo"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_TORCH=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
export WARP_CACHE_PATH="$output/warp-cache"
unset XVLA_ONLINE_ACCEPTANCE XVLA_SAMPLER_DIAGNOSTIC RLINF_XVLA_ISOLATED_RAY RAY_ADDRESS
mkdir -p "$output" "$WARP_CACHE_PATH"
cd "$repo"
# A new session confines timeout and the GPU watcher to our own process group.
# Reserve 15s for KILL within the selected diagnostic budget.
exec setsid --wait timeout --signal=TERM --kill-after=15s "${term_seconds}s" nice -n 10 \
    "$prefix/bin/python" - "$repo" "$commit" "$gpu_uuid" > "$output/login-diagnostic.log" 2>&1 <<'PY'
import csv
import importlib.util
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

repo, commit, requested = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
root = pathlib.Path(os.environ["XVLA_RUN_DIR"])
time_limit_seconds = int(os.environ["XVLA_LOGIN_SECONDS"])
share_free_capacity = os.environ["XVLA_LOGIN_SHARE_FREE_CAPACITY"] == "1"
if socket.gethostname() != "4090node1" or "SLURM_JOB_ID" in os.environ:
    raise RuntimeError("Login diagnostics require 4090node1 outside Slurm")
cpus = sorted(os.sched_getaffinity(0))[:4]
if len(cpus) != 4:
    raise RuntimeError("The login diagnostic requires four available CPU cores")
os.sched_setaffinity(0, cpus)

def query(arguments):
    return list(csv.reader(subprocess.check_output(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).splitlines()))

def idle_gpu():
    rows = query([f"--id={requested}", "--query-gpu=uuid,pci.bus_id,memory.used,memory.free,name"])
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError(f"Expected exactly one GPU for {requested}: {rows}")
    uuid, pci, used, free, name = (value.strip() for value in rows[0])
    if uuid.lower() != requested.lower():
        raise RuntimeError(f"Requested GPU UUID does not match: {rows[0]}")
    processes = query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory"])
    pids, foreign = [], []
    for row in processes:
        if not row or row[0].strip().lower() != uuid.lower():
            continue
        pid = int(row[1])
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        pids.append(pid)
        if group != os.getpgrp():
            foreign.append({"pid": pid, "memory_used_mib": int(row[2])})
    foreign_memory = sum(process["memory_used_mib"] for process in foreign)
    if share_free_capacity:
        if int(free) < 16 * 1024 or foreign_memory > 6 * 1024:
            raise RuntimeError(f"Insufficient shared GPU capacity: free={free} MiB, foreign={foreign_memory} MiB")
    elif int(used) >= 256 or pids:
        raise RuntimeError(f"Requested GPU is not idle: used={used} MiB, compute_pids={pids}")
    return {"uuid": uuid, "pci_bus_id": pci, "memory_used_mib": int(used),
            "memory_free_mib": int(free), "name": name, "compute_pids": pids,
            "foreign_compute_processes": foreign, "foreign_compute_memory_mib": foreign_memory,
            "checked_epoch": time.time()}

first_check = idle_gpu()
import torch
import rlinf

origin = pathlib.Path(rlinf.__file__).resolve()
if not origin.is_relative_to(repo):
    raise RuntimeError(f"Expected RLinf from {repo}, got {origin}")
# Import Torch before this second check to keep the CUDA-init race window small.
last_check = idle_gpu()
if torch.cuda.device_count() != 1:
    raise RuntimeError("The login diagnostic requires exactly one visible GPU")
properties = torch.cuda.get_device_properties(0)
uuid = str(properties.uuid).lower().removeprefix("gpu-")
if uuid != requested.lower().removeprefix("gpu-"):
    raise RuntimeError(f"Visible CUDA UUID {uuid} does not match {requested}")
pci = re.fullmatch(r"[\da-fA-F]+:([\da-fA-F]{2}):([\da-fA-F]{2})\.([\da-fA-F])", last_check["pci_bus_id"])
if not pci:
    raise RuntimeError(f"Invalid allocated GPU PCI address: {last_check['pci_bus_id']}")
vulkan = f"pci:{pci[1]}:{pci[2]}.{pci[3]}"
os.environ["FYP_ROBOTWIN_VULKAN_DEVICE"] = vulkan
record = {
    "code_root": str(repo), "code_commit": commit,
    "hostname": socket.gethostname(), "started_epoch": time.time(),
    "python": sys.executable, "prefix": sys.prefix, "torch": torch.__version__,
    "imports": {name: getattr(importlib.util.find_spec(name), "origin", None)
                for name in ("rlinf", "transformers", "xvla_rlinf")},
    "login_diagnostic": True, "scope": "Frozen sampler diagnosis; no training or Ray",
    "time_limit_seconds": time_limit_seconds, "term_after_seconds": time_limit_seconds - 15, "kill_after_seconds": 15,
    "cuda_visible_devices": requested, "gpu_uuid": uuid, "gpu_model": properties.name,
    "vulkan_device": vulkan, "gpu_idle_checks": [first_check, last_check],
    "share_free_capacity": share_free_capacity, "max_torch_memory_gib": 10,
    "gpu_preflight_policy": "shared_free_capacity" if share_free_capacity else "idle_only",
    "cpu_affinity": cpus, "threads": 4, "nice": os.getpriority(os.PRIO_PROCESS, 0),
    "process_group": os.getpgrp(), "gpu_watch_interval_seconds": 1 if share_free_capacity else 10,
}
with (root / "login-runtime.json").open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")

# This CPU-only watcher shares the private timeout group. It never signals a
# foreign compute process, and also clears any children left after our parent exits.
watcher = r'''
import csv, json, os, pathlib, signal, subprocess, sys, time
uuid, parent, group, output = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
shared = os.environ["XVLA_LOGIN_SHARE_FREE_CAPACITY"] == "1"
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(1 if shared else 10)
    reason = None
    try:
        if os.getpgid(parent) != group:
            reason = "diagnostic_parent_exited"
    except ProcessLookupError:
        reason = "diagnostic_parent_exited"
    if reason is None:
        try:
            text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True, timeout=5)
            foreign = []
            for row in csv.reader(text.splitlines()):
                if not row or row[0].strip().lower() != uuid.lower():
                    continue
                pid = int(row[1])
                try:
                    if os.getpgid(pid) != group:
                        foreign.append({"pid": pid, "memory_used_mib": int(row[2])})
                except ProcessLookupError:
                    pass
            if shared:
                used = int(subprocess.check_output(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5).strip())
                foreign_memory = sum(process["memory_used_mib"] for process in foreign)
                if foreign_memory > 8 * 1024 or used > 20 * 1024:
                    reason = {"foreign_compute_processes": foreign, "foreign_compute_memory_mib": foreign_memory, "gpu_memory_used_mib": used}
            elif foreign:
                reason = {"foreign_compute_pids": [process["pid"] for process in foreign]}
        except Exception as error:
            reason = {"gpu_watch_failed": str(error)}
    if reason is not None:
        with (output / "gpu-watch-stop.json").open("w") as stream:
            json.dump({"reason": reason, "time": time.time(), "terminated_group": group}, stream)
        os.killpg(group, signal.SIGTERM)
        time.sleep(5)
        os.killpg(group, signal.SIGKILL)
'''
subprocess.Popen([sys.executable, "-c", watcher, requested, str(os.getpid()),
                  str(os.getpgrp()), str(root)])
os.execv(sys.executable, [sys.executable, "-m", "tools.xvla.sampler_diagnostic",
                        "--config-name", "robotwin_grpo_xvla"])
PY
