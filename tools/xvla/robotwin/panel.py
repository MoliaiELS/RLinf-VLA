"""Run independently recoverable task/condition shards inside one Slurm allocation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PRIORITY = ("move_stapler_pad", "turn_switch", "click_alarmclock", "move_pillbottle_pad", "move_can_pot",
            "open_microwave", "pick_dual_bottles", "lift_pot", "place_bread_basket", "press_stapler")
MISSING_IN_RLINF = ("place_fan", "open_laptop", "place_object_scale", "put_object_cabinet")


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(args):
    if args.mode == "probe":
        from bind_renderer import probe

        probe()
        model_root = Path(os.environ["FYP_ROBOTWIN_MODEL"])
        if (model_root / "verified.json").is_file():
            probe_model(model_root)
        return 0
    root = Path(os.environ["FYP_ROBOTWIN_DATA"])
    job = Path(os.environ["FYP_ROBOTWIN_JOB"])
    code = json.loads((job / "runtime.json").read_text())["code"]["repository"]["commit"]
    output = root / ("easy-preflight" if args.mode == "easy-preflight" else "panels") / code
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"lane-{args.lane}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_lane(args, job, output, code)


def run_lane(args, job, output, code):
    conditions = ("easy", "hard") if args.lane == "both" else (args.lane,)
    preflight = args.mode == "easy-preflight"
    if preflight and conditions != ("easy",):
        raise ValueError("Easy preflight requires --lane easy")
    phase = "coverage" if preflight else args.mode
    smoke = phase in ("smoke", "coverage")
    if preflight or args.mode == "smoke":
        tasks = PRIORITY[:2]
    elif args.mode == "coverage":
        tasks = MISSING_IN_RLINF
    elif args.mode == "extension":
        tasks = PRIORITY
    else:
        tasks_json = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
        tasks = tuple(item["task_id"] for item in tasks_json["tasks"])
        assert len(set(tasks)) == 50 and set(PRIORITY) <= set(tasks)
        tasks = PRIORITY + tuple(t for t in tasks if t not in PRIORITY)
    if not smoke:
        for condition in ("easy", "hard"):
            if not ((output / "gate" / f"smoke-{condition}.json").is_file()
                    or (output / "gate" / "smoke-both.json").is_file()):
                raise ValueError("Both conditions must pass the same commit's parity and smoke jobs")
    phases = [2, 3] if args.mode == "extension" else [1]
    states = []
    for batch in phases:
        for task in tasks:
            for condition in conditions:
                if args.task and task != args.task:
                    continue
                base = ["--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"], "--task", task, "--condition", condition,
                        "--phase", phase if smoke else "formal", "--output", str(output),
                        "--limit", str(1 if phase == "coverage" else 5 if smoke else 300 if task in PRIORITY else 100)]
                model = ["--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"],
                         "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"]]
                steps = [("screen", []), ("eval", model + ["--batch", str(batch)])]
                if args.mode == "smoke":
                    steps.insert(1, ("parity", model))
                ok = True
                for step, opts in steps:
                    started = time.monotonic()
                    proc = subprocess.run([sys.executable, "-u", "-m", "tools.xvla.robotwin.runner", step] + base + opts)
                    state = dict(task_id=task, condition=condition, mode=args.mode, step=step, batch=batch,
                                 returncode=proc.returncode, seconds=round(time.monotonic() - started, 3))
                    append(job / "shards.jsonl", state)
                    states.append(state)
                    if proc.returncode:
                        ok = False
                        break
                print(f"ROBOTWIN_SHARD {task} {condition} {batch}: {'PASS' if ok else 'INCOMPLETE'}", flush=True)
    if not states or any(row["returncode"] for row in states):
        return 1
    if args.mode == "smoke" and not args.task:
        gate = output / "gate"
        gate.mkdir(exist_ok=True)
        tmp = gate / f"smoke-{args.lane}.json.tmp"
        tmp.write_text(json.dumps({"commit": code, "job_id": os.environ["SLURM_JOB_ID"], "shards": states}, indent=2) + "\n")
        tmp.replace(gate / f"smoke-{args.lane}.json")
    return 0


def probe_model(model_root):
    import numpy as np
    import torch
    from tools.xvla.robotwin import protocol
    from tools.xvla.robotwin.parity import reference_client
    from tools.xvla.robotwin.runner import checkpoint, pinned_checkout

    policy, _ = checkpoint(model_root, "cuda:0", True)
    images = np.random.default_rng(0)
    obs = {"observation": {camera: {"rgb": images.integers(0, 256, protocol.RGB_SHAPE, dtype=np.uint8)}
                           for camera in (*protocol.CAMERAS, "front_camera")},
           "endpose": {"left_endpose": [0.1, 0.2, 0.3, 1., 0., 0., 0.], "left_gripper": 0.5,
                       "right_endpose": [-0.1, 0.3, 0.2, 0.5, 0.5, 0.5, -0.5], "right_gripper": 0.8}}
    instruction, seed = "move stapler pad", 0
    policy.begin(instruction, seed)
    reference_root = Path(os.environ["FYP_ROBOTWIN_XVLA"])
    pinned_checkout(reference_root, protocol.XVLA_COMMIT)
    evidence = {"input": "fixed synthetic camera arrays and poses", "policy_seed": seed}
    with torch.inference_mode():
        official = reference_client(reference_root, policy, seed, policy.inputs(obs), evidence)
        official.set_instruction(instruction)
        raw = policy.step(obs)
        official_raw = official.step(obs)
    np.testing.assert_allclose(raw, official_raw, rtol=1e-5, atol=1e-6)
    commands = protocol.actions_to_env(raw)
    evidence.update(raw_shape=raw.shape, ee_shape=commands.shape,
                    raw_max_abs_diff=float(np.max(np.abs(raw - official_raw))),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
    print("ROBOTWIN_MODEL_PROBE " + json.dumps(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "easy-preflight", "smoke", "coverage", "formal", "extension"])
    parser.add_argument("--lane", choices=["easy", "hard", "both"], default="both")
    parser.add_argument("--task")
    raise SystemExit(run(parser.parse_args()))
