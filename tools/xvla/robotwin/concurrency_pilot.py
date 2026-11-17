"""Compare serial and concurrent Easy rollouts on one allocated Slurm GPU."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


EVALUATION_COMMIT = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
TASKS = ("move_stapler_pad", "turn_switch")
METRICS = ("status", "reason", "actual_seed", "policy_seed", "queries", "anchors_sent", "anchors",
           "physics_ticks", "left_plan_failures", "right_plan_failures", "wall_seconds")


def load_qualification(panel):
    gate = json.loads((panel / "gate/smoke-easy.json").read_bytes())
    passed = {(row["task_id"], row["step"]) for row in gate["shards"]
              if row["condition"] == "easy" and row["mode"] == "smoke" and row["returncode"] == 0}
    required = {(task, step) for task in TASKS for step in ("screen", "parity", "eval")}
    if gate["commit"] != EVALUATION_COMMIT or not required <= passed:
        raise ValueError("Both pilot tasks require Easy smoke qualification on the frozen evaluation")
    qualified = {}
    for task in TASKS:
        raw = (panel / "smoke" / task / "easy/seed_manifest.json").read_bytes()
        manifest = json.loads(raw)
        episodes = {row["episode_id"]: (row["actual_seed"], row["policy_seed"]) for row in manifest["episodes"]}
        if ((manifest["task"], manifest["condition"], manifest["phase"], manifest["limit"]) != (task, "easy", "smoke", 5)
                or len(episodes) != 5 or any(row["batch"] != 1 for row in manifest["episodes"])):
            raise ValueError(f"Unexpected Easy smoke manifest for {task}")
        qualified[task] = (raw, episodes)
    return gate["job_id"], qualified


def read_episodes(path, expected):
    rows = path.read_bytes().splitlines(keepends=True) if path.is_file() else []
    attempts = Counter()
    reasons = Counter()
    metrics = {}
    for line in rows:
        if not line.endswith(b"\n"):
            reasons["partial_journal_row"] += 1
            break
        row = json.loads(line)
        episode_id = row["episode_id"]
        if (row["actual_seed"], row["policy_seed"]) != expected.get(episode_id):
            raise ValueError(f"Pilot episode differs from qualified manifest: {episode_id}")
        if metrics.get(episode_id, {}).get("status") in {"success", "failure", "unresolved"}:
            raise ValueError(f"Pilot episode repeated after a terminal result: {episode_id}")
        attempts[row["status"]] += 1
        reasons[row["reason"]] += 1
        metrics[episode_id] = {key: row.get(key) for key in METRICS}
    completed = {key for key, row in metrics.items() if row["status"] in {"success", "failure"}}
    return {"complete": completed == expected.keys() and "partial_journal_row" not in reasons,
            "attempt_statuses": dict(attempts), "attempt_reasons": dict(reasons), "episodes": metrics}


def run_worker(args, scenario, task, qualified):
    root = args.output / scenario
    raw, expected = qualified
    shard = root / "smoke" / task / "easy"
    log = root / f"{task}.log"
    command = [sys.executable, "-u", "-m", "tools.xvla.robotwin.runner", "eval",
               "--robotwin-root", str(args.robotwin_root), "--xvla-root", str(args.xvla_root),
               "--checkpoint", str(args.checkpoint), "--output", str(root), "--task", task,
               "--condition", "easy", "--phase", "smoke", "--batch", "1"]
    env = dict(os.environ, PYTHONPATH=str(args.evaluation_root), PYTHONDONTWRITEBYTECODE="1",
               WARP_CACHE_PATH=str(root / "cache" / task), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    result = {"task": task, "log": str(log.relative_to(args.output)), "returncode": None,
              "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    started = time.perf_counter()
    try:
        shard.mkdir(parents=True)
        (shard / "seed_manifest.json").write_bytes(raw)
        with log.open("wb") as stream:
            result["returncode"] = subprocess.run(command, cwd=args.evaluation_root, env=env, stdout=stream,
                                                  stderr=subprocess.STDOUT).returncode
        result.update(read_episodes(shard / "episodes.jsonl", expected))
    except Exception:
        result.update(complete=False, traceback=traceback.format_exc())
    result["wall_seconds"] = time.perf_counter() - started
    return result


def run_scenario(args, scenario, qualified):
    started = time.perf_counter()
    workers = 1 if scenario == "serial" else 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_worker, args, scenario, task, qualified[task]) for task in TASKS]
        results = [future.result() for future in futures]
    return {"wall_seconds": time.perf_counter() - started, "workers": results,
            "complete": all(row["returncode"] == 0 and row["complete"] for row in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-root", "input-panel", "output", "robotwin-root", "xvla-root", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not os.environ.get("SLURM_JOB_ID") or not visible or "," in visible or visible in {"-1", "NoDevFiles"}:
        parser.error("Pilot requires one Slurm-assigned visible GPU")
    if args.input_panel.name != EVALUATION_COMMIT or args.input_panel.parent.name != "panels":
        parser.error("Expected the original evaluation under its qualified panels directory")
    read_only = (args.input_panel.parents[1], args.evaluation_root, args.robotwin_root, args.xvla_root, args.checkpoint)
    if any(args.output.is_relative_to(root) for root in read_only):
        parser.error("Pilot output must be outside all read-only inputs and the qualified results tree")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"label": "Easy same-GPU concurrency pilot", "job_id": os.environ["SLURM_JOB_ID"],
               "cuda_visible_devices": visible, "evaluation_commit": EVALUATION_COMMIT,
               "qualified_panel": str(args.input_panel), "scenarios": {}}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.evaluation_root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.evaluation_root, text=True)
        if revision != EVALUATION_COMMIT or dirty:
            raise ValueError("Pilot requires the clean frozen evaluation commit")
        summary["qualification_job"], qualified = load_qualification(args.input_panel)
        for scenario in ("serial", "concurrent"):
            summary["scenarios"][scenario] = run_scenario(args, scenario, qualified)
        serial, concurrent = (summary["scenarios"][name] for name in ("serial", "concurrent"))
        summary["complete"] = serial["complete"] and concurrent["complete"]
        if summary["complete"]:
            summary["serial_over_concurrent"] = serial["wall_seconds"] / concurrent["wall_seconds"]
            summary["episodes_per_hour"] = {name: 3600 * sum(len(row["episodes"]) for row in data["workers"])
                                            / data["wall_seconds"] for name, data in summary["scenarios"].items()}
            summary["outcome_changes"] = [episode_id for before, after in zip(serial["workers"], concurrent["workers"])
                                          for episode_id, row in before["episodes"].items()
                                          if row["status"] != after["episodes"][episode_id]["status"]]
    except Exception:
        summary.update(complete=False, traceback=traceback.format_exc())
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
