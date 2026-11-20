# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Submit one bounded, fixed-commit X-VLA online acceptance job."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence


@dataclass(frozen=True)
class Submission:
    """Fixed paths and resources passed verbatim to one Slurm job."""

    repo: str
    commit: str
    env_prefix: str
    output_root: str
    robotwin_root: str
    model_path: str
    algorithm: str
    partition: str = "GPU"
    gpu: str = "4090"
    cpus: int = 8
    time_minutes: int = 20

    def validate(self) -> None:
        """Reject paths or resources outside the short-test contract."""
        if re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise ValueError("--commit must be the complete 40-character Git commit")
        for name in (
            "repo",
            "env_prefix",
            "output_root",
            "robotwin_root",
            "model_path",
        ):
            value = getattr(self, name)
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or path == PurePosixPath("/")
                or ".." in path.parts
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError(f"{name} must be an absolute, non-root POSIX path")
        if PurePosixPath(self.repo).name != self.commit:
            raise ValueError(
                "The fixed worktree directory must be named by its full commit"
            )
        output = PurePosixPath(self.output_root)
        for name in ("repo", "env_prefix", "robotwin_root", "model_path"):
            if output.is_relative_to(PurePosixPath(getattr(self, name))):
                raise ValueError(
                    "Keep output_root outside code, model and environment directories"
                )
        if self.algorithm not in {"ppo", "grpo"}:
            raise ValueError("The online acceptance algorithm must be ppo or grpo")
        if self.gpu not in {"4090", "3090"}:
            raise ValueError("The short test requires one known 4090 or 3090 GPU")
        if self.cpus not in {8, 16}:
            raise ValueError("The short test requires 8 or 16 CPU cores")
        if not 1 <= self.time_minutes <= 30:
            raise ValueError(
                "The short-test time limit must be between 1 and 30 minutes"
            )
        if re.fullmatch(r"[A-Za-z0-9_.-]+", self.partition) is None:
            raise ValueError("Invalid Slurm partition name")


@dataclass(frozen=True)
class QueueJob:
    """The own-user Slurm state needed for the submission gate."""

    job_id: str
    state: str
    tres: str


def parse_queue(text: str) -> list[QueueJob]:
    """Parse the explicit ``job|state|TRES`` squeue response."""
    jobs = []
    for row in text.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split("|")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            raise ValueError(f"Unexpected squeue row: {row!r}")
        jobs.append(QueueJob(*fields))
    return jobs


def require_submission_capacity(jobs: Sequence[QueueJob]) -> None:
    """Wait for all running own jobs and preserve the account's two queue slots."""
    active = [job for job in jobs if job.state != "PENDING"]
    if active:
        found = ", ".join(f"{job.job_id}:{job.state}" for job in active)
        raise RuntimeError(
            f"Wait for current own jobs to end before submitting: {found}"
        )
    if len(jobs) >= 2:
        raise RuntimeError(
            "The account already has two submitted jobs; no queue slot is free"
        )
    requested_gpus = 0
    for job in jobs:
        for token in re.split(r"[\s,]+", job.tres):
            if "gpu" not in token:
                continue
            match = re.fullmatch(r"(?:gres[/.:])?gpu(?::[A-Za-z0-9_-]+)?:(\d+)", token)
            if match is None:
                raise RuntimeError(
                    f"Cannot determine requested GPUs for job {job.job_id}: {job.tres}"
                )
            requested_gpus += int(match[1])
    if requested_gpus + 1 > 2:
        raise RuntimeError("The new job would exceed the two-own-GPU request limit")


def sbatch_command(plan: Submission) -> list[str]:
    """Build an argv vector without changing the task, reward or model settings."""
    plan.validate()
    duration = f"00:{plan.time_minutes:02d}:00"
    logs = PurePosixPath(plan.output_root) / "slurm"
    return [
        "sbatch",
        "--parsable",
        f"--partition={plan.partition}",
        "--exclude=4090node1",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={plan.cpus}",
        f"--time={duration}",
        f"--gres=gpu:{plan.gpu}:1",
        f"--job-name=xvla-online-{plan.algorithm}",
        f"--output={logs}/%j.out",
        f"--error={logs}/%j.err",
        f"--chdir={plan.repo}",
        "--export=ALL",
        str(PurePosixPath(plan.repo) / "tools/xvla/online_job.sh"),
        plan.repo,
        plan.env_prefix,
        plan.output_root,
        plan.robotwin_root,
        plan.model_path,
        plan.algorithm,
        str(plan.time_minutes),
        plan.commit,
    ]


def validate_worktree(plan: Submission) -> None:
    """Verify the exact committed checkout and required local job assets."""
    repo = Path(plan.repo).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != plan.commit:
        raise ValueError(f"Expected worktree {plan.commit}, found {actual}")
    attached = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        check=False,
    )
    if attached.returncode != 1:
        raise ValueError("Run the online test from a detached fixed-commit worktree")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    if dirty:
        raise ValueError("The online-test worktree has uncommitted files")
    for path in (
        repo / "tools/xvla/online_job.sh",
        repo / "tools/xvla/online_acceptance.py",
        repo / "tools/xvla/train.sh",
        Path(plan.env_prefix) / "bin/activate",
        Path(plan.env_prefix) / "bin/python",
        Path(plan.model_path) / "verified.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required online-test asset is missing: {path}")
    if not Path(plan.robotwin_root).is_dir():
        raise FileNotFoundError(
            f"Native RoboTwin checkout is missing: {plan.robotwin_root}"
        )


def submit(plan: Submission) -> dict:
    """Submit one job after rechecking the account under an output-root lock."""
    import fcntl

    plan.validate()
    validate_worktree(plan)
    output = Path(plan.output_root)
    (output / "slurm").mkdir(parents=True, exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    with (output / "submit.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        own_queue = subprocess.check_output(
            ["squeue", "--noheader", "--user", str(os.getuid()), "--format=%i|%T|%b"],
            text=True,
        )
        jobs = parse_queue(own_queue)
        require_submission_capacity(jobs)
        command = sbatch_command(plan)
        result = subprocess.check_output(command, text=True).strip()
        if re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", result) is None:
            raise RuntimeError(
                f"Unexpected sbatch response; inspect the queue before retrying: {result!r}"
            )
        job_id = result.split(";", 1)[0]
        receipt = {
            "job_id": job_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "plan": asdict(plan),
            "command": command,
            "queue_before": [asdict(job) for job in jobs],
            "memory_isolation": "No Slurm memory request; physical memory is not reliably isolated",
        }
        path = output / "evidence" / f"submitted-{job_id}.json"
        try:
            path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            raise RuntimeError(
                f"Slurm accepted job {job_id}, but writing its receipt failed: {error}"
            ) from error
        return receipt


@dataclass(frozen=True)
class OwnedProcess:
    """One Linux process identity, including start time to detect PID reuse."""

    pid: int
    parent_pid: int
    uid: int
    start_time: int
    argv: tuple[str, ...]


def marker_processes(
    processes: Sequence[OwnedProcess], uid: int, marker: str, exclude: set[int]
) -> list[OwnedProcess]:
    """Find own job-marked Ray processes and their own-user descendants."""
    pattern = re.compile(re.escape(marker) + r"(?:/|$)")
    own = {
        process.pid: process
        for process in processes
        if process.uid == uid and process.pid not in exclude
    }
    selected = {
        pid
        for pid, process in own.items()
        if any(pattern.search(arg) for arg in process.argv)
    }
    while True:
        descendants = {
            pid for pid, process in own.items() if process.parent_pid in selected
        }
        expanded = selected | descendants
        if expanded == selected:
            return [own[pid] for pid in sorted(selected, reverse=True)]
        selected = expanded


def read_process(pid: int, root: Path = Path("/proc")) -> OwnedProcess | None:
    """Read a stable-enough Linux process identity without following other users."""
    folder = root / str(pid)
    try:
        uid = folder.stat().st_uid
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        argv = tuple(
            part.decode(errors="replace")
            for part in (folder / "cmdline").read_bytes().split(b"\0")
            if part
        )
        return OwnedProcess(pid, int(stat[1]), uid, int(stat[19]), argv)
    except (OSError, ValueError, IndexError):
        return None


def cleanup_job_ray(marker: str) -> dict:
    """Terminate only this job's own marked Ray tree, never a shared runtime."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise RuntimeError("Ray cleanup requires a real Slurm job ID")
    expected = f"/tmp/xvla_rl_{getpass.getuser()}_{job_id}"
    if marker != expected:
        raise ValueError(
            f"Ray cleanup marker must be this job's private directory: {expected}"
        )
    processes = [
        record
        for path in Path("/proc").iterdir()
        if path.name.isdigit() and (record := read_process(int(path.name))) is not None
    ]
    by_pid = {process.pid: process for process in processes}
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent in by_pid and parent not in ancestors:
        ancestors.add(parent)
        parent = by_pid[parent].parent_pid
    candidates = marker_processes(processes, os.getuid(), marker, ancestors)
    terminated = []
    for process in candidates:
        current = read_process(process.pid)
        if (
            current is not None
            and current.uid == os.getuid()
            and current.start_time == process.start_time
        ):
            try:
                os.kill(process.pid, signal.SIGTERM)
                terminated.append(process.pid)
            except ProcessLookupError:
                pass
    if terminated:
        time.sleep(2)
    killed = []
    for process in candidates:
        current = read_process(process.pid)
        if (
            current is not None
            and current.uid == os.getuid()
            and current.start_time == process.start_time
        ):
            try:
                os.kill(process.pid, signal.SIGKILL)
                killed.append(process.pid)
            except ProcessLookupError:
                pass
    return {"ray_temp": marker, "terminated_pids": terminated, "killed_pids": killed}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-commit single-job submission options."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "commit",
        "env-prefix",
        "output-root",
        "robotwin-root",
        "model-path",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--partition", default="GPU")
    parser.add_argument("--gpu", choices=("4090", "3090"), default="4090")
    parser.add_argument("--cpus", type=int, choices=(8, 16), default=8)
    parser.add_argument("--time-minutes", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print a plan, submit one job, or clean only the current job's Ray tree."""
    values = sys.argv[1:] if argv is None else argv
    if values and values[0] == "--cleanup-ray-temp":
        parser = argparse.ArgumentParser(
            description="Clean this Slurm job's private Ray processes"
        )
        parser.add_argument("--cleanup-ray-temp", required=True)
        cleanup = parser.parse_args(values)
        result = cleanup_job_ray(cleanup.cleanup_ray_temp)
    else:
        args = arguments(values)
        plan = Submission(
            **{key: value for key, value in vars(args).items() if key != "dry_run"}
        )
        command = sbatch_command(plan)
        if args.dry_run:
            result = {"dry_run": True, "plan": asdict(plan), "command": command}
        else:
            result = submit(plan)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
