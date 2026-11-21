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
from dataclasses import asdict, dataclass, field, replace
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
    sampler_diagnostic: bool = False
    config_name: str | None = None
    after_job: str | None = None

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
        if self.config_name is not None and self.config_name not in {
            f"robotwin_{self.algorithm}_xvla",
            f"robotwin_{self.algorithm}_xvla_acceptance",
        }:
            raise ValueError(
                "The config name must select the requested X-VLA algorithm"
            )
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
        if (
            self.after_job is not None
            and re.fullmatch(r"[1-9][0-9]*", self.after_job) is None
        ):
            raise ValueError(
                "--after-job must be one canonical positive numeric Slurm job ID"
            )


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


def require_submission_capacity(
    jobs: Sequence[QueueJob], after_job: str | None = None
) -> None:
    """Allow no active own job, or only the explicitly named dependency target."""
    active = [job for job in jobs if job.state != "PENDING"]
    if after_job is not None:
        if re.fullmatch(r"[1-9][0-9]*", after_job) is None:
            raise ValueError(
                "--after-job must be one canonical positive numeric Slurm job ID"
            )
        target = [job for job in jobs if job.job_id == after_job]
        if len(target) != 1:
            raise RuntimeError(
                "The dependency target must be present in the own-user queue"
            )
        if len(active) != 1 or active[0].job_id != after_job:
            raise RuntimeError("The dependency target must be the only active own job")
    elif active:
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
    job_name = (
        "xvla-sampler-diagnostic"
        if plan.sampler_diagnostic
        else f"xvla-online-{plan.algorithm}"
    )
    exported = "ALL,XVLA_SAMPLER_DIAGNOSTIC=1" if plan.sampler_diagnostic else "ALL"
    command = [
        "sbatch",
        "--parsable",
        *([f"--dependency=afterany:{plan.after_job}"] if plan.after_job else []),
        f"--partition={plan.partition}",
        "--exclude=4090node1",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={plan.cpus}",
        f"--time={duration}",
        f"--gres=gpu:{plan.gpu}:1",
        f"--job-name={job_name}",
        f"--output={logs}/%j.out",
        f"--error={logs}/%j.err",
        f"--chdir={plan.repo}",
        f"--export={exported}",
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
    command.append(
        plan.config_name
        or f"robotwin_{plan.algorithm}_xvla"
        + ("" if plan.sampler_diagnostic else "_acceptance")
    )
    return command


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
        repo
        / "tools/xvla"
        / (
            "sampler_diagnostic.py"
            if plan.sampler_diagnostic
            else "online_acceptance.py"
        ),
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
        require_submission_capacity(jobs, plan.after_job)
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
    state: str = ""


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


def read_process_identity(pid: int, root: Path = Path("/proc")) -> OwnedProcess | None:
    """Read UID, start time and state without requiring cmdline or environ access."""
    folder = root / str(pid)
    try:
        uid = folder.stat().st_uid
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        return OwnedProcess(pid, int(stat[1]), uid, int(stat[19]), (), stat[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_process(pid: int, root: Path = Path("/proc")) -> OwnedProcess | None:
    """Read a stable-enough Linux process identity without following other users."""
    try:
        identity = read_process_identity(pid, root)
        if identity is None:
            return None
        argv = tuple(
            part.decode(errors="replace")
            for part in (root / str(pid) / "cmdline").read_bytes().split(b"\0")
            if part
        )
        return replace(identity, argv=argv)
    except (OSError, ValueError, IndexError):
        return None


def login_environment_matches(environment: bytes, run_id: str, marker: str) -> bool:
    """Require one exact initial environment entry for each private login tag."""
    entries = environment.split(b"\0")
    for key, value in (("XVLA_LOGIN_RUN_ID", run_id), ("RAY_TMPDIR", marker)):
        prefix = os.fsencode(key + "=")
        if [entry for entry in entries if entry.startswith(prefix)] != [
            prefix + os.fsencode(value)
        ]:
            return False
    return True


@dataclass
class LoginProcessTracker:
    """Retain verified login identities across transient environment access loss."""

    uid: int
    run_id: str
    marker: str
    verified: set[tuple[int, int, int]] = field(default_factory=set, init=False)

    def foreign_pids(
        self, pids: Sequence[int], root: Path = Path("/proc")
    ) -> list[int]:
        """Reject unknown identities; cache fallback requires unchanged UID/starttime."""
        foreign = []
        for pid in pids:
            try:
                before = read_process_identity(pid, root)
                if before is None or before.state in {"Z", "X"}:
                    self.verified = {key for key in self.verified if key[1] != pid}
                    continue
                key = (before.uid, pid, before.start_time)
                self.verified = {
                    item for item in self.verified if item[1] != pid or item == key
                }
                if before.uid != self.uid:
                    foreign.append(pid)
                    continue
                try:
                    environment = (root / str(pid) / "environ").read_bytes()
                except OSError:
                    environment = None
                after = read_process_identity(pid, root)
                if after is None or after.state in {"Z", "X"}:
                    self.verified.discard(key)
                    continue
                if (after.uid, pid, after.start_time) != key:
                    self.verified.discard(key)
                    foreign.append(pid)
                elif environment is None:
                    if key not in self.verified:
                        foreign.append(pid)
                elif login_environment_matches(environment, self.run_id, self.marker):
                    self.verified.add(key)
                else:
                    self.verified.discard(key)
                    foreign.append(pid)
            except (OSError, ValueError, IndexError):
                foreign.append(pid)
        return foreign


def acceptance_finished(folder: Path) -> bool:
    """Stop watching resources after the acceptance entry records its final status."""
    try:
        result = json.loads((folder / "online-acceptance.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("status") in {"PASS", "FAILED"}


def read_login_process(
    pid: int, uid: int, run_id: str, marker: str, root: Path = Path("/proc")
) -> OwnedProcess | None:
    """Verify both initial login tags and a stable own-user process identity."""
    before = read_process(pid, root)
    if before is None or before.uid != uid or before.state in {"Z", "X"}:
        return None
    try:
        environment = (root / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    if not login_environment_matches(environment, run_id, marker):
        return None
    after = read_process(pid, root)
    if (
        after is None
        or after.uid != uid
        or after.start_time != before.start_time
        or after.state in {"Z", "X"}
    ):
        return None
    return after


def cleanup_candidates(
    processes: Sequence[OwnedProcess],
    uid: int,
    marker: str,
    exclude: set[int],
    login_run_id: str | None = None,
    root: Path = Path("/proc"),
) -> list[OwnedProcess]:
    """Select the marker tree, plus double-tagged login orphans only when enabled."""
    selected = {
        process.pid: process
        for process in marker_processes(processes, uid, marker, exclude)
        if process.state not in {"Z", "X"}
    }
    if login_run_id is not None:
        for process in processes:
            if process.pid in exclude or process.uid != uid:
                continue
            verified = read_login_process(process.pid, uid, login_run_id, marker, root)
            if verified is not None and verified.start_time == process.start_time:
                selected[process.pid] = verified
    return [selected[pid] for pid in sorted(selected, reverse=True)]


def private_ray_marker() -> str:
    """Identify the current Slurm job or explicitly bounded login debug run."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if os.environ.get("XVLA_LOGIN_TRAINING_DEBUG") == "1":
        import hashlib
        import socket

        if "SLURM_JOB_ID" in os.environ or socket.gethostname() != "4090node1":
            raise RuntimeError("Login Ray cleanup requires 4090node1 outside Slurm")
        run_id = os.environ.get("XVLA_LOGIN_RUN_ID", "")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", run_id) is None:
            raise ValueError("Set an 8-64 character private XVLA_LOGIN_RUN_ID")
        seconds = int(os.environ.get("XVLA_LOGIN_SECONDS", "0"))
        if not 30 <= seconds < 1500 or os.environ.get("XVLA_LOGIN_CPUS") != "4":
            raise ValueError("Login Ray cleanup requires a bounded four-CPU debug run")
        key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        return f"/tmp/xvla_l_{getpass.getuser()}_{key}"
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise RuntimeError("Ray cleanup requires a real Slurm job ID")
    return f"/tmp/xvla_rl_{getpass.getuser()}_{job_id}"


def cleanup_job_ray(marker: str, *, proc_root: Path = Path("/proc")) -> dict:
    """Terminate only this job's own marked Ray tree, never a shared runtime."""
    expected = private_ray_marker()
    if marker != expected:
        raise ValueError(
            f"Ray cleanup marker must be this job's private directory: {expected}"
        )

    def scan():
        return [
            record
            for path in proc_root.iterdir()
            if path.name.isdigit()
            and (record := read_process(int(path.name), proc_root)) is not None
        ]

    processes = scan()
    by_pid = {process.pid: process for process in processes}
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent in by_pid and parent not in ancestors:
        ancestors.add(parent)
        parent = by_pid[parent].parent_pid
    uid = os.getuid()
    login_run_id = (
        os.environ["XVLA_LOGIN_RUN_ID"]
        if os.environ.get("XVLA_LOGIN_TRAINING_DEBUG") == "1"
        else None
    )

    def candidates(snapshot):
        return cleanup_candidates(
            snapshot, uid, marker, ancestors, login_run_id, proc_root
        )

    initial = candidates(processes)
    terminated = []
    for process in initial:
        current = read_process(process.pid, proc_root)
        if (
            current is not None
            and current.uid == uid
            and current.start_time == process.start_time
        ):
            try:
                os.kill(process.pid, signal.SIGTERM)
                terminated.append(process.pid)
            except ProcessLookupError:
                pass
    if terminated:
        time.sleep(2)
    targets = {(process.pid, process.start_time): process for process in initial}
    targets.update(
        {(process.pid, process.start_time): process for process in candidates(scan())}
    )
    killed = []
    for process in targets.values():
        current = read_process(process.pid, proc_root)
        if (
            current is not None
            and current.uid == uid
            and current.start_time == process.start_time
        ):
            try:
                os.kill(process.pid, signal.SIGKILL)
                killed.append(process.pid)
            except ProcessLookupError:
                pass

    def survivors():
        snapshot = scan()
        selected = {process.pid: process for process in candidates(snapshot)}
        for process in snapshot:
            if (
                (process.pid, process.start_time) in targets
                and process.uid == uid
                and process.state not in {"Z", "X"}
            ):
                selected[process.pid] = process
        return [selected[pid] for pid in sorted(selected)]

    remaining = survivors()
    deadline = time.monotonic() + 2
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = survivors()
    return {
        "ray_temp": marker,
        "terminated_pids": terminated,
        "killed_pids": killed,
        "remaining_pids": [process.pid for process in remaining],
    }


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
    parser.add_argument("--sampler-diagnostic", action="store_true")
    parser.add_argument("--config-name")
    parser.add_argument("--after-job")
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
    if result.get("remaining_pids"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
