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

"""Check the short-training submission and job-private process boundaries."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from tools.xvla import online_submit
from tools.xvla.online_submit import (
    LoginProcessTracker,
    OwnedProcess,
    Submission,
    acceptance_finished,
    cleanup_candidates,
    cleanup_job_ray,
    marker_processes,
    parse_queue,
    private_ray_marker,
    read_login_process,
    read_process,
    require_submission_capacity,
    sbatch_command,
)


def plan() -> Submission:
    commit = "a" * 40
    return Submission(
        repo=f"/home/user/RLinf-VLA-runs/{commit}",
        commit=commit,
        env_prefix="/home/user/.venvs/xvla_flow_debug",
        output_root="/storage/user/xvla/online",
        robotwin_root="/home/user/RoboTwin-pinned",
        model_path="/storage/user/models/X-VLA-RoboTwin2-pinned",
        algorithm="ppo",
    )


def test_submission_uses_one_gpu_and_a_bounded_compute_node_job():
    config = plan()
    command = sbatch_command(config)
    assert command[:2] == ["sbatch", "--parsable"]
    assert "--exclude=4090node1" in command
    assert "--gres=gpu:4090:1" in command
    assert "--cpus-per-task=8" in command
    assert "--time=00:20:00" in command
    assert "--job-name=xvla-online-ppo" in command
    assert "--export=ALL" in command
    assert not any(value.startswith("--mem") for value in command)
    assert command[-9:] == [
        config.repo,
        config.env_prefix,
        config.output_root,
        config.robotwin_root,
        config.model_path,
        "ppo",
        "20",
        config.commit,
        "robotwin_ppo_xvla_acceptance",
    ]
    grpo = sbatch_command(
        replace(config, algorithm="grpo", gpu="3090", cpus=16, time_minutes=30)
    )
    assert "--gres=gpu:3090:1" in grpo
    assert "--time=00:30:00" in grpo
    assert grpo[-4] == "grpo"
    assert grpo[-1] == "robotwin_grpo_xvla_acceptance"


def test_sampler_diagnostic_keeps_the_same_bounded_job_resources():
    config = plan()
    regular = sbatch_command(config)
    diagnostic = sbatch_command(replace(config, sampler_diagnostic=True))
    assert "--job-name=xvla-sampler-diagnostic" in diagnostic
    assert "--export=ALL,XVLA_SAMPLER_DIAGNOSTIC=1" in diagnostic
    assert diagnostic[-1] == "robotwin_ppo_xvla"
    assert [
        value
        for value in diagnostic[:-1]
        if not value.startswith(("--job-name=", "--export="))
    ] == [
        value
        for value in regular[:-1]
        if not value.startswith(("--job-name=", "--export="))
    ]


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
def test_acceptance_config_is_forwarded_without_changing_default_submission(algorithm):
    config = replace(plan(), algorithm=algorithm)
    command = sbatch_command(config)
    assert command[-1] == f"robotwin_{algorithm}_xvla_acceptance"
    selected = f"robotwin_{algorithm}_xvla"
    assert sbatch_command(replace(config, config_name=selected)) == [
        *command[:-1],
        selected,
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"commit": "HEAD"},
        {"repo": "/home/user/mutable-checkout"},
        {"output_root": "/"},
        {"output_root": "/storage/../root"},
        {"output_root": "/storage/user/models/X-VLA-RoboTwin2-pinned/results"},
        {"output_root": "/storage/user/\nresults"},
        {"time_minutes": 31},
        {"time_minutes": 0},
        {"cpus": 64},
        {"gpu": "all"},
        {"partition": "GPU --exclusive"},
        {"algorithm": "sft"},
        {"config_name": "robotwin_grpo_xvla_acceptance"},
        {"config_name": "arbitrary_training"},
    ],
)
def test_invalid_paths_or_unbounded_resources_are_rejected(override):
    with pytest.raises(ValueError):
        sbatch_command(replace(plan(), **override))


def test_submission_waits_for_the_current_job_and_keeps_the_held_job():
    held = "20326|PENDING|gres/gpu:4090:1\n"
    require_submission_capacity(parse_queue(held))
    for state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "UNKNOWN"):
        with pytest.raises(RuntimeError, match="Wait for current"):
            require_submission_capacity(
                parse_queue(held + f"20730|{state}|gres/gpu:4090:1\n")
            )
    with pytest.raises(RuntimeError, match="two submitted"):
        require_submission_capacity(
            parse_queue(held + "20731|PENDING|gres/gpu:4090:1\n")
        )
    with pytest.raises(RuntimeError, match="two-own-GPU"):
        require_submission_capacity(parse_queue("20326|PENDING|gres/gpu:4090:2\n"))
    with pytest.raises(RuntimeError, match="Cannot determine"):
        require_submission_capacity(parse_queue("20326|PENDING|gpu:unknown\n"))
    with pytest.raises(ValueError, match="Unexpected squeue"):
        parse_queue("20730 RUNNING")


def test_scoped_cleanup_keeps_other_users_jobs_and_similar_directory_names():
    marker = "/tmp/xvla_rl_user_123"
    entries = [
        OwnedProcess(10, 1, 1000, 1, ("raylet", f"--session-dir={marker}/session")),
        OwnedProcess(11, 10, 1000, 1, ("ray::Actor",)),
        OwnedProcess(12, 11, 1000, 1, ("worker",)),
        OwnedProcess(13, 10, 2000, 1, ("ray::OtherUser",)),
        OwnedProcess(
            20, 1, 1000, 1, ("raylet", "--session-dir=/tmp/xvla_rl_user_1234/session")
        ),
        OwnedProcess(21, 20, 1000, 1, ("ray::OtherJob",)),
        OwnedProcess(30, 1, 2000, 1, ("raylet", f"--session-dir={marker}/session")),
        OwnedProcess(40, 1, 1000, 1, ("python", "--cleanup-ray-temp", marker)),
    ]
    selected = marker_processes(entries, uid=1000, marker=marker, exclude={40})
    assert {entry.pid for entry in selected} == {10, 11, 12}


def proc_entry(root, pid, run_id, marker, *, parent=1, start=900, state="S", argv=None):
    folder = root / str(pid)
    folder.mkdir(exist_ok=True)
    fields = [state, str(parent), *("0" for _ in range(17)), str(start), "0"]
    (folder / "stat").write_text(f"{pid} (ray worker) " + " ".join(fields))
    (folder / "cmdline").write_bytes(
        b"\0".join(os.fsencode(arg) for arg in (argv or ["ray::Actor"]))
    )
    (folder / "environ").write_bytes(
        os.fsencode(f"XVLA_LOGIN_RUN_ID={run_id}\0RAY_TMPDIR={marker}\0")
    )
    return folder


def test_login_orphans_are_selected_through_proc_files_only_with_both_exact_tags(
    tmp_path,
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    valid = proc_entry(tmp_path, 101, run_id, marker)
    proc_entry(tmp_path, 102, run_id + "_other", marker)
    proc_entry(tmp_path, 103, run_id, marker + "_other")
    duplicate = proc_entry(tmp_path, 104, run_id, marker)
    with (duplicate / "environ").open("ab") as stream:
        stream.write(b"XVLA_LOGIN_RUN_ID=another_run\0")
    proc_entry(tmp_path, 105, run_id, marker, state="Z")
    proc_entry(tmp_path, 106, run_id, marker)
    proc_entry(
        tmp_path,
        201,
        "other_run",
        "other_marker",
        argv=["raylet", f"--session-dir={marker}/session"],
    )
    proc_entry(tmp_path, 202, "other_run", "other_marker", parent=201)
    processes = [read_process(int(path.name), tmp_path) for path in tmp_path.iterdir()]
    uid = valid.stat().st_uid
    default = cleanup_candidates(processes, uid, marker, {106}, root=tmp_path)
    login = cleanup_candidates(processes, uid, marker, {106}, run_id, tmp_path)
    assert {process.pid for process in default} == {201, 202}
    assert {process.pid for process in login} == {101, 201, 202}
    assert read_process(101, tmp_path).parent_pid == 1
    assert cleanup_candidates(processes, uid + 1, marker, set(), run_id, tmp_path) == []


def test_login_proc_identity_is_rechecked_after_reading_initial_environment(
    tmp_path, monkeypatch
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    folder = proc_entry(tmp_path, 101, run_id, marker)
    original = Path.read_bytes

    def read_and_reuse(path):
        value = original(path)
        if path == folder / "environ":
            proc_entry(tmp_path, 101, run_id, marker, start=901)
        return value

    monkeypatch.setattr(Path, "read_bytes", read_and_reuse)
    assert (
        read_login_process(101, folder.stat().st_uid, run_id, marker, tmp_path) is None
    )


def test_tracker_allows_environment_permission_loss_only_for_verified_identity(
    tmp_path, monkeypatch
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    folder = proc_entry(tmp_path, 101, run_id, marker)
    uid = folder.stat().st_uid
    tracker = LoginProcessTracker(uid, run_id, marker)
    assert tracker.foreign_pids([101], tmp_path) == []
    original = Path.read_bytes

    def unreadable_environment(path):
        if path == folder / "environ":
            raise PermissionError(str(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable_environment)
    assert tracker.foreign_pids([101], tmp_path) == []
    assert LoginProcessTracker(uid, run_id, marker).foreign_pids([101], tmp_path) == [
        101
    ]
    assert LoginProcessTracker(uid + 1, run_id, marker).foreign_pids(
        [101], tmp_path
    ) == [101]
    proc_entry(tmp_path, 101, run_id, marker, start=901)
    assert tracker.foreign_pids([101], tmp_path) == [101]


def test_tracker_cannot_use_cached_environment_when_current_identity_is_unreadable(
    tmp_path, monkeypatch
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    folder = proc_entry(tmp_path, 101, run_id, marker)
    tracker = LoginProcessTracker(folder.stat().st_uid, run_id, marker)
    assert tracker.foreign_pids([101], tmp_path) == []
    original = Path.read_text

    def unreadable_stat(path, *args, **kwargs):
        if path == folder / "stat":
            raise PermissionError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable_stat)
    assert tracker.foreign_pids([101], tmp_path) == [101]


@pytest.mark.parametrize("state", ["Z", "X"])
def test_tracker_ignores_exited_processes_when_environment_becomes_unreadable(
    tmp_path, monkeypatch, state
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    folder = proc_entry(tmp_path, 101, run_id, marker)
    tracker = LoginProcessTracker(folder.stat().st_uid, run_id, marker)
    assert tracker.foreign_pids([101], tmp_path) == []
    proc_entry(tmp_path, 101, run_id, marker, state=state)
    original = Path.read_bytes

    def unreadable_environment(path):
        if path == folder / "environ":
            raise PermissionError(str(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable_environment)
    assert tracker.foreign_pids([101], tmp_path) == []


def test_tracker_rejects_readable_wrong_tags_and_reuse_during_environment_read(
    tmp_path, monkeypatch
):
    run_id, marker = "debug_12345", "/tmp/xvla_l_user_private"
    folder = proc_entry(tmp_path, 101, run_id, marker)
    tracker = LoginProcessTracker(folder.stat().st_uid, run_id, marker)
    assert tracker.foreign_pids([101], tmp_path) == []
    proc_entry(tmp_path, 101, run_id + "_other", marker)
    assert tracker.foreign_pids([101], tmp_path) == [101]
    proc_entry(tmp_path, 101, run_id, marker)
    assert tracker.foreign_pids([101], tmp_path) == []
    original = Path.read_bytes

    def read_and_reuse(path):
        value = original(path)
        if path == folder / "environ":
            proc_entry(tmp_path, 101, run_id, marker, start=901)
        return value

    monkeypatch.setattr(Path, "read_bytes", read_and_reuse)
    assert tracker.foreign_pids([101], tmp_path) == [101]


def test_acceptance_terminal_status_uses_the_actual_result_file(tmp_path):
    assert not acceptance_finished(tmp_path)
    path = tmp_path / "online-acceptance.json"
    path.write_text('{"status":')
    assert not acceptance_finished(tmp_path)
    for status, expected in (("RUNNING", False), ("PASS", True), ("FAILED", True)):
        path.write_text(json.dumps({"status": status}))
        assert acceptance_finished(tmp_path) is expected


@pytest.mark.parametrize("status", ["PASS", "FAILED"])
def test_login_watcher_exits_before_gpu_queries_after_acceptance_terminal_status(
    tmp_path, monkeypatch, status
):
    path = tmp_path / "online-acceptance.json"
    path.write_text(json.dumps({"status": status}))
    assert execute_login_watcher(tmp_path, monkeypatch, lambda: None) == []
    assert not (tmp_path / "resource-stop.json").exists()


def execute_login_watcher(folder, monkeypatch, on_query):
    launcher = Path(__file__).resolve().parents[2] / "tools/xvla/login_training.sh"
    blocks = [
        part.split("\n", 1)[1].split("\nPY", 1)[0]
        for part in launcher.read_text().split("<<'PY'")[1:]
    ]
    source = next(block for block in blocks if "LoginProcessTracker" in block)
    for key, value in {
        "XVLA_RUN_DIR": str(folder),
        "FYP_XVLA_RAY_SPILL": str(folder / "spill"),
        "CUDA_VISIBLE_DEVICES": "GPU-test",
        "XVLA_LOGIN_RUN_ID": "debug_12345",
        "RAY_TMPDIR": "/tmp/private-ray",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(os, "getuid", lambda: folder.stat().st_uid, raising=False)
    monkeypatch.setattr(os, "getpgrp", lambda: 9999, raising=False)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    queries = []
    signals = []

    def query(*args, **kwargs):
        queries.append(args)
        on_query()
        raise RuntimeError("simulated query interruption")

    monkeypatch.setattr(subprocess, "check_output", query)
    monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args), raising=False)
    exec(compile(source, str(launcher), "exec"), {})
    assert signals == []
    return queries


def test_login_watcher_rechecks_terminal_status_before_signaling(tmp_path, monkeypatch):
    path = tmp_path / "online-acceptance.json"
    path.write_text(json.dumps({"status": "RUNNING"}))
    queries = execute_login_watcher(
        tmp_path, monkeypatch, lambda: path.write_text(json.dumps({"status": "PASS"}))
    )
    assert len(queries) == 1
    assert not (tmp_path / "resource-stop.json").exists()


def test_login_watcher_honors_cleanup_stop_without_signaling_a_stale_pid(
    tmp_path, monkeypatch
):
    (tmp_path / "resource-watcher-stop").touch()
    assert execute_login_watcher(tmp_path, monkeypatch, lambda: None) == []
    assert not (tmp_path / "resource-stop.json").exists()


def login_cleanup_context(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "user")
    monkeypatch.setattr("socket.gethostname", lambda: "4090node1")
    monkeypatch.setattr(os, "getpid", lambda: 9000)
    monkeypatch.setattr(os, "getppid", lambda: 9001)
    monkeypatch.setattr(signal, "SIGKILL", getattr(signal, "SIGKILL", 9), raising=False)
    for key, value in {
        "XVLA_LOGIN_TRAINING_DEBUG": "1",
        "XVLA_LOGIN_RUN_ID": "debug_12345",
        "XVLA_LOGIN_SECONDS": "1200",
        "XVLA_LOGIN_CPUS": "4",
    }.items():
        monkeypatch.setenv(key, value)
    return private_ray_marker()


def test_cleanup_reaps_login_orphans_and_keeps_slurm_scope_unchanged(
    tmp_path, monkeypatch
):
    marker = login_cleanup_context(monkeypatch)
    folder = proc_entry(tmp_path, 101, "debug_12345", marker)
    proc_entry(tmp_path, 9000, "debug_12345", marker, parent=9001)
    proc_entry(tmp_path, 9001, "debug_12345", marker)
    monkeypatch.setattr(os, "getuid", lambda: folder.stat().st_uid, raising=False)
    calls = []

    def signal_process(pid, signum):
        calls.append((pid, signum))
        if signum == signal.SIGKILL:
            proc_entry(tmp_path, pid, "debug_12345", marker, state="Z")

    monkeypatch.setattr(os, "kill", signal_process)
    result = cleanup_job_ray(marker, proc_root=tmp_path)
    assert calls == [(101, signal.SIGTERM), (101, signal.SIGKILL)]
    assert result["remaining_pids"] == []
    monkeypatch.delenv("XVLA_LOGIN_TRAINING_DEBUG")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    marker = private_ray_marker()
    proc_entry(tmp_path, 101, "debug_12345", marker)
    calls.clear()
    assert cleanup_job_ray(marker, proc_root=tmp_path)["remaining_pids"] == []
    assert calls == []


def test_cleanup_does_not_signal_a_reused_pid_after_term(tmp_path, monkeypatch):
    marker = login_cleanup_context(monkeypatch)
    folder = proc_entry(tmp_path, 101, "debug_12345", marker)
    monkeypatch.setattr(os, "getuid", lambda: folder.stat().st_uid, raising=False)
    calls = []

    def reuse_process(pid, signum):
        calls.append((pid, signum))
        proc_entry(tmp_path, pid, "different_run", "/tmp/different", start=901)

    monkeypatch.setattr(os, "kill", reuse_process)
    result = cleanup_job_ray(marker, proc_root=tmp_path)
    assert calls == [(101, signal.SIGTERM)]
    assert result["remaining_pids"] == []


def test_cleanup_cli_reports_remaining_processes_and_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    marker = login_cleanup_context(monkeypatch)
    folder = proc_entry(tmp_path, 101, "debug_12345", marker)
    monkeypatch.setattr(os, "getuid", lambda: folder.stat().st_uid, raising=False)
    monkeypatch.setattr(
        os, "kill", lambda pid, signum: (folder / "environ").write_bytes(b"")
    )
    monkeypatch.setattr(
        online_submit, "cleanup_job_ray", partial(cleanup_job_ray, proc_root=tmp_path)
    )
    with pytest.raises(SystemExit) as result:
        online_submit.main(["--cleanup-ray-temp", marker])
    assert result.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["remaining_pids"] == [101]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Requires the live Linux /proc boundary",
)
def test_live_proc_initial_environment_contract_without_sending_signals(tmp_path):
    run_id, marker = "debug_live_12345", str(tmp_path / "private-ray")
    release = tmp_path / "release"
    child = (
        "import os,pathlib,time\n"
        "release=pathlib.Path(os.environ['XVLA_TEST_RELEASE'])\n"
        "deadline=time.monotonic()+5\n"
        "while not release.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
    )
    parent = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]], "
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "print(os.getpid(),p.pid)",
            child,
        ],
        env=dict(
            os.environ,
            XVLA_LOGIN_RUN_ID=run_id,
            RAY_TMPDIR=marker,
            XVLA_TEST_RELEASE=str(release),
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    creator_pid, pid = map(int, parent.stdout.split())
    try:
        identity = read_process(pid)
        assert identity.parent_pid != creator_pid
        verified = read_login_process(pid, os.getuid(), run_id, marker)
        assert verified is not None and verified.start_time == identity.start_time
        assert cleanup_candidates([identity], os.getuid(), marker, set()) == []
        assert [
            item.pid
            for item in cleanup_candidates(
                [identity], os.getuid(), marker, set(), run_id
            )
        ] == [pid]
        assert read_login_process(pid, os.getuid(), run_id + "_other", marker) is None
    finally:
        release.touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            process = read_process(pid)
            if process is None or process.state in {"Z", "X"}:
                break
            time.sleep(0.02)


def test_cleanup_marker_uses_real_slurm_or_explicit_bounded_login_identity(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("XVLA_LOGIN_TRAINING_DEBUG", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "user")
    with pytest.raises(RuntimeError, match="real Slurm"):
        private_ray_marker()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert private_ray_marker() == "/tmp/xvla_rl_user_12345"
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr("socket.gethostname", lambda: "4090node1")
    values = {
        "XVLA_LOGIN_TRAINING_DEBUG": "1",
        "XVLA_LOGIN_RUN_ID": "debug_12345",
        "XVLA_LOGIN_SECONDS": "1200",
        "XVLA_LOGIN_CPUS": "4",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    key = hashlib.sha256(b"debug_12345").hexdigest()[:16]
    assert private_ray_marker() == f"/tmp/xvla_l_user_{key}"
    monkeypatch.setenv("XVLA_LOGIN_RUN_ID", "x" * 64)
    assert (
        private_ray_marker()
        == "/tmp/xvla_l_user_" + hashlib.sha256(b"x" * 64).hexdigest()[:16]
    )
    monkeypatch.setenv("XVLA_LOGIN_RUN_ID", values["XVLA_LOGIN_RUN_ID"])
    with pytest.raises(ValueError, match="this job's private directory"):
        cleanup_job_ray("/tmp/xvla_l_user_some_other_run")
    for key, value in (
        ("XVLA_LOGIN_RUN_ID", "../other"),
        ("XVLA_LOGIN_SECONDS", "1500"),
        ("XVLA_LOGIN_CPUS", "8"),
    ):
        monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            private_ray_marker()
        monkeypatch.setenv(key, values[key])
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    with pytest.raises(RuntimeError, match="outside Slurm"):
        private_ray_marker()
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr("socket.gethostname", lambda: "4090node2")
    with pytest.raises(RuntimeError, match="4090node1"):
        private_ray_marker()


def test_login_training_argument_count_and_literal_hydra_forwarding(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if not candidate.is_file():
            pytest.skip("Bash is unavailable for the launcher argument contract")
        bash = str(candidate)
    launcher = Path(__file__).resolve().parents[2] / "tools/xvla/login_training.sh"
    parser = launcher.read_text().split('\nparse_login_training_arguments "$@"', 1)[0]
    command = (
        parser
        + '\nparse_login_training_arguments "$@"\nprintf "%s\\0" "$config" "${hydra_overrides[@]}"'
    )
    args = [
        "/repo",
        "/venv",
        "/new-output",
        "/robotwin",
        "/model",
        "GPU-uuid",
        "ppo",
        "debug_12345",
    ]
    default = subprocess.run([bash, "-c", command, "--", *args], capture_output=True)
    assert default.returncode == 0
    assert default.stdout.rstrip(b"\0") == b"robotwin_ppo_xvla_acceptance"
    overrides = [
        "runner.resume_dir=/storage/checkpoint with spaces/global_step_3",
        "runner.max_epochs=4",
        "runner.max_steps=4",
        "runner.logger.experiment_name=$(touch unexpected)",
    ]
    forwarded = subprocess.run(
        [bash, "-c", command, "--", *args, "robotwin_ppo_xvla_acceptance", *overrides],
        capture_output=True,
        cwd=tmp_path,
    )
    assert forwarded.returncode == 0
    assert forwarded.stdout.rstrip(b"\0").decode().split("\0") == [
        "robotwin_ppo_xvla_acceptance",
        *overrides,
    ]
    assert not list(tmp_path.iterdir())
    for invalid in (args[:-1], [*args, "robotwin_grpo_xvla_acceptance"]):
        rejected = subprocess.run(
            [bash, "-c", command, "--", *invalid], capture_output=True
        )
        assert rejected.returncode == 2


@pytest.mark.parametrize("sampler_diagnostic", [False, True])
@pytest.mark.parametrize("config_name", [None, "robotwin_ppo_xvla_acceptance"])
def test_dry_run_prints_the_exact_plan_without_git_or_slurm(
    tmp_path, sampler_diagnostic, config_name
):
    config = replace(
        plan(), sampler_diagnostic=sampler_diagnostic, config_name=config_name
    )
    script = Path(__file__).resolve().parents[2] / "tools/xvla/online_submit.py"
    args = []
    for field in (
        "repo",
        "commit",
        "env_prefix",
        "output_root",
        "robotwin_root",
        "model_path",
        "algorithm",
    ):
        args.extend(("--" + field.replace("_", "-"), str(getattr(config, field))))
    if sampler_diagnostic:
        args.append("--sampler-diagnostic")
    if config_name is not None:
        args.extend(("--config-name", config_name))
    completed = subprocess.run(
        [sys.executable, str(script), *args, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert result["plan"]["sampler_diagnostic"] is sampler_diagnostic
    assert result["command"] == sbatch_command(config)
    assert not list(tmp_path.iterdir())
