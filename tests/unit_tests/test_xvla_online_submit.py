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
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.xvla.online_submit import (
    OwnedProcess,
    Submission,
    cleanup_job_ray,
    marker_processes,
    parse_queue,
    private_ray_marker,
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
