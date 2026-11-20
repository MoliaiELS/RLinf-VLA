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

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.xvla.online_submit import (
    OwnedProcess,
    Submission,
    marker_processes,
    parse_queue,
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
    assert not any(value.startswith("--mem") for value in command)
    assert command[-8:] == [
        config.repo,
        config.env_prefix,
        config.output_root,
        config.robotwin_root,
        config.model_path,
        "ppo",
        "20",
        config.commit,
    ]
    grpo = sbatch_command(
        replace(config, algorithm="grpo", gpu="3090", cpus=16, time_minutes=30)
    )
    assert "--gres=gpu:3090:1" in grpo
    assert "--time=00:30:00" in grpo
    assert grpo[-3] == "grpo"


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


def test_dry_run_prints_the_exact_plan_without_git_or_slurm(tmp_path):
    config = plan()
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
    completed = subprocess.run(
        [sys.executable, str(script), *args, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert result["command"] == sbatch_command(config)
    assert not list(tmp_path.iterdir())
