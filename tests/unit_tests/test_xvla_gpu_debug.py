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

"""Check diagnostic input, checkpoint and execution boundaries without a GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.xvla import gpu_debug


def test_cli_refuses_unallocated_login_execution_before_gpu_imports(tmp_path):
    environment = dict(os.environ)
    environment.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(gpu_debug.__file__)),
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "explicitly permit bounded login debug" in result.stderr
    assert not (tmp_path / "output").exists()


def test_deterministic_synthetic_observations_and_explicit_source():
    arguments = SimpleNamespace(observations=None, instruction="move stapler pad")
    first, source = gpu_debug._observation(np, torch, arguments)
    second, second_source = gpu_debug._observation(np, torch, arguments)
    assert source == second_source == "deterministic_rgb_and_proprio_placeholders"
    assert first["main_images"].shape == (2, 240, 320, 3)
    assert first["wrist_images"].shape == (2, 2, 240, 320, 3)
    assert first["states"].shape == (2, 20)
    for key in ("main_images", "wrist_images", "states"):
        assert torch.equal(first[key], second[key])
    assert not torch.equal(first["main_images"][0], first["main_images"][1])


def test_observation_file_must_match_full_dual_arm_batch(tmp_path):
    source = tmp_path / "observations.npz"
    np.savez(
        source,
        main_images=np.zeros((1, 240, 320, 3), dtype=np.uint8),
        wrist_images=np.zeros((1, 2, 240, 320, 3), dtype=np.uint8),
        states=np.zeros((1, 20)),
    )
    with pytest.raises(ValueError, match="main_images"):
        gpu_debug._observation(
            np, torch, SimpleNamespace(observations=source, instruction="test")
        )


def test_trainable_payload_restores_parameters_and_does_not_include_frozen_weights():
    model = nn.Linear(3, 2)
    model.weight.requires_grad_(False)
    state = gpu_debug._trainable_state(model)
    assert set(state) == {"bias"}
    frozen = model.weight.detach().clone()
    with torch.no_grad():
        model.bias.add_(1.0)
    assert not torch.equal(model.bias, state["bias"])
    gpu_debug._restore_trainable(model, state)
    assert torch.equal(model.bias, state["bias"])
    assert torch.equal(model.weight, frozen)
    with pytest.raises(RuntimeError, match="names"):
        gpu_debug._restore_trainable(model, {})


def test_summary_cannot_claim_real_rollouts_or_distributed_checkpoint(tmp_path):
    recorder = gpu_debug.Recorder(tmp_path)
    recorder.event("synthetic_advantages", rewards=[1.0, 0.0])
    recorder.summary("PASS")
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["evidence"] == gpu_debug.EVIDENCE
    assert result["real_environment_steps"] == result["real_rollouts"] == 0
    assert result["checkpoint_kind"] == "trainable_parameters_only_debug_roundtrip"
    assert json.loads((tmp_path / "events.jsonl").read_text())["rewards"] == [1.0, 0.0]
