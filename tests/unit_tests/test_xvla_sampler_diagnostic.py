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

"""Check the paired sampler diagnostic's comparisons and fixed trial coverage."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from tools.xvla.sampler_diagnostic import action_comparison, resume_trials, trial_plan


def test_diagnostic_plan_covers_each_control_on_both_original_initial_states():
    plan = trial_plan()
    assert len(plan) == 10
    for seed in (12003, 13001):
        assert {item["mode"] for item in plan if item["seed"] == seed} == {
            "expert",
            "native_measured",
            "native_history",
            "flow_ode",
            "flow_noise",
        }


def test_action_differences_keep_position_rotation_and_grippers_separate():
    reference = np.zeros((1, 30, 20))
    actions = reference.copy()
    actions[..., (0, 1, 2, 10, 11, 12)] = 0.003
    actions[..., (9, 19)] = 0.8
    result = action_comparison(actions, reference)
    assert result["xyz_rms"] == pytest.approx(0.003)
    assert result["xyz_max_abs"] == pytest.approx(0.003)
    assert result["rotation6d_rms"] == 0
    assert result["gripper_threshold_disagreements"] == 60
    assert action_comparison(actions, actions)["gripper_threshold_disagreements"] == 0


def test_resume_keeps_only_complete_trials_and_rejects_changed_protocol(tmp_path):
    previous, destination = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    destination.mkdir()
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"precision": "fp32"}},
            "env": {"train": {"max_episode_steps": 300}},
        }
    )
    (previous / "diagnostic-config.yaml").write_text(OmegaConf.to_yaml(cfg))
    trial = {
        "seed": 12003,
        "mode": "expert",
        "status": "COMPLETE",
        "expert_success": True,
    }
    manifest = {"status": "INTERRUPTED", "plan": trial_plan(), "trials": [trial]}
    (previous / "sampler-diagnostic.json").write_text(json.dumps(manifest))
    folder = previous / "12003-expert"
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps(trial))
    (folder / "initial-state.json").write_text(json.dumps({"reset_seeds": [12003]}))
    (previous / "12003-native_measured").mkdir()
    assert resume_trials(previous, destination, cfg) == [trial]
    assert (destination / "12003-expert/summary.json").exists()
    assert not (destination / "12003-native_measured").exists()
    cfg.env.train.max_episode_steps = 400
    with pytest.raises(ValueError, match="changed model or native environment"):
        resume_trials(previous, destination, cfg)
