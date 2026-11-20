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

import numpy as np
import pytest

from tools.xvla.sampler_diagnostic import action_comparison, trial_plan


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
