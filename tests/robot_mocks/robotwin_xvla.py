# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native RoboTwin task fake at the simulator boundary."""

from contextlib import nullcontext

import numpy as np

from tools.xvla.robotwin.protocol import CAMERAS, RGB_SHAPE, Budget


class FakeNativeTask:
    """Record native commands and expose measured end poses and raw cameras."""

    success_at = None
    fail_at = None
    native_limit = 60
    advance_counter = True

    def __init__(self, cfg):
        del cfg
        self.env = self
        self.close_count = 0
        self.reset_count = 0
        self.commands = []
        self.budget = Budget()
        self.initialized = False

    def reset(self, index, seed):
        del index
        self.reset_count += 1
        self.initialized = True
        self.actual_seed = seed
        self.take_action_cnt = 0
        self.step_lim = self.native_limit
        self.actor_pose = True
        self.commands = []
        self.budget = Budget(step_lim=self.step_lim, settling_ticks=2500)
        self.left_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.right_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.grippers = [0.3, 0.7]

    def observe(self):
        return {
            "observation": {
                name: {"rgb": np.full(RGB_SHAPE, index, dtype=np.uint8)}
                for index, name in enumerate(CAMERAS, 1)
            },
            "endpose": {
                "left_endpose": self.left_pose.copy(),
                "right_endpose": self.right_pose.copy(),
                "left_gripper": self.grippers[0],
                "right_gripper": self.grippers[1],
            },
        }

    def scene_signature(self):
        return [("actor", float(self.actual_seed), 0.0, 0.0)]

    def action_context(self):
        return nullcontext()

    def take_action(self, command):
        self.commands.append(command.copy())
        if self.advance_counter:
            self.take_action_cnt += 1
        self.budget.physics_ticks += 5
        self.left_pose = command[:7].tolist()
        self.right_pose = command[8:15].tolist()
        self.grippers = [(1 - command[7]) / 2, (1 - command[15]) / 2]
        if self.fail_at == self.take_action_cnt:
            self.actor_pose = False

    def check_success(self):
        return self.success_at is not None and self.take_action_cnt >= self.success_at

    def close(self):
        self.close_count += 1
        self.initialized = False
