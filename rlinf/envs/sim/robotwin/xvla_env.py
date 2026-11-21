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

"""Native dual-arm RoboTwin transitions for the X-VLA flow policy."""

from __future__ import annotations

import gc
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from tools.xvla.robotwin import protocol


class NativeTask:
    """Own one native task and preserve its camera, planner and action conventions."""

    def __init__(self, cfg):
        from tools.xvla.robotwin import runner

        root = Path(cfg.robotwin_root).resolve()
        runner.pinned_checkout(root, protocol.ROBOTWIN_COMMIT)
        self.module, task_type, self.config, _ = runner.load_native(
            root, cfg.task_name, cfg.condition
        )
        self._task_type = task_type
        self.env = None
        self.task_name = cfg.task_name
        self.runner = runner
        self.initialized = False
        self.budget = protocol.Budget()

    def reset(self, index: int, seed: int) -> None:
        self.close()
        self.env = self._task_type()
        self.budget = protocol.Budget()
        try:
            self.runner.reset_env(
                self.env, deepcopy(self.config), index, seed, self.budget
            )
            self.runner.init_success_fields(self.module, self.task_name, self.env)
            self.env._update_render()
            self.env.actor_pose = True
            self.initialized = True
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def observe(self) -> dict:
        return self.env.get_obs()

    def scene_signature(self) -> list:
        return self.runner.scene_values(self.env)

    def action_context(self):
        return self.runner.count_actions(self.env, self.budget)

    def take_action(self, command: np.ndarray) -> None:
        self.env.take_action(command, action_type="ee")

    def close(self) -> None:
        episode = self.env
        self.env = None
        self.initialized = False
        if episode is None:
            return
        try:
            episode.close_env(clear_cache=False)
        finally:
            # Pinned close_env delegates to Gym's no-op close; planners retain scenes.
            episode.__dict__.clear()
            del episode
            gc.collect()


class XVLARoboTwinEnv:
    """Batch fixed-task native episodes without autoreset or post-done actions.

    Observations contain unmodified head/left/right RGB and measured 20-D
    proprioception. Each input action is a postprocessed 30-by-20 dual-arm chunk.
    Adjacent ``group_size`` episodes share an explicitly selected initial seed.
    """

    metadata = {"render_fps": 30}

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        record_metrics: bool = True,
    ):
        del worker_info
        if cfg.get("auto_reset", False) or cfg.get("ignore_terminations", False):
            raise ValueError(
                "Native X-VLA requires auto_reset=false and ignore_terminations=false"
            )
        if cfg.get("enable_offload", False):
            raise ValueError("Native X-VLA environments do not support offload")
        if cfg.get("condition") not in ("easy", "hard"):
            raise ValueError("Native X-VLA condition must be easy or hard")
        self.group_size = int(cfg.group_size)
        if num_envs < 1 or self.group_size < 1 or num_envs % self.group_size:
            raise ValueError("num_envs must be positive and divisible by group_size")
        seeds = cfg.get("reset_seeds")
        if not seeds or any(
            isinstance(seed, bool) or int(seed) != seed or seed < 0 for seed in seeds
        ):
            raise ValueError(
                "reset_seeds must explicitly contain nonnegative integer initial seeds"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("reset_seeds must not contain duplicate initial seeds")
        self.reset_seed_mode = cfg.get("reset_seed_mode", "random")
        if self.reset_seed_mode not in ("random", "cycle"):
            raise ValueError("reset_seed_mode must be random or cycle")
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_group = num_envs // self.group_size
        self.seed = int(cfg.seed) + seed_offset
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics
        self.auto_reset = False
        self.ignore_terminations = False
        self.max_episode_steps = int(cfg.max_episode_steps)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        if not np.isfinite(self.reward_coef) or self.reward_coef <= 0:
            raise ValueError("reward_coef must be finite and positive")
        self._seed_pool = np.asarray(seeds, dtype=np.int64)
        self._generator = np.random.default_rng(self.seed)
        self._cycle_cursor = 0
        self._tasks = []
        try:
            for _ in range(num_envs):
                self._tasks.append(NativeTask(cfg))
        except Exception:
            for task in self._tasks:
                with suppress(Exception):
                    task.close()
            raise
        self._raw_obs = [None] * num_envs
        self._is_start = True
        self._reset_count = 0
        self._closed = False
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self._terminations = torch.zeros(num_envs, dtype=torch.bool)
        self._truncations = torch.zeros(num_envs, dtype=torch.bool)
        self._success = torch.zeros(num_envs, dtype=torch.bool)
        self._returns = torch.zeros(num_envs)
        self.update_reset_state_ids()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def update_reset_state_ids(self, env_idx=None) -> None:
        if env_idx is not None:
            raise ValueError("Native grouped episodes require a full-batch reset")
        if self.cfg.get("use_fixed_reset_state_ids", False) and hasattr(
            self, "reset_state_ids"
        ):
            return
        if self.reset_seed_mode == "cycle":
            indices = (np.arange(self.num_group) + self._cycle_cursor) % len(
                self._seed_pool
            )
            self._cycle_cursor += self.num_group
        else:
            indices = self._generator.integers(
                0, len(self._seed_pool), size=self.num_group
            )
        self.reset_state_ids = torch.from_numpy(
            np.repeat(self._seed_pool[indices], self.group_size).copy()
        )

    def _schedule_metadata(self) -> dict:
        return {
            "version": 1,
            "seed_pool": self._seed_pool.tolist(),
            "reset_seed_mode": self.reset_seed_mode,
            "group_size": self.group_size,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "seed_offset": self.seed_offset,
            "total_num_processes": self.total_num_processes,
            "use_fixed_reset_state_ids": bool(
                self.cfg.get("use_fixed_reset_state_ids", False)
            ),
            "task_name": self.cfg.task_name,
            "condition": self.cfg.condition,
        }

    def get_reset_schedule_state(self) -> dict:
        """Export next-episode scheduling at a fresh or completed episode boundary."""
        if self._closed or (
            any(task.initialized for task in self._tasks)
            and not bool((self._terminations | self._truncations).all())
        ):
            raise RuntimeError("Export reset scheduling only at an episode boundary")
        return {
            **self._schedule_metadata(),
            "generator_state": deepcopy(self._generator.bit_generator.state),
            "cycle_cursor": self._cycle_cursor,
            "reset_count": self._reset_count,
            "reset_state_ids": self.reset_state_ids.tolist(),
        }

    def set_reset_schedule_state(self, state: dict) -> None:
        """Restore scheduling into a fresh environment without restoring physics."""
        if self._closed or self._reset_count:
            raise RuntimeError("Restore reset scheduling before the first episode")
        if any(
            type(state.get(key)) is not type(value) or state.get(key) != value
            for key, value in self._schedule_metadata().items()
        ):
            raise ValueError("Reset schedule metadata does not match this environment")
        for name in ("cycle_cursor", "reset_count"):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Reset schedule {name} must be a nonnegative integer")
        seeds = state.get("reset_state_ids")
        if not isinstance(seeds, list):
            raise ValueError(
                "Reset schedule must include selected next reset_state_ids"
            )
        seeds = self._validate_reset_seeds(seeds)
        if self.reset_seed_mode == "cycle":
            cursor = state["cycle_cursor"]
            expected = np.repeat(
                self._seed_pool[
                    (np.arange(self.num_group) + cursor - self.num_group)
                    % len(self._seed_pool)
                ],
                self.group_size,
            ).tolist()
            if cursor < self.num_group or seeds != expected:
                raise ValueError("Cycle cursor does not match the selected next seeds")
        elif state["cycle_cursor"] != 0:
            raise ValueError("Random reset scheduling requires a zero cycle cursor")
        generator = np.random.default_rng()
        try:
            generator.bit_generator.state = deepcopy(state["generator_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Reset schedule requires a compatible NumPy RNG state"
            ) from error
        self._generator = generator
        self._cycle_cursor = state["cycle_cursor"]
        self._reset_count = state["reset_count"]
        self.reset_state_ids = torch.tensor(seeds, dtype=torch.long)

    def _validate_reset_seeds(self, seeds: list) -> list[int]:
        if len(seeds) != self.num_envs or any(
            isinstance(seed, bool) or int(seed) != seed or seed not in self._seed_pool
            for seed in seeds
        ):
            raise ValueError(
                "Reset seeds must belong to the configured initial-state pool"
            )
        if any(
            len(set(seeds[start : start + self.group_size])) != 1
            for start in range(0, self.num_envs, self.group_size)
        ):
            raise ValueError("Every GRPO group must reset to the same native seed")
        return [int(seed) for seed in seeds]

    def _observations(self, reset: bool = False) -> dict:
        values = [protocol.observation_values(obs) for obs in self._raw_obs]
        images = np.stack([np.stack(row[0]) for row in values])
        states = np.stack([row[1] for row in values]).astype(np.float32)
        return {
            "main_images": torch.from_numpy(images[:, 0].copy()),
            "wrist_images": torch.from_numpy(images[:, 1:].copy()),
            "states": torch.from_numpy(states),
            "task_descriptions": [self.cfg.task_name.replace("_", " ")] * self.num_envs,
            "reset_mask": torch.full((self.num_envs,), reset, dtype=torch.bool),
        }

    def reset(self, env_idx=None, env_seeds=None) -> tuple[dict, dict]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed native X-VLA environment")
        if env_idx is not None:
            raise ValueError("Native grouped episodes require a full-batch reset")
        seeds = self.reset_state_ids.tolist() if env_seeds is None else list(env_seeds)
        seeds = self._validate_reset_seeds(seeds)
        self._elapsed_steps.zero_()
        self._terminations.zero_()
        self._truncations.zero_()
        self._success.zero_()
        self._returns.zero_()
        self._is_start = False
        from tools.xvla.robotwin.runner import same_scene

        scenes = []
        try:
            for index, (task, seed) in enumerate(zip(self._tasks, seeds)):
                task.reset(self._reset_count, int(seed))
                if int(task.env.step_lim) < 1 or int(task.env.take_action_cnt) != 0:
                    raise RuntimeError(
                        "Expected a positive native action limit and zero initial action count"
                    )
                if task.env.check_success():
                    raise RuntimeError(
                        "Native initial state already satisfies the task success condition"
                    )
                scenes.append(task.scene_signature())
                self._raw_obs[index] = task.observe()
            for start in range(0, self.num_envs, self.group_size):
                if any(
                    not same_scene(scenes[start], scenes[index])
                    for index in range(start + 1, start + self.group_size)
                ):
                    raise RuntimeError(
                        "Grouped native resets produced different initial scenes"
                    )
        except Exception:
            with suppress(Exception):
                self.close()
            raise
        self._reset_count += 1
        return self._observations(reset=True), {"reset_seeds": torch.tensor(seeds)}

    def _episode_info(self) -> dict:
        budgets = [task.budget for task in self._tasks]
        return {
            "success_once": self._success.clone(),
            "success_at_end": self._success.clone(),
            "return": self._returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self._returns / self._elapsed_steps.clamp(min=1),
            "policy_env_steps": self._elapsed_steps.clone(),
            "settling_ticks": torch.tensor([b.settling_ticks for b in budgets]),
            "physics_ticks": torch.tensor([b.physics_ticks for b in budgets]),
            "planning_failures": torch.tensor(
                [b.left_plan_failures + b.right_plan_failures for b in budgets]
            ),
        }

    def chunk_step(self, chunk_actions):
        if self._closed or self._is_start or any(obs is None for obs in self._raw_obs):
            raise RuntimeError("Call reset before executing native X-VLA actions")
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        actions = np.asarray(chunk_actions)
        expected = (self.num_envs, protocol.ANCHORS, 20)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(
                f"Expected finite native X-VLA actions {expected}, got {actions.shape}"
            )
        commands = [protocol.actions_to_env(chunk) for chunk in actions]
        rewards = torch.zeros(self.num_envs, protocol.ANCHORS)
        terminations = torch.zeros_like(rewards, dtype=torch.bool)
        truncations = torch.zeros_like(rewards, dtype=torch.bool)
        executed = torch.zeros_like(rewards, dtype=torch.bool)
        try:
            for index, (task, native_commands) in enumerate(zip(self._tasks, commands)):
                if self._terminations[index] or self._truncations[index]:
                    terminations[index] = self._terminations[index]
                    truncations[index] = self._truncations[index]
                    continue
                with task.action_context():
                    for slot, command in enumerate(native_commands):
                        if not (self._terminations[index] or self._truncations[index]):
                            before = int(task.env.take_action_cnt)
                            limit = min(self.max_episode_steps, int(task.env.step_lim))
                            if before >= limit:
                                self._truncations[index] = True
                            else:
                                task.take_action(command)
                                after = int(task.env.take_action_cnt)
                                if after != before + 1:
                                    raise RuntimeError(
                                        "Native action did not advance its action counter once"
                                    )
                                task.budget.anchors_sent += 1
                                task.budget.anchors = after
                                self._elapsed_steps[index] += 1
                                executed[index, slot] = True
                                self._raw_obs[index] = task.observe()
                                if task.env.check_success():
                                    self._success[index] = True
                                    self._terminations[index] = True
                                    rewards[index, slot] = self.reward_coef
                                    self._returns[index] += self.reward_coef
                                elif not bool(task.env.actor_pose):
                                    self._terminations[index] = True
                                elif after >= limit:
                                    self._truncations[index] = True
                        terminations[index, slot] = self._terminations[index]
                        truncations[index, slot] = self._truncations[index]
                if self._terminations[index] or self._truncations[index]:
                    self._raw_obs[index] = deepcopy(self._raw_obs[index])
                    task.close()
        except Exception:
            with suppress(Exception):
                self.close()
            raise
        obs = self._observations()
        infos = {
            "episode": self._episode_info(),
            "success": self._success.clone(),
            "executed_mask": executed,
            "valid_action_mask": executed.clone(),
            "final_observation": obs,
        }
        return [obs], rewards, terminations, truncations, [infos]

    def close(self) -> None:
        if not self._closed:
            failure = None
            for task in self._tasks:
                try:
                    task.close()
                except Exception as error:
                    if failure is None:
                        failure = error
            self._closed = True
            if failure is not None:
                raise failure
