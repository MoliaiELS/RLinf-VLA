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

"""Exercise native X-VLA episodes through RLinf's environment contract."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from robot_mocks.robotwin_xvla import FakeNativeTask

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.robotwin import xvla_env


@pytest.fixture
def native_task(monkeypatch):
    class Task(FakeNativeTask):
        pass

    monkeypatch.setattr(xvla_env, "NativeTask", Task)
    return Task


def config(**overrides):
    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "xvla_native": True,
            "robotwin_root": "/vendor/robotwin",
            "task_name": "move_stapler_pad",
            "condition": "easy",
            "seed": 123,
            "group_size": 1,
            "reset_seeds": [10000, 10001, 10002, 10003],
            "max_episode_steps": 60,
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            **overrides,
        }
    )


def environment(cfg=None, num_envs=1):
    cfg = config() if cfg is None else cfg
    return get_env_cls("robotwin", cfg)(cfg, num_envs, 0, 1, None)


def actions(batch=1):
    chunk = np.zeros((batch, 30, 20), dtype=np.float32)
    for offset in (0, 10):
        chunk[..., offset + 3 : offset + 9] = [1, 0, 0, 1, 0, 0]
        chunk[..., offset + 9] = 0.8
    chunk[:, :, 0] = np.arange(30) / 10
    chunk[:, :, 10] = -np.arange(30) / 10
    return chunk


def test_factory_and_action_dispatch_preserve_native_dual_arm_values(native_task):
    cfg = config()
    assert get_env_cls("robotwin", cfg) is xvla_env.XVLARoboTwinEnv
    chunk = actions()
    actual = prepare_actions(chunk, "robotwin", "xvla", 30, 20, env_cfg=cfg)
    np.testing.assert_array_equal(actual, chunk)
    bad = chunk.copy()
    bad[..., 19] = -0.1
    with pytest.raises(ValueError, match="postprocessed probabilities"):
        prepare_actions(bad, "robotwin", "xvla", 30, 20, env_cfg=cfg)


def test_measured_observations_preserve_all_three_raw_cameras(native_task):
    env = environment()
    obs, _ = env.reset()
    assert obs["main_images"].shape == (1, 240, 320, 3)
    assert obs["wrist_images"].shape == (1, 2, 240, 320, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["main_images"].unique().tolist() == [1]
    assert obs["wrist_images"][:, 0].unique().tolist() == [2]
    assert obs["wrist_images"][:, 1].unique().tolist() == [3]
    assert obs["task_descriptions"] == ["move stapler pad"]
    torch.testing.assert_close(obs["states"][:, [9, 19]], torch.tensor([[0.4, -0.4]]))
    assert obs["reset_mask"].tolist() == [True]


def test_success_stops_at_exact_anchor_and_never_executes_post_done(native_task):
    native_task.success_at = 2
    env = environment()
    env.reset()
    obs, rewards, terms, truncs, infos = env.chunk_step(actions())
    assert len(env._tasks[0].commands) == 2
    assert rewards.sum().item() == 1
    assert rewards[0, 1] == 1
    assert terms[0].tolist() == [False] + [True] * 29
    assert not truncs.any()
    assert infos[-1]["executed_mask"][0].tolist() == [True, True] + [False] * 28
    assert infos[-1]["episode"]["policy_env_steps"].item() == 2
    assert infos[-1]["episode"]["settling_ticks"].item() == 2500
    assert obs[-1]["states"][0, 0] == pytest.approx(0.1)
    assert obs[-1]["states"][0, 10] == pytest.approx(-0.1)
    assert obs[-1]["reset_mask"].tolist() == [False]
    _, rewards, terms, _, infos = env.chunk_step(actions())
    assert len(env._tasks[0].commands) == 2
    assert rewards.sum().item() == 0
    assert terms.all()
    assert not infos[-1]["executed_mask"].any()


@pytest.mark.parametrize(
    "native_limit,configured_limit,expected", [(60, 3, 3), (4, 60, 4)]
)
def test_native_and_configured_time_limits_truncate_without_noop_commands(
    native_task, native_limit, configured_limit, expected
):
    native_task.native_limit = native_limit
    env = environment(config(max_episode_steps=configured_limit))
    env.reset()
    _, rewards, terms, truncs, infos = env.chunk_step(actions())
    assert len(env._tasks[0].commands) == expected
    assert not rewards.any() and not terms.any()
    assert truncs[0, expected - 1 :].all()
    assert infos[-1]["executed_mask"].sum().item() == expected
    assert env.elapsed_steps.item() == expected


def test_native_actor_failure_is_terminal_with_zero_reward(native_task):
    native_task.fail_at = 3
    env = environment()
    env.reset()
    _, rewards, terms, truncs, infos = env.chunk_step(actions())
    assert len(env._tasks[0].commands) == 3
    assert not rewards.any() and not truncs.any()
    assert terms[0, 2:].all()
    assert not infos[-1]["episode"]["success_once"].any()


def test_grpo_group_resets_share_initial_state_and_keep_explicit_seed_pool(native_task):
    env = environment(config(group_size=2), num_envs=4)
    for _ in range(3):
        _, info = env.reset()
        seeds = info["reset_seeds"].tolist()
        assert seeds[0] == seeds[1] and seeds[2] == seeds[3]
        assert set(seeds).issubset(set(env.cfg.reset_seeds))
        env.update_reset_state_ids()
    with pytest.raises(ValueError, match="same native seed"):
        env.reset(env_seeds=[10000, 10001, 10002, 10002])


def test_native_action_counter_error_is_propagated(native_task):
    native_task.advance_counter = False
    env = environment()
    env.reset()
    with pytest.raises(RuntimeError, match="advance its action counter"):
        env.chunk_step(actions())


@pytest.mark.parametrize(
    "override",
    [
        {"auto_reset": True},
        {"ignore_terminations": True},
        {"reset_seeds": []},
        {"reset_seeds": [1, 1]},
        {"condition": "modified"},
        {"enable_offload": True},
    ],
)
def test_unsupported_protocols_fail_before_native_initialization(native_task, override):
    with pytest.raises(ValueError):
        environment(config(**override))


def test_close_releases_each_native_task_once(native_task):
    env = environment()
    env.reset()
    env.close()
    env.close()
    assert env._tasks[0].close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_invalid_initial_success_fails_and_releases_the_batch(native_task):
    native_task.success_at = 0
    env = environment(num_envs=2)
    with pytest.raises(RuntimeError, match="initial state already"):
        env.reset()
    assert all(task.close_count == 1 for task in env._tasks)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
