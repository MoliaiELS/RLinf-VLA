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

import json
import sys
import weakref
from types import ModuleType, SimpleNamespace

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
        def action_context(self):
            if not self.initialized:
                raise RuntimeError("Cannot access a released native scene")
            return super().action_context()

        def close(self):
            if self.initialized:
                super().close()

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
    assert not env._tasks[0].initialized
    assert env._tasks[0].close_count == 1
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
    assert not env._tasks[0].initialized
    assert env._tasks[0].close_count == 1


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
    assert env._tasks[0].close_count == 1
    assert not any(task.initialized for task in env._tasks)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_cycle_seeds_cover_only_the_declared_training_pool_in_group_order(native_task):
    pool = [12000, 12001, 12002, 12003]
    env = environment(
        config(
            group_size=4, reset_seeds=pool, reset_seed_mode="cycle", max_episode_steps=1
        ),
        num_envs=4,
    )
    for seed in (*pool, pool[0]):
        _, info = env.reset()
        assert info["reset_seeds"].tolist() == [seed] * 4
        assert all(task.actual_seed == seed for task in env._tasks)
        env.chunk_step(actions(4))
        assert not any(task.initialized for task in env._tasks)
        env.update_reset_state_ids()


def test_random_seed_selection_retains_the_existing_default_stream(native_task):
    env = environment(config(group_size=2), num_envs=4)
    generator = np.random.default_rng(env.seed)
    for _ in range(4):
        expected = np.repeat(generator.choice(env.cfg.reset_seeds, size=2), 2).tolist()
        assert env.reset_state_ids.tolist() == expected
        env.update_reset_state_ids()


def test_fixed_cycle_ids_remain_fixed_and_multiple_groups_share_their_own_seed(
    native_task,
):
    env = environment(
        config(group_size=2, reset_seed_mode="cycle", use_fixed_reset_state_ids=True),
        num_envs=4,
    )
    assert env.reset_state_ids.tolist() == [10000, 10000, 10001, 10001]
    state = env.get_reset_schedule_state()
    env.update_reset_state_ids()
    assert env.get_reset_schedule_state() == state


@pytest.mark.parametrize("mode", ["random", "cycle"])
def test_episode_boundary_schedule_resume_preserves_selected_next_seeds_and_rng(
    native_task, mode
):
    cfg = config(
        group_size=2,
        reset_seed_mode=mode,
        reset_seeds=[12000, 12001, 12002, 12003],
        max_episode_steps=1,
    )
    original = environment(cfg, num_envs=4)
    original.reset()
    with pytest.raises(RuntimeError, match="episode boundary"):
        original.get_reset_schedule_state()
    original.chunk_step(actions(4))
    original.update_reset_state_ids()
    saved = json.loads(json.dumps(original.get_reset_schedule_state()))
    assert saved["reset_count"] == 1
    resumed = environment(cfg, num_envs=4)
    resumed.set_reset_schedule_state(saved)
    assert resumed.get_reset_schedule_state() == saved
    assert not resumed._terminations.any() and not resumed._truncations.any()
    for _ in range(4):
        original_seeds = original.reset()[1]["reset_seeds"]
        resumed_seeds = resumed.reset()[1]["reset_seeds"]
        torch.testing.assert_close(original_seeds, resumed_seeds)
        assert set(resumed_seeds.tolist()) <= {12000, 12001, 12002, 12003}
        for env in (original, resumed):
            env.chunk_step(actions(4))
            env.update_reset_state_ids()
        assert resumed.get_reset_schedule_state() == original.get_reset_schedule_state()
    with pytest.raises(RuntimeError, match="before the first episode"):
        resumed.set_reset_schedule_state(saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed_pool", [13000, 13001, 13002, 13003]),
        ("group_size", 2),
        ("reset_seed_mode", "random"),
        ("seed_offset", 1),
        ("use_fixed_reset_state_ids", True),
        ("reset_state_ids", [13000]),
        ("cycle_cursor", 2),
        ("reset_count", -1),
        ("generator_state", {}),
    ],
)
def test_schedule_restore_rejects_incompatible_state_without_mutating_stream(
    native_task, field, value
):
    env = environment(config(reset_seed_mode="cycle"))
    before = env.get_reset_schedule_state()
    incompatible = {**before, field: value}
    with pytest.raises(ValueError):
        env.set_reset_schedule_state(incompatible)
    assert env.get_reset_schedule_state() == before


def test_schedule_restore_rejects_split_grpo_groups(native_task):
    env = environment(config(group_size=2, reset_seed_mode="cycle"), num_envs=4)
    state = env.get_reset_schedule_state()
    state["reset_state_ids"] = [10000, 10001, 10001, 10001]
    with pytest.raises(ValueError, match="same native seed"):
        env.set_reset_schedule_state(state)


class NativeResource:
    """Scene-bound simulator resource with a weak-referenceable lifetime."""

    def __init__(self, scene=None):
        self.scene = scene

    def step(self):
        pass

    def get_timestep(self):
        return 1 / 250

    def reset(self, *, reset_seed):
        assert reset_seed


class NativeEpisode:
    """Pinned vendor boundary, including its no-op episode close."""

    closed = []
    scene_refs = []
    fail_setup = False
    fail_close = False

    def setup_demo(self, **kwargs):
        del kwargs
        self.scene = NativeResource()
        type(self).scene_refs.append(weakref.ref(self.scene))
        self.renderer = NativeResource(self.scene)
        self.self_cycle = self
        self.load_robot()
        self.check_stable()
        self.step_lim = 60
        if self.fail_setup:
            raise RuntimeError("vendor setup failed")

    def get_cluttered_table(self, offsets=()):
        pass

    def load_robot(self):
        self.robot = NativeResource(self.scene)
        self.robot.communication_flag = False
        for side in ("left", "right"):
            planner = NativeResource(self.scene)
            planner.motion_gen = NativeResource()
            planner.motion_gen_batch = NativeResource()
            setattr(self.robot, f"{side}_planner", planner)

    def check_stable(self):
        self.scene.step()

    def _update_render(self):
        pass

    def close_env(self, *, clear_cache):
        type(self).closed.append(clear_cache)
        if self.fail_close:
            raise RuntimeError("vendor close failed")


def task_with_native_episode():
    from tools.xvla.robotwin import runner

    class Episode(NativeEpisode):
        closed = []
        scene_refs = []

    task = object.__new__(xvla_env.NativeTask)
    task._task_type = Episode
    task.env = None
    task.config = {}
    task.module = SimpleNamespace()
    task.task_name = "move_stapler_pad"
    task.runner = runner
    task.initialized = False
    return task, Episode


def test_native_task_recreates_episode_and_releases_scene_robot_planner_cycles():
    task, episode_type = task_with_native_episode()
    task.reset(0, 12000)
    episode = weakref.ref(task.env)
    scene = weakref.ref(task.env.scene)
    planner = weakref.ref(task.env.robot.left_planner)
    task.reset(1, 12001)
    assert episode() is None and scene() is None and planner() is None
    budget = task.budget
    assert budget.settling_ticks == 1 and budget.step_lim == 60
    task.close()
    task.close()
    assert task.env is None and not task.initialized
    assert task.budget is budget
    assert not any(reference() is not None for reference in episode_type.scene_refs)
    assert episode_type.closed == [False, False]


def test_native_task_releases_partial_setup_and_does_not_mask_the_failure():
    task, episode_type = task_with_native_episode()
    episode_type.fail_setup = True
    with pytest.raises(RuntimeError, match="vendor setup failed"):
        task.reset(0, 12000)
    assert task.env is None and not task.initialized
    assert episode_type.scene_refs[0]() is None
    assert episode_type.closed == [False]
    task.close()
    assert episode_type.closed == [False]


def test_native_task_close_releases_resources_even_when_vendor_close_raises():
    task, episode_type = task_with_native_episode()
    task.reset(0, 12000)
    episode_type.fail_close = True
    with pytest.raises(RuntimeError, match="vendor close failed"):
        task.close()
    assert task.env is None and not task.initialized
    assert episode_type.scene_refs[0]() is None
    task.close()
    assert episode_type.closed == [False]


def test_training_renderer_and_scene_share_the_explicit_vulkan_device(monkeypatch):
    from tools.xvla.robotwin.bind_renderer import bind_renderer

    class NativeRenderer:
        def __init__(self, device=None):
            self.device = device

    class LegacyRenderer(NativeRenderer):
        def __init__(self, **kwargs):
            super().__init__()

        def create_material(self):
            return "legacy material"

    class Engine:
        def create_scene(self, config=None):
            raise AssertionError("The default scene would use an unbound GPU")

    core = ModuleType("sapien.core")
    core.Engine = Engine
    core.SapienRenderer = LegacyRenderer
    core.SceneConfig = object
    sapien = ModuleType("sapien")
    sapien.core = core
    sapien.SapienRenderer = LegacyRenderer
    sapien.Device = lambda alias: SimpleNamespace(alias=alias)
    sapien.Scene = tuple
    sapien.physx = SimpleNamespace(
        PhysxCpuSystem=object, set_scene_config=lambda config: None
    )
    sapien.render = SimpleNamespace(
        RenderSystem=lambda device: SimpleNamespace(device=device)
    )
    sapien.pysapien = SimpleNamespace(
        render=SimpleNamespace(SapienRenderer=NativeRenderer)
    )
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.core", core)
    monkeypatch.setenv("FYP_ROBOTWIN_VULKAN_DEVICE", "pci:49:00.0")

    bind_renderer()
    scene = Engine().create_scene()
    assert LegacyRenderer().device is None
    bind_renderer(pin_legacy_renderer=True)
    renderer = core.SapienRenderer(device="cuda:0")
    assert type(renderer) is sapien.SapienRenderer
    assert renderer.create_material() == "legacy material"
    assert renderer.device is scene[1].device
    assert renderer.device.alias == "pci:49:00.0"
    init = LegacyRenderer.__init__
    bind_renderer(pin_legacy_renderer=True)
    assert LegacyRenderer.__init__ is init
    assert Engine().create_scene()[1].device is renderer.device
