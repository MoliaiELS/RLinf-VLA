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

"""Exercise X-VLA through the policy and actual RLinf algorithm contracts."""

import copy
import inspect
import socket
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.registry import policy_loss
from rlinf.models.embodiment.xvla.policy import (
    DomainEmbedding,
    XVLAPolicy,
    pack_joint_logprob,
)


class VendorSpace:
    dim_action = 20

    def preprocess(self, proprio, action):
        action = action.clone()
        action[..., [9, 19]] = 0
        return proprio, action

    def postprocess(self, action):
        action[..., [9, 19]] = action[..., [9, 19]].sigmoid()
        return action


class VendorPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.linspace(0.1, 0.3, 20))

    def forward(self, action_with_noise, vlm_features, **kwargs):
        return self.theta + action_with_noise * 0.1 + vlm_features.mean() * 0.01


class VendorModel(nn.Module):
    num_actions = 30
    action_mode = "ee6d"

    def __init__(self):
        super().__init__()
        self.action_space = VendorSpace()
        self.transformer = VendorPredictor()
        self.encoder = nn.Linear(1, 8)
        self.config = SimpleNamespace(florence_config=SimpleNamespace(projection_dim=8))

    def forward_vlm(self, ids, images, mask):
        features = self.encoder(images.mean(dim=(1, 2, 3, 4))[:, None])
        return {
            "vlm_features": features[:, None],
            "aux_visual_inputs": features[:, None],
        }


class VendorProcessor:
    def __call__(self, images, text):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "image_input": torch.zeros(1, 3, 3, 4, 4),
            "image_mask": torch.ones(1, 3, dtype=torch.bool),
        }


def make_policy(critic=True):
    cfg = OmegaConf.create(
        {
            "action_dim": 20,
            "num_action_chunks": 30,
            "is_lora": False,
            "add_value_head": critic,
            "xvla": {
                "update_scope": "full",
                "steps": 2,
                "noise_hidden_dim": 8,
                "noise_std": [0.1] * 20,
                "checkpoint_transformer": False,
            },
        }
    )
    torch.manual_seed(1234)
    return XVLAPolicy(VendorModel(), VendorProcessor(), cfg)


def observation():
    return {
        "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 20),
        "task_descriptions": ["move stapler pad"] * 2,
    }


class ImageVendorProcessor:
    tokenizer = SimpleNamespace(pad_token_id=17)

    def __call__(self, images, text):
        pixels = torch.stack(
            [torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images]
        )
        return {
            "input_ids": torch.arange(1, len(text.split()) + 1)[None],
            "image_input": pixels[None].float() / 255,
            "image_mask": torch.ones(1, 3, dtype=torch.bool),
        }


def unequal_observation():
    return {
        "main_images": torch.arange(4, dtype=torch.uint8)[:, None, None, None]
        .expand(4, 4, 4, 3)
        .mul(31),
        "wrist_images": torch.arange(8, dtype=torch.uint8)
        .reshape(4, 2, 1, 1, 1)
        .expand(4, 2, 4, 4, 3)
        .mul(23),
        "states": torch.linspace(-0.7, 0.9, 80).reshape(4, 20),
        "task_descriptions": ["a", "a b", "a b c", "a b c d"],
    }


def recorded_policy():
    model = make_policy()
    model.processor = ImageVendorProcessor()
    calls = {name: [] for name in ("encoder", "predictor", "noise", "value")}
    forward_vlm = model.core.forward_vlm

    def encode(ids, images, mask):
        calls["encoder"].append(tuple(ids.shape))
        return forward_vlm(ids, images, mask)

    model.core.forward_vlm = encode
    model.core.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: calls["predictor"].append(
            kwargs["action_with_noise"].shape[0]
        ),
        with_kwargs=True,
    )
    model.sampler.noise_head.register_forward_pre_hook(
        lambda module, args: calls["noise"].append(args[0].shape[0])
    )
    model.value_head.register_forward_pre_hook(
        lambda module, args: calls["value"].append(args[0].shape[0])
    )
    return model, calls


def test_rollout_preserves_full_initial_noise_draw_and_processes_samples_independently():
    model, calls = recorded_policy()
    torch.manual_seed(715)
    expected_initial = torch.randn(4, 30, 20)
    torch.manual_seed(715)
    _, rollout = model.predict_action_batch(unequal_observation(), mode="train")
    cache = rollout["forward_inputs"]
    torch.testing.assert_close(cache["chains"][:, 0], expected_initial, rtol=0, atol=0)
    assert cache["input_ids"].tolist() == [
        [1, 17, 17, 17],
        [1, 2, 17, 17],
        [1, 2, 3, 17],
        [1, 2, 3, 4],
    ]
    assert calls["encoder"] == [(1, 4)] * 4
    assert calls["predictor"] == [1] * (4 * model.steps)
    assert calls["noise"] == [1] * (4 * model.steps)
    assert calls["value"] == [1] * 4
    assert rollout["prev_logprobs"].dtype == torch.float32


@pytest.mark.parametrize("partition", [(1, 1, 1, 1), (2, 2), (3, 1)])
def test_cached_batch_replay_is_invariant_to_permuted_actor_partitions(partition):
    model, _ = recorded_policy()
    initial = torch.linspace(-1.1, 1.3, 4 * 30 * 20).reshape(4, 30, 20)
    torch.manual_seed(716)
    _, rollout = model.predict_action_batch(
        unequal_observation(), mode="train", initial_noise=initial
    )
    cache = rollout["forward_inputs"]
    cache["executed_mask"] = (
        torch.arange(30)[None] < torch.tensor([30, 11, 1, 0])[:, None]
    )
    expected = model.default_forward(cache, compute_values=True, compute_entropy=True)
    torch.testing.assert_close(
        expected["logprobs"][:3], rollout["prev_logprobs"][:3], rtol=0, atol=0
    )
    assert expected["logprobs"][3].count_nonzero() == 0
    weights = torch.tensor([0.7, -0.3, 1.1, -0.8])

    def objective(output):
        return (
            (output["logprobs"].flatten(1).sum(1) * weights).sum()
            + 0.2 * output["values"].square().sum()
            - 0.01 * output["entropy"].sum()
        )

    objective(expected).backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    for name in (
        "core.transformer.theta",
        "sampler.noise_head.net.2.weight",
        "value_head.2.weight",
    ):
        assert torch.isfinite(reference_gradients[name]).all()
        assert reference_gradients[name].norm() > 0

    replay, calls = recorded_policy()
    replay.load_state_dict(model.state_dict(), strict=True)
    order = torch.tensor([2, 0, 3, 1])
    outputs = []
    start = 0
    for count in partition:
        ids = order[start : start + count]
        inputs = {name: value[ids] for name, value in cache.items()}
        outputs.append(
            replay.default_forward(inputs, compute_values=True, compute_entropy=True)
        )
        start += count
    inverse = order.argsort()
    actual = {
        name: torch.cat([output[name] for output in outputs])[inverse]
        for name in ("logprobs", "values", "entropy")
    }
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    objective(actual).backward()
    for name, parameter in replay.named_parameters():
        if name in reference_gradients:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_gradients[name], rtol=2e-5, atol=2e-5
            )
    assert calls["encoder"] == [(1, 4)] * 4
    assert calls["predictor"] == [1] * (4 * replay.steps)
    assert calls["noise"] == [1] * (4 * replay.steps)
    assert calls["value"] == [1] * 4
    assert actual["logprobs"].dtype == torch.float32


def test_rollout_recompute_full_path_and_prefix_mask():
    model = make_policy()
    actions, rollout = model.predict_action_batch(observation(), mode="train")
    assert actions.shape == (2, 30, 20)
    assert np.all((actions[..., [9, 19]] > 0) & (actions[..., [9, 19]] < 1))
    cache = rollout["forward_inputs"]
    assert cache["chains"].shape == (2, 3, 30, 20)
    assert not cache["chains"].requires_grad
    output = model.default_forward(cache, compute_values=True)
    torch.testing.assert_close(
        output["logprobs"], rollout["prev_logprobs"], rtol=0, atol=0
    )
    cache["executed_mask"] = torch.zeros(2, 30, dtype=torch.bool)
    cache["executed_mask"][0, :3] = True
    masked = model.default_forward(cache)
    torch.testing.assert_close(masked["logprobs"][0].sum(), output["logprobs"][0].sum())
    assert masked["logprobs"][1].count_nonzero() == 0
    cache["executed_mask"][1, 2] = True
    with pytest.raises(ValueError, match="prefix"):
        model.default_forward(cache)


def test_zero_reward_ppo_updates_actor_and_corrects_critic():
    model = make_policy()
    _, rollout = model.predict_action_batch(observation(), mode="train")
    output = model.default_forward(rollout["forward_inputs"], compute_values=True)
    values = torch.tensor([[0.5, 0.2], [0.0, 0.0]])
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=torch.zeros(1, 2),
        values=values,
        dones=torch.tensor([[False, False], [True, True]]),
        normalize_advantages=False,
    )
    torch.testing.assert_close(advantages, torch.tensor([[-0.5, -0.2]]))
    before_actor = model.core.transformer.theta.detach().clone()
    before_value = model.value_head[-1].weight.detach().clone()
    loss, _ = policy_loss(
        task_type="embodied",
        loss_type="actor_critic",
        reward_type="chunk_level",
        logprob_type="chunk_level",
        single_action_dim=1,
        logprobs=output["logprobs"],
        old_logprobs=rollout["prev_logprobs"],
        advantages=advantages.squeeze(0),
        returns=returns.squeeze(0),
        values=output["values"],
        prev_values=rollout["prev_values"].squeeze(-1),
        loss_mask=torch.ones(2, dtype=torch.bool),
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
        value_clip=0.2,
        huber_delta=10.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss.backward()
    assert model.core.transformer.theta.grad.norm() > 0
    assert model.sampler.noise_head.net[-1].weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(model.core.transformer.theta, before_actor)
    assert not torch.equal(model.value_head[-1].weight, before_value)


def test_grpo_actor_without_critic_and_checkpoint_roundtrip():
    from rlinf.algorithms.advantages import compute_grpo_advantages

    model = make_policy(critic=False)
    _, rollout = model.predict_action_batch(observation(), mode="train")
    output = model.default_forward(rollout["forward_inputs"])
    advantages, _ = compute_grpo_advantages(
        rewards=torch.tensor([1.0, 0.0]),
        loss_mask=torch.ones(1, 2),
        group_size=2,
    )
    loss, _ = policy_loss(
        task_type="embodied",
        loss_type="actor",
        reward_type="chunk_level",
        logprob_type="chunk_level",
        single_action_dim=1,
        logprobs=output["logprobs"],
        old_logprobs=rollout["prev_logprobs"],
        advantages=advantages.squeeze(0),
        loss_mask=torch.ones(2, dtype=torch.bool),
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
    )
    loss.backward()
    assert not hasattr(model, "value_head")
    assert model.core.transformer.theta.grad.norm() > 0
    restored = make_policy(critic=False)
    restored.load_state_dict(model.state_dict(), strict=True)
    actual = restored.default_forward(rollout["forward_inputs"])
    torch.testing.assert_close(actual["logprobs"], output["logprobs"], rtol=0, atol=0)


def test_domain_embedding_preserves_other_rows_under_adamw():
    original = nn.Embedding(8, 4)
    expected = original.weight.detach().clone()
    module = DomainEmbedding(original, 6)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.1)
    module(torch.tensor([6])).sum().backward()
    optimizer.step()
    torch.testing.assert_close(module(torch.tensor([0, 1, 7])), expected[[0, 1, 7]])
    assert not torch.equal(module(torch.tensor([6]))[0], expected[6])


def test_deployment_sampler_is_explicit_and_value_ignores_path_noise():
    model = make_policy()
    _, first = model.predict_action_batch(observation(), mode="eval")
    _, second = model.predict_action_batch(observation(), mode="eval")
    torch.testing.assert_close(first["prev_values"], second["prev_values"])
    assert first["prev_logprobs"].count_nonzero() == 0
    with pytest.raises(ValueError, match="mode"):
        model.predict_action_batch(observation())


def test_native_chunk_rewards_reach_real_gae_and_grpo_preprocessing():
    from rlinf.algorithms.registry import calculate_adv_and_returns

    rewards = torch.zeros(1, 2, 30)
    rewards[0, 0, 1] = 1.0
    dones = torch.zeros(2, 2, 30, dtype=torch.bool)
    dones[1, 0, 1:] = True
    dones[1, 1, -1] = True
    mask = torch.ones_like(rewards, dtype=torch.bool)
    mask[0, 0, 2:] = False
    arguments = {
        "task_type": "embodied",
        "reward_type": "chunk_level",
        "rewards": rewards,
        "dones": dones,
        "loss_mask": mask,
        "normalize_advantages": False,
    }
    gae = calculate_adv_and_returns(
        **arguments,
        adv_type="gae",
        gamma=0.99,
        gae_lambda=0.95,
        values=torch.tensor([[[0.5], [0.2]], [[5.0], [5.0]]]),
    )
    torch.testing.assert_close(gae["returns"].flatten(), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(gae["advantages"].flatten(), torch.tensor([0.5, -0.2]))
    grpo = calculate_adv_and_returns(**arguments, adv_type="grpo", group_size=2)
    torch.testing.assert_close(
        grpo["advantages"].flatten(),
        torch.tensor([2**-0.5, -(2**-0.5)]),
        atol=2e-6,
        rtol=1e-5,
    )


def test_builtin_registration_and_config_contracts(monkeypatch, tmp_path):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    import rlinf.models as models
    from rlinf.config import EMBODIED_MODEL, SupportedModel, validate_xvla_cfg

    assert SupportedModel.XVLA in EMBODIED_MODEL
    assert "xvla" in models._MODEL_REGISTRY
    validate_xvla_cfg(OmegaConf.create({"runner": {"only_eval": True}}))
    for key in ("XVLA_MODEL_PATH", "ROBOTWIN_PATH", "XVLA_RUN_DIR"):
        monkeypatch.setenv(key, str(tmp_path))
    config_root = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        for config_name in (
            "robotwin_ppo_xvla",
            "robotwin_grpo_xvla",
            "robotwin_ppo_xvla_acceptance",
            "robotwin_grpo_xvla_acceptance",
        ):
            cfg = compose(config_name=config_name)
            validate_xvla_cfg(cfg)
            cfg.actor.model.logprob_action_dim = 20
            with pytest.raises(ValueError, match="logprob_action_dim"):
                validate_xvla_cfg(cfg)
        cfg = compose(config_name="robotwin_ppo_xvla")
        cfg.algorithm.filter_rewards = True
        with pytest.raises(ValueError, match="zero-reward"):
            validate_xvla_cfg(cfg)


def test_prompt_heads_with_real_vendor_transformer_and_activation_checkpoint():
    from xvla_rlinf.vendor.xvla.transformer import SoftPromptedTransformer

    core = VendorModel()
    core.transformer = SoftPromptedTransformer(
        hidden_size=16,
        multi_modal_input_size=8,
        depth=1,
        num_heads=2,
        num_domains=8,
        max_len_seq=128,
    )
    cfg = OmegaConf.create(
        {
            "action_dim": 20,
            "num_action_chunks": 30,
            "is_lora": False,
            "add_value_head": False,
            "xvla": {
                "update_scope": "prompt_heads",
                "steps": 2,
                "noise_std": [0.1] * 20,
                "checkpoint_transformer": True,
            },
        }
    )
    model = XVLAPolicy(core, VendorProcessor(), cfg)
    _, rollout = model.predict_action_batch(observation(), mode="train")
    output = model.default_forward(rollout["forward_inputs"])
    torch.testing.assert_close(
        output["logprobs"], rollout["prev_logprobs"], rtol=0, atol=0
    )
    (-output["logprobs"].sum()).backward()
    selected = {
        name: parameter
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    }
    assert len(selected) == 5
    assert all("selected_weight" in name for name in selected)
    assert all(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in selected.values()
    )
    assert all(parameter.grad is None for parameter in core.encoder.parameters())


def fsdp_actor_config():
    return OmegaConf.create(
        {
            "model": {"model_type": "xvla", "is_lora": False},
            "enable_offload": False,
            "fsdp_config": {
                "sharding_strategy": "no_shard",
                "disable": True,
                "use_orig_params": True,
                "cpu_offload": False,
                "forward_prefetch": False,
                "backward_prefetch": None,
                "limit_all_gathers": False,
                "mixed_precision": {
                    "param_dtype": "fp32",
                    "reduce_dtype": "fp32",
                    "buffer_dtype": "fp32",
                },
            },
        }
    )


@pytest.mark.parametrize(
    "invalid",
    ["multiple_ranks", "sharding", "child_wrap", "orig_params", "offload", "precision"],
)
def test_frozen_fsdp_parameters_require_the_single_rank_fp32_contract(invalid):
    from torch.distributed.fsdp import MixedPrecision

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy

    model = make_policy()
    model.core.encoder.requires_grad_(False)
    cfg = fsdp_actor_config()
    world_size = 2 if invalid == "multiple_ranks" else 1
    wrap_policy = object() if invalid == "child_wrap" else None
    if invalid == "sharding":
        cfg.fsdp_config.sharding_strategy = "full_shard"
    elif invalid == "orig_params":
        cfg.fsdp_config.use_orig_params = False
    elif invalid == "offload":
        cfg.enable_offload = True
    precision = torch.bfloat16 if invalid == "precision" else torch.float32
    strategy = FSDPStrategy(cfg, world_size)
    with pytest.raises(ValueError, match="one FP32 NO_SHARD root"):
        strategy._get_ignored_parameters(
            model, MixedPrecision(param_dtype=precision), wrap_policy
        )


def test_other_models_do_not_opt_into_ignored_frozen_parameters():
    from torch.distributed.fsdp import MixedPrecision

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy

    cfg = fsdp_actor_config()
    cfg.model.model_type = "openpi"
    model = nn.Linear(2, 2).requires_grad_(False)
    assert (
        FSDPStrategy(cfg, 2)._get_ignored_parameters(model, MixedPrecision(), None)
        is None
    )


def test_frozen_fsdp_weights_avoid_flat_grad_and_roundtrip_full_state(
    monkeypatch, tmp_path
):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel

    from rlinf.hybrid_engines.fsdp.strategy import fsdp as strategy_module
    from rlinf.scheduler import Worker

    if dist.is_initialized():
        pytest.skip("CPU FSDP contract owns its single-rank process group")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("USE_LIBUV", "0")
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        model = make_policy()
        model.core.encoder.requires_grad_(False)
        model.register_buffer("frozen_buffer", torch.tensor([17.0]))
        frozen = model.get_fsdp_ignored_parameters()
        expected_ids = {id(parameter) for parameter in frozen}
        trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        _, rollout = model.predict_action_batch(observation(), mode="train")
        captured = {}

        def cpu_fsdp(**kwargs):
            captured.update(kwargs)
            # Keep the real FSDP implementation; adapt only the accelerator boundary.
            kwargs["device_id"] = torch.device("cpu")
            kwargs["sync_module_states"] = False
            return FullyShardedDataParallel(**kwargs)

        monkeypatch.setattr(strategy_module, "FSDP", cpu_fsdp)
        strategy = strategy_module.FSDPStrategy(fsdp_actor_config(), 1)
        wrapped = strategy.wrap_model(model, None)
        assert {
            id(parameter) for parameter in captured["ignored_states"]
        } == expected_ids
        assert sum(wrapped._handle.flat_param._numels) == trainable_numel
        result = wrapped(forward_inputs=rollout["forward_inputs"], compute_values=True)
        (-result["logprobs"].sum() + result["values"].square().sum()).backward()
        assert wrapped._handle.flat_param.grad is not None
        assert all(parameter.grad is None for parameter in frozen)
        assert model.core.transformer.theta.grad.norm() > 0
        full_state = strategy.get_model_state_dict(wrapped, True, True)
        assert full_state.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(full_state[name], expected[name], rtol=0, atol=0)
        checkpoint = tmp_path / "full_weights.pt"
        torch.save(full_state, checkpoint)
        with torch.no_grad():
            model.core.encoder.weight.add_(1)
            model.core.transformer.theta.add_(1)
            model.frozen_buffer.add_(1)
        strategy.load_model_with_state_dict(
            wrapped, torch.load(checkpoint, weights_only=True), True, True
        )
        restored = strategy.get_model_state_dict(wrapped, True, True)
        for name in expected:
            torch.testing.assert_close(restored[name], expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class _DensityPolicy(torch.nn.Module):
    """Small differentiable policy using the embodied actor forward contract."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25]))

    def forward(self, forward_inputs, **kwargs):
        return {"logprobs": self.weight * forward_inputs["features"]}


@pytest.fixture
def checkpoint_actor_factory(monkeypatch):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.scheduler import Worker
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    if dist.is_initialized():
        pytest.skip("CPU checkpoint contract owns its single-rank process group")
    monkeypatch.setenv("USE_LIBUV", "0")
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            synchronize=lambda: None,
            ipc_collect=lambda: None,
            empty_cache=lambda: None,
            is_available=lambda: False,
        ),
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )

    def factory(optimizer_type, *, model_type="xvla", warm=False, updates=0):
        worker = object.__new__(EmbodiedFSDPActor)
        worker._logger = worker.logger
        worker._stacklevel = 1
        worker._cfg = fsdp_actor_config()
        worker._cfg.model.model_type = model_type
        worker._cfg.fsdp_config.save_full_model_weights = False
        worker.cfg = OmegaConf.create({"actor": worker._cfg})
        worker.model = FullyShardedDataParallel(
            nn.Linear(2, 1),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=torch.device("cpu"),
        )
        worker.optimizer = optimizer_type(worker.model.parameters(), lr=0.01)
        worker.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            worker.optimizer, 1, gamma=0.5
        )
        worker.optimizer_steps = updates
        worker.version = 7
        worker.is_weight_offloaded = False
        worker.is_optimizer_offloaded = False
        worker._strategy = FSDPStrategy(worker._cfg, 1, dist.group.WORLD)
        if warm:
            worker.model(torch.ones(1, 2)).sum().backward()
            worker.optimizer.step()
            worker.optimizer.zero_grad()
            worker.lr_scheduler.step()
        return worker

    try:
        yield factory
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("runner_created_directory", [False, True])
def test_normal_xvla_actor_zero_update_checkpoint_preserves_empty_adam(
    checkpoint_actor_factory,
    tmp_path,
    caplog,
    optimizer_type,
    runner_created_directory,
):
    worker = checkpoint_actor_factory(optimizer_type)
    checkpoint = tmp_path / "actor"
    if runner_created_directory:
        checkpoint.mkdir()  # EmbodiedRunner creates this before its actor RPC.
    model_before = copy.deepcopy(worker.model.state_dict())
    optimizer_before = copy.deepcopy(worker.optimizer.state_dict())
    scheduler_before = copy.deepcopy(worker.lr_scheduler.state_dict())

    worker.save_checkpoint(str(checkpoint), step=3)

    torch.testing.assert_close(worker.model.state_dict(), model_before, rtol=0, atol=0)
    torch.testing.assert_close(
        worker.optimizer.state_dict(), optimizer_before, rtol=0, atol=0
    )
    assert not worker.optimizer.state
    assert all(parameter.grad is None for parameter in worker.model.parameters())
    assert worker.lr_scheduler.state_dict() == scheduler_before
    assert worker.optimizer_steps == 0
    assert worker.version == 7
    assert "ZERO_UPDATE_EMPTY_ADAM_STATE" in caplog.text
    assert not checkpoint.exists() or not list(checkpoint.iterdir())


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
def test_normal_xvla_actor_rejects_missing_trained_adam_state(
    checkpoint_actor_factory, tmp_path, optimizer_type
):
    worker = checkpoint_actor_factory(optimizer_type, updates=1)
    with pytest.raises(RuntimeError, match="trained X-VLA.*empty Adam"):
        worker.save_checkpoint(str(tmp_path / "actor"), step=3)
    assert not worker.optimizer.state
    assert worker.optimizer_steps == 1
    assert not (tmp_path / "actor").exists()


@pytest.mark.parametrize(
    "model_type,optimizer_type,warm,updates",
    [
        ("xvla", torch.optim.Adam, True, 1),
        # A loaded warm optimizer is safe even if the manager counter starts at 0.
        ("xvla", torch.optim.AdamW, True, 0),
        ("mlp_policy", torch.optim.Adam, False, 0),
        ("xvla", torch.optim.SGD, False, 0),
    ],
)
def test_normal_actor_checkpoint_keeps_native_dcp_for_supported_states(
    checkpoint_actor_factory, tmp_path, model_type, optimizer_type, warm, updates
):
    worker = checkpoint_actor_factory(
        optimizer_type, model_type=model_type, warm=warm, updates=updates
    )
    model_before = copy.deepcopy(worker.model.state_dict())
    optimizer_before = copy.deepcopy(worker.optimizer.state_dict())
    scheduler_before = copy.deepcopy(worker.lr_scheduler.state_dict())
    checkpoint = tmp_path / "actor"

    worker.save_checkpoint(str(checkpoint), step=3)

    assert (checkpoint / "dcp_checkpoint" / ".metadata").is_file()
    torch.testing.assert_close(worker.model.state_dict(), model_before, rtol=0, atol=0)
    assert worker.lr_scheduler.state_dict() == scheduler_before
    assert worker.optimizer_steps == updates
    if warm or optimizer_type is torch.optim.SGD:
        torch.testing.assert_close(
            worker.optimizer.state_dict(), optimizer_before, rtol=0, atol=0
        )
    else:
        # Other models retain upstream DCP's empty-Adam initialization behavior.
        assert all(state["step"] == 1 for state in worker.optimizer.state.values())


def _actor_valid_count_process(rank, init_method, output):
    import torch.distributed as dist

    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        worker = object.__new__(EmbodiedFSDPActor)
        worker.device = torch.device("cpu")
        worker._dp_group = dist.new_group([0, 1], backend="gloo")
        counts = []
        for case in range(3):
            batch = {"prev_logprobs": torch.zeros(3)}
            if case < 2 or rank == 1:
                batch["loss_mask"] = torch.tensor([case == 1 and rank == 1])
            counts.append(worker._global_valid_sample_count(batch))
        output.put((rank, counts))
    finally:
        dist.destroy_process_group()


def test_embodied_actor_empty_batch_preserves_adam_momentum_and_scheduler(monkeypatch):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.scheduler import Worker
    from rlinf.utils.utils import get_loss_agg_func
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    if dist.is_initialized():
        pytest.skip("CPU actor contract owns its process group")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("USE_LIBUV", "0")
    cpu_platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"),
        empty_cache=lambda: None,
        synchronize=lambda: None,
        ipc_collect=lambda: None,
    )
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(Worker, "torch_platform", cpu_platform)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    native_all_reduce = dist.all_reduce

    def cpu_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        # Torch 2.6 Gloo lacks AVG; SUM is exact for this single-rank CPU boundary.
        if (
            op == dist.ReduceOp.AVG
            and dist.get_backend(group) == "gloo"
            and dist.get_world_size(group) == 1
        ):
            op = dist.ReduceOp.SUM
        return native_all_reduce(tensor, op=op, group=group, async_op=async_op)

    monkeypatch.setattr(dist, "all_reduce", cpu_all_reduce)
    try:
        worker = object.__new__(EmbodiedFSDPActor)
        worker.cfg = OmegaConf.create(
            {
                "runner": {"task_type": "embodied"},
                "actor": {
                    "seed": 1234,
                    "global_batch_size": 2,
                    "micro_batch_size": 1,
                    "model": {"model_type": "mlp_policy", "action_dim": 1},
                    "optim": {"clip_grad": 1.0},
                    "fsdp_config": {"enable_gradient_accumulation": True},
                },
                "algorithm": {
                    "loss_type": "actor",
                    "adv_type": "grpo",
                    "reward_type": "chunk_level",
                    "logprob_type": "chunk_level",
                    "entropy_bonus": 0.0,
                    "clip_ratio_high": 0.2,
                    "clip_ratio_low": 0.2,
                },
                "env": {"train": {"max_episode_steps": 1}},
            }
        )
        worker._cfg = worker.cfg.actor
        worker._world_size = 1
        worker._rank = 0
        worker._dp_group = dist.group.WORLD
        worker.device = torch.device("cpu")
        worker.torch_platform = cpu_platform
        worker.is_weight_offloaded = False
        worker.is_optimizer_offloaded = False
        worker.enable_sft_co_train = False
        worker.gradient_accumulation = 2
        worker.critic_warmup_steps = 0
        worker.optimizer_steps = 0
        worker.version = 7
        worker.amp_context = nullcontext()
        worker.loss_agg_func = get_loss_agg_func("token-mean")
        worker._strategy = FSDPStrategy(worker.cfg.actor, 1, worker._dp_group)
        worker.model = FullyShardedDataParallel(
            _DensityPolicy(),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=torch.device("cpu"),
        )
        worker.optimizer = torch.optim.AdamW(
            worker.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0
        )
        worker.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            worker.optimizer, 1, gamma=0.5
        )
        worker.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)

        def training_batch(valid):
            features = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
            return {
                "prev_logprobs": worker.model.module.weight.detach() * features,
                "forward_inputs": {"features": features},
                "advantages": torch.tensor([1.0, -1.0]).reshape(1, 2, 1),
                "loss_mask": torch.full((1, 2, 1), valid, dtype=torch.bool),
            }

        worker.rollout_batch = training_batch(True)
        initial_weight = worker.model.module.weight.detach().clone()
        inspect.unwrap(EmbodiedFSDPActor.run_training)(worker)
        assert worker.optimizer_steps == 1
        assert not torch.equal(worker.model.module.weight, initial_weight)
        assert any(
            state["exp_avg"].count_nonzero()
            for state in worker.optimizer.state.values()
        )
        parameters = {
            name: value.detach().clone()
            for name, value in worker.model.named_parameters()
        }
        optimizer_state = copy.deepcopy(worker.optimizer.state_dict())
        scheduler_state = copy.deepcopy(worker.lr_scheduler.state_dict())
        worker.rollout_batch = training_batch(False)
        metrics = inspect.unwrap(EmbodiedFSDPActor.run_training)(worker)
        assert metrics["actor/skipped_no_valid_samples"] == 1
        assert metrics["actor/valid_samples"] == 0
        assert worker.optimizer_steps == 1
        assert worker.version == 7
        torch.testing.assert_close(
            worker.optimizer.state_dict(), optimizer_state, rtol=0, atol=0
        )
        assert worker.lr_scheduler.state_dict() == scheduler_state
        for name, parameter in worker.model.named_parameters():
            torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_embodied_actor_valid_sample_count_reduces_across_cpu_ranks(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setenv("USE_LIBUV", "0")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_actor_valid_count_process,
            args=(rank, f"tcp://127.0.0.1:{port}?use_libuv=0", output),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(40)
            assert not process.is_alive(), (
                "Actor sample-count collective did not complete"
            )
            assert process.exitcode == 0
        assert sorted(output.get(timeout=5) for _ in processes) == [
            (0, [0, 1, 3]),
            (1, [0, 1, 3]),
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        output.close()


class _MacroDensityPolicy(nn.Module):
    """Independent per-trajectory score/value derivatives for loss-unit checks."""

    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.zeros(2))
        self.value = nn.Parameter(torch.zeros(2))

    def forward(self, forward_inputs, **kwargs):
        ids = forward_inputs["trajectory_ids"].flatten()
        return {
            "logprobs": pack_joint_logprob(self.score[ids], 30),
            "values": self.value[ids],
        }


def partial_chunk_actor_batch(
    anchors, maximum=300, *, algorithm="grpo", model_type="xvla", success=True
):
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {
                "model": {
                    "model_type": model_type,
                    "num_action_chunks": 30,
                    "action_dim": 20,
                    "logprob_action_dim": 1,
                },
                "fsdp_config": {"enable_gradient_accumulation": False},
            },
            "algorithm": {
                "adv_type": "gae" if algorithm == "ppo" else "grpo",
                "loss_type": "actor_critic" if algorithm == "ppo" else "actor",
                "reward_type": "chunk_level",
                "logprob_type": "chunk_level",
                "group_size": 2,
                "filter_rewards": algorithm == "grpo",
                "rewards_lower_bound": 0.01,
                "rewards_upper_bound": 0.99,
                "entropy_bonus": 0.0,
                "clip_ratio_high": 0.2,
                "clip_ratio_low": 0.2,
                "value_clip": 0.2,
                "huber_delta": 10.0,
            },
            "env": {
                "train": {
                    "max_episode_steps": maximum,
                    "rollout_epoch": 1,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
        }
    )
    chunks = (maximum + 29) // 30
    timeline = torch.arange(1, chunks * 30 + 1).reshape(chunks, 30)
    dones = torch.zeros(chunks + 1, 2, 30, dtype=torch.bool)
    dones[1:, 0] = timeline >= anchors
    dones[1:, 1] = timeline >= maximum
    rewards = torch.zeros(chunks, 2, 30)
    if success:
        rewards[(anchors - 1) // 30, 0, (anchors - 1) % 30] = 1
    batch = {
        "dones": dones,
        "rewards": rewards,
        "prev_logprobs": torch.zeros(chunks, 2, 30, 1),
        "prev_values": torch.zeros(chunks + 1, 2, 1),
        "forward_inputs": {
            "trajectory_ids": torch.arange(2)[None, :, None].expand(chunks, -1, -1)
        },
    }
    return worker, worker._process_received_rollout_batch(batch)


@pytest.mark.parametrize("anchors", [1, 30, 31, 109, 300])
@pytest.mark.parametrize("maximum", [300, 400])
@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
def test_partial_terminal_chunks_use_macro_units_through_real_actor_loss(
    monkeypatch, anchors, maximum, algorithm
):
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.scheduler import Worker
    from rlinf.utils.nested_dict_process import process_nested_dict_for_train
    from rlinf.utils.utils import get_loss_agg_func

    worker, batch = partial_chunk_actor_batch(anchors, maximum, algorithm=algorithm)
    chunks = (maximum + 29) // 30
    counts = batch["loss_mask_sum"][0, :, 0]
    torch.testing.assert_close(counts, torch.tensor([(anchors + 29) // 30, chunks]))
    assert worker._loss_max_episode_steps() == chunks
    # Equal-magnitude score advantages and critic errors isolate normalization units.
    batch["advantages"] = torch.tensor([1.0, -1.0])[None, :, None].expand(
        chunks, -1, -1
    )
    batch["returns"] = torch.ones(chunks, 2, 1)
    minibatch = process_nested_dict_for_train(batch, torch.arange(chunks * 2))
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )
    worker.device = torch.device("cpu")
    worker.model = _MacroDensityPolicy()
    worker._strategy = FSDPStrategy(worker.cfg.actor, 1)
    worker.optimizer_steps = 1
    worker.critic_warmup_steps = 0
    worker.enable_sft_co_train = False
    worker.amp_context = nullcontext()
    worker.gradient_accumulation = 1
    worker.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
    worker.loss_agg_func = get_loss_agg_func("token-mean")
    metrics = {}
    worker.train_micro_batch(minibatch, metrics, is_last=True)
    torch.testing.assert_close(worker.model.score.grad, torch.tensor([-0.5, 0.5]))
    if algorithm == "ppo":
        assert worker.model.value.grad.abs().min() > 0
        torch.testing.assert_close(
            worker.model.value.grad[0], worker.model.value.grad[1]
        )


def test_other_models_keep_anchor_normalization_and_tied_grpo_still_filters():
    worker, batch = partial_chunk_actor_batch(31, model_type="openpi")
    torch.testing.assert_close(batch["loss_mask_sum"][0, :, 0], torch.tensor([31, 300]))
    assert worker._loss_max_episode_steps() == 300
    worker, batch = partial_chunk_actor_batch(31, success=False)
    assert not batch["loss_mask"].any()
    torch.testing.assert_close(batch["loss_mask_sum"][0, :, 0], torch.tensor([2, 10]))
