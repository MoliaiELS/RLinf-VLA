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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.registry import policy_loss
from rlinf.models.embodiment.xvla.policy import DomainEmbedding, XVLAPolicy


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
        for config_name in ("robotwin_ppo_xvla", "robotwin_grpo_xvla"):
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
