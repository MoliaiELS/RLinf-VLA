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

"""Analytic density and gradient checks for the X-VLA stochastic flow policy."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

# This mathematical component needs only Torch, including in environments
# without the scheduler, simulators, or vendor model dependencies.
_SOURCE = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/xvla/sampling.py"
)
_SPEC = importlib.util.spec_from_file_location("xvla_flow_sampling", _SOURCE)
_SAMPLING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAMPLING)
FlowSampler = _SAMPLING.FlowSampler
BoundedNoiseNet = _SAMPLING.BoundedNoiseNet


class AffineClean(nn.Module):
    def __init__(self, bias=0.2, slope=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))
        self.slope = slope

    def forward(self, x, time):
        del time
        return self.bias + self.slope * x


def _independent_logprob(value, mean, std):
    return (
        torch.distributions.Normal(mean.double(), std.double())
        .log_prob(value.double())
        .sum(dim=(-2, -1), dtype=torch.float64)
    )


def test_full_path_density_preserves_small_mean_changes_at_native_scales():
    sampler = FlowSampler(steps=10, action_dim=20, noise_std=[0.003] * 20)
    predictor = AffineClean(bias=0.2)
    z = torch.zeros(1, 30, 20)
    sampled = sampler.sample(
        predictor, z, torch.zeros(1, 20), generator=torch.Generator().manual_seed(17)
    )
    before = sampler.recompute(predictor, sampled["chains"], torch.zeros(1, 20))
    with torch.no_grad():
        predictor.bias.add_(1e-7)
    after = sampler.recompute(predictor, sampled["chains"], torch.zeros(1, 20))
    assert before["logprobs"].dtype == after["logprobs"].dtype == torch.float64
    assert 0 < (after["logprobs"] - before["logprobs"]).abs().item() < 0.01


def test_single_transition_density_and_clean_mean_score_match_analytic_gaussian():
    predictor = AffineClean()
    sampler = FlowSampler(steps=1, action_dim=2, noise_std=[0.3, 0.8])
    z = torch.tensor([[[0.1, -0.4], [0.7, 0.9]]])
    q = torch.tensor([[[0.5, -0.2], [0.3, 0.8]]])
    chain = torch.stack((z, q), dim=1)
    result = sampler.recompute(predictor, chain, torch.zeros(1, 2))
    std = torch.tensor([0.3, 0.8]).expand_as(q)
    expected = _independent_logprob(z, torch.zeros_like(z), torch.ones_like(z))
    expected = expected + _independent_logprob(q, torch.full_like(q, 0.2), std)
    torch.testing.assert_close(result["logprobs"], expected)
    result["logprobs"].sum().backward()
    expected_score = ((q - 0.2) / std.square()).sum()
    torch.testing.assert_close(predictor.bias.grad, expected_score)
    assert not chain.requires_grad


def test_learned_noise_score_and_entropy_have_correct_gradients():
    sampler = FlowSampler(steps=1, action_dim=1, noise_std=[0.4])
    predictor = AffineClean(bias=0.0)
    chain = torch.tensor([[[[0.1]], [[0.8]]]])
    result = sampler.recompute(predictor, chain, torch.zeros(1, 1))
    last_bias = sampler.noise_head.net[-1].bias
    density_grad = torch.autograd.grad(
        result["logprobs"].sum(), last_bias, retain_graph=True
    )[0]
    entropy_grad = torch.autograd.grad(result["entropy"].sum(), last_bias)[0]
    fraction = (1.0 - 0.1) / (2.0 - 0.1)
    d_log_std = (2.0 - 0.1) * fraction * (1.0 - fraction)
    expected_density_grad = ((0.8 / 0.4) ** 2 - 1.0) * d_log_std
    torch.testing.assert_close(density_grad, torch.tensor([expected_density_grad]))
    torch.testing.assert_close(entropy_grad, torch.tensor([d_log_std]))
    assert density_grad.abs().item() > 0
    assert entropy_grad.item() > 0


@pytest.mark.parametrize("method", ["flow_noise", "flow_sde"])
def test_sample_and_cached_recompute_have_unit_ratio_and_finite_actor_gradient(method):
    sampler = FlowSampler(method=method, steps=4, action_dim=20, noise_std=[0.2] * 20)
    predictor = AffineClean(slope=0.3)
    z = torch.linspace(-0.7, 0.9, 120).reshape(2, 3, 20)
    proprio = torch.zeros(2, 20)
    sampled = sampler.sample(
        predictor, z, proprio, generator=torch.Generator().manual_seed(42)
    )
    assert sampled["chains"].shape == (2, 5, 3, 20)
    assert not sampled["chains"].requires_grad
    assert sampled["logprobs"].shape == (2,)
    recomputed = sampler.recompute(predictor, sampled["chains"], proprio)
    torch.testing.assert_close(
        recomputed["logprobs"], sampled["logprobs"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        (recomputed["logprobs"] - sampled["logprobs"]).exp(),
        torch.ones_like(recomputed["logprobs"]),
    )
    recomputed["logprobs"].sum().backward()
    assert torch.isfinite(predictor.bias.grad)
    assert predictor.bias.grad.abs().item() > 0
    with torch.no_grad():
        predictor.bias.add_(0.04)
    updated = sampler.recompute(predictor, sampled["chains"], proprio)
    assert not torch.allclose(updated["logprobs"], sampled["logprobs"])


def test_multistep_score_conditions_on_cached_states_without_reparameterization():
    predictor = AffineClean(slope=0.25)
    sampler = FlowSampler(steps=2, action_dim=1, noise_std=[0.5])
    chain = torch.tensor([[[[0.1], [0.3]], [[0.7], [-0.4]], [[0.6], [0.9]]]])
    stats = sampler.recompute(predictor, chain, torch.zeros(1, 1))
    stats["logprobs"].sum().backward()
    expected_score = 0.0
    for index in range(2):
        t = (2 - index) / 2
        incoming = chain[:, index]
        mean = incoming - 0.5 * (incoming - (0.2 + 0.25 * incoming)) / t
        expected_score += ((chain[:, index + 1] - mean) / 0.5**2 * (0.5 / t)).sum()
    torch.testing.assert_close(predictor.bias.grad, expected_score)


def test_joint_probability_includes_all_denoising_steps_coordinates_and_suffixes():
    sampler = FlowSampler(steps=3, action_dim=20, noise_std=[0.6] * 20)
    predictor = AffineClean(slope=0.15)
    chain = torch.linspace(-0.8, 0.9, 240).reshape(1, 4, 3, 20)
    actual = sampler.recompute(predictor, chain, torch.zeros(1, 20))["logprobs"]
    expected = _independent_logprob(
        chain[:, 0], torch.zeros_like(chain[:, 0]), torch.ones_like(chain[:, 0])
    )
    for index in range(3):
        t = (3 - index) / 3
        incoming = chain[:, index]
        mean = incoming - (incoming - (0.2 + 0.15 * incoming)) / (3 * t)
        expected += _independent_logprob(
            chain[:, index + 1], mean, torch.full_like(incoming, 0.6)
        )
    torch.testing.assert_close(actual, expected)
    changed = chain.clone()
    changed[:, -1, -1, -1] += 0.7
    changed_density = sampler.recompute(predictor, changed, torch.zeros(1, 20))[
        "logprobs"
    ]
    assert not torch.allclose(actual, changed_density)


@pytest.mark.parametrize("steps", [2, 10, 100])
def test_flow_sde_boundary_and_last_transition_match_independent_drift(steps):
    predictor = AffineClean(slope=0.2)
    sampler = FlowSampler(method="flow_sde", steps=steps, action_dim=1)
    x = torch.tensor([[[0.3], [-0.4]]])
    for index in (0, steps - 1):
        mean, std = sampler._transition(
            predictor, x, torch.zeros(1, 1), index, context=None, deterministic=False
        )
        t = (steps - index) / steps
        dt = 1 / steps
        clean = 0.2 + 0.2 * x
        velocity = (x - clean) / t
        denom_t = t - dt if index == 0 else t
        sigma_squared = 0.5**2 * t / (1 - denom_t)
        expected_mean = (
            x - dt * velocity - sigma_squared * dt / (2 * t) * (x + (1 - t) * velocity)
        )
        expected_std = torch.full_like(x, math.sqrt(dt * sigma_squared))
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(std, expected_std)
        assert bool(torch.isfinite(mean).all())
        assert bool(torch.isfinite(std).all())
        assert bool((std > 0).all())


def test_noise_head_starts_at_coordinate_scales_and_stays_bounded():
    head = BoundedNoiseNet([0.003, 0.15])
    x = torch.zeros(2, 3, 2)
    proprio = torch.zeros(2, 2)
    t = torch.tensor([1.0, 0.1])
    initial = head(x, proprio, t)
    torch.testing.assert_close(initial, torch.tensor([0.003, 0.15]).expand_as(x))
    for bias in (-100.0, 100.0):
        with torch.no_grad():
            head.net[-1].bias.fill_(bias)
        std = head(x, proprio, t)
        assert bool(torch.isfinite(std).all())
        assert bool((std >= torch.tensor([0.003, 0.15]) * 0.1).all())
        assert bool((std <= torch.tensor([0.003, 0.15]) * 2.0).all())


def test_observation_context_reaches_trainable_noise_network():
    head = BoundedNoiseNet([0.3], context_dim=2)
    with torch.no_grad():
        head.net[-1].weight.fill_(0.2)
    x = torch.zeros(1, 2, 1)
    proprio = torch.zeros(1, 1)
    t = torch.ones(1)
    context = torch.tensor([[0.2, -0.5]], requires_grad=True)
    head(x, proprio, t, context).sum().backward()
    assert bool(torch.isfinite(context.grad).all())
    assert bool((context.grad != 0).any())


def test_deterministic_euler_evaluation_returns_no_training_density():
    predictor = AffineClean(bias=0.4)
    z = torch.tensor([[[0.1], [-0.5]]])
    proprio = torch.zeros(1, 1)
    sampler = FlowSampler(method="flow_noise", steps=4, action_dim=1)
    result = sampler.sample(predictor, z, proprio, deterministic=True)
    torch.testing.assert_close(result["actions"], torch.full_like(z, 0.4))
    assert result["logprobs"].item() == 0
    assert result["entropy"].item() == 0
    ode_sampler = FlowSampler(method="flow_ode", steps=4, action_dim=1)
    with pytest.raises(ValueError, match="no Gaussian training density"):
        ode_sampler.recompute(predictor, result["chains"], proprio)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": True},
        {"method": "terminal_clean_gaussian_v1"},
        {"method": "flow_sde", "steps": 1},
        {"noise_std": [0.0]},
        {"noise_std": [math.nan]},
        {"noise_std": [0.2, 0.3]},
        {"noise_min_ratio": 0.0},
        {"sde_noise_level": 0.0},
    ],
)
def test_invalid_sampler_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FlowSampler(action_dim=1, **kwargs)


def test_recompute_rejects_reparameterized_or_truncated_cached_chains():
    sampler = FlowSampler(steps=2, action_dim=1)
    predictor = AffineClean()
    proprio = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="must be detached"):
        sampler.recompute(
            predictor, torch.zeros(1, 3, 2, 1, requires_grad=True), proprio
        )
    with pytest.raises(ValueError, match="cached chain shape"):
        sampler.recompute(predictor, torch.zeros(1, 2, 2, 1), proprio)
