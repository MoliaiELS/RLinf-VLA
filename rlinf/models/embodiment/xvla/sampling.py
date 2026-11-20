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

"""Full-path stochastic flow policies for X-VLA clean-action predictions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

CleanPredictor = Callable[[Tensor, Tensor], Tensor]


def _normal_logprob(sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (
        -0.5 * ((sample - mean) / std).square()
        - std.log()
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=(-2, -1), dtype=torch.float64)


def _normal_entropy(std: Tensor) -> Tensor:
    return (std.log() + 0.5 * math.log(2.0 * math.pi * math.e)).sum(
        dim=(-2, -1), dtype=torch.float64
    )


class BoundedNoiseNet(nn.Module):
    """Predict positive per-coordinate transition standard deviations.

    The inputs are the current noisy action, measured proprioception, denoising
    time, and optional observation features. Coordinate scales preserve the
    units of position, rotation, and gripper logits.
    """

    def __init__(
        self,
        noise_std: Sequence[float],
        *,
        noise_min_ratio: float = 0.1,
        noise_max_ratio: float = 2.0,
        hidden_dim: int = 128,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if not (0.0 < noise_min_ratio < 1.0 < noise_max_ratio):
            raise ValueError("Expected 0 < noise_min_ratio < 1 < noise_max_ratio")
        scales = torch.as_tensor(noise_std, dtype=torch.float32)
        if (
            scales.ndim != 1
            or scales.numel() == 0
            or not bool(torch.isfinite(scales).all())
            or not bool((scales > 0).all())
        ):
            raise ValueError("noise_std must contain finite positive coordinate scales")
        if hidden_dim < 1 or context_dim < 0:
            raise ValueError("Expected positive hidden_dim and nonnegative context_dim")
        self.action_dim = int(scales.numel())
        self.context_dim = context_dim
        self.noise_min_ratio = float(noise_min_ratio)
        self.noise_max_ratio = float(noise_max_ratio)
        self.register_buffer("noise_std", scales)
        self.net = nn.Sequential(
            nn.Linear(2 * self.action_dim + 1 + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        initial_fraction = (1.0 - noise_min_ratio) / (noise_max_ratio - noise_min_ratio)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(
            self.net[-1].bias, math.log(initial_fraction / (1.0 - initial_fraction))
        )

    def forward(
        self,
        noisy_action: Tensor,
        proprio: Tensor,
        time: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Return transition standard deviations with shape ``[B,H,D]``."""
        batch, horizon, _ = noisy_action.shape
        features = [
            noisy_action,
            proprio[:, None, :].expand(-1, horizon, -1),
            time[:, None, None].expand(-1, horizon, 1),
        ]
        if self.context_dim:
            if context is None or context.shape != (batch, self.context_dim):
                raise ValueError(
                    f"Expected observation context shape {(batch, self.context_dim)}"
                )
            features.append(context[:, None, :].expand(-1, horizon, -1))
        elif context is not None:
            raise ValueError("Expected no observation context when context_dim is zero")
        inputs = torch.cat(features, dim=-1).to(self.net[0].weight.dtype)
        fraction = self.net(inputs).float().sigmoid()
        ratio = (
            self.noise_min_ratio
            + (self.noise_max_ratio - self.noise_min_ratio) * fraction
        )
        return ratio * self.noise_std.float()


class FlowSampler(nn.Module):
    """Sample and score a complete stochastic denoising trajectory.

    X-VLA predicts clean actions for ``x_t = t*z + (1-t)*action``. The velocity
    for decreasing noise time is ``(x_t-clean)/t``. Flow-Noise adds a learned
    Gaussian transition to its Euler mean; Flow-SDE uses the corrected drift
    and noise schedule from the RLinf OpenPI sampler.

    The scored augmented action includes the initial noise and every sampled
    coordinate at every denoising step. Unused arms and prediction suffixes
    remain part of this joint policy. No execution mask or averaging belongs
    inside this density. A caller may mask the entire environment macro-step.
    """

    def __init__(
        self,
        *,
        method: str = "flow_noise",
        steps: int = 10,
        action_dim: int = 20,
        noise_std: Sequence[float] | None = None,
        noise_min_ratio: float = 0.1,
        noise_max_ratio: float = 2.0,
        sde_noise_level: float = 0.5,
        hidden_dim: int = 128,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if method not in {"flow_noise", "flow_sde", "flow_ode"}:
            raise ValueError(f"Unsupported X-VLA flow sampling method: {method}")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("Expected a positive integer denoising step count")
        if action_dim < 1:
            raise ValueError("Expected a positive action_dim")
        if method == "flow_sde" and steps < 2:
            raise ValueError("Flow-SDE requires at least two denoising steps")
        if not math.isfinite(sde_noise_level) or sde_noise_level <= 0:
            raise ValueError("Expected a finite positive sde_noise_level")
        self.method = method
        self.steps = steps
        self.action_dim = action_dim
        self.sde_noise_level = float(sde_noise_level)
        if noise_std is None:
            noise_std = [1.0] * action_dim
        if len(noise_std) != action_dim:
            raise ValueError(f"Expected {action_dim} noise_std coordinate scales")
        self.noise_head = (
            BoundedNoiseNet(
                noise_std,
                noise_min_ratio=noise_min_ratio,
                noise_max_ratio=noise_max_ratio,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
            )
            if method == "flow_noise"
            else None
        )

    def _validate_inputs(self, initial_noise: Tensor, proprio: Tensor) -> None:
        if (
            initial_noise.ndim != 3
            or initial_noise.shape[-1] != self.action_dim
            or initial_noise.shape[0] < 1
            or initial_noise.shape[1] < 1
        ):
            raise ValueError(f"Expected initial noise shape [B,H,{self.action_dim}]")
        if proprio.shape != (initial_noise.shape[0], self.action_dim):
            raise ValueError(
                f"Expected proprio shape {(initial_noise.shape[0], self.action_dim)}"
            )
        if initial_noise.device != proprio.device:
            raise ValueError("Expected initial noise and proprio on the same device")
        if not initial_noise.is_floating_point() or not proprio.is_floating_point():
            raise ValueError("Expected floating point initial noise and proprio")
        if initial_noise.requires_grad:
            raise ValueError("The cached denoising trajectory must be detached")
        if not bool(torch.isfinite(initial_noise).all()) or not bool(
            torch.isfinite(proprio).all()
        ):
            raise ValueError("Expected finite initial noise and proprio")

    def _transition(
        self,
        clean_predict: CleanPredictor,
        noisy_action: Tensor,
        proprio: Tensor,
        index: int,
        *,
        context: Tensor | None,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        time_value = (self.steps - index) / self.steps
        delta = 1.0 / self.steps
        next_time = (self.steps - index - 1) / self.steps
        time = torch.full(
            (noisy_action.shape[0],),
            time_value,
            device=noisy_action.device,
            dtype=torch.float32,
        )
        clean = clean_predict(noisy_action, time).float()
        if clean.shape != noisy_action.shape:
            raise ValueError(
                "Expected clean prediction to match the noisy action shape"
            )
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Expected finite clean-action predictions")
        velocity = (noisy_action - clean) / time_value
        if deterministic or self.method == "flow_ode":
            return noisy_action - delta * velocity, torch.zeros_like(noisy_action)
        if self.method == "flow_noise":
            mean = noisy_action - delta * velocity
            std = self.noise_head(noisy_action, proprio, time, context)
            return mean, std

        # The t=1 boundary uses the next grid time, matching OpenPI's finite
        # discretization of sigma(t)=level*sqrt(t/(1-t)).
        denominator_time = next_time if index == 0 else time_value
        sigma = self.sde_noise_level * math.sqrt(time_value / (1.0 - denominator_time))
        noise_prediction = noisy_action + (1.0 - time_value) * velocity
        noise_weight = next_time - sigma * sigma * delta / (2.0 * time_value)
        mean = clean * (1.0 - next_time) + noise_prediction * noise_weight
        std = torch.full_like(noisy_action, math.sqrt(delta) * sigma)
        return mean, std

    @torch.no_grad()
    def sample(
        self,
        clean_predict: CleanPredictor,
        initial_noise: Tensor,
        proprio: Tensor,
        *,
        context: Tensor | None = None,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        """Return actions, detached chains, and joint scores of shape ``[B]``.

        ``deterministic=True`` is the matched Euler deployment policy. Its zero
        scores are placeholders for evaluation and must never be used as a
        behavior-policy density for PPO or GRPO.
        """
        self._validate_inputs(initial_noise, proprio)
        noisy_action = initial_noise.float()
        chains = [noisy_action]
        deterministic = deterministic or self.method == "flow_ode"
        logprobs = _normal_logprob(
            noisy_action, torch.zeros_like(noisy_action), torch.ones_like(noisy_action)
        )
        entropy = _normal_entropy(torch.ones_like(noisy_action))
        for index in range(self.steps):
            mean, std = self._transition(
                clean_predict,
                noisy_action,
                proprio,
                index,
                context=context,
                deterministic=deterministic,
            )
            if deterministic:
                noisy_action = mean
            else:
                noise = torch.randn(
                    mean.shape,
                    device=mean.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                noisy_action = mean + std * noise
                logprobs = logprobs + _normal_logprob(noisy_action, mean, std)
                entropy = entropy + _normal_entropy(std)
            chains.append(noisy_action)
        if deterministic:
            logprobs = torch.zeros_like(logprobs)
            entropy = torch.zeros_like(entropy)
        return {
            "actions": noisy_action.detach().clone(),
            "chains": torch.stack(chains, dim=1).detach(),
            "logprobs": logprobs.detach(),
            "entropy": entropy.detach(),
        }

    def recompute(
        self,
        clean_predict: CleanPredictor,
        chains: Tensor,
        proprio: Tensor,
        *,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Score cached transitions with gradients and without resampling.

        The incoming and outgoing states in each transition are fixed samples;
        gradients flow through the new transition mean and noise head only.
        """
        if self.method == "flow_ode":
            raise ValueError("Deterministic Flow-ODE has no Gaussian training density")
        if chains.ndim != 4 or chains.shape[1] != self.steps + 1:
            raise ValueError(
                f"Expected cached chain shape [B,{self.steps + 1},H,{self.action_dim}]"
            )
        if chains.requires_grad:
            raise ValueError("The cached denoising trajectory must be detached")
        self._validate_inputs(chains[:, 0], proprio)
        if not bool(torch.isfinite(chains).all()):
            raise ValueError("Expected a finite cached denoising trajectory")
        chains = chains.float()
        logprobs = _normal_logprob(
            chains[:, 0], torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0])
        )
        entropy = _normal_entropy(torch.ones_like(chains[:, 0]))
        for index in range(self.steps):
            mean, std = self._transition(
                clean_predict,
                chains[:, index],
                proprio,
                index,
                context=context,
                deterministic=False,
            )
            logprobs = logprobs + _normal_logprob(chains[:, index + 1], mean, std)
            entropy = entropy + _normal_entropy(std)
        return {"logprobs": logprobs, "entropy": entropy}
