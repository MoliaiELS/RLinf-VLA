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

"""Acceptance must distinguish real effective updates from no-signal runs."""

import json
import random

import numpy as np
import pytest
import torch

from tools.xvla.online_acceptance import (
    fixed_probe,
    parameter_groups,
    real_batch_evidence,
    record,
    require_effective_update,
    require_informative_grpo,
    require_matching_states,
    state_digest,
    summarize_evidence,
)


def batch_with_returns(returns):
    rewards = torch.zeros(2, len(returns), 3)
    rewards[0, :, 0] = torch.tensor(returns)
    mask = torch.ones_like(rewards, dtype=torch.bool)
    mask[1, :, 2] = False
    return {
        "rewards": rewards,
        "forward_inputs": {"executed_mask": mask},
        "advantages": torch.tensor(returns)[None, :, None].expand(2, -1, 1),
        "loss_mask": torch.ones(2, len(returns), 1, dtype=torch.bool),
    }


def test_state_digest_covers_frozen_buffers_and_storage_dtype():
    state = {"weight": torch.arange(3.0), "frozen": torch.ones(2, dtype=torch.bfloat16)}
    digest = state_digest(state)
    assert digest == state_digest(
        {f"_fsdp_wrapped_module.{key}": value for key, value in state.items()}
    )
    assert digest != state_digest({**state, "frozen": state["frozen"].float()})
    state["frozen"][0] += 1
    assert digest != state_digest(state)


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"weight": None},
        {"a": torch.ones(1), "_fsdp_wrapped_module.a": torch.ones(1)},
    ],
)
def test_state_digest_rejects_incomplete_or_ambiguous_states(state):
    with pytest.raises(ValueError):
        state_digest(state)


def test_native_batch_counts_only_executed_actions_and_records_real_groups():
    batch = batch_with_returns([1.0, 0.0, 1.0, 1.0])
    evidence = real_batch_evidence(batch, "grpo", 2)
    assert evidence["real_environment_steps"] == 20
    assert evidence["per_trajectory_control_steps"] == [5, 5, 5, 5]
    assert evidence["trajectory_returns"] == [1.0, 0.0, 1.0, 1.0]
    assert evidence["groups"]["variance"] == [0.25, 0.0]
    assert evidence["groups"]["informative_fraction"] == 0.5
    assert evidence["groups"]["all_success_fraction"] == 0.5
    assert evidence["groups"]["retained_fraction"] == 1.0
    assert evidence["advantages"]["nonzero"] == 6
    require_informative_grpo(evidence)


def test_native_top_level_mask_is_supported_and_conflicts_fail():
    batch = batch_with_returns([0.0, 0.0])
    batch["executed_mask"] = batch.pop("forward_inputs")["executed_mask"]
    assert real_batch_evidence(batch, "gae", 1)["real_environment_steps"] == 10
    batch["forward_inputs"] = {"executed_mask": ~batch["executed_mask"]}
    with pytest.raises(RuntimeError, match="masks disagree"):
        real_batch_evidence(batch, "gae", 1)


@pytest.mark.parametrize("returns", [[0.0] * 4, [1.0] * 4])
def test_tied_real_grpo_rewards_cannot_pass_update_acceptance(returns):
    evidence = real_batch_evidence(batch_with_returns(returns), "grpo", 4)
    with pytest.raises(RuntimeError, match="NO_INFORMATIVE_GROUP"):
        require_informative_grpo(evidence)


def test_filtered_informative_grpo_cannot_pass():
    batch = batch_with_returns([1.0, 0.0])
    batch["loss_mask"].zero_()
    with pytest.raises(RuntimeError, match="NO_RETAINED_GROUP"):
        require_informative_grpo(real_batch_evidence(batch, "grpo", 2))


def test_zero_reward_ppo_is_recorded_without_success_gate():
    batch = batch_with_returns([0.0, 0.0])
    batch["advantages"] = torch.tensor([[[1.0], [-1.0]], [[0.5], [-0.5]]])
    evidence = real_batch_evidence(batch, "gae", 1)
    assert evidence["algorithm"] == "ppo"
    assert evidence["advantage_estimator"] == "gae"
    assert evidence["successful_trajectories"] == 0
    assert evidence["advantages"]["nonzero"] == 4


@pytest.mark.parametrize(
    "invalid", ["missing_mask", "unexecuted_reward", "no_actions", "nonfinite"]
)
def test_invalid_native_rollout_evidence_fails(invalid):
    batch = batch_with_returns([1.0, 0.0])
    if invalid == "missing_mask":
        batch.pop("forward_inputs")
    elif invalid == "unexecuted_reward":
        batch["rewards"][1, 0, 2] = 1
    elif invalid == "no_actions":
        batch["forward_inputs"]["executed_mask"].zero_()
    else:
        batch["rewards"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        real_batch_evidence(batch, "grpo", 2)


def test_real_optimizer_update_requires_actor_gradient_and_parameter_change():
    actor = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    noise = torch.nn.Parameter(torch.tensor([0.5]))
    critic = torch.nn.Parameter(torch.tensor([0.1]))
    parameters = {
        "_fsdp_wrapped_module.core.transformer.head": actor,
        "_fsdp_wrapped_module.sampler.noise_head.weight": noise,
        "_fsdp_wrapped_module.value_head.weight": critic,
    }
    optimizer = torch.optim.SGD(parameters.values(), lr=0.1)
    (actor.square().sum() + noise.square().sum() + critic.square().sum()).backward()
    before = {name: value.detach().clone() for name, value in parameters.items()}
    gradients = {name: float(value.grad.norm()) for name, value in parameters.items()}
    optimizer.step()
    changes = {
        name: float((value.detach() - before[name]).abs().max())
        for name, value in parameters.items()
    }
    groups = parameter_groups(parameters)
    require_effective_update(
        gradients, changes, groups, has_noise_head=True, has_critic=True
    )
    gradients[groups["actor"][0]] = 0.0
    with pytest.raises(RuntimeError, match="NO_EFFECTIVE_ACTOR_UPDATE"):
        require_effective_update(
            gradients, changes, groups, has_noise_head=True, has_critic=True
        )


@pytest.mark.parametrize(
    "gradient,change", [(None, 1.0), (1.0, 0.0), (float("nan"), 1.0)]
)
def test_missing_gradient_or_no_parameter_change_is_not_an_effective_update(
    gradient, change
):
    parameters = {"core.transformer.head": torch.nn.Parameter(torch.ones(1))}
    with pytest.raises(
        RuntimeError, match="NONFINITE_UPDATE|NO_EFFECTIVE_ACTOR_UPDATE"
    ):
        require_effective_update(
            {"core.transformer.head": gradient},
            {"core.transformer.head": change},
            parameter_groups(parameters),
            has_noise_head=False,
            has_critic=False,
        )


def test_sync_requires_exact_state_and_consistent_probe():
    state = {"digest": "whole-model", "probe": {"actions": [[1.0]], "values": [[0.1]]}}
    require_matching_states(state, state)
    with pytest.raises(RuntimeError, match="full model states differ"):
        require_matching_states(state, {**state, "digest": "different-buffer"})
    with pytest.raises(AssertionError):
        require_matching_states(
            state, {**state, "probe": {"actions": [[2.0]], "values": [[0.1]]}}
        )


class ProbePolicy(torch.nn.Module):
    """External inference boundary that consumes random values during encoding."""

    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(1, 1)
        self.add_value_head = True

    def predict_action_batch(self, observation, *, mode, initial_noise, compute_values):
        assert mode == "eval" and compute_values
        assert observation["main_images"].shape == (1, 240, 320, 3)
        random.random()
        np.random.rand()
        torch.rand(3)
        return initial_noise.numpy(), {"prev_values": torch.zeros(1, 1)}


def test_fixed_consistency_probe_preserves_training_rng():
    policy = ProbePolicy()
    python_state, numpy_state, tensor_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    first = fixed_probe(policy)
    second = fixed_probe(policy)
    assert first == second
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_equal(current_numpy[1:], numpy_state[1:])
    assert torch.equal(torch.get_rng_state(), tensor_state)


def test_environment_audit_tracks_actual_rlinf_step_counter():
    pytest.importorskip("gymnasium")
    from tools.xvla.online_acceptance import _worker_classes

    _, _, env_type, _ = _worker_classes()
    worker = object.__new__(env_type)
    worker.set_global_step(7)
    assert worker.audit_global_step == 7
    worker.set_global_step(8)
    assert worker.audit_global_step == 8


def complete_events():
    return [
        {
            "event": "training_batch",
            "real_environment_steps": 20,
            "trajectory_returns": [0.0, 1.0],
        },
        {"event": "optimizer_update", "status": "PASS"},
        {"event": "weight_sync", "status": "PASS", "global_step": 1},
        {
            "event": "environment_chunk",
            "mode": "eval",
            "executed_actions": 9,
            "global_step": 1,
        },
        {"event": "evaluation_completed", "global_step": 1},
        {"event": "checkpoint_reload", "status": "PASS", "path": "/native/actor"},
    ]


def test_acceptance_summary_requires_real_post_update_eval_and_native_reload(tmp_path):
    events = complete_events()
    path = tmp_path / "audit-1.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    summary = summarize_evidence(tmp_path)
    assert summary["status"] == "PASS"
    assert summary["real_training_environment_steps"] == 20
    assert summary["real_evaluation_environment_steps"] == 9
    assert "Single rank NO_SHARD acceptance" in summary["limitations"]
    events[2]["global_step"] = 0
    path.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(RuntimeError, match="updated weights were not synchronized"):
        summarize_evidence(tmp_path)


@pytest.mark.parametrize(
    "missing",
    [
        "optimizer_update",
        "environment_chunk",
        "evaluation_completed",
        "checkpoint_reload",
    ],
)
def test_synthetic_probe_alone_cannot_satisfy_online_acceptance(tmp_path, missing):
    events = [event for event in complete_events() if event["event"] != missing]
    (tmp_path / "audit-1.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    with pytest.raises(RuntimeError, match="INCOMPLETE_ONLINE_ACCEPTANCE"):
        summarize_evidence(tmp_path)


def test_audit_records_native_tensors_and_numpy_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XVLA_RUN_DIR", str(tmp_path))
    record(
        "native_metrics",
        values={"physics_ticks": torch.tensor([3]), "success": np.float32(0)},
    )
    path = next(tmp_path.glob("audit-*.jsonl"))
    item = json.loads(path.read_text())
    assert item["values"] == {"physics_ticks": [3], "success": 0.0}
