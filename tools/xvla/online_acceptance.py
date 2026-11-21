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

"""Audit real X-VLA training through RLinf's existing workers and runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _json_value(value: Any) -> Any:
    import numpy as np

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize acceptance value of type {type(value).__name__}")


def record(event: str, **payload: Any) -> dict:
    """Append one process-local acceptance event without altering training data."""
    folder = Path(os.environ["XVLA_RUN_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    item = {"event": event, "time": time.time(), "pid": os.getpid(), **payload}
    with (folder / f"audit-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, allow_nan=False, default=_json_value) + "\n")
        stream.flush()
    return item


def canonical_name(name: str) -> str:
    """Remove only FSDP's implementation prefix from a model state name."""
    return name.replace("_fsdp_wrapped_module.", "")


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash every parameter and buffer including name, shape and storage dtype."""
    digest = hashlib.sha256()
    names = set()
    for raw_name, value in sorted(state.items()):
        name = canonical_name(raw_name)
        if name in names or not torch.is_tensor(value):
            raise ValueError(
                "Expected unique tensor names in an ordinary NO_SHARD state"
            )
        names.add(name)
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        data = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(data).cast("B"))
    if not names:
        raise ValueError("Cannot accept an empty model state")
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    """Fingerprint nested optimizer, scheduler, RNG and reset-schedule state."""
    import numpy as np

    digest = hashlib.sha256()

    def feed(kind: str, payload: bytes | memoryview) -> None:
        digest.update(f"{kind}:{len(payload)}:".encode("ascii"))
        digest.update(payload)

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().contiguous().cpu()
            feed("tensor", str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
            feed(
                "data",
                memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B"),
            )
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            feed("array", str((array.shape, array.dtype.str)).encode("ascii"))
            feed("data", array.tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            feed("dict", str(len(item)).encode("ascii"))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            feed(type(item).__name__, str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int, float)):
            feed(type(item).__name__, repr(item).encode("utf-8"))
        else:
            raise TypeError(f"Unsupported training-state value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def process_identity() -> str:
    """Identify a worker process across separate Slurm launches."""
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("XVLA_LOGIN_RUN_ID", "")
    return f"{socket.gethostname()}:{run_id}:{os.getpid()}"


def capture_rng_state() -> dict:
    """Capture the native checkpoint RNG state for this worker."""
    from rlinf.utils.utils import get_rng_state

    return get_rng_state()


def restore_rng_state(state: dict) -> None:
    """Restore native CPU, Python, NumPy and worker-device RNG streams."""
    from rlinf.utils.utils import set_rng_state

    set_rng_state(state)


def training_fingerprints(
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
    scheduler_state: dict,
    rng_state: dict,
    optimizer_steps: int,
) -> dict:
    """Describe all actor state that must survive a checkpoint reload."""
    return {
        "model": state_digest(model_state),
        "optimizer": object_digest(optimizer_state),
        "scheduler": object_digest(scheduler_state),
        "rng": object_digest(rng_state),
        "optimizer_steps": int(optimizer_steps),
    }


def require_matching_training_states(before: dict, after: dict) -> None:
    """Reject any model, optimizer, scheduler, RNG or update-counter mismatch."""
    for name in ("model", "optimizer", "scheduler", "rng", "optimizer_steps"):
        if before[name] != after[name]:
            raise RuntimeError(f"TRAINING_STATE_MISMATCH: {name} changed")


def tensor_stats(value: torch.Tensor) -> dict:
    """Summarize finite real rollout or advantage tensors."""
    values = value.detach().float().cpu()
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise RuntimeError("Acceptance requires nonempty finite rollout statistics")
    return {
        "shape": list(values.shape),
        "sum": float(values.sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
        "nonzero": int(values.count_nonzero()),
    }


def real_batch_evidence(batch: dict, algorithm: str, group_size: int) -> dict:
    """Read sampled rewards and control counts before actor minibatch reshuffling."""
    estimator = algorithm
    algorithm = "ppo" if estimator == "gae" else estimator
    rewards = batch["rewards"].detach().float().cpu()
    if rewards.ndim != 3 or not bool(torch.isfinite(rewards).all()):
        raise RuntimeError("Expected finite real rewards [time,batch,anchor]")
    mask = batch.get("executed_mask")
    cached_mask = batch.get("forward_inputs", {}).get("executed_mask")
    if mask is None:
        mask = cached_mask
    elif cached_mask is not None and not torch.equal(mask, cached_mask):
        raise RuntimeError("Top-level and cached native execution masks disagree")
    if mask is None or mask.shape != rewards.shape:
        raise RuntimeError("Native X-VLA trajectories require an anchor execution mask")
    mask = mask.detach().bool().cpu()
    if not bool(mask.any()):
        raise RuntimeError(
            "NO_EXECUTED_ACTIONS: no real native control action was recorded"
        )
    if bool((rewards[~mask] != 0).any()):
        raise RuntimeError("Unexecuted native actions must have zero rewards")
    returns = rewards.sum(dim=(0, 2))
    evidence = {
        "algorithm": algorithm,
        "advantage_estimator": estimator,
        "real_environment_steps": int(mask.sum()),
        "per_trajectory_control_steps": mask.sum(dim=(0, 2)).tolist(),
        "trajectory_returns": returns.tolist(),
        "successful_trajectories": int((returns > 0).sum()),
        "trajectory_count": int(rewards.shape[1]),
        "rewards": tensor_stats(rewards),
    }
    for name in (
        "advantages",
        "returns",
        "dones",
        "terminations",
        "truncations",
        "loss_mask",
    ):
        if torch.is_tensor(batch.get(name)):
            evidence[name] = tensor_stats(batch[name])
    if algorithm == "grpo":
        if group_size < 2 or returns.numel() % group_size:
            raise RuntimeError("GRPO trajectories must form complete adjacent groups")
        grouped = returns.reshape(-1, group_size)
        variance = grouped.var(dim=1, unbiased=False)
        loss_mask = batch.get("loss_mask")
        retained = None
        if loss_mask is not None:
            retained_trajectories = loss_mask.detach().bool().cpu().any(dim=(0, 2))
            retained = retained_trajectories.reshape(-1, group_size).all(dim=1)
        evidence["groups"] = {
            "returns": grouped.tolist(),
            "variance": variance.tolist(),
            "informative_fraction": float((variance > 0).float().mean()),
            "retained_fraction": None
            if retained is None
            else float(retained.float().mean()),
            "all_success_fraction": float((grouped > 0).all(dim=1).float().mean()),
            "all_failure_fraction": float((grouped == 0).all(dim=1).float().mean()),
        }
    return evidence


def require_informative_grpo(evidence: dict) -> None:
    """Reject tied real-reward groups as an effective-GRPO acceptance result."""
    groups = evidence["groups"]
    if not any(value > 0 for value in groups["variance"]):
        raise RuntimeError("NO_INFORMATIVE_GROUP: real within-group returns all tie")
    if groups["retained_fraction"] == 0:
        raise RuntimeError("NO_RETAINED_GROUP: every real GRPO group was filtered")


def parameter_groups(
    named_parameters: dict[str, torch.nn.Parameter],
) -> dict[str, list[str]]:
    """Identify the actor, learned flow noise, and observation-only critic."""
    groups = {"actor": [], "noise": [], "critic": []}
    for raw_name in named_parameters:
        name = canonical_name(raw_name)
        if "core.transformer." in name:
            groups["actor"].append(raw_name)
        elif "sampler.noise_head." in name:
            groups["noise"].append(raw_name)
        elif "value_head." in name:
            groups["critic"].append(raw_name)
    return groups


def require_effective_update(
    gradients: dict[str, float | None],
    changes: dict[str, float],
    groups: dict[str, list[str]],
    *,
    has_noise_head: bool,
    has_critic: bool,
) -> None:
    """Require real finite score gradients and optimizer changes in each active head."""
    for value in (*gradients.values(), *changes.values()):
        if value is not None and not math.isfinite(value):
            raise RuntimeError(
                "NONFINITE_UPDATE: nonfinite gradient or parameter change"
            )
    required = ["actor"]
    if has_noise_head:
        required.append("noise")
    if has_critic:
        required.append("critic")
    for group in required:
        if not any(
            gradients.get(name) is not None
            and gradients[name] > 0
            and changes.get(name, 0) > 0
            for name in groups[group]
        ):
            raise RuntimeError(
                f"NO_EFFECTIVE_{group.upper()}_UPDATE: no gradient-backed parameter change"
            )


def unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    """Access the policy behind a single root FSDP wrapper."""
    while hasattr(model, "module"):
        model = model.module
    return model


@torch.no_grad()
def fixed_probe(policy: torch.nn.Module) -> dict:
    """Use a separate fixed diagnostic input only for synchronization consistency."""
    import numpy as np

    state = torch.tensor(
        [[0.3, 0.2, 0.2, 1, 0, 0, 1, 0, 0, -1, 0.3, -0.2, 0.2, 1, 0, 0, 1, 0, 0, -1]],
        dtype=torch.float32,
    )
    observation = {
        "main_images": torch.full((1, 240, 320, 3), 80, dtype=torch.uint8),
        "wrist_images": torch.full((1, 2, 240, 320, 3), 140, dtype=torch.uint8),
        "states": state,
        "task_descriptions": ["move stapler pad"],
    }
    device = next(policy.core.parameters()).device
    noise = torch.randn((1, 30, 20), generator=torch.Generator().manual_seed(717)).to(
        device
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(
            devices=[] if device.type == "cpu" else [device.index],
            device_type=device.type,
        ):
            actions, output = policy.predict_action_batch(
                observation,
                mode="eval",
                initial_noise=noise,
                compute_values=policy.add_value_head,
            )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return {"actions": actions.tolist(), "values": output["prev_values"].cpu().tolist()}


def require_matching_states(actor: dict, rollout: dict) -> None:
    """Check exact parameter/buffer synchronization and a fixed numerical probe."""
    if actor["digest"] != rollout["digest"]:
        raise RuntimeError(
            "Actor/rollout full model states differ after weight synchronization"
        )
    for name in ("actions", "values"):
        torch.testing.assert_close(
            torch.as_tensor(actor["probe"][name]),
            torch.as_tensor(rollout["probe"][name]),
            rtol=1e-5,
            atol=1e-6,
        )


def summarize_evidence(
    folder: Path, min_updates: int = 3, require_resume: bool = False
) -> dict:
    """Require all real-update, synchronization, evaluation and reload evidence."""
    events = []
    for path in sorted(folder.glob("audit-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    optimizers = [
        item
        for item in events
        if item["event"] == "optimizer_update" and item.get("status") == "PASS"
    ]
    batches = [item for item in events if item["event"] == "training_batch"]
    syncs = [
        item
        for item in events
        if item["event"] == "weight_sync" and item.get("status") == "PASS"
    ]
    evaluations = [
        item
        for item in events
        if item["event"] == "evaluation_completed" and item.get("global_step", 0) > 0
    ]
    checkpoints = [
        item
        for item in events
        if item["event"] == "checkpoint_reload" and item.get("status") == "PASS"
    ]
    resumes = [
        item
        for item in events
        if item["event"] == "fresh_process_resume"
        and item.get("status") == "PASS"
        and item.get("fresh_process") is True
        and item.get("source_process") != item.get("destination_process")
    ]
    worker_resumes = [
        item
        for item in events
        if item["event"] == "worker_resume"
        and item.get("status") == "PASS"
        and item.get("fresh_process") is True
    ]
    worker_checkpoints = [
        item
        for item in events
        if item["event"] == "worker_checkpoint" and item.get("status") == "PASS"
    ]
    skipped = [
        item
        for item in events
        if item["event"] == "training_skipped" and item.get("status") == "PASS"
    ]
    if (
        isinstance(min_updates, bool)
        or not isinstance(min_updates, int)
        or min_updates < 1
    ):
        raise ValueError("acceptance.min_updates must be a positive integer")
    baseline_steps = int(resumes[-1]["optimizer_steps"]) if resumes else 0
    update_steps = [int(item["optimizer_steps"]) for item in optimizers]
    if update_steps != list(
        range(baseline_steps + 1, baseline_steps + len(optimizers) + 1)
    ):
        raise RuntimeError(
            "INCOMPLETE_ONLINE_ACCEPTANCE: effective update counters are not consecutive"
        )
    final_steps = baseline_steps + len(optimizers)
    eval_steps = sum(
        item.get("executed_actions", 0)
        for item in events
        if item["event"] == "environment_chunk"
        and item.get("mode") == "eval"
        and item.get("global_step", 0) > 0
    )
    if not optimizers and skipped:
        raise RuntimeError(
            f"INSUFFICIENT_EFFECTIVE_UPDATES: no effective update; {len(skipped)} real batches skipped"
        )
    if (
        not optimizers
        or not batches
        or not syncs
        or not evaluations
        or not checkpoints
        or not eval_steps
    ):
        raise RuntimeError(
            "INCOMPLETE_ONLINE_ACCEPTANCE: missing update/sync/eval/checkpoint evidence"
        )
    if not any(item.get("global_step", 0) > 0 for item in syncs):
        raise RuntimeError(
            "INCOMPLETE_ONLINE_ACCEPTANCE: updated weights were not synchronized"
        )
    if final_steps < min_updates:
        raise RuntimeError(
            f"INSUFFICIENT_EFFECTIVE_UPDATES: {final_steps} < {min_updates}; "
            f"{len(skipped)} real batches had no usable training samples"
        )
    for name, items in (
        ("sync", syncs),
        ("evaluation", evaluations),
        ("checkpoint", checkpoints),
        ("worker checkpoint", worker_checkpoints),
    ):
        if not any(item.get("optimizer_steps") == final_steps for item in items):
            raise RuntimeError(
                f"INCOMPLETE_ONLINE_ACCEPTANCE: final updated weights lack {name}"
            )
    if require_resume and (not resumes or not worker_resumes):
        raise RuntimeError(
            "INCOMPLETE_ONLINE_ACCEPTANCE: fresh-process actor/worker resume was not verified"
        )
    return {
        "status": "PASS",
        "evidence": "real_native_robotwin_rollouts",
        "optimizer_updates": len(optimizers),
        "optimizer_steps": final_steps,
        "required_effective_updates": min_updates,
        "skipped_batches": len(skipped),
        "fresh_process_resume_verified": bool(resumes and worker_resumes),
        "real_training_environment_steps": sum(
            item["real_environment_steps"] for item in batches
        ),
        "real_evaluation_environment_steps": eval_steps,
        "trajectory_returns": [item["trajectory_returns"] for item in batches],
        "synchronization_checks": len(syncs),
        "checkpoints": [item["path"] for item in checkpoints],
        "limitations": [
            "Single rank NO_SHARD acceptance",
            "Environment resume restores episode-boundary RNG/reset schedules, not physical state replay",
            "No performance replication claim",
        ],
    }


def _worker_classes():
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    class AuditedActor(EmbodiedFSDPActor):
        def run_training(self):
            evidence = real_batch_evidence(
                self.rollout_batch,
                self.cfg.algorithm.adv_type,
                self.cfg.algorithm.group_size,
            )
            record(
                "training_batch", rank=self._rank, global_step=self.version, **evidence
            )
            skip_reason = None
            if self.cfg.algorithm.adv_type == "grpo":
                try:
                    require_informative_grpo(evidence)
                except RuntimeError as error:
                    skip_reason = str(error)
                    if evidence.get("loss_mask", {}).get("nonzero") != 0:
                        raise RuntimeError(
                            "Uninformative GRPO groups must have an empty effective loss mask"
                        ) from error
            before_skip = self.audit_training_state() if skip_reason else None
            started = time.monotonic()
            result = super().run_training()
            if before_skip is not None:
                after_skip = self.audit_training_state()
                require_matching_training_states(before_skip, after_skip)
                if result.get("actor/skipped_no_valid_samples", 0) <= 0:
                    raise RuntimeError(
                        "Uninformative GRPO batch did not use the native empty-batch skip"
                    )
                record(
                    "training_skipped",
                    status="PASS",
                    reason=skip_reason,
                    rank=self._rank,
                    global_step=self.version,
                    optimizer_steps=self.optimizer_steps,
                    fingerprints=after_skip,
                    native_empty_batch_skip=True,
                )
            record(
                "training_finished", rank=self._rank, seconds=time.monotonic() - started
            )
            return result

        def optimizer_step(self):
            parameters = {
                name: value
                for name, value in self.model.named_parameters()
                if value.requires_grad
            }
            before = {
                name: value.detach().clone() for name, value in parameters.items()
            }
            gradients = {
                name: None
                if value.grad is None
                else float(value.grad.detach().float().norm())
                for name, value in parameters.items()
            }
            groups = parameter_groups(parameters)
            gradient_report = {
                name: value if value is None or math.isfinite(value) else str(value)
                for name, value in gradients.items()
            }
            group_norms = {
                group: math.sqrt(
                    sum(
                        gradients[name] ** 2
                        for name in names
                        if gradients[name] is not None
                    )
                )
                if all(
                    gradients[name] is None or math.isfinite(gradients[name])
                    for name in names
                )
                else None
                for group, names in groups.items()
            }
            record(
                "optimizer_gradients",
                rank=self._rank,
                optimizer_steps=self.optimizer_steps,
                gradients=gradient_report,
                gradient_norms=group_norms,
                groups=groups,
                stage="before_optimizer_step",
            )
            result = super().optimizer_step()
            changes = {
                name: float((value.detach() - before[name]).abs().max())
                for name, value in parameters.items()
            }
            policy = unwrapped(self.model)
            required = {
                "has_noise_head": policy.sampler.noise_head is not None,
                "has_critic": policy.add_value_head,
            }
            report = {
                "rank": self._rank,
                "global_step": self.version,
                "optimizer_steps": self.optimizer_steps,
                "gradients": gradient_report,
                "gradient_norms": group_norms,
                "parameter_max_changes": {
                    name: value if math.isfinite(value) else str(value)
                    for name, value in changes.items()
                },
                "groups": groups,
                "peak_device_allocated_bytes": int(
                    self.torch_platform.max_memory_allocated()
                ),
            }
            try:
                require_effective_update(gradients, changes, groups, **required)
            except RuntimeError as error:
                record("optimizer_update", status="FAILED", error=str(error), **report)
                raise
            record("optimizer_update", status="PASS", **report)
            return result

        def audit_state(self):
            if self._world_size != 1:
                raise RuntimeError("Online acceptance requires one NO_SHARD actor rank")
            if self.is_weight_offloaded:
                raise RuntimeError(
                    "Online acceptance requires actor weight offload disabled"
                )
            state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
            digest = state_digest(state)
            del state
            return {
                "digest": digest,
                "probe": fixed_probe(unwrapped(self.model)),
                "optimizer_steps": self.optimizer_steps,
            }

        def audit_training_state(self):
            if (
                self._world_size != 1
                or self.is_weight_offloaded
                or self.is_optimizer_offloaded
            ):
                raise RuntimeError(
                    "Training fingerprints require one actor rank with offload disabled"
                )
            state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
            return training_fingerprints(
                state,
                self.optimizer.state_dict(),
                self.lr_scheduler.state_dict(),
                capture_rng_state(),
                self.optimizer_steps,
            )

        def load_checkpoint(self, load_path):
            metadata = json.loads(
                (Path(load_path) / "xvla_training_state.json").read_text(
                    encoding="utf-8"
                )
            )
            super().load_checkpoint(load_path)
            self.optimizer_steps = int(metadata["fingerprints"]["optimizer_steps"])
            self.version = int(metadata["policy_version"])
            actual = self.audit_training_state()
            require_matching_training_states(metadata["fingerprints"], actual)
            fresh_process = metadata["source_process"] != process_identity()
            record(
                "fresh_process_resume"
                if fresh_process
                else "checkpoint_state_restored",
                status="PASS",
                path=str(load_path),
                optimizer_steps=self.optimizer_steps,
                global_step=metadata["global_step"],
                fingerprints=actual,
                source_process=metadata["source_process"],
                destination_process=process_identity(),
                fresh_process=fresh_process,
            )

        def save_checkpoint(self, save_path, step=0):
            if not self.optimizer.state:
                if self.optimizer_steps:
                    raise RuntimeError(
                        "A trained actor cannot checkpoint an empty Adam state"
                    )
                record(
                    "checkpoint_skipped",
                    status="SKIPPED",
                    step=step,
                    optimizer_steps=0,
                    path=str(save_path),
                    resumable=False,
                    reason="ZERO_UPDATE_EMPTY_ADAM_STATE: native DCP would create a synthetic Adam step",
                )
                return
            before = self.audit_state()
            training_before = self.audit_training_state()
            super().save_checkpoint(save_path, step)
            metadata = {
                "format_version": 1,
                "source_process": process_identity(),
                "global_step": step,
                "policy_version": int(self.version),
                "fingerprints": training_before,
            }
            (Path(save_path) / "xvla_training_state.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            parameter = next(
                value
                for value in self.model.parameters()
                if value.requires_grad and value.numel()
            )
            buffer = next(
                (
                    value
                    for name, value in self.model.named_buffers()
                    if name.endswith(".frozen_weight") and value.numel()
                ),
                None,
            )
            if buffer is None:
                raise RuntimeError(
                    "Checkpoint acceptance requires a persistent frozen X-VLA domain buffer"
                )
            moment_perturbed = False
            with torch.no_grad():
                index = (0,) * parameter.ndim
                initial = parameter[index].detach().clone()
                parameter[index].add_(1.0)
                if torch.equal(parameter[index], initial):
                    raise RuntimeError(
                        "Checkpoint verification failed to perturb a trainable parameter"
                    )
                buffer_index = (0,) * buffer.ndim
                buffer_initial = buffer[buffer_index].detach().clone()
                buffer[buffer_index].add_(1.0)
                if torch.equal(buffer[buffer_index], buffer_initial):
                    raise RuntimeError(
                        "Checkpoint verification failed to perturb a persistent buffer"
                    )
                for state in self.optimizer.state.values():
                    moment = state.get("exp_avg")
                    if torch.is_tensor(moment) and moment.numel():
                        moment.reshape(-1)[0].add_(1.0)
                        moment_perturbed = True
                        break
                if not moment_perturbed and self.optimizer_steps > 0:
                    raise RuntimeError(
                        "Checkpoint acceptance requires initialized Adam moments"
                    )
            self.lr_scheduler.last_epoch += 1
            self.optimizer_steps += 11
            random.random()
            import numpy as np

            np.random.rand()
            torch.rand(1, device="cpu")
            torch.rand(1, device=self.device)
            self.load_checkpoint(save_path)
            after = self.audit_state()
            require_matching_states(before, after)
            training_after = self.audit_training_state()
            require_matching_training_states(training_before, training_after)
            from omegaconf import OmegaConf

            evidence = {
                "model_config": OmegaConf.to_container(
                    unwrapped(self.model).cfg, resolve=True
                ),
                **after,
                "step": step,
                "native_checkpoint_api": True,
                "training_fingerprints": training_after,
            }
            (Path(save_path) / "xvla_online_evidence.json").write_text(
                json.dumps(evidence, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            record(
                "checkpoint_reload",
                status="PASS",
                rank=self._rank,
                step=step,
                optimizer_steps=self.optimizer_steps,
                path=str(save_path),
                digest=after["digest"],
                actual_parameter_perturbation=True,
                actual_buffer_perturbation=True,
                actual_scheduler_rng_perturbation=True,
                actual_optimizer_moment_perturbation=moment_perturbed,
                fingerprints=training_after,
            )

    class AuditedRollout(MultiStepRolloutWorker):
        def export_resume_state(self):
            return {
                "rng": capture_rng_state(),
                "version": int(self.version),
                "finished_episodes": None
                if self.finished_episodes is None
                else int(self.finished_episodes),
            }

        def import_resume_state(self, state):
            self.version = int(state["version"])
            self.finished_episodes = state["finished_episodes"]
            if hasattr(self.hf_model, "set_global_step"):
                self.hf_model.set_global_step(self.version)
            restore_rng_state(state["rng"])
            return object_digest(self.export_resume_state())

        def audit_state(self):
            if self._world_size != 1 or self.enable_offload:
                raise RuntimeError(
                    "Online acceptance requires one rollout rank without offload"
                )
            return {
                "digest": state_digest(self.hf_model.state_dict()),
                "probe": fixed_probe(self.hf_model),
            }

    class AuditedEnv(EnvWorker):
        def export_resume_state(self):
            if self._prefetched_train_bootstrap is not None:
                raise RuntimeError(
                    "Episode-boundary resume cannot serialize prefetched observations"
                )
            return {
                "rng": capture_rng_state(),
                "trajectory_step": self._trajectory_step,
                "train": [env.get_reset_schedule_state() for env in self.env_list],
                "eval": [env.get_reset_schedule_state() for env in self.eval_env_list],
            }

        def import_resume_state(self, state):
            for name, envs in (("train", self.env_list), ("eval", self.eval_env_list)):
                if len(state[name]) != len(envs):
                    raise ValueError("Checkpoint environment stage count differs")
                for env, schedule in zip(envs, state[name]):
                    env.set_reset_schedule_state(schedule)
            self._trajectory_step = int(state["trajectory_step"])
            restore_rng_state(state["rng"])
            return object_digest(self.export_resume_state())

        @property
        def audit_global_step(self) -> int:
            """Expose the counter maintained by EnvWorker.set_global_step."""
            return self._trajectory_step

        def env_interact_step(self, chunk_actions, stage_id):
            result = super().env_interact_step(chunk_actions, stage_id)
            output, info, _ = result
            mask = output.transition.executed_mask
            if mask is None:
                raise RuntimeError(
                    "Native training transitions did not expose executed_mask"
                )
            record(
                "environment_chunk",
                mode="train",
                rank=self._rank,
                global_step=self.audit_global_step,
                stage=stage_id,
                reset_state_ids=self.env_list[stage_id].reset_state_ids,
                executed_actions=int(mask.sum()),
                rewards=output.transition.rewards.detach().cpu().tolist(),
                episode={
                    key: value.detach().cpu().tolist()
                    if torch.is_tensor(value)
                    else value
                    for key, value in info.items()
                },
            )
            return result

        def env_evaluate_step(self, raw_actions, stage_id):
            before = self.eval_env_list[stage_id].elapsed_steps.detach().clone()
            result = super().env_evaluate_step(raw_actions, stage_id)
            after = self.eval_env_list[stage_id].elapsed_steps.detach()
            _, info = result
            record(
                "environment_chunk",
                mode="eval",
                rank=self._rank,
                global_step=self.audit_global_step,
                stage=stage_id,
                reset_state_ids=self.eval_env_list[stage_id].reset_state_ids,
                executed_actions=int((after - before).sum()),
                episode={
                    key: value.detach().cpu().tolist()
                    if torch.is_tensor(value)
                    else value
                    for key, value in info.items()
                },
            )
            return result

    class AuditedRunner(EmbodiedRunner):
        def init_workers(self):
            if self.overlap_env_bootstrap:
                raise ValueError(
                    "Online acceptance checkpoints require episode boundaries without prefetch"
                )
            super().init_workers()
            resume_dir = self.cfg.runner.get("resume_dir", None)
            if resume_dir is None:
                return
            state = torch.load(
                Path(resume_dir) / "xvla_worker_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                state["global_step"] != self.global_step
                or state["source_process"] == process_identity()
            ):
                raise RuntimeError(
                    "Worker resume requires the matching checkpoint in a fresh driver process"
                )
            for name in ("rollout", "env"):
                if (
                    len(state[name]) != 1
                    or [object_digest(item) for item in state[name]]
                    != state[f"{name}_digests"]
                ):
                    raise RuntimeError(f"Invalid {name} checkpoint payload fingerprint")
            rollout_digests = self.rollout.import_resume_state(
                state["rollout"][0]
            ).wait()
            env_digests = self.env.import_resume_state(state["env"][0]).wait()
            if (
                rollout_digests != state["rollout_digests"]
                or env_digests != state["env_digests"]
            ):
                raise RuntimeError(
                    "Worker RNG/reset schedule differs after fresh-process resume"
                )
            record(
                "worker_resume",
                status="PASS",
                global_step=self.global_step,
                rollout_digests=rollout_digests,
                env_digests=env_digests,
                physical_state_replay=False,
                fresh_process=True,
                source_process=state["source_process"],
                destination_process=process_identity(),
            )

        def _save_checkpoint(self):
            super()._save_checkpoint()
            checkpoint_root = (
                Path(self.cfg.runner.logger.log_path)
                / self.cfg.runner.logger.experiment_name
                / f"checkpoints/global_step_{self.global_step}"
            )
            if not (checkpoint_root / "actor/xvla_training_state.json").is_file():
                record(
                    "worker_checkpoint",
                    status="SKIPPED",
                    global_step=self.global_step,
                    resumable=False,
                    reason="No native actor checkpoint was saved",
                )
                return
            rollout_states = self.rollout.export_resume_state().wait()
            env_states = self.env.export_resume_state().wait()
            if len(rollout_states) != 1 or len(env_states) != 1:
                raise RuntimeError(
                    "Worker resume acceptance requires one rollout and environment rank"
                )
            state = {
                "format_version": 1,
                "source_process": process_identity(),
                "global_step": self.global_step,
                "rollout": rollout_states,
                "env": env_states,
                "rollout_digests": [object_digest(item) for item in rollout_states],
                "env_digests": [object_digest(item) for item in env_states],
            }
            path = checkpoint_root / "xvla_worker_state.pt"
            torch.save(state, path)
            record(
                "worker_checkpoint",
                status="PASS",
                path=str(path),
                global_step=self.global_step,
                optimizer_steps=self._audit_synced_optimizer_steps,
            )

        def update_rollout_weights(self):
            started = time.monotonic()
            super().update_rollout_weights()
            actors = self.actor.audit_state().wait()
            rollouts = self.rollout.audit_state().wait()
            if len(actors) != 1 or len(rollouts) != 1:
                raise RuntimeError(
                    "Online acceptance requires one actor and rollout rank"
                )
            require_matching_states(actors[0], rollouts[0])
            self._audit_synced_optimizer_steps = actors[0]["optimizer_steps"]
            record(
                "weight_sync",
                status="PASS",
                global_step=self.global_step,
                optimizer_steps=self._audit_synced_optimizer_steps,
                digest=actors[0]["digest"],
                seconds=time.monotonic() - started,
                probe="fixed synthetic observation/noise for consistency only",
            )

        def evaluate(self):
            self.env.set_global_step(self.global_step).wait()
            result = super().evaluate()
            record(
                "evaluation_completed",
                global_step=self.global_step,
                metrics=result,
                optimizer_steps=self._audit_synced_optimizer_steps,
            )
            return result

    return AuditedActor, AuditedRollout, AuditedEnv, AuditedRunner


def validate_acceptance_cfg(cfg: Any) -> None:
    """Validate audited training configuration before starting any Ray workers."""
    if str(cfg.actor.model.model_type) != "xvla" or cfg.runner.get("only_eval", False):
        raise ValueError("Online acceptance requires X-VLA training")
    if cfg.reward.get("use_reward_model", False) or cfg.runner.get(
        "use_training_pipeline", False
    ):
        raise ValueError(
            "Acceptance uses native rewards and the synchronous RLinf runner"
        )
    if cfg.actor.fsdp_config.sharding_strategy != "no_shard":
        raise ValueError("Online acceptance currently requires single-rank NO_SHARD")
    if cfg.actor.optim.get("critic_warmup_steps", 0) != 0:
        raise ValueError(
            "Online acceptance requires a real actor update without critic warmup"
        )
    if cfg.runner.val_check_interval <= 0 or cfg.runner.save_interval <= 0:
        raise ValueError(
            "Acceptance requires positive evaluation/checkpoint intervals and final updated-weight evidence"
        )
    if cfg.runner.get("weight_sync_interval", 1) != 1:
        raise ValueError(
            "Online acceptance requires weight_sync_interval=1, including fresh-process resume"
        )
    if cfg.runner.get("overlap_env_bootstrap", False):
        raise ValueError(
            "Online acceptance requires episode-boundary checkpoints without bootstrap prefetch"
        )
    minimum = cfg.get("acceptance", {}).get("min_updates", 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("acceptance.min_updates must be a positive integer")


def _record_failure(destination: Path, error: BaseException, elapsed: float) -> None:
    from rlinf.scheduler.cluster.cluster import IsolatedWorkerFailure

    private_worker_failure = isinstance(error, IsolatedWorkerFailure)
    details = {"error_type": type(error).__name__, "error": str(error)}
    if private_worker_failure:
        details.update(
            error_type=error.error_type,
            worker_class=error.worker_class,
            method=error.method,
        )
    record("acceptance_failed", **details)
    summary = {
        "status": "FAILED",
        "evidence": "real_native_robotwin_rollouts",
        **details,
        "elapsed_seconds": elapsed,
    }
    with destination.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if private_worker_failure:
            stream.flush()
            os.fsync(stream.fileno())
    if private_worker_failure:
        # The launcher cleans the private Ray tree without reentering failed RPCs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run(cfg) -> None:
    import ray
    import torch.multiprocessing as mp
    from omegaconf import OmegaConf

    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    login_debug = (
        not os.environ.get("SLURM_JOB_ID")
        and os.environ.get("XVLA_LOGIN_TRAINING_DEBUG") == "1"
        and socket.gethostname() == "4090node1"
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", os.environ.get("XVLA_LOGIN_RUN_ID", "")
        )
        is not None
        and os.environ.get("XVLA_LOGIN_CPUS") == "4"
        and 30 <= int(os.environ.get("XVLA_LOGIN_SECONDS", "1200")) <= 1499
    )
    if not os.environ.get("SLURM_JOB_ID") and not login_debug:
        raise RuntimeError(
            "Online acceptance requires Slurm or the explicit bounded 4090node1 debug context"
        )
    validate_acceptance_cfg(cfg)
    folder = Path(os.environ["XVLA_RUN_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "online-acceptance.json"
    if destination.exists() or any(folder.glob("audit-*.jsonl")):
        raise FileExistsError("Choose a new job result directory for online acceptance")
    destination.write_text(
        json.dumps({"status": "RUNNING", "evidence": "real_native_robotwin_rollouts"})
        + "\n"
    )
    mp.set_start_method("spawn", force=True)
    started = time.monotonic()
    try:
        cfg = validate_cfg(cfg)
        (folder / "online-config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )
        cluster = Cluster(
            cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
        )
        placement = HybridComponentPlacement(cfg, cluster)
        actor_type, rollout_type, env_type, runner_type = _worker_classes()
        rollout = rollout_type.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env = env_type.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        actor = actor_type.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy("actor"),
        )
        runner = runner_type(cfg=cfg, actor=actor, rollout=rollout, env=env)
        runner.init_workers()
        runner.run()
        acceptance = cfg.get("acceptance", {})
        summary = summarize_evidence(
            folder,
            min_updates=acceptance.get("min_updates", 3),
            require_resume=bool(
                cfg.runner.get("resume_dir") or acceptance.get("require_resume", False)
            ),
        )
        summary.update(
            algorithm="ppo"
            if cfg.algorithm.adv_type == "gae"
            else cfg.algorithm.adv_type,
            advantage_estimator=cfg.algorithm.adv_type,
            elapsed_seconds=time.monotonic() - started,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            login_debug_run_id=os.environ.get("XVLA_LOGIN_RUN_ID")
            if login_debug
            else None,
        )
        destination.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except BaseException as error:
        _record_failure(destination, error, time.monotonic() - started)
        raise
    finally:
        ray.shutdown()


def main() -> None:
    """Compose the standard X-VLA config and launch audited RLinf components."""
    import hydra

    config_root = str(
        Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    )
    hydra.main(
        version_base="1.1",
        config_path=config_root,
        config_name="robotwin_ppo_xvla_acceptance",
    )(_run)()


if __name__ == "__main__":
    main()
