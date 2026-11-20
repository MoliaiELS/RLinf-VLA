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

"""Bounded fixed-model GPU diagnostics with synthetic observations and rewards.

This entry never starts Ray, Slurm or a simulator. Its optimizer/checkpoint
checks cover the policy and RLinf loss contracts, not online RL performance or
distributed training-state restoration.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("xvla.gpu_debug")
EVIDENCE = "synthetic_observation_and_rewards"
CHECKPOINT_KIND = "trainable_parameters_only_debug_roundtrip"
CHUNK = 30
BATCH = 2


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("ppo", "grpo", "both"), default="both")
    parser.add_argument(
        "--method", choices=("flow_noise", "flow_sde"), default="flow_noise"
    )
    parser.add_argument("--allow-login-debug", action="store_true")
    parser.add_argument(
        "--observations",
        type=Path,
        help="Optional native-observation NPZ; rewards remain synthetic",
    )
    parser.add_argument("--instruction", default="move stapler pad")
    return parser.parse_args(argv)


def _execution_guard(args: argparse.Namespace) -> tuple[Path, str]:
    hostname = socket.gethostname().split(".", 1)[0]
    if not os.environ.get("SLURM_JOB_ID") and not (
        args.allow_login_debug and hostname == "4090node1"
    ):
        raise RuntimeError(
            "Use an allocated Slurm job, or explicitly permit bounded login debug on 4090node1"
        )
    if not args.checkpoint.is_absolute() or not args.checkpoint.is_dir():
        raise ValueError(
            "--checkpoint must name an absolute existing verified model directory"
        )
    if not args.output.is_absolute() or args.output == Path(args.output.anchor):
        raise ValueError("--output must name an absolute dedicated result directory")
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(repo) or output.is_relative_to(args.checkpoint.resolve()):
        raise ValueError(
            "Keep diagnostic outputs outside the code worktree and checkpoint"
        )
    if (
        subprocess.run(
            ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ):
        raise RuntimeError("Run GPU diagnostics from a detached fixed-commit worktree")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    if dirty:
        raise RuntimeError("Commit diagnostic code before creating the GPU worktree")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / filename).exists()
        for filename in ("events.jsonl", "runtime.json", "summary.json")
    ):
        raise FileExistsError(
            "Diagnostic evidence already exists in --output; choose a new directory"
        )
    return repo, commit


class Recorder:
    """Append flushed phase evidence and keep one atomic terminal summary."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.started = time.monotonic()

    def event(self, name: str, **details: Any) -> None:
        item = {
            "event": name,
            "elapsed_seconds": time.monotonic() - self.started,
            "evidence": EVIDENCE,
            **details,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
        LOGGER.info("%s %s", name, json.dumps(details, allow_nan=False))

    def summary(self, status: str, **details: Any) -> None:
        result = {
            "status": status,
            "evidence": EVIDENCE,
            "real_environment_steps": 0,
            "real_rollouts": 0,
            "elapsed_seconds": time.monotonic() - self.started,
            "checkpoint_kind": CHECKPOINT_KIND,
            **details,
        }
        temporary = self.directory / "summary.json.tmp"
        temporary.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(self.directory / "summary.json")


def _observation(np, torch, args):
    if args.observations is not None:
        if not args.observations.is_absolute():
            raise ValueError("--observations must be an absolute NPZ path")
        with np.load(args.observations, allow_pickle=False) as data:
            arrays = {
                key: np.array(data[key], copy=True)
                for key in ("main_images", "wrist_images", "states")
            }
        source = str(args.observations.resolve())
    else:
        y, x = np.indices((240, 320), dtype=np.uint16)
        images = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        main = np.stack((images, np.roll(images, 13, axis=1)))
        wrists = np.stack(
            (np.roll(main, 29, axis=2), np.roll(main, 67, axis=1)), axis=1
        )
        arm_left = [0.30, 0.20, 0.20, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
        arm_right = [0.30, -0.20, 0.20, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
        arrays = {
            "main_images": main,
            "wrist_images": wrists,
            "states": np.repeat(
                np.asarray([arm_left + arm_right], dtype=np.float32), BATCH, axis=0
            ),
        }
        source = "deterministic_rgb_and_proprio_placeholders"
    expected = {
        "main_images": (BATCH, 240, 320, 3),
        "wrist_images": (BATCH, 2, 240, 320, 3),
        "states": (BATCH, 20),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(
                f"Expected finite {key} with shape {shape}; got {arrays[key].shape}"
            )
    if (
        arrays["main_images"].dtype != np.uint8
        or arrays["wrist_images"].dtype != np.uint8
    ):
        raise ValueError("RGB diagnostic observations must have uint8 dtype")
    return {key: torch.from_numpy(value) for key, value in arrays.items()} | {
        "task_descriptions": [args.instruction] * BATCH
    }, source


def _cache_row(cache: dict, index: int) -> dict:
    return {key: value[index : index + 1] for key, value in cache.items()}


def _trainable_state(model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable(model, state: dict) -> None:
    import torch

    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if parameters.keys() != state.keys():
        raise RuntimeError(
            "Trainable parameter names differ from the saved debug payload"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise RuntimeError(f"Trainable parameter shape differs: {name}")
            parameter.copy_(state[name].to(parameter))


def _frozen_vlm(model) -> dict:
    parameters = list(model.core.vlm.named_parameters())
    if not parameters or any(
        parameter.requires_grad or parameter.grad is not None
        for _, parameter in parameters
    ):
        raise RuntimeError(
            "The fixed-model debug scope must keep the VLM frozen without gradients"
        )
    return {
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(parameter.numel() for _, parameter in parameters),
        "check": "requires_grad_false_and_grad_none",
        "full_weight_hash_checked": False,
    }


def _gradient_records(model) -> tuple[dict, dict[str, list[str]]]:
    import torch

    groups = {"actor": [], "noise": [], "critic": []}
    records = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("core.transformer."):
            groups["actor"].append(name)
        elif name.startswith("sampler.noise_head."):
            groups["noise"].append(name)
        elif name.startswith("value_head."):
            groups["critic"].append(name)
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"Nonfinite diagnostic gradient: {name}")
        records[name] = {
            "elements": parameter.numel(),
            "gradient_norm": None
            if gradient is None
            else float(gradient.float().norm()),
        }
    return records, groups


def _require_group_signal(records, names, group: str) -> None:
    if not names or not any(
        (records[name]["gradient_norm"] or 0.0) > 0 for name in names
    ):
        raise RuntimeError(f"No nonzero {group} gradient in the diagnostic update")


def _algorithm_check(
    args, algorithm: str, observation: dict, recorder: Recorder
) -> dict:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
    from rlinf.models.embodiment.xvla.policy import XVLAPolicy

    recorder.event("model_load_started", algorithm=algorithm)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cfg = OmegaConf.create(
        {
            "model_type": "xvla",
            "model_path": str(args.checkpoint),
            "precision": "fp32",
            "is_lora": False,
            "action_dim": 20,
            "num_action_chunks": CHUNK,
            "logprob_action_dim": 1,
            "add_value_head": algorithm == "ppo",
            "xvla": {
                "domain_id": 6,
                "steps": 10,
                "update_scope": "prompt_heads",
                "noise_method": args.method,
                "eval_sampler": "flow_ode",
                "checkpoint_transformer": True,
                "noise_hidden_dim": 128,
            },
        }
    )
    model = XVLAPolicy.from_checkpoint(cfg, torch.float32).to("cuda:0")
    if algorithm == "ppo":
        # The fixture tests failed-return learning from a known nonzero baseline.
        # This initializes only the newly added critic, never the pretrained actor.
        with torch.no_grad():
            model.value_head[-1].weight.zero_()
            model.value_head[-1].bias.fill_(0.5)
    frozen = _frozen_vlm(model)
    recorder.event(
        "model_loaded",
        algorithm=algorithm,
        frozen_vlm=frozen,
        trainable_elements=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    recorder.event("sampling_started", algorithm=algorithm)
    actions, rollout = model.predict_action_batch(
        observation, mode="train", compute_values=algorithm == "ppo"
    )
    if actions.shape != (BATCH, CHUNK, 20) or not np.isfinite(actions).all():
        raise RuntimeError(
            "The full checkpoint returned invalid dual-arm diagnostic actions"
        )
    cache = rollout["forward_inputs"]
    cache["executed_mask"] = torch.ones(BATCH, CHUNK, dtype=torch.bool, device="cuda:0")
    if cache["chains"].shape != (BATCH, 11, CHUNK, 20):
        raise RuntimeError(
            "The diagnostic did not retain the full ten-step denoising path"
        )
    recorder.event(
        "sampled",
        algorithm=algorithm,
        actions_shape=list(actions.shape),
        chain_shape=list(cache["chains"].shape),
        synthetic_chunks=BATCH,
        joint_logprobs=rollout["prev_logprobs"].sum(dim=(1, 2)).cpu().tolist(),
    )

    # Advantage preprocessing in the framework runs on the received CPU rollout.
    rewards = torch.zeros(1, BATCH, CHUNK)
    if algorithm == "grpo":
        rewards[0, 0, -1] = 1.0
    dones = torch.zeros(2, BATCH, CHUNK, dtype=torch.bool)
    dones[-1] = True
    values = torch.cat(
        (rollout["prev_values"].detach().cpu()[None], torch.zeros(1, BATCH, 1)), dim=0
    )
    advantages = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae" if algorithm == "ppo" else "grpo",
        reward_type="chunk_level",
        rewards=rewards,
        dones=dones,
        values=values if algorithm == "ppo" else None,
        loss_mask=torch.ones_like(rewards, dtype=torch.bool),
        gamma=0.99 if algorithm == "ppo" else 1.0,
        gae_lambda=0.95,
        group_size=1 if algorithm == "ppo" else BATCH,
        normalize_advantages=False,
    )
    advantage = advantages["advantages"].reshape(BATCH).to("cuda:0")
    returns = advantages.get("returns")
    if returns is not None:
        returns = returns.reshape(BATCH).to("cuda:0")
    if not bool(torch.isfinite(advantage).all()) or not bool((advantage != 0).any()):
        raise RuntimeError(
            "Synthetic diagnostic advantages are not finite and informative"
        )
    recorder.event(
        "synthetic_advantages",
        algorithm=algorithm,
        rewards=rewards.sum(dim=(0, 2)).tolist(),
        advantages=advantage.detach().cpu().tolist(),
        critic_initialization=0.5 if algorithm == "ppo" else None,
        group_size=1 if algorithm == "ppo" else BATCH,
        group_variance=float(rewards.sum(dim=(0, 2)).var(unbiased=False))
        if algorithm == "grpo"
        else None,
        note="Diagnostic fixture only; no simulator or environment reward was used",
    )

    trainable_before = _trainable_state(model)
    actor_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("value_head.")
    ]
    parameter_groups = [{"params": actor_parameters, "lr": 1e-6}]
    if algorithm == "ppo":
        parameter_groups.append({"params": model.value_head.parameters(), "lr": 1e-5})
    optimizer = torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)
    errors, losses = [], []
    for index in range(BATCH):
        recorder.event("microbatch_started", algorithm=algorithm, index=index)
        output = model.default_forward(
            _cache_row(cache, index), compute_values=algorithm == "ppo"
        )
        previous = rollout["prev_logprobs"][index : index + 1]
        difference = (output["logprobs"].detach() - previous).sum()
        error = float(difference)
        errors.append(error)
        recorder.event(
            "pre_update_recompute",
            algorithm=algorithm,
            index=index,
            joint_logprob_difference=error,
            ratio=float(difference.exp()),
            tolerance=0.01,
        )
        if not bool(torch.isfinite(difference)) or abs(error) > 0.01:
            raise RuntimeError(
                "Microbatch recomputation differs from the behavior policy by more than 0.01"
            )
        loss, metrics = policy_loss(
            task_type="embodied",
            loss_type="actor_critic" if algorithm == "ppo" else "actor",
            reward_type="chunk_level",
            logprob_type="chunk_level",
            single_action_dim=1,
            logprobs=output["logprobs"],
            old_logprobs=previous,
            advantages=advantage[index : index + 1],
            returns=None if returns is None else returns[index : index + 1],
            values=output["values"] if algorithm == "ppo" else None,
            prev_values=rollout["prev_values"][index : index + 1].reshape(-1)
            if algorithm == "ppo"
            else None,
            loss_mask=torch.ones(1, dtype=torch.bool, device="cuda:0"),
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            value_clip=0.2,
            huber_delta=10.0,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite RLinf policy loss")
        losses.append(float(loss.detach()))
        (loss / BATCH).backward()
        recorder.event(
            "microbatch_backward", algorithm=algorithm, index=index, loss=losses[-1]
        )
        del output, loss, metrics
    records, groups = _gradient_records(model)
    _require_group_signal(records, groups["actor"], "actor")
    if args.method == "flow_noise":
        _require_group_signal(records, groups["noise"], "noise head")
    if algorithm == "ppo":
        _require_group_signal(records, groups["critic"], "critic")
    recorder.event("gradients", algorithm=algorithm, parameters=records, groups=groups)
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    )
    if not bool(torch.isfinite(clipped_norm)):
        raise RuntimeError("Nonfinite aggregate diagnostic gradient norm")
    optimizer.step()
    torch.cuda.synchronize()
    changes = {
        name: float((parameter.detach().cpu() - trainable_before[name]).abs().max())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    required_groups = (
        ["actor"]
        + (["noise"] if args.method == "flow_noise" else [])
        + (["critic"] if algorithm == "ppo" else [])
    )
    for group in required_groups:
        if not any(changes[name] > 0 for name in groups[group]):
            raise RuntimeError(f"No actual {group} parameter change after AdamW")
    _frozen_vlm(model)
    recorder.event(
        "optimizer_update",
        algorithm=algorithm,
        updates=1,
        gradient_norm_before_clip=float(clipped_norm),
        parameter_max_changes=changes,
        frozen_vlm=frozen,
    )

    with torch.no_grad():
        reference = model.default_forward(
            _cache_row(cache, 0), compute_values=algorithm == "ppo"
        )
        reference = {
            key: value.detach().cpu().clone()
            for key, value in reference.items()
            if key in ("logprobs", "values")
        }
    saved = _trainable_state(model)
    checkpoint = recorder.directory / f"{algorithm}-trainable-debug.pt"
    torch.save(
        {
            "kind": CHECKPOINT_KIND,
            "algorithm": algorithm,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "parameters": saved,
        },
        checkpoint,
    )
    with torch.no_grad():
        next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).view(-1)[0].add_(1.0)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _restore_trainable(model, loaded["parameters"])
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(
            parameter.detach().cpu(), loaded["parameters"][name]
        ):
            raise RuntimeError(f"Trainable parameter restoration differs: {name}")
    with torch.no_grad():
        restored = model.default_forward(
            _cache_row(cache, 0), compute_values=algorithm == "ppo"
        )
    roundtrip_error = {}
    for key, expected in reference.items():
        actual = restored[key].detach().cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        roundtrip_error[key] = float((actual - expected).abs().max())
    recorder.event(
        "trainable_checkpoint_roundtrip",
        algorithm=algorithm,
        status="PASS",
        checkpoint=str(checkpoint),
        bytes=checkpoint.stat().st_size,
        errors=roundtrip_error,
        kind=CHECKPOINT_KIND,
        not_tested="FSDP, optimizer, scheduler and RNG checkpoint restoration",
    )
    result = {
        "algorithm": algorithm,
        "status": "PASS",
        "optimizer_updates": 1,
        "pre_update_joint_logprob_differences": errors,
        "losses": losses,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint": str(checkpoint),
        "checkpoint_kind": CHECKPOINT_KIND,
    }
    del optimizer, model, rollout, cache, restored, saved, loaded
    gc.collect()
    torch.cuda.empty_cache()
    recorder.event("algorithm_finished", **result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    repo, commit = _execution_guard(args)
    recorder = Recorder(args.output)
    recorder.summary("RUNNING", code_commit=commit)

    def interrupted(signum, _frame):
        raise TimeoutError(f"Diagnostic received termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        import numpy as np
        import ray
        import torch

        if ray.is_initialized():
            raise RuntimeError("The diagnostic must not connect to a Ray runtime")
        if (
            len(
                [
                    part
                    for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                    if part.strip()
                ]
            )
            != 1
        ):
            raise RuntimeError(
                "Expose exactly one explicitly selected GPU for bounded debug"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA GPU")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(0)
        rlinf_origin = Path(importlib.import_module("rlinf").__file__).resolve()
        if not rlinf_origin.is_relative_to(repo):
            raise RuntimeError(f"Expected RLinf from {repo}, got {rlinf_origin}")
        model_receipt = json.loads(
            (args.checkpoint / "verified.json").read_text(encoding="utf-8")
        )
        runtime = {
            "code_commit": commit,
            "code_root": str(repo),
            "python": sys.version,
            "executable": sys.executable,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "login_debug_opt_in": args.allow_login_debug,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu": {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "total_memory_bytes": properties.total_memory,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": {
                name: metadata.version(name)
                for name in (
                    "transformers",
                    "timm",
                    "peft",
                    "numpy",
                    "scipy",
                    "ray",
                    "hydra-core",
                    "omegaconf",
                )
            },
            "imports": {
                name: importlib.import_module(name).__file__
                for name in ("rlinf", "transformers", "xvla_rlinf")
            },
            "checkpoint": str(args.checkpoint),
            "evidence": EVIDENCE,
            "model_repo": model_receipt.get("repo_id"),
            "model_revision": model_receipt.get("revision"),
            "method": args.method,
            "chunk": CHUNK,
            "denoising_steps": 10,
            "training_scope": "prompt_heads",
            "storage_precision": "fp32",
            "ray_started": False,
            "simulation_started": False,
        }
        (args.output / "runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        observation, source = _observation(np, torch, args)
        recorder.event("diagnostic_started", runtime=runtime, observation_source=source)
        algorithms = ("ppo", "grpo") if args.algorithm == "both" else (args.algorithm,)
        results = [
            _algorithm_check(args, algorithm, observation, recorder)
            for algorithm in algorithms
        ]
        if ray.is_initialized():
            raise RuntimeError("Diagnostic unexpectedly initialized Ray")
        recorder.summary(
            "PASS",
            code_commit=commit,
            algorithms=results,
            method=args.method,
            observation_source=source,
            limitations=[
                "No simulator or online reward was used",
                "No RLinf Worker/FSDP/weight-sync acceptance was performed",
                "Checkpoint test saves trainable parameters only",
            ],
        )
    except BaseException as error:
        recorder.event(
            "diagnostic_failed", error_type=type(error).__name__, error=str(error)
        )
        recorder.summary(
            "FAILED",
            code_commit=commit,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    main()
