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

"""X-VLA clean-prediction flow policies for on-policy embodied RL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .sampling import FlowSampler

MODEL_REPO = "2toINF/X-VLA-RoboTwin2"
MODEL_REVISION = "a157c580cfe6f9f445614490f3bec1b2f9ef9f18"
INPUT_KEYS = ("input_ids", "image_input", "image_mask", "domain_id", "proprio")


class DomainEmbedding(nn.Module):
    """Train one robot domain while retaining the other domains as buffers."""

    def __init__(self, embedding: nn.Embedding, domain_id: int):
        super().__init__()
        if not 0 <= domain_id < embedding.num_embeddings:
            raise ValueError(
                f"Domain {domain_id} exceeds {embedding.num_embeddings} embeddings"
            )
        self.domain_id = domain_id
        self.register_buffer("frozen_weight", embedding.weight.detach().clone())
        self.selected_weight = nn.Parameter(
            embedding.weight[domain_id].detach().clone()
        )

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        frozen = F.embedding(domain_ids, self.frozen_weight)
        return torch.where(
            (domain_ids == self.domain_id)[..., None], self.selected_weight, frozen
        )


def pack_joint_logprob(joint: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Transport one augmented macro-action density through RLinf's row mask.

    The first slot owns the entire sampled denoising path, including unused
    coordinates. Later slots are zero. An executed prefix keeps this joint
    density exactly once; an entirely unexecuted chunk removes it entirely.
    These transport slots are not individual physical-action densities.
    """
    if joint.ndim != 1 or chunk_size < 1:
        raise ValueError("Expected joint density [B] and a positive chunk size")
    return F.pad(joint.float()[:, None, None], (0, 0, 0, chunk_size - 1))


class XVLAPolicy(nn.Module, BasePolicy):
    """Expose stochastic denoising paths and observation values to RLinf."""

    def __init__(self, core: nn.Module, processor: Any, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.xcfg = cfg.get("xvla", {})
        self.core = core
        self.processor = processor
        self.domain_id = int(self.xcfg.get("domain_id", 6))
        self.num_action_chunks = int(cfg.get("num_action_chunks", 30))
        self.action_dim = 20
        self.steps = int(self.xcfg.get("steps", 10))
        if cfg.get("precision", "fp32") != "fp32":
            raise ValueError("X-VLA flow RL currently requires precision=fp32")
        if self.domain_id != 6:
            raise ValueError("The pinned RoboTwin checkpoint requires domain_id=6")
        if int(cfg.get("action_dim", 20)) != 20:
            raise ValueError("RoboTwin X-VLA requires both arms and action_dim=20")
        if core.action_space.dim_action != 20 or core.action_mode != "ee6d":
            raise ValueError("X-VLA RL requires the ee6d/20D checkpoint action space")
        if self.num_action_chunks != 30 or core.num_actions != 30:
            raise ValueError(
                "Native RoboTwin X-VLA requires a 30-anchor checkpoint and chunk"
            )
        if self.xcfg.get("noise_method", "flow_noise") not in {
            "flow_noise",
            "flow_sde",
        }:
            raise ValueError(
                "Training requires flow_noise or flow_sde stochastic transitions"
            )
        if cfg.get("is_lora", False):
            raise ValueError(
                "X-VLA uses xvla.update_scope; factory LoRA is unsupported"
            )
        self._set_trainable_scope(str(self.xcfg.get("update_scope", "prompt_heads")))
        feature_dim = int(core.config.florence_config.projection_dim)
        noise_std = self.xcfg.get("noise_std", ([0.003] * 3 + [0.01] * 6 + [0.15]) * 2)
        self.sampler = FlowSampler(
            method=str(self.xcfg.get("noise_method", "flow_noise")),
            steps=self.steps,
            action_dim=20,
            noise_std=noise_std,
            noise_min_ratio=float(self.xcfg.get("noise_min_ratio", 0.1)),
            noise_max_ratio=float(self.xcfg.get("noise_max_ratio", 2.0)),
            sde_noise_level=float(self.xcfg.get("sde_noise_level", 0.5)),
            hidden_dim=int(self.xcfg.get("noise_hidden_dim", 128)),
            context_dim=feature_dim,
        )
        self.add_value_head = bool(cfg.get("add_value_head", False))
        if self.add_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim + 20, 256), nn.Tanh(), nn.Linear(256, 1)
            )
        self._no_split_modules = ["SoftPromptedTransformer"]
        self.global_step = 0

    def _set_trainable_scope(self, scope: str) -> None:
        if scope not in {"prompt_heads", "action_head", "full"}:
            raise ValueError(f"Expected prompt_heads, action_head or full; got {scope}")
        self.core.requires_grad_(scope == "full")
        if scope == "action_head":
            self.core.transformer.requires_grad_(True)
        elif scope == "prompt_heads":
            transformer = self.core.transformer
            for name in ("action_encoder", "action_decoder"):
                layer = getattr(transformer, name)
                for part in ("fc", "bias"):
                    embedding = getattr(layer, part)
                    if not isinstance(embedding, nn.Embedding):
                        raise TypeError(f"Expected DomainAwareLinear.{part} embedding")
                    setattr(layer, part, DomainEmbedding(embedding, self.domain_id))
            transformer.soft_prompt_hub = DomainEmbedding(
                transformer.soft_prompt_hub, self.domain_id
            )

    @classmethod
    def from_checkpoint(cls, cfg: Any, torch_dtype=None) -> XVLAPolicy:
        """Load local model assets only after checking pinned file hashes."""
        from xvla_rlinf.vendor.xvla.modeling_xvla import XVLA
        from xvla_rlinf.vendor.xvla.processing_xvla import XVLAProcessor

        path = Path(cfg.model_path).expanduser()
        receipt_path = path / "verified.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Expected pinned checkpoint receipt at {receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("repo_id"), receipt.get("revision")) != (
            MODEL_REPO,
            MODEL_REVISION,
        ):
            raise ValueError(f"Expected {MODEL_REPO}@{MODEL_REVISION}")
        for item in receipt.get("files", []):
            if not isinstance(item, dict) or "sha256" not in item:
                raise ValueError("Checkpoint receipt requires per-file SHA256 entries")
            relative = item.get("file", item.get("path", item.get("name", "")))
            file_path = (path / relative).resolve()
            if not file_path.is_relative_to(path.resolve()) or not file_path.is_file():
                raise ValueError(f"Invalid checkpoint asset {relative!r}")
            with file_path.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Checkpoint asset hash mismatch: {relative}")
        if not receipt.get("files"):
            raise ValueError("Expected nonempty checkpoint file hashes")
        core = XVLA.from_pretrained(
            str(path),
            torch_dtype=torch_dtype or torch.float32,
            trust_remote_code=False,
            local_files_only=True,
        )
        processor = XVLAProcessor.from_pretrained(
            str(path), trust_remote_code=False, local_files_only=True
        )
        model = cls(core, processor, cfg)
        return model.to(dtype=torch_dtype or torch.float32)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def _device_inputs(self, inputs: dict) -> dict[str, torch.Tensor]:
        parameter = next(self.core.parameters())
        result = {}
        for key in INPUT_KEYS:
            value = torch.as_tensor(inputs[key])
            result[key] = value.to(
                device=parameter.device,
                dtype=parameter.dtype if value.is_floating_point() else value.dtype,
            )
        return result

    def _inputs(self, obs: dict) -> dict[str, torch.Tensor]:
        from PIL import Image

        main = torch.as_tensor(obs["main_images"]).cpu().numpy()
        wrists = torch.as_tensor(obs["wrist_images"]).cpu().numpy()
        states = torch.as_tensor(obs["states"])
        batch = len(states)
        if states.shape != (batch, 20) or wrists.shape[:2] != (batch, 2):
            raise ValueError("Expected measured states [B,20] and two wrist cameras")
        processed = []
        for index in range(batch):
            images = [Image.fromarray(main[index])]
            images.extend(Image.fromarray(im) for im in wrists[index])
            processed.append(self.processor(images, obs["task_descriptions"][index]))
        # Florence token sequences may differ across tasks in one rollout batch.
        ids = [torch.as_tensor(item["input_ids"]).squeeze(0) for item in processed]
        pad_id = int(
            getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", 1) or 1
        )
        inputs = {
            "input_ids": nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=pad_id
            ),
            "image_input": torch.cat(
                [torch.as_tensor(p["image_input"]) for p in processed]
            ),
            "image_mask": torch.cat(
                [torch.as_tensor(p["image_mask"]) for p in processed]
            ),
            "domain_id": torch.full((batch,), self.domain_id, dtype=torch.long),
            "proprio": states,
        }
        return self._device_inputs(inputs)

    def _encode(self, inputs: dict) -> dict:
        # eval disables dropout while preserving autograd during recomputation.
        self.core.eval()
        return self.core.forward_vlm(
            inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
        )

    def _predictor(self, inputs: dict, encoded: dict):
        def predict(noisy, time):
            proprio, action = self.core.action_space.preprocess(
                inputs["proprio"], noisy
            )
            kwargs = dict(
                domain_id=inputs["domain_id"],
                action_with_noise=action,
                proprio=proprio,
                t=time,
                **encoded,
            )
            if (
                self.xcfg.get("checkpoint_transformer", True)
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                return checkpoint(self.core.transformer, use_reentrant=False, **kwargs)
            return self.core.transformer(**kwargs)

        return predict

    @staticmethod
    def _context(encoded: dict) -> torch.Tensor:
        return encoded["vlm_features"].float().mean(dim=1)

    def _values(self, encoded: dict, inputs: dict, enabled: bool) -> torch.Tensor:
        if not self.add_value_head or not enabled:
            return torch.zeros(len(inputs["proprio"]), device=inputs["proprio"].device)
        features = torch.cat(
            (self._context(encoded), inputs["proprio"].float()), dim=-1
        )
        return self.value_head(features).squeeze(-1).float()

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict, mode=None, compute_values=True, **kwargs
    ):
        if mode not in {"train", "eval"}:
            raise ValueError("X-VLA requires explicit rollout mode train/eval")
        if mode == "eval" and self.xcfg.get("eval_sampler", "flow_ode") != "flow_ode":
            raise ValueError(
                "RL deployment evaluation uses the matching flow_ode sampler"
            )
        inputs = self._inputs(env_obs)
        encoded = self._encode(inputs)
        parameter = inputs["proprio"]
        initial_noise = kwargs.get("initial_noise")
        if initial_noise is None:
            initial_noise = torch.randn(
                (len(parameter), self.core.num_actions, 20),
                device=parameter.device,
                dtype=parameter.dtype,
            )
        sampled = self.sampler.sample(
            self._predictor(inputs, encoded),
            initial_noise,
            parameter,
            context=self._context(encoded),
            deterministic=mode == "eval",
        )
        raw = sampled["actions"][:, : self.num_action_chunks].detach()
        actions = self.core.action_space.postprocess(raw.clone()).float().cpu().numpy()
        cache = {key: value.detach().clone() for key, value in inputs.items()}
        cache.update(
            chains=sampled["chains"].detach(),
            action=torch.as_tensor(actions).flatten(1),
            model_action=raw.flatten(1),
        )
        return actions, {
            "prev_logprobs": pack_joint_logprob(
                sampled["logprobs"], self.num_action_chunks
            ),
            "prev_values": self._values(encoded, inputs, compute_values)[:, None],
            "forward_inputs": cache,
        }

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError("X-VLA supports on-policy PPO/GRPO forward only")
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_values=False,
        compute_entropy=False,
        **kwargs,
    ):
        if not compute_logprobs:
            raise ValueError("On-policy X-VLA forward must recompute the path density")
        inputs = self._device_inputs(forward_inputs)
        encoded = self._encode(inputs)
        stats = self.sampler.recompute(
            self._predictor(inputs, encoded),
            forward_inputs["chains"],
            inputs["proprio"],
            context=self._context(encoded),
        )
        logprobs = pack_joint_logprob(stats["logprobs"], self.num_action_chunks)
        entropy = pack_joint_logprob(stats["entropy"], self.num_action_chunks)
        mask = forward_inputs.get("executed_mask")
        if mask is not None:
            if mask.shape != logprobs.shape[:2] or bool(
                (mask[:, 1:] & ~mask[:, :-1]).any()
            ):
                raise ValueError("Expected an executed prefix mask [B,C]")
            logprobs = torch.where(mask[..., None].bool(), logprobs, 0.0)
            entropy = torch.where(mask[..., None].bool(), entropy, 0.0)
        return {
            "logprobs": logprobs,
            "values": self._values(encoded, inputs, compute_values),
            "entropy": entropy if compute_entropy else None,
        }
