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

"""Compare frozen X-VLA samplers on matched native RoboTwin initial states."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (12003, 13001)
MODES = ("expert", "native_measured", "native_history", "flow_ode", "flow_noise")


def _json(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value.tolist()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False) + "\n")


def trial_plan() -> list[dict]:
    """Declare every trial before observing any diagnostic outcome."""
    return [{"seed": seed, "mode": mode} for seed in SEEDS for mode in MODES]


def action_comparison(actions, reference) -> dict:
    """Compare physical XYZ, rotation-6D coordinates and discrete grippers."""
    import numpy as np

    actions, reference = np.asarray(actions), np.asarray(reference)
    xyz = (0, 1, 2, 10, 11, 12)
    rotation = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
    difference = actions - reference
    return {
        "xyz_rms": float(np.sqrt(np.mean(difference[..., xyz] ** 2))),
        "xyz_max_abs": float(np.abs(difference[..., xyz]).max()),
        "rotation6d_rms": float(np.sqrt(np.mean(difference[..., rotation] ** 2))),
        "gripper_threshold_disagreements": int(
            ((actions[..., (9, 19)] > 0.7) != (reference[..., (9, 19)] > 0.7)).sum()
        ),
    }


def _predictions(model, observation: dict, query: int) -> dict:
    import torch

    device = next(model.parameters()).device
    # Separate streams keep the same initial z across all four samplers.
    initial_seed = 1234 + query
    initial_generator = torch.Generator(device=device).manual_seed(initial_seed)
    initial = torch.randn((1, 30, 20), device=device, generator=initial_generator)
    inputs = model._inputs(observation)
    encoded = model._encode(inputs)
    predictor = model._predictor(inputs, encoded)
    context = model._context(encoded)
    output = {}
    for mode in ("flow_ode", "flow_noise"):
        sampled = model.sampler.sample(
            predictor,
            initial,
            inputs["proprio"],
            context=context,
            deterministic=mode == "flow_ode",
            generator=torch.Generator(device=device).manual_seed(1001234 + query),
        )
        output[mode] = model.core.action_space.postprocess(sampled["actions"].clone())
    with torch.random.fork_rng(devices=[device.index]):
        torch.cuda.set_rng_state(
            torch.Generator(device=device).manual_seed(initial_seed).get_state(), device
        )
        output["native"] = model.core.generate_actions(**inputs, steps=10)
    return {name: actions.float().cpu().numpy() for name, actions in output.items()}


def _child(config_path: str, seed: int, mode: str, folder: str) -> None:
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from rlinf.envs.sim.robotwin.xvla_env import XVLARoboTwinEnv
    from rlinf.models.embodiment.xvla.policy import XVLAPolicy
    from tools.xvla.robotwin import protocol
    from tools.xvla.robotwin.runner import screen_expert

    cfg = OmegaConf.load(config_path)
    destination = Path(folder).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1234)
    if os.environ.get("XVLA_LOGIN_DIAGNOSTIC") == "1":
        budget = int(os.environ.get("XVLA_LOGIN_MAX_TORCH_GIB", "10")) * 1024**3
        total = torch.cuda.get_device_properties(0).total_memory
        if not 0 < budget <= total:
            raise ValueError("Invalid login diagnostic PyTorch memory budget")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
    env_cfg.group_size = 1
    env_cfg.total_num_envs = 1
    env_cfg.reset_seeds = [seed]
    env_cfg.use_fixed_reset_state_ids = True
    env = XVLARoboTwinEnv(env_cfg, 1, 0, 1, None)
    started = time.monotonic()
    try:
        obs, reset_info = env.reset()
        scene = env._tasks[0].scene_signature()
        _write(destination / "initial-state.json", {"scene": scene, **reset_info})
        Image.fromarray(obs["main_images"][0].numpy()).save(destination / "initial.png")
        if mode == "expert":
            native = env._tasks[0]
            exclusion = screen_expert(native.env, env_cfg.task_name)
            _write(
                destination / "summary.json",
                {
                    "mode": mode,
                    "seed": seed,
                    "status": "COMPLETE",
                    "expert_success": exclusion is None,
                    "exclusion": exclusion,
                    "real_policy_environment_steps": 0,
                    "note": "Native scripted expert control; its motions are not policy training steps",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            return
        model = XVLAPolicy.from_checkpoint(cfg.actor.model).to("cuda:0").eval()
        with torch.no_grad(), (destination / "actions.jsonl").open("w") as journal:
            for query in range(10):
                predictions = _predictions(model, obs, query)
                actions = predictions["native" if mode.startswith("native_") else mode]
                row = {
                    "query": query,
                    "initial_noise_seed": 1234 + query,
                    "transition_noise_seed": 1001234 + query,
                    "proprio": obs["states"],
                    "executed_mode": mode,
                    "actions": predictions,
                    "native_vs_ode": action_comparison(
                        predictions["native"], predictions["flow_ode"]
                    ),
                    "noise_vs_ode": action_comparison(
                        predictions["flow_noise"], predictions["flow_ode"]
                    ),
                }
                next_obs, rewards, terminated, truncated, infos = env.chunk_step(
                    actions
                )
                row.update(
                    rewards=rewards,
                    executed_mask=infos[0]["executed_mask"],
                    episode=infos[0]["episode"],
                )
                journal.write(json.dumps(row, default=_json, allow_nan=False) + "\n")
                journal.flush()
                obs = next_obs[0]
                if mode == "native_history":
                    count = int(infos[0]["executed_mask"][0].sum())
                    if count:
                        command = protocol.actions_to_env(actions[0])[count - 1]
                        protocol.update_history(env._raw_obs[0], command)
                        obs = env._observations()
                if bool(terminated[0, -1] or truncated[0, -1]):
                    break
        Image.fromarray(obs["main_images"][0].numpy()).save(destination / "final.png")
        _write(
            destination / "summary.json",
            {
                "mode": mode,
                "seed": seed,
                "actual_reset_seeds": reset_info["reset_seeds"],
                "status": "COMPLETE",
                "episode": env._episode_info(),
                "real_policy_environment_steps": int(env.elapsed_steps.sum()),
                "termination": env._terminations,
                "truncation": env._truncations,
                "frozen_checkpoint": cfg.actor.model.model_path,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    finally:
        env.close()


def _run(cfg) -> None:
    from omegaconf import OmegaConf

    from tools.xvla.robotwin.runner import same_scene

    login_debug = (
        os.environ.get("XVLA_LOGIN_DIAGNOSTIC") == "1"
        and socket.gethostname() == "4090node1"
    )
    if not os.environ.get("SLURM_JOB_ID") and not login_debug:
        raise RuntimeError(
            "Use a Slurm allocation or the bounded login diagnostic launcher"
        )
    if (
        cfg.actor.model.model_type != "xvla"
        or cfg.actor.model.xvla.noise_method != "flow_noise"
    ):
        raise ValueError(
            "This paired diagnostic requires the unchanged X-VLA Flow-Noise configuration"
        )
    root = Path(os.environ["XVLA_RUN_DIR"]).resolve()
    config = root / "diagnostic-config.yaml"
    config.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    plan = trial_plan()
    result = {
        "status": "RUNNING",
        "hostname": socket.gethostname(),
        "login_diagnostic": login_debug,
        "plan": plan,
        "trials": [],
        "scope": "Frozen-model paired diagnosis; no optimizer or RL update",
        "limitations": [
            "Two fixed initial states, one paired initial-noise schedule per state",
            "Seeds reconstructed from earlier configuration/RNG; previous reset seeds were not logged",
            "Policy random schedule differs from earlier training; this does not replay those actions",
            "Native recurrence/history controls use the same 300-step budget and task text as RL",
        ],
    }
    target = root / "sampler-diagnostic.json"
    if target.exists():
        raise FileExistsError("Choose a new sampler diagnostic result directory")
    _write(target, result)

    def interrupted(signum, frame):
        del frame
        result.update(status="INTERRUPTED", signal=signum)
        _write(target, result)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for trial in plan:
        folder = root / f"{trial['seed']}-{trial['mode']}"
        command = [
            sys.executable,
            "-m",
            "tools.xvla.sampler_diagnostic",
            "--child",
            str(config),
            str(trial["seed"]),
            trial["mode"],
            str(folder),
        ]
        try:
            with (root / f"{folder.name}.log").open("w") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=210,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            result.update(
                status="FAILED", failed_trial=trial, error="Child exceeded 210 seconds"
            )
            _write(target, result)
            raise
        if completed.returncode:
            result.update(
                status="FAILED", failed_trial=trial, exit_code=completed.returncode
            )
            _write(target, result)
            raise RuntimeError(f"Sampler diagnostic child failed: {trial}")
        summary = json.loads((folder / "summary.json").read_text())
        initial = json.loads((folder / "initial-state.json").read_text())
        reference = next(
            (entry for entry in result["trials"] if entry["seed"] == trial["seed"]),
            None,
        )
        if reference is not None:
            previous = json.loads(
                (
                    root
                    / f"{reference['seed']}-{reference['mode']}"
                    / "initial-state.json"
                ).read_text()
            )
            if not same_scene(previous["scene"], initial["scene"]):
                result.update(
                    status="FAILED",
                    failed_trial=trial,
                    error="Paired initial scenes differ",
                )
                _write(target, result)
                raise RuntimeError("Paired diagnostic initial scenes differ")
        result["trials"].append(summary)
        _write(target, result)
    result["status"] = "COMPLETE"
    _write(target, result)


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--child", action="store_true")
        parser.add_argument("config")
        parser.add_argument("seed", type=int, choices=SEEDS)
        parser.add_argument("mode", choices=MODES)
        parser.add_argument("folder")
        args = parser.parse_args()
        _child(args.config, args.seed, args.mode, args.folder)
        return
    import hydra

    config_root = str(
        Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    )
    hydra.main(
        version_base="1.1", config_path=config_root, config_name="robotwin_grpo_xvla"
    )(_run)()


if __name__ == "__main__":
    main()
