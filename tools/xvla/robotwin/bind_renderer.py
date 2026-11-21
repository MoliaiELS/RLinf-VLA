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

"""Bind SAPIEN 3.0.0b1 rendering to the allocated GPU."""

from __future__ import annotations

import os


def bind_renderer(*, pin_legacy_renderer: bool = False) -> None:
    """Bind scenes and optionally the legacy renderer to the selected GPU."""
    alias = os.environ.get("FYP_ROBOTWIN_VULKAN_DEVICE")
    if not alias:
        if os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Cannot use a default Vulkan device under Slurm")
        return
    import sapien
    import sapien.core as core

    create_scene = core.Engine.create_scene
    if getattr(create_scene, "_robotwin_device", None) == alias:
        device = create_scene._robotwin_render_device
    else:
        device = sapien.Device(alias)

        def create_scene(self, config=None):
            sapien.physx.set_scene_config(
                config if config is not None else core.SceneConfig()
            )
            return sapien.Scene(
                [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)]
            )

        create_scene._robotwin_device = alias
        create_scene._robotwin_render_device = device
        core.Engine.create_scene = create_scene

    if (
        pin_legacy_renderer
        and getattr(core.SapienRenderer.__init__, "_robotwin_device", None) != alias
    ):

        def init_renderer(self, **kwargs) -> None:
            # The legacy wrapper discards device arguments before reaching pybind.
            sapien.pysapien.render.SapienRenderer.__init__(self, device)

        init_renderer._robotwin_device = alias
        core.SapienRenderer.__init__ = init_renderer


def probe():
    # isort: off
    import numpy as np
    import sapien.core as core
    import torch
    from curobo.wrap.reacher.motion_gen import MotionGenConfig  # noqa: F401
    from curobo.curobolib import geom_cu
    # isort: on

    bind_renderer()
    engine = core.Engine()
    engine.set_renderer(core.SapienRenderer())
    scene = engine.create_scene(core.SceneConfig())
    camera = scene.add_camera("probe", 32, 24, 1.0, 0.1, 100)
    scene.step()
    scene.update_render()
    camera.take_picture()
    color = camera.get_picture("Color")
    if color.shape != (24, 32, 4) or not np.isfinite(color).all():
        raise ValueError(f"Invalid SAPIEN Vulkan frame: {color.shape}")
    assert torch.empty(1, device="cuda:0").is_cuda
    print(
        f"ROBOTWIN_RENDER_PROBE shape={color.shape} extension={geom_cu.__file__}",
        flush=True,
    )
