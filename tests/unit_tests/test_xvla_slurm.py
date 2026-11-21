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

"""Slurm resource and visibility checks without starting Ray or GPU drivers."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
import warnings
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
CLUSTER_SOURCE = (ROOT / "rlinf/scheduler/cluster/cluster.py").read_text(
    encoding="utf-8"
)
NVIDIA_SOURCE = (
    ROOT / "rlinf/scheduler/hardware/accelerators/nvidia_gpu.py"
).read_text(encoding="utf-8")
ALLOCATION_ENV = "RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"


def _execute_source_nodes(nodes, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<slurm-source>", "exec"), namespace
    )


class SlurmRayTests(unittest.TestCase):
    def setUp(self):
        self.subprocess_env = os.environ.copy()
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.env = {
            "RLINF_XVLA_ISOLATED_RAY": "1",
            "SLURM_JOB_ID": "12345",
            "SLURM_CPUS_PER_TASK": "8",
            "CUDA_VISIBLE_DEVICES": "3,1",
            "RAY_TMPDIR": str(Path(self.temp) / "ray_12345"),
            "FYP_XVLA_RAY_SPILL": str(Path(self.temp) / "spill_12345"),
        }
        nodes = [
            node
            for node in ast.parse(CLUSTER_SOURCE).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_xvla_isolated_ray_kwargs"
        ]
        namespace = {"os": os, "re": re, "Path": Path}
        _execute_source_nodes(nodes, namespace)
        self.kwargs = namespace["_xvla_isolated_ray_kwargs"]

    def test_isolation_is_explicit_and_preserves_the_slurm_mask(self):
        self.assertIsNone(self.kwargs())
        with patch.dict(os.environ, self.env, clear=True):
            actual = self.kwargs()
            self.assertEqual(dict(os.environ), self.env)
        self.assertEqual(actual["address"], "local")
        self.assertEqual(actual["num_cpus"], 8)
        self.assertEqual(actual["num_gpus"], 2)
        self.assertEqual(actual["object_store_memory"], 2 * 1024**3)
        self.assertFalse(actual["include_dashboard"])
        spill = json.loads(actual["_system_config"]["object_spilling_config"])
        self.assertEqual(
            spill["params"]["directory_path"], self.env["FYP_XVLA_RAY_SPILL"]
        )

    def test_missing_allocation_or_private_paths_are_rejected(self):
        for missing in ("SLURM_JOB_ID", "RAY_TMPDIR", "FYP_XVLA_RAY_SPILL"):
            env = {key: value for key, value in self.env.items() if key != missing}
            with self.subTest(missing=missing), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    self.kwargs()

    def login_env(self):
        run_id = "login_test_12345"
        return {
            key: value
            for key, value in self.env.items()
            if not key.startswith("SLURM_")
        } | {
            "XVLA_LOGIN_TRAINING_DEBUG": "1",
            "XVLA_LOGIN_RUN_ID": run_id,
            "XVLA_LOGIN_CPUS": "4",
            "XVLA_LOGIN_SECONDS": "1200",
            "CUDA_VISIBLE_DEVICES": "GPU-16767e50-647f-d107-5073-8b2a27f3773a",
            "RAY_TMPDIR": str(
                Path(self.temp) / hashlib.sha256(run_id.encode()).hexdigest()[:16]
            ),
            "FYP_XVLA_RAY_SPILL": str(Path(self.temp) / f"spill_{run_id}"),
        }

    def test_login_debug_has_its_own_identity_without_fabricating_slurm(self):
        env = self.login_env()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("socket.gethostname", return_value="4090node1"),
        ):
            actual = self.kwargs()
            self.assertEqual(dict(os.environ), env)
        self.assertEqual(actual["address"], "local")
        self.assertEqual(actual["num_cpus"], 4)
        self.assertEqual(actual["num_gpus"], 1)
        self.assertEqual(actual["object_store_memory"], 2 * 1024**3)
        self.assertFalse(actual["include_dashboard"])
        for seconds in ("30", "1440", "1499"):
            with (
                patch.dict(
                    os.environ, env | {"XVLA_LOGIN_SECONDS": seconds}, clear=True
                ),
                patch("socket.gethostname", return_value="4090node1"),
            ):
                self.assertEqual(self.kwargs()["num_cpus"], 4)

    def test_login_debug_rejects_unbounded_or_ambiguous_context(self):
        invalid = [
            {"XVLA_LOGIN_TRAINING_DEBUG": "0"},
            {"SLURM_JOB_ID": "12345"},
            {"XVLA_LOGIN_RUN_ID": "../other"},
            {"XVLA_LOGIN_SECONDS": "0"},
            {"XVLA_LOGIN_SECONDS": "1500"},
            {"XVLA_LOGIN_CPUS": "8"},
            {"RLINF_XVLA_RAY_CPUS": "5"},
            {"CUDA_VISIBLE_DEVICES": "0,1"},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {"RAY_TMPDIR": str(Path(self.temp) / "other_run")},
        ]
        for override in invalid:
            with (
                self.subTest(override=override),
                patch.dict(os.environ, self.login_env() | override, clear=True),
                patch("socket.gethostname", return_value="4090node1"),
            ):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.kwargs()
        with (
            patch.dict(os.environ, self.login_env(), clear=True),
            patch("socket.gethostname", return_value="4090node2"),
        ):
            with self.assertRaises(RuntimeError):
                self.kwargs()

    def test_cpu_object_store_and_job_path_bounds_are_enforced(self):
        invalid = [
            {"RLINF_XVLA_RAY_CPUS": "9"},
            {"RLINF_XVLA_RAY_CPUS": "0"},
            {"RLINF_XVLA_RAY_OBJECT_STORE_BYTES": str(3 * 1024**3)},
            {"RLINF_XVLA_RAY_OBJECT_STORE_BYTES": "100"},
            {"RAY_TMPDIR": "relative_12345"},
            {"FYP_XVLA_RAY_SPILL": str(Path(self.temp) / "other-job")},
        ]
        for extra in invalid:
            with (
                self.subTest(extra=extra),
                patch.dict(os.environ, {**self.env, **extra}, clear=True),
            ):
                with self.assertRaises(ValueError):
                    self.kwargs()

    def test_driver_uses_a_private_instance_instead_of_an_available_shared_ray(self):
        tree = ast.parse(CLUSTER_SOURCE)
        constructor = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_init_and_launch_managers"
        )
        index = next(
            index
            for index, node in enumerate(constructor.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_xvla_isolated_ray_kwargs"
        )
        local_start = Mock()
        ray = SimpleNamespace(init=Mock())
        namespace = {
            "os": os,
            "_xvla_isolated_ray_kwargs": lambda: {"address": "local"},
            "self": SimpleNamespace(
                _num_nodes=1,
                _audited_worker_failure=None,
                _ray_code_sync_fragment=None,
                _start_local_ray=local_start,
            ),
            "ray": ray,
            "Cluster": SimpleNamespace(LOGGING_LEVEL="INFO", NAMESPACE="rlinf"),
        }
        _execute_source_nodes(constructor.body[index : index + 2], namespace)
        local_start.assert_called_once()
        self.assertEqual(local_start.call_args.args[0]["address"], "local")
        ray.init.assert_not_called()
        self.assertIsNone(namespace["self"]._audited_worker_failure)
        with patch.dict(os.environ, {"XVLA_ONLINE_ACCEPTANCE": "1"}):
            _execute_source_nodes(constructor.body[index : index + 2], namespace)
        self.assertEqual(namespace["self"]._audited_worker_failure, {})

    def test_detached_worker_refuses_auto_connect(self):
        tree = ast.parse(CLUSTER_SOURCE)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_init_from_existing_managers"
        )
        namespace = {"os": os, "ray": SimpleNamespace(is_initialized=lambda: False)}
        _execute_source_nodes([method], namespace)
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "refusing address=auto"):
                namespace["_init_from_existing_managers"](SimpleNamespace())

    def test_worker_failure_does_not_query_a_disabled_dashboard(self):
        method = next(
            node
            for node in ast.walk(ast.parse(CLUSTER_SOURCE))
            if isinstance(node, ast.FunctionDef) and node.name == "signal_handler"
        )
        logger = Mock()
        dashboard = Mock(side_effect=AssertionError("Dashboard must not be queried"))
        namespace = {
            "os": os,
            "sys": SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO()),
            "self": SimpleNamespace(
                _audited_worker_failure=None,
                _distributed_log_collector=None,
                _logger=logger,
            ),
            "list_actors": dashboard,
        }
        _execute_source_nodes([method], namespace)
        with patch.dict(os.environ, self.env, clear=True):
            with self.assertRaises(SystemExit) as stopped:
                namespace["signal_handler"](None, None)
        self.assertEqual(stopped.exception.code, 1)
        dashboard.assert_not_called()
        logger.error.assert_called_once()

    def test_audited_worker_failure_interrupts_once_with_original_cause(self):
        from rlinf.scheduler.cluster.cluster import Cluster, IsolatedWorkerFailure

        handler = next(
            node
            for node in ast.walk(ast.parse(CLUSTER_SOURCE))
            if isinstance(node, ast.FunctionDef) and node.name == "signal_handler"
        )
        cluster = SimpleNamespace(_audited_worker_failure={})
        namespace = {"self": cluster, "IsolatedWorkerFailure": IsolatedWorkerFailure}
        _execute_source_nodes([handler], namespace)
        with (
            patch.object(signal, "SIGUSR1", 10, create=True),
            patch.object(os, "kill") as interrupt,
        ):
            self.assertTrue(
                Cluster._interrupt_audited_worker_failure(
                    cluster, "AuditedEnv", "interact", RuntimeError("no Vulkan device")
                )
            )
            self.assertTrue(
                Cluster._interrupt_audited_worker_failure(
                    cluster,
                    "AuditedRollout",
                    "generate",
                    RuntimeError("channel closed"),
                )
            )
        interrupt.assert_called_once_with(os.getpid(), 10)
        with self.assertRaisesRegex(
            IsolatedWorkerFailure, "no Vulkan device"
        ) as raised:
            namespace["signal_handler"](10, None)
        self.assertEqual(raised.exception.worker_class, "AuditedEnv")
        self.assertEqual(raised.exception.method, "interact")
        self.assertEqual(raised.exception.error_type, "RuntimeError")

    def test_audited_worker_failure_records_reason_before_nonzero_exit(self):
        folder = Path(self.temp)
        script = """
import os
from pathlib import Path
from rlinf.scheduler.cluster.cluster import IsolatedWorkerFailure
from tools.xvla.online_acceptance import _record_failure

folder = Path(os.environ["XVLA_RUN_DIR"])
try:
    _record_failure(
        folder / "online-acceptance.json",
        IsolatedWorkerFailure("AuditedEnv", "interact", "RuntimeError", "no Vulkan device"),
        12.0,
    )
finally:
    (folder / "teardown-entered").touch()
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**self.subprocess_env, "XVLA_RUN_DIR": str(folder)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertFalse((folder / "teardown-entered").exists())
        self.assertTrue((folder / "online-acceptance.json").is_file(), proc.stderr)
        summary = json.loads((folder / "online-acceptance.json").read_text())
        event = json.loads(next(folder.glob("audit-*.jsonl")).read_text())
        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(event["event"], "acceptance_failed")
        for data in (summary, event):
            self.assertEqual(data["worker_class"], "AuditedEnv")
            self.assertEqual(data["method"], "interact")
            self.assertEqual(data["error_type"], "RuntimeError")
            self.assertEqual(data["error"], "no Vulkan device")


class SlurmGPUVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        functions = {
            "_xvla_parse_cuda_tokens",
            "_xvla_initialize_cuda_visibility",
            "_xvla_allocation_tokens",
            "_egl_index_by_cuda_device",
            "_renders_with_egl",
        }
        methods = {
            "get_num_devices",
            "get_visible_devices",
            "get_accelerator_env_var",
            "get_egl_device_id",
        }
        tree = ast.parse(NVIDIA_SOURCE)
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
        original = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NvidiaGPUManager"
        )
        nodes.append(
            ast.ClassDef(
                name="NvidiaGPUManager",
                bases=[],
                keywords=[],
                decorator_list=[],
                body=[
                    node
                    for node in original.body
                    if isinstance(node, ast.FunctionDef) and node.name in methods
                ],
            )
        )
        self.inventory = Mock(return_value=4)
        self.egl = Mock(return_value={0: 7, 1: 2})
        self.namespace = {
            "os": os,
            "cache": cache,
            "logger": logging.getLogger(__name__),
            "warnings": warnings,
            "_XVLA_ALLOCATION_ENV": ALLOCATION_ENV,
            "EGL_DEVICE_ID_ENV_VARS": ("MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID"),
            "_CPU_RENDERING_BACKENDS": frozenset({"osmesa", "glx"}),
            "NvidiaGPUAcceleratorManager": SimpleNamespace(
                get_current_node_num_accelerators=self.inventory
            ),
            "_query_egl_index_by_cuda_ordinal": self.egl,
            "_torch_needs_avoid_record_streams": lambda: False,
        }
        _execute_source_nodes(nodes, self.namespace)
        self.manager = self.namespace["NvidiaGPUManager"]

    def isolated(self, mask):
        os.environ.update(RLINF_XVLA_ISOLATED_RAY="1", CUDA_VISIBLE_DEVICES=mask)
        self.namespace["_xvla_initialize_cuda_visibility"]()

    def test_generic_model_visibility_is_unchanged(self):
        self.assertEqual(self.manager.get_num_devices(), 4)
        self.inventory.assert_called_once()
        os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"
        self.assertEqual(self.manager.get_visible_devices(), [3, 1])
        self.assertNotIn(ALLOCATION_ENV, os.environ)

    def test_nvml_cannot_expand_a_single_slurm_gpu_into_the_full_node(self):
        self.isolated("3")
        self.assertEqual(self.manager.get_num_devices(), 1)
        self.assertEqual(self.manager.get_visible_devices(), [0])
        self.assertEqual(os.environ[ALLOCATION_ENV], "3")
        self.assertEqual(os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"], "1")
        self.inventory.assert_not_called()

    def test_nonzero_discontiguous_gpu_tokens_and_egl_are_mapped_explicitly(self):
        self.isolated("3,1")
        worker = self.manager.get_accelerator_env_var(["1"])
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(worker[ALLOCATION_ENV], "3,1")
        self.assertEqual(worker["MUJOCO_EGL_DEVICE_ID"], "2")
        os.environ.update(worker)
        self.assertEqual(self.manager.get_visible_devices(), [1])
        self.assertEqual(self.manager.get_num_devices(), 2)
        self.egl.return_value = {0: 9}
        self.namespace["_egl_index_by_cuda_device"].cache_clear()
        self.assertEqual(self.manager.get_egl_device_id(1), 9)
        self.assertIsNone(self.manager.get_egl_device_id(0))

    def test_uuid_order_and_full_allocation_survive_worker_subsetting(self):
        self.isolated("GPU-aaaa1111-bbbb,GPU-cccc2222-dddd")
        worker = self.manager.get_accelerator_env_var(["1", "0"])
        self.assertEqual(
            worker["CUDA_VISIBLE_DEVICES"], "GPU-cccc2222-dddd,GPU-aaaa1111-bbbb"
        )
        os.environ.update(worker)
        self.assertEqual(self.manager.get_visible_devices(), [1, 0])
        self.assertEqual(self.manager.get_num_devices(), 2)

    def test_invalid_masks_and_out_of_allocation_placements_fail_closed(self):
        for value in ("0,0", "0,,1", "0,-1", "GPU-not-a-uuid", "all"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"RLINF_XVLA_ISOLATED_RAY": "1", "CUDA_VISIBLE_DEVICES": value},
                    clear=True,
                ),
            ):
                with self.assertRaises(ValueError):
                    self.manager.get_num_devices()
        self.isolated("2,3")
        for placement in (["2"], ["-1"], ["0", "0"], ["GPU-aaaa"], ["01"]):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                self.manager.get_accelerator_env_var(placement)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.manager.get_visible_devices()
        self.inventory.assert_not_called()

    def test_egl_failure_is_rejected_instead_of_guessing_a_gpu(self):
        self.isolated("3")
        self.egl.return_value = {}
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            self.manager.get_accelerator_env_var(["0"])
        os.environ["MUJOCO_GL"] = "osmesa"
        self.assertEqual(
            self.manager.get_accelerator_env_var(["0"])["CUDA_VISIBLE_DEVICES"], "3"
        )


if __name__ == "__main__":
    unittest.main()
