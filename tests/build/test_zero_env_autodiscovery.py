"""Regression tests for zero-env compiler autodiscovery and PyTorch-like defaults."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jittor as jt
from jittor_utils import cuda_wheel


class _FakeDistribution:
    def __init__(self, version, root):
        self.version = version
        self._root = Path(root)

    def locate_file(self, relative):
        return self._root / relative


class TestZeroEnvAutodiscovery(unittest.TestCase):
    def test_discover_cuda_compiler_reads_cuda_home_and_honors_cpu_override(self):
        from jittor.build import compiler

        with tempfile.TemporaryDirectory() as tmp:
            cuda_home = Path(tmp) / "cuda-sdk"
            nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text(
                "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n",
                encoding="utf-8",
            )
            nvcc.chmod(0o755)

            clean_env = {"CUDA_HOME": str(cuda_home)}
            with mock.patch.dict(os.environ, clean_env, clear=True), \
                 mock.patch.object(compiler.install_cuda, "has_installation", return_value=False), \
                 mock.patch.object(compiler.install_cuda, "install_cuda", return_value=""), \
                 mock.patch.object(compiler.install_cuda.cuda_wheel, "discover_wheel_nvcc", return_value=None):
                self.assertEqual(compiler._discover_cuda_compiler(None), str(nvcc))
                self.assertEqual(compiler._discover_cuda_compiler("cuda"), str(nvcc))
                self.assertEqual(compiler._discover_cuda_compiler("cpu"), "")

            # Explicit empty JT_BUILD_NVCC_PATH overrides CUDA_HOME to force CPU-only.
            with mock.patch.dict(
                os.environ, {"CUDA_HOME": str(cuda_home), "JT_BUILD_NVCC_PATH": ""}, clear=True
            ), mock.patch.object(compiler.install_cuda, "has_installation", return_value=True):
                self.assertEqual(compiler._discover_cuda_compiler(None), "")

    @unittest.skipUnless(os.name == "posix", "POSIX wheel path contract")
    def test_system_nvcc_does_not_treat_partial_cuda13_wheels_as_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-packages"
            (site / "nvidia" / "cu13").mkdir(parents=True)
            partial = {
                "nvidia-cuda-runtime": _FakeDistribution("13.0.48", site),
            }

            def dist(name):
                if name in partial:
                    return partial[name]
                raise cuda_wheel.importlib_metadata.PackageNotFoundError(name)

            with mock.patch.dict(os.environ, {"JITTOR_CUDA_WHEEL_DISABLE": ""}, clear=False):
                # System/CUDA_HOME nvcc falls back cleanly without JITTOR_CUDA_WHEEL_DISABLE=1.
                sys_report = cuda_wheel.inspect_cuda_wheel_stack(
                    "13.0.88", distribution=dist, nvcc_path="/usr/local/cuda-13.0/bin/nvcc"
                )
                self.assertIsNone(sys_report.stack)
                self.assertFalse(sys_report.broken)

                # A wheel-located nvcc still flags an incomplete CUDA 13 wheel set as broken.
                wheel_nvcc = str(site / "nvidia" / "cu13" / "bin" / "nvcc")
                wheel_report = cuda_wheel.inspect_cuda_wheel_stack(
                    "13.0.88", distribution=dist, nvcc_path=wheel_nvcc
                )
                self.assertIsNone(wheel_report.stack)
                self.assertTrue(wheel_report.broken)

    def test_setup_mpi_skips_single_rank_process_by_default(self):
        from jittor.build import compile_extern

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(compile_extern, "inside_mpi", return_value=False), \
             mock.patch.object(compile_extern, "env_or_try_find") as finder:
            prev_use = compile_extern.use_mpi
            prev_has = compile_extern.has_mpi
            try:
                compile_extern.setup_mpi()
                self.assertFalse(compile_extern.use_mpi)
                self.assertFalse(compile_extern.has_mpi)
                finder.assert_not_called()
            finally:
                compile_extern.use_mpi = prev_use
                compile_extern.has_mpi = prev_has


if __name__ == "__main__":
    unittest.main()
