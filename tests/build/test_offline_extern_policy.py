"""Offline-safe and graceful-degradation contracts for external libraries."""

import ast
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import jittor as jt


ROOT = Path(__file__).resolve().parents[2]
EXTERN = ROOT / "python/jittor/build/compile_extern.py"


class TestOfflineSafeExternDeps(unittest.TestCase):
    def test_cutt_defaults_off_and_refuses_unasked_download(self):
        from jittor.build import compile_extern as ce

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(ce, "has_cuda", True), \
             mock.patch.object(ce, "is_cuda", True), \
             mock.patch.object(ce, "install_cutt",
                               side_effect=AssertionError("cutt should default off")):
            ce.setup_cutt()
            self.assertFalse(ce.use_cutt)

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(ce, "cache_path", tmp), \
             mock.patch.object(ce, "download_url_to_local",
                               side_effect=AssertionError("must not download cutt unasked")):
            with self.assertRaisesRegex(RuntimeError, "JT_AUTO_INSTALL_CUTT=1"):
                ce.install_cutt(tmp)

    def test_nccl_refuses_unasked_source_download_and_build(self):
        from jittor.build import compile_extern as ce

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(os.environ, {"JT_NCCL_WORLD_SIZE": "2"}, clear=True), \
             mock.patch.object(ce.core, "get_device_count", return_value=2), \
             mock.patch.object(ce, "download_url_to_local",
                               side_effect=AssertionError("must not download nccl unasked")):
            with self.assertRaisesRegex(RuntimeError, "JT_BUILD_NCCL_FROM_SOURCE=1"):
                ce.install_nccl(tmp)

    def test_setup_cuda_extern_degrades_gracefully_when_cudnn_missing_and_defers_cufft_cusparse(self):
        from jittor.build import compile_extern as ce

        called = []

        def fake_setup_cuda_lib(name, extra_flags=""):
            called.append(name)
            if name == "cudnn":
                raise RuntimeError("cudnn.h not found. Searched: /usr/include.")

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(ce, "has_cuda", True), \
             mock.patch.object(ce, "is_cuda", True), \
             mock.patch.object(ce, "compile_if_stale"), \
             mock.patch.object(ce.ctypes, "CDLL"), \
             mock.patch.object(ce, "register_library_resources"), \
             mock.patch.object(ce, "setup_cub"), \
             mock.patch.object(ce, "setup_cuda_lib", side_effect=fake_setup_cuda_lib):
            ce.setup_cuda_extern()

        # cublas + curand are eager required; cudnn is tried and degrades cleanly
        # when absent; cufft and cusparse remain lazy on first use.
        self.assertEqual(called, ["cublas", "curand", "cudnn"])

    def test_pyproject_cuda13_extra_includes_complete_math_and_comm_stack(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        # Extract the cuda13 list block and verify core CUDA 13 libraries exist.
        start = text.index("cuda13 = [")
        end = text.index("]", start)
        block = text[start:end]
        for pkg in (
            "nvidia-cuda-nvcc", "nvidia-cuda-runtime", "nvidia-cublas",
            "nvidia-curand", "nvidia-cufft", "nvidia-cusparse",
            "nvidia-cuda-nvrtc", "nvidia-cudnn-cu13", "nvidia-nccl-cu13",
        ):
            self.assertIn(pkg, block, f"cuda13 extra missing {pkg}")


if __name__ == "__main__":
    unittest.main()
