"""Getting hold of the cuTT ops, which are built on first use."""

from _helpers import capability as _test_capability

import os
import unittest

import jittor as jt

from jittor._runtime.backend_libraries import get_library_ops
from jittor_utils.env_config import build_flag


def require_cutt_ops():
    """Return the cuTT ops module, or skip if cuTT is disabled or unavailable.

    Reading ``compile_extern.cutt_ops`` reports ``None`` on a machine that has
    cuTT: the library is compiled the first time a CUDA transpose asks for it,
    not during import, so it has to be requested with ``load=True``. cuTT is
    opt-in by default (``JT_BUILD_USE_CUTT=1`` / ``use_cutt=1``) so standard
    runs do not download or compile a third-party transpose library.

    Call this from ``setUpClass``, not at module scope: collecting a test file
    must not compile a backend.
    """
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        raise unittest.SkipTest("no CUDA on this machine, cuTT cannot be built")
    if not build_flag("use_cutt", False, os.environ):
        raise unittest.SkipTest("cuTT is disabled (opt-in via JT_BUILD_USE_CUTT=1)")
    ops = get_library_ops("cutt", load=True)
    if ops is None:
        # Only two things can get us here now: CUDA is present and cuTT is
        # enabled, so a None means the build itself failed. Skipping on that
        # is how a broken cuTT build stays invisible -- transposes quietly fall
        # back to the built-in kernel and every cuTT test reports "skipped".
        raise AssertionError(
            "cuTT is enabled and CUDA is present, but the ops did not load: "
            "the cuTT build failed. Re-run with log_v=1 to see the compile "
            "command; a missing CUDA SDK include path is the usual cause.")
    return ops
