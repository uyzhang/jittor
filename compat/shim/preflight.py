"""Import-time environment preparation for the Torch compatibility runtime.

This module is deliberately stdlib-only: :mod:`jittor.__init__` imports it
before the compiler and native core exist.
"""

from __future__ import absolute_import

import glob
import hashlib
import os
import pathlib
import sys
import tempfile
import importlib.util
from ..diagnostics import EXPECTED, swallowed


_TRUTHY = frozenset(("1", "true", "yes", "on"))


def require_independent_frontend(independent_namespace=True, environ=None):
    """Reject the removed frontend before environment or installation writes."""
    if independent_namespace is not True:
        raise RuntimeError(
            "legacy Jittor-alias Torch installation was removed; "
            "use activate() and import torch, leaving native jittor unchanged")
    env = os.environ if environ is None else environ
    requested = env.get("JITTOR_TORCH_INDEPENDENT")
    if requested is not None and not is_truthy(requested):
        raise RuntimeError(
            "JITTOR_TORCH_INDEPENDENT=%r requests the removed legacy frontend; "
            "remove this environment variable and import torch" % requested)
_DRIVER_LIBRARY_CANDIDATES = (
    "/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
)


class PreflightResult:
    __slots__ = (
        "active", "project_root", "runtime_root", "real_home", "jtcuda", "trigger"
    )

    def __init__(
        self,
        active,
        project_root="",
        runtime_root="",
        real_home="",
        jtcuda="",
        trigger="",
    ):
        self.active = bool(active)
        self.project_root = project_root
        self.runtime_root = runtime_root
        self.real_home = real_home
        self.jtcuda = jtcuda
        self.trigger = trigger


def is_truthy(value):
    return str(value or "").strip().lower() in _TRUTHY


def package_root():
    return pathlib.Path(__file__).resolve().parent


def resources_root():
    return package_root() / "resources"


def jittor_python_root():
    spec = importlib.util.find_spec("jittor")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if not locations:
        raise RuntimeError("cannot locate the installed jittor package")
    return pathlib.Path(locations[0]).resolve().parent


def project_dir(path=None):
    value = pathlib.Path.cwd() if path is None else pathlib.Path(os.fspath(path)).expanduser()
    if value.exists() and value.is_file():
        return value.parent.resolve()
    if str(value).endswith(".py"):
        return value.parent.resolve()
    return value.resolve()


def project_runtime_root(project_root, environ=None):
    env = os.environ if environ is None else environ
    root = project_dir(project_root)
    cache_root = env.get("JITTOR_TORCH_CACHE_ROOT")
    if cache_root:
        base = pathlib.Path(cache_root).expanduser()
    else:
        xdg_cache = env.get("XDG_CACHE_HOME")
        if not xdg_cache:
            home = env.get("REAL_HOME") or env.get("HOME") or os.path.expanduser("~")
            xdg_cache = os.path.join(home, ".cache")
        base = pathlib.Path(xdg_cache).expanduser() / "jittor" / "torch-shim"
    safe_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in (root.name or "project")
    )[:64] or "project"
    digest = hashlib.sha256(os.fsencode(os.fspath(root))).hexdigest()[:16]
    return (base / (safe_name + "-" + digest)).resolve()


def _prepend_env_path(environ, name, path):
    value = os.fspath(path)
    current = [item for item in environ.get(name, "").split(os.pathsep) if item]
    environ[name] = os.pathsep.join([value] + [item for item in current if item != value])


def prepend_sys_path(path, after=None):
    """Put ``path`` ahead of everything already on ``sys.path``.

    Reserved for directories this layer owns: anything placed here precedes the
    standard library, so a stray ``types.py`` or ``copy.py`` inside it breaks
    the interpreter. Use :func:`append_sys_path` for directories that belong to
    the user's project.
    """
    value = os.fspath(path)
    if not value:
        return
    if value in sys.path:
        sys.path.remove(value)
    if after is not None:
        marker = os.fspath(after)
        if marker in sys.path:
            sys.path.insert(sys.path.index(marker) + 1, value)
            return
    sys.path.insert(0, value)


def append_sys_path(path):
    """Make ``path`` importable without letting it shadow anything.

    The shim used to insert the *project* directory at ``sys.path[0]``, ahead of
    the standard library and of Jittor's own package root. A project holding a
    file named after a stdlib module -- ``types.py`` and ``copy.py`` are the
    ones seen in practice, and both are imported by Jittor itself -- then broke
    the interpreter from the moment ``enable()`` ran, with a traceback pointing
    anywhere but at the shim. A project directory only has to be *reachable*;
    it has no claim to precedence, so it goes at the end.
    """
    value = os.fspath(path)
    if not value or value in sys.path:
        return
    sys.path.append(value)


_UNWRITABLE_HINT = (
    "This path is derived from HOME/XDG_CACHE_HOME. On a read-only or "
    "unwritable HOME, point the shim somewhere writable with "
    "JITTOR_TORCH_CACHE_ROOT=<dir> (moves every project's runtime) or "
    "JITTOR_TORCH_RUNTIME_ROOT=<dir> (this project only). Set "
    "JITTOR_TORCH_SHIM=0 to start Jittor without the torch runtime at all."
)


def _ensure_dir(path, purpose=None):
    """Create a directory the shim needs, or say what it was for and how to move it.

    This runs during ``import jittor``, before the compiler and the native core
    exist. A bare ``PermissionError: '/home/x/.cache/jittor'`` from here reads
    as a Jittor installation problem, names no cause and offers no fix -- while
    the actual cause is that the torch shim decided to put its runtime under an
    unwritable HOME.
    """
    value = pathlib.Path(path)
    what = (" (%s)" % purpose) if purpose else ""
    try:
        value.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise type(exc)(
            "Jittor's torch compatibility layer could not create its runtime "
            "directory %s%s: %s. %s" % (value, what, exc, _UNWRITABLE_HINT)
        ) from exc
    if not os.access(os.fspath(value), os.W_OK):
        raise PermissionError(
            "Jittor's torch compatibility layer needs to write to %s%s, but "
            "that directory is not writable. %s" % (value, what, _UNWRITABLE_HINT)
        )
    return value


#: What a caller may still append to $TMPDIR and expect to work.
#:
#: A unix socket path is capped at 108 bytes including the terminator by
#: ``sockaddr_un.sun_path``, and the error a program gets for exceeding it is
#: not about length -- vLLM-Omni's orchestrator simply dies at startup. The
#: names that land here are of the form ``<tmpdir>/ipc_<uuid4hex>``, so 45
#: bytes of headroom covers them with room to spare.
_TMPDIR_SOCKET_HEADROOM = 45
_SUN_PATH_MAX = 107


def _short_enough_tmpdir(preferred):
    """The isolated tmpdir, or a short stand-in when that one cannot hold a socket.

    The shim points $TMPDIR inside its per-project runtime so a run leaves
    nothing behind. That directory is nested deep enough to be 90+ bytes on a
    normal install, which leaves no room for a socket name and is why
    deployments set ``JITTOR_TORCH_KEEP_TMPDIR=1`` -- a variable nobody can be
    expected to know about, whose absence shows up as a startup that dies
    without saying why.

    Measure instead. When the preferred path has room, use it. When it does
    not, fall back to a short directory that is still per-project, rather than
    to the caller's real $TMPDIR, so the isolation survives.
    """
    preferred = pathlib.Path(preferred)
    if len(os.fsencode(os.fspath(preferred))) + _TMPDIR_SOCKET_HEADROOM <= _SUN_PATH_MAX:
        return preferred
    digest = hashlib.sha1(os.fspath(preferred).encode("utf-8")).hexdigest()[:10]
    fallback = pathlib.Path(tempfile.gettempdir()) / ("jt-%s" % digest)
    if len(os.fsencode(os.fspath(fallback))) + _TMPDIR_SOCKET_HEADROOM <= _SUN_PATH_MAX:
        return fallback
    # tempfile.gettempdir() may itself be a long caller-provided TMPDIR. The
    # hashed fallback adds another path component, so recheck the complete path
    # before handing it to libraries that create AF_UNIX sockets beneath it.
    if os.name == "posix":
        fallback = pathlib.Path("/tmp") / ("jt-%s" % digest)
        if len(os.fsencode(os.fspath(fallback))) + _TMPDIR_SOCKET_HEADROOM <= _SUN_PATH_MAX:
            return fallback
    raise RuntimeError(
        "Jittor's Torch compatibility temporary directory cannot leave enough "
        "pathname space for a Unix socket: %s" % fallback
    )


def _set_env_dir(environ, name, path, override=False):
    if override or not environ.get(name):
        environ[name] = os.fspath(path)
    _ensure_dir(environ[name], "the directory $%s points at" % name)


#: The pre-2.22 unprefixed names, which still configure the build.
#:
#: This module runs before ``jittor_utils`` is importable -- it is what decides
#: whether the import can proceed at all -- so it cannot go through
#: ``env_config`` like every other reader, and
#: ``tests/structure/test_env_var_manifest.py`` names it as the one exception.
#: Keeping the two names in constants keeps the exception to one place.
_DEPRECATED_TIKCC_PATH = "tikcc_path"
_DEPRECATED_NVCC_FLAGS = "nvcc_flags"


def _remove_strict_math_nvcc_flags(value):
    strict_tokens = {"--fmad=false", "--prec-div=true", "--prec-sqrt=true"}
    tokens = [token for token in str(value or "").split()
              if token not in strict_tokens]
    return " " + " ".join(tokens) + " " if tokens else ""


def _acl_environment(environ):
    return bool(
        environ.get("ASCEND_TOOLKIT_HOME")
        or environ.get("ASCEND_HOME_PATH")
        # 2.22 gave build settings the JT_BUILD_ namespace; the old unprefixed
        # name still configures the toolchain, so both have to be read here.
        or environ.get("JT_BUILD_TIKCC_PATH")
        or environ.get(_DEPRECATED_TIKCC_PATH)
    )


def _prepare_kernel_math(environ):
    # The canonical names (see ``jittor_utils/env_config.py``). Spelled out
    # rather than imported because this module is loaded by the shim bootstrap
    # before ``jittor_utils`` is importable, and read back by the structure gate
    # without importing anything.
    if _acl_environment(environ):
        environ["JT_BUILD_NVCC_FLAGS"] = _remove_strict_math_nvcc_flags(
            environ.get("JT_BUILD_NVCC_FLAGS",
                        environ.get(_DEPRECATED_NVCC_FLAGS, ""))
        )
        environ["JT_CUDA_KERNEL_MATH"] = "backend"
        return
    # Kernel policy belongs in operator keys, not in the core build fingerprint.
    # Strict is also the core's default; keeping fast math is an explicit opt-in.
    if is_truthy(environ.get("JITTOR_TORCH_KEEP_FAST_MATH")):
        environ["JT_CUDA_KERNEL_MATH"] = "fast"
        return
    environ["JT_CUDA_KERNEL_MATH"] = "strict"


def configure_torch_math_flags(jittor_module):
    compiler = getattr(jittor_module, "compiler", None)
    flags = jittor_module.flags
    if getattr(compiler, "has_acl", False):
        os.environ["JT_BUILD_NVCC_FLAGS"] = _remove_strict_math_nvcc_flags(
            os.environ.get("JT_BUILD_NVCC_FLAGS",
                           os.environ.get(_DEPRECATED_NVCC_FLAGS, ""))
        )
        flags.cuda_kernel_math = "backend"
        return
    _prepare_kernel_math(os.environ)
    if is_truthy(os.environ.get("JITTOR_TORCH_KEEP_FAST_MATH")):
        flags.cuda_kernel_math = "fast"
        return
    flags.cuda_kernel_math = "strict"


def _best_jtcuda(environ, real_home):
    candidates = []
    if environ.get("JTCUDA"):
        candidates.append(environ["JTCUDA"])
    for home in (real_home, environ.get("HOME")):
        if home:
            candidates.extend(
                glob.glob(os.path.join(home, ".cache", "jittor", "jtcuda", "cuda*_linux"))
            )
    valid = []
    for candidate in dict.fromkeys(candidates):
        path = pathlib.Path(candidate)
        if (path / "bin" / "nvcc").is_file():
            valid.append(path)
    if not valid:
        return None
    valid.sort(
        key=lambda path: (
            not (path / "include" / "cudnn.h").is_file(),
            "cuda12.2" not in path.name,
            os.fspath(path),
        )
    )
    return valid[0]


def _configure_cuda(environ, real_home):
    if is_truthy(environ.get("JITTOR_TORCH_KEEP_CUDA")):
        return None
    if environ.get("JT_BUILD_NVCC_PATH") is not None or environ.get("nvcc_path") is not None:
        return None
    nvcc_exe = "nvcc.exe" if os.name == "nt" else "nvcc"
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = environ.get(env_name)
        if cuda_home:
            nvcc_candidate = pathlib.Path(cuda_home) / "bin" / nvcc_exe
            if nvcc_candidate.is_file():
                environ.setdefault("JT_BUILD_NVCC_PATH", os.fspath(nvcc_candidate))
                return None
    jtcuda = _best_jtcuda(environ, real_home)
    if jtcuda is None:
        return None
    environ.setdefault("JTCUDA", os.fspath(jtcuda))
    environ.setdefault("JT_BUILD_NVCC_PATH", os.fspath(jtcuda / "bin" / "nvcc"))
    environ.setdefault("CUDA_HOME", os.fspath(jtcuda))
    _prepend_env_path(environ, "PATH", jtcuda / "bin")
    _prepend_env_path(environ, "LD_LIBRARY_PATH", jtcuda / "lib64")
    _prepend_env_path(environ, "LIBRARY_PATH", jtcuda / "lib64")
    return jtcuda


def configure_runtime_driver_lib(runtime, environ=None):
    env = os.environ if environ is None else environ
    candidates = tuple(pathlib.Path(path) for path in _DRIVER_LIBRARY_CANDIDATES)
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return
    lib_dir = _ensure_dir(pathlib.Path(runtime) / "lib")
    target = lib_dir / "libcuda.so"
    try:
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                target.unlink()
        if not target.exists():
            target.symlink_to(source)
    except OSError:
        return
    _prepend_env_path(env, "LD_LIBRARY_PATH", lib_dir)
    _prepend_env_path(env, "LIBRARY_PATH", lib_dir)
    env.setdefault("TRITON_LIBCUDA_PATH", os.fspath(lib_dir))


#: Environment variables a launcher sets to say how many ranks it started:
#: Open MPI (``mpirun``), MPICH and Slurm, jittor's own MPI-free rendezvous,
#: and torchrun. Every one of them carries a *size*; ``PMIX_RANK`` and friends
#: are deliberately absent because a rank cannot answer "how many are there".
_WORLD_SIZE_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "JT_NCCL_WORLD_SIZE",
    "WORLD_SIZE",
)

#: Written next to ``JT_BUILD_USE_NCCL`` / ``JT_BUILD_USE_MPI`` whenever
#: *preflight* chose its value, holding the value it chose.
_NCCL_CHOSEN_BY_SHIM = "JITTOR_TORCH_SHIM_CHOSE_NCCL"
_MPI_CHOSEN_BY_SHIM = "JITTOR_TORCH_SHIM_CHOSE_MPI"


def _launched_with_multiple_ranks(env):
    """Whether this process is one rank of a multi-rank job.

    Read from the environment because it has to be answered *before* the core
    is imported -- ``setup_nccl()`` and ``setup_mpi()`` read these once during
    that import. ``jt.world_size`` does not exist yet here.
    """
    for name in _WORLD_SIZE_ENV:
        raw = env.get(name)
        if raw is None:
            continue
        try:
            if int(str(raw).strip()) > 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _choose_distributed_extern(env, canonical, legacy, marker):
    """Decide a distributed extern flag for *this* process without overruling
    a user-supplied value, and reconsidering only our own prior default.

    See :func:`_choose_nccl`: a single-rank launcher that defaults an extern to
    ``"0"`` stamps ``marker`` so child ranks spawned under ``mpirun`` or
    ``torchrun`` can flip it back to ``"1"``; any explicit user setting carries
    no matching marker and is preserved verbatim.
    """
    wanted = "1" if _launched_with_multiple_ranks(env) else "0"
    explicit_legacy = env.get(legacy)
    if explicit_legacy is not None and env.get(marker) != explicit_legacy:
        return explicit_legacy
    current = env.get(canonical)
    if current is not None and env.get(marker) != current:
        return current
    env[canonical] = wanted
    env[marker] = wanted
    return wanted


def _choose_nccl(env):
    """Decide ``JT_BUILD_USE_NCCL`` for *this* process, overruling only our own default."""
    return _choose_distributed_extern(
        env, "JT_BUILD_USE_NCCL", "use_nccl", _NCCL_CHOSEN_BY_SHIM
    )


def _choose_mpi(env):
    """Decide ``JT_BUILD_USE_MPI`` for *this* process, overruling only our own default."""
    return _choose_distributed_extern(
        env, "JT_BUILD_USE_MPI", "use_mpi", _MPI_CHOSEN_BY_SHIM
    )


def prepare_import_environment(
    argv=None,
    environ=None,
    project_root=None,
    runtime_root=None,
    force=False,
    local_home=False,
    configure_cuda=True,
):
    """Prepare the shim environment before native compiler/core import."""

    env = os.environ if environ is None else environ
    # ``argv`` remains accepted for source compatibility, but process mode must
    # never depend on an entry script's text or spelling.
    del argv
    explicit_project = (
        project_root
        if project_root is not None
        else env.get("JITTOR_TORCH_PROJECT_ROOT")
    )
    explicit_runtime = (
        runtime_root
        if runtime_root is not None
        else env.get("JITTOR_TORCH_RUNTIME_ROOT")
    )
    active = bool(
        force
        or explicit_project
        or explicit_runtime
        or is_truthy(env.get("JITTOR_TORCH_SHIM"))
    )
    if not active:
        return PreflightResult(False)

    require_independent_frontend(environ=env)

    trigger = (
        "forced"
        if force
        else "environment"
        if (
            explicit_project
            or explicit_runtime
            or is_truthy(env.get("JITTOR_TORCH_SHIM"))
        )
        else "explicit"
    )
    project = project_dir(explicit_project or pathlib.Path.cwd())
    runtime = pathlib.Path(
        os.fspath(explicit_runtime or project_runtime_root(project, env))
    ).expanduser().resolve()
    _ensure_dir(runtime, "the torch shim's per-project runtime root")
    real_home = env.get("REAL_HOME") or env.get("HOME") or os.path.expanduser("~")
    env.setdefault("REAL_HOME", real_home)
    env["JITTOR_TORCH_PROJECT_ROOT"] = os.fspath(project)
    env["JITTOR_TORCH_RUNTIME_ROOT"] = os.fspath(runtime)
    # The optional accelerator externs below stay off by default for shim runs:
    # measurements on cuDNN and cuBLAS show no benefit from cuTT or CUTLASS for
    # Torch compatibility workloads. Use canonical JT_BUILD_* names to avoid
    # deprecation warnings, while honoring any explicit legacy spelling.
    #
    # oneDNN is deliberately NOT in that list. Turning it off removes the
    # `mkl_conv` and `mkl_matmul` relays, so every CPU convolution falls back to
    # the generic reindex kernel: a 4x64x32x32 conv went from 1.4ms to 156ms and
    # a 512x512 matmul from 0.6ms to 11ms. That made ordinary CPU inference under
    # `import torch` unusable, which is the opposite of what the shim is for.
    #
    # DISABLE_MULTIPROCESSING is also NOT forced here: forcing it serialized
    # jittor_core's 260+ TU build onto a single core and forced DataLoader
    # num_workers=0 in ordinary user scripts.
    defaults = [
        ("JITTOR_TORCH_SHIM", "1"),
        ("FIX_TORCH_ERROR", "0"),
        ("use_cutlass", "0"),
    ]
    if env.get("JT_BUILD_USE_CUTT") is None and env.get("use_cutt") is None:
        env["JT_BUILD_USE_CUTT"] = "0"
    _choose_nccl(env)
    _choose_mpi(env)
    for name, value in defaults:
        env.setdefault(name, value)

    # Preserve the user's real HOME, XDG_CACHE_HOME (e.g. HuggingFace / model
    # caches), and PIP_CACHE_DIR by default -- matching PyTorch. Sandboxing HOME
    # into the per-project runtime is an explicit opt-in via local_home=True or
    # JITTOR_TORCH_ISOLATE_HOME=1 (and JITTOR_TORCH_KEEP_HOME=1 always wins).
    isolate_home = (
        bool(local_home) or is_truthy(env.get("JITTOR_TORCH_ISOLATE_HOME"))
    ) and not is_truthy(env.get("JITTOR_TORCH_KEEP_HOME"))

    # Reuse a shared user-level JITTOR_HOME across working directories by default
    # so changing cwd does not rebuild jittor_core from scratch; keep per-runtime
    # isolation when the caller explicitly requested runtime/cache/home isolation.
    isolate_jittor_home = bool(
        explicit_runtime or env.get("JITTOR_TORCH_CACHE_ROOT") or isolate_home
    )
    default_jittor_home = (
        runtime / "jittor_cache" if isolate_jittor_home else pathlib.Path(real_home)
    )
    _set_env_dir(env, "JITTOR_HOME", default_jittor_home)

    # Share compiled JIT kernels across projects via the CAS shared cache when
    # the user has not explicitly configured JITTOR_SHARED_CACHE (including "0").
    if env.get("JITTOR_SHARED_CACHE") is None:
        shared_cas = runtime.parent / "shared_cas"
        _set_env_dir(env, "JITTOR_SHARED_CACHE", shared_cas)

    shim_dirs = [
        ("TORCH_HOME", "torch_home"),
        ("JITTOR_TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("CUDA_CACHE_PATH", "cuda_cache"),
        ("TRITON_HOME", "triton_home"),
        ("TRITON_CACHE_DIR", "triton_home/cache"),
        ("TRITON_OVERRIDE_DIR", "triton_home/override"),
        ("TRITON_DUMP_DIR", "triton_home/dump"),
    ]
    if isolate_home:
        shim_dirs.extend([
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("PIP_CACHE_DIR", "pip_cache"),
        ])
    for name, subdir in shim_dirs:
        _set_env_dir(env, name, runtime / subdir)
    if isolate_home:
        env["HOME"] = env.get("JITTOR_TORCH_HOME", os.fspath(runtime / "home"))
        _ensure_dir(env["HOME"], "HOME for this process; "
                                "JITTOR_TORCH_KEEP_HOME=1 keeps the real one")
    _set_env_dir(
        env,
        "TMPDIR",
        _short_enough_tmpdir(runtime / "tmp"),
        override=isolate_home and not is_truthy(env.get("JITTOR_TORCH_KEEP_TMPDIR")),
    )
    _prepare_kernel_math(env)
    jtcuda = _configure_cuda(env, real_home) if configure_cuda else None
    configure_runtime_driver_lib(runtime, env)
    return PreflightResult(
        True,
        os.fspath(project),
        os.fspath(runtime),
        real_home,
        os.fspath(jtcuda) if jtcuda else "",
        trigger,
    )
