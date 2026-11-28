"""Native FFT operations shared by Jittor and the Torch compatibility layer.

The implementation uses differentiable real/imag DFT matrix operations so it
works on every backend supported by the surrounding native complex bridge.  A
Torch-mode process publishes this same module object as ``torch.fft``; it does
not install a second FFT implementation.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
from collections import OrderedDict

import jittor as jt
from jittor import _arg_policy
from jittor._core.var import device_scope_like
from jittor._runtime.backend_libraries import get_library, get_library_ops
from jittor._runtime.dispatch import dispatch_context


_ComplexNumber = jt.nn.ComplexNumber
_dft_mat_cache = OrderedDict()
_dft_mat_cache_limit = 16
_cufft_ops_cache = None
_cufft_ops_checked = False
_cufft_eager_rfft = None
_cufft_eager_irfft = None
_cufft_eager_c2c = None
_cufft_eager_fft_real = None
_cufft_eager_checked = False


def _get_cufft_ops():
    global _cufft_ops_cache, _cufft_ops_checked
    if not _cufft_ops_checked:
        _cufft_ops_checked = True
        _cufft_ops_cache = get_library_ops("cufft", load=True)
    return _cufft_ops_cache


def _ensure_cufft_eager():
    global _cufft_eager_rfft, _cufft_eager_irfft, _cufft_eager_c2c
    global _cufft_eager_fft_real, _cufft_eager_checked
    if not _cufft_eager_checked:
        _cufft_eager_checked = True
        lib = get_library("cufft", load=True)
        if lib is not None:
            _cufft_eager_rfft = getattr(lib, "cufft_eager_rfft_1d", None)
            _cufft_eager_irfft = getattr(lib, "cufft_eager_irfft_1d", None)
            _cufft_eager_c2c = getattr(lib, "cufft_eager_c2c_1d", None)
            _cufft_eager_fft_real = getattr(lib, "cufft_eager_fft_real_1d", None)


def _fft_norm_scale(length, inverse, norm):
    if norm == "ortho":
        return 1.0 / (length ** 0.5)
    if norm == "forward":
        return (1.0 / length) if not inverse else 1.0
    return (1.0 / length) if inverse else 1.0


def _zeros_on(shape, dtype, like):
    z = jt.zeros(shape, dtype)
    ctx = dispatch_context(like)
    if ctx.device_id >= 0:
        z = z.to_device(ctx.device_id)
    return z


_REAL_TO_PAIR_HEADER = r"""
template <typename T>
__global__ void _jt_real_to_pair_k(const T* __restrict__ xp, T* __restrict__ yp, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        yp[i * 2] = xp[i];
        yp[i * 2 + 1] = (T)0;
    }
}

template <typename T>
__global__ void _jt_pair_slice_scale_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t batch, int64_t full_len, int64_t keep_len, T scale
) {
    const int64_t total = batch * keep_len;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t b = idx / keep_len;
        const int64_t k = idx - b * keep_len;
        const int64_t src = (b * full_len + k) * 2;
        const int64_t dst = idx * 2;
        yp[dst] = xp[src] * scale;
        yp[dst + 1] = xp[src + 1] * scale;
    }
}

template <typename T>
__global__ void _jt_pair_to_real_k(const T* __restrict__ xp, T* __restrict__ yp, int64_t n, T scale) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        yp[i] = xp[i * 2] * scale;
    }
}

template <typename T>
__global__ void _jt_irfft_mirror_pair_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t batch, int64_t in_len, int64_t keep, int64_t full_len
) {
    const int64_t total = batch * full_len;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t b = idx / full_len;
        const int64_t k = idx - b * full_len;
        const int64_t dst = idx * 2;
        if (k < keep) {
            if (k < in_len) {
                const int64_t src = (b * in_len + k) * 2;
                yp[dst] = xp[src];
                yp[dst + 1] = xp[src + 1];
            } else {
                yp[dst] = (T)0;
                yp[dst + 1] = (T)0;
            }
        } else {
            const int64_t mk = full_len - k;
            if (mk < in_len) {
                const int64_t src = (b * in_len + mk) * 2;
                yp[dst] = xp[src];
                yp[dst + 1] = -xp[src + 1];
            } else {
                yp[dst] = (T)0;
                yp[dst + 1] = (T)0;
            }
        }
    }
}
"""


class _RealToPairCuda(jt.Function):
    def execute(self, real):
        rc = real.contiguous()
        out_shape = list(rc.shape) + [2]
        with device_scope_like(rc):
            return jt.code(
                out_shape,
                rc.dtype,
                [rc],
                cuda_header=_REAL_TO_PAIR_HEADER,
                cuda_src="""
                const int64_t n = (int64_t)in0->num;
                if (n > 0) {
                    constexpr int threads = 256;
                    const int64_t raw_blocks = (n + threads - 1) / threads;
                    const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                    _jt_real_to_pair_k<in0_type><<<blocks, threads>>>(in0_p, out0_p, n);
                    CHECK(0 == cudaGetLastError());
                }
                """,
            )

    def grad(self, dout):
        if dout is None:
            return None
        dc = dout.contiguous()
        out_shape = list(dc.shape[:-1])
        with device_scope_like(dc):
            return jt.code(
                out_shape,
                dc.dtype,
                [dc],
                cuda_header=_REAL_TO_PAIR_HEADER,
                cuda_src="""
                const int64_t n = (int64_t)out0->num;
                if (n > 0) {
                    constexpr int threads = 256;
                    const int64_t raw_blocks = (n + threads - 1) / threads;
                    const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                    _jt_pair_to_real_k<in0_type><<<blocks, threads>>>(in0_p, out0_p, n, (in0_type)1);
                    CHECK(0 == cudaGetLastError());
                }
                """,
            )


def _as_real_pair(value, size):
    """Convert real or complex ``value`` (with transform on last dim) to ``[..., L, 2]``."""
    is_native_complex = (
        isinstance(value, jt.Var)
        and "complex" in _jittor_dtype_name(value.dtype)
    )
    if is_native_complex:
        pair = jt.nn.view_as_real(value)
        if size is not None:
            length = pair.shape[-2]
            if length > size:
                pair = pair[..., :size, :]
            elif length < size:
                pad_shape = list(pair.shape[:-2]) + [size - length, 2]
                pair = jt.concat([pair, _zeros_on(pad_shape, pair.dtype, pair)], dim=-2)
        return pair
    real0, imag0 = _real_imag(value)
    real = _resize_last(real0, size)
    rname = _jittor_dtype_name(real.dtype)
    if rname not in ("float32", "float64"):
        real = real.float32()
    if imag0 is None:
        if dispatch_context(real).backend == "cuda":
            if jt.flags.no_grad:
                rc = real.contiguous()
                with device_scope_like(rc):
                    return jt.code(
                        list(rc.shape) + [2],
                        rc.dtype,
                        [rc],
                        cuda_header=_REAL_TO_PAIR_HEADER,
                        cuda_src="""
                        const int64_t n = (int64_t)in0->num;
                        if (n > 0) {
                            constexpr int threads = 256;
                            const int64_t raw_blocks = (n + threads - 1) / threads;
                            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                            _jt_real_to_pair_k<in0_type><<<blocks, threads>>>(in0_p, out0_p, n);
                            CHECK(0 == cudaGetLastError());
                        }
                        """,
                    )
            return _RealToPairCuda()(real)
        imag = _zeros_on(real.shape, real.dtype, real)
    else:
        imag = _resize_last(imag0, size)
        if _jittor_dtype_name(imag.dtype) != _jittor_dtype_name(real.dtype):
            imag = imag.cast(real.dtype)
    return jt.stack([real, imag], dim=-1)


def _cufft_1d(value, size, dim, inverse, norm=None, keep=None):
    ops = _get_cufft_ops()
    if ops is None:
        return None
    probe = value.real if isinstance(value, _ComplexNumber) else value
    with device_scope_like(probe):
        moved, restore = _to_last(value, dim)
        pair = _as_real_pair(moved, size)
        dtype_name = _jittor_dtype_name(pair.dtype)
        if dtype_name not in ("float32", "float64"):
            return None
        length = int(pair.shape[-2])
        if length <= 0 or pair.numel() == 0:
            return None
        batch = int(pair.numel() // (length * 2))
        out_pair = ops.cufft_fft(pair.reshape([batch, 1, length, 2]), bool(inverse))
        scale = _fft_norm_scale(length, inverse, norm)
        out_len = keep if (keep is not None and keep < length) else length
        if jt.flags.no_grad and (out_len < length or scale != 1.0):
            out_pair = jt.code(
                [batch, 1, out_len, 2],
                out_pair.dtype,
                [out_pair],
                cuda_header=_REAL_TO_PAIR_HEADER,
                cuda_src="""
                const int64_t batch = (int64_t)in0->shape[0];
                const int64_t full_len = (int64_t)in0->shape[2];
                const int64_t keep_len = (int64_t)out0->shape[2];
                const int64_t total = batch * keep_len;
                const in0_type scale = (in0_type)data["scale"];
                if (total > 0) {
                    constexpr int threads = 256;
                    const int64_t raw_blocks = (total + threads - 1) / threads;
                    const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                    _jt_pair_slice_scale_k<in0_type><<<blocks, threads>>>(
                        in0_p, out0_p, batch, full_len, keep_len, scale);
                    CHECK(0 == cudaGetLastError());
                }
                """,
                data={"scale": float(scale)},
            )
            out_pair = out_pair.reshape(list(pair.shape[:-2]) + [out_len, 2])
        else:
            out_pair = out_pair.reshape(list(pair.shape[:-2]) + [length, 2])
            if out_len < length:
                out_pair = out_pair[..., :out_len, :].contiguous()
            if scale != 1.0:
                out_pair = out_pair * scale
        output = jt.nn.view_as_complex(out_pair)
        if restore is not None:
            output = output.permute(*restore)
        return output


def _cufft_2d(value, sizes, dims, inverse, norm=None):
    ops = _get_cufft_ops()
    if ops is None or len(dims) != 2:
        return None
    ndim = value.real.ndim if isinstance(value, _ComplexNumber) else value.ndim
    if ndim < 2:
        return None
    d0 = dims[0] if dims[0] >= 0 else dims[0] + ndim
    d1 = dims[1] if dims[1] >= 0 else dims[1] + ndim
    if d0 == d1:
        return None
    probe = value.real if isinstance(value, _ComplexNumber) else value
    with device_scope_like(probe):
        if d0 == ndim - 2 and d1 == ndim - 1:
            moved, restore = value, None
        else:
            perm = [a for a in range(ndim) if a not in (d0, d1)] + [d0, d1]
            restore = [0] * ndim
            for new_a, old_a in enumerate(perm):
                restore[old_a] = new_a
            moved = value.permute(*perm)
        s0 = sizes[0] if sizes is not None else None
        s1 = sizes[1] if sizes is not None else None
        pair = _as_real_pair(moved, s1)
        if s0 is not None:
            pair = _resize_at(pair, pair.ndim - 3, s0)
        dtype_name = _jittor_dtype_name(pair.dtype)
        if dtype_name not in ("float32", "float64"):
            return None
        h = int(pair.shape[-3])
        w = int(pair.shape[-2])
        if h <= 0 or w <= 0 or pair.numel() == 0:
            return None
        batch = int(pair.numel() // (h * w * 2))
        out_pair = ops.cufft_fft(pair.reshape([batch, h, w, 2]), bool(inverse))
        out_pair = out_pair.reshape(list(pair.shape[:-3]) + [h, w, 2])
        scale = _fft_norm_scale(h * w, inverse, norm)
        if scale != 1.0:
            out_pair = out_pair * scale
        output = jt.nn.view_as_complex(out_pair.contiguous())
        if restore is not None:
            output = output.permute(*restore)
        return output


def _dft_mats(size, inverse, like):
    context = dispatch_context(like)
    key = (int(size), bool(inverse), context.backend, context.device_id)
    cached = _dft_mat_cache.get(key)
    if cached is not None:
        _dft_mat_cache.move_to_end(key)
        return cached
    indices = np.arange(size)
    angle = (2.0 * np.pi / size) * np.outer(indices, indices)
    if not inverse:
        angle = -angle
    matrices = (
        jt.array(np.cos(angle).astype("float32")),
        jt.array(np.sin(angle).astype("float32")),
    )
    if context.device_id >= 0:
        matrices = tuple(matrix.to_device(context.device_id) for matrix in matrices)
    _dft_mat_cache[key] = matrices
    if len(_dft_mat_cache) > _dft_mat_cache_limit:
        _dft_mat_cache.popitem(last=False)
    return matrices


def _to_last(value, dim):
    ndim = value.real.ndim if isinstance(value, _ComplexNumber) else value.ndim
    dimension = dim if dim >= 0 else dim + ndim
    if dimension == ndim - 1:
        return value, None
    permutation = [axis for axis in range(ndim) if axis != dimension] + [dimension]
    inverse = [0] * ndim
    for new_axis, old_axis in enumerate(permutation):
        inverse[old_axis] = new_axis
    return value.permute(*permutation), inverse


def _resize_last(value, size):
    if size is None:
        return value
    length = value.shape[-1]
    if length == size:
        return value
    if length > size:
        return value[..., :size]
    padding = _zeros_on(list(value.shape[:-1]) + [size - length], value.dtype, value)
    return jt.concat([value, padding], dim=-1)


def _make_complex(real, imag):
    return jt.nn.view_as_complex(jt.stack([real, imag], dim=-1))


def _real_imag(value):
    if isinstance(value, _ComplexNumber):
        return value.real, value.imag
    if isinstance(value, jt.Var) and "complex" in _jittor_dtype_name(value.dtype):
        return value.real, value.imag
    return value, None


def _fft_core(value, size, dim, inverse, norm=None):
    probe = value.real if isinstance(value, _ComplexNumber) else value
    if dispatch_context(probe).backend == "cuda":
        fast = _cufft_1d(value, size, dim, inverse, norm=norm)
        if fast is not None:
            return fast
    value, restore = _to_last(value, dim)
    real0, imag0 = _real_imag(value)
    real = _resize_last(real0, size)
    imag = _resize_last(imag0, size) if imag0 is not None else None
    length = real.shape[-1]
    cosine, sine = _dft_mats(length, inverse, real)

    out_real = jt.matmul(real, cosine.transpose(1, 0))
    out_imag = jt.matmul(real, sine.transpose(1, 0))
    if imag is not None:
        out_real = out_real - jt.matmul(imag, sine.transpose(1, 0))
        out_imag = out_imag + jt.matmul(imag, cosine.transpose(1, 0))

    scale = _fft_norm_scale(length, inverse, norm)
    if scale != 1.0:
        out_real = out_real * scale
        out_imag = out_imag * scale

    output = _make_complex(out_real, out_imag)
    if restore is not None:
        output = output.permute(*restore)
    return output


def fft(input, n=None, dim=-1, norm=None):
    if (n is None or (isinstance(input, jt.Var) and input.ndim > 0 and n == input.shape[-1])) and (
        dim == -1 or (isinstance(input, jt.Var) and dim == input.ndim - 1)
    ) and isinstance(input, jt.Var) and input.ndim > 0:
        if not _cufft_eager_checked:
            _ensure_cufft_eager()
        if _cufft_eager_c2c is not None:
            scale = _fft_norm_scale(int(input.shape[-1]), False, norm)
            dname = _jittor_dtype_name(input.dtype)
            if "complex" in dname:
                res = _cufft_eager_c2c(input, False, float(scale))
                if res is not None:
                    return res
            elif dname in ("float32", "float64") and _cufft_eager_fft_real is not None:
                res = _cufft_eager_fft_real(input, False, float(scale))
                if res is not None:
                    return res
    return _fft_core(input, n, dim, False, norm)


def ifft(input, n=None, dim=-1, norm=None):
    if (n is None or (isinstance(input, jt.Var) and input.ndim > 0 and n == input.shape[-1])) and (
        dim == -1 or (isinstance(input, jt.Var) and dim == input.ndim - 1)
    ) and isinstance(input, jt.Var) and input.ndim > 0:
        if not _cufft_eager_checked:
            _ensure_cufft_eager()
        if _cufft_eager_c2c is not None:
            scale = _fft_norm_scale(int(input.shape[-1]), True, norm)
            dname = _jittor_dtype_name(input.dtype)
            if "complex" in dname:
                res = _cufft_eager_c2c(input, True, float(scale))
                if res is not None:
                    return res
            elif dname in ("float32", "float64") and _cufft_eager_fft_real is not None:
                res = _cufft_eager_fft_real(input, True, float(scale))
                if res is not None:
                    return res
    return _fft_core(input, n, dim, True, norm)


def _fftn(input, sizes=None, dims=(-2, -1), norm=None, inverse=False):
    dimensions = list(dims)
    lengths = list(sizes) if sizes is not None else [None] * len(dimensions)
    probe = input.real if isinstance(input, _ComplexNumber) else input
    if len(dimensions) == 2 and dispatch_context(probe).backend == "cuda":
        fast = _cufft_2d(input, lengths, dimensions, inverse, norm=norm)
        if fast is not None:
            return fast
    output = input
    for dim, size in zip(dimensions, lengths):
        output = _fft_core(output, size, dim, inverse, norm)
    return output


def fft2(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, False)


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, True)


def fftn(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, False)


def ifftn(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, True)


def rfft(input, n=None, dim=-1, norm=None):
    if (n is None or (isinstance(input, jt.Var) and input.ndim > 0 and n == input.shape[-1])) and (
        dim == -1 or (isinstance(input, jt.Var) and dim == input.ndim - 1)
    ) and isinstance(input, jt.Var) and input.ndim > 0:
        if not _cufft_eager_checked:
            _ensure_cufft_eager()
        if _cufft_eager_rfft is not None:
            scale = _fft_norm_scale(int(input.shape[-1]), False, norm)
            res = _cufft_eager_rfft(input, float(scale))
            if res is not None:
                return res
    length = input.shape[dim] if n is None else n
    keep = length // 2 + 1
    probe = input.real if isinstance(input, _ComplexNumber) else input
    if dispatch_context(probe).backend == "cuda":
        fast = _cufft_1d(input, n, dim, False, norm=norm, keep=keep)
        if fast is not None:
            return fast
    full = _fft_core(input, n, dim, False, norm)
    real, imag = full.real, full.imag
    slices = [slice(None)] * real.ndim
    slices[dim if dim >= 0 else dim + real.ndim] = slice(0, keep)
    selection = tuple(slices)
    return _make_complex(real[selection], imag[selection])


def _resize_at(value, dim, size):
    """Truncate or zero-pad ``value`` along ``dim`` to exactly ``size``."""
    length = value.shape[dim]
    if length == size:
        return value
    slices = [slice(None)] * value.ndim
    if length > size:
        slices[dim] = slice(0, size)
        return value[tuple(slices)]
    padding = list(value.shape)
    padding[dim] = size - length
    return jt.concat([value, _zeros_on(padding, value.dtype, value)], dim=dim)


def _cufft_irfft_1d(value, length, dim, norm=None):
    ops = _get_cufft_ops()
    if ops is None or not jt.flags.no_grad:
        return None
    probe = value.real if isinstance(value, _ComplexNumber) else value
    with device_scope_like(probe):
        moved, restore = _to_last(value, dim)
        pair = _as_real_pair(moved, None)
        dtype_name = _jittor_dtype_name(pair.dtype)
        if dtype_name not in ("float32", "float64"):
            return None
        in_len = int(pair.shape[-2])
        if length <= 0 or in_len <= 0 or pair.numel() == 0:
            return None
        keep = length // 2 + 1
        batch = int(pair.numel() // (in_len * 2))
        in_flat = pair.contiguous().reshape([batch, in_len, 2])
        full_pair = jt.code(
            [batch, 1, length, 2],
            pair.dtype,
            [in_flat],
            cuda_header=_REAL_TO_PAIR_HEADER,
            cuda_src="""
            const int64_t batch = (int64_t)in0->shape[0];
            const int64_t in_len = (int64_t)in0->shape[1];
            const int64_t full_len = (int64_t)out0->shape[2];
            const int64_t keep = full_len / 2 + 1;
            const int64_t total = batch * full_len;
            if (total > 0) {
                constexpr int threads = 256;
                const int64_t raw_blocks = (total + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_irfft_mirror_pair_k<in0_type><<<blocks, threads>>>(
                    in0_p, out0_p, batch, in_len, keep, full_len);
                CHECK(0 == cudaGetLastError());
            }
            """,
        )
        out_pair = ops.cufft_fft(full_pair, True)
        scale = _fft_norm_scale(length, True, norm)
        out_real = jt.code(
            [batch, length],
            pair.dtype,
            [out_pair],
            cuda_header=_REAL_TO_PAIR_HEADER,
            cuda_src="""
            const int64_t n = (int64_t)out0->num;
            const in0_type scale = (in0_type)data["scale"];
            if (n > 0) {
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_pair_to_real_k<in0_type><<<blocks, threads>>>(in0_p, out0_p, n, scale);
                CHECK(0 == cudaGetLastError());
            }
            """,
            data={"scale": float(scale)},
        )
        out_real = out_real.reshape(list(pair.shape[:-2]) + [length])
        if restore is not None:
            out_real = out_real.permute(*restore)
        return out_real


def irfft(input, n=None, dim=-1, norm=None):
    # ``_real_imag`` is the accessor that tells a real input apart from a
    # complex one; ``input.real``/``input.imag`` cannot, because a real Var
    # answers both (returning itself and a zero Var).
    real, imag = _real_imag(input)
    dimension = dim if dim >= 0 else dim + real.ndim
    half = real.shape[dimension]
    length = (2 * (half - 1)) if n is None else int(n)
    if length < 1:
        raise RuntimeError(
            f"irfft: invalid number of data points ({length}) specified")
    if isinstance(input, jt.Var) and (dim == -1 or dim == input.ndim - 1):
        if not _cufft_eager_checked:
            _ensure_cufft_eager()
        if _cufft_eager_irfft is not None:
            scale = _fft_norm_scale(length, True, norm)
            res = _cufft_eager_irfft(input, int(length), float(scale))
            if res is not None:
                return res
    if dispatch_context(real).backend == "cuda":
        fast = _cufft_irfft_1d(input, length, dim, norm=norm)
        if fast is not None:
            return fast
    # The half spectrum that describes a signal of `length` samples has exactly
    # length//2 + 1 entries, so resize the input to that before mirroring it.
    # Resizing the *mirrored* spectrum instead (which is what passing `length`
    # to _fft_core did) truncates or pads in the middle of the conjugate pairs.
    keep = length // 2 + 1
    with device_scope_like(real):
        real = _resize_at(real, dimension, keep)
        imag = (_zeros_on(real.shape, real.dtype, real) if imag is None
                else _resize_at(imag, dimension, keep))
        # Bin `length - k` is the conjugate of bin `k`; for even `length` the
        # Nyquist bin is its own mirror and must not be repeated.
        first_mirror = keep - 2 if length % 2 == 0 else keep - 1
        if first_mirror >= 1:
            slices = [slice(None)] * real.ndim
            slices[dimension] = slice(1, first_mirror + 1)
            mirrored = tuple(slices)
            real = jt.concat([real, jt.flip(real[mirrored], dim=dimension)], dim=dimension)
            imag = jt.concat([imag, -jt.flip(imag[mirrored], dim=dimension)], dim=dimension)
        return _fft_core(_make_complex(real, imag), None, dim, True, norm).real


def _shift_dims(value, dim, inverse):
    dims = list(range(value.ndim)) if dim is None else (
        [dim] if isinstance(dim, int) else list(dim)
    )
    shifts = [
        (-(int(value.shape[d]) // 2) if inverse else int(value.shape[d]) // 2)
        for d in dims
    ]
    return jt.roll(value, shifts, dims)


def fftshift(input, dim=None):
    if isinstance(input, _ComplexNumber):
        return _ComplexNumber(
            _shift_dims(input.real, dim, False),
            _shift_dims(input.imag, dim, False),
        )
    return _shift_dims(input, dim, False)


def ifftshift(input, dim=None):
    if isinstance(input, _ComplexNumber):
        return _ComplexNumber(
            _shift_dims(input.real, dim, True),
            _shift_dims(input.imag, dim, True),
        )
    return _shift_dims(input, dim, True)


def _freq_common(api, np_fn, n, d, dtype, device, requires_grad, out, kwargs):
    """Shared tail of fftfreq/rfftfreq.

    These two used to be ``def fftfreq(n, d=1.0, **kwargs)`` with the body
    ignoring ``kwargs`` entirely. Three separate problems:

    * ``dtype`` was swallowed and the result was ALWAYS float32, so
      ``fftfreq(8, dtype="float64")`` silently returned float32 and every
      downstream computation quietly lost precision. That is an observable
      value, so it is implemented here rather than reported.
    * ``device`` was swallowed. jittor has no per-Var device (placement follows
      the global ``jt.flags.use_cuda``), so it is reported as `ignored` -- it
      changes no value.
    * **``**kwargs`` also swallowed misspellings.** ``fftfreq(8, dtpye="f64")``
      returned a float32 Var and said nothing. A signature that accepts
      anything and reads nothing is the hardest kind of lie to debug, so
      unknown keywords now raise TypeError the way any other function would.
    """
    if kwargs:
        raise TypeError("%s() got an unexpected keyword argument %r"
                        % (api.rsplit(".", 1)[-1], sorted(kwargs)[0]))
    if out is not None:
        _arg_policy.unsupported(
            api, "out", out,
            "the result is returned as a new Var and nothing is written into "
            "the supplied out Var, so the caller's buffer keeps its old values")
    if device is not None:
        _arg_policy.ignored(
            api, "device", device,
            "jittor has no per-Var device; placement follows the global "
            "jt.flags.use_cuda")
    values = np_fn(n, d)
    # torch's default for these is the default floating dtype (float32 here).
    name = _np_dtype(dtype) if dtype is not None else "float32"
    # dtype= is NOT optional here: jt.array(np_float64_array) silently narrows
    # to float32, which is exactly the bug this parameter is being added to fix.
    result = jt.array(values.astype(name), dtype=name)
    if requires_grad:
        result.start_grad()
    return result


def _np_dtype(dtype):
    """Accept the spellings jittor/torch users pass: 'float64', jt.float64, ..."""
    name = getattr(dtype, "__name__", None) or _jittor_dtype_name(dtype)
    name = name.replace("jittor.", "").replace("torch.", "")
    if name not in ("float16", "float32", "float64"):
        raise TypeError(
            "fftfreq/rfftfreq need a floating dtype, got %r" % (_jittor_dtype_name(dtype),))
    return name


def fftfreq(n, d=1.0, *, out=None, dtype=None, device=None,
            requires_grad=False, **kwargs):
    return _freq_common("jittor.fft.fftfreq", np.fft.fftfreq, n, d, dtype,
                        device, requires_grad, out, kwargs)


def rfftfreq(n, d=1.0, *, out=None, dtype=None, device=None,
             requires_grad=False, **kwargs):
    return _freq_common("jittor.fft.rfftfreq", np.fft.rfftfreq, n, d, dtype,
                        device, requires_grad, out, kwargs)


from .spectral import stft  # noqa: E402 - load after native FFT definitions


__all__ = [
    "stft",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
]
