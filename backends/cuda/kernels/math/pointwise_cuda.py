"""Fused 1-D CUDA kernels for slow pointwise math and activation ops.

Each kernel flattens contiguous inputs to 1-D so a single compiled module per
dtype serves every tensor rank and shape, with exact analytical backward passes.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import register_kernel


_POINTWISE_CUDA_HEADER = r"""
#include <cmath>
#include <type/fp16_compute.h>

__device__ __forceinline__ float _pw_exp(float x) { return ::expf(x); }
__device__ __forceinline__ double _pw_exp(double x) { return ::exp(x); }
__device__ __forceinline__ float _pw_expm1(float x) { return ::expm1f(x); }
__device__ __forceinline__ double _pw_expm1(double x) { return ::expm1(x); }
__device__ __forceinline__ float _pw_log1p(float x) { return ::log1pf(x); }
__device__ __forceinline__ double _pw_log1p(double x) { return ::log1p(x); }
__device__ __forceinline__ float _pw_tanh(float x) { return ::tanhf(x); }
__device__ __forceinline__ double _pw_tanh(double x) { return ::tanh(x); }
__device__ __forceinline__ float _pw_rsqrt(float x) { return ::rsqrtf(x); }
__device__ __forceinline__ double _pw_rsqrt(double x) { return ::rsqrt(x); }
__device__ __forceinline__ float _pw_atan2(float y, float x) { return ::atan2f(y, x); }
__device__ __forceinline__ double _pw_atan2(double y, double x) { return ::atan2(y, x); }
__device__ __forceinline__ float _pw_fabs(float x) { return ::fabsf(x); }
__device__ __forceinline__ double _pw_fabs(double x) { return ::fabs(x); }
__device__ __forceinline__ float _pw_sqrt(float x) { return ::sqrtf(x); }
__device__ __forceinline__ double _pw_sqrt(double x) { return ::sqrt(x); }

template <typename T, typename AccT>
__global__ void _jt_var_std_lastdim_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ out_p,
    AccT* __restrict__ mean_p,
    int64_t rows, int64_t width,
    AccT correction, int take_sqrt
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t total_warps = (int64_t)gridDim.x * warps_per_block;

    for (int64_t row = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
         row < rows;
         row += total_warps) {
        const T* r = xp + row * width;
        // Welford per-lane accumulation
        AccT count = (AccT)0;
        AccT mean = (AccT)0;
        AccT m2 = (AccT)0;
        for (int64_t c = lane; c < width; c += 32) {
            AccT v = (AccT)r[c];
            count += (AccT)1;
            AccT delta = v - mean;
            mean += delta / count;
            m2 += delta * (v - mean);
        }
        // Parallel Chan/Welford warp merge across 32 lanes
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            AccT o_cnt = __shfl_down_sync(0xffffffffu, count, d);
            AccT o_mean = __shfl_down_sync(0xffffffffu, mean, d);
            AccT o_m2 = __shfl_down_sync(0xffffffffu, m2, d);
            AccT new_cnt = count + o_cnt;
            if (o_cnt > (AccT)0) {
                if (count == (AccT)0) {
                    mean = o_mean;
                    m2 = o_m2;
                } else {
                    AccT delta = o_mean - mean;
                    mean += delta * (o_cnt / new_cnt);
                    m2 += o_m2 + delta * delta * (count * o_cnt / new_cnt);
                }
                count = new_cnt;
            }
        }
        if (lane == 0) {
            AccT denom = (AccT)width - correction;
            AccT var_v = (denom > (AccT)0) ? (m2 / denom) : (AccT)0;
            if (var_v < (AccT)0) var_v = (AccT)0;
            mean_p[row] = mean;
            out_p[row] = (T)(take_sqrt ? _pw_sqrt(var_v) : var_v);
        }
    }
}

template <typename T, typename AccT>
__global__ void _jt_var_std_lastdim_bwd_k(
    const T* __restrict__ xp,
    const AccT* __restrict__ mean_p,
    const T* __restrict__ out_p,
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t rows, int64_t width,
    AccT correction, int take_sqrt
) {
    const int64_t total = rows * width;
    const AccT denom = (AccT)width - correction;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = idx / width;
        AccT g = (AccT)gp[row];
        AccT scale;
        if (denom <= (AccT)0) {
            scale = (AccT)0;
        } else if (take_sqrt) {
            AccT s = (AccT)out_p[row];
            scale = (s > (AccT)0) ? (g / (denom * s)) : (AccT)0;
        } else {
            scale = ((AccT)2 * g) / denom;
        }
        dxp[idx] = (T)(((AccT)xp[idx] - mean_p[row]) * scale);
    }
}

template <typename T, typename AccT>
__global__ void _jt_prod_lastdim_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ out_p,
    int64_t rows, int64_t width
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t total_warps = (int64_t)gridDim.x * warps_per_block;

    for (int64_t row = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
         row < rows;
         row += total_warps) {
        const T* r = xp + row * width;
        AccT acc = (AccT)1;
        for (int64_t c = lane; c < width; c += 32) {
            acc = acc * (AccT)r[c];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            acc = acc * __shfl_down_sync(0xffffffffu, acc, d);
        }
        if (lane == 0) {
            out_p[row] = (T)acc;
        }
    }
}

template <typename T, typename AccT>
__global__ void _jt_prod_lastdim_bwd_k(
    const T* __restrict__ xp,
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t rows, int64_t width
) {
    for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += (int64_t)gridDim.x * blockDim.x) {
        const T* r = xp + row * width;
        T* dr = dxp + row * width;
        AccT g = (AccT)gp[row];
        int zero_cnt = 0;
        int64_t zero_pos = -1;
        AccT nz_prod = (AccT)1;
        for (int64_t c = 0; c < width; ++c) {
            AccT v = (AccT)r[c];
            if (v == (AccT)0) {
                ++zero_cnt;
                zero_pos = c;
            } else {
                nz_prod = nz_prod * v;
            }
        }
        if (zero_cnt >= 2) {
            for (int64_t c = 0; c < width; ++c) dr[c] = (T)0;
        } else if (zero_cnt == 1) {
            for (int64_t c = 0; c < width; ++c)
                dr[c] = (c == zero_pos) ? (T)(g * nz_prod) : (T)0;
        } else {
            AccT total = g * nz_prod;
            for (int64_t c = 0; c < width; ++c)
                dr[c] = (T)(total / (AccT)r[c]);
        }
    }
}

// op_id:
//   0: rsqrt
//   1: expm1
//   2: reciprocal
//   3: softplus (p0=beta, p1=threshold)
//   4: mish
//   5: logsigmoid
//   6: elu (p0=alpha)
//   7: celu (p0=alpha)
template <typename T, typename AccT, int OP>
__global__ void _jt_pw_unary_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t n,
    AccT p0, AccT p1
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT x = (AccT)xp[i];
        AccT y;
        if (OP == 0) {
            y = _pw_rsqrt(x);
        } else if (OP == 1) {
            y = _pw_expm1(x);
        } else if (OP == 2) {
            y = (AccT)1 / x;
        } else if (OP == 3) {
            AccT bx = p0 * x;
            y = (bx > p1) ? x : (_pw_log1p(_pw_exp(bx)) / p0);
        } else if (OP == 4) {
            AccT sp = (x > (AccT)20) ? x : _pw_log1p(_pw_exp(x));
            y = x * _pw_tanh(sp);
        } else if (OP == 5) {
            AccT m = (x < (AccT)0) ? x : (AccT)0;
            y = m - _pw_log1p(_pw_exp(-_pw_fabs(x)));
        } else if (OP == 6) {
            y = (x > (AccT)0) ? x : (p0 * _pw_expm1(x));
        } else {
            y = (x > (AccT)0) ? x : (p0 * _pw_expm1(x / p0));
        }
        yp[i] = (T)y;
    }
}

template <typename T, typename AccT, int OP>
__global__ void _jt_pw_unary_bwd_k(
    const T* __restrict__ xp,
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t n,
    AccT p0, AccT p1
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT x = (AccT)xp[i];
        AccT g = (AccT)gp[i];
        AccT dx;
        if (OP == 0) {
            AccT r = _pw_rsqrt(x);
            dx = (AccT)(-0.5) * r * r * r * g;
        } else if (OP == 1) {
            dx = _pw_exp(x) * g;
        } else if (OP == 2) {
            dx = -(g / (x * x));
        } else if (OP == 3) {
            AccT bx = p0 * x;
            if (bx > p1) {
                dx = g;
            } else {
                AccT z = _pw_exp(bx);
                dx = g * (z / (z + (AccT)1));
            }
        } else if (OP == 4) {
            AccT sp = (x > (AccT)20) ? x : _pw_log1p(_pw_exp(x));
            AccT th = _pw_tanh(sp);
            AccT sig = (AccT)1 / ((AccT)1 + _pw_exp(-x));
            dx = g * (th + x * ((AccT)1 - th * th) * sig);
        } else if (OP == 5) {
            AccT sig_neg = (AccT)1 / ((AccT)1 + _pw_exp(x));
            dx = g * sig_neg;
        } else if (OP == 6) {
            dx = (x > (AccT)0) ? g : (g * p0 * _pw_exp(x));
        } else {
            dx = (x > (AccT)0) ? g : (g * _pw_exp(x / p0));
        }
        dxp[i] = (T)dx;
    }
}

template <typename T, typename AccT>
__global__ void _jt_pw_atan2_fwd_k(
    const T* __restrict__ yp,
    const T* __restrict__ xp,
    T* __restrict__ out_p,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        out_p[i] = (T)_pw_atan2((AccT)yp[i], (AccT)xp[i]);
    }
}

template <typename T, typename AccT>
__global__ void _jt_pw_atan2_bwd_k(
    const T* __restrict__ yp,
    const T* __restrict__ xp,
    const T* __restrict__ gp,
    T* __restrict__ dyp,
    T* __restrict__ dxp,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT y = (AccT)yp[i];
        AccT x = (AccT)xp[i];
        AccT g = (AccT)gp[i];
        AccT denom = x * x + y * y;
        dyp[i] = (T)(g * x / denom);
        dxp[i] = (T)(-g * y / denom);
    }
}
"""


def _pw_unary_fwd_code(x, op_id, p0=0.0, p1=0.0):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n = int(x.numel())
    x1 = x.contiguous().reshape([n])
    with device_scope_like(x):
        y1 = jt.code(
            [n],
            x.dtype,
            [x1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            if (n > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_pw_unary_fwd_k<in0_type, {acc_t}, {op_id}><<<blocks, threads>>>(
                    in0_p, out0_p, n, ({acc_t})data["p0"], ({acc_t})data["p1"]);
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"p0": float(p0), "p1": float(p1)},
        )
    return y1.reshape(x.shape)


def _pw_unary_bwd_code(x, dout, op_id, p0=0.0, p1=0.0):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n = int(x.numel())
    x1 = x.contiguous().reshape([n])
    g1 = dout.contiguous().reshape([n])
    with device_scope_like(x):
        dx1 = jt.code(
            [n],
            x.dtype,
            [x1, g1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            if (n > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_pw_unary_bwd_k<in0_type, {acc_t}, {op_id}><<<blocks, threads>>>(
                    in0_p, in1_p, out0_p, n, ({acc_t})data["p0"], ({acc_t})data["p1"]);
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"p0": float(p0), "p1": float(p1)},
        )
    return dx1.reshape(x.shape)


class _PointwiseUnaryCuda(jt.Function):
    def __init__(self, op_id, p0=0.0, p1=0.0):
        self.op_id = op_id
        self.p0 = p0
        self.p1 = p1

    def execute(self, x):
        self.saved_x = x
        return _pw_unary_fwd_code(x, self.op_id, self.p0, self.p1)

    def grad(self, dout):
        if dout is None:
            return None
        return _pw_unary_bwd_code(self.saved_x, dout, self.op_id, self.p0, self.p1)


def _dispatch_pw_unary(x, op_id, p0=0.0, p1=0.0):
    dname = _dtype_name(x.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16") or x.numel() == 0:
        return None
    if jt.flags.no_grad:
        return _pw_unary_fwd_code(x, op_id, p0, p1)
    return _PointwiseUnaryCuda(op_id, p0, p1)(x)


def _rsqrt_cuda(x):
    return _dispatch_pw_unary(x, 0)


def _expm1_cuda(x):
    return _dispatch_pw_unary(x, 1)


def _reciprocal_cuda(x):
    return _dispatch_pw_unary(x, 2)


def _softplus_cuda(x, beta=1.0, threshold=20.0):
    return _dispatch_pw_unary(x, 3, float(beta), float(threshold))


def _mish_cuda(x):
    return _dispatch_pw_unary(x, 4)


def _logsigmoid_cuda(x):
    return _dispatch_pw_unary(x, 5)


def _elu_cuda(x, alpha=1.0):
    return _dispatch_pw_unary(x, 6, float(alpha))


def _celu_cuda(x, alpha=1.0):
    return _dispatch_pw_unary(x, 7, float(alpha))


def _atan2_fwd_code(y, x):
    acc_t = "double" if _dtype_name(y.dtype) == "float64" else "float"
    n = int(y.numel())
    y1 = y.contiguous().reshape([n])
    x1 = x.contiguous().reshape([n])
    with device_scope_like(y):
        o1 = jt.code(
            [n],
            y.dtype,
            [y1, x1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            if (n > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_pw_atan2_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(in0_p, in1_p, out0_p, n);
                CHECK(0 == cudaGetLastError());
            }}
            """,
        )
    return o1.reshape(y.shape)


def _atan2_bwd_code(y, x, dout):
    acc_t = "double" if _dtype_name(y.dtype) == "float64" else "float"
    n = int(y.numel())
    y1 = y.contiguous().reshape([n])
    x1 = x.contiguous().reshape([n])
    g1 = dout.contiguous().reshape([n])
    with device_scope_like(y):
        dy1, dx1 = jt.code(
            [[n], [n]],
            [y.dtype, x.dtype],
            [y1, x1, g1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            if (n > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_pw_atan2_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                    in0_p, in1_p, in2_p, out0_p, out1_p, n);
                CHECK(0 == cudaGetLastError());
            }}
            """,
        )
    return dy1.reshape(y.shape), dx1.reshape(x.shape)


class _Atan2Cuda(jt.Function):
    def execute(self, y, x):
        self.saved = (y, x)
        return _atan2_fwd_code(y, x)

    def grad(self, dout):
        if dout is None:
            return None, None
        y, x = self.saved
        return _atan2_bwd_code(y, x, dout)


def _atan2_cuda(y, x):
    if not isinstance(y, jt.Var) or not isinstance(x, jt.Var):
        return None
    if y.shape != x.shape or y.dtype != x.dtype or y.numel() == 0:
        return None
    dname = _dtype_name(y.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    if jt.flags.no_grad:
        return _atan2_fwd_code(y, x)
    return _Atan2Cuda()(y, x)


def _var_std_fwd_code(x, correction, take_sqrt, keepdim):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    x2 = x.contiguous().reshape([rows, width])
    with device_scope_like(x):
        out1, mean1 = jt.code(
            [[rows], [rows]],
            [x.dtype, acc_dtype],
            [x2],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t rows = (int64_t)in0->shape[0];
            const int64_t width = (int64_t)in0->shape[1];
            if (rows > 0 && width > 0) {{
                const int64_t raw_blocks = (rows + 7) / 8;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_var_std_lastdim_fwd_k<in0_type, {acc_t}><<<blocks, 256>>>(
                    in0_p, out0_p, out1_p, rows, width,
                    ({acc_t})data["corr"], (int)data["take_sqrt"]);
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"corr": float(correction), "take_sqrt": float(1 if take_sqrt else 0)},
        )
    out_shape = list(x.shape[:-1]) + ([1] if keepdim else [])
    return out1.reshape(out_shape), mean1, out1


def _var_std_bwd_code(x, mean1, out1, dout, correction, take_sqrt):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    x2 = x.contiguous().reshape([rows, width])
    g1 = dout.contiguous().reshape([rows])
    with device_scope_like(x):
        dx2 = jt.code(
            [rows, width],
            x.dtype,
            [x2, mean1, out1, g1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t rows = (int64_t)in0->shape[0];
            const int64_t width = (int64_t)in0->shape[1];
            const int64_t total = rows * width;
            if (total > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (total + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_var_std_lastdim_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                    in0_p, in1_p, in2_p, in3_p, out0_p, rows, width,
                    ({acc_t})data["corr"], (int)data["take_sqrt"]);
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"corr": float(correction), "take_sqrt": float(1 if take_sqrt else 0)},
        )
    return dx2.reshape(x.shape)


class _VarStdLastdimCuda(jt.Function):
    def __init__(self, correction, take_sqrt, keepdim):
        self.correction = correction
        self.take_sqrt = take_sqrt
        self.keepdim = keepdim

    def execute(self, x):
        out, mean1, out1 = _var_std_fwd_code(x, self.correction, self.take_sqrt, self.keepdim)
        self.saved = (x, mean1, out1)
        return out

    def grad(self, dout):
        if dout is None:
            return None
        x, mean1, out1 = self.saved
        return _var_std_bwd_code(x, mean1, out1, dout, self.correction, self.take_sqrt)


def _var_std_cuda(x, dim, correction, keepdim, take_sqrt):
    if x.ndim < 2 or x.numel() == 0:
        return None
    d = int(dim)
    if d < 0:
        d += x.ndim
    if d != x.ndim - 1:
        return None
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    if width < 2 or width > 2048 or rows < 8:
        return None
    dname = _dtype_name(x.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    if jt.flags.no_grad:
        out, _, _ = _var_std_fwd_code(x, correction, take_sqrt, keepdim)
        return out
    return _VarStdLastdimCuda(correction, take_sqrt, keepdim)(x)


def _prod_fwd_code(x, keepdim):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else ("float" if _dtype_name(x.dtype) in ("float16", "bfloat16") else "in0_type")
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    x2 = x.contiguous().reshape([rows, width])
    with device_scope_like(x):
        out1 = jt.code(
            [rows],
            x.dtype,
            [x2],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t rows = (int64_t)in0->shape[0];
            const int64_t width = (int64_t)in0->shape[1];
            if (rows > 0 && width > 0) {{
                const int64_t raw_blocks = (rows + 7) / 8;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_prod_lastdim_fwd_k<in0_type, {acc_t}><<<blocks, 256>>>(in0_p, out0_p, rows, width);
                CHECK(0 == cudaGetLastError());
            }}
            """,
        )
    out_shape = list(x.shape[:-1]) + ([1] if keepdim else [])
    return out1.reshape(out_shape)


def _prod_bwd_code(x, dout):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    x2 = x.contiguous().reshape([rows, width])
    g1 = dout.contiguous().reshape([rows])
    with device_scope_like(x):
        dx2 = jt.code(
            [rows, width],
            x.dtype,
            [x2, g1],
            cuda_header=_POINTWISE_CUDA_HEADER,
            cuda_src=f"""
            const int64_t rows = (int64_t)in0->shape[0];
            const int64_t width = (int64_t)in0->shape[1];
            if (rows > 0 && width > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (rows + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_prod_lastdim_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(in0_p, in1_p, out0_p, rows, width);
                CHECK(0 == cudaGetLastError());
            }}
            """,
        )
    return dx2.reshape(x.shape)


class _ProdLastdimCuda(jt.Function):
    def __init__(self, keepdim):
        self.keepdim = keepdim

    def execute(self, x):
        self.saved_x = x
        return _prod_fwd_code(x, self.keepdim)

    def grad(self, dout):
        if dout is None:
            return None
        return _prod_bwd_code(self.saved_x, dout)


def _prod_cuda(x, dim, keepdim):
    if x.ndim < 2 or x.numel() == 0:
        return None
    d = int(dim)
    if d < 0:
        d += x.ndim
    if d != x.ndim - 1:
        return None
    width = int(x.shape[-1])
    rows = int(x.numel() // width)
    if width < 2 or width > 2048 or rows < 8:
        return None
    dname = _dtype_name(x.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16", "int32", "int64"):
        return None
    if jt.flags.no_grad:
        return _prod_fwd_code(x, keepdim)
    if dname in ("int32", "int64"):
        return None
    return _ProdLastdimCuda(keepdim)(x)


register_kernel("math.rsqrt", "cuda", _rsqrt_cuda)
register_kernel("math.expm1", "cuda", _expm1_cuda)
register_kernel("math.reciprocal", "cuda", _reciprocal_cuda)
register_kernel("math.atan2", "cuda", _atan2_cuda)
register_kernel("nn.softplus", "cuda", _softplus_cuda)
register_kernel("nn.mish", "cuda", _mish_cuda)
register_kernel("nn.logsigmoid", "cuda", _logsigmoid_cuda)
register_kernel("nn.elu", "cuda", _elu_cuda)
register_kernel("nn.celu", "cuda", _celu_cuda)
register_kernel("math.var_std_lastdim", "cuda", _var_std_cuda)
register_kernel("math.prod_lastdim", "cuda", _prod_cuda)
