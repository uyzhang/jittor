"""Fused CUDA kernels for cross_entropy_loss, nll_loss, and bce_with_logits."""

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import dispatch_context, register_kernel


_LOSS_CUDA_HEADER = r"""
#include <cmath>
#include <type/fp16_compute.h>
#include <ops/composite/index_bounds.h>

__device__ __forceinline__ float _jt_loss_exp(float v) { return ::expf(v); }
__device__ __forceinline__ double _jt_loss_exp(double v) { return ::exp(v); }
__device__ __forceinline__ float _jt_loss_log(float v) { return ::logf(v); }
__device__ __forceinline__ double _jt_loss_log(double v) { return ::log(v); }
__device__ __forceinline__ float _jt_loss_log1p(float v) { return ::log1pf(v); }
__device__ __forceinline__ double _jt_loss_log1p(double v) { return ::log1p(v); }
__device__ __forceinline__ float _jt_loss_fabs(float v) { return ::fabsf(v); }
__device__ __forceinline__ double _jt_loss_fabs(double v) { return ::fabs(v); }

template <typename AccT>
__device__ __forceinline__ AccT _jt_loss_neg_inf();
template <>
__device__ __forceinline__ float _jt_loss_neg_inf<float>() { return -3.402823466e38f; }
template <>
__device__ __forceinline__ double _jt_loss_neg_inf<double>() { return -1e300; }

template <typename AccT>
__device__ __forceinline__ void _jt_online_merge(
    AccT& m_a, AccT& s_a, AccT m_b, AccT s_b
) {
    if (s_b == (AccT)0) return;
    if (s_a == (AccT)0) { m_a = m_b; s_a = s_b; return; }
    if (m_a >= m_b) {
        s_a = s_a + s_b * _jt_loss_exp(m_b - m_a);
    } else {
        s_a = s_b + s_a * _jt_loss_exp(m_a - m_b);
        m_a = m_b;
    }
}

template <typename XType, typename TType, typename LossOutType, typename AccT>
__global__ void _jt_ce_fwd_k(
    const XType* __restrict__ xp,
    const TType* __restrict__ tp,
    const AccT* __restrict__ wp,
    LossOutType* __restrict__ loss_p,
    AccT* __restrict__ max_p,
    AccT* __restrict__ logsum_p,
    AccT* __restrict__ rw_p,
    int64_t N, int64_t C,
    int64_t ignore_idx, int has_ignore, int has_weight,
    jittor::IndexFault* fault
) {
    __shared__ AccT s_max[8];
    __shared__ AccT s_sum[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    for (int64_t row = blockIdx.x; row < N; row += gridDim.x) {
        const int64_t t = (int64_t)tp[row];
        const bool ignored = (has_ignore && t == ignore_idx);
        if (!ignored && (t < 0 || t >= C)) {
            if (tid == 0) {
                if (fault && atomicCAS(&fault->claimed, 0, 1) == 0) {
                    fault->index = t;
                    fault->size = C;
                    __threadfence_system();
                    fault->dim = 1;
                    __threadfence_system();
                } else {
                    printf("[jittor] index %lld is out of bounds for dimension 1 with size %lld\n",
                           (long long)t, (long long)C);
                }
                __trap();
            }
            return;
        }

        const AccT w_t = ignored ? (AccT)0 : (has_weight ? wp[t] : (AccT)1);
        if (ignored) {
            if (tid == 0) {
                loss_p[row] = (LossOutType)0;
                max_p[row] = (AccT)0;
                logsum_p[row] = (AccT)0;
                rw_p[row] = (AccT)0;
            }
            continue;
        }

        const XType* row_x = xp + row * C;
        AccT m = _jt_loss_neg_inf<AccT>();
        AccT s = (AccT)0;
        for (int64_t c = tid; c < C; c += 256) {
            AccT v = (AccT)row_x[c];
            if (s == (AccT)0) {
                m = v;
                s = (AccT)1;
            } else if (v > m) {
                s = s * _jt_loss_exp(m - v) + (AccT)1;
                m = v;
            } else {
                s += _jt_loss_exp(v - m);
            }
        }

        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            AccT om = __shfl_down_sync(0xffffffffu, m, d);
            AccT os = __shfl_down_sync(0xffffffffu, s, d);
            _jt_online_merge(m, s, om, os);
        }
        if (lane == 0) {
            s_max[wid] = m;
            s_sum[wid] = s;
        }
        __syncthreads();

        if (wid == 0) {
            m = (lane < 8) ? s_max[lane] : _jt_loss_neg_inf<AccT>();
            s = (lane < 8) ? s_sum[lane] : (AccT)0;
            #pragma unroll
            for (int d = 4; d > 0; d >>= 1) {
                AccT om = __shfl_down_sync(0xffffffffu, m, d);
                AccT os = __shfl_down_sync(0xffffffffu, s, d);
                _jt_online_merge(m, s, om, os);
            }
            if (lane == 0) {
                // Keep shifted coordinates so large common offsets (e.g. 1e8)
                // do not cancel out log(sum(exp(x - m))).
                AccT log_s = _jt_loss_log(s);
                AccT picked = (AccT)row_x[t];
                max_p[row] = m;
                logsum_p[row] = log_s;
                rw_p[row] = w_t;
                loss_p[row] = (LossOutType)((log_s - (picked - m)) * w_t);
            }
        }
        __syncthreads();
    }
}

template <typename AccT, typename OutT>
__global__ void _jt_loss_reduce_k(
    const AccT* __restrict__ losses,
    const AccT* __restrict__ weights,
    OutT* __restrict__ out_scalar,
    int64_t N,
    int is_mean
) {
    __shared__ AccT s_loss[8];
    __shared__ AccT s_w[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    AccT lsum = (AccT)0;
    AccT wsum = (AccT)0;
    for (int64_t i = tid; i < N; i += 256) {
        lsum += losses[i];
        wsum += weights[i];
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        lsum += __shfl_down_sync(0xffffffffu, lsum, d);
        wsum += __shfl_down_sync(0xffffffffu, wsum, d);
    }
    if (lane == 0) { s_loss[wid] = lsum; s_w[wid] = wsum; }
    __syncthreads();
    if (wid == 0) {
        lsum = (lane < 8) ? s_loss[lane] : (AccT)0;
        wsum = (lane < 8) ? s_w[lane] : (AccT)0;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) {
            lsum += __shfl_down_sync(0xffffffffu, lsum, d);
            wsum += __shfl_down_sync(0xffffffffu, wsum, d);
        }
        if (lane == 0) {
            out_scalar[0] = (OutT)(is_mean ? (lsum / wsum) : lsum);
        }
    }
}

template <typename XType, typename TType, typename AccT>
__global__ void _jt_ce_bwd_k(
    const XType* __restrict__ xp,
    const TType* __restrict__ tp,
    const AccT* __restrict__ max_p,
    const AccT* __restrict__ logsum_p,
    const AccT* __restrict__ rw_p,
    const AccT* __restrict__ dl_p,
    const AccT* __restrict__ wsum_p,
    XType* __restrict__ dxp,
    int64_t N, int64_t C,
    int red_mode
) {
    const AccT denom = (red_mode == 2) ? wsum_p[0] : (AccT)1;
    const AccT global_dl = (red_mode == 0) ? (AccT)0 : (dl_p[0] / denom);
    for (int64_t row = blockIdx.x; row < N; row += gridDim.x) {
        const AccT rw = rw_p[row];
        const XType* row_x = xp + row * C;
        XType* row_dx = dxp + row * C;
        if (rw == (AccT)0) {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                row_dx[c] = (XType)0;
            }
            continue;
        }
        const AccT scale = ((red_mode == 0) ? dl_p[row] : global_dl) * rw;
        const AccT m = max_p[row];
        const AccT log_s = logsum_p[row];
        const int64_t t = (int64_t)tp[row];
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            AccT prob = _jt_loss_exp(((AccT)row_x[c] - m) - log_s);
            AccT g = (c == t) ? (prob - (AccT)1) : prob;
            row_dx[c] = (XType)(g * scale);
        }
    }
}

template <typename XType, typename TType, typename LossOutType, typename AccT>
__global__ void _jt_nll_fwd_k(
    const XType* __restrict__ xp,
    const TType* __restrict__ tp,
    const AccT* __restrict__ wp,
    LossOutType* __restrict__ loss_p,
    AccT* __restrict__ rw_p,
    int64_t N, int64_t C,
    int64_t ignore_idx, int has_weight,
    jittor::IndexFault* fault
) {
    for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < N;
         row += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = (int64_t)tp[row];
        if (t == ignore_idx) {
            loss_p[row] = (LossOutType)0;
            rw_p[row] = (AccT)0;
            continue;
        }
        if (t < 0 || t >= C) {
            if (fault && atomicCAS(&fault->claimed, 0, 1) == 0) {
                fault->index = t;
                fault->size = C;
                __threadfence_system();
                fault->dim = 1;
                __threadfence_system();
            } else {
                printf("[jittor] index %lld is out of bounds for dimension 1 with size %lld\n",
                       (long long)t, (long long)C);
            }
            __trap();
            return;
        }
        const AccT w = has_weight ? wp[t] : (AccT)1;
        rw_p[row] = w;
        loss_p[row] = (LossOutType)(-(AccT)xp[row * C + t] * w);
    }
}

template <typename XType, typename TType, typename AccT>
__global__ void _jt_nll_bwd_k(
    const TType* __restrict__ tp,
    const AccT* __restrict__ rw_p,
    const AccT* __restrict__ dl_p,
    const AccT* __restrict__ wsum_p,
    XType* __restrict__ dxp,
    int64_t N, int64_t C,
    int red_mode
) {
    const AccT denom = (red_mode == 2) ? wsum_p[0] : (AccT)1;
    const AccT global_dl = (red_mode == 0) ? (AccT)0 : (dl_p[0] / denom);
    for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < N;
         row += (int64_t)gridDim.x * blockDim.x) {
        const AccT rw = rw_p[row];
        if (rw == (AccT)0) continue;
        const int64_t t = (int64_t)tp[row];
        const AccT g = -((red_mode == 0) ? dl_p[row] : global_dl) * rw;
        dxp[row * C + t] = (XType)g;
    }
}

template <typename T, typename AccT>
__global__ void _jt_bce_logits_elem_k(
    const T* __restrict__ xp,
    const T* __restrict__ tp,
    T* __restrict__ out_p,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT x = (AccT)xp[i];
        if (x < (AccT)(-50)) x = (AccT)(-50);
        else if (x > (AccT)50) x = (AccT)50;
        AccT t = (AccT)tp[i];
        AccT max_v = (-x > (AccT)0) ? -x : (AccT)0;
        AccT loss = ((AccT)1 - t) * x + max_v + _jt_loss_log1p(_jt_loss_exp(-_jt_loss_fabs(x)));
        out_p[i] = (T)loss;
    }
}

template <typename T, typename AccT>
__global__ void _jt_bce_logits_reduce_k(
    const T* __restrict__ xp,
    const T* __restrict__ tp,
    AccT* __restrict__ part_p,
    int64_t n
) {
    __shared__ AccT s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    AccT acc = (AccT)0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + tid;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT x = (AccT)xp[i];
        if (x < (AccT)(-50)) x = (AccT)(-50);
        else if (x > (AccT)50) x = (AccT)50;
        AccT t = (AccT)tp[i];
        AccT max_v = (-x > (AccT)0) ? -x : (AccT)0;
        acc += ((AccT)1 - t) * x + max_v + _jt_loss_log1p(_jt_loss_exp(-_jt_loss_fabs(x)));
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < 8) ? s[lane] : (AccT)0;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
        if (lane == 0) part_p[blockIdx.x] = acc;
    }
}

template <typename AccT, typename OutT>
__global__ void _jt_sum_finalize_k(
    const AccT* __restrict__ part_p,
    OutT* __restrict__ out_p,
    int64_t count,
    AccT inv_scale
) {
    __shared__ AccT s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    AccT acc = (AccT)0;
    for (int64_t i = tid; i < count; i += 256) acc += part_p[i];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < 8) ? s[lane] : (AccT)0;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
        if (lane == 0) out_p[0] = (OutT)(acc * inv_scale);
    }
}

template <typename T, typename AccT>
__global__ void _jt_bce_logits_bwd_k(
    const T* __restrict__ xp,
    const T* __restrict__ tp,
    const AccT* __restrict__ dl_p,
    T* __restrict__ dxp,
    T* __restrict__ dtp,
    int64_t n,
    int red_mode,
    AccT inv_scale
) {
    const AccT g_scalar = (red_mode == 0) ? (AccT)0 : (dl_p[0] * inv_scale);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        AccT raw_x = (AccT)xp[i];
        AccT x = raw_x;
        AccT clamp_mask = (AccT)1;
        if (x < (AccT)(-50)) { x = (AccT)(-50); clamp_mask = (AccT)0; }
        else if (x > (AccT)50) { x = (AccT)50; clamp_mask = (AccT)0; }
        AccT t = (AccT)tp[i];
        AccT g = (red_mode == 0) ? dl_p[i] : g_scalar;
        AccT sig = (AccT)1 / ((AccT)1 + _jt_loss_exp(-x));
        dxp[i] = (T)(g * (sig - t) * clamp_mask);
        dtp[i] = (T)(-g * x);
    }
}
"""


def _ce_fwd_cuda(x, target, weight, ignore_index, has_ignore, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    out_dtype = "float64" if dname == "float64" else ("float32" if dname in ("float16", "bfloat16") else dname)
    acc_dtype = "float64" if dname == "float64" else "float32"

    has_weight = 1 if weight is not None else 0
    inputs = [x, target] if weight is None else [x, target, weight.cast(acc_dtype).contiguous()]
    w_ptr = f"(const {acc_t}*)nullptr" if weight is None else "in2_p"

    n_rows = int(x.shape[0])
    out_shape = [n_rows] if reduction_mode == 0 else [1]
    if reduction_mode == 0:
        shapes = [out_shape, [n_rows], [n_rows], [n_rows]]
        dtypes = [out_dtype, acc_dtype, acc_dtype, acc_dtype]
        loss_ptr = "out0_p"
        loss_t = "out0_type"
        reduce_launch = ""
    else:
        shapes = [out_shape, [n_rows], [n_rows], [n_rows], [n_rows]]
        dtypes = [out_dtype, acc_dtype, acc_dtype, acc_dtype, acc_dtype]
        loss_ptr = "out4_p"
        loss_t = acc_t
        reduce_launch = f"""
        _jt_loss_reduce_k<{acc_t}, out0_type><<<1, 256>>>(out4_p, out3_p, out0_p, N, {1 if reduction_mode == 2 else 0});
        """

    cuda_src = f"""
    const int64_t N = (int64_t)in0->shape[0];
    const int64_t C = (int64_t)in0->shape[1];
    const int64_t ignore_idx = (int64_t)data["ignore_index"];
    const int has_ignore = (int)data["has_ignore"];
    const int has_weight = {has_weight};
    if (N > 0 && C > 0) {{
        const int blocks = (int)(N > 65535 ? 65535 : N);
        _jt_ce_fwd_k<in0_type, in1_type, {loss_t}, {acc_t}><<<blocks, 256>>>(
            in0_p, in1_p, {w_ptr}, {loss_ptr}, out1_p, out2_p, out3_p,
            N, C, ignore_idx, has_ignore, has_weight, jittor::device_index_fault()
        );
        {reduce_launch}
        CHECK(0 == cudaGetLastError());
    }}
    """
    with device_scope_like(x):
        outs = jt.code(
            shapes,
            dtypes,
            inputs,
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=cuda_src,
            data={
                "ignore_index": float(ignore_index if has_ignore else 0),
                "has_ignore": float(has_ignore),
            },
        )
    return outs[0], outs[1], outs[2], outs[3]


def _ce_bwd_cuda(x, target, row_max, row_logsum, row_weight, dloss, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    dloss_c = dloss.cast(acc_dtype).contiguous()
    if reduction_mode == 2:
        w_sum = row_weight.sum().reshape([1])
        inputs = [x, target, row_max, row_logsum, row_weight, dloss_c, w_sum]
        wsum_ptr = "in6_p"
    else:
        inputs = [x, target, row_max, row_logsum, row_weight, dloss_c]
        wsum_ptr = f"(const {acc_t}*)nullptr"

    cuda_src = f"""
    const int64_t N = (int64_t)in0->shape[0];
    const int64_t C = (int64_t)in0->shape[1];
    const int red_mode = (int)data["red_mode"];
    if (N > 0 && C > 0) {{
        const int blocks = (int)(N > 65535 ? 65535 : N);
        _jt_ce_bwd_k<in0_type, in1_type, {acc_t}><<<blocks, 256>>>(
            in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, {wsum_ptr}, out0_p, N, C, red_mode
        );
        CHECK(0 == cudaGetLastError());
    }}
    """
    with device_scope_like(x):
        return jt.code(
            x.shape,
            x.dtype,
            inputs,
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=cuda_src,
            data={"red_mode": float(reduction_mode)},
        )


class _CrossEntropyCuda(jt.Function):
    def __init__(self, ignore_index, has_ignore, reduction_mode, weight=None):
        self.ignore_index = ignore_index
        self.has_ignore = has_ignore
        self.reduction_mode = reduction_mode
        self.weight = weight

    def execute(self, x, target):
        xc = x.contiguous()
        tc = target.contiguous()
        out, row_max, row_logsum, row_weight = _ce_fwd_cuda(
            xc, tc, self.weight, self.ignore_index, self.has_ignore, self.reduction_mode
        )
        self.saved = (xc, tc, row_max, row_logsum, row_weight)
        return out

    def grad(self, dloss):
        if dloss is None:
            return None, None
        xc, tc, row_max, row_logsum, row_weight = self.saved
        dx = _ce_bwd_cuda(xc, tc, row_max, row_logsum, row_weight, dloss, self.reduction_mode)
        return dx, None


def _cross_entropy_cuda(output, target, weight=None, ignore_index=None, reduction="mean"):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of ['none', 'mean', 'sum'], got %r" % (reduction,)
        )
    dname = _dtype_name(output.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    tname = _dtype_name(target.dtype)
    if tname not in ("int32", "int64", "int16", "int8", "uint8"):
        return None
    if output.ndim not in (2, 4):
        return None
    if weight is not None and weight.ndim != 1:
        return None

    target_shape = target.shape
    x = output
    if x.ndim == 4:
        c_dim = x.shape[1]
        x = x.transpose((0, 2, 3, 1)).reshape((-1, c_dim))
    t = target.reshape((-1,))
    if t.shape[0] != x.shape[0]:
        raise jt.JittorShapeError(
            "cross_entropy_loss: expected one target per input row, but the "
            "input has %d rows and the target %d elements (target shape %s)."
            % (x.shape[0], t.shape[0], list(target_shape))
        )
    if x.shape[0] == 0 or x.shape[1] == 0:
        return None

    ctx = dispatch_context(x)
    if ctx.backend == "cuda":
        t = t.to_device(ctx.device_id)
        if weight is not None:
            weight = weight.to_device(ctx.device_id)

    has_ignore = 1 if ignore_index is not None else 0
    ign = int(ignore_index) if has_ignore else 0
    red_mode = 0 if reduction == "none" else (1 if reduction == "sum" else 2)

    if jt.flags.no_grad:
        out, _, _, _ = _ce_fwd_cuda(x.contiguous(), t.contiguous(), weight, ign, has_ignore, red_mode)
    else:
        out = _CrossEntropyCuda(ign, has_ignore, red_mode, weight=weight)(x, t)

    if reduction == "none":
        return out.reshape(target_shape)
    return out.reshape(())


def _nll_fwd_cuda(x, target, weight, ignore_index, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    out_dtype = "float64" if dname == "float64" else ("float32" if dname in ("float16", "bfloat16") else dname)
    acc_dtype = "float64" if dname == "float64" else "float32"
    has_weight = 1 if weight is not None else 0
    inputs = [x, target] if weight is None else [x, target, weight.cast(acc_dtype).contiguous()]
    w_ptr = f"(const {acc_t}*)nullptr" if weight is None else "in2_p"

    n_rows = int(x.shape[0])
    out_shape = [n_rows] if reduction_mode == 0 else [1]
    if reduction_mode == 0:
        shapes = [out_shape, [n_rows]]
        dtypes = [out_dtype, acc_dtype]
        loss_ptr = "out0_p"
        loss_t = "out0_type"
        reduce_launch = ""
    else:
        shapes = [out_shape, [n_rows], [n_rows]]
        dtypes = [out_dtype, acc_dtype, acc_dtype]
        loss_ptr = "out2_p"
        loss_t = acc_t
        reduce_launch = f"""
        _jt_loss_reduce_k<{acc_t}, out0_type><<<1, 256>>>(out2_p, out1_p, out0_p, N, {1 if reduction_mode == 2 else 0});
        """

    cuda_src = f"""
    const int64_t N = (int64_t)in0->shape[0];
    const int64_t C = (int64_t)in0->shape[1];
    const int64_t ignore_idx = (int64_t)data["ignore_index"];
    const int has_weight = {has_weight};
    if (N > 0 && C > 0) {{
        constexpr int threads = 256;
        const int64_t raw_blocks = (N + threads - 1) / threads;
        const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
        _jt_nll_fwd_k<in0_type, in1_type, {loss_t}, {acc_t}><<<blocks, threads>>>(
            in0_p, in1_p, {w_ptr}, {loss_ptr}, out1_p,
            N, C, ignore_idx, has_weight, jittor::device_index_fault()
        );
        {reduce_launch}
        CHECK(0 == cudaGetLastError());
    }}
    """
    with device_scope_like(x):
        outs = jt.code(
            shapes,
            dtypes,
            inputs,
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=cuda_src,
            data={"ignore_index": float(ignore_index)},
        )
    return outs[0], outs[1]


def _nll_bwd_cuda(x, target, row_weight, dloss, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    dloss_c = dloss.cast(acc_dtype).contiguous()
    if reduction_mode == 2:
        w_sum = row_weight.sum().reshape([1])
        inputs = [x, target, row_weight, dloss_c, w_sum]
        wsum_ptr = "in4_p"
    else:
        inputs = [x, target, row_weight, dloss_c]
        wsum_ptr = f"(const {acc_t}*)nullptr"
    cuda_src = f"""
    const int64_t N = (int64_t)in0->shape[0];
    const int64_t C = (int64_t)in0->shape[1];
    const int red_mode = (int)data["red_mode"];
    cudaMemsetAsync(out0_p, 0, out0->size);
    if (N > 0 && C > 0) {{
        constexpr int threads = 256;
        const int64_t raw_blocks = (N + threads - 1) / threads;
        const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
        _jt_nll_bwd_k<in0_type, in1_type, {acc_t}><<<blocks, threads>>>(
            in1_p, in2_p, in3_p, {wsum_ptr}, out0_p, N, C, red_mode
        );
        CHECK(0 == cudaGetLastError());
    }}
    """
    with device_scope_like(x):
        return jt.code(
            x.shape,
            x.dtype,
            inputs,
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=cuda_src,
            data={"red_mode": float(reduction_mode)},
        )


class _NllLossCuda(jt.Function):
    def __init__(self, ignore_index, reduction_mode, weight=None):
        self.ignore_index = ignore_index
        self.reduction_mode = reduction_mode
        self.weight = weight

    def execute(self, x, target):
        xc = x.contiguous()
        tc = target.contiguous()
        out, row_weight = _nll_fwd_cuda(xc, tc, self.weight, self.ignore_index, self.reduction_mode)
        self.saved = (xc, tc, row_weight)
        return out

    def grad(self, dloss):
        if dloss is None:
            return None, None
        xc, tc, row_weight = self.saved
        dx = _nll_bwd_cuda(xc, tc, row_weight, dloss, self.reduction_mode)
        return dx, None


def _nll_loss_cuda(output, target, weight=None, ignore_index=-100, reduction="mean"):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of ['none', 'mean', 'sum'], got %r" % (reduction,)
        )
    if output.ndim != 2 or target.ndim != 1 or output.shape[0] != target.shape[0]:
        return None
    dname = _dtype_name(output.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    tname = _dtype_name(target.dtype)
    if tname not in ("int32", "int64", "int16", "int8", "uint8"):
        return None
    if output.shape[0] == 0 or output.shape[1] == 0:
        return None

    ctx = dispatch_context(output)
    t = target
    if ctx.backend == "cuda":
        t = t.to_device(ctx.device_id)
        if weight is not None:
            weight = weight.to_device(ctx.device_id)

    ign = int(ignore_index)
    red_mode = 0 if reduction == "none" else (1 if reduction == "sum" else 2)
    if jt.flags.no_grad:
        out, _ = _nll_fwd_cuda(output.contiguous(), t.contiguous(), weight, ign, red_mode)
    else:
        out = _NllLossCuda(ign, red_mode, weight=weight)(output, t)
    if reduction == "none":
        return out
    return out.reshape(())


def _bce_logits_fwd_cuda(x, target, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    n = int(x.numel())
    x1 = x.contiguous().reshape([n])
    t1 = target.cast(x.dtype).contiguous().reshape([n])
    with device_scope_like(x):
        if reduction_mode == 0:
            out1 = jt.code(
                [n],
                x.dtype,
                [x1, t1],
                cuda_header=_LOSS_CUDA_HEADER,
                cuda_src=f"""
                const int64_t n = (int64_t)in0->num;
                if (n > 0) {{
                    constexpr int threads = 256;
                    const int64_t raw_blocks = (n + threads - 1) / threads;
                    const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                    _jt_bce_logits_elem_k<in0_type, {acc_t}><<<blocks, threads>>>(in0_p, in1_p, out0_p, n);
                    CHECK(0 == cudaGetLastError());
                }}
                """,
            )
            return out1.reshape(x.shape)
        n_blocks = min(max((n + 255) // 256, 1), 256)
        inv_scale = (1.0 / n) if reduction_mode == 2 else 1.0
        outs = jt.code(
            [[1], [n_blocks]],
            [x.dtype, acc_dtype],
            [x1, t1],
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            const int blocks = (int)out1->shape[0];
            if (n > 0) {{
                _jt_bce_logits_reduce_k<in0_type, {acc_t}><<<blocks, 256>>>(in0_p, in1_p, out1_p, n);
                _jt_sum_finalize_k<{acc_t}, out0_type><<<1, 256>>>(
                    out1_p, out0_p, blocks, ({acc_t})data["inv_scale"]);
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"inv_scale": float(inv_scale)},
        )
        return outs[0].reshape(())


def _bce_logits_bwd_cuda(x, target, dloss, reduction_mode):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    n = int(x.numel())
    x1 = x.contiguous().reshape([n])
    t1 = target.cast(x.dtype).contiguous().reshape([n])
    dl1 = dloss.cast(acc_dtype).contiguous().reshape([n] if reduction_mode == 0 else [1])
    inv_scale = (1.0 / n) if reduction_mode == 2 else 1.0
    with device_scope_like(x):
        dx1, dt1 = jt.code(
            [[n], [n]],
            [x.dtype, x.dtype],
            [x1, t1, dl1],
            cuda_header=_LOSS_CUDA_HEADER,
            cuda_src=f"""
            const int64_t n = (int64_t)in0->num;
            if (n > 0) {{
                constexpr int threads = 256;
                const int64_t raw_blocks = (n + threads - 1) / threads;
                const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
                _jt_bce_logits_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                    in0_p, in1_p, in2_p, out0_p, out1_p, n,
                    (int)data["red_mode"], ({acc_t})data["inv_scale"]
                );
                CHECK(0 == cudaGetLastError());
            }}
            """,
            data={"red_mode": float(reduction_mode), "inv_scale": float(inv_scale)},
        )
    return dx1.reshape(x.shape), dt1.reshape(target.shape).cast(target.dtype)


class _BceWithLogitsCuda(jt.Function):
    def __init__(self, reduction_mode):
        self.reduction_mode = reduction_mode

    def execute(self, x, target):
        self.saved = (x, target)
        return _bce_logits_fwd_cuda(x, target, self.reduction_mode)

    def grad(self, dloss):
        if dloss is None:
            return None, None
        x, target = self.saved
        return _bce_logits_bwd_cuda(x, target, dloss, self.reduction_mode)


def _bce_with_logits_cuda(output, target, weight=None, pos_weight=None, reduction="mean"):
    if weight is not None or pos_weight is not None:
        return None
    if output.numel() == 0 or output.shape != target.shape:
        return None
    dname = _dtype_name(output.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    if reduction not in ("none", "mean", "sum"):
        return None
    red_mode = 0 if reduction == "none" else (1 if reduction == "sum" else 2)
    if jt.flags.no_grad:
        return _bce_logits_fwd_cuda(output, target, red_mode)
    return _BceWithLogitsCuda(red_mode)(output, target)


register_kernel("nn.cross_entropy_loss", "cuda", _cross_entropy_cuda)
register_kernel("nn.nll_loss", "cuda", _nll_loss_cuda)
register_kernel("nn.bce_with_logits", "cuda", _bce_with_logits_cuda)
