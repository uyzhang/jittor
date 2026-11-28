// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// Precompiled float4-vectorized CUDA kernels for eager C++ fast paths.
// Compiled once with the CUDA backend; launched directly on cudaStreamPerThread
// with zero JIT, zero JitKey hashing, and zero graph-fusion overhead.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "core/common.h"
#include "helper_cuda.h"
#include "ops/composite/index_bounds.h"
#include "runtime/cuda_fast_ops.h"

namespace jittor {
namespace {

static inline int grid_blocks(int64_t work, int threads = 256) {
    int64_t b = (work + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

__host__ __device__ static inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

__device__ __forceinline__ float f_signed_powf(float x, float y) {
    if (x < 0.0f && ::floorf(y) == y)
        return ::powf(-x, y) * (::fmodf(y, 2.0f) != 0.0f ? -1.0f : 1.0f);
    return ::powf(x, y);
}

__device__ __forceinline__ float f_digamma_f32(float x) {
    constexpr float PSI_10 = 2.25175258906672110764f;
    if (x == 0.0f) return ::copysignf(CUDART_INF_F, -x);
    bool x_is_int = (x == ::truncf(x));
    if (x < 0.0f) {
        if (x_is_int) return ::nanf("");
        float q = 0.0f;
        float r = ::modff(x, &q);
        float pi_over_tan = 3.14159265358979323846f / ::tanf(3.14159265358979323846f * r);
        return f_digamma_f32(1.0f - x) - pi_over_tan;
    }
    float result = 0.0f;
    while (x < 10.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    if (x == 10.0f) return result + PSI_10;
    static constexpr float A[7] = {
         8.33333333333333333333e-2f,
        -2.10927960927960927961e-2f,
         7.57575757575757575758e-3f,
        -4.16666666666666666667e-3f,
         3.96825396825396825397e-3f,
        -8.33333333333333333333e-3f,
         8.33333333333333333333e-2f,
    };
    float y = 0.0f;
    if (x < 1.0e17f) {
        float z = 1.0f / (x * x);
        float p = A[0];
        #pragma unroll
        for (int i = 1; i <= 6; ++i) p = p * z + A[i];
        y = z * p;
    }
    return result + ::logf(x) - (0.5f / x) - y;
}

template <int OP>
__device__ __forceinline__ float apply_unary_f32(float x, float p0, float p1, int flags) {
    switch (OP) {
        case (int)FastUnaryOp::Exp:         return ::expf(x);
        case (int)FastUnaryOp::Log:         return ::logf(x);
        case (int)FastUnaryOp::Log1p:       return ::log1pf(x);
        case (int)FastUnaryOp::Log2:        return ::log2f(x);
        case (int)FastUnaryOp::Exp2:        return ::exp2f(x);
        case (int)FastUnaryOp::Expm1:       return ::expm1f(x);
        case (int)FastUnaryOp::Sqrt:        return ::sqrtf(x);
        case (int)FastUnaryOp::Rsqrt:       return ::rsqrtf(x);
        case (int)FastUnaryOp::Reciprocal:  return 1.0f / x;
        case (int)FastUnaryOp::Abs:         return ::fabsf(x);
        case (int)FastUnaryOp::Neg:         return -x;
        case (int)FastUnaryOp::Sin:         return ::sinf(x);
        case (int)FastUnaryOp::Cos:         return ::cosf(x);
        case (int)FastUnaryOp::Tan:         return ::tanf(x);
        case (int)FastUnaryOp::Tanh:        return ::tanhf(x);
        case (int)FastUnaryOp::Sigmoid:     return 1.0f / (1.0f + ::expf(-x));
        case (int)FastUnaryOp::Relu:        return x > 0.0f ? x : 0.0f;
        case (int)FastUnaryOp::Erf:         return ::erff(x);
        case (int)FastUnaryOp::Floor:       return ::floorf(x);
        case (int)FastUnaryOp::Ceil:        return ::ceilf(x);
        case (int)FastUnaryOp::Round:       return ::rintf(x);
        case (int)FastUnaryOp::Trunc:       return ::truncf(x);
        case (int)FastUnaryOp::Sign:        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : (x == 0.0f ? 0.0f : x));
        case (int)FastUnaryOp::Deg2Rad:     return x * 0.017453292519943295f;
        case (int)FastUnaryOp::Rad2Deg:     return x * 57.29577951308232f;
        case (int)FastUnaryOp::Lgamma:      return ::lgammaf(x);
        case (int)FastUnaryOp::Digamma:     return f_digamma_f32(x);
        case (int)FastUnaryOp::Clamp: {
            if (::isnan(x)) return x;
            float v = x;
            if ((flags & 1) && v < p0) v = p0;
            if ((flags & 2) && v > p1) v = p1;
            return v;
        }
        case (int)FastUnaryOp::Hardswish: {
            if (::isnan(x)) return x;
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return x;
            return x * (x + 3.0f) * (1.0f / 6.0f);
        }
        case (int)FastUnaryOp::Hardsigmoid: {
            if (::isnan(x)) return x;
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return 1.0f;
            return (x + 3.0f) * (1.0f / 6.0f);
        }
        case (int)FastUnaryOp::Hardtanh: {
            if (::isnan(x)) return x;
            return x < p0 ? p0 : (x > p1 ? p1 : x);
        }
        case (int)FastUnaryOp::Elu:         return x > 0.0f ? x : p0 * ::expm1f(x);
        case (int)FastUnaryOp::Celu:        return x > 0.0f ? x : p0 * ::expm1f(x / p0);
        case (int)FastUnaryOp::Selu: {
            const float a = 1.6732632423543772848170429916717f;
            const float s = 1.0507009873554804934193349852946f;
            return x > 0.0f ? (s * x) : (s * a * ::expm1f(x));
        }
        case (int)FastUnaryOp::Softplus: {
            float bx = p0 * x;
            return bx > p1 ? x : (::log1pf(::expf(bx)) / p0);
        }
        case (int)FastUnaryOp::Mish: {
            float sp = x > 20.0f ? x : ::log1pf(::expf(x));
            return x * ::tanhf(sp);
        }
        case (int)FastUnaryOp::Logsigmoid: {
            float m = x < 0.0f ? x : 0.0f;
            return m - ::log1pf(::expf(-::fabsf(x)));
        }
        case (int)FastUnaryOp::Gelu:        return 0.5f * x * (1.0f + ::erff(x * 0.7071067811865476f));
        case (int)FastUnaryOp::Silu:        return x / (1.0f + ::expf(-x));
        case (int)FastUnaryOp::LeakyRelu:   return x >= 0.0f ? x : p0 * x;
        default:                            return x;
    }
}

template <int OP>
__global__ void k_fast_unary_f32_vec4(
    const float* __restrict__ x, float* __restrict__ y,
    int64_t n4, int64_t n, float p0, float p1, int flags
) {
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += (int64_t)gridDim.x * blockDim.x) {
        float4 v = x4[i];
        float4 r;
        r.x = apply_unary_f32<OP>(v.x, p0, p1, flags);
        r.y = apply_unary_f32<OP>(v.y, p0, p1, flags);
        r.z = apply_unary_f32<OP>(v.z, p0, p1, flags);
        r.w = apply_unary_f32<OP>(v.w, p0, p1, flags);
        y4[i] = r;
    }
    int64_t tail_base = n4 * 4;
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        int64_t idx = tail_base + threadIdx.x;
        if (idx < n) y[idx] = apply_unary_f32<OP>(x[idx], p0, p1, flags);
    }
}

template <int OP>
__global__ void k_fast_unary_f32_scalar(
    const float* __restrict__ x, float* __restrict__ y,
    int64_t n, float p0, float p1, int flags
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        y[i] = apply_unary_f32<OP>(x[i], p0, p1, flags);
    }
}

template <int OP>
static void launch_unary(const float* x, float* y, int64_t n, float p0, float p1, int flags) {
    if (n >= 4 && is_aligned16(x) && is_aligned16(y)) {
        int64_t n4 = n >> 2;
        k_fast_unary_f32_vec4<OP><<<grid_blocks(n4), 256, 0, cudaStreamPerThread>>>(
            x, y, n4, n, p0, p1, flags);
    } else {
        k_fast_unary_f32_scalar<OP><<<grid_blocks(n), 256, 0, cudaStreamPerThread>>>(
            x, y, n, p0, p1, flags);
    }
}

template <int OP>
__device__ __forceinline__ float apply_binary_f32(float a, float b) {
    switch (OP) {
        case (int)FastBinaryOp::Pow:       return f_signed_powf(a, b);
        case (int)FastBinaryOp::Atan2:     return ::atan2f(a, b);
        case (int)FastBinaryOp::Add:       return a + b;
        case (int)FastBinaryOp::Sub:       return a - b;
        case (int)FastBinaryOp::Mul:       return a * b;
        case (int)FastBinaryOp::Div:       return a / b;
        case (int)FastBinaryOp::Maximum:   return (::isnan(a) || ::isnan(b)) ? ::nanf("") : (a > b ? a : b);
        case (int)FastBinaryOp::Minimum:   return (::isnan(a) || ::isnan(b)) ? ::nanf("") : (a < b ? a : b);
        case (int)FastBinaryOp::Hypot:     return ::hypotf(a, b);
        case (int)FastBinaryOp::LogAddExp: {
            if (a == b && ::isinf(a)) return a;
            float m = a > b ? a : b;
            float d = ::fabsf(a - b);
            return m + ::log1pf(::expf(-d));
        }
        default: return a;
    }
}

template <int OP>
__global__ void k_fast_binary_f32_vec4(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ z, int64_t n4, int64_t n
) {
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    const float4* __restrict__ y4 = reinterpret_cast<const float4*>(y);
    float4* __restrict__ z4 = reinterpret_cast<float4*>(z);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += (int64_t)gridDim.x * blockDim.x) {
        float4 a = x4[i];
        float4 b = y4[i];
        float4 r;
        r.x = apply_binary_f32<OP>(a.x, b.x);
        r.y = apply_binary_f32<OP>(a.y, b.y);
        r.z = apply_binary_f32<OP>(a.z, b.z);
        r.w = apply_binary_f32<OP>(a.w, b.w);
        z4[i] = r;
    }
    int64_t tail_base = n4 * 4;
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        int64_t idx = tail_base + threadIdx.x;
        if (idx < n) z[idx] = apply_binary_f32<OP>(x[idx], y[idx]);
    }
}

template <int OP>
__global__ void k_fast_binary_f32_scalar(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ z, int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        z[i] = apply_binary_f32<OP>(x[i], y[i]);
    }
}

template <int OP>
static void launch_binary(const float* x, const float* y, float* z, int64_t n) {
    if (n >= 4 && is_aligned16(x) && is_aligned16(y) && is_aligned16(z)) {
        int64_t n4 = n >> 2;
        k_fast_binary_f32_vec4<OP><<<grid_blocks(n4), 256, 0, cudaStreamPerThread>>>(x, y, z, n4, n);
    } else {
        k_fast_binary_f32_scalar<OP><<<grid_blocks(n), 256, 0, cudaStreamPerThread>>>(x, y, z, n);
    }
}

// =========================================================================
// Fast float4 warp-per-row var/std (ZERO inner-loop divisions!)
// =========================================================================
__global__ void k_var_std_warp_f32(
    const float* __restrict__ xp, float* __restrict__ out,
    int64_t rows, int64_t width, float correction, int take_sqrt
) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int64_t total_w = (int64_t)gridDim.x * wpb;
    const bool vec4_ok = ((width & 3) == 0) && is_aligned16(xp);
    const int64_t w4 = width >> 2;

    for (int64_t row = (int64_t)blockIdx.x * wpb + wid; row < rows; row += total_w) {
        const float* r = xp + row * width;
        float sum = 0.0f;
        if (vec4_ok) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int64_t c = lane; c < w4; c += 32) {
                float4 v = r4[c];
                sum += (v.x + v.y) + (v.z + v.w);
            }
        } else {
            for (int64_t c = lane; c < width; c += 32) sum += r[c];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        float mean = __shfl_sync(0xffffffffu, sum, 0) / (float)width;

        float sq = 0.0f;
        if (vec4_ok) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int64_t c = lane; c < w4; c += 32) {
                float4 v = r4[c];
                float dx = v.x - mean, dy = v.y - mean, dz = v.z - mean, dw = v.w - mean;
                sq += dx * dx + dy * dy + dz * dz + dw * dw;
            }
        } else {
            for (int64_t c = lane; c < width; c += 32) {
                float d = r[c] - mean;
                sq += d * d;
            }
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, d);
        if (lane == 0) {
            float denom = (float)width - correction;
            float v = denom > 0.0f ? (sq / denom) : 0.0f;
            if (v < 0.0f) v = 0.0f;
            out[row] = take_sqrt ? ::sqrtf(v) : v;
        }
    }
}

__global__ void k_var_std_block_f32(
    const float* __restrict__ xp, float* __restrict__ out,
    int64_t rows, int64_t width, float correction, int take_sqrt
) {
    __shared__ float s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* r = xp + row * width;
        float sum = 0.0f;
        for (int64_t c = tid; c < width; c += 256) sum += r[c];
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (lane == 0) s[wid] = sum;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? s[lane] : 0.0f;
            #pragma unroll
            for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
            if (lane == 0) s[0] = v / (float)width;
        }
        __syncthreads();
        float mean = s[0];
        float sq = 0.0f;
        for (int64_t c = tid; c < width; c += 256) {
            float d = r[c] - mean;
            sq += d * d;
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, d);
        if (lane == 0) s[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? s[lane] : 0.0f;
            #pragma unroll
            for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
            if (lane == 0) {
                float denom = (float)width - correction;
                float res = denom > 0.0f ? (v / denom) : 0.0f;
                if (res < 0.0f) res = 0.0f;
                out[row] = take_sqrt ? ::sqrtf(res) : res;
            }
        }
        __syncthreads();
    }
}

__global__ void k_prod_warp_f32(
    const float* __restrict__ xp, float* __restrict__ out, int64_t rows, int64_t width
) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int64_t total_w = (int64_t)gridDim.x * wpb;
    const bool vec4_ok = ((width & 3) == 0) && is_aligned16(xp);
    const int64_t w4 = width >> 2;
    for (int64_t row = (int64_t)blockIdx.x * wpb + wid; row < rows; row += total_w) {
        const float* r = xp + row * width;
        float acc = 1.0f;
        if (vec4_ok) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int64_t c = lane; c < w4; c += 32) {
                float4 v = r4[c];
                acc *= (v.x * v.y) * (v.z * v.w);
            }
        } else {
            for (int64_t c = lane; c < width; c += 32) acc *= r[c];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) acc *= __shfl_down_sync(0xffffffffu, acc, d);
        if (lane == 0) out[row] = acc;
    }
}

template <int IS_MIN>
__global__ void k_arg_reduce_warp_f32(
    const float* __restrict__ xp, float* __restrict__ val_out,
    int64_t* __restrict__ idx_out, int64_t rows, int64_t width
) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int64_t total_w = (int64_t)gridDim.x * wpb;
    const bool vec4_ok = ((width & 3) == 0) && is_aligned16(xp);
    const int64_t w4 = width >> 2;

    for (int64_t row = (int64_t)blockIdx.x * wpb + wid; row < rows; row += total_w) {
        const float* r = xp + row * width;
        bool has_v = false;
        float best_v = 0.0f;
        int64_t best_i = 0;

        auto update = [&](float v, int64_t i) {
            if (!has_v) {
                best_v = v; best_i = i; has_v = true;
            } else {
                bool better = IS_MIN ? (v < best_v) : (v > best_v);
                if (better) { best_v = v; best_i = i; }
            }
        };

        if (vec4_ok) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int64_t c = lane; c < w4; c += 32) {
                float4 v = r4[c];
                int64_t base = c << 2;
                update(v.x, base);
                update(v.y, base + 1);
                update(v.z, base + 2);
                update(v.w, base + 3);
            }
        } else {
            for (int64_t c = lane; c < width; c += 32) update(r[c], c);
        }

        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best_v, d);
            int64_t oi = __shfl_down_sync(0xffffffffu, best_i, d);
            int oh = __shfl_down_sync(0xffffffffu, (int)has_v, d);
            if (oh) {
                if (!has_v) {
                    best_v = ov; best_i = oi; has_v = true;
                } else {
                    bool better = IS_MIN ? ((ov < best_v) || (ov == best_v && oi < best_i))
                                         : ((ov > best_v) || (ov == best_v && oi < best_i));
                    if (better) { best_v = ov; best_i = oi; }
                }
            }
        }
        if (lane == 0) {
            val_out[row] = best_v;
            idx_out[row] = best_i;
        }
    }
}

template <int IS_PROD>
__global__ void k_scan_warp_f32(
    const float* __restrict__ xp, float* __restrict__ yp, int64_t rows, int64_t width
) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int64_t total_w = (int64_t)gridDim.x * wpb;
    for (int64_t row = (int64_t)blockIdx.x * wpb + wid; row < rows; row += total_w) {
        const int64_t row_start = row * width;
        float carry = IS_PROD ? 1.0f : 0.0f;
        for (int64_t tile = 0; tile < width; tile += 32) {
            int64_t col = tile + lane;
            bool active = col < width;
            float v = active ? xp[row_start + col] : (IS_PROD ? 1.0f : 0.0f);
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, v, d);
                if (lane >= d) v = IS_PROD ? (v * n) : (v + n);
            }
            v = IS_PROD ? (v * carry) : (v + carry);
            if (active) yp[row_start + col] = v;
            carry = __shfl_sync(0xffffffffu, v, 31);
        }
    }
}

template <int IS_PROD>
__global__ void k_scan_strided_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t outer, int64_t dim_size, int64_t inner
) {
    const int64_t total_fibers = outer * inner;
    for (int64_t f = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         f < total_fibers;
         f += (int64_t)gridDim.x * blockDim.x) {
        int64_t o = f / inner;
        int64_t in_i = f - o * inner;
        int64_t base = o * dim_size * inner + in_i;
        float acc = IS_PROD ? 1.0f : 0.0f;
        for (int64_t k = 0; k < dim_size; ++k) {
            float v = xp[base + k * inner];
            acc = IS_PROD ? (acc * v) : (acc + v);
            yp[base + k * inner] = acc;
        }
    }
}

// =========================================================================
// Fast float4-vectorized Cross-Entropy, NLL, BCE-with-Logits
// =========================================================================
template <typename TType>
__global__ void k_ce_fwd_f32(
    const float* __restrict__ xp,
    const TType* __restrict__ tp,
    const float* __restrict__ wp,
    float* __restrict__ loss_p,
    float* __restrict__ rw_p,
    int64_t N, int64_t C,
    int has_ignore, int64_t ignore_idx,
    IndexFault* fault
) {
    __shared__ float s_buf[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const bool vec4_ok = ((C & 3) == 0) && is_aligned16(xp);
    const int64_t C4 = C >> 2;

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
        if (ignored) {
            if (tid == 0) {
                loss_p[row] = 0.0f;
                if (rw_p) rw_p[row] = 0.0f;
            }
            continue;
        }
        const float* rx = xp + row * C;
        // Pass 1: branch-free float4 max
        float m = -CUDART_INF_F;
        if (vec4_ok) {
            const float4* rx4 = reinterpret_cast<const float4*>(rx);
            for (int64_t c = tid; c < C4; c += 256) {
                float4 v = rx4[c];
                float m0 = v.x > v.y ? v.x : v.y;
                float m1 = v.z > v.w ? v.z : v.w;
                float mv = m0 > m1 ? m0 : m1;
                if (mv > m) m = mv;
            }
        } else {
            for (int64_t c = tid; c < C; c += 256) {
                float v = rx[c];
                if (v > m) m = v;
            }
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            float o = __shfl_down_sync(0xffffffffu, m, d);
            if (o > m) m = o;
        }
        if (lane == 0) s_buf[wid] = m;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? s_buf[lane] : -CUDART_INF_F;
            #pragma unroll
            for (int d = 4; d > 0; d >>= 1) {
                float o = __shfl_down_sync(0xffffffffu, v, d);
                if (o > v) v = o;
            }
            if (lane == 0) s_buf[0] = v;
        }
        __syncthreads();
        float row_max = s_buf[0];

        // Pass 2: branch-free float4 exp sum (hits L1/L2 cache!)
        float s = 0.0f;
        if (vec4_ok) {
            const float4* rx4 = reinterpret_cast<const float4*>(rx);
            for (int64_t c = tid; c < C4; c += 256) {
                float4 v = rx4[c];
                s += ::expf(v.x - row_max) + ::expf(v.y - row_max)
                   + ::expf(v.z - row_max) + ::expf(v.w - row_max);
            }
        } else {
            for (int64_t c = tid; c < C; c += 256) {
                s += ::expf(rx[c] - row_max);
            }
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d);
        if (lane == 0) s_buf[wid] = s;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? s_buf[lane] : 0.0f;
            #pragma unroll
            for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
            if (lane == 0) {
                float w = wp ? wp[t] : 1.0f;
                float log_s = ::logf(v);
                loss_p[row] = (log_s - (rx[t] - row_max)) * w;
                if (rw_p) rw_p[row] = w;
            }
        }
        __syncthreads();
    }
}

__global__ void k_loss_reduce_f32(
    const float* __restrict__ losses,
    const float* __restrict__ weights,
    float* __restrict__ out_scalar,
    int64_t N, int is_mean
) {
    __shared__ float sl[8];
    __shared__ float sw[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float ls = 0.0f, ws = 0.0f;
    for (int64_t i = tid; i < N; i += 256) {
        ls += losses[i];
        ws += weights[i];
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        ls += __shfl_down_sync(0xffffffffu, ls, d);
        ws += __shfl_down_sync(0xffffffffu, ws, d);
    }
    if (lane == 0) { sl[wid] = ls; sw[wid] = ws; }
    __syncthreads();
    if (wid == 0) {
        ls = (lane < 8) ? sl[lane] : 0.0f;
        ws = (lane < 8) ? sw[lane] : 0.0f;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) {
            ls += __shfl_down_sync(0xffffffffu, ls, d);
            ws += __shfl_down_sync(0xffffffffu, ws, d);
        }
        if (lane == 0) out_scalar[0] = is_mean ? (ls / ws) : ls;
    }
}

template <typename TType>
__global__ void k_nll_fwd_f32(
    const float* __restrict__ xp, const TType* __restrict__ tp,
    const float* __restrict__ wp, float* __restrict__ loss_p,
    float* __restrict__ rw_p, int64_t N, int64_t C,
    int64_t ignore_idx, IndexFault* fault
) {
    for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < N;
         row += (int64_t)gridDim.x * blockDim.x) {
        int64_t t = (int64_t)tp[row];
        if (t == ignore_idx) {
            loss_p[row] = 0.0f;
            if (rw_p) rw_p[row] = 0.0f;
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
        float w = wp ? wp[t] : 1.0f;
        loss_p[row] = -xp[row * C + t] * w;
        if (rw_p) rw_p[row] = w;
    }
}

__device__ __forceinline__ float bce_logits_one(float x, float t) {
    if (x < -50.0f) x = -50.0f;
    else if (x > 50.0f) x = 50.0f;
    float mv = (-x > 0.0f) ? -x : 0.0f;
    return (1.0f - t) * x + mv + ::log1pf(::expf(-::fabsf(x)));
}

__global__ void k_bce_elem_f32(
    const float* __restrict__ xp, const float* __restrict__ tp,
    float* __restrict__ out, int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = bce_logits_one(xp[i], tp[i]);
    }
}

__global__ void k_bce_reduce_f32(
    const float* __restrict__ xp, const float* __restrict__ tp,
    float* __restrict__ part, int64_t n
) {
    __shared__ float s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float acc = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + tid;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        acc += bce_logits_one(xp[i], tp[i]);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s[lane] : 0.0f;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
        if (lane == 0) part[blockIdx.x] = v;
    }
}

__global__ void k_sum_finalize_f32(
    const float* __restrict__ part, float* __restrict__ out,
    int64_t count, float inv_scale
) {
    __shared__ float s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float acc = 0.0f;
    for (int64_t i = tid; i < count; i += 256) acc += part[i];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s[lane] : 0.0f;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
        if (lane == 0) out[0] = v * inv_scale;
    }
}

// =========================================================================
// Fast Pooling, Grid Sample, Bilinear Resize, Affine Grid, Triu/Tril, Trace
// =========================================================================
__global__ void k_aap2d_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t nc = idx / oHW;
        int64_t rem = idx - nc * oHW;
        int64_t oh = rem / oW;
        int64_t ow = rem - oh * oW;
        int64_t h0 = (oh * H) / oH;
        int64_t h1 = ((oh + 1) * H + oH - 1) / oH;
        int64_t w0 = (ow * W) / oW;
        int64_t w1 = ((ow + 1) * W + oW - 1) / oW;
        const float* plane = xp + nc * HW;
        float sum = 0.0f;
        for (int64_t ih = h0; ih < h1; ++ih) {
            int64_t ro = ih * W;
            for (int64_t iw = w0; iw < w1; ++iw) sum += plane[ro + iw];
        }
        yp[idx] = sum / (float)((h1 - h0) * (w1 - w0));
    }
}

__global__ void k_avgpool2d_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW, int count_pad
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t nc = idx / oHW;
        int64_t rem = idx - nc * oHW;
        int64_t oh = rem / oW;
        int64_t ow = rem - oh * oW;
        int64_t hs = oh * sH - pH, ws = ow * sW - pW;
        int64_t he_p = hs + kH < H + pH ? hs + kH : H + pH;
        int64_t we_p = ws + kW < W + pW ? ws + kW : W + pW;
        int64_t h0 = hs > 0 ? hs : 0, w0 = ws > 0 ? ws : 0;
        int64_t h1 = hs + kH < H ? hs + kH : H, w1 = ws + kW < W ? ws + kW : W;
        const float* plane = xp + nc * HW;
        float sum = 0.0f;
        for (int64_t ih = h0; ih < h1; ++ih) {
            int64_t ro = ih * W;
            for (int64_t iw = w0; iw < w1; ++iw) sum += plane[ro + iw];
        }
        int64_t dh = count_pad ? (he_p - hs) : (h1 - h0);
        int64_t dw = count_pad ? (we_p - ws) : (w1 - w0);
        if (dh < 1) dh = 1;
        if (dw < 1) dw = 1;
        yp[idx] = sum / (float)(dh * dw);
    }
}

__global__ void k_avgpool3d_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int count_pad
) {
    const int64_t oHW = oH * oW, oDHW = oD * oHW;
    const int64_t HW = H * W, DHW = D * HW;
    const int64_t total = N * C * oDHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t nc = idx / oDHW;
        int64_t rem = idx - nc * oDHW;
        int64_t od = rem / oHW; rem -= od * oHW;
        int64_t oh = rem / oW; int64_t ow = rem - oh * oW;
        int64_t ds = od * sD - pD, hs = oh * sH - pH, ws = ow * sW - pW;
        int64_t de_p = ds + kD < D + pD ? ds + kD : D + pD;
        int64_t he_p = hs + kH < H + pH ? hs + kH : H + pH;
        int64_t we_p = ws + kW < W + pW ? ws + kW : W + pW;
        int64_t d0 = ds > 0 ? ds : 0, h0 = hs > 0 ? hs : 0, w0 = ws > 0 ? ws : 0;
        int64_t d1 = ds + kD < D ? ds + kD : D;
        int64_t h1 = hs + kH < H ? hs + kH : H;
        int64_t w1 = ws + kW < W ? ws + kW : W;
        const float* vol = xp + nc * DHW;
        float sum = 0.0f;
        for (int64_t id = d0; id < d1; ++id) {
            int64_t dof = id * HW;
            for (int64_t ih = h0; ih < h1; ++ih) {
                int64_t ro = dof + ih * W;
                for (int64_t iw = w0; iw < w1; ++iw) sum += vol[ro + iw];
            }
        }
        int64_t dd = count_pad ? (de_p - ds) : (d1 - d0);
        int64_t dh = count_pad ? (he_p - hs) : (h1 - h0);
        int64_t dw = count_pad ? (we_p - ws) : (w1 - w0);
        if (dd < 1) dd = 1;
        if (dh < 1) dh = 1;
        if (dw < 1) dw = 1;
        yp[idx] = sum / (float)(dd * dh * dw);
    }
}

__device__ __forceinline__ float gs_coord_f32(float c, int64_t sz, int pad, int align) {
    float v = align ? (((c + 1.0f) * 0.5f) * (float)(sz - 1))
                    : (((c + 1.0f) * (float)sz - 1.0f) * 0.5f);
    if (pad == 1) {
        if (v < 0.0f) v = 0.0f;
        if (v > (float)(sz - 1)) v = (float)(sz - 1);
    } else if (pad == 2) {
        float tlo = align ? 0.0f : -1.0f;
        float thi = align ? (float)(2 * (sz - 1)) : (float)(2 * sz - 1);
        if (tlo == thi) {
            v = 0.0f;
        } else {
            float m = tlo * 0.5f;
            float span = (thi - tlo) * 0.5f;
            float a = ::fabsf(v - m);
            float ex = ::fmodf(a, span);
            int64_t fl = (int64_t)::floorf(a / span);
            v = ((fl & 1LL) == 0) ? (ex + m) : (span - ex + m);
        }
        if (v < 0.0f) v = 0.0f;
        if (v > (float)(sz - 1)) v = (float)(sz - 1);
    }
    return v;
}

__global__ void k_grid_sample_2d_f32(
    const float* __restrict__ xp, const float* __restrict__ gp,
    float* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int mode, int pad, int align
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t nc = idx / oHW;
        int64_t hw = idx - nc * oHW;
        int64_t n = nc / C;
        int64_t go = (n * oHW + hw) * 2;
        float ix = gs_coord_f32(gp[go], W, pad, align);
        float iy = gs_coord_f32(gp[go + 1], H, pad, align);
        const float* plane = xp + nc * HW;
        if (mode == 1) {
            int64_t ixn = (int64_t)::rintf(ix), iyn = (int64_t)::rintf(iy);
            yp[idx] = (ixn >= 0 && ixn < W && iyn >= 0 && iyn < H) ? plane[iyn * W + ixn] : 0.0f;
        } else {
            int64_t ix0 = (int64_t)::floorf(ix), iy0 = (int64_t)::floorf(iy);
            int64_t ix1 = ix0 + 1, iy1 = iy0 + 1;
            float dx = ix - (float)ix0, dy = iy - (float)iy0;
            float dnx = (float)ix1 - ix, dny = (float)iy1 - iy;
            float val = 0.0f;
            if (iy0 >= 0 && iy0 < H && ix0 >= 0 && ix0 < W) val += plane[iy0 * W + ix0] * dnx * dny;
            if (iy1 >= 0 && iy1 < H && ix0 >= 0 && ix0 < W) val += plane[iy1 * W + ix0] * dnx * dy;
            if (iy0 >= 0 && iy0 < H && ix1 >= 0 && ix1 < W) val += plane[iy0 * W + ix1] * dx * dny;
            if (iy1 >= 0 && iy1 < H && ix1 >= 0 && ix1 < W) val += plane[iy1 * W + ix1] * dx * dy;
            yp[idx] = val;
        }
    }
}

__global__ void k_resize_bilinear_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int align, int tf_mode
) {
    const int64_t oHW = oH * oW, HW = H * W, total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t nc = idx / oHW;
        int64_t rem = idx - nc * oHW;
        int64_t oh = rem / oW, ow = rem - oh * oW;
        float x, y;
        if (align) {
            x = ((float)oh * (float)(H - 1)) / (float)(oH > 1 ? oH - 1 : 1);
            y = ((float)ow * (float)(W - 1)) / (float)(oW > 1 ? oW - 1 : 1);
        } else if (tf_mode) {
            x = ((float)oh * (float)H) / (float)oH;
            y = ((float)ow * (float)W) / (float)oW;
            if (oH > H) { if (x < 0.0f) x = 0.0f; if (x > (float)(H - 1)) x = (float)(H - 1); }
            if (oW > W) { if (y < 0.0f) y = 0.0f; if (y > (float)(W - 1)) y = (float)(W - 1); }
        } else {
            x = (((float)oh + 0.5f) * (float)H) / (float)oH - 0.5f;
            y = (((float)ow + 0.5f) * (float)W) / (float)oW - 0.5f;
            if (oH > H) { if (x < 0.0f) x = 0.0f; if (x > (float)(H - 1)) x = (float)(H - 1); }
            if (oW > W) { if (y < 0.0f) y = 0.0f; if (y > (float)(W - 1)) y = (float)(W - 1); }
        }
        int64_t fx = (int64_t)::floorf(x), fy = (int64_t)::floorf(y);
        int64_t cx = fx + 1, cy = fy + 1;
        float dx = x - (float)fx, dy = y - (float)fy;
        float dnx = 1.0f - dx, dny = 1.0f - dy;
        if (fx < 0) fx = 0; else if (fx >= H) fx = H - 1;
        if (cx < 0) cx = 0; else if (cx >= H) cx = H - 1;
        if (fy < 0) fy = 0; else if (fy >= W) fy = W - 1;
        if (cy < 0) cy = 0; else if (cy >= W) cy = W - 1;
        const float* plane = xp + nc * HW;
        float ab = dx * plane[cx * W + fy] + dnx * plane[fx * W + fy];
        float cd = dx * plane[cx * W + cy] + dnx * plane[fx * W + cy];
        yp[idx] = ab * dny + cd * dy;
    }
}

__global__ void k_affine_grid_4d_f32(
    const float* __restrict__ theta, float* __restrict__ out,
    int64_t N, int64_t H, int64_t W, int align
) {
    const int64_t HW = H * W, total = N * HW;
    const float sx = (W > 1) ? (2.0f / (float)(W - 1)) : 0.0f;
    const float sy = (H > 1) ? (2.0f / (float)(H - 1)) : 0.0f;
    const float fx = align ? 1.0f : ((float)(W - 1) / (float)W);
    const float fy = align ? 1.0f : ((float)(H - 1) / (float)H);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t n = idx / HW;
        int64_t rem = idx - n * HW;
        int64_t h = rem / W, w = rem - h * W;
        float x = (W > 1) ? (((float)w * sx - 1.0f) * fx) : 0.0f;
        float y = (H > 1) ? (((float)h * sy - 1.0f) * fy) : 0.0f;
        const float* th = theta + n * 6;
        out[idx * 2]     = th[0] * x + th[1] * y + th[2];
        out[idx * 2 + 1] = th[3] * x + th[4] * y + th[5];
    }
}

__global__ void k_triu_tril_f32(
    const float* __restrict__ xp, float* __restrict__ yp,
    int64_t batch, int64_t rows, int64_t cols, int64_t diag, int is_lower
) {
    const int64_t rc = rows * cols;
    const int64_t total = batch * rc;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t rem = idx % rc;
        int64_t r = rem / cols;
        int64_t c = rem - r * cols;
        bool keep = is_lower ? (r >= c - diag) : (r <= c - diag);
        yp[idx] = keep ? xp[idx] : 0.0f;
    }
}

__global__ void k_trace_f32(
    const float* __restrict__ xp, float* __restrict__ out,
    int64_t size, int64_t stride
) {
    __shared__ float s[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float acc = 0.0f;
    for (int64_t i = tid; i < size; i += 256) acc += xp[i * stride];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s[lane] : 0.0f;
        #pragma unroll
        for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
        if (lane == 0) out[0] = v;
    }
}

} // namespace

void cuda_fast_unary_f32(
    FastUnaryOp op, const float* x, float* y, int64_t n,
    float p0, float p1, int flags
) {
    if (n <= 0) return;
    switch (op) {
        case FastUnaryOp::Exp:         launch_unary<(int)FastUnaryOp::Exp>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Log:         launch_unary<(int)FastUnaryOp::Log>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Log1p:       launch_unary<(int)FastUnaryOp::Log1p>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Log2:        launch_unary<(int)FastUnaryOp::Log2>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Exp2:        launch_unary<(int)FastUnaryOp::Exp2>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Expm1:       launch_unary<(int)FastUnaryOp::Expm1>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Sqrt:        launch_unary<(int)FastUnaryOp::Sqrt>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Rsqrt:       launch_unary<(int)FastUnaryOp::Rsqrt>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Reciprocal:  launch_unary<(int)FastUnaryOp::Reciprocal>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Abs:         launch_unary<(int)FastUnaryOp::Abs>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Neg:         launch_unary<(int)FastUnaryOp::Neg>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Sin:         launch_unary<(int)FastUnaryOp::Sin>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Cos:         launch_unary<(int)FastUnaryOp::Cos>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Tan:         launch_unary<(int)FastUnaryOp::Tan>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Tanh:        launch_unary<(int)FastUnaryOp::Tanh>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Sigmoid:     launch_unary<(int)FastUnaryOp::Sigmoid>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Relu:        launch_unary<(int)FastUnaryOp::Relu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Erf:         launch_unary<(int)FastUnaryOp::Erf>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Floor:       launch_unary<(int)FastUnaryOp::Floor>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Ceil:        launch_unary<(int)FastUnaryOp::Ceil>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Round:       launch_unary<(int)FastUnaryOp::Round>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Trunc:       launch_unary<(int)FastUnaryOp::Trunc>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Sign:        launch_unary<(int)FastUnaryOp::Sign>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Deg2Rad:     launch_unary<(int)FastUnaryOp::Deg2Rad>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Rad2Deg:     launch_unary<(int)FastUnaryOp::Rad2Deg>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Lgamma:      launch_unary<(int)FastUnaryOp::Lgamma>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Digamma:     launch_unary<(int)FastUnaryOp::Digamma>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Clamp:       launch_unary<(int)FastUnaryOp::Clamp>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Hardswish:   launch_unary<(int)FastUnaryOp::Hardswish>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Hardsigmoid: launch_unary<(int)FastUnaryOp::Hardsigmoid>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Hardtanh:    launch_unary<(int)FastUnaryOp::Hardtanh>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Elu:         launch_unary<(int)FastUnaryOp::Elu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Celu:        launch_unary<(int)FastUnaryOp::Celu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Selu:        launch_unary<(int)FastUnaryOp::Selu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Softplus:    launch_unary<(int)FastUnaryOp::Softplus>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Mish:        launch_unary<(int)FastUnaryOp::Mish>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Logsigmoid:  launch_unary<(int)FastUnaryOp::Logsigmoid>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Gelu:        launch_unary<(int)FastUnaryOp::Gelu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::Silu:        launch_unary<(int)FastUnaryOp::Silu>(x, y, n, p0, p1, flags); break;
        case FastUnaryOp::LeakyRelu:   launch_unary<(int)FastUnaryOp::LeakyRelu>(x, y, n, p0, p1, flags); break;
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_binary_f32(
    FastBinaryOp op, const float* x, const float* y, float* z, int64_t n
) {
    if (n <= 0) return;
    switch (op) {
        case FastBinaryOp::Pow:       launch_binary<(int)FastBinaryOp::Pow>(x, y, z, n); break;
        case FastBinaryOp::Atan2:     launch_binary<(int)FastBinaryOp::Atan2>(x, y, z, n); break;
        case FastBinaryOp::Add:       launch_binary<(int)FastBinaryOp::Add>(x, y, z, n); break;
        case FastBinaryOp::Sub:       launch_binary<(int)FastBinaryOp::Sub>(x, y, z, n); break;
        case FastBinaryOp::Mul:       launch_binary<(int)FastBinaryOp::Mul>(x, y, z, n); break;
        case FastBinaryOp::Div:       launch_binary<(int)FastBinaryOp::Div>(x, y, z, n); break;
        case FastBinaryOp::Maximum:   launch_binary<(int)FastBinaryOp::Maximum>(x, y, z, n); break;
        case FastBinaryOp::Minimum:   launch_binary<(int)FastBinaryOp::Minimum>(x, y, z, n); break;
        case FastBinaryOp::Hypot:     launch_binary<(int)FastBinaryOp::Hypot>(x, y, z, n); break;
        case FastBinaryOp::LogAddExp: launch_binary<(int)FastBinaryOp::LogAddExp>(x, y, z, n); break;
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_var_std_lastdim_f32(
    const float* x, float* out, int64_t rows, int64_t width,
    float correction, int take_sqrt
) {
    if (rows <= 0 || width <= 0) return;
    if (width <= 1024 && rows >= 8) {
        int blocks = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
        k_var_std_warp_f32<<<blocks, 256, 0, cudaStreamPerThread>>>(x, out, rows, width, correction, take_sqrt);
    } else {
        int blocks = static_cast<int>(std::min<int64_t>(rows, 65535));
        k_var_std_block_f32<<<blocks, 256, 0, cudaStreamPerThread>>>(x, out, rows, width, correction, take_sqrt);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_prod_lastdim_f32(
    const float* x, float* out, int64_t rows, int64_t width
) {
    if (rows <= 0 || width <= 0) return;
    int blocks = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
    k_prod_warp_f32<<<blocks, 256, 0, cudaStreamPerThread>>>(x, out, rows, width);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_arg_reduce_lastdim_f32(
    const float* x, float* val_out, int64_t* idx_out,
    int64_t rows, int64_t width, int is_min
) {
    if (rows <= 0 || width <= 0) return;
    int blocks = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
    if (is_min) {
        k_arg_reduce_warp_f32<1><<<blocks, 256, 0, cudaStreamPerThread>>>(x, val_out, idx_out, rows, width);
    } else {
        k_arg_reduce_warp_f32<0><<<blocks, 256, 0, cudaStreamPerThread>>>(x, val_out, idx_out, rows, width);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_scan_f32(
    const float* x, float* y, int64_t outer, int64_t dim_size,
    int64_t inner, int is_prod
) {
    if (outer <= 0 || dim_size <= 0 || inner <= 0) return;
    if (inner == 1 && dim_size <= 256) {
        int blocks = static_cast<int>(std::min<int64_t>((outer + 7) / 8, 65535));
        if (is_prod) k_scan_warp_f32<1><<<blocks, 256, 0, cudaStreamPerThread>>>(x, y, outer, dim_size);
        else         k_scan_warp_f32<0><<<blocks, 256, 0, cudaStreamPerThread>>>(x, y, outer, dim_size);
    } else {
        int64_t fibers = outer * inner;
        if (is_prod) k_scan_strided_f32<1><<<grid_blocks(fibers), 256, 0, cudaStreamPerThread>>>(x, y, outer, dim_size, inner);
        else         k_scan_strided_f32<0><<<grid_blocks(fibers), 256, 0, cudaStreamPerThread>>>(x, y, outer, dim_size, inner);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_cross_entropy_f32(
    const float* x, const void* target, int target_is_i64,
    const float* weight, float* out, float* temp_losses, float* temp_weights,
    int64_t N, int64_t C, int has_ignore, int64_t ignore_idx,
    int red_mode, IndexFault* fault
) {
    if (N <= 0 || C <= 0) return;
    int blocks = static_cast<int>(std::min<int64_t>(N, 65535));
    float* loss_dst = (red_mode == 0) ? out : temp_losses;
    float* rw_dst = (red_mode == 0) ? nullptr : temp_weights;
    if (target_is_i64) {
        k_ce_fwd_f32<int64_t><<<blocks, 256, 0, cudaStreamPerThread>>>(
            x, static_cast<const int64_t*>(target), weight, loss_dst, rw_dst,
            N, C, has_ignore, ignore_idx, fault);
    } else {
        k_ce_fwd_f32<int32_t><<<blocks, 256, 0, cudaStreamPerThread>>>(
            x, static_cast<const int32_t*>(target), weight, loss_dst, rw_dst,
            N, C, has_ignore, ignore_idx, fault);
    }
    if (red_mode != 0) {
        k_loss_reduce_f32<<<1, 256, 0, cudaStreamPerThread>>>(
            temp_losses, temp_weights, out, N, red_mode == 2 ? 1 : 0);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_nll_loss_f32(
    const float* x, const void* target, int target_is_i64,
    const float* weight, float* out, float* temp_losses, float* temp_weights,
    int64_t N, int64_t C, int64_t ignore_idx,
    int red_mode, IndexFault* fault
) {
    if (N <= 0 || C <= 0) return;
    float* loss_dst = (red_mode == 0) ? out : temp_losses;
    float* rw_dst = (red_mode == 0) ? nullptr : temp_weights;
    if (target_is_i64) {
        k_nll_fwd_f32<int64_t><<<grid_blocks(N), 256, 0, cudaStreamPerThread>>>(
            x, static_cast<const int64_t*>(target), weight, loss_dst, rw_dst,
            N, C, ignore_idx, fault);
    } else {
        k_nll_fwd_f32<int32_t><<<grid_blocks(N), 256, 0, cudaStreamPerThread>>>(
            x, static_cast<const int32_t*>(target), weight, loss_dst, rw_dst,
            N, C, ignore_idx, fault);
    }
    if (red_mode != 0) {
        k_loss_reduce_f32<<<1, 256, 0, cudaStreamPerThread>>>(
            temp_losses, temp_weights, out, N, red_mode == 2 ? 1 : 0);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_bce_with_logits_f32(
    const float* x, const float* target, float* out, float* part_buf,
    int64_t n, int n_blocks, int red_mode
) {
    if (n <= 0) return;
    if (red_mode == 0) {
        k_bce_elem_f32<<<grid_blocks(n), 256, 0, cudaStreamPerThread>>>(x, target, out, n);
    } else {
        k_bce_reduce_f32<<<n_blocks, 256, 0, cudaStreamPerThread>>>(x, target, part_buf, n);
        float inv_scale = (red_mode == 2) ? (1.0f / (float)n) : 1.0f;
        k_sum_finalize_f32<<<1, 256, 0, cudaStreamPerThread>>>(part_buf, out, n_blocks, inv_scale);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_adaptive_avg_pool2d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW
) {
    int64_t total = N * C * oH * oW;
    if (total <= 0) return;
    k_aap2d_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(x, y, N, C, H, W, oH, oW);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_avg_pool2d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int count_pad
) {
    int64_t total = N * C * oH * oW;
    if (total <= 0) return;
    k_avgpool2d_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        x, y, N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, count_pad);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_avg_pool3d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int count_pad
) {
    int64_t total = N * C * oD * oH * oW;
    if (total <= 0) return;
    k_avgpool3d_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        x, y, N, C, D, H, W, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_pad);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_grid_sample_2d_f32(
    const float* x, const float* grid, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int mode, int padding_mode, int align_corners
) {
    int64_t total = N * C * oH * oW;
    if (total <= 0) return;
    k_grid_sample_2d_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        x, grid, y, N, C, H, W, oH, oW, mode, padding_mode, align_corners);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_resize_bilinear_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int align_corners, int tf_mode
) {
    int64_t total = N * C * oH * oW;
    if (total <= 0) return;
    k_resize_bilinear_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        x, y, N, C, H, W, oH, oW, align_corners, tf_mode);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_affine_grid_4d_f32(
    const float* theta, float* out,
    int64_t N, int64_t H, int64_t W, int align_corners
) {
    int64_t total = N * H * W;
    if (total <= 0) return;
    k_affine_grid_4d_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        theta, out, N, H, W, align_corners);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_triu_tril_f32(
    const float* x, float* y,
    int64_t batch, int64_t rows, int64_t cols, int64_t diagonal, int is_lower
) {
    int64_t total = batch * rows * cols;
    if (total <= 0) return;
    k_triu_tril_f32<<<grid_blocks(total), 256, 0, cudaStreamPerThread>>>(
        x, y, batch, rows, cols, diagonal, is_lower);
    checkCudaErrors(cudaGetLastError());
}

void cuda_fast_trace_f32(
    const float* x, float* out, int64_t size, int64_t stride
) {
    if (size <= 0) return;
    k_trace_f32<<<1, 256, 0, cudaStreamPerThread>>>(x, out, size, stride);
    checkCudaErrors(cudaGetLastError());
}

} // namespace jittor
