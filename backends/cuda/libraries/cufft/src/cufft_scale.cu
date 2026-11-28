// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cuda_runtime.h>
#include <cstdint>
#include "helper_cuda.h"

namespace jittor {

namespace {

static inline int cufft_grid(int64_t n, int threads = 256) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

__global__ void k_cufft_scale_f32_vec4(float* __restrict__ p, int64_t n4, int64_t n, float s) {
    float4* p4 = reinterpret_cast<float4*>(p);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += (int64_t)gridDim.x * blockDim.x) {
        float4 v = p4[i];
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        p4[i] = v;
    }
    int64_t tail = n4 * 4;
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        int64_t idx = tail + threadIdx.x;
        if (idx < n) p[idx] *= s;
    }
}

__global__ void k_cufft_scale_f32(float* __restrict__ p, int64_t n, float s) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        p[i] *= s;
    }
}

__global__ void k_cufft_scale_f64(double* __restrict__ p, int64_t n, double s) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        p[i] *= s;
    }
}

// Expand R2C half-spectrum [batch, keep] into full complex spectrum [batch, L]
// with optional inverse conjugate sign flip (for ifft of real input) and scaling.
template <typename T>
__global__ void k_cufft_expand_hermitian(
    const T* __restrict__ half, T* __restrict__ full,
    int64_t batch, int64_t keep, int64_t L, T scale, int conj_all
) {
    const int64_t total = batch * L;
    const T im_sign = conj_all ? (T)(-1) : (T)1;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t b = idx / L;
        int64_t k = idx - b * L;
        int64_t dst = idx * 2;
        if (k < keep) {
            int64_t src = (b * keep + k) * 2;
            full[dst] = half[src] * scale;
            full[dst + 1] = half[src + 1] * scale * im_sign;
        } else {
            int64_t mk = L - k;
            int64_t src = (b * keep + mk) * 2;
            full[dst] = half[src] * scale;
            full[dst + 1] = -half[src + 1] * scale * im_sign;
        }
    }
}

} // namespace

void cufft_scale_inplace_f32(float* p, int64_t n, float s, cudaStream_t stream) {
    if (n <= 0) return;
    if (n >= 4 && ((reinterpret_cast<uintptr_t>(p) & 15u) == 0)) {
        int64_t n4 = n >> 2;
        k_cufft_scale_f32_vec4<<<cufft_grid(n4), 256, 0, stream>>>(p, n4, n, s);
    } else {
        k_cufft_scale_f32<<<cufft_grid(n), 256, 0, stream>>>(p, n, s);
    }
    checkCudaErrors(cudaGetLastError());
}

void cufft_scale_inplace_f64(double* p, int64_t n, double s, cudaStream_t stream) {
    if (n <= 0) return;
    k_cufft_scale_f64<<<cufft_grid(n), 256, 0, stream>>>(p, n, s);
    checkCudaErrors(cudaGetLastError());
}

void cufft_expand_hermitian_f32(
    const float* half, float* full,
    int64_t batch, int64_t keep, int64_t L, float scale, int conj_all, cudaStream_t stream
) {
    int64_t total = batch * L;
    if (total <= 0) return;
    k_cufft_expand_hermitian<float><<<cufft_grid(total), 256, 0, stream>>>(
        half, full, batch, keep, L, scale, conj_all);
    checkCudaErrors(cudaGetLastError());
}

void cufft_expand_hermitian_f64(
    const double* half, double* full,
    int64_t batch, int64_t keep, int64_t L, double scale, int conj_all, cudaStream_t stream
) {
    int64_t total = batch * L;
    if (total <= 0) return;
    k_cufft_expand_hermitian<double><<<cufft_grid(total), 256, 0, stream>>>(
        half, full, batch, keep, L, scale, conj_all);
    checkCudaErrors(cudaGetLastError());
}

} // namespace jittor
