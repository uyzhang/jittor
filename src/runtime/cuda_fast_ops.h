// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <cstdint>
#include "core/common.h"

namespace jittor {

struct IndexFault;

enum class FastUnaryOp : int {
    Exp = 0,
    Log = 1,
    Log1p = 2,
    Log2 = 3,
    Exp2 = 4,
    Expm1 = 5,
    Sqrt = 6,
    Rsqrt = 7,
    Reciprocal = 8,
    Abs = 9,
    Neg = 10,
    Sin = 11,
    Cos = 12,
    Tan = 13,
    Tanh = 14,
    Sigmoid = 15,
    Relu = 16,
    Erf = 17,
    Floor = 18,
    Ceil = 19,
    Round = 20,
    Trunc = 21,
    Sign = 22,
    Deg2Rad = 23,
    Rad2Deg = 24,
    Lgamma = 25,
    Digamma = 26,
    Clamp = 27,        // p0=min (or -inf if has_min==0), p1=max (or +inf)
    Hardswish = 28,
    Hardsigmoid = 29,
    Hardtanh = 30,     // p0=min_val, p1=max_val
    Elu = 31,          // p0=alpha
    Celu = 32,         // p0=alpha
    Selu = 33,
    Softplus = 34,     // p0=beta, p1=threshold
    Mish = 35,
    Logsigmoid = 36,
    Gelu = 37,
    Silu = 38,
    LeakyRelu = 39,    // p0=negative_slope
};

enum class FastBinaryOp : int {
    Pow = 0,
    Atan2 = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    Maximum = 6,
    Minimum = 7,
    Hypot = 8,
    LogAddExp = 9,
};

#if defined(IS_CUDA) && !defined(IS_COREX)

EXTERN_LIB void cuda_fast_unary_f32(
    FastUnaryOp op, const float* x, float* y, int64_t n,
    float p0, float p1, int flags);

EXTERN_LIB void cuda_fast_binary_f32(
    FastBinaryOp op, const float* x, const float* y, float* z, int64_t n);

EXTERN_LIB void cuda_fast_var_std_lastdim_f32(
    const float* x, float* out, int64_t rows, int64_t width,
    float correction, int take_sqrt);

EXTERN_LIB void cuda_fast_prod_lastdim_f32(
    const float* x, float* out, int64_t rows, int64_t width);

EXTERN_LIB void cuda_fast_arg_reduce_lastdim_f32(
    const float* x, float* val_out, int64_t* idx_out,
    int64_t rows, int64_t width, int is_min);

EXTERN_LIB void cuda_fast_scan_f32(
    const float* x, float* y, int64_t outer, int64_t dim_size,
    int64_t inner, int is_prod);

EXTERN_LIB void cuda_fast_cross_entropy_f32(
    const float* x, const void* target, int target_is_i64,
    const float* weight, float* out, float* temp_losses, float* temp_weights,
    int64_t N, int64_t C, int has_ignore, int64_t ignore_idx,
    int red_mode, IndexFault* fault);

EXTERN_LIB void cuda_fast_nll_loss_f32(
    const float* x, const void* target, int target_is_i64,
    const float* weight, float* out, float* temp_losses, float* temp_weights,
    int64_t N, int64_t C, int64_t ignore_idx,
    int red_mode, IndexFault* fault);

EXTERN_LIB void cuda_fast_bce_with_logits_f32(
    const float* x, const float* target, float* out, float* part_buf,
    int64_t n, int n_blocks, int red_mode);

EXTERN_LIB void cuda_fast_adaptive_avg_pool2d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW);

EXTERN_LIB void cuda_fast_avg_pool2d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int count_pad);

EXTERN_LIB void cuda_fast_avg_pool3d_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int count_pad);

EXTERN_LIB void cuda_fast_grid_sample_2d_f32(
    const float* x, const float* grid, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int mode, int padding_mode, int align_corners);

EXTERN_LIB void cuda_fast_resize_bilinear_f32(
    const float* x, float* y,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW,
    int align_corners, int tf_mode);

EXTERN_LIB void cuda_fast_affine_grid_4d_f32(
    const float* theta, float* out,
    int64_t N, int64_t H, int64_t W, int align_corners);

EXTERN_LIB void cuda_fast_triu_tril_f32(
    const float* x, float* y,
    int64_t batch, int64_t rows, int64_t cols, int64_t diagonal, int is_lower);

EXTERN_LIB void cuda_fast_trace_f32(
    const float* x, float* out, int64_t size, int64_t stride);

#endif

} // namespace jittor
