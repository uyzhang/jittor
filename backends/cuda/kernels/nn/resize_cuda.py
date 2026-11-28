"""Fused CUDA kernel for 2-D bilinear resize / interpolate."""

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import register_kernel


_RESIZE_CUDA_HEADER = r"""
#include <cmath>
#include <type/fp16_compute.h>

__device__ __forceinline__ float _jt_rs_floor(float x) { return ::floorf(x); }
__device__ __forceinline__ double _jt_rs_floor(double x) { return ::floor(x); }

template <typename AccT>
__device__ __forceinline__ AccT _jt_rs_coord(
    int64_t idx, int64_t in_size, int64_t out_size, int align_corners, int tf_mode
) {
    AccT c;
    if (align_corners) {
        int64_t denom = out_size > 1 ? (out_size - 1) : 1;
        c = ((AccT)idx * (AccT)(in_size - 1)) / (AccT)denom;
    } else if (tf_mode) {
        c = ((AccT)idx * (AccT)in_size) / (AccT)out_size;
        if (out_size > in_size) {
            if (c < (AccT)0) c = (AccT)0;
            if (c > (AccT)(in_size - 1)) c = (AccT)(in_size - 1);
        }
    } else {
        c = (((AccT)idx + (AccT)0.5) * (AccT)in_size) / (AccT)out_size - (AccT)0.5;
        if (out_size > in_size) {
            if (c < (AccT)0) c = (AccT)0;
            if (c > (AccT)(in_size - 1)) c = (AccT)(in_size - 1);
        }
    }
    return c;
}

template <typename T, typename AccT>
__global__ void _jt_resize_bilinear_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int align_corners, int tf_mode
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oHW;
        const int64_t rem = idx - nc * oHW;
        const int64_t oh = rem / oW;
        const int64_t ow = rem - oh * oW;

        AccT x = _jt_rs_coord<AccT>(oh, H, oH, align_corners, tf_mode);
        AccT y = _jt_rs_coord<AccT>(ow, W, oW, align_corners, tf_mode);

        int64_t fx = (int64_t)_jt_rs_floor(x);
        int64_t fy = (int64_t)_jt_rs_floor(y);
        int64_t cx = fx + 1;
        int64_t cy = fy + 1;
        AccT dx = x - (AccT)fx;
        AccT dy = y - (AccT)fy;
        AccT dnx = (AccT)1 - dx;
        AccT dny = (AccT)1 - dy;

        if (fx < 0) fx = 0; else if (fx >= H) fx = H - 1;
        if (cx < 0) cx = 0; else if (cx >= H) cx = H - 1;
        if (fy < 0) fy = 0; else if (fy >= W) fy = W - 1;
        if (cy < 0) cy = 0; else if (cy >= W) cy = W - 1;

        const T* plane = xp + nc * HW;
        AccT a = (AccT)plane[fx * W + fy];
        AccT b = (AccT)plane[cx * W + fy];
        AccT c = (AccT)plane[fx * W + cy];
        AccT d = (AccT)plane[cx * W + cy];
        AccT ab = dx * b + dnx * a;
        AccT cd = dx * d + dnx * c;
        yp[idx] = (T)(ab * dny + cd * dy);
    }
}

template <typename T, typename AccT>
__global__ void _jt_resize_bilinear_bwd_k(
    const T* __restrict__ dop,
    AccT* __restrict__ dxp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int align_corners, int tf_mode
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oHW;
        const int64_t rem = idx - nc * oHW;
        const int64_t oh = rem / oW;
        const int64_t ow = rem - oh * oW;

        AccT x = _jt_rs_coord<AccT>(oh, H, oH, align_corners, tf_mode);
        AccT y = _jt_rs_coord<AccT>(ow, W, oW, align_corners, tf_mode);

        int64_t fx = (int64_t)_jt_rs_floor(x);
        int64_t fy = (int64_t)_jt_rs_floor(y);
        int64_t cx = fx + 1;
        int64_t cy = fy + 1;
        AccT dx = x - (AccT)fx;
        AccT dy = y - (AccT)fy;
        AccT dnx = (AccT)1 - dx;
        AccT dny = (AccT)1 - dy;

        if (fx < 0) fx = 0; else if (fx >= H) fx = H - 1;
        if (cx < 0) cx = 0; else if (cx >= H) cx = H - 1;
        if (fy < 0) fy = 0; else if (fy >= W) fy = W - 1;
        if (cy < 0) cy = 0; else if (cy >= W) cy = W - 1;

        AccT g = (AccT)dop[idx];
        AccT* dxplane = dxp + nc * HW;
        atomicAdd(&dxplane[fx * W + fy], g * dnx * dny);
        atomicAdd(&dxplane[cx * W + fy], g * dx * dny);
        atomicAdd(&dxplane[fx * W + cy], g * dnx * dy);
        atomicAdd(&dxplane[cx * W + cy], g * dx * dy);
    }
}
"""


def _resize_bilinear_fwd_code(x, out_h, out_w, align_corners, tf_mode):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n, c, _, _ = x.shape
    return jt.code(
        [n, c, out_h, out_w],
        x.dtype,
        [x],
        cuda_header=_RESIZE_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)in0->shape[0];
        const int64_t C = (int64_t)in0->shape[1];
        const int64_t H = (int64_t)in0->shape[2];
        const int64_t W = (int64_t)in0->shape[3];
        const int64_t oH = (int64_t)out0->shape[2];
        const int64_t oW = (int64_t)out0->shape[3];
        const int64_t total = N * C * oH * oW;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_resize_bilinear_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, H, W, oH, oW,
                (int)data["align"], (int)data["tf_mode"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "align": float(1 if align_corners else 0),
            "tf_mode": float(1 if tf_mode else 0),
        },
    )


def _resize_bilinear_bwd_code(dout, in_shape, align_corners, tf_mode):
    dname = _dtype_name(dout.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    dx_acc = jt.code(
        list(in_shape),
        acc_dtype,
        [dout],
        cuda_header=_RESIZE_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)out0->shape[0];
        const int64_t C = (int64_t)out0->shape[1];
        const int64_t H = (int64_t)out0->shape[2];
        const int64_t W = (int64_t)out0->shape[3];
        const int64_t oH = (int64_t)in0->shape[2];
        const int64_t oW = (int64_t)in0->shape[3];
        cudaMemsetAsync(out0_p, 0, out0->size);
        const int64_t total = N * C * oH * oW;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_resize_bilinear_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, H, W, oH, oW,
                (int)data["align"], (int)data["tf_mode"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "align": float(1 if align_corners else 0),
            "tf_mode": float(1 if tf_mode else 0),
        },
    )
    return dx_acc.cast(dout.dtype) if acc_dtype != dname else dx_acc


class _ResizeBilinear2dCuda(jt.Function):
    def __init__(self, out_h, out_w, align_corners, tf_mode):
        self.out_h = out_h
        self.out_w = out_w
        self.align_corners = align_corners
        self.tf_mode = tf_mode

    def execute(self, x):
        self.in_shape = tuple(x.shape)
        return _resize_bilinear_fwd_code(
            x.contiguous(), self.out_h, self.out_w, self.align_corners, self.tf_mode
        )

    def grad(self, dout):
        if dout is None:
            return None
        with device_scope_like(dout):
            return _resize_bilinear_bwd_code(
                dout.contiguous(), self.in_shape, self.align_corners, self.tf_mode
            )


def _resize_bilinear_cuda(img, size, mode="nearest", align_corners=False, tf_mode=False):
    if mode != "bilinear" or img.ndim != 4:
        return None
    dname = _dtype_name(img.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    # Preserve existing channels-last optimization path when active
    from jittor.nn.functional._layout import channels_last_source, records_no_grad
    if records_no_grad(img) and channels_last_source(img) is not None:
        return None
    out_h, out_w = int(size[0]), int(size[1])
    with device_scope_like(img):
        if jt.flags.no_grad:
            return _resize_bilinear_fwd_code(
                img.contiguous(), out_h, out_w, bool(align_corners), bool(tf_mode)
            )
        return _ResizeBilinear2dCuda(out_h, out_w, bool(align_corners), bool(tf_mode))(img)


register_kernel("nn.resize", "cuda", _resize_bilinear_cuda)
