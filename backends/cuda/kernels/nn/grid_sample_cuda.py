"""Fused CUDA kernels for 2-D grid_sample (bilinear and nearest)."""

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import register_kernel


_GRID_SAMPLE_CUDA_HEADER = r"""
#include <cmath>
#include <type/fp16_compute.h>

__device__ __forceinline__ float _jt_gs_floor(float x) { return ::floorf(x); }
__device__ __forceinline__ double _jt_gs_floor(double x) { return ::floor(x); }
__device__ __forceinline__ float _jt_gs_rint(float x) { return ::rintf(x); }
__device__ __forceinline__ double _jt_gs_rint(double x) { return ::rint(x); }
__device__ __forceinline__ float _jt_gs_fabs(float x) { return ::fabsf(x); }
__device__ __forceinline__ double _jt_gs_fabs(double x) { return ::fabs(x); }
__device__ __forceinline__ float _jt_gs_fmod(float a, float b) { return ::fmodf(a, b); }
__device__ __forceinline__ double _jt_gs_fmod(double a, double b) { return ::fmod(a, b); }

template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_unnormalize(AccT coord, int64_t size, int align_corners) {
    if (align_corners) {
        return ((coord + (AccT)1) * (AccT)0.5) * (AccT)(size - 1);
    }
    return ((coord + (AccT)1) * (AccT)size - (AccT)1) * (AccT)0.5;
}

template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_unnormalize_with_grad(
    AccT coord, int64_t size, int align_corners, AccT& grad_mult
) {
    if (align_corners) {
        grad_mult = (AccT)0.5 * (AccT)(size - 1);
        return ((coord + (AccT)1) * (AccT)0.5) * (AccT)(size - 1);
    }
    grad_mult = (AccT)0.5 * (AccT)size;
    return ((coord + (AccT)1) * (AccT)size - (AccT)1) * (AccT)0.5;
}

template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_clip(AccT x, int64_t size, AccT& grad_mult) {
    AccT hi = (AccT)(size - 1);
    if (x < (AccT)0) { grad_mult = (AccT)0; return (AccT)0; }
    if (x > hi) { grad_mult = (AccT)0; return hi; }
    grad_mult = (AccT)1;
    return x;
}

template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_reflect(
    AccT x, AccT twice_low, AccT twice_high, AccT& grad_mult
) {
    if (twice_low == twice_high) {
        grad_mult = (AccT)0;
        return (AccT)0;
    }
    AccT m = twice_low * (AccT)0.5;
    AccT span = (twice_high - twice_low) * (AccT)0.5;
    AccT diff = x - m;
    AccT sign = diff < (AccT)0 ? (AccT)(-1) : (AccT)1;
    AccT a = _jt_gs_fabs(diff);
    AccT extra = _jt_gs_fmod(a, span);
    int64_t flips = (int64_t)_jt_gs_floor(a / span);
    if ((flips & 1LL) == 0) {
        grad_mult = sign;
        return extra + m;
    }
    grad_mult = -sign;
    return span - extra + m;
}

// padding_mode: 0 = zeros, 1 = border, 2 = reflection
template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_compute_coord(
    AccT coord, int64_t size, int padding_mode, int align_corners
) {
    AccT c = _jt_gs_unnormalize(coord, size, align_corners);
    AccT dummy;
    if (padding_mode == 1) {
        c = _jt_gs_clip(c, size, dummy);
    } else if (padding_mode == 2) {
        if (align_corners) {
            c = _jt_gs_reflect(c, (AccT)0, (AccT)(2 * (size - 1)), dummy);
        } else {
            c = _jt_gs_reflect(c, (AccT)(-1), (AccT)(2 * size - 1), dummy);
        }
        c = _jt_gs_clip(c, size, dummy);
    }
    return c;
}

template <typename AccT>
__device__ __forceinline__ AccT _jt_gs_compute_coord_with_grad(
    AccT coord, int64_t size, int padding_mode, int align_corners, AccT& grad_mult
) {
    AccT gm_un = (AccT)1;
    AccT c = _jt_gs_unnormalize_with_grad(coord, size, align_corners, gm_un);
    AccT gm_pad = (AccT)1;
    if (padding_mode == 1) {
        c = _jt_gs_clip(c, size, gm_pad);
    } else if (padding_mode == 2) {
        AccT gm_ref = (AccT)1;
        AccT gm_clip = (AccT)1;
        if (align_corners) {
            c = _jt_gs_reflect(c, (AccT)0, (AccT)(2 * (size - 1)), gm_ref);
        } else {
            c = _jt_gs_reflect(c, (AccT)(-1), (AccT)(2 * size - 1), gm_ref);
        }
        c = _jt_gs_clip(c, size, gm_clip);
        gm_pad = gm_ref * gm_clip;
    }
    grad_mult = gm_un * gm_pad;
    return c;
}

// mode: 0 = bilinear, 1 = nearest
template <typename T, typename AccT>
__global__ void _jt_grid_sample_2d_fwd_k(
    const T* __restrict__ xp,
    const T* __restrict__ gp,
    T* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int mode, int padding_mode, int align_corners
) {
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total = N * C * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oHW;
        const int64_t hw = idx - nc * oHW;
        const int64_t n = nc / C;

        const int64_t g_off = (n * oHW + hw) * 2;
        AccT ix = _jt_gs_compute_coord((AccT)gp[g_off], W, padding_mode, align_corners);
        AccT iy = _jt_gs_compute_coord((AccT)gp[g_off + 1], H, padding_mode, align_corners);

        const T* plane = xp + nc * HW;
        if (mode == 1) {
            int64_t ix_n = (int64_t)_jt_gs_rint(ix);
            int64_t iy_n = (int64_t)_jt_gs_rint(iy);
            if (ix_n >= 0 && ix_n < W && iy_n >= 0 && iy_n < H) {
                yp[idx] = plane[iy_n * W + ix_n];
            } else {
                yp[idx] = (T)0;
            }
        } else {
            int64_t ix_nw = (int64_t)_jt_gs_floor(ix);
            int64_t iy_nw = (int64_t)_jt_gs_floor(iy);
            int64_t ix_ne = ix_nw + 1;
            int64_t iy_sw = iy_nw + 1;

            AccT dx = ix - (AccT)ix_nw;
            AccT dy = iy - (AccT)iy_nw;
            AccT dnx = (AccT)ix_ne - ix;
            AccT dny = (AccT)iy_sw - iy;

            AccT val = (AccT)0;
            if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
                val += (AccT)plane[iy_nw * W + ix_nw] * dnx * dny;
            if (iy_sw >= 0 && iy_sw < H && ix_nw >= 0 && ix_nw < W)
                val += (AccT)plane[iy_sw * W + ix_nw] * dnx * dy;
            if (iy_nw >= 0 && iy_nw < H && ix_ne >= 0 && ix_ne < W)
                val += (AccT)plane[iy_nw * W + ix_ne] * dx * dny;
            if (iy_sw >= 0 && iy_sw < H && ix_ne >= 0 && ix_ne < W)
                val += (AccT)plane[iy_sw * W + ix_ne] * dx * dy;
            yp[idx] = (T)val;
        }
    }
}

template <typename T, typename AccT>
__global__ void _jt_grid_sample_2d_bwd_k(
    const T* __restrict__ xp,
    const T* __restrict__ gp,
    const T* __restrict__ dop,
    AccT* __restrict__ dxp,
    T* __restrict__ dgp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int mode, int padding_mode, int align_corners
) {
    // 1 thread per (n, oh, ow), loops across channels C so dgrid needs zero atomics
    // and dX accumulates cleanly via atomicAdd on float/double buffer.
    const int64_t oHW = oH * oW;
    const int64_t HW = H * W;
    const int64_t total_pts = N * oHW;
    for (int64_t pt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pt < total_pts;
         pt += (int64_t)gridDim.x * blockDim.x) {
        const int64_t n = pt / oHW;
        const int64_t hw = pt - n * oHW;
        const int64_t g_off = pt * 2;

        AccT gix_mult = (AccT)1, giy_mult = (AccT)1;
        AccT ix = _jt_gs_compute_coord_with_grad((AccT)gp[g_off], W, padding_mode, align_corners, gix_mult);
        AccT iy = _jt_gs_compute_coord_with_grad((AccT)gp[g_off + 1], H, padding_mode, align_corners, giy_mult);

        if (mode == 1) {
            int64_t ix_n = (int64_t)_jt_gs_rint(ix);
            int64_t iy_n = (int64_t)_jt_gs_rint(iy);
            if (ix_n >= 0 && ix_n < W && iy_n >= 0 && iy_n < H) {
                for (int64_t c = 0; c < C; ++c) {
                    const int64_t nc = n * C + c;
                    AccT g_out = (AccT)dop[nc * oHW + hw];
                    atomicAdd(&dxp[nc * HW + iy_n * W + ix_n], g_out);
                }
            }
            dgp[g_off] = (T)0;
            dgp[g_off + 1] = (T)0;
        } else {
            int64_t ix_nw = (int64_t)_jt_gs_floor(ix);
            int64_t iy_nw = (int64_t)_jt_gs_floor(iy);
            int64_t ix_ne = ix_nw + 1;
            int64_t iy_sw = iy_nw + 1;

            AccT dx = ix - (AccT)ix_nw;
            AccT dy = iy - (AccT)iy_nw;
            AccT dnx = (AccT)ix_ne - ix;
            AccT dny = (AccT)iy_sw - iy;

            const bool nw_ok = (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W);
            const bool sw_ok = (iy_sw >= 0 && iy_sw < H && ix_nw >= 0 && ix_nw < W);
            const bool ne_ok = (iy_nw >= 0 && iy_nw < H && ix_ne >= 0 && ix_ne < W);
            const bool se_ok = (iy_sw >= 0 && iy_sw < H && ix_ne >= 0 && ix_ne < W);

            AccT gix = (AccT)0;
            AccT giy = (AccT)0;
            for (int64_t c = 0; c < C; ++c) {
                const int64_t nc = n * C + c;
                const T* xplane = xp + nc * HW;
                AccT* dxplane = dxp + nc * HW;
                AccT g_out = (AccT)dop[nc * oHW + hw];

                if (nw_ok) {
                    AccT v = (AccT)xplane[iy_nw * W + ix_nw];
                    atomicAdd(&dxplane[iy_nw * W + ix_nw], g_out * dnx * dny);
                    gix -= v * dny * g_out;
                    giy -= v * dnx * g_out;
                }
                if (sw_ok) {
                    AccT v = (AccT)xplane[iy_sw * W + ix_nw];
                    atomicAdd(&dxplane[iy_sw * W + ix_nw], g_out * dnx * dy);
                    gix -= v * dy * g_out;
                    giy += v * dnx * g_out;
                }
                if (ne_ok) {
                    AccT v = (AccT)xplane[iy_nw * W + ix_ne];
                    atomicAdd(&dxplane[iy_nw * W + ix_ne], g_out * dx * dny);
                    gix += v * dny * g_out;
                    giy -= v * dx * g_out;
                }
                if (se_ok) {
                    AccT v = (AccT)xplane[iy_sw * W + ix_ne];
                    atomicAdd(&dxplane[iy_sw * W + ix_ne], g_out * dx * dy);
                    gix += v * dy * g_out;
                    giy += v * dx * g_out;
                }
            }
            dgp[g_off] = (T)(gix * gix_mult);
            dgp[g_off + 1] = (T)(giy * giy_mult);
        }
    }
}
template <typename T, typename AccT>
__global__ void _jt_affine_grid_4d_fwd_k(
    const T* __restrict__ theta,
    T* __restrict__ out,
    int64_t N, int64_t H, int64_t W,
    int align_corners
) {
    const int64_t HW = H * W;
    const int64_t total = N * HW;
    const AccT sx = (W > 1) ? ((AccT)2 / (AccT)(W - 1)) : (AccT)0;
    const AccT sy = (H > 1) ? ((AccT)2 / (AccT)(H - 1)) : (AccT)0;
    const AccT fx = align_corners ? (AccT)1 : ((AccT)(W - 1) / (AccT)W);
    const AccT fy = align_corners ? (AccT)1 : ((AccT)(H - 1) / (AccT)H);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t n = idx / HW;
        const int64_t rem = idx - n * HW;
        const int64_t h = rem / W;
        const int64_t w = rem - h * W;

        const AccT x = (W > 1) ? (((AccT)w * sx - (AccT)1) * fx) : (AccT)0;
        const AccT y = (H > 1) ? (((AccT)h * sy - (AccT)1) * fy) : (AccT)0;

        const T* th = theta + n * 6;
        out[idx * 2]     = (T)((AccT)th[0] * x + (AccT)th[1] * y + (AccT)th[2]);
        out[idx * 2 + 1] = (T)((AccT)th[3] * x + (AccT)th[4] * y + (AccT)th[5]);
    }
}

template <typename T, typename AccT>
__global__ void _jt_affine_grid_4d_bwd_k(
    const T* __restrict__ dgrid,
    T* __restrict__ dtheta,
    int64_t N, int64_t H, int64_t W,
    int align_corners
) {
    // 1 block of 256 threads per batch element n: reduces 6 scalars per sample
    const int64_t HW = H * W;
    const AccT sx = (W > 1) ? ((AccT)2 / (AccT)(W - 1)) : (AccT)0;
    const AccT sy = (H > 1) ? ((AccT)2 / (AccT)(H - 1)) : (AccT)0;
    const AccT fx = align_corners ? (AccT)1 : ((AccT)(W - 1) / (AccT)W);
    const AccT fy = align_corners ? (AccT)1 : ((AccT)(H - 1) / (AccT)H);
    __shared__ AccT s[6][8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    for (int64_t n = blockIdx.x; n < N; n += gridDim.x) {
        AccT a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
        const T* dg = dgrid + n * HW * 2;
        for (int64_t rem = tid; rem < HW; rem += 256) {
            const int64_t h = rem / W;
            const int64_t w = rem - h * W;
            const AccT x = (W > 1) ? (((AccT)w * sx - (AccT)1) * fx) : (AccT)0;
            const AccT y = (H > 1) ? (((AccT)h * sy - (AccT)1) * fy) : (AccT)0;
            const AccT gx = (AccT)dg[rem * 2];
            const AccT gy = (AccT)dg[rem * 2 + 1];
            a0 += gx * x; a1 += gx * y; a2 += gx;
            a3 += gy * x; a4 += gy * y; a5 += gy;
        }
        AccT vals[6] = {a0, a1, a2, a3, a4, a5};
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1)
                vals[k] += __shfl_down_sync(0xffffffffu, vals[k], d);
            if (lane == 0) s[k][wid] = vals[k];
        }
        __syncthreads();
        if (wid == 0) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                AccT v = (lane < 8) ? s[k][lane] : (AccT)0;
                #pragma unroll
                for (int d = 4; d > 0; d >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, d);
                if (lane == 0) dtheta[n * 6 + k] = (T)v;
            }
        }
        __syncthreads();
    }
}
"""


_MODE_MAP = {"bilinear": 0, "nearest": 1}
_PAD_MAP = {"zeros": 0, "border": 1, "reflection": 2}


def _grid_sample_2d_fwd_code(x, grid, mode_id, pad_id, align_corners):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n, c, _, _ = x.shape
    oh, ow = grid.shape[1], grid.shape[2]
    return jt.code(
        [n, c, oh, ow],
        x.dtype,
        [x, grid],
        cuda_header=_GRID_SAMPLE_CUDA_HEADER,
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
            _jt_grid_sample_2d_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, in1_p, out0_p, N, C, H, W, oH, oW,
                (int)data["mode"], (int)data["pad"], (int)data["align"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "mode": float(mode_id),
            "pad": float(pad_id),
            "align": float(1 if align_corners else 0),
        },
    )


def _grid_sample_2d_bwd_code(x, grid, dout, mode_id, pad_id, align_corners):
    dname = _dtype_name(x.dtype)
    acc_t = "double" if dname == "float64" else "float"
    acc_dtype = "float64" if dname == "float64" else "float32"
    dx_acc, dgrid = jt.code(
        [list(x.shape), list(grid.shape)],
        [acc_dtype, grid.dtype],
        [x, grid, dout],
        cuda_header=_GRID_SAMPLE_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)in0->shape[0];
        const int64_t C = (int64_t)in0->shape[1];
        const int64_t H = (int64_t)in0->shape[2];
        const int64_t W = (int64_t)in0->shape[3];
        const int64_t oH = (int64_t)in1->shape[1];
        const int64_t oW = (int64_t)in1->shape[2];
        cudaMemsetAsync(out0_p, 0, out0->size);
        const int64_t total_pts = N * oH * oW;
        if (total_pts > 0 && C > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total_pts + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_grid_sample_2d_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, in1_p, in2_p, out0_p, out1_p, N, C, H, W, oH, oW,
                (int)data["mode"], (int)data["pad"], (int)data["align"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "mode": float(mode_id),
            "pad": float(pad_id),
            "align": float(1 if align_corners else 0),
        },
    )
    dx = dx_acc.cast(x.dtype) if acc_dtype != dname else dx_acc
    return dx, dgrid


class _GridSample2dCuda(jt.Function):
    def __init__(self, mode_id, pad_id, align_corners):
        self.mode_id = mode_id
        self.pad_id = pad_id
        self.align_corners = align_corners

    def execute(self, x, grid):
        xc = x.contiguous()
        gc = grid.contiguous()
        self.saved = (xc, gc)
        return _grid_sample_2d_fwd_code(xc, gc, self.mode_id, self.pad_id, self.align_corners)

    def grad(self, dout):
        if dout is None:
            return None, None
        xc, gc = self.saved
        with device_scope_like(dout):
            return _grid_sample_2d_bwd_code(
                xc, gc, dout.contiguous(), self.mode_id, self.pad_id, self.align_corners
            )


def _supports_grid_sample_2d(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    if input.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
        return False
    if input.dtype != grid.dtype or input.shape[0] != grid.shape[0]:
        return False
    if input.numel() == 0 or grid.numel() == 0:
        return False
    dname = _dtype_name(input.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return False
    return mode in _MODE_MAP and padding_mode in _PAD_MAP


def _grid_sample_cuda(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    mode_id = _MODE_MAP[mode]
    pad_id = _PAD_MAP[padding_mode]
    with device_scope_like(input):
        if jt.flags.no_grad:
            return _grid_sample_2d_fwd_code(
                input.contiguous(), grid.contiguous(), mode_id, pad_id, bool(align_corners)
            )
        return _GridSample2dCuda(mode_id, pad_id, bool(align_corners))(input, grid)


def _affine_grid_4d_fwd_code(theta, n, h, w, align_corners):
    acc_t = "double" if _dtype_name(theta.dtype) == "float64" else "float"
    return jt.code(
        [n, h, w, 2],
        theta.dtype,
        [theta],
        cuda_header=_GRID_SAMPLE_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)out0->shape[0];
        const int64_t H = (int64_t)out0->shape[1];
        const int64_t W = (int64_t)out0->shape[2];
        const int64_t total = N * H * W;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_affine_grid_4d_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, H, W, (int)data["align"]);
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={"align": float(1 if align_corners else 0)},
    )


def _affine_grid_4d_bwd_code(dgrid, align_corners):
    acc_t = "double" if _dtype_name(dgrid.dtype) == "float64" else "float"
    n = int(dgrid.shape[0])
    return jt.code(
        [n, 2, 3],
        dgrid.dtype,
        [dgrid],
        cuda_header=_GRID_SAMPLE_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)in0->shape[0];
        const int64_t H = (int64_t)in0->shape[1];
        const int64_t W = (int64_t)in0->shape[2];
        if (N > 0 && H > 0 && W > 0) {{
            const int blocks = (int)(N > 65535 ? 65535 : N);
            _jt_affine_grid_4d_bwd_k<in0_type, {acc_t}><<<blocks, 256>>>(
                in0_p, out0_p, N, H, W, (int)data["align"]);
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={"align": float(1 if align_corners else 0)},
    )


class _AffineGrid4dCuda(jt.Function):
    def __init__(self, n, h, w, align_corners):
        self.n = n
        self.h = h
        self.w = w
        self.align_corners = align_corners

    def execute(self, theta):
        return _affine_grid_4d_fwd_code(theta.contiguous(), self.n, self.h, self.w, self.align_corners)

    def grad(self, dgrid):
        if dgrid is None:
            return None
        with device_scope_like(dgrid):
            return _affine_grid_4d_bwd_code(dgrid.contiguous(), self.align_corners)


def _affine_grid_4d_cuda(theta, N, C, H, W, align_corners):
    dname = _dtype_name(theta.dtype)
    if dname not in ("float32", "float64"):
        return None
    with device_scope_like(theta):
        if jt.flags.no_grad:
            return _affine_grid_4d_fwd_code(theta.contiguous(), int(N), int(H), int(W), bool(align_corners))
        return _AffineGrid4dCuda(int(N), int(H), int(W), bool(align_corners))(theta)


register_kernel("nn.grid_sample", "cuda", _grid_sample_cuda, supports=_supports_grid_sample_2d)
register_kernel("nn.affine_grid_4d", "cuda", _affine_grid_4d_cuda)
