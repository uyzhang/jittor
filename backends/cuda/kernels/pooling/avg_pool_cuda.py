"""Fused CUDA kernels for adaptive_avg_pool2d, avg_pool2d, and avg_pool3d."""

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import register_kernel


_POOL_CUDA_HEADER = r"""
#include <type/fp16_compute.h>

template <typename T, typename AccT>
__global__ void _jt_aap2d_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW
) {
    const int64_t oHW = oH * oW;
    const int64_t total = N * C * oHW;
    const int64_t HW = H * W;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oHW;
        const int64_t rem = idx - nc * oHW;
        const int64_t oh = rem / oW;
        const int64_t ow = rem - oh * oW;

        const int64_t h0 = (oh * H) / oH;
        const int64_t h1 = ((oh + 1) * H + oH - 1) / oH;
        const int64_t w0 = (ow * W) / oW;
        const int64_t w1 = ((ow + 1) * W + oW - 1) / oW;

        const T* plane = xp + nc * HW;
        AccT sum = (AccT)0;
        for (int64_t ih = h0; ih < h1; ++ih) {
            const int64_t row_off = ih * W;
            for (int64_t iw = w0; iw < w1; ++iw) {
                sum += (AccT)plane[row_off + iw];
            }
        }
        const AccT cnt = (AccT)((h1 - h0) * (w1 - w0));
        yp[idx] = (T)(sum / cnt);
    }
}

template <typename T, typename AccT>
__global__ void _jt_aap2d_bwd_k(
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW
) {
    const int64_t HW = H * W;
    const int64_t total = N * C * HW;
    const int64_t oHW = oH * oW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / HW;
        const int64_t rem = idx - nc * HW;
        const int64_t ih = rem / W;
        const int64_t iw = rem - ih * W;

        const int64_t oh0 = (ih * oH) / H;
        const int64_t oh1 = ((ih + 1) * oH - 1) / H + 1;
        const int64_t ow0 = (iw * oW) / W;
        const int64_t ow1 = ((iw + 1) * oW - 1) / W + 1;

        const T* gplane = gp + nc * oHW;
        AccT acc = (AccT)0;
        for (int64_t oh = oh0; oh < oh1; ++oh) {
            const int64_t kh = ((oh + 1) * H + oH - 1) / oH - (oh * H) / oH;
            const int64_t grow = oh * oW;
            for (int64_t ow = ow0; ow < ow1; ++ow) {
                const int64_t kw = ((ow + 1) * W + oW - 1) / oW - (ow * W) / oW;
                acc += (AccT)gplane[grow + ow] / (AccT)(kh * kw);
            }
        }
        dxp[idx] = (T)acc;
    }
}

template <typename T, typename AccT>
__global__ void _jt_avgpool2d_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int count_pad
) {
    const int64_t oHW = oH * oW;
    const int64_t total = N * C * oHW;
    const int64_t HW = H * W;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oHW;
        const int64_t rem = idx - nc * oHW;
        const int64_t oh = rem / oW;
        const int64_t ow = rem - oh * oW;

        const int64_t hs = oh * sH - pH;
        const int64_t ws = ow * sW - pW;
        const int64_t he_pad = hs + kH < H + pH ? hs + kH : H + pH;
        const int64_t we_pad = ws + kW < W + pW ? ws + kW : W + pW;
        const int64_t h0 = hs > 0 ? hs : 0;
        const int64_t w0 = ws > 0 ? ws : 0;
        const int64_t h1 = hs + kH < H ? hs + kH : H;
        const int64_t w1 = ws + kW < W ? ws + kW : W;

        const T* plane = xp + nc * HW;
        AccT sum = (AccT)0;
        for (int64_t ih = h0; ih < h1; ++ih) {
            const int64_t row_off = ih * W;
            for (int64_t iw = w0; iw < w1; ++iw) {
                sum += (AccT)plane[row_off + iw];
            }
        }
        int64_t dh = count_pad ? (he_pad - hs) : (h1 - h0);
        int64_t dw = count_pad ? (we_pad - ws) : (w1 - w0);
        if (dh < 1) dh = 1;
        if (dw < 1) dw = 1;
        yp[idx] = (T)(sum / (AccT)(dh * dw));
    }
}

template <typename T, typename AccT>
__global__ void _jt_avgpool2d_bwd_k(
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t oH, int64_t oW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int count_pad
) {
    const int64_t HW = H * W;
    const int64_t total = N * C * HW;
    const int64_t oHW = oH * oW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / HW;
        const int64_t rem = idx - nc * HW;
        const int64_t ih = rem / W;
        const int64_t iw = rem - ih * W;

        const int64_t oh_start = (ih + pH < kH) ? 0 : ((ih + pH - kH) / sH + 1);
        int64_t oh_end = (ih + pH) / sH + 1;
        if (oh_end > oH) oh_end = oH;
        const int64_t ow_start = (iw + pW < kW) ? 0 : ((iw + pW - kW) / sW + 1);
        int64_t ow_end = (iw + pW) / sW + 1;
        if (ow_end > oW) ow_end = oW;

        const T* gplane = gp + nc * oHW;
        AccT acc = (AccT)0;
        for (int64_t oh = oh_start; oh < oh_end; ++oh) {
            const int64_t hs = oh * sH - pH;
            const int64_t he_pad = hs + kH < H + pH ? hs + kH : H + pH;
            const int64_t h0 = hs > 0 ? hs : 0;
            const int64_t h1 = hs + kH < H ? hs + kH : H;
            int64_t dh = count_pad ? (he_pad - hs) : (h1 - h0);
            if (dh < 1) dh = 1;
            const int64_t grow = oh * oW;
            for (int64_t ow = ow_start; ow < ow_end; ++ow) {
                const int64_t ws = ow * sW - pW;
                const int64_t we_pad = ws + kW < W + pW ? ws + kW : W + pW;
                const int64_t w0 = ws > 0 ? ws : 0;
                const int64_t w1 = ws + kW < W ? ws + kW : W;
                int64_t dw = count_pad ? (we_pad - ws) : (w1 - w0);
                if (dw < 1) dw = 1;
                acc += (AccT)gplane[grow + ow] / (AccT)(dh * dw);
            }
        }
        dxp[idx] = (T)acc;
    }
}

template <typename T, typename AccT>
__global__ void _jt_avgpool3d_fwd_k(
    const T* __restrict__ xp,
    T* __restrict__ yp,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int count_pad
) {
    const int64_t oHW = oH * oW;
    const int64_t oDHW = oD * oHW;
    const int64_t total = N * C * oDHW;
    const int64_t HW = H * W;
    const int64_t DHW = D * HW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / oDHW;
        int64_t rem = idx - nc * oDHW;
        const int64_t od = rem / oHW;
        rem -= od * oHW;
        const int64_t oh = rem / oW;
        const int64_t ow = rem - oh * oW;

        const int64_t ds = od * sD - pD;
        const int64_t hs = oh * sH - pH;
        const int64_t ws = ow * sW - pW;
        const int64_t de_pad = ds + kD < D + pD ? ds + kD : D + pD;
        const int64_t he_pad = hs + kH < H + pH ? hs + kH : H + pH;
        const int64_t we_pad = ws + kW < W + pW ? ws + kW : W + pW;
        const int64_t d0 = ds > 0 ? ds : 0;
        const int64_t h0 = hs > 0 ? hs : 0;
        const int64_t w0 = ws > 0 ? ws : 0;
        const int64_t d1 = ds + kD < D ? ds + kD : D;
        const int64_t h1 = hs + kH < H ? hs + kH : H;
        const int64_t w1 = ws + kW < W ? ws + kW : W;

        const T* vol = xp + nc * DHW;
        AccT sum = (AccT)0;
        for (int64_t id = d0; id < d1; ++id) {
            const int64_t d_off = id * HW;
            for (int64_t ih = h0; ih < h1; ++ih) {
                const int64_t row_off = d_off + ih * W;
                for (int64_t iw = w0; iw < w1; ++iw) {
                    sum += (AccT)vol[row_off + iw];
                }
            }
        }
        int64_t dd = count_pad ? (de_pad - ds) : (d1 - d0);
        int64_t dh = count_pad ? (he_pad - hs) : (h1 - h0);
        int64_t dw = count_pad ? (we_pad - ws) : (w1 - w0);
        if (dd < 1) dd = 1;
        if (dh < 1) dh = 1;
        if (dw < 1) dw = 1;
        yp[idx] = (T)(sum / (AccT)(dd * dh * dw));
    }
}

template <typename T, typename AccT>
__global__ void _jt_avgpool3d_bwd_k(
    const T* __restrict__ gp,
    T* __restrict__ dxp,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int count_pad
) {
    const int64_t HW = H * W;
    const int64_t DHW = D * HW;
    const int64_t total = N * C * DHW;
    const int64_t oHW = oH * oW;
    const int64_t oDHW = oD * oHW;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t nc = idx / DHW;
        int64_t rem = idx - nc * DHW;
        const int64_t id = rem / HW;
        rem -= id * HW;
        const int64_t ih = rem / W;
        const int64_t iw = rem - ih * W;

        const int64_t od0 = (id + pD < kD) ? 0 : ((id + pD - kD) / sD + 1);
        int64_t od1 = (id + pD) / sD + 1;
        if (od1 > oD) od1 = oD;
        const int64_t oh0 = (ih + pH < kH) ? 0 : ((ih + pH - kH) / sH + 1);
        int64_t oh1 = (ih + pH) / sH + 1;
        if (oh1 > oH) oh1 = oH;
        const int64_t ow0 = (iw + pW < kW) ? 0 : ((iw + pW - kW) / sW + 1);
        int64_t ow1 = (iw + pW) / sW + 1;
        if (ow1 > oW) ow1 = oW;

        const T* gvol = gp + nc * oDHW;
        AccT acc = (AccT)0;
        for (int64_t od = od0; od < od1; ++od) {
            const int64_t ds = od * sD - pD;
            const int64_t de_pad = ds + kD < D + pD ? ds + kD : D + pD;
            const int64_t d0 = ds > 0 ? ds : 0;
            const int64_t d1 = ds + kD < D ? ds + kD : D;
            int64_t dd = count_pad ? (de_pad - ds) : (d1 - d0);
            if (dd < 1) dd = 1;
            const int64_t gd_off = od * oHW;
            for (int64_t oh = oh0; oh < oh1; ++oh) {
                const int64_t hs = oh * sH - pH;
                const int64_t he_pad = hs + kH < H + pH ? hs + kH : H + pH;
                const int64_t h0 = hs > 0 ? hs : 0;
                const int64_t h1 = hs + kH < H ? hs + kH : H;
                int64_t dh = count_pad ? (he_pad - hs) : (h1 - h0);
                if (dh < 1) dh = 1;
                const int64_t gh_off = gd_off + oh * oW;
                for (int64_t ow = ow0; ow < ow1; ++ow) {
                    const int64_t ws = ow * sW - pW;
                    const int64_t we_pad = ws + kW < W + pW ? ws + kW : W + pW;
                    const int64_t w0 = ws > 0 ? ws : 0;
                    const int64_t w1 = ws + kW < W ? ws + kW : W;
                    int64_t dw = count_pad ? (we_pad - ws) : (w1 - w0);
                    if (dw < 1) dw = 1;
                    acc += (AccT)gvol[gh_off + ow] / (AccT)(dd * dh * dw);
                }
            }
        }
        dxp[idx] = (T)acc;
    }
}
"""


def _adaptive_avg_pool2d_fwd_code(x, out_h, out_w):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n, c, _, _ = x.shape
    return jt.code(
        [n, c, out_h, out_w],
        x.dtype,
        [x],
        cuda_header=_POOL_CUDA_HEADER,
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
            _jt_aap2d_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(in0_p, out0_p, N, C, H, W, oH, oW);
            CHECK(0 == cudaGetLastError());
        }}
        """,
    )


def _adaptive_avg_pool2d_bwd_code(dout, in_shape):
    acc_t = "double" if _dtype_name(dout.dtype) == "float64" else "float"
    return jt.code(
        list(in_shape),
        dout.dtype,
        [dout],
        cuda_header=_POOL_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)out0->shape[0];
        const int64_t C = (int64_t)out0->shape[1];
        const int64_t H = (int64_t)out0->shape[2];
        const int64_t W = (int64_t)out0->shape[3];
        const int64_t oH = (int64_t)in0->shape[2];
        const int64_t oW = (int64_t)in0->shape[3];
        const int64_t total = N * C * H * W;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_aap2d_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(in0_p, out0_p, N, C, H, W, oH, oW);
            CHECK(0 == cudaGetLastError());
        }}
        """,
    )


class _AdaptiveAvgPool2dCuda(jt.Function):
    def __init__(self, out_h, out_w):
        self.out_h = out_h
        self.out_w = out_w

    def execute(self, x):
        self.in_shape = tuple(x.shape)
        return _adaptive_avg_pool2d_fwd_code(x.contiguous(), self.out_h, self.out_w)

    def grad(self, dout):
        if dout is None:
            return None
        with device_scope_like(dout):
            return _adaptive_avg_pool2d_bwd_code(dout.contiguous(), self.in_shape)


def _adaptive_avg_pool2d_cuda(input, output_size):
    dname = _dtype_name(input.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    out_h, out_w = int(output_size[0]), int(output_size[1])
    if out_h == 1 and out_w == 1:
        return None
    with device_scope_like(input):
        if jt.flags.no_grad:
            return _adaptive_avg_pool2d_fwd_code(input.contiguous(), out_h, out_w)
        return _AdaptiveAvgPool2dCuda(out_h, out_w)(input)


def _avg_pool2d_fwd_code(x, out_h, out_w, kh, kw, sh, sw, ph, pw, count_include_pad):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n, c, _, _ = x.shape
    return jt.code(
        [n, c, out_h, out_w],
        x.dtype,
        [x],
        cuda_header=_POOL_CUDA_HEADER,
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
            _jt_avgpool2d_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, H, W, oH, oW,
                (int64_t)data["kH"], (int64_t)data["kW"],
                (int64_t)data["sH"], (int64_t)data["sW"],
                (int64_t)data["pH"], (int64_t)data["pW"],
                (int)data["count_pad"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "kH": float(kh), "kW": float(kw),
            "sH": float(sh), "sW": float(sw),
            "pH": float(ph), "pW": float(pw),
            "count_pad": float(1 if count_include_pad else 0),
        },
    )


def _avg_pool2d_bwd_code(dout, in_shape, kh, kw, sh, sw, ph, pw, count_include_pad):
    acc_t = "double" if _dtype_name(dout.dtype) == "float64" else "float"
    return jt.code(
        list(in_shape),
        dout.dtype,
        [dout],
        cuda_header=_POOL_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)out0->shape[0];
        const int64_t C = (int64_t)out0->shape[1];
        const int64_t H = (int64_t)out0->shape[2];
        const int64_t W = (int64_t)out0->shape[3];
        const int64_t oH = (int64_t)in0->shape[2];
        const int64_t oW = (int64_t)in0->shape[3];
        const int64_t total = N * C * H * W;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_avgpool2d_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, H, W, oH, oW,
                (int64_t)data["kH"], (int64_t)data["kW"],
                (int64_t)data["sH"], (int64_t)data["sW"],
                (int64_t)data["pH"], (int64_t)data["pW"],
                (int)data["count_pad"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "kH": float(kh), "kW": float(kw),
            "sH": float(sh), "sW": float(sw),
            "pH": float(ph), "pW": float(pw),
            "count_pad": float(1 if count_include_pad else 0),
        },
    )


class _AvgPool2dCuda(jt.Function):
    def __init__(self, out_h, out_w, kh, kw, sh, sw, ph, pw, count_include_pad):
        self.params = (out_h, out_w, kh, kw, sh, sw, ph, pw, count_include_pad)

    def execute(self, x):
        self.in_shape = tuple(x.shape)
        return _avg_pool2d_fwd_code(x.contiguous(), *self.params)

    def grad(self, dout):
        if dout is None:
            return None
        _, _, kh, kw, sh, sw, ph, pw, count_pad = self.params
        with device_scope_like(dout):
            return _avg_pool2d_bwd_code(dout.contiguous(), self.in_shape, kh, kw, sh, sw, ph, pw, count_pad)


def _avg_pool2d_cuda(x, kernel, strides, pads, dilation, return_indices, ceil_mode, count_include_pad, op):
    if op != "mean" or return_indices or dilation is not None:
        return None
    dname = _dtype_name(x.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    from jittor.nn.functional.pooling.average import _pool_output_size
    out_h = _pool_output_size(int(x.shape[2]), kernel[0], strides[0], pads[0], ceil_mode)
    out_w = _pool_output_size(int(x.shape[3]), kernel[1], strides[1], pads[1], ceil_mode)
    with device_scope_like(x):
        if jt.flags.no_grad:
            return _avg_pool2d_fwd_code(
                x.contiguous(), out_h, out_w,
                kernel[0], kernel[1], strides[0], strides[1], pads[0], pads[1], count_include_pad
            )
        return _AvgPool2dCuda(
            out_h, out_w, kernel[0], kernel[1], strides[0], strides[1], pads[0], pads[1], count_include_pad
        )(x)


def _avg_pool3d_fwd_code(x, out_d, out_h, out_w, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_include_pad):
    acc_t = "double" if _dtype_name(x.dtype) == "float64" else "float"
    n, c, _, _, _ = x.shape
    return jt.code(
        [n, c, out_d, out_h, out_w],
        x.dtype,
        [x],
        cuda_header=_POOL_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)in0->shape[0];
        const int64_t C = (int64_t)in0->shape[1];
        const int64_t D = (int64_t)in0->shape[2];
        const int64_t H = (int64_t)in0->shape[3];
        const int64_t W = (int64_t)in0->shape[4];
        const int64_t oD = (int64_t)out0->shape[2];
        const int64_t oH = (int64_t)out0->shape[3];
        const int64_t oW = (int64_t)out0->shape[4];
        const int64_t total = N * C * oD * oH * oW;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_avgpool3d_fwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, D, H, W, oD, oH, oW,
                (int64_t)data["kD"], (int64_t)data["kH"], (int64_t)data["kW"],
                (int64_t)data["sD"], (int64_t)data["sH"], (int64_t)data["sW"],
                (int64_t)data["pD"], (int64_t)data["pH"], (int64_t)data["pW"],
                (int)data["count_pad"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "kD": float(kd), "kH": float(kh), "kW": float(kw),
            "sD": float(sd), "sH": float(sh), "sW": float(sw),
            "pD": float(pd), "pH": float(ph), "pW": float(pw),
            "count_pad": float(1 if count_include_pad else 0),
        },
    )


def _avg_pool3d_bwd_code(dout, in_shape, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_include_pad):
    acc_t = "double" if _dtype_name(dout.dtype) == "float64" else "float"
    return jt.code(
        list(in_shape),
        dout.dtype,
        [dout],
        cuda_header=_POOL_CUDA_HEADER,
        cuda_src=f"""
        const int64_t N = (int64_t)out0->shape[0];
        const int64_t C = (int64_t)out0->shape[1];
        const int64_t D = (int64_t)out0->shape[2];
        const int64_t H = (int64_t)out0->shape[3];
        const int64_t W = (int64_t)out0->shape[4];
        const int64_t oD = (int64_t)in0->shape[2];
        const int64_t oH = (int64_t)in0->shape[3];
        const int64_t oW = (int64_t)in0->shape[4];
        const int64_t total = N * C * D * H * W;
        if (total > 0) {{
            constexpr int threads = 256;
            const int64_t raw_blocks = (total + threads - 1) / threads;
            const int blocks = (int)(raw_blocks > 65535 ? 65535 : raw_blocks);
            _jt_avgpool3d_bwd_k<in0_type, {acc_t}><<<blocks, threads>>>(
                in0_p, out0_p, N, C, D, H, W, oD, oH, oW,
                (int64_t)data["kD"], (int64_t)data["kH"], (int64_t)data["kW"],
                (int64_t)data["sD"], (int64_t)data["sH"], (int64_t)data["sW"],
                (int64_t)data["pD"], (int64_t)data["pH"], (int64_t)data["pW"],
                (int)data["count_pad"]
            );
            CHECK(0 == cudaGetLastError());
        }}
        """,
        data={
            "kD": float(kd), "kH": float(kh), "kW": float(kw),
            "sD": float(sd), "sH": float(sh), "sW": float(sw),
            "pD": float(pd), "pH": float(ph), "pW": float(pw),
            "count_pad": float(1 if count_include_pad else 0),
        },
    )


class _AvgPool3dCuda(jt.Function):
    def __init__(self, out_d, out_h, out_w, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_include_pad):
        self.params = (out_d, out_h, out_w, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_include_pad)

    def execute(self, x):
        self.in_shape = tuple(x.shape)
        return _avg_pool3d_fwd_code(x.contiguous(), *self.params)

    def grad(self, dout):
        if dout is None:
            return None
        _, _, _, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_pad = self.params
        with device_scope_like(dout):
            return _avg_pool3d_bwd_code(
                dout.contiguous(), self.in_shape, kd, kh, kw, sd, sh, sw, pd, ph, pw, count_pad
            )


def _avg_pool3d_cuda(x, kernel, strides, pads, ceil_mode, count_include_pad):
    dname = _dtype_name(x.dtype)
    if dname not in ("float32", "float64", "float16", "bfloat16"):
        return None
    from jittor.nn.functional.pooling.average import _pool_output_size
    out_d = _pool_output_size(int(x.shape[2]), kernel[0], strides[0], pads[0], ceil_mode)
    out_h = _pool_output_size(int(x.shape[3]), kernel[1], strides[1], pads[1], ceil_mode)
    out_w = _pool_output_size(int(x.shape[4]), kernel[2], strides[2], pads[2], ceil_mode)
    with device_scope_like(x):
        if jt.flags.no_grad:
            return _avg_pool3d_fwd_code(
                x.contiguous(), out_d, out_h, out_w,
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pads[0], pads[1], pads[2], count_include_pad,
            )
        return _AvgPool3dCuda(
            out_d, out_h, out_w,
            kernel[0], kernel[1], kernel[2],
            strides[0], strides[1], strides[2],
            pads[0], pads[1], pads[2], count_include_pad,
        )(x)


register_kernel("nn.adaptive_avg_pool2d", "cuda", _adaptive_avg_pool2d_cuda)
register_kernel("nn.pool2d", "cuda", _avg_pool2d_cuda)
register_kernel("nn.avg_pool3d", "cuda", _avg_pool3d_cuda)
