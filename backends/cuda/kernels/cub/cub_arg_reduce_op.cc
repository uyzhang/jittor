// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "core/var.h"
#include "cub_arg_reduce_op.h"
#include <vector>
#include "core/executor.h"
#include "ops/composite/arg_reduce_op.h"
#ifdef JIT_cuda
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/cub.cuh>
#endif

namespace jittor {

#ifndef JIT
CubArgReduceOp::CubArgReduceOp(Var* x, Var* offsets, NanoString op, bool keepdims)
    : x(x), offsets(offsets), op(op), keepdims(keepdims) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    USER_CHECK(offsets->dtype()==ns_int32);
    y = create_output(nullptr, ns_int32);
    y_key = create_output(nullptr, x->dtype());
    set_flag(OpFlags::_manual_set_vnbb);
    y->set_flag(VarFlags::_needed_by_backward);
}

VarPtr CubArgReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return ArgReduceOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

void CubArgReduceOp::infer_shape() {
    int n = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        if (i < x->shape.size() - 1) {
            n *= x->shape[i];
        }
    }
    USER_CHECKop(offsets->shape.size(),==,1);
    USER_CHECKop(offsets->shape[0],==,n + 1);
    NanoVector shape;
    for (int i = 0; i < x->shape.size() - 1; ++i) {
        shape.push_back(x->shape[i]);
    }
    if (keepdims) {
        shape.push_back(1);
    }
    y->set_shape(shape);
    y_key->set_shape(shape);
    // A rank-0 result -- arg_reduce over a 1-D input without keepdims -- is a
    // scalar, which is what ArgReduceOp's own kernel produces and what numpy
    // and torch return. Padding it to shape (1,) here made the same call
    // answer with a different rank depending on whether a CUDA device was
    // present.
    if (shape.size() == 0) {
        y->set_flag(VarFlags::_is_scalar);
        y_key->set_flag(VarFlags::_is_scalar);
    }
}

void CubArgReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Toffsets:" << offsets->dtype();
    jk << "«IS_MIN=" << JK::hex(op == ns_minimum ? 1 : 0);
    jk << "«WARP_ARG=" << JK::hex(
        x->dtype() == ns_float32 || x->dtype() == ns_float64 ||
        x->dtype() == ns_int32 || x->dtype() == ns_int64 ||
        x->dtype() == ns_uint32 || x->dtype() == ns_uint64);
    jk << "«FUNC:";
    if (op==ns_minimum)
        jk << "ArgMin";
    else 
        jk << "ArgMax";
}

#else // JIT
#ifdef JIT_cuda

static __global__ void split(cub::KeyValuePair<int, Tx>* a, Tx* key, int* val, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = blockDim.x * gridDim.x;
    for (int i=tid; i<n; i+=tnum) {
        val[i] = a[i].key;
        key[i] = a[i].value;
    }
}

#if WARP_ARG
// Direct 1-pass warp-per-segment argmax/argmin: writes indices and values
// directly without allocating temporary KeyValuePair buffers or a split kernel.
static __global__ void warp_seg_arg_reduce(
    const Tx* __restrict__ xp,
    const Toffsets* __restrict__ offsets,
    int* __restrict__ yp,
    Tx* __restrict__ y_keyp,
    int num_segments
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_warps = gridDim.x * warps_per_block;

    for (int seg = blockIdx.x * warps_per_block + warp_in_block;
         seg < num_segments;
         seg += total_warps) {
        const int seg_start = (int)offsets[seg];
        const int seg_end = (int)offsets[seg + 1];
        const int seg_len = seg_end - seg_start;
        if (seg_len <= 0) continue;

        bool has_val = false;
        Tx best_v = Tx(0);
        int best_i = 0;
        for (int i = lane; i < seg_len; i += 32) {
            Tx v = xp[seg_start + i];
            if (!has_val) {
                best_v = v;
                best_i = i;
                has_val = true;
            } else {
                #if IS_MIN
                bool better = (v < best_v);
                #else
                bool better = (v > best_v);
                #endif
                if (better) {
                    best_v = v;
                    best_i = i;
                }
            }
        }
        // Warp shuffle reduction preserving first-occurrence tie-break (smaller index wins)
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            Tx ov = __shfl_down_sync(0xffffffffu, best_v, d);
            int oi = __shfl_down_sync(0xffffffffu, best_i, d);
            int ohas = __shfl_down_sync(0xffffffffu, (int)has_val, d);
            if (ohas) {
                if (!has_val) {
                    best_v = ov;
                    best_i = oi;
                    has_val = true;
                } else {
                    #if IS_MIN
                    bool better = (ov < best_v) || (ov == best_v && oi < best_i);
                    #else
                    bool better = (ov > best_v) || (ov == best_v && oi < best_i);
                    #endif
                    if (better) {
                        best_v = ov;
                        best_i = oi;
                    }
                }
            }
        }
        if (lane == 0) {
            yp[seg] = best_i;
            y_keyp[seg] = best_v;
        }
    }
}
#endif

void CubArgReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ offsetsp = offsets->ptr<Toffsets>();

    int num_segments = 1;
    for (int i = 0; i < x->shape.size() - 1; ++i) {
        num_segments *= x->shape[i];
    }
    auto* __restrict__ yp = y->ptr<int>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();

    #if WARP_ARG
    const int last_dim = x->shape.size() ? (int)x->shape[x->shape.size() - 1] : 0;
    if (num_segments >= 8 && last_dim > 0 && last_dim <= 2048) {
        const int blocks = std::min((num_segments + 7) / 8, 65535);
        warp_seg_arg_reduce<<<blocks, 256>>>(xp, offsetsp, yp, y_keyp, num_segments);
        return;
    }
    #endif

    size_t allocation_dout;
    cub::KeyValuePair<int, Tx> *d_out = (cub::KeyValuePair<int, Tx> *)runtime_executor().temp_allocator->alloc(sizeof(cub::KeyValuePair<int, Tx>) * num_segments, allocation_dout);

    // Determine temporary device storage requirementse = NULL;
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, d_out, num_segments, offsetsp, offsetsp + 1);
    // Allocate temporary storage
    size_t allocation;
    d_temp_storage = runtime_executor().temp_allocator->alloc(temp_storage_bytes, allocation);
    // Run sorting operation
    cub::DeviceSegmentedReduce::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, d_out, num_segments, offsetsp, offsetsp + 1);

    split<<<std::max(1,num_segments/1024),1024>>>(d_out, y_keyp, yp, num_segments);

    runtime_executor().temp_allocator->free(d_temp_storage, temp_storage_bytes, allocation);
    runtime_executor().temp_allocator->free(d_out, sizeof(cub::KeyValuePair<int, Tx>) * num_segments, allocation_dout);
}
#endif // JIT_cuda
#endif // JIT

} // jittor
