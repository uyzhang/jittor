// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <map>
#include "stream_compat.h"
#include "cufft_wrapper.h"
#include "runtime/device.h"
#include "runtime/backend.h"
#include "runtime/graph_capture.h"
#include "runtime/tensor_placement.h"
#include "core/var_holder.h"
#include "mem/allocator.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "device_plan_cache.h"

namespace jittor {

DECLARE_FLAG(bool, no_grad);

int cufft_max_cache_size = 32;
struct DestroyCufftPlan {
    static bool destroy(cufftHandle plan) {
        auto status = cufftDestroy(plan);
        peekCudaErrorsAlways(status);
        return status == CUFFT_SUCCESS;
    }
};
using CufftDeviceCache = DevicePlanCache<CufftPlanKey, cufftHandle,
    CufftPlanKeyHash, CufftPlanKeyEq, DestroyCufftPlan>;
static std::map<int, std::unique_ptr<CufftDeviceCache>> cufft_devices;

static CufftDeviceCache& cufft_cache(int device) {
    auto found = cufft_devices.find(device);
    if (found != cufft_devices.end()) return *found->second;
    auto cache = std::unique_ptr<CufftDeviceCache>(
        new CufftDeviceCache(device, cuda_compute_stream(device)));
    auto* result = cache.get();
    cufft_devices.emplace(device, std::move(cache));
    return *result;
}

int cufft_plan_cache_size(int device) {
    ExecutorEntryScope entry;
    int total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->plans.size();
    return total;
}

void cufft_set_plan_cache_size(int size) {
    ExecutorEntryScope entry;
    cufft_max_cache_size = size < 1 ? 1 : size;
    for (auto& bank : cufft_devices) bank.second->trim(cufft_max_cache_size);
}

cufftHandle cufft_get_plan(const CufftPlanKey& key) {
    ExecutorEntryScope entry;
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    // Same as the cuTT cache: cufft_fft_op fills key.device from
    // cudaGetDevice() immediately before calling in, so this is an internal
    // invariant and not a caller-supplied value.
    ASSERT(device == key.device) << "cuFFT plan key must name the current device";
    auto& cache = cufft_cache(device);
    if (auto* found = cache.find(key)) {
        CUFFT_CALL(cufftSetStream(found->value, cache.stream));
        cache.binds++;
        return found->value;
    }

    cache.reserve_one(cufft_max_cache_size);
    std::unique_ptr<CufftDeviceCache::Plan> plan(new CufftDeviceCache::Plan(&cache));
    // MakePlanMany initializes this existing handle (unlike PlanMany, which
    // creates another one). The owner is armed before workspace setup so a
    // failed build or stream bind also releases the handle.
    CUFFT_CALL(cufftCreate(&plan->value));
    plan->ready = true;
    size_t workspace = 0;
    if (key.n0 == 1) {
        int n1[1] = {(int)key.n1};
        int idist = n1[0];
        int odist = n1[0];
        if (key.type == CUFFT_R2C || key.type == CUFFT_D2Z) {
            odist = n1[0] / 2 + 1;
        } else if (key.type == CUFFT_C2R || key.type == CUFFT_Z2D) {
            idist = n1[0] / 2 + 1;
        }
        CUFFT_CALL(cufftMakePlanMany(plan->value, 1, n1,
            nullptr, 1, idist, nullptr, 1, odist,
            (cufftType)key.type, (int)key.batch, &workspace));
    } else {
        int n[2] = {(int)key.n0, (int)key.n1};
        CUFFT_CALL(cufftMakePlanMany(plan->value, 2, n,
            nullptr, 1, n[0] * n[1], nullptr, 1, n[0] * n[1],
            (cufftType)key.type, (int)key.batch, &workspace));
    }
    CUFFT_CALL(cufftSetStream(plan->value, cache.stream));
    cache.binds++;
    return cache.publish(key, std::move(plan));
}

uint64 cufft_stream_bind_count(int device) {
    ExecutorEntryScope entry;
    auto found = cufft_devices.find(device);
    return found == cufft_devices.end() ? 0 : found->second->binds;
}

uint64 cufft_plan_build_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->builds;
    return total;
}

uint64 cufft_plan_destroy_count(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroys;
    return total;
}

uint64 cufft_plan_destroy_failures(int device) {
    ExecutorEntryScope entry;
    uint64 total = 0;
    for (const auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) total += bank.second->destroy_failures;
    return total;
}

void cufft_clear_plan_cache(int device) {
    ExecutorEntryScope entry;
    for (auto& bank : cufft_devices)
        if (device < 0 || bank.first == device) bank.second->clear();
}

void cufft_shutdown() {
    // No Runtime access or executor entry during static destruction. Each bank
    // remembers its device/stream and restores the caller's raw CUDA device.
    for (auto& bank : cufft_devices) bank.second->clear();
    cufft_devices.clear();
    LOGv << "cufftDestroy finished";
}

struct cufft_initer {
    ~cufft_initer() { cufft_shutdown(); }
} init;

namespace {

inline PyObject* cufft_none() {
    Py_INCREF(Py_None);
    return Py_None;
}

inline bool cufft_can_eager(PyObject* obj, Var*& out_var, int& dev_id) {
    if (graph_capture_active() || accelerator_backend_id() != BackendId::Cuda) return false;
    if (!PyObject_TypeCheck(obj, &PyjtVarHolder.ht_type)) return false;
    PyTypeObject* t = Py_TYPE(obj);
    if (t != &PyjtVarHolder.ht_type
            && (!t->tp_dict || PyDict_GetItemString(t->tp_dict, "_frontend_backend") == nullptr))
        return false;
    auto* vh = GET_RAW_PTR(VarHolder, obj);
    if (vh->view) return false;
    Var* v = vh->var;
    if (!no_grad && !v->is_stop_grad()) return false;
    if (!v->mem_ptr || !v->is_finished() || !v->allocator || !v->allocator->is_cuda()) return false;
    if (!v->is_contiguous() || v->device_id < 0 || v->num <= 0 || v->shape.size() == 0) return false;
    out_var = v;
    dev_id = v->device_id;
    return true;
}

inline VarPtr cufft_alloc_var(const NanoVector& shape, NanoString dtype, int dev_id) {
    VarPtr vp(shape, dtype);
    Device dev{BackendId::Cuda, dev_id};
    Allocator* alloc = get_allocator(dev, false);
    Allocation buf(alloc, vp->size);
    vp->mem_ptr = buf.ptr;
    vp->allocator = buf.allocator;
    vp->allocation = buf.allocation;
    buf.ptr = nullptr;
    buf.allocator = nullptr;
    buf.allocation = 0;
    vp->placement = TensorPlacement(dev);
    vp->device_id = dev_id;
    vp->set_flag(VarFlags::_stop_fuse, 1);
    vp->flags.set(NodeFlags::_stop_grad, 1);
    return vp;
}

inline PyObject* cufft_wrap(PyObject* like, VarPtr&& vp) {
    unique_ptr<VarHolder> owned(new VarHolder(std::move(vp)));
    PyTensorFrontendScope scope(like, nullptr, 0, false);
    return to_py_object<VarHolder*>(owned.release());
}

} // namespace

PyObject* cufft_eager_rfft_1d(PyObject* x, double scale) {
    Var* xv = nullptr;
    int dev_id = -1;
    if (!cufft_can_eager(x, xv, dev_id)) return cufft_none();
    NanoString dt = xv->dtype();
    if (dt != ns_float32 && dt != ns_float64) return cufft_none();
    int nd = (int)xv->shape.size();
    int64 L = xv->shape[nd - 1];
    if (L <= 0) return cufft_none();
    int64 keep = L / 2 + 1;
    int64 batch = xv->num / L;
    if (dev_id != current_device()) set_current_device(dev_id);

    CufftPlanKey key;
    std::memset(&key, 0, sizeof(key));
    key.batch = batch;
    key.n0 = 1;
    key.n1 = L;
    key.type = (int64)(dt == ns_float64 ? CUFFT_D2Z : CUFFT_R2C);
    key.device = dev_id;
    cufftHandle plan = cufft_get_plan(key);

    NanoVector out_shape;
    for (int i = 0; i < nd - 1; ++i) out_shape.push_back(xv->shape[i]);
    out_shape.push_back(keep);
    NanoString out_dt = (dt == ns_float64) ? ns_complex128 : ns_complex64;
    VarPtr out = cufft_alloc_var(out_shape, out_dt, dev_id);

    if (dt == ns_float32) {
        CUFFT_CALL(cufftExecR2C(plan, (cufftReal*)xv->mem_ptr, (cufftComplex*)out->mem_ptr));
        if (scale != 1.0) {
            cufft_scale_inplace_f32((float*)out->mem_ptr, out->num * 2, (float)scale, cuda_compute_stream(dev_id));
        }
    } else {
        CUFFT_CALL(cufftExecD2Z(plan, (cufftDoubleReal*)xv->mem_ptr, (cufftDoubleComplex*)out->mem_ptr));
        if (scale != 1.0) {
            cufft_scale_inplace_f64((double*)out->mem_ptr, out->num * 2, scale, cuda_compute_stream(dev_id));
        }
    }
    return cufft_wrap(x, std::move(out));
}

PyObject* cufft_eager_irfft_1d(PyObject* x, int64 length, double scale) {
    Var* xv = nullptr;
    int dev_id = -1;
    if (!cufft_can_eager(x, xv, dev_id)) return cufft_none();
    NanoString dt = xv->dtype();
    if (dt != ns_complex64 && dt != ns_complex128) return cufft_none();
    int nd = (int)xv->shape.size();
    int64 keep = length / 2 + 1;
    if (length <= 0 || xv->shape[nd - 1] != keep) return cufft_none();
    int64 batch = xv->num / keep;
    if (dev_id != current_device()) set_current_device(dev_id);

    CufftPlanKey key;
    std::memset(&key, 0, sizeof(key));
    key.batch = batch;
    key.n0 = 1;
    key.n1 = length;
    key.type = (int64)(dt == ns_complex128 ? CUFFT_Z2D : CUFFT_C2R);
    key.device = dev_id;
    cufftHandle plan = cufft_get_plan(key);

    auto stream = cuda_compute_stream(dev_id);
    Allocator* talloc = get_allocator(Device{BackendId::Cuda, dev_id}, true);
    Allocation scratch(talloc, xv->size);
    checkCudaErrors(cudaMemcpyAsync(scratch.ptr, xv->mem_ptr, xv->size, cudaMemcpyDeviceToDevice, stream));

    NanoVector out_shape;
    for (int i = 0; i < nd - 1; ++i) out_shape.push_back(xv->shape[i]);
    out_shape.push_back(length);
    NanoString out_dt = (dt == ns_complex128) ? ns_float64 : ns_float32;
    VarPtr out = cufft_alloc_var(out_shape, out_dt, dev_id);

    if (dt == ns_complex64) {
        CUFFT_CALL(cufftExecC2R(plan, (cufftComplex*)scratch.ptr, (cufftReal*)out->mem_ptr));
        if (scale != 1.0) {
            cufft_scale_inplace_f32((float*)out->mem_ptr, out->num, (float)scale, stream);
        }
    } else {
        CUFFT_CALL(cufftExecZ2D(plan, (cufftDoubleComplex*)scratch.ptr, (cufftDoubleReal*)out->mem_ptr));
        if (scale != 1.0) {
            cufft_scale_inplace_f64((double*)out->mem_ptr, out->num, scale, stream);
        }
    }
    return cufft_wrap(x, std::move(out));
}

PyObject* cufft_eager_c2c_1d(PyObject* x, bool inverse, double scale) {
    Var* xv = nullptr;
    int dev_id = -1;
    if (!cufft_can_eager(x, xv, dev_id)) return cufft_none();
    NanoString dt = xv->dtype();
    if (dt != ns_complex64 && dt != ns_complex128) return cufft_none();
    int nd = (int)xv->shape.size();
    int64 L = xv->shape[nd - 1];
    if (L <= 0) return cufft_none();
    int64 batch = xv->num / L;
    if (dev_id != current_device()) set_current_device(dev_id);

    CufftPlanKey key;
    std::memset(&key, 0, sizeof(key));
    key.batch = batch;
    key.n0 = 1;
    key.n1 = L;
    key.type = (int64)(dt == ns_complex128 ? CUFFT_Z2Z : CUFFT_C2C);
    key.device = dev_id;
    cufftHandle plan = cufft_get_plan(key);

    VarPtr out = cufft_alloc_var(xv->shape, dt, dev_id);
    auto stream = cuda_compute_stream(dev_id);
    if (dt == ns_complex64) {
        CUFFT_CALL(cufftExecC2C(plan, (cufftComplex*)xv->mem_ptr, (cufftComplex*)out->mem_ptr,
                                inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
        if (scale != 1.0) {
            cufft_scale_inplace_f32((float*)out->mem_ptr, out->num * 2, (float)scale, stream);
        }
    } else {
        CUFFT_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex*)xv->mem_ptr, (cufftDoubleComplex*)out->mem_ptr,
                                inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
        if (scale != 1.0) {
            cufft_scale_inplace_f64((double*)out->mem_ptr, out->num * 2, scale, stream);
        }
    }
    return cufft_wrap(x, std::move(out));
}

PyObject* cufft_eager_fft_real_1d(PyObject* x, bool inverse, double scale) {
    Var* xv = nullptr;
    int dev_id = -1;
    if (!cufft_can_eager(x, xv, dev_id)) return cufft_none();
    NanoString dt = xv->dtype();
    if (dt != ns_float32 && dt != ns_float64) return cufft_none();
    int nd = (int)xv->shape.size();
    int64 L = xv->shape[nd - 1];
    if (L <= 0) return cufft_none();
    int64 keep = L / 2 + 1;
    int64 batch = xv->num / L;
    if (dev_id != current_device()) set_current_device(dev_id);

    CufftPlanKey key;
    std::memset(&key, 0, sizeof(key));
    key.batch = batch;
    key.n0 = 1;
    key.n1 = L;
    key.type = (int64)(dt == ns_float64 ? CUFFT_D2Z : CUFFT_R2C);
    key.device = dev_id;
    cufftHandle plan = cufft_get_plan(key);

    auto stream = cuda_compute_stream(dev_id);
    Allocator* talloc = get_allocator(Device{BackendId::Cuda, dev_id}, true);
    size_t half_bytes = (size_t)batch * (size_t)keep * (dt == ns_float64 ? 16u : 8u);
    Allocation half_buf(talloc, half_bytes);

    NanoString out_dt = (dt == ns_float64) ? ns_complex128 : ns_complex64;
    VarPtr out = cufft_alloc_var(xv->shape, out_dt, dev_id);
    if (dt == ns_float32) {
        CUFFT_CALL(cufftExecR2C(plan, (cufftReal*)xv->mem_ptr, (cufftComplex*)half_buf.ptr));
        cufft_expand_hermitian_f32((const float*)half_buf.ptr, (float*)out->mem_ptr,
                                   batch, keep, L, (float)scale, inverse ? 1 : 0, stream);
    } else {
        CUFFT_CALL(cufftExecD2Z(plan, (cufftDoubleReal*)xv->mem_ptr, (cufftDoubleComplex*)half_buf.ptr));
        cufft_expand_hermitian_f64((const double*)half_buf.ptr, (double*)out->mem_ptr,
                                   batch, keep, L, scale, inverse ? 1 : 0, stream);
    }
    return cufft_wrap(x, std::move(out));
}

} // jittor
