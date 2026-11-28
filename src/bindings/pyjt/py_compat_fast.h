#pragma once

#include <Python.h>
#include "core/common.h"
#include "core/var_holder.h"

namespace jittor {

// Native fast paths for the torch frontend's per-tensor accessors: each
// replaces a Python function on a path every operator of a model takes.

// `Tensor.dtype` on a frontend tensor: the frontend's dtype object for the
// native dtype, from the `_frontend_dtype_objects` table its type carries, or
// the native name where the table has none (`method_api._dtype_get`).
// @pyjt(_frontend_dtype)
PyObject* frontend_dtype(PyObject* self);

// Differentiate once, preserving missing targets. If every result already
// has its target's dtype, prepare ordinary stopped gradients in native code;
// otherwise return the untouched results for the frontend's normalization.
// The tuple is (gradients, prepared); create_graph keeps derivative edges.
// @pyjt(_grad_optional_prepare)
PyObject* grad_optional_prepare(VarHolder* loss,
    const vector<VarHolder*>& targets, bool retain_graph, bool create_graph);

// Batch the existing imperative storage writes after the optimizer has
// resolved its group readers. False leaves every destination untouched.
// Only materialized, dense CUDA tensors outside capture are accepted.
// @pyjt(_try_optimizer_copies)
bool try_optimizer_copies(PyObject* targets, PyObject* values);

// Read current canonical holder metadata as one batch. None leaves observable
// Python descriptors, subclasses and container iteration to the caller.
// @pyjt(_try_optimizer_shape_changes)
PyObject* try_optimizer_shape_changes(PyObject* params, PyObject* moments,
    PyObject* variances, PyObject* canonical_types);

// Fresh Adam entries or independently advanced, unexposed integer counters.
// None leaves custom descriptors, counters and containers untouched. The
// caller keeps provider selection between these two separate operations.
// @pyjt(_try_optimizer_adam_entries)
PyObject* try_optimizer_adam_entries(PyObject* group, PyObject* gradients,
    PyObject* steps, PyObject* canonical_types, PyObject* protocol, bool advance);

// Fresh optimizer update metadata only: unknown Python protocols stay on the
// caller's ordinary path. No holder, value or eligibility survives the call.
// @pyjt(_try_pack_optimizer_updates)
PyObject* try_pack_optimizer_updates(PyObject* active, PyObject* updates,
    PyObject* canonical_types, PyObject* writer, PyObject* protocol);

// True/False for ordinary class attributes; None preserves Python descriptors.
// @pyjt(_try_optimizer_writers)
PyObject* try_optimizer_writers(PyObject* targets, PyObject* writer,
    PyObject* protocol);

// Publish this backward's fresh stopped gradients into empty ordinary slots.
// False leaves custom accessors, accumulation and shared targets to Python.
// All eligibility and holder metadata are read anew on every call.
// @pyjt(_try_publish_fresh_gradients)
bool try_publish_fresh_gradients(PyObject* groups, PyObject* gradients,
    PyObject* filled, PyObject* canonical_types, PyObject* metadata_protocol,
    PyObject* state_protocol);


// Defer the binary update of an unallocated, unaliased result. The caller
// may opt in for a cast when the update is its own temporary, which can be
// assigned without copying an externally owned RHS. False keeps the store.
// @pyjt(_try_defer_unaliased_inplace)
bool try_defer_unaliased_inplace(PyObject* self, PyObject* value, bool fresh_result = false);

// An already allocated, unaliased CUDA destination can be updated by a fresh
// write_back op. Its reader dependencies preserve snapshots without draining
// unrelated pending work; its output retains the original storage.
// @pyjt(_try_ordered_unaliased_inplace)
bool try_ordered_unaliased_inplace(PyObject* self, PyObject* value);

// What the fast paths below call back to decide: `tracing()`, whether a
// step is being captured (`jittor._runtime.step_capture.tracing`). The optional
// concat continuation resumes after provider selection; selector_state keeps
// the Python selection owners so an override leaves the native short path.
// @pyjt(_compat_fast_bind)
void compat_fast_bind(PyObject* tracing, PyObject* concat_fallback=nullptr,
                      PyObject* selector_state=nullptr, PyObject* getitem_selector_state=nullptr);

// `torch.cat(tensors, dim)` for the common case -- two to 64 non-empty Vars
// of one dtype, same rank, on the current device -- calls the selected backend
// provider once, or builds `empty` plus `setitem` when none is registered.
// A declining provider resumes the Python concat after selection. Anything
// outside this contract answers NotImplemented for the Python `cat`.
// @pyjt(_fast_cat)
PyObject* fast_cat(PyObject* tensors, int64 dim);

// `tensor[index]` for a basic index -- ints, slices, None and Ellipsis,
// alone or in a tuple -- on a frontend tensor that carries no CPU residency
// hint and no `.data` ownership record. A registered provider is tried once
// for no-grad, static positive-step indexing, then a decline continues with
// native getitem. New views record `self`; an existing provider view retains
// its recorded owner. Unsupported spellings use the Python path.
// @pyjt(_fast_getitem)
PyObject* fast_getitem(PyObject* self, PyObject* index);

// The native operators the binary fast path calls: `natives` holds the
// jittor Var's own `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`,
// `__rmul__`, `__truediv__` and `__rtruediv__` as captured before the frontend
// replaced them (slot wrappers; anything else leaves that operator on the
// Python path), and `mark_cpu_like` is the frontend's `_mark_cpu_like`.
// `widen_scalar_division`: take a floating tensor over a Python float here,
// widened as `_true_division` does (everywhere but ACL).
// @pyjt(_compat_fast_bind_binary)
void compat_fast_bind_binary(PyObject* natives, PyObject* mark_cpu_like,
                             bool widen_scalar_division=false);

// `tensor <op> other` for the common case of `method_api._promoting_binary`
// and `_true_division`: a Var of the same dtype (not unsigned; floating for a
// division), or a Python float against a floating tensor or a Python int
// against a floating or signed-integer one, which keep the tensor's dtype.
// `code` indexes the operators above. None for anything else, and the Python
// operator takes the call.
// @pyjt(_fast_binary)
PyObject* fast_binary(PyObject* self, PyObject* other, int code);

// `nn.gelu(x)` (exact) as its Python body builds it, operator for operator:
// `0.5 * x * (1.0 + erf(x * 0.7071067811865476))`, x widened to float32 for a
// half type and the result cast back, each product and sum taken as
// `_fast_binary` takes it. None when the binary operators are not bound, x is
// not floating, a kernel is registered for "nn.gelu", or any step declines.
// @pyjt(_fast_gelu)
PyObject* fast_gelu(PyObject* x);

// `tensor.view(*shape)` / `reshape(*shape)` of a dense tensor, shape given as
// ints or one tuple or list of them: the reshape op, recorded as a storage
// view of `self`, as `jittor.view` does. None for anything else.
// @pyjt(_fast_view)
PyObject* fast_view(PyObject* self, PyObject* shape);

// Materialize a CPU/CUDA storage view under its frontend owner's type and
// its own explicit placement. None for legacy placement or other backends.
// @pyjt(_fast_contiguous)
PyObject* fast_contiguous(PyObject* self, PyObject* tensor_type);

// Torch's two-axis transpose as a physical storage view, retaining the
// immediate input's autograd history. Exact integer axes only; other spellings
// and invalid axes return None for the Python API's validation.
// @pyjt(_fast_storage_transpose)
PyObject* fast_storage_transpose(PyObject* self, PyObject* dims);

// `tensor.unsqueeze(dim)` of a dense tensor, as the reshape above.
// @pyjt(_fast_unsqueeze)
PyObject* fast_unsqueeze(PyObject* self, int64 dim);

// `tensor.transpose(dim0, dim1)`, as `jittor.transpose` builds it: composed
// with the transpose this tensor is a view of, or is computed by, and
// recorded as a transpose view. None when a transpose kernel is registered.
// @pyjt(_fast_transpose)
PyObject* fast_transpose(PyObject* self, int64 dim0, int64 dim1);

// `tensor.permute(axes)` / `transpose(*axes)` for a permutation of exact,
// in-range, distinct non-negative ints, built as `_fast_transpose` builds it.
// None for anything else, and `jittor.transpose` reports it.
// @pyjt(_fast_permute)
PyObject* fast_permute(PyObject* self, PyObject* axes);

// `jittor.Var.clamp(min_v, max_v)` for a plain Var of a float or integer
// dtype between Python int/float bounds (either may be None, not both): the
// graph `_clamp_cpu` builds, op for op -- each bound converted the way the
// generated bindings convert a Python number, the integer path's maximum and
// minimum, the general path's select and NaN keep -- built in one call
// instead of sixteen. NotImplemented for anything else, and for an empty or
// dynamically shaped Var, which `_clamp_cpu` answers itself.
// @pyjt(_fast_clamp)
PyObject* fast_clamp(PyObject* self, PyObject* min_v, PyObject* max_v);

// `jittor.any(x, dims, keepdims)` (`all` when `is_and`) of a bool Var on CPU
// or CUDA: the logical reduction with a bool result, where `any_`/`all_`
// reduce a bool input into int32 and the caller then casts that back -- a
// second kernel for every truth test. NotImplemented for any other dtype or
// backend, for `dims` other than an int or a sequence of in-range ints,
// and `any_(...).bool()` takes the call.
// @pyjt(_logical_reduce)
PyObject* logical_reduce(PyObject* self, PyObject* dims, bool keepdims, bool is_and);

// `_fast_getitem` and `_fast_cat` as jittor's own `getitem` and `concat`
// reach them: each Var operand is first read the way a generated binding
// reads it (`VarHolder::grad_input`, which brings an aliased view up to date
// with its source's in-place writes while gradients are recorded), so the
// native path sees what the Python path it stands in for would have built on.
//
// `gather`: also answer the index forms holding index Vars -- a lone bool
// mask, a lone index Var, a tuple with index Vars in it -- as jittor's
// `_fast_getitem` answers them. The caller passes whether that path applies,
// i.e. whether `Var.getitem` is still jittor's own.
// @pyjt(_jt_basic_getitem)
PyObject* jt_basic_getitem(PyObject* self, PyObject* index, bool gather=false);
// @pyjt(_jt_concat)
PyObject* jt_concat(PyObject* tensors, int64 dim);

// `x[index] = value` as jittor's `setitem` and `var_setitem` together build
// it, for a plain Var `x` (not complex64, not on ACL) and a Var or exact
// Python int, float or bool `value`: an index of ints, slices, None,
// Ellipsis and index Vars, or a lone bool mask -- the `ternary` a mask of
// `x`'s own shape becomes under a scalar or one-element value, its `where()`
// coordinates otherwise. Returns `x`, assigned. NotImplemented for anything
// else, and the Python path takes the call. The caller has checked that no
// setitem kernel is registered and that `Var.setitem` is still jittor's own.
// @pyjt(_jt_setitem)
PyObject* jt_setitem(PyObject* self, PyObject* index, PyObject* value);

// Direct eager precompiled CUDA float32 fast paths for already-materialized
// contiguous CUDA tensors under no_grad / stop_grad outside graph capture.
// Each returns None whenever any guard declines so the caller falls through
// to the standard lazy/autograd/multi-dtype implementation.
// @pyjt(_cuda_eager_unary)
PyObject* cuda_eager_unary(PyObject* x, int op_id, double p0=0.0, double p1=0.0, int flags=0);
// @pyjt(_cuda_eager_binary)
PyObject* cuda_eager_binary(PyObject* x, PyObject* y, int op_id);
// @pyjt(_cuda_eager_var_std)
PyObject* cuda_eager_var_std(PyObject* x, int dim, double correction, bool keepdim, bool take_sqrt);
// @pyjt(_cuda_eager_prod)
PyObject* cuda_eager_prod(PyObject* x, int dim, bool keepdim);
// @pyjt(_cuda_eager_minmax_dim)
PyObject* cuda_eager_minmax_dim(PyObject* x, int dim, bool keepdim, bool is_min);
// @pyjt(_cuda_eager_scan)
PyObject* cuda_eager_scan(PyObject* x, int dim, bool is_prod);
// @pyjt(_cuda_eager_cross_entropy)
PyObject* cuda_eager_cross_entropy(PyObject* x, PyObject* target, PyObject* weight,
    bool has_ignore, int64 ignore_idx, int red_mode);
// @pyjt(_cuda_eager_nll_loss)
PyObject* cuda_eager_nll_loss(PyObject* x, PyObject* target, PyObject* weight,
    int64 ignore_idx, int red_mode);
// @pyjt(_cuda_eager_bce_with_logits)
PyObject* cuda_eager_bce_with_logits(PyObject* x, PyObject* target, int red_mode);
// @pyjt(_cuda_eager_adaptive_avg_pool2d)
PyObject* cuda_eager_adaptive_avg_pool2d(PyObject* x, int64 oh, int64 ow);
// @pyjt(_cuda_eager_avg_pool2d)
PyObject* cuda_eager_avg_pool2d(PyObject* x, int64 kh, int64 kw, int64 sh, int64 sw,
    int64 ph, int64 pw, bool ceil_mode, bool count_include_pad);
// @pyjt(_cuda_eager_avg_pool3d)
PyObject* cuda_eager_avg_pool3d(PyObject* x, int64 kd, int64 kh, int64 kw,
    int64 sd, int64 sh, int64 sw, int64 pd, int64 ph, int64 pw,
    bool ceil_mode, bool count_include_pad);
// @pyjt(_cuda_eager_grid_sample_2d)
PyObject* cuda_eager_grid_sample_2d(PyObject* x, PyObject* grid, int mode_id, int pad_id, bool align_corners);
// @pyjt(_cuda_eager_resize_bilinear)
PyObject* cuda_eager_resize_bilinear(PyObject* x, int64 oh, int64 ow, bool align_corners, bool tf_mode);
// @pyjt(_cuda_eager_affine_grid_4d)
PyObject* cuda_eager_affine_grid_4d(PyObject* theta, int64 n, int64 h, int64 w, bool align_corners);
// @pyjt(_cuda_eager_triu_tril)
PyObject* cuda_eager_triu_tril(PyObject* x, int64 diagonal, bool is_lower);
// @pyjt(_cuda_eager_trace)
PyObject* cuda_eager_trace(PyObject* x);

} // namespace jittor
