"""Non-persistent buffers built under the meta context survive materialization."""
import unittest

import numpy as np

import jittor as jt
import torch
import torch.nn as nn

from _helpers import capability as _test_capability


class TestMetaBufferMaterialization(unittest.TestCase):
    def test_registered_meta_buffer_is_tagged(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "position_ids", torch.arange(6).expand((1, -1)), persistent=False)
            self.assertTrue(getattr(module.position_ids, "_jittor_torch_meta_data", False))

    def test_empty_like_of_meta_buffer_preserves_values(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "position_ids", torch.arange(6).expand((1, -1)), persistent=False)
            materialized = torch.empty_like(module.position_ids, device="cpu")
            np.testing.assert_array_equal(np.asarray(materialized).ravel(), np.arange(6))

    def test_materialization_round_trip_through_module_attribute(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "ids", torch.arange(5).expand((1, -1)), persistent=False)
            # transformers' _move_missing_keys_from_meta_to_device: empty_like
            # the non-persistent buffer and assign it back.
            module.ids = torch.empty_like(module.ids, device="cpu")
            np.testing.assert_array_equal(np.asarray(module.ids).ravel(), np.arange(5))

    def test_plain_empty_like_stays_uninitialized(self):
        with jt.flag_scope(use_cuda=0):
            source = torch.arange(6)
            out = torch.empty_like(source)
            self.assertEqual(tuple(out.shape), (6,))
            self.assertFalse(getattr(source, "_jittor_torch_meta_data", False))


class TestMetaBufferPlacement(unittest.TestCase):
    """Materialized meta buffers must follow the requested device."""

    def test_materialization_forwards_the_requested_device(self):
        # White-box on CPU: the CUDA move itself needs a GPU, so record that
        # the requested device reaches the residency helper with force=True
        # instead of being dropped by the constructor adapter.
        from jittor.compat.torch.installers import factories

        recorded = []
        original = factories._make_cuda_resident

        def recording(value, **kwargs):
            recorded.append(kwargs)
            return value

        factories._make_cuda_resident = recording
        try:
            with jt.flag_scope(use_cuda=0):
                with torch.device("meta"):
                    module = nn.Module()
                    module.register_buffer(
                        "inv_freq", torch.arange(4), persistent=False)
                materialized = factories._materialize_meta_like(
                    module.inv_freq, torch.device("cuda", 0))
            self.assertEqual(len(recorded), 1)
            self.assertIs(recorded[0]["device"].type, "cuda")
            self.assertTrue(recorded[0]["force"])
            np.testing.assert_array_equal(
                np.asarray(materialized).ravel(), np.arange(4))
        finally:
            factories._make_cuda_resident = original

    def test_module_to_casts_a_non_persistent_buffer(self):
        # torch's Module.to touches _buffers, which includes non-persistent
        # entries; the dtype cast is observable without a second device.
        with jt.flag_scope(use_cuda=0):
            module = nn.Module()
            module.register_buffer(
                "inv_freq", torch.arange(4, dtype=torch.float32),
                persistent=False)
            module.to(torch.float64)
            self.assertEqual(str(module.inv_freq.dtype), "torch.float64")

    @unittest.skipUnless(
        _test_capability.check_accelerator("cuda", backend=jt).enabled,
        "requires CUDA",
    )
    def test_meta_buffer_materializes_on_the_requested_cuda_device(self):
        with jt.flag_scope(use_cuda=1):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "inv_freq", torch.arange(4, dtype=torch.float32),
                    persistent=False)
            materialized = torch.empty_like(
                module.inv_freq, device=torch.device("cuda", 0))
            self.assertTrue(materialized.is_cuda)
            np.testing.assert_array_equal(
                materialized.numpy(), np.arange(4, dtype="float32"))

    @unittest.skipUnless(
        _test_capability.check_accelerator("cuda", backend=jt).enabled,
        "requires CUDA",
    )
    def test_module_to_moves_a_non_persistent_buffer(self):
        module = nn.Module()
        module.register_buffer(
            "inv_freq", torch.arange(4, dtype=torch.float32), persistent=False)
        module.to("cuda")
        self.assertTrue(module.inv_freq.is_cuda)


class TestMetaDeviceContextSemantics(unittest.TestCase):
    """``with torch.device('meta')`` marks newly constructed tensors accurately."""

    def test_tensor_and_factories_honor_meta_context_and_nesting(self):
        with jt.flag_scope(use_cuda=0):
            pre_existing = torch.zeros((2,))
            with torch.device("meta"):
                # Pre-existing tensors must keep their original device.
                self.assertEqual(pre_existing.device.type, "cpu")
                self.assertFalse(pre_existing.is_meta)

                # Transformers probes ``torch.tensor([]).device`` to detect
                # meta construction and skip eager weight init.
                probe = torch.tensor([])
                self.assertEqual(probe.device, torch.device("meta"))
                self.assertTrue(probe.is_meta)

                non_empty = torch.tensor([1.0, 2.0])
                self.assertEqual(non_empty.device, torch.device("meta"))
                self.assertTrue(non_empty.is_meta)

                created = torch.empty((3, 4))
                self.assertEqual(created.device, torch.device("meta"))
                self.assertTrue(created.is_meta)

                # Explicit real devices override the ambient meta context.
                explicit_cpu = torch.linspace(0.0, 1.0, 5, device="cpu")
                self.assertEqual(explicit_cpu.device.type, "cpu")
                self.assertFalse(explicit_cpu.is_meta)

                # Nested real-device context overrides outer meta context.
                with torch.device("cpu"):
                    nested = torch.tensor([])
                    self.assertEqual(nested.device.type, "cpu")
                    self.assertFalse(nested.is_meta)

            # Tensors built under meta keep their meta identity after exit,
            # while fresh allocations outside the block are on CPU.
            self.assertEqual(created.device, torch.device("meta"))
            self.assertTrue(created.is_meta)
            materialized = torch.empty_like(created, device="cpu")
            self.assertEqual(materialized.device.type, "cpu")
            self.assertFalse(materialized.is_meta)

    def test_module_to_empty_clears_meta_mark_on_cpu(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.Module()
            with torch.device("meta"):
                module.register_buffer("w", torch.empty((2, 3)))
            self.assertTrue(module.w.is_meta)
            module.to_empty(device="cpu")
            self.assertFalse(module.w.is_meta)
            self.assertEqual(module.w.device.type, "cpu")
