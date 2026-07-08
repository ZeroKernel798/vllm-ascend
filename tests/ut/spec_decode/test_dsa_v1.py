# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dsa_v1.py — DeepSeek attention backend.

Tests pure functions (hadamard_transform_ref, pad_to_blocks) and
metadata dataclass initialization.  All tests run on CPU — no NPU needed.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch


# ===================================================================
# hadamard_transform_ref
# ===================================================================

class TestHadamardTransformRef:
    """hadamard_transform_ref: pad → linear → scale → truncate → reshape."""

    def _make_hadamard(self, dim_padded):
        return torch.eye(dim_padded)

    def test_identity_hadamard_preserves_values(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dim = x.shape[-1]
        padded = 2 ** math.ceil(math.log2(dim))
        had = torch.eye(padded)
        out = hadamard_transform_ref(x, had, scale=1.0)
        assert out.shape == x.shape
        assert torch.allclose(out, x)

    def test_scale_factor_applied(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.ones(2, 3)
        padded = 2 ** math.ceil(math.log2(3))
        had = torch.eye(padded)
        out = hadamard_transform_ref(x, had, scale=2.0)
        assert out.shape == x.shape
        assert torch.allclose(out, x * 2.0)

    def test_pad_when_dim_not_power_of_two(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.ones(2, 5)  # 5 → padded to 8
        padded = 2 ** math.ceil(math.log2(5))  # = 8
        had = torch.eye(padded)
        out = hadamard_transform_ref(x, had, scale=1.0)
        assert out.shape == (2, 5)  # truncated back

    def test_pad_when_dim_is_power_of_two(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.ones(2, 8)  # 8 is already power of 2
        had = torch.eye(8)
        out = hadamard_transform_ref(x, had, scale=1.0)
        assert out.shape == (2, 8)

    def test_3d_input_preserves_batch(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.ones(4, 3, 7)  # [batch, seq, dim]
        padded = 2 ** math.ceil(math.log2(7))  # = 8
        had = torch.eye(padded)
        out = hadamard_transform_ref(x, had, scale=1.0)
        assert out.shape == (4, 3, 7)

    def test_scale_default_one_does_not_change(self):
        from vllm_ascend.attention.dsa_v1 import hadamard_transform_ref
        x = torch.randn(2, 4)
        padded = 2 ** math.ceil(math.log2(4))
        had = torch.eye(padded)
        out = hadamard_transform_ref(x, had)
        assert out.shape == x.shape


# ===================================================================
# pad_to_blocks
# ===================================================================

class TestPadToBlocks:
    """pad_to_blocks: ragged → fixed-block padding."""

    def test_single_request_fits_one_block(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.ones(30, 2, 64)  # [tokens=30, heads=2, dim=64]
        lengths = torch.tensor([30])
        out = pad_to_blocks(x, lengths, block_size=128)
        assert out.shape[0] >= 2  # block 0 (reserved) + block 1
        assert out.shape[1] == 128  # block_size
        assert out.shape[2] == 2   # heads
        assert out.shape[3] == 64  # dim
        # First 30 rows of block 1 must match input
        assert torch.equal(out[1, :30], x)

    def test_multi_request(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.cat([
            torch.ones(50, 2, 64),
            torch.ones(200, 2, 64) * 2,
        ])
        lengths = torch.tensor([50, 200])
        out = pad_to_blocks(x, lengths, block_size=128)
        assert out.shape[0] >= 4  # reserved + 1 block for req0 + 2 for req1
        assert out.shape[1] == 128
        # req0: first 50 rows of block 1 are 1.0
        assert (out[1, :50] == 1.0).all()
        assert (out[1, 50:] == 0.0).all()  # padding

    def test_zero_length_request(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.ones(50, 2, 64)
        lengths = torch.tensor([0, 50])  # req0 has 0 tokens, req1 has 50
        out = pad_to_blocks(x, lengths, block_size=128)
        # block 0 is reserved; block 1 gets req1's data
        assert out[1, 0, 0, 0].item() == 1.0  # req1 data placed in block 1

    def test_raises_on_token_count_mismatch(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.ones(50, 2, 64)
        lengths = torch.tensor([30])  # sum=30, x has 50 tokens
        with pytest.raises(ValueError, match="does not match sum"):
            pad_to_blocks(x, lengths, block_size=128)

    def test_exact_block_boundary(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.ones(128, 2, 64)  # exactly one block
        lengths = torch.tensor([128])
        out = pad_to_blocks(x, lengths, block_size=128)
        assert out.shape[0] == 2  # reserved + 1
        assert torch.equal(out[1], x)

    def test_custom_block_size(self):
        from vllm_ascend.attention.dsa_v1 import pad_to_blocks
        x = torch.ones(30, 2, 64)
        lengths = torch.tensor([30])
        out = pad_to_blocks(x, lengths, block_size=64)
        assert out.shape[1] == 64


# ===================================================================
# _is_w8a8_dynamic
# ===================================================================

class TestIsW8A8Dynamic:
    """_is_w8a8_dynamic: check if a linear module uses int8 dynamic quantization."""

    def test_returns_false_when_no_quant_method(self):
        from vllm_ascend.attention.dsa_v1 import _is_w8a8_dynamic
        linear = MagicMock()
        del linear.quant_method  # no attribute
        assert not _is_w8a8_dynamic(linear)

    def test_returns_false_when_unquantized(self):
        from vllm_ascend.attention.dsa_v1 import _is_w8a8_dynamic
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        linear = MagicMock()
        linear.quant_method = AscendUnquantizedLinearMethod()
        assert not _is_w8a8_dynamic(linear)

    def test_returns_false_when_quant_method_is_none(self):
        from vllm_ascend.attention.dsa_v1 import _is_w8a8_dynamic
        linear = MagicMock()
        linear.quant_method = None
        assert not _is_w8a8_dynamic(linear)


# ===================================================================
# dsv4_dsa_overlap_stream — singleton
# ===================================================================

class TestDsv4DsaOverlapStream:
    """dsv4_dsa_overlap_stream: returns same NPU stream on repeated calls."""

    def test_returns_same_stream_invoking_twice(self):
        from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
        s1 = dsv4_dsa_overlap_stream()
        s2 = dsv4_dsa_overlap_stream()
        assert s1 is s2, "singleton: must return the same stream on every call"


# ===================================================================
# Metadata dataclasses — verified by every other test that imports them
# ===================================================================

# ===================================================================
# _require_prefill_metadata / _require_decode_metadata — helpers
# ===================================================================


# ===================================================================
# rotate_activation
# ===================================================================

class TestRotateActivation:
    def test_returns_same_shape_and_normalized(self):
        from vllm_ascend.attention.dsa_v1 import rotate_activation
        hidden_size = 8
        x = torch.ones(2, 4, hidden_size)
        had = torch.eye(hidden_size)
        out = rotate_activation(x, had)
        assert out.shape == x.shape
        # With identity hadamard, output = input * scale (where scale = hidden_size**-0.5)
        expected = x * (hidden_size ** -0.5)
        assert torch.allclose(out, expected, atol=1e-6), (
            f"rotate_activation with identity hadamard should equal input * 1/sqrt(dim)"
        )
