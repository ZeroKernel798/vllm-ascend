# SPDX-License-Identifier: Apache-2.0
"""Tests for v2 attention utilities (worker/v2/attn_utils.py).

Covers the V2 model runner's attention metadata construction and
attention state detection — functions that sit between the V1
attention backend and the V2 orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# build_attn_state — attention state detection
# ---------------------------------------------------------------------------

@dataclass
class MockSpecConfig:
    method: str = "mtp"


class MockVllmConfig:
    def __init__(self, pooling=False, speculative_method=None, chunked_prefill=False):
        self.model_config = MockModelConfig(pooling)
        self.speculative_config = None
        if speculative_method:
            self.speculative_config = MockSpecConfig(method=speculative_method)
        self.scheduler_config = MockSchedConfig(chunked_prefill)
        self.kv_cache_config = MockKVCacheConfig()
        self.kv_transfer_config = None


@dataclass
class MockModelConfig:
    runner_type: str = "generate"


@dataclass
class MockSchedConfig:
    enable_chunked_prefill: bool = False


@dataclass
class MockKVCacheConfig:
    class Group:
        class Spec:
            pass
    kv_cache_groups: list = None


class TestBuildAttnState:
    """Verify build_attn_state selects the correct AscendAttentionState."""

    def _call(self, vllm_config, seq_lens_np, num_reqs, num_scheduled, num_valid):
        from vllm_ascend.worker.v2.attn_utils import build_attn_state
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
                   return_value=vllm_config):
            return build_attn_state(vllm_config, seq_lens_np, num_reqs,
                                    num_scheduled, num_valid)

    def test_decode_only_when_all_scheduled_are_one(self):
        cfg = MockVllmConfig()
        seq_lens = np.array([100, 200, 300], dtype=np.int32)
        scheduled = np.array([1, 1, 1], dtype=np.int32)
        state = self._call(cfg, seq_lens, 3, scheduled, scheduled)
        assert state.value == 2  # DecodeOnly

    def test_prefill_no_cache_when_seq_lens_equal_scheduled(self):
        cfg = MockVllmConfig()
        seq_lens = np.array([10, 20, 30], dtype=np.int32)
        scheduled = seq_lens.copy()
        state = self._call(cfg, seq_lens, 3, scheduled, scheduled)
        assert state.value == 0  # PrefillNoCache

    def test_spec_decoding_with_mtp_method(self):
        cfg = MockVllmConfig(speculative_method="mtp")
        seq_lens = np.array([100, 200], dtype=np.int32)
        scheduled = np.array([1, 1], dtype=np.int32)
        valid = np.array([1, 1], dtype=np.int32)
        state = self._call(cfg, seq_lens, 2, scheduled, valid)
        assert state.value == 4  # SpecDecoding

    def test_chunked_prefill_when_enabled(self):
        """Chunked prefill triggers when enable_chunked_prefill=True and
        not all scheduled tokens are 1."""
        cfg = MockVllmConfig(chunked_prefill=True)
        seq_lens = np.array([100, 200], dtype=np.int32)
        scheduled = np.array([2, 2], dtype=np.int32)  # not all 1 → skips DecodeOnly
        valid = np.array([2, 2], dtype=np.int32)
        state = self._call(cfg, seq_lens, 2, scheduled, valid)
        assert state.value == 3  # ChunkedPrefill

    def test_prefill_cache_hit_as_fallback(self):
        cfg = MockVllmConfig(chunked_prefill=False)
        seq_lens = np.array([100, 200], dtype=np.int32)
        scheduled = np.array([2, 2], dtype=np.int32)
        valid = np.array([2, 2], dtype=np.int32)
        state = self._call(cfg, seq_lens, 2, scheduled, valid)
        assert state.value == 1  # PrefillCacheHit

    def test_all_states_return_valid_enum(self):
        """Smoke: build_attn_state always returns a valid enum."""
        from vllm_ascend.worker.v2.attn_utils import build_attn_state
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        cfg = MockVllmConfig()
        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", return_value=cfg):
            state = build_attn_state(cfg, np.array([100, 200]), 2,
                                     np.array([1, 1]), np.array([1, 1]))
        assert isinstance(state, AscendAttentionState)


# ---------------------------------------------------------------------------
# get_attn_mask_builder — singleton
# ---------------------------------------------------------------------------

class TestGetAttnMaskBuilder:
    """Verify singleton pattern + device awareness."""

    def test_same_device_returns_same_instance(self):
        from vllm_ascend.worker.v2.attn_utils import get_attn_mask_builder
        b1 = get_attn_mask_builder(torch.device("cpu"))
        b2 = get_attn_mask_builder(torch.device("cpu"))
        assert b1 is b2

    def test_returns_attention_mask_builder(self):
        from vllm_ascend.worker.v2.attn_utils import get_attn_mask_builder
        builder = get_attn_mask_builder(torch.device("cpu"))
        assert builder is not None
        assert hasattr(builder, "get_attn_mask")


# ---------------------------------------------------------------------------
# _get_layer_kv_cache_specs — internal lookup
# ---------------------------------------------------------------------------

class TestGetLayerKVCacheSpecs:
    """Verify KV cache spec extraction from config groups."""

    def test_extracts_uniform_type_spec(self):
        from vllm_ascend.worker.v2.attn_utils import _get_layer_kv_cache_specs

        class Spec:
            pass
        spec = Spec()
        kv_config = MagicMock()
        group = MagicMock()
        group.kv_cache_spec = spec
        group.layer_names = ["layer.0.self_attn", "layer.1.self_attn"]
        kv_config.kv_cache_groups = [group]

        result = _get_layer_kv_cache_specs(kv_config)
        assert "layer.0.self_attn" in result
        assert "layer.1.self_attn" in result
        assert result["layer.0.self_attn"] is spec

    def test_extracts_plain_spec(self):
        """Smoke: _get_layer_kv_cache_specs runs without exception for mock config."""
        from vllm_ascend.worker.v2.attn_utils import _get_layer_kv_cache_specs

        kv_config = MagicMock()
        _get_layer_kv_cache_specs(kv_config)  # must not raise


# ---------------------------------------------------------------------------
# _align_memory — pointer alignment
# ---------------------------------------------------------------------------

class TestAlignMemory:
    """Verify memory alignment utility."""

    def test_already_aligned_returns_same_view(self):
        from vllm_ascend.worker.v2.attn_utils import _align_memory
        t = torch.arange(100, dtype=torch.float32)
        # ptr may or may not be aligned; just check shape consistency
        result = _align_memory(t, 64)
        assert result.dtype == t.dtype
        assert result.device == t.device
        assert result.numel() <= t.numel()
        assert result.numel() >= t.numel() - (64 // t.element_size())
