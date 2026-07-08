# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for speculative token tree attention pipeline.

Exercises the Config → Tree metadata → Attention builder → build_for_drafting
chain without a model load.  Tests the builder-level integration — tree bias
slicing, edge-case handling, and hash stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers: minimal config objects that pass __post_init__ validation
# ---------------------------------------------------------------------------

@dataclass
class MockModelConfig:
    max_model_len: int = 1024
    runner_type: str = "generate"
    model: str = "gpt2"
    tokenizer: str = "gpt2"
    hf_config: object = None
    hf_text_config: object = None
    quantization: str | None = None
    skip_tokenizer_init: bool = True

    def verify_with_parallel_config(self, parallel_config):
        pass

@dataclass
class MockSchedulerConfig:
    enable_chunked_prefill: bool = False

@dataclass
class MockCompilationConfig:
    capture_model_init_state: bool = False


class MockSpeculativeConfig:
    def __init__(self, tree_str, num_spec):
        self.num_speculative_tokens = num_spec
        self.speculative_token_tree = tree_str
        self.parallel_drafting = False

    def compute_hash(self):
        return hash((self.num_speculative_tokens, self.speculative_token_tree))


def _create_speculative_config(tree_str, num_spec):
    return MockSpeculativeConfig(tree_str, num_spec)


def _make_vllm_config(tree_str, num_spec):
    spec_cfg = _create_speculative_config(tree_str, num_spec)
    class VllmConfig:
        speculative_config = spec_cfg
        model_config = MockModelConfig()
        scheduler_config = MockSchedulerConfig()
        compilation_config = MockCompilationConfig()
    return VllmConfig()


# ---------------------------------------------------------------------------
# Tree strategies for parametrized tests
# ---------------------------------------------------------------------------
TREE_STRATEGIES = [
    ("linear_chain_3", "[(0,), (0,0), (0,0,0)]", 3, 4, 3, (1, 1, 1)),
    ("binary_2deep",  "[(0,), (0,0), (0,1)]",       3, 4, 2, (1, 2)),
    ("ternary",       "[(0,), (0,0), (0,1), (0,2)]", 4, 5, 2, (1, 3)),
]


# ---------------------------------------------------------------------------
# NPU availability check
# ---------------------------------------------------------------------------
_npu_available = False
try:
    import torch_npu  # noqa: F401
    _npu_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Builder integration — core smoke
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _npu_available, reason="requires NPU device")
@pytest.mark.parametrize("name,tree_str,num_spec,exp_len,exp_depth,exp_drafts", TREE_STRATEGIES)
def test_builder_build_for_drafting_per_level(name, tree_str, num_spec, exp_len, exp_depth, exp_drafts):
    """build_for_drafting slices tree bias correctly for each draft level."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendCommonAttentionMetadata,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    vllm_cfg = _make_vllm_config(tree_str, num_spec)
    attach_speculative_tree_metadata(vllm_cfg)

    kv_cache_spec = FullAttentionSpec(block_size=128, num_kv_heads=16, head_size=128, dtype=torch.float16)
    builder = AscendAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layer.0.self_attn"],
        vllm_config=vllm_cfg,
        device=torch.device("npu:0"),
    )

    # draft_index=0 → prefill, no tree mask
    common = AscendCommonAttentionMetadata(
        num_reqs=1, num_actual_tokens=num_spec,
        query_start_loc=torch.tensor([0, num_spec], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, num_spec], dtype=torch.int32),
        max_seq_len=num_spec,
        block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_spec], dtype=torch.int32),
        seq_lens=torch.tensor([num_spec], dtype=torch.int32),
        slot_mapping=torch.zeros((num_spec,), dtype=torch.int64),
        max_query_len=num_spec, causal=True,
    )
    meta_prefill = builder.build_for_drafting(0, common)
    assert not meta_prefill.is_tree_mask

    # draft_index=1..depth → each level has tree mask with correct shape
    for level_idx in range(1, exp_depth):
        level_tokens = exp_drafts[level_idx - 1]
        common_lvl = AscendCommonAttentionMetadata(
            num_reqs=1, num_actual_tokens=level_tokens,
            query_start_loc=torch.tensor([0, level_tokens], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, level_tokens], dtype=torch.int32),
            max_seq_len=level_tokens,
            block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
            seq_lens_cpu=torch.tensor([level_tokens], dtype=torch.int32),
            seq_lens=torch.tensor([level_tokens], dtype=torch.int32),
            slot_mapping=torch.zeros((level_tokens,), dtype=torch.int64),
            max_query_len=level_tokens, causal=True,
        )
        meta = builder.build_for_drafting(level_idx, common_lvl)
        assert meta.is_tree_mask
        assert meta.attn_mask.shape == (level_tokens, level_tokens)
        if level_tokens > 1:
            assert torch.isneginf(meta.attn_mask[0, 1]), "siblings must be masked"


# ---------------------------------------------------------------------------
# Builder — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _npu_available, reason="requires NPU device")
def test_builder_build_for_drafting_index_beyond_depth():
    """draft_index past tree depth returns metadata without tree mask."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendCommonAttentionMetadata,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    vllm_cfg = _make_vllm_config("[(0,), (0,0), (0,1)]", 3)
    attach_speculative_tree_metadata(vllm_cfg)

    builder = AscendAttentionMetadataBuilder(
        kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=16, head_size=128, dtype=torch.float16),
        layer_names=["layer.0.self_attn"],
        vllm_config=vllm_cfg,
        device=torch.device("npu:0"),
    )
    common = AscendCommonAttentionMetadata(
        num_reqs=1, num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        slot_mapping=torch.zeros((1,), dtype=torch.int64),
        max_query_len=1, causal=True,
    )
    meta = builder.build_for_drafting(99, common)
    assert not meta.is_tree_mask


@pytest.mark.skipif(not _npu_available, reason="requires NPU device")
def test_builder_build_for_drafting_prefill_no_tree():
    """draft_index=0 is always prefill — no tree mask regardless of config."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendCommonAttentionMetadata,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    vllm_cfg = _make_vllm_config("[(0,), (0,0), (0,1)]", 3)
    attach_speculative_tree_metadata(vllm_cfg)

    builder = AscendAttentionMetadataBuilder(
        kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=16, head_size=128, dtype=torch.float16),
        layer_names=["layer.0.self_attn"],
        vllm_config=vllm_cfg,
        device=torch.device("npu:0"),
    )
    common = AscendCommonAttentionMetadata(
        num_reqs=1, num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        max_seq_len=3,
        block_table_tensor=torch.zeros((1, 64), dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        slot_mapping=torch.zeros((3,), dtype=torch.int64),
        max_query_len=3, causal=True,
    )
    meta = builder.build_for_drafting(0, common)
    assert not meta.is_tree_mask


# ---------------------------------------------------------------------------
# Hash stability
# ---------------------------------------------------------------------------

def test_compute_hash_incorporates_tree():
    """SpeculativeConfig.compute_hash changes when tree differs."""
    cfg_a = _create_speculative_config("[(0,), (0,0), (0,0,0)]", 3)
    cfg_b = _create_speculative_config("[(0,), (0,0), (0,1)]", 3)
    assert cfg_a.compute_hash() != cfg_b.compute_hash(), (
        "different trees must produce different hashes"
    )
