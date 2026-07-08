# SPDX-License-Identifier: Apache-2.0
"""Performance regression guards for speculative token tree pipeline.

Each test measures a single operation, compares against a 5× baseline
threshold (generous for SSH/NPU variance), and fails only on significant
regressions (>5× slowdown).

All thresholds are calibrated from observed remote NPU data.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import pytest
import torch


# ---------------------------------------------------------------------------
# Tree generators
# ---------------------------------------------------------------------------

def _linear_chain(n: int) -> str:
    return str([tuple(0 for _ in range(d)) for d in range(1, n + 1)])


def _binary_tree(depth: int) -> str:
    paths: list[tuple[int, ...]] = []
    for d in range(1, depth + 1):
        for leaf in range(2 ** d):
            paths.append(tuple((leaf >> (d - 1 - i)) & 1 for i in range(d)))
    return str(paths)


def _wide_shallow(n: int) -> str:
    return str([(i,) for i in range(n)])


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _bench(fn: Callable[[], object], iters: int = 200) -> float:
    """Return median execution time in seconds (warmup + measured)."""
    for _ in range(max(5, iters // 20)):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _check(label: str, median_us: float, limit_us: float, detail: str = ""):
    """Assert median is under limit; report actual value on failure."""
    assert median_us < limit_us, (
        f"{label}: {median_us:.0f}us > {limit_us:.0f}us{detail}"
    )


# ===================================================================
# Hot-path regression: bias construction (O(n²) CPU fill)
# -------------------------------------------------------------------
# Observed median (remote NPU, 500 iters):
#   n=4  ~12us    n=16 ~400us    n=64 ~5600us
# Limits: 5× observed to tolerate SSH variance, fail on true regressions.
# ===================================================================

BIAS_SIZES_REGRESSION = {4: 60, 16: 2000, 64: 30000}


@pytest.mark.parametrize("num_tokens,limit_us", BIAS_SIZES_REGRESSION.items())
def test_bias_construction_regression(num_tokens: int, limit_us: int):
    """Bias construction: O(n²) fill.  5× threshold over baseline."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    tree_str = _linear_chain(num_tokens)
    choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))

    median_us = _bench(lambda: prepare_speculative_token_tree_attn_bias(choices), iters=500) * 1e6
    _check(f"bias n={num_tokens}", median_us, limit_us,
           f" ({ (num_tokens+1)**2 } elements)")


# ===================================================================
# Hot-path regression: bias slicing (per-level tensor views)
# ===================================================================

SLICE_CASES = [
    # (name, tree_str, drafts, limit_us)
    ("uniform_1x8",    _linear_chain(8),   [1]*8,   500),
    ("binary_d3",      _binary_tree(3),    [2,4,8], 500),
    ("wide_16",        _wide_shallow(16),  [16],    500),
    ("long_chain_64",  _linear_chain(64),  [1]*64,  2000),
]


@pytest.mark.parametrize("name,tree_str,drafts,limit_us", SLICE_CASES)
def test_bias_slicing_regression(name: str, tree_str: str, drafts: list[int], limit_us: int):
    """Per-level bias slicing: tensor views + contiguous()."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))
    bias = prepare_speculative_token_tree_attn_bias(choices)

    def _slice():
        slices: list[torch.Tensor] = []
        offset = 0
        for qlen in drafts:
            if qlen <= 0:
                slices.append(bias.new_empty((0, 0)))
            else:
                start = 1 + offset
                end = start + qlen
                slices.append(bias[start:end, start:end].contiguous())
            offset += qlen
        return slices

    median_us = _bench(_slice, iters=500) * 1e6
    _check(f"slice {name}", median_us, limit_us,
           f" ({len(drafts)} levels, {sum(drafts)} tokens)")


# ===================================================================
# Cold-path regression: metadata attach (Config + bias + plan)
# This includes SpeculativeConfig creation overhead — inherently
# noisier, so use wider thresholds.
# ===================================================================

ATTACH_LIMITS = {4: 3000, 16: 15000, 64: 80000}


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
    def verify_with_parallel_config(self, p): pass


@dataclass
class MockParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1


def _make_config(tree_str: str, num_spec: int):
    from vllm.config.speculative import SpeculativeConfig
    return SpeculativeConfig(
        num_speculative_tokens=num_spec,
        speculative_token_tree=tree_str,
        method="ngram",
        target_model_config=MockModelConfig(),
        target_parallel_config=MockParallelConfig(),
    )


@pytest.mark.parametrize("num_tokens,limit_us", ATTACH_LIMITS.items())
def test_metadata_attach_regression(num_tokens: int, limit_us: int):
    """Full attach pipeline: Config → tree_choices → bias → plan → slices."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata

    tree_str = _linear_chain(num_tokens)

    def _attach():
        sc = _make_config(tree_str, num_tokens)
        class V:
            speculative_config = sc
            model_config = MockModelConfig()
        attach_speculative_tree_metadata(V())

    median_us = _bench(_attach, iters=200) * 1e6
    _check(f"attach n={num_tokens}", median_us, limit_us)


# ===================================================================
# Memory footprint: O(n²) float32 — exact element count + regression
# guard at 1.1× expected.  n=256 added as stress ceiling (258 KB).
# ===================================================================

MEMORY_SIZES = [4, 8, 16, 32, 64, 128, 256]
# (n+1)² × 4 bytes / 1024, rounded to 3 decimals
EXPECTED_KB = {
    4: 0.098, 8: 0.316, 16: 1.129, 32: 4.254,
    64: 16.504, 128: 65.004, 256: 258.004,
}


@pytest.mark.parametrize("num_tokens", MEMORY_SIZES)
def test_bias_memory_ok(num_tokens: int):
    """Bias tensor: O(n²) float32.  Regression guard at 1.1× expected KB."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    choices = sort_speculative_token_tree(
        parse_speculative_token_tree(_linear_chain(num_tokens))
    )
    bias = prepare_speculative_token_tree_attn_bias(choices)

    # Exact element count
    expected_elts = (num_tokens + 1) ** 2
    assert bias.numel() == expected_elts, (
        f"n={num_tokens}: {bias.numel()} elements != {expected_elts} expected"
    )

    # Footprint
    kb = bias.numel() * 4 / 1024
    expected_kb = EXPECTED_KB[num_tokens]

    # Tight check: exact to 0.01 KB (10 bytes)
    assert abs(kb - expected_kb) < 0.01, (
        f"n={num_tokens}: {kb:.3f}KB != {expected_kb:.3f}KB (off by {abs(kb-expected_kb)*1000:.0f} bytes)"
    )

    # Regression guard: no more than 1.1× expected (catches dtype change, duplicate alloc)
    assert kb <= expected_kb * 1.1, (
        f"n={num_tokens}: {kb:.1f}KB exceeds 1.1× expected {expected_kb*1.1:.1f}KB"
    )


# ===================================================================
# Rapid-fire: 50 small trees under 1s (wide regression guard)
# ===================================================================

def test_rapid_fire_50_trees():
    """50 random small trees (n=2..12) must complete under 1 second."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    import random
    rng = random.Random(42)

    t0 = time.perf_counter()
    for _ in range(50):
        n = rng.randint(2, 12)
        tree_str = _linear_chain(n)
        choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))
        bias = prepare_speculative_token_tree_attn_bias(choices)
        assert bias.shape == (n + 1, n + 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _check("rapid-fire 50 trees", elapsed_ms * 1000, 1_000_000,
           f" ({elapsed_ms:.0f}ms)")


# ===================================================================
# Builder hot-path (NPU only)  — needs torch_npu
# ===================================================================

_npu = False
try:
    import torch_npu  # noqa: F401
    _npu = True
except ImportError:
    pass


@pytest.mark.skipif(not _npu, reason="requires NPU device")
@pytest.mark.parametrize("num_tokens,limit_us", [(4, 2000), (8, 3000)])
def test_build_for_drafting_regression(num_tokens: int, limit_us: int):
    """Per-level build_for_drafting latency (n≤8 to stay in TND limit)."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendCommonAttentionMetadata,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    tree_str = _linear_chain(num_tokens)
    sc = _make_config(tree_str, num_tokens)

    @dataclass
    class MS: enable_chunked_prefill: bool = False
    @dataclass
    class MC: capture_model_init_state: bool = False
    class V:
        speculative_config = sc
        model_config = MockModelConfig()
        scheduler_config = MS()
        compilation_config = MC()
    attach_speculative_tree_metadata(V())

    builder = AscendAttentionMetadataBuilder(
        kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=16, head_size=128, dtype=torch.float16),
        layer_names=["layer.0.self_attn"],
        vllm_config=V(),
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

    median_us = _bench(lambda: builder.build_for_drafting(1, common), iters=200) * 1e6
    _check(f"build_for_drafting n={num_tokens}", median_us, limit_us)


# ===================================================================
# Large-tree topology benchmarks
# ===================================================================

LARGE_TOPO_BENCH = [
    # (label, tree_str, num_tokens, limit_us)
    ("binary_d3",  _binary_tree(3),    14, 10000),
    ("binary_d4",  _binary_tree(4),    30, 40000),
    ("wide_32",    _wide_shallow(32),  32, 20000),
    ("linear_64",  _linear_chain(64),  64, 30000),
]


@pytest.mark.parametrize("label,tree_str,num_tokens,limit_us", LARGE_TOPO_BENCH)
def test_bias_construction_large_topologies(label, tree_str, num_tokens, limit_us):
    """Bias construction for large/complex tree topologies (not just linear chains)."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))
    assert len(choices) == num_tokens, f"{label}: expected {num_tokens}, got {len(choices)}"

    median_us = _bench(lambda: prepare_speculative_token_tree_attn_bias(choices), iters=300) * 1e6
    _check(f"bias {label}", median_us, limit_us,
           f" ({num_tokens} tokens, {(num_tokens+1)**2} elements)")


@pytest.mark.parametrize("label,tree_str,num_tokens,limit_us", LARGE_TOPO_BENCH)
def test_attach_metadata_large_topologies(label, tree_str, num_tokens, limit_us):
    """Metadata attach for large/complex tree topologies."""
    from vllm_ascend.spec_decode.speculative_token_tree import attach_speculative_tree_metadata

    def _attach():
        sc = _make_config(tree_str, num_tokens)
        class V:
            speculative_config = sc
            model_config = MockModelConfig()
        attach_speculative_tree_metadata(V())

    median_us = _bench(_attach, iters=200) * 1e6
    _check(f"attach {label}", median_us, limit_us,
           f" ({num_tokens} tokens)")


# ===================================================================
# Rapid-fire: mixed topologies
# ===================================================================

def test_rapid_fire_50_mixed_topologies():
    """50 random trees of mixed topologies (linear, binary, wide) under 2s."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    import random
    rng = random.Random(42)

    t0 = time.perf_counter()
    for _ in range(50):
        kind = rng.choice(["linear", "binary", "wide"])
        if kind == "linear":
            n = rng.randint(2, 16)
            tree_str = _linear_chain(n)
        elif kind == "binary":
            d = rng.randint(1, 3)
            tree_str = _binary_tree(d)
            n = 2**(d+1) - 2
        else:  # wide
            n = rng.randint(2, 20)
            tree_str = _wide_shallow(n)
        choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))
        bias = prepare_speculative_token_tree_attn_bias(choices)
        assert bias.shape == (n + 1, n + 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _check("rapid-fire 50 mixed", elapsed_ms * 1000, 2_000_000,
           f" ({elapsed_ms:.0f}ms)")


# ===================================================================
# Slicing benchmark for large asymmetric tree
# ===================================================================

def test_slicing_asymmetric_large_tree():
    """Per-level bias slicing for an asymmetric depth-6 tree (10 tokens)."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    tree_str = str([(0,), (0,0), (0,1), (0,0,0), (0,0,1),
                    (0,0,0,0), (0,0,0,1), (0,0,0,0,0),
                    (0,0,0,0,1), (0,0,0,0,0,0)])
    choices = sort_speculative_token_tree(parse_speculative_token_tree(tree_str))
    bias = prepare_speculative_token_tree_attn_bias(choices)

    drafts_per_level = [1, 2, 2, 2, 2, 1]
    def _slice():
        slices: list[torch.Tensor] = []
        offset = 0
        for qlen in drafts_per_level:
            if qlen <= 0:
                slices.append(bias.new_empty((0, 0)))
            else:
                start = 1 + offset
                end = start + qlen
                slices.append(bias[start:end, start:end].contiguous())
            offset += qlen
        return slices

    median_us = _bench(_slice, iters=500) * 1e6
    _check("slice asymmetric_d6", median_us, 1000,
           f" ({len(drafts_per_level)} levels)")
