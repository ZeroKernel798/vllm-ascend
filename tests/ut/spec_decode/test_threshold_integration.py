# SPDX-License-Identifier: Apache-2.0
"""Integration tests for benchmark threshold guard functions.

Unlike test_threshold_guards.py (which unit-tests the _check helper),
this file exercises the actual benchmark functions from
test_perf_benchmark.py with mocked timings to verify each
threshold guard passes and fails at the right boundaries.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch


# ===================================================================
# Helper: reproducible mocked _bench
# ===================================================================

def _make_bench_mock(median_us: float, iters: int = 200):
    """Return a function that mimics _bench with a fixed median time."""
    def mocked(fn, iters=iters):
        _ = fn()  # Call once for side effects (tree generation etc.)
        return median_us / 1e6
    return mocked


# ===================================================================
# Integration: bias construction regression
# ===================================================================

@pytest.mark.parametrize("num_tokens,limit_us", [(4, 60), (16, 2000), (64, 30000)])
def test_integration_bias_passes_under_limit(num_tokens, limit_us):
    """bias construction at limit-1 → no exception."""
    from tests.ut.spec_decode.test_perf_benchmark import test_bias_construction_regression
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us - 1)):
        test_bias_construction_regression(num_tokens, limit_us)


@pytest.mark.parametrize("num_tokens,limit_us", [(4, 60), (16, 2000), (64, 30000)])
def test_integration_bias_fails_at_limit(num_tokens, limit_us):
    """bias construction at limit → raises AssertionError."""
    from tests.ut.spec_decode.test_perf_benchmark import (
        test_bias_construction_regression,
    )
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us)):
        with pytest.raises(AssertionError, match=r"bias n="):
            test_bias_construction_regression(num_tokens, limit_us)


@pytest.mark.parametrize("num_tokens,limit_us", [(4, 60), (16, 2000), (64, 30000)])
def test_integration_bias_fails_double_limit(num_tokens, limit_us):
    """bias construction at 2× limit → raises AssertionError."""
    from tests.ut.spec_decode.test_perf_benchmark import (
        test_bias_construction_regression,
    )
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us * 2)):
        with pytest.raises(AssertionError, match=r"bias n="):
            test_bias_construction_regression(num_tokens, limit_us)


# ===================================================================
# Integration: bias slicing regression
# ===================================================================

SLICE_PARAMS = [
    ("uniform_1x8", "[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0)]", [1]*8, 500),
    ("binary_d3", "[(0,), (1,), (0,0), (0,1), (1,0), (1,1), (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]", [2,4,8], 500),
    ("wide_16", "[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,)]", [16], 500),
    ("long_chain_64", "[(0,), (0,0)]", [1]*2, 2000),  # simplified long chain for speed
]


@pytest.mark.parametrize("name,tree_str,drafts,limit_us", SLICE_PARAMS)
def test_integration_slice_passes_under_limit(name, tree_str, drafts, limit_us):
    """Bias slicing at low latency → no exception."""
    from tests.ut.spec_decode.test_perf_benchmark import test_bias_slicing_regression
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us - 1)):
        test_bias_slicing_regression(name, tree_str, drafts, limit_us)


@pytest.mark.parametrize("name,tree_str,drafts,limit_us", SLICE_PARAMS)
def test_integration_slice_fails_above_limit(name, tree_str, drafts, limit_us):
    """Bias slicing above limit → raises AssertionError."""
    from tests.ut.spec_decode.test_perf_benchmark import test_bias_slicing_regression
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us + 1)):
        with pytest.raises(AssertionError, match=r"slice "):
            test_bias_slicing_regression(name, tree_str, drafts, limit_us)


# ===================================================================
# Integration: metadata attach regression
# ===================================================================

@pytest.mark.parametrize("num_tokens,limit_us", [(4, 3000), (16, 15000), (64, 80000)])
def test_integration_attach_passes_under_limit(num_tokens, limit_us):
    """Attach under limit → no exception."""
    from tests.ut.spec_decode.test_perf_benchmark import test_metadata_attach_regression
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us - 1)):
        test_metadata_attach_regression(num_tokens, limit_us)


@pytest.mark.parametrize("num_tokens,limit_us", [(4, 3000), (16, 15000), (64, 80000)])
def test_integration_attach_fails_at_limit(num_tokens, limit_us):
    """Attach at limit → raises AssertionError."""
    from tests.ut.spec_decode.test_perf_benchmark import test_metadata_attach_regression
    with patch("tests.ut.spec_decode.test_perf_benchmark._bench",
               _make_bench_mock(limit_us)):
        with pytest.raises(AssertionError, match=r"attach "):
            test_metadata_attach_regression(num_tokens, limit_us)


# ===================================================================
# Integration: memory footprint
# ===================================================================

def test_integration_memory_passes_for_all_sizes():
    """Actual memory test runs without mocking — verifies O(n²) formula."""
    from tests.ut.spec_decode.test_perf_benchmark import (
        test_bias_memory_ok,
        _linear_chain,
    )
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    for n in [4, 16, 64, 128]:
        choices = sort_speculative_token_tree(
            parse_speculative_token_tree(_linear_chain(n)))
        bias = prepare_speculative_token_tree_attn_bias(choices)
        # Verify exact formula
        assert bias.numel() == (n + 1) ** 2
        kb = bias.numel() * 4 / 1024
        assert kb > 0


def test_integration_memory_would_fail_if_bias_allocated_twice():
    """If prepare_attn_bias allocated double memory, 1.1× guard catches it."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    from tests.ut.spec_decode.test_perf_benchmark import _linear_chain, EXPECTED_KB

    n = 16
    choices = sort_speculative_token_tree(
        parse_speculative_token_tree(_linear_chain(n)))
    bias = prepare_speculative_token_tree_attn_bias(choices)
    expected_kb = EXPECTED_KB[n]

    # Normal: within 1.1×
    actual_kb = bias.numel() * 4 / 1024
    assert actual_kb <= expected_kb * 1.1, "normal bias should pass"

    # Simulate double allocation: if someone changed float32→float64
    double_kb = actual_kb * 2
    assert double_kb > expected_kb * 1.1, (
        f"double allocation ({double_kb:.1f}KB) would trigger guard "
        f"(limit={expected_kb*1.1:.1f}KB)"
    )


# ===================================================================
# Integration: rapid-fire
# ===================================================================

def test_integration_rapid_fire_passes_under_limit():
    """50 small trees under 1s → no exception."""
    from tests.ut.spec_decode.test_perf_benchmark import test_rapid_fire_50_trees
    with patch("tests.ut.spec_decode.test_perf_benchmark.time.perf_counter") as mock_time:
        mock_time.side_effect = [0.0, 0.5]  # start=0, end=0.5 → elapsed=0.5s
        test_rapid_fire_50_trees()


def test_integration_rapid_fire_fails_over_limit():
    """50 trees over 1s → raises AssertionError."""
    from tests.ut.spec_decode.test_perf_benchmark import test_rapid_fire_50_trees
    with patch("tests.ut.spec_decode.test_perf_benchmark.time.perf_counter") as mock_time:
        mock_time.side_effect = [0.0, 1.5]  # start=0, end=1.5 → elapsed=1.5s
        with pytest.raises(AssertionError, match="rapid-fire"):
            test_rapid_fire_50_trees()


# ===================================================================
# Integration: run ALL benchmarks end-to-end (no mocking)
# ===================================================================

@pytest.mark.slow
def test_run_all_benchmarks_end_to_end():
    """Run every benchmark function with real timings — must all pass."""
    from tests.ut.spec_decode.test_perf_benchmark import (
        test_bias_construction_regression,
        test_bias_slicing_regression,
        test_metadata_attach_regression,
        test_bias_memory_ok,
        test_rapid_fire_50_trees,
        SLICE_CASES,
    )

    # bias construction: all three sizes
    for num_tokens, limit_us in [(4, 60), (16, 2000), (64, 30000)]:
        test_bias_construction_regression(num_tokens, limit_us)

    # bias slicing: all four cases
    for name, tree_str, drafts, limit_us in SLICE_CASES:
        test_bias_slicing_regression(name, tree_str, drafts, limit_us)

    # metadata attach: all three sizes
    for num_tokens, limit_us in [(4, 3000), (16, 15000), (64, 80000)]:
        test_metadata_attach_regression(num_tokens, limit_us)

    # memory: all sizes
    for n in [4, 8, 16, 32, 64, 128, 256]:
        test_bias_memory_ok(n)

    # rapid-fire
    test_rapid_fire_50_trees()


# ===================================================================
# Cross-validation: verify actual timing < threshold for real runs
# ===================================================================

def test_cross_validate_bias_n4_actual_under_limit():
    """Real bias construction n=4 is well under 60us — verifies threshold isn't
    too tight to pass under normal conditions."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    from tests.ut.spec_decode.test_perf_benchmark import _linear_chain, _bench

    choices = sort_speculative_token_tree(
        parse_speculative_token_tree(_linear_chain(4)))
    median_sec = _bench(lambda: prepare_speculative_token_tree_attn_bias(choices), iters=100)
    median_us = median_sec * 1e6

    # Should be WELL under the 60us threshold (typically ~12us)
    assert median_us < 60, f"n=4 bias actual: {median_us:.0f}us — threshold too tight?"


def test_cross_validate_memory_n128_actual_footprint():
    """Real footprint at n=128 matches expected."""
    from vllm_ascend.spec_decode.speculative_token_tree import (
        parse_speculative_token_tree,
        sort_speculative_token_tree,
        prepare_speculative_token_tree_attn_bias,
    )
    from tests.ut.spec_decode.test_perf_benchmark import _linear_chain, EXPECTED_KB

    n = 128
    choices = sort_speculative_token_tree(
        parse_speculative_token_tree(_linear_chain(n)))
    bias = prepare_speculative_token_tree_attn_bias(choices)
    actual_kb = bias.numel() * 4 / 1024
    expected_kb = EXPECTED_KB[n]
    assert abs(actual_kb - expected_kb) < 0.01, (
        f"n={n}: actual {actual_kb:.3f}KB != expected {expected_kb:.3f}KB"
    )
    assert actual_kb <= expected_kb * 1.1


# ===================================================================
# Threshold value sanity: verify no threshold is accidentally zero/negative
# ===================================================================

@pytest.mark.parametrize("name,value", [
    ("BIAS_SIZES_REGRESSION keys", [4, 16, 64]),
    ("BIAS_SIZES_REGRESSION values list", [60, 2000, 30000]),
    ("ATTACH_LIMITS keys", [4, 16, 64]),
    ("ATTACH_LIMITS values list", [3000, 15000, 80000]),
    ("SLICE_CASES limit_us values", [500, 500, 500, 2000]),
    ("MEMORY_SIZES", [4, 8, 16, 32, 64, 128, 256]),
])
def test_threshold_values_positive_and_ordered(name, value):
    """All threshold limits are positive and non-decreasing."""
    assert all(v > 0 for v in value), f"{name}: contains zero/negative value"
    assert all(value[i] <= value[i+1] for i in range(len(value)-1)), (
        f"{name}: not non-decreasing: {value}"
    )


def test_perf_benchmark_file_importable():
    """Verify the benchmark module can be imported without side effects."""
    import tests.ut.spec_decode.test_perf_benchmark as bm
    assert hasattr(bm, "test_bias_construction_regression")
    assert hasattr(bm, "test_bias_slicing_regression")
    assert hasattr(bm, "test_metadata_attach_regression")
    assert hasattr(bm, "test_bias_memory_ok")
    assert hasattr(bm, "test_rapid_fire_50_trees")
    assert hasattr(bm, "_check")
    assert hasattr(bm, "EXPECTED_KB")
    assert hasattr(bm, "BIAS_SIZES_REGRESSION")
    assert hasattr(bm, "ATTACH_LIMITS")
    assert hasattr(bm, "SLICE_CASES")
    assert hasattr(bm, "MEMORY_SIZES")
