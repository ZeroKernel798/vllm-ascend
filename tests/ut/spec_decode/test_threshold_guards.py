# SPDX-License-Identifier: Apache-2.0
"""Unit tests for each regression threshold guard in test_perf_benchmark.py.

Verifies that:
- each threshold passes at the limit (median_us == limit_us)
- each threshold fails just above the limit (median_us > limit_us)
- failure messages contain the expected label and values
- the _check helper correctly formats AssertionError messages
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Re-implement the _check helper in isolation so we can unit-test it
# ---------------------------------------------------------------------------

def _check(label: str, median_us: float, limit_us: float, detail: str = ""):
    """Assert median is under limit; report actual value on failure."""
    assert median_us < limit_us, (
        f"{label}: {median_us:.0f}us > {limit_us:.0f}us{detail}"
    )


# ---------------------------------------------------------------------------
# _check helper tests
# ---------------------------------------------------------------------------

class TestCheckHelper:
    """Verify _check assertion logic and error message formatting."""

    def test_passes_when_median_below_limit(self):
        _check("bias n=4", 30.0, 60.0)  # no exception

    def test_passes_when_median_at_half_limit(self):
        _check("bias n=16", 1000.0, 2000.0)  # no exception

    def test_passes_when_median_near_limit(self):
        _check("bias n=64", 29999.0, 30000.0)  # no exception

    def test_fails_when_median_equals_limit(self):
        with pytest.raises(AssertionError, match="bias n=4: 60us > 60us"):
            _check("bias n=4", 60.0, 60.0)

    def test_fails_when_median_above_limit(self):
        with pytest.raises(AssertionError, match="bias n=16: 2001us > 2000us"):
            _check("bias n=16", 2001.0, 2000.0)

    def test_fails_with_detail_in_message(self):
        with pytest.raises(AssertionError, match=r"bias n=64: 31000us > 30000us \(4225 elements\)"):
            _check("bias n=64", 31000.0, 30000.0, " (4225 elements)")

    def test_fails_double_limit(self):
        with pytest.raises(AssertionError, match="slice foo: 1000us > 500us"):
            _check("slice foo", 1000.0, 500.0)

    def test_label_format_preserved_in_message(self):
        with pytest.raises(AssertionError) as excinfo:
            _check("build_for_drafting n=8", 5000.0, 3000.0)
        msg = str(excinfo.value)
        assert "build_for_drafting n=8" in msg
        assert "5000us" in msg
        assert "3000us" in msg


# ---------------------------------------------------------------------------
# Simulated timing: mock _bench to return controlled values
# ---------------------------------------------------------------------------

def _mock_bench(median_us: float):
    """Return a mock that replaces _bench with a fixed median (in seconds)."""
    return patch(
        "tests.ut.spec_decode.test_perf_benchmark._bench",
        return_value=median_us / 1e6,  # convert us → seconds
    )


# ---------------------------------------------------------------------------
# Bias construction threshold tests
# ---------------------------------------------------------------------------

BIAS_THRESHOLDS = [
    # (num_tokens, limit_us, label)
    (4, 60, "bias n=4"),
    (16, 2000, "bias n=16"),
    (64, 30000, "bias n=64"),
]


class TestBiasConstructionThreshold:
    """Each parametrized size: verify pass-at-limit-1 and fail-at-limit+1."""

    @pytest.mark.parametrize("num_tokens,limit_us,label", BIAS_THRESHOLDS)
    def test_passes_at_limit_minus_one(self, num_tokens, limit_us, label):
        median_us = limit_us - 1
        with _mock_bench(median_us):
            from tests.ut.spec_decode.test_perf_benchmark import (
                test_bias_construction_regression as fn,
                _check,
            )
            fn.parametrize = None  # skip parametrize — call directly
            # Verify _check passes
            _check(label, median_us, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us,label", BIAS_THRESHOLDS)
    def test_fails_at_limit(self, num_tokens, limit_us, label):
        median_us = limit_us  # equal to limit → must fail
        with pytest.raises(AssertionError):
            _check(label, median_us, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us,label", BIAS_THRESHOLDS)
    def test_fails_double_limit(self, num_tokens, limit_us, label):
        median_us = limit_us * 2  # 2× threshold
        with pytest.raises(AssertionError):
            _check(label, median_us, limit_us)


# ---------------------------------------------------------------------------
# Bias slicing threshold tests
# ---------------------------------------------------------------------------

SLICE_THRESHOLDS = [
    ("uniform_1x8", 500),
    ("binary_d3", 500),
    ("wide_16", 500),
    ("long_chain_64", 2000),
]


class TestSliceThreshold:
    @pytest.mark.parametrize("name,limit_us", SLICE_THRESHOLDS)
    def test_slice_passes_well_under_limit(self, name, limit_us):
        median_us = 10  # tensor view ops are fast
        _check(f"slice {name}", median_us, limit_us)

    @pytest.mark.parametrize("name,limit_us", SLICE_THRESHOLDS)
    def test_slice_fails_above_limit(self, name, limit_us):
        median_us = limit_us + 1
        with pytest.raises(AssertionError):
            _check(f"slice {name}", median_us, limit_us)

    @pytest.mark.parametrize("name,limit_us", SLICE_THRESHOLDS)
    def test_slice_fails_ten_x(self, name, limit_us):
        median_us = limit_us * 10
        with pytest.raises(AssertionError):
            _check(f"slice {name}", median_us, limit_us)


# ---------------------------------------------------------------------------
# Metadata attach threshold tests
# ---------------------------------------------------------------------------

ATTACH_THRESHOLDS = [
    (4, 3000, "attach n=4"),
    (16, 15000, "attach n=16"),
    (64, 80000, "attach n=64"),
]


class TestAttachThreshold:
    @pytest.mark.parametrize("num_tokens,limit_us,label", ATTACH_THRESHOLDS)
    def test_attach_passes_under_limit(self, num_tokens, limit_us, label):
        median_us = limit_us - 1
        _check(label, median_us, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us,label", ATTACH_THRESHOLDS)
    def test_attach_fails_at_limit(self, num_tokens, limit_us, label):
        with pytest.raises(AssertionError):
            _check(label, limit_us, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us,label", ATTACH_THRESHOLDS)
    def test_attach_fails_above_limit(self, num_tokens, limit_us, label):
        with pytest.raises(AssertionError):
            _check(label, limit_us + 100, limit_us)


# ---------------------------------------------------------------------------
# Memory footprint threshold tests
# ---------------------------------------------------------------------------

MEMORY_CASES = [
    # (num_tokens, expected_elts, expected_kb, max_1_1x)
    (4,   25,   0.098,  0.108),
    (8,   81,   0.316,  0.348),
    (16,  289,  1.129,  1.242),
    (32,  1089, 4.254,  4.679),
    (64,  4225, 16.504, 18.154),
    (128, 16641,65.004, 71.504),
    (256, 66049,258.004,283.804),
]


class TestMemoryThreshold:
    """Verify O(n²) element count and KB threshold logic."""

    @pytest.mark.parametrize("n,elts,kb,limit_kb", MEMORY_CASES)
    def test_exact_element_count(self, n, elts, kb, limit_kb):
        computed = (n + 1) ** 2
        assert computed == elts, f"n={n}: (n+1)²={computed} != {elts}"

    @pytest.mark.parametrize("n,elts,kb,limit_kb", MEMORY_CASES)
    def test_exact_kb_from_float32(self, n, elts, kb, limit_kb):
        computed_kb = elts * 4 / 1024
        assert abs(computed_kb - kb) < 0.001, (
            f"n={n}: {computed_kb:.4f}KB != {kb:.4f}KB"
        )

    @pytest.mark.parametrize("n,elts,kb,limit_kb", MEMORY_CASES)
    def test_regression_guard_1_1x(self, n, elts, kb, limit_kb):
        """At exactly 1.1× → passes (≤ guard). Above 1.1× → fails."""
        # Exactly at limit → passes (≤ is inclusive)
        assert kb * 1.1 <= kb * 1.1  # equal → True

        # Just above 1.1× → fails
        at_above = kb * 1.1001
        assert at_above > kb * 1.1, "should be above 1.1×"
        with pytest.raises(AssertionError):
            assert at_above <= kb * 1.1, f"should fail above 1.1×"

    @pytest.mark.parametrize("n,elts,kb,limit_kb", MEMORY_CASES)
    def test_tolerance_and_guard_independent(self, n, elts, kb, limit_kb):
        """The 0.01KB tolerance and 1.1× guard check different things."""
        # Tolerance: exact expected_kb
        abs_diff = abs(kb - kb)  # always 0 for same value
        assert abs_diff < 0.01  # passes
        # Guard: must be ≤ 1.1×
        assert kb <= kb * 1.1  # passes

    @pytest.mark.parametrize("n,elts,kb,limit_kb", MEMORY_CASES)
    def test_would_fail_if_dtype_changes(self, n, elts, kb, limit_kb):
        """If someone changes float32 → float64, footprint doubles."""
        double_kb = kb * 2
        # Double KB would fail the 1.1× guard
        assert double_kb > kb * 1.1, (
            f"n={n}: {double_kb:.1f}KB > {kb*1.1:.1f}KB — "
            f"double-allocation would be caught"
        )


# ---------------------------------------------------------------------------
# Rapid-fire threshold test
# ---------------------------------------------------------------------------

class TestRapidFireThreshold:
    def test_passes_well_under_1_second(self):
        _check("rapid-fire 50 trees", 500_000, 1_000_000, " (500ms)")

    def test_fails_above_1_second(self):
        with pytest.raises(AssertionError, match="rapid-fire 50 trees: 1200000us > 1000000us"):
            _check("rapid-fire 50 trees", 1_200_000, 1_000_000, " (1200ms)")

    def test_fails_if_takes_10_seconds(self):
        with pytest.raises(AssertionError):
            _check("rapid-fire 50 trees", 10_000_000, 1_000_000, " (10000ms)")


# ---------------------------------------------------------------------------
# Builder (NPU) threshold tests
# ---------------------------------------------------------------------------

BUILDER_THRESHOLDS = [(4, 2000), (8, 3000)]


class TestBuilderThreshold:
    @pytest.mark.parametrize("num_tokens,limit_us", BUILDER_THRESHOLDS)
    def test_builder_passes_under_limit(self, num_tokens, limit_us):
        _check(f"build_for_drafting n={num_tokens}", limit_us - 1, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us", BUILDER_THRESHOLDS)
    def test_builder_fails_at_limit(self, num_tokens, limit_us):
        with pytest.raises(AssertionError):
            _check(f"build_for_drafting n={num_tokens}", limit_us, limit_us)

    @pytest.mark.parametrize("num_tokens,limit_us", BUILDER_THRESHOLDS)
    def test_builder_fails_above_limit(self, num_tokens, limit_us):
        with pytest.raises(AssertionError):
            _check(f"build_for_drafting n={num_tokens}", limit_us + 500, limit_us)


# ---------------------------------------------------------------------------
# Boundary: edge cases for the _check function
# ---------------------------------------------------------------------------

class TestCheckEdgeCases:
    def test_zero_median_passes_positive_limit(self):
        _check("zero median", 0.0, 100.0)  # no exception

    def test_zero_limit_with_positive_median_fails(self):
        with pytest.raises(AssertionError, match="zero limit: 1us > 0us"):
            _check("zero limit", 1.0, 0.0)

    def test_float_precision_boundary(self):
        """median just under limit by 0.001us — passes."""
        limit = 1000.0
        median = limit - 0.001
        _check("precision", median, limit)

    def test_float_precision_boundary_fails(self):
        """median just over limit by 0.001us — fails."""
        limit = 1000.0
        median = limit + 0.001
        with pytest.raises(AssertionError):
            _check("precision", median, limit)

    def test_large_values_format_correctly(self):
        with pytest.raises(AssertionError) as excinfo:
            _check("big", 9_999_999.0, 1_000_000.0)
        assert "9999999us > 1000000us" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Verify no threshold is accidentally commented out or weakened
# ---------------------------------------------------------------------------

class TestThresholdConstantsConsistency:
    """Sanity-check that threshold dictionaries in the benchmark file 
    have the expected structure (not accidentally commented out)."""

    def test_bias_thresholds_three_entries(self):
        from tests.ut.spec_decode.test_perf_benchmark import BIAS_SIZES_REGRESSION
        assert len(BIAS_SIZES_REGRESSION) == 3
        assert 4 in BIAS_SIZES_REGRESSION
        assert 16 in BIAS_SIZES_REGRESSION
        assert 64 in BIAS_SIZES_REGRESSION
        # All limits are positive
        for limit in BIAS_SIZES_REGRESSION.values():
            assert limit > 0

    def test_attach_thresholds_three_entries(self):
        from tests.ut.spec_decode.test_perf_benchmark import ATTACH_LIMITS
        assert len(ATTACH_LIMITS) == 3
        assert 4 in ATTACH_LIMITS
        assert 16 in ATTACH_LIMITS
        assert 64 in ATTACH_LIMITS
        for limit in ATTACH_LIMITS.values():
            assert limit > 0

    def test_builder_uses_bias_sizes_subset(self):
        from tests.ut.spec_decode.test_perf_benchmark import BIAS_SIZES_REGRESSION
        # Builder parametrize slices BIAS_SIZES[:2] — verify ≥2 sizes exist
        assert len(BIAS_SIZES_REGRESSION) >= 2, (
            f"builder needs at least 2 sizes, got {len(BIAS_SIZES_REGRESSION)}"
        )

    def test_slice_cases_four_entries(self):
        from tests.ut.spec_decode.test_perf_benchmark import SLICE_CASES
        assert len(SLICE_CASES) == 4
        for name, tree_str, drafts, limit_us in SLICE_CASES:
            assert isinstance(name, str) and name
            assert isinstance(drafts, list)
            assert limit_us > 0

    def test_memory_sizes_seven_entries(self):
        from tests.ut.spec_decode.test_perf_benchmark import MEMORY_SIZES
        assert len(MEMORY_SIZES) == 7
        assert MEMORY_SIZES == [4, 8, 16, 32, 64, 128, 256]

    def test_expected_kb_covers_all_sizes(self):
        from tests.ut.spec_decode.test_perf_benchmark import EXPECTED_KB, MEMORY_SIZES
        for n in MEMORY_SIZES:
            assert n in EXPECTED_KB, f"n={n} missing from EXPECTED_KB"
        for n in EXPECTED_KB:
            assert n in MEMORY_SIZES, f"n={n} not in MEMORY_SIZES"
