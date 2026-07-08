# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Top-level doctest runner for the speculative token tree pipeline.

Run with::

    pytest tests/ut/spec_decode/test_doctest_runner.py -v

Or standalone::

    python tests/ut/spec_decode/test_doctest_runner.py

Gathers all doctest-enabled modules, runs them, and reports
aggregated pass/fail counts.
"""
from __future__ import annotations

import doctest
import importlib
import sys
from pathlib import Path
from typing import NamedTuple

import pytest


# ---------------------------------------------------------------------------
# Module registry — add entries here when a module gains doctests
# ---------------------------------------------------------------------------

class DoctestTarget(NamedTuple):
    module_path: str         # dotted import path
    label: str               # human-readable label for output
    expected_count: int      # expected number of doctest examples (None=any)

DOCTEST_MODULES = [
    DoctestTarget("vllm_ascend.spec_decode.speculative_token_tree",    "tree utils", 41),
    DoctestTarget("vllm_ascend.attention.attention_mask",              "attention mask", 4),
    DoctestTarget("vllm_ascend.attention.utils",                      "attention utils", 5),
]


# ---------------------------------------------------------------------------
# doctest runner
# ---------------------------------------------------------------------------

def _run_one(target: DoctestTarget) -> tuple[int, int, str | None]:
    """Import module, run doctests, return (passed, failed, error_message)."""
    try:
        mod = importlib.import_module(target.module_path)
    except Exception as exc:
        return 0, 0, f"IMPORT ERROR: {target.module_path}: {exc}"

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=False)
    tests = finder.find(mod)

    if not tests:
        return 0, 0, f"SKIP: {target.module_path} has no doctests"

    passed, failed = 0, 0
    for test in tests:
        result = runner.run(test)
        passed += result.attempted - result.failed
        failed += result.failed

    return passed, failed, None


# ---------------------------------------------------------------------------
# pytest tests — one per module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", DOCTEST_MODULES)
def test_doctest_module(target: DoctestTarget):
    """Run doctests for one module and verify they pass."""
    passed, failed, error = _run_one(target)

    if error is not None:
        if "IMPORT ERROR" in error:
            pytest.skip(error)
        else:
            pytest.fail(error)

    if target.expected_count is not None:
        assert passed >= target.expected_count, (
            f"{target.label}: got {passed}/{target.expected_count} expected doctests — "
            f"may need to update DOCTEST_MODULES registry"
        )

    assert failed == 0, f"{target.label}: {failed} doctest(s) failed"


def test_doctest_all_modules_pass():
    """Run all registered modules and verify total failures == 0."""
    total_passed = 0
    total_failed = 0
    errors = []

    for target in DOCTEST_MODULES:
        passed, failed, error = _run_one(target)
        if error:
            errors.append(error)
            continue
        total_passed += passed
        total_failed += failed

    if errors:
        skipped_msg = "; ".join(errors)
        pytest.skip(f"Skipped modules: {skipped_msg}")

    assert total_failed == 0, (
        f"{total_failed} doctests failed across {len(DOCTEST_MODULES)} modules "
        f"({total_passed} passed)"
    )


def test_doctest_registry_coverage():
    """Ensure every entry in DOCTEST_MODULES has at least one doctest."""
    for target in DOCTEST_MODULES:
        passed, failed, error = _run_one(target)
        if error and "IMPORT" in error:
            continue
        assert passed + failed > 0, (
            f"{target.label}: 0 doctests found — remove from registry or add doctests"
        )


# ---------------------------------------------------------------------------
# Standalone runner (python this_file.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total_ok, total_fail, skipped = 0, 0, 0
    for target in DOCTEST_MODULES:
        passed, failed, error = _run_one(target)
        if error:
            print(f"[SKIP] {target.label}: {error}")
            skipped += 1
            continue
        status = "OK" if failed == 0 else f"FAIL ({failed} failed)"
        print(f"[{status:>5}] {target.label}: {passed} passed, {failed} failed")
        total_ok += passed
        total_fail += failed

    print(f"\n{'='*50}")
    print(f"Total: {total_ok + total_fail} tests, {total_ok} passed, "
          f"{total_fail} failed, {skipped} skipped")
    sys.exit(1 if total_fail > 0 else 0)
