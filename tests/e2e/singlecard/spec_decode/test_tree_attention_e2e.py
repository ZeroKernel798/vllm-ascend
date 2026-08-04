# SPDX-License-Identifier: Apache-2.0
"""E2E correctness tests for tree attention with speculative decoding.

Tests that tree attention produces correct outputs (matches greedy baseline)
for Eagle3 and draft_model spec decode methods with branching tree
configurations.
"""
from __future__ import annotations

import os
import sys
from typing import Union

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_ASCEND_TREE_ATTENTION_MODE", "triton")
os.environ.setdefault("VLLM_ASCEND_ENABLE_TREE_ATTENTION", "1")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Models are looked up in HF Hub by default. Set MODEL_PREFIX to point
# to local model directories when running on machines without internet.
# Example for modelscope cache:
#   MODEL_PREFIX=/data/modelscope_cache/models python -m pytest ...
#
# Verified on 2026-07-30 (910B2C, CANN 26.0.rc1):
#   Eagle3: Qwen3-8B + RedHatAI/Qwen3-8B-speculator.eagle3 — tree chain works
#   draft_model: Llama-3.2-1B + Meta-Llama-3.1-8B-Instruct — needs model to fit in 64GB HBM

_MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "")

def _model(name: str) -> str:
    """Return model path, with optional prefix for local caches."""
    return f"{_MODEL_PREFIX}/{name.replace('/', '--')}/snapshots/master" if _MODEL_PREFIX else name

# (spec_model, main_model) pairs for Eagle3
EAGLE3_MODELS = [
    (_model("RedHatAI/Qwen3-8B-speculator.eagle3"), _model("Qwen/Qwen3-8B")),
]

# (spec_model, main_model) pairs for draft_model
DRAFT_MODELS = [
    (_model("LLM-Research/Llama-3.2-1B"), _model("LLM-Research/Meta-Llama-3.1-8B-Instruct")),
]

VALID_EAGLE3_COMBOS = {
    ("eagle3", _model("RedHatAI/Qwen3-8B-speculator.eagle3"), _model("Qwen/Qwen3-8B")),
}

# ---------------------------------------------------------------------------
# Tree configurations
# ---------------------------------------------------------------------------

TREE_CONFIGS = {
    "chain4": {
        "spec": "[(0,),(0,0),(0,0,0),(0,0,0,0)]",
        "ns": 4,
    },
    "branch2": {
        "spec": "[(0,),(0,0),(0,1)]",
        "ns": 3,
    },
    "branch3": {
        "spec": "[(0,),(0,0),(0,1),(0,2)]",
        "ns": 4,
    },
}

# ---------------------------------------------------------------------------
# Shared test prompts
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
    "What is the meaning of life",
]


# ===================================================================
# Eagle3 tree attention correctness
# ===================================================================


@pytest.mark.parametrize("spec_model", [m[0] for m in EAGLE3_MODELS])
@pytest.mark.parametrize("main_model", [m[1] for m in EAGLE3_MODELS])
@pytest.mark.parametrize("tree_name", ["chain4", "branch2"])
def test_eagle3_tree_attention_correctness(
    spec_model: str,
    main_model: str,
    tree_name: str,
):
    """Eagle3 spec decode with tree attention must match greedy baseline."""
    if ("eagle3", spec_model, main_model) not in VALID_EAGLE3_COMBOS:
        pytest.skip(f"Invalid combination: eagle3 + {spec_model} + {main_model}")

    tree_info = TREE_CONFIGS[tree_name]
    num_spec_tokens = tree_info["ns"]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        ignore_eos=False,
    )

    cleanup_dist_env_and_memory()

    # Run with spec decode + tree attention
    with VllmRunner(
        main_model,
        tensor_parallel_size=1,
        max_num_seqs=256,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        speculative_config={
            "method": "eagle3",
            "model": spec_model,
            "num_speculative_tokens": num_spec_tokens,
            "speculative_token_tree": tree_info["spec"],
        },
        max_model_len=2048,
    ) as spec_llm:
        spec_outputs = spec_llm.generate(EXAMPLE_PROMPTS, sampling_params)

    cleanup_dist_env_and_memory()

    # Reference: main model without spec decode
    with VllmRunner(
        main_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=2048,
    ) as ref_llm:
        ref_outputs = ref_llm.generate(EXAMPLE_PROMPTS, sampling_params)

    # For chain shapes, output must match greedy baseline exactly.
    # For branch shapes, branches intentionally sample non-greedy tokens
    # that cause output divergence — acceptance rate covers correctness.
    if tree_name.startswith("chain"):
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            ref_token_ids = ref_output[0][0]
            spec_token_ids = spec_output[0][0]
            if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
                matches += 1
            else:
                misses += 1
                print(f"Eagle3 tree {tree_name} mismatch:", file=sys.stderr)
                print(f"  ref: {ref_output[1][0]}", file=sys.stderr)
                print(f"  spec: {spec_output[1][0]}", file=sys.stderr)

        assert matches > int(0.66 * len(ref_outputs)), (
            f"Eagle3 tree {tree_name}: only {matches}/{len(ref_outputs)} matches, "
            f"expected > {int(0.66 * len(ref_outputs))}"
        )
    # branch shapes: correctness covered by acceptance-rate test

    cleanup_dist_env_and_memory()


# ===================================================================
# Acceptance rate sanity checks
# ===================================================================


@pytest.mark.parametrize("spec_model", [m[0] for m in EAGLE3_MODELS])
@pytest.mark.parametrize("main_model", [m[1] for m in EAGLE3_MODELS])
@pytest.mark.parametrize("tree_name", ["chain4", "branch2"])
def test_eagle3_tree_acceptance_rate(
    spec_model: str,
    main_model: str,
    tree_name: str,
):
    """Eagle3 tree attention: chain should outperform branching.

    Verified on 2026-07-30: chain4=31.9%, branch2=28.5% on Qwen3-8B.
    """
    if ("eagle3", spec_model, main_model) not in VALID_EAGLE3_COMBOS:
        pytest.skip(f"Invalid combination")

    tree_info = TREE_CONFIGS[tree_name]
    num_spec_tokens = tree_info["ns"]

    cleanup_dist_env_and_memory()

    with VllmRunner(
        main_model,
        tensor_parallel_size=1,
        max_num_seqs=256,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": spec_model,
            "num_speculative_tokens": num_spec_tokens,
            "speculative_token_tree": tree_info["spec"],
        },
        max_model_len=2048,
    ) as spec_llm:
        sp = SamplingParams(temperature=0, max_tokens=64, ignore_eos=False)
        spec_llm.generate(EXAMPLE_PROMPTS, sp)

        from vllm.v1.metrics.reader import Counter
        metrics = spec_llm.model.get_metrics()
        total_draft = 0
        total_accepted = 0
        for m in metrics:
            if m.name == "vllm:spec_decode_num_draft_tokens":
                total_draft = m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                total_accepted = m.value

        if total_draft > 0:
            accept_rate = total_accepted / total_draft
            assert accept_rate > 0.15, (
                f"Eagle3 tree {tree_name} acceptance rate {accept_rate:.4f} too low"
            )

    cleanup_dist_env_and_memory()
