"""E2E test: Qwen3-0.6B MTP spec decode + speculative_token_tree correctness."""
import gc
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import pytest
import torch
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/huggingface_home/models--Qwen--Qwen3-0.6B"

TEST_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
]

TREE_TEST_CASES = [
    pytest.param(None, 3, id="linear_chain"),
    pytest.param("[(0,), (0,0), (0,1)]", 3, id="binary_tree"),
    pytest.param("[(0,), (0,0), (0,1), (0,2)]", 4, id="ternary_tree"),
    pytest.param("[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]", 5, id="mixed_shape"),
]


@pytest.fixture(scope="module")
def ref_outputs():
    sampling_params = SamplingParams(temperature=0, max_tokens=16, seed=42)
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=512,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    del llm
    gc.collect()
    try:
        torch.accelerator.empty_cache()
    except Exception:
        pass
    return outputs


@pytest.mark.parametrize("tree_str,num_spec_tokens", TREE_TEST_CASES)
def test_mtp_tree_output_correctness(ref_outputs, tree_str, num_spec_tokens):
    sampling_params = SamplingParams(temperature=0, max_tokens=16, seed=42)

    spec_kwargs = {
        "method": "draft_model",
        "model": MODEL_PATH,
        "num_speculative_tokens": num_spec_tokens,
    }
    if tree_str is not None:
        spec_kwargs["speculative_token_tree"] = tree_str

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=512,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        trust_remote_code=True,
        speculative_config=spec_kwargs,
    )
    try:
        spec_outputs = llm.generate(TEST_PROMPTS, sampling_params)
    finally:
        del llm
        gc.collect()
        try:
            torch.accelerator.empty_cache()
        except Exception:
            pass

    for i, (ref, spec) in enumerate(zip(ref_outputs, spec_outputs)):
        assert ref.outputs[0].text == spec.outputs[0].text, (
            f"Prompt {i} mismatch with tree={tree_str}:\n"
            f"  prompt: {TEST_PROMPTS[i]}\n"
            f"  ref:    {ref.outputs[0].text}\n"
            f"  spec:   {spec.outputs[0].text}"
        )
