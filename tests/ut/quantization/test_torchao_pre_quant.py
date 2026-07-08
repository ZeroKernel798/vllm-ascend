"""Scenario B: Create pre-quantized torchao checkpoint via torch.save."""

import json
import os
import sys

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "/root/models/qwen2.5-0.5b-torchao-int8wo"


def main():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from torchao.core.config import config_to_dict

    print("=== Step 1: Quantize with TorchAoConfig ===")
    quant_config = TorchAoConfig(Int8WeightOnlyConfig())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("=== Step 2: Save with torch.save ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save weights via torch.save (handles torchao tensor subclasses via pickle)
    state_dict = model.state_dict()
    torch.save(state_dict, f"{OUTPUT_DIR}/pytorch_model.bin")
    size_mb = os.path.getsize(f"{OUTPUT_DIR}/pytorch_model.bin") / 1024 / 1024
    print(f"Saved weights: {size_mb:.1f} MB")

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Build proper config.json with quantization_config format that upstream
    # TorchAOConfig.from_config expects (quant_type.default = config_to_dict(...))
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True).to_dict()
    quant_dict = config_to_dict(Int8WeightOnlyConfig())
    cfg["quantization_config"] = {
        "quant_method": "torchao",
        "quant_type": {"default": quant_dict},
    }
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("config.json written with quant_method=torchao + quant_type.default")

    # Verify
    with open(f"{OUTPUT_DIR}/config.json") as f:
        cfg_check = json.load(f)
    qm = cfg_check.get("quantization_config", {}).get("quant_method", "NOT FOUND")
    print(f"Verified quant_method: {qm}")
    assert "torchao" in str(qm).lower()

    print("=== Step 3: Verify with vLLM ===")
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=8)
    llm = LLM(
        model=OUTPUT_DIR,
        quantization="torchao",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        max_model_len=256,
    )
    out = llm.generate(["Hello, world!"], sp)
    text = out[0].outputs[0].text
    print(f"OUTPUT: {repr(text)}")
    assert text, "SCENARIO B FAILED: empty output"
    print("SCENARIO B PASSED")
    del llm


if __name__ == "__main__":
    main()
