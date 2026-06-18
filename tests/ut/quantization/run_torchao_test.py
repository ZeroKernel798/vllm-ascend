#!/usr/bin/env python3
"""
简化版 torchao 量化测试脚本
可以直接运行，不依赖 pytest
使用方法:
    python run_torchao_test.py
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检查 NPU 是否可用
if not torch.cuda.is_available():
    print("❌ NPU 不可用，请检查驱动和安装")
    exit(1)

print("=" * 70)
print("torchao 量化测试 - NPU")
print("=" * 70)

# 配置
MODEL_NAME = "gpt2"  # 小模型，适合测试
DEVICE = "npu:0"

# 测试 1: 基本量化流程
print("\n[测试 1] INT8 权重量化 - 基本流程")
print("-" * 70)

try:
    # 1. 加载模型
    print(f"1. 加载模型: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    
    original_params = sum(p.numel() for p in model.parameters())
    original_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"   ✓ 模型加载成功: {original_params:,} 参数")
    print(f"   ✓ 模型大小: {original_size_mb:.2f} MB")
    
    # 2. 量化模型
    print(f"\n2. 应用 INT8 权重量化")
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    
    start_time = time.time()
    quantize_(model, Int8WeightOnlyConfig())
    quant_time = time.time() - start_time
    
    print(f"   ✓ 量化成功 (耗时: {quant_time:.2f}s)")
    
    # 3. 推理测试
    print(f"\n3. 推理测试")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    test_inputs = [
        "Hello, world!",
        "The quick brown fox",
        "Once upon a time"
    ]
    
    for test_input in test_inputs:
        inputs = tokenizer(test_input, return_tensors="pt").to(DEVICE)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        infer_time = time.time() - start_time
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输入: {test_input}")
        print(f"   输出: {generated}")
        print(f"   耗时: {infer_time:.3f}s\n")
    
    print("✅ 测试 1 通过: INT8 权重量化")
    
except Exception as e:
    print(f"❌ 测试 1 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 2: INT4 量化
print("\n[测试 2] INT4 权重量化")
print("-" * 70)

try:
    print(f"1. 重新加载模型")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    print(f"   ✓ 模型加载成功")
    
    print(f"\n2. 应用 INT4 权重量化")
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    
    quantize_(model, Int4WeightOnlyConfig())
    print(f"   ✓ 量化成功")
    
    print(f"\n3. 推理测试")
    inputs = tokenizer("Hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   生成: {generated}")
    
    print("✅ 测试 2 通过: INT4 权重量化")
    
except Exception as e:
    print(f"❌ 测试 2 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 内存对比
print("\n[测试 3] 量化前后内存对比")
print("-" * 70)

try:
    import psutil
    import os
    
    # 获取 NPU 内存信息（简化版，实际需要 ascend 官方 API）
    print("   提示: 完整的内存测试需要 ascend 官方 API")
    print("   当前可以使用 npu-smi info 手动查看")
    
    print("✅ 测试 3 完成")
    
except ImportError:
    print("   跳过: 需要 psutil 库")
except Exception as e:
    print(f"❌ 测试 3 失败: {e}")

# 测试 4: vLLM + torchao 集成（如果 vLLM 可用）
print("\n[测试 4] vLLM + torchao 集成测试")
print("-" * 70)

try:
    from vllm import LLM, SamplingParams
    from vllm_ascend.quantization.torchao_config import AscendTorchAOConfig
    
    print("1. 创建 LLM 实例（带 torchao 量化）")
    print("   注意: 这需要完整的 vLLM 环境")
    
    # 这里只是示例，实际需要更多配置
    # llm = LLM(
    #     model=MODEL_NAME,
    #     quantization="torchao",
    #     dtype="float16"
    # )
    
    print("   提示: 完整集成测试需要 vLLM 正确安装")
    print("✅ 测试 4 跳过（需要完整环境）")
    
except ImportError as e:
    print(f"   跳过: {e}")
except Exception as e:
    print(f"❌ 测试 4 失败: {e}")

# 总结
print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
print("\n建议下一步:")
print("1. 如果测试通过，尝试更大的模型（如 gpt2-medium）")
print("2. 运行性能基准测试（速度、内存）")
print("3. 集成到 vLLM 中进行端到端测试")
print("4. 提交 PR 到上游仓库\n")
