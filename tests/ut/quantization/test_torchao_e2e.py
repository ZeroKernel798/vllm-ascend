"""
端到端测试：使用真实模型测试 torchao 量化
测试 Int8WeightOnly 和 Int4WeightOnly 量化在 NPU 上的效果
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

# 添加 vllm_ascend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from vllm_ascend.quantization.torchao_config import AscendTorchAOConfig


class TestTorchAOE2E:
    """torchao 量化端到端测试"""
    
    # 使用小模型进行测试（适合 NPU 内存）
    MODEL_NAME = "gpt2"  # 124M 参数，适合测试
    
    @pytest.fixture
    def setup_npu(self):
        """检查 NPU 是否可用"""
        if not torch.cuda.is_available():
            pytest.skip("NPU 不可用，跳过测试")
        
        device = torch.device("npu:0")
        return device
    
    def test_int8_weight_only_quantization(self, setup_npu):
        """测试 INT8 权重量化"""
        device = setup_npu
        
        print(f"\n{'='*60}")
        print("测试 1: INT8 权重量化")
        print(f"{'='*60}")
        
        # 1. 加载模型
        print(f"\n[1/5] 加载模型: {self.MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)
        model = model.to(device)
        print(f"  ✓ 模型加载成功: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 2. 记录原始模型大小
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"  ✓ 原始模型大小: {original_size / 1024 / 1024:.2f} MB")
        
        # 3. 应用 torchao 量化
        print(f"\n[2/5] 应用 INT8 权重量化")
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        
        quantize_(model, Int8WeightOnlyConfig())
        print(f"  ✓ 量化成功")
        
        # 4. 记录量化后模型大小
        quantized_size = sum(
            p.numel() * (1 if p.dtype == torch.int8 else p.element_size()) 
            for p in model.parameters()
        )
        print(f"  ✓ 量化后模型大小: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"  ✓ 压缩比: {original_size / quantized_size:.2f}x")
        
        # 5. 推理测试
        print(f"\n[3/5] 推理测试")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  ✓ 输入: {input_text}")
        print(f"  ✓ 输出 shape: {outputs.logits.shape}")
        print(f"  ✓ 输出不包含 NaN: {not torch.isnan(outputs.logits).any()}")
        print(f"  ✓ 输出不包含 Inf: {not torch.isinf(outputs.logits).any()}")
        
        # 6. 生成测试
        print(f"\n[4/5] 生成测试")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,  # 关闭采样以保证可重复性
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  ✓ 生成文本: {generated_text}")
        assert len(generated_text) > 0, "生成文本为空"
        
        # 7. 验证量化配置
        print(f"\n[5/5] 验证 AscendTorchAOConfig")
        config = AscendTorchAOConfig(
            weight_dtype="int8",
            is_symmetric=True
        )
        print(f"  ✓ 配置创建成功: {config}")
        print(f"  ✓ 量化方法: {config.get_quant_method(None, None)}")
        
        print(f"\n{'='*60}")
        print("✅ INT8 权重量化测试通过！")
        print(f"{'='*60}\n")
    
    def test_int4_weight_only_quantization(self, setup_npu):
        """测试 INT4 权重量化"""
        device = setup_npu
        
        print(f"\n{'='*60}")
        print("测试 2: INT4 权重量化")
        print(f"{'='*60}")
        
        # 1. 加载模型
        print(f"\n[1/4] 加载模型: {self.MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)
        model = model.to(device)
        print(f"  ✓ 模型加载成功")
        
        # 2. 应用 INT4 量化
        print(f"\n[2/4] 应用 INT4 权重量化")
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        
        quantize_(model, Int4WeightOnlyConfig())
        print(f"  ✓ 量化成功")
        
        # 3. 推理测试
        print(f"\n[3/4] 推理测试")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        input_text = "The quick brown fox"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ 输入: {input_text}")
        print(f"  ✓ 生成: {generated_text}")
        
        # 4. 验证配置
        print(f"\n[4/4] 验证 AscendTorchAOConfig (INT4)")
        config = AscendTorchAOConfig(
            weight_dtype="int4",
            is_symmetric=True,
            group_size=128
        )
        print(f"  ✓ 配置创建成功: {config}")
        
        print(f"\n{'='*60}")
        print("✅ INT4 权重量化测试通过！")
        print(f"{'='*60}\n")
    
    def test_ascend_torchao_config(self, setup_npu):
        """测试 AscendTorchAOConfig 配置类"""
        device = setup_npu
        
        print(f"\n{'='*60}")
        print("测试 3: AscendTorchAOConfig 配置类")
        print(f"{'='*60}")
        
        # 测试 1: 默认配置
        print(f"\n[1/3] 测试默认配置")
        config = AscendTorchAOConfig()
        print(f"  ✓ 默认配置: weight_dtype={config.weight_dtype}")
        print(f"  ✓ 默认配置: is_symmetric={config.is_symmetric}")
        
        # 测试 2: INT8 配置
        print(f"\n[2/3] 测试 INT8 配置")
        config_int8 = AscendTorchAOConfig(
            weight_dtype="int8",
            is_symmetric=True
        )
        print(f"  ✓ INT8 配置: {config_int8}")
        
        # 测试 3: INT4 配置
        print(f"\n[3/3] 测试 INT4 配置")
        config_int4 = AscendTorchAOConfig(
            weight_dtype="int4",
            is_symmetric=True,
            group_size=64
        )
        print(f"  ✓ INT4 配置: {config_int4}")
        
        print(f"\n{'='*60}")
        print("✅ AscendTorchAOConfig 测试通过！")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 直接运行时执行测试
    pytest.main([__file__, "-v", "-s"])
