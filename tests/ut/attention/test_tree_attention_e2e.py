"""
Tree Attention 端到端测试 (Phase 4)
最小化测试：验证 tree attention 的完整流程，不依赖真实模型
"""
import os
import sys
import torch
import torch_npu

# 设置 NPU 设备
os.environ['ASCEND_DEVICE_ID'] = '0'

def test_tree_attention_e2e():
    """端到端测试：模拟完整的 tree attention 流程"""
    print("=== Tree Attention 端到端测试 (Phase 4) ===\n")
    
    try:
        # 1. 导入模块
        print("1. 导入模块...")
        from vllm_ascend.attention.tree_attention_v1 import AscendTreeAttentionMetadataBuilder
        from vllm_ascend.speculative_token_tree import (
            build_speculative_token_tree_plan,
            prepare_speculative_token_tree_attn_bias,
        )
        print("   ✓ 模块导入成功\n")
        
        # 2. 模拟 EAGLE 配置
        print("2. 模拟 EAGLE 配置...")
        
        # 使用一个简单的分支树
        tree_choices = [(0,), (1,), (0, 0), (1, 0)]
        print(f"   Tree choices: {tree_choices}")
        
        # 构建树计划
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        print(f"   Tree length: {tree_plan.tree_len}")
        print(f"   Depth counts: {tree_plan.depth_counts}\n")
        
        # 3. 准备 attention bias
        print("3. 准备 attention bias...")
        tree_attn_bias = prepare_speculative_token_tree_attn_bias(tree_choices)
        print(f"   Attention bias shape: {tree_attn_bias.shape}")
        print(f"   Attention bias (CPU):\n{tree_attn_bias.cpu().numpy()}\n")
        
        # 4. 创建 AscendTreeAttentionMetadataBuilder
        print("4. 创建 AscendTreeAttentionMetadataBuilder...")
        
        # 创建模拟的 vllm_config
        class MockVllmConfig:
            def __init__(self):
                self.speculative_config = type('SpeculativeConfig', (), {
                    'speculative_token_tree': str(tree_choices)
                })()
                self.additional_config = {
                    'enable_ascend_tree_attention_experimental': True
                }
        
        mock_config = MockVllmConfig()
        builder = AscendTreeAttentionMetadataBuilder(vllm_config=mock_config)
        print("   ✓ Builder 创建成功\n")
        
        # 5. 测试 _convert_bias_to_4d_mask
        print("5. 测试 _convert_bias_to_4d_mask...")
        mask_4d = builder._convert_bias_to_4d_mask(
            tree_attn_bias=tree_attn_bias,
            dtype=torch.bool
        )
        
        # 确保 mask 在正确的设备上
        if mask_4d is not None and mask_4d.device.type == 'cpu':
            mask_4d = mask_4d.to("npu")
        
        print(f"   ✓ 4D mask shape: {mask_4d.shape}")
        print(f"   4D mask dtype: {mask_4d.dtype}")
        print(f"   4D mask device: {mask_4d.device}")
        print(f"   4D mask sample:\n{mask_4d[0, 0].cpu().numpy()}\n")
        
        # 6. 模拟 NPU attention 计算
        print("6. 模拟 NPU attention 计算...")
        
        if not torch.npu.is_available():
            print("   ✗ NPU 设备不可用，跳过 NPU 计算")
            return False
        
        num_heads = 8
        head_size = 64
        tree_len = tree_plan.tree_len
        dtype = torch.float16
        
        # 创建 Q, K, V
        torch.manual_seed(42)
        query = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        key = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        value = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        
        # 转换到 BNSD layout
        query_bnsd = query.view(tree_len, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        key_bnsd = key.view(tree_len, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        value_bnsd = value.view(tree_len, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        
        # 执行 attention
        output = torch.randn(
            (1, num_heads, tree_len, head_size),
            dtype=dtype, device="npu"
        )
        softmax_lse = torch.empty(
            (1, num_heads, tree_len), dtype=dtype, device="npu"
        )
        
        torch_npu.npu_fused_infer_attention_score.out(
            query=query_bnsd,
            key=key_bnsd,
            value=value_bnsd,
            atten_mask=mask_4d,
            block_table=None,
            input_layout="BNSD",
            block_size=0,
            actual_seq_lengths=[tree_len],
            actual_seq_lengths_kv=[tree_len],
            num_key_value_heads=num_heads,
            num_heads=num_heads,
            scale=1.0 / (head_size ** 0.5),
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            out=[output, softmax_lse],
        )
        
        print("   ✓ NPU attention 计算成功")
        print(f"   Output shape: {output.shape}")
        
        # 验证输出
        output_tnd = output.squeeze(0).permute(1, 0, 2).contiguous().view(tree_len, num_heads * head_size)
        has_nan = torch.isnan(output_tnd).any()
        has_inf = torch.isinf(output_tnd).any()
        
        if has_nan or has_inf:
            print("   ✗ 输出包含 NaN 或 Inf")
            return False
        
        print(f"   Output mean: {output_tnd.mean().item():.6f}")
        print(f"   Output std: {output_tnd.std().item():.6f}\n")
        
        print("=== 端到端测试通过! ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Tree Attention Phase 4 端到端测试")
    print("="*60 + "\n")
    
    if not torch.npu.is_available():
        print("✗ NPU 设备不可用，退出测试")
        sys.exit(1)
    
    print(f"✓ NPU 设备: {torch.npu.get_device_name(0)}\n")
    torch.npu.set_device(0)
    
    result = test_tree_attention_e2e()
    
    if result:
        print("=== 所有测试通过! ===")
    else:
        print("=== 测试失败 ===")
        sys.exit(1)
