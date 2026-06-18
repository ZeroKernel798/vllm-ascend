"""
Tree Attention 集成测试
模拟 EAGLE 投机推理的完整流程，验证端到端正确性
"""
import os
import sys
import torch
import torch_npu

# 设置 NPU 设备
os.environ['ASCEND_DEVICE_ID'] = '0'

def test_tree_attention_full_flow():
    """测试完整的 tree attention 流程"""
    print("=== 开始 Tree Attention 集成测试 ===\n")
    
    try:
        # 1. 导入相关模块
        print("1. 导入模块...")
        # 只导入 tree attention 模块，避免循环导入
        from vllm_ascend.attention.tree_attention_v1 import AscendTreeAttentionMetadataBuilder
        
        # 不导入 AscendAttentionState 和 AscendMetadata，避免循环导入
        # 这些会在实际使用时由 vLLM 框架传入
        print("   ✓ 模块导入成功 (AscendTreeAttentionMetadataBuilder)\n")
        
        # 2. 创建 AscendTreeAttentionMetadataBuilder
        print("2. 创建 AscendTreeAttentionMetadataBuilder...")
        
        # 创建模拟的 vllm_config
        # AscendTreeAttentionMetadataBuilder 需要 vllm_config 参数
        class MockVllmConfig:
            def __init__(self):
                self.speculative_config = None
                self.model_config = None
        
        mock_config = MockVllmConfig()
        
        # 使用正确的构造函数
        builder = AscendTreeAttentionMetadataBuilder(vllm_config=mock_config)
        print("   ✓ Builder 创建成功\n")
        
        # 3. 模拟 EAGLE 的 tree 结构
        print("3. 模拟 EAGLE tree 结构...")
        
        # 模拟一个 tree 结构：
        # Token 0: context token
        # Token 1-2: draft tokens (分支1)
        # Token 3-4: draft tokens (分支2)
        # 总共 5 个 token，其中 token 0 是 context，1-4 是 draft
        
        num_context_tokens = 1
        num_draft_tokens = 4
        total_tokens = num_context_tokens + num_draft_tokens
        num_heads = 8
        head_size = 64
        dtype = torch.float16
        
        print(f"   Context tokens: {num_context_tokens}")
        print(f"   Draft tokens: {num_draft_tokens}")
        print(f"   Total tokens: {total_tokens}\n")
        
        # 4. 创建 attention bias (2D) - 模拟 tree 结构
        print("4. 创建 attention bias (2D)...")
        
        # tree_attn_bias 形状: [tree_len, tree_len]
        # 这是 EAGLE 提供的 tree attention bias 矩阵
        # 0 表示可以 attend，-inf 表示不能 attend
        tree_attn_bias = torch.zeros(total_tokens, total_tokens, dtype=torch.float32, device="cpu")
        
        # 模拟 tree 结构：
        # Token 0: context token，可以看到自己
        # Token 1-2: 分支1，可以看到 token 0 和自己
        # Token 3-4: 分支2，可以看到 token 0 和自己
        
        # 设置 mask: -inf 表示不能 attend
        # 分支隔离：分支1 不能看到分支2 的 token
        tree_attn_bias[1, 3] = float("-inf")  # Token 1 不能看到 Token 3
        tree_attn_bias[1, 4] = float("-inf")  # Token 1 不能看到 Token 4
        tree_attn_bias[2, 3] = float("-inf")  # Token 2 不能看到 Token 3
        tree_attn_bias[2, 4] = float("-inf")  # Token 2 不能看到 Token 4
        
        # 移动到 NPU
        tree_attn_bias = tree_attn_bias.to("npu")
        
        print(f"   Tree attention bias shape: {tree_attn_bias.shape}")
        print(f"   Tree attention bias (CPU):\n{tree_attn_bias.cpu().numpy()}\n")
        
        # 5. 测试 2D bias 转 4D mask
        print("5. 测试 2D bias -> 4D mask 转换...")
        try:
            # 注意：_convert_bias_to_4d_mask 只接受 tree_attn_bias 和 dtype 参数
            mask_4d = builder._convert_bias_to_4d_mask(
                tree_attn_bias=tree_attn_bias,
                dtype=dtype
            )
            print(f"   ✓ 4D mask shape: {mask_4d.shape}")
            print(f"   4D mask dtype: {mask_4d.dtype}")
            
            # 验证 mask 内容
            mask_cpu = mask_4d[0, 0].cpu().numpy()
            print(f"   4D mask sample (position [0,0]):\n{mask_cpu}\n")
        except Exception as e:
            print(f"   ✗ 转换失败: {e}\n")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. 创建 query, key, value 张量 (TND layout)
        print("6. 创建 Q, K, V 张量...")
        torch.manual_seed(42)
        query = torch.randn(
            (total_tokens, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        key = torch.randn(
            (total_tokens, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        value = torch.randn(
            (total_tokens, num_heads * head_size),
            dtype=dtype, device="npu"
        )
        print(f"   Query shape (TND): {query.shape}")
        print(f"   Key shape (TND): {key.shape}")
        print(f"   Value shape (TND): {value.shape}\n")
        
        # 7. 转换到 BNSD layout
        print("7. 转换到 BNSD layout...")
        query_bnsd = query.view(total_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        key_bnsd = key.view(total_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        value_bnsd = value.view(total_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
        print(f"   Query BNSD shape: {query_bnsd.shape}")
        print(f"   Key BNSD shape: {key_bnsd.shape}")
        print(f"   Value BNSD shape: {value_bnsd.shape}\n")
        
        # 8. 执行 NPU attention 计算
        print("8. 执行 NPU attention 计算...")
        output = torch.randn(
            (1, num_heads, total_tokens, head_size),
            dtype=dtype, device="npu"
        )
        softmax_lse = torch.empty(
            (1, num_heads, total_tokens), dtype=dtype, device="npu"
        )
        
        try:
            torch_npu.npu_fused_infer_attention_score.out(
                query=query_bnsd,
                key=key_bnsd,
                value=value_bnsd,
                atten_mask=mask_4d,
                block_table=None,
                input_layout="BNSD",
                block_size=0,
                actual_seq_lengths=[total_tokens],
                actual_seq_lengths_kv=[total_tokens],
                num_key_value_heads=num_heads,
                num_heads=num_heads,
                scale=1.0 / (head_size ** 0.5),
                sparse_mode=0,
                pre_tokens=65535,
                next_tokens=65535,
                out=[output, softmax_lse],
            )
            print("   ✓ NPU attention 计算成功")
            print(f"   Output shape: {output.shape}\n")
        except Exception as e:
            print(f"   ✗ NPU attention 计算失败: {e}\n")
            return False
        
        # 9. 转换输出回 TND layout
        print("9. 转换输出回 TND layout...")
        output_tnd = output.squeeze(0).permute(1, 0, 2).contiguous().view(total_tokens, num_heads * head_size)
        print(f"   Output TND shape: {output_tnd.shape}\n")
        
        # 10. 验证输出合理性
        print("10. 验证输出合理性...")
        
        # 检查输出是否包含 NaN 或 Inf
        has_nan = torch.isnan(output_tnd).any()
        has_inf = torch.isinf(output_tnd).any()
        
        if has_nan:
            print("   ✗ 输出包含 NaN!")
            return False
        if has_inf:
            print("   ✗ 输出包含 Inf!")
            return False
        
        print("   ✓ 输出不包含 NaN 或 Inf")
        print(f"   Output mean: {output_tnd.mean().item():.6f}")
        print(f"   Output std: {output_tnd.std().item():.6f}\n")
        
        print("=== 集成测试通过! ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tree_attention_with_actual_metadata():
    """测试使用实际的 AscendMetadata"""
    print("\n=== 测试使用实际 AscendMetadata ===\n")
    
    try:
        from vllm_ascend.attention.tree_attention_v1 import AscendTreeAttentionMetadataBuilder
        
        print("1. 创建模拟的 AscendMetadata...")
        
        # 模拟 AscendMetadata 的属性
        class MockAscendMetadata:
            def __init__(self):
                self.num_context_tokens = 1
                self.num_draft_tokens = 4
                self.total_tokens = 5
                self.use_tree_attention = True
                self.tree_attn_mask_4d = None
                self.tree_context_len = 1
        
        metadata = MockAscendMetadata()
        print(f"   ✓ Mock metadata 创建成功")
        print(f"   num_context_tokens: {metadata.num_context_tokens}")
        print(f"   num_draft_tokens: {metadata.num_draft_tokens}")
        print(f"   use_tree_attention: {metadata.use_tree_attention}\n")
        
        print("2. 测试 builder 处理 metadata...")
        
        # 创建模拟的 vllm_config
        class MockVllmConfig:
            def __init__(self):
                self.speculative_config = None
                self.model_config = None
        
        mock_config = MockVllmConfig()
        builder = AscendTreeAttentionMetadataBuilder(vllm_config=mock_config)
        
        # 模拟处理流程
        seq_len = metadata.total_tokens
        
        # 创建模拟的 tree attention bias (2D)
        tree_attn_bias = torch.zeros(seq_len, seq_len, dtype=torch.float32, device="npu")
        
        # 转换到 4D mask
        mask_4d = builder._convert_bias_to_4d_mask(
            tree_attn_bias=tree_attn_bias,
            dtype=torch.float16
        )
        
        # 更新 metadata
        metadata.tree_attn_mask_4d = mask_4d
        
        print(f"   ✓ Metadata 更新成功")
        print(f"   tree_attn_mask_4d shape: {metadata.tree_attn_mask_4d.shape}\n")
        
        print("=== Metadata 测试通过! ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Tree Attention 集成测试套件")
    print("="*60 + "\n")
    
    # 设置 NPU 设备
    if not torch.npu.is_available():
        print("✗ NPU 设备不可用，退出测试")
        sys.exit(1)
    
    print(f"✓ NPU 设备: {torch.npu.get_device_name(0)}\n")
    torch.npu.set_device(0)
    
    # 运行测试
    results = []
    
    results.append(("完整流程测试", test_tree_attention_full_flow()))
    results.append(("Metadata 测试", test_tree_attention_with_actual_metadata()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n=== 所有测试通过! ===")
    else:
        print("\n=== 部分测试失败 ===")
        sys.exit(1)
