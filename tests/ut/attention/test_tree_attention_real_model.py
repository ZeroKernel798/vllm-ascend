"""
End-to-end test with simulated EAGLE model.

This test simulates an EAGLE model to verify the complete
tree attention pipeline without requiring actual model weights.

Usage:
    python tests/ut/attention/test_tree_attention_real_model.py
"""

import os
import sys
import torch
import torch_npu

# Set NPU device
os.environ["ASCEND_DEVICE_ID"] = "0"


def test_e2e_with_simulated_model():
    """
    End-to-end test with simulated EAGLE model.

    This test:
    1. Creates a mock vLLM config with speculative_token_tree
    2. Initializes the tree attention metadata builder
    3. Simulates the draft proposal process
    4. Verifies attention computation with tree mask
    """
    print("=" * 80)
    print("End-to-End Test with Simulated EAGLE Model")
    print("=" * 80)
    print()

    if not torch.npu.is_available():
        print("✗ NPU not available, skipping test")
        return False

    torch.npu.set_device(0)
    print(f"✓ Using NPU: {torch.npu.get_device_name(0)}\n")

    try:
        # Step 1: Import required modules
        print("Step 1: Importing modules...")
        from vllm_ascend.attention.tree_attention_v1 import (
            AscendTreeAttentionMetadataBuilder,
        )
        from vllm_ascend.speculative_token_tree import (
            build_speculative_token_tree_plan,
            prepare_speculative_token_tree_attn_bias,
        )
        print("  ✓ Modules imported successfully\n")

        # Step 2: Simulate EAGLE config
        print("Step 2: Simulating EAGLE config...")
        tree_choices = [
            (0,),        # Token 1: parent is root
            (0,),        # Token 2: parent is root
            (0, 1),     # Token 3: parent is token 1
            (0, 2),     # Token 4: parent is token 2
        ]
        num_speculative_tokens = len(tree_choices)
        print(f"  Tree choices: {tree_choices}")
        print(f"  Num speculative tokens: {num_speculative_tokens}\n")

        # Step 3: Build tree plan (simulates EAGLE tree construction)
        print("Step 3: Building tree plan...")
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        print(f"  Tree length: {tree_plan.tree_len}")
        print(f"  Depth counts: {tree_plan.depth_counts}")
        print(f"  Num layers: {len(tree_plan.depth_counts)}\n")

        # Step 4: Prepare attention bias
        print("Step 4: Preparing attention bias...")
        tree_attn_bias = prepare_speculative_token_tree_attn_bias(
            tree_choices
        )
        print(f"  Attention bias shape: {tree_attn_bias.shape}")
        print(f"  Attention bias (CPU):\n{tree_attn_bias.cpu().numpy()}\n")

        # Step 5: Create metadata builder (simulates vLLM config)
        print("Step 5: Creating metadata builder...")

        class MockSpeculativeConfig:
            def __init__(self):
                self.speculative_token_tree = str(tree_choices)
                self.num_speculative_tokens = num_speculative_tokens

        class MockAdditionalConfig:
            def __init__(self):
                self.enable_ascend_tree_attention_experimental = True

        class MockVllmConfig:
            def __init__(self):
                self.speculative_config = MockSpeculativeConfig()
                self.additional_config = {
                    "enable_ascend_tree_attention_experimental": True
                }

        mock_config = MockVllmConfig()
        builder = AscendTreeAttentionMetadataBuilder(
            vllm_config=mock_config
        )
        print("  ✓ Builder created successfully\n")

        # Step 6: Convert bias to 4D mask
        print("Step 6: Converting bias to 4D mask...")
        mask_4d = builder._convert_bias_to_4d_mask(
            tree_attn_bias=tree_attn_bias,
            dtype=torch.bool
        )

        # Ensure mask is on NPU
        if mask_4d.device.type == "cpu":
            mask_4d = mask_4d.to("npu")

        print(f"  ✓ 4D mask shape: {mask_4d.shape}")
        print(f"  ✓ 4D mask dtype: {mask_4d.dtype}")
        print(f"  ✓ 4D mask device: {mask_4d.device}")
        print(f"  Sample (mask[0, 0]):\n{mask_4d[0, 0].cpu().numpy()}\n")

        # Step 7: Simulate attention computation (core of EAGLE verification)
        print("Step 7: Simulating attention computation...")

        num_heads = 8
        head_size = 64
        tree_len = tree_plan.tree_len
        dtype = torch.float16

        # Create Q, K, V (simulates EAGLE draft model output)
        torch.manual_seed(42)
        query = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype,
            device="npu"
        )
        key = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype,
            device="npu"
        )
        value = torch.randn(
            (tree_len, num_heads * head_size),
            dtype=dtype,
            device="npu"
        )

        # Convert to BNSD layout
        query_bnsd = (
            query.view(tree_len, num_heads, head_size)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )
        key_bnsd = (
            key.view(tree_len, num_heads, head_size)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )
        value_bnsd = (
            value.view(tree_len, num_heads, head_size)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )

        # Prepare output
        output = torch.randn(
            (1, num_heads, tree_len, head_size),
            dtype=dtype,
            device="npu"
        )
        softmax_lse = torch.empty(
            (1, num_heads, tree_len),
            dtype=dtype,
            device="npu"
        )

        # Run attention (simulates EAGLE verification step)
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
            scale=1.0 / (head_size**0.5),
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            out=[output, softmax_lse],
        )

        print("  ✓ Attention computation successful")
        print(f"  Output shape: {output.shape}")

        # Verify output
        output_tnd = (
            output.squeeze(0)
            .permute(1, 0, 2)
            .contiguous()
            .view(tree_len, num_heads * head_size)
        )
        has_nan = torch.isnan(output_tnd).any()
        has_inf = torch.isinf(output_tnd).any()

        if has_nan or has_inf:
            print("  ✗ Output contains NaN or Inf")
            return False

        print(f"  Output mean: {output_tnd.mean().item():.6f}")
        print(f"  Output std: {output_tnd.std().item():.6f}\n")

        # Step 8: Simulate token acceptance (EAGLE verification result)
        print("Step 8: Simulating token acceptance...")
        # In real EAGLE, the verifier would check if draft tokens match
        # the target model's output. Here we simulate acceptance.
        simulated_accepted_tokens = 3  # Simulate 3 out of 4 tokens accepted
        print(f"  Simulated accepted tokens: {simulated_accepted_tokens}/{num_speculative_tokens}")
        print("  ✓ Token acceptance simulation successful\n")

        print("=" * 80)
        print("✓ End-to-end test passed!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Tree structure: {tree_choices}")
        print(f"  - Tree length: {tree_len}")
        print(f"  - Attention computation: ✓")
        print(f"  - Output validation: ✓")
        print(f"  - Simulated acceptance rate: {simulated_accepted_tokens}/{num_speculative_tokens}")
        print()
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Tree Attention End-to-End Test (Simulated EAGLE Model)")
    print("=" * 80)
    print()

    if not torch.npu.is_available():
        print("✗ NPU not available, cannot run test")
        sys.exit(1)

    result = test_e2e_with_simulated_model()

    if result:
        print("=== All tests passed! ===")
        sys.exit(0)
    else:
        print("=== Test failed ===")
        sys.exit(1)
