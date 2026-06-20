"""
Simple test for tree attention implementation.
Tests the 2D bias to 2D mask conversion and basic attention computation.
"""

import pytest
import torch
import torch_npu


def test_2d_bias_to_2d_mask():
    """Test converting 2D attention bias to 2D mask."""
    from vllm_ascend.attention.tree_attention_v1 import (
        AscendTreeAttentionMetadataBuilder,
    )

    # Create a simple 2D bias matrix
    # Shape: [num_tokens, num_tokens]
    num_tokens = 5
    attn_bias = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)

    # Simulate a simple causal mask: each token can only attend to itself and previous tokens
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            attn_bias[i, j] = float("-inf")

    # Convert to 2D mask (ND layout)
    # The builder should convert this to a 2D mask with shape [num_tokens, num_tokens]
    mask_2d = AscendTreeAttentionMetadataBuilder._convert_bias_to_2d_mask(
        attn_bias, dtype=torch.int8
    )

    # Check shape
    assert mask_2d.shape == (
        num_tokens,
        num_tokens,
    ), f"Expected shape ({num_tokens}, {num_tokens}), got {mask_2d.shape}"

    # Check values
    # The mask should be 0 where attention is allowed, 1 where masked
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j <= i:
                assert mask_2d[i, j] == 0, f"Expected 0 at ({i}, {j}), got {mask_2d[i, j]}"
            else:
                assert mask_2d[i, j] == 1, f"Expected 1 at ({i}, {j}), got {mask_2d[i, j]}"

    print("test_2d_bias_to_2d_mask passed!")


def test_tree_attention_metadata():
    """Test AscendTreeAttentionMetadata dataclass."""
    from vllm_ascend.attention.tree_attention_v1 import (
        AscendTreeAttentionMetadata,
    )

    metadata = AscendTreeAttentionMetadata(
        attn_bias=torch.zeros((5, 5)),
        tree_attn_mask=torch.zeros((5, 5), dtype=torch.int8),
        use_tree_attention=True,
        tree_context_len=3,
    )

    assert metadata.use_tree_attention == True
    assert metadata.tree_context_len == 3
    assert metadata.tree_attn_mask.shape == (5, 5)
    assert metadata.tree_attn_mask.dtype == torch.int8

    print("test_tree_attention_metadata passed!")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_tree_attention_computation():
    """Test basic attention computation with tree attention mask on NPU."""
    import os

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

    torch.npu.set_device(0)

    # Create test tensors
    num_tokens = 5
    num_heads = 4
    head_size = 16

    query = torch.randn((num_tokens, num_heads * head_size), device="npu")
    key = torch.randn((num_tokens, num_heads * head_size), device="npu")
    value = torch.randn((num_tokens, num_heads * head_size), device="npu")

    # Create a simple 2D mask (causal mask) in ND layout
    # 0 = visible, 1 = masked
    mask_2d = torch.zeros((num_tokens, num_tokens), dtype=torch.int8, device="npu")
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j > i:
                mask_2d[i, j] = 1

    # Reshape to ND layout: [seq_len, num_heads, head_size]
    query_nd = query.view(num_tokens, num_heads, head_size)
    key_nd = key.view(num_tokens, num_heads, head_size)
    value_nd = value.view(num_tokens, num_heads, head_size)

    # Compute attention using npu_fused_infer_attention_score with ND layout
    output = torch.randn((num_tokens, num_heads, head_size), dtype=torch.float16, device="npu")
    softmax_lse = torch.empty((1, num_heads, num_tokens), dtype=torch.float16, device="npu")

    torch_npu.npu_fused_infer_attention_score.out(
        query=query_nd,
        key=key_nd,
        value=value_nd,
        atten_mask=mask_2d,
        block_table=None,
        input_layout="ND",
        block_size=0,
        actual_seq_lengths=[num_tokens],
        actual_seq_lengths_kv=[num_tokens],
        num_key_value_heads=num_heads,
        num_heads=num_heads,
        softmax_scale=1.0 / (head_size**0.5),
        sparse_mode=0,
        pre_tokens=65535,
        next_tokens=65535,
        out=[output, softmax_lse],
    )

    # Check output shape
    assert output.shape == (num_tokens, num_heads, head_size)

    print("test_tree_attention_computation passed!")


if __name__ == "__main__":
    test_2d_bias_to_2d_mask()
    test_tree_attention_metadata()
    if torch.npu.is_available():
        test_tree_attention_computation()
    print("All tests passed!")
