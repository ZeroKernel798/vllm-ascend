"""
Simple test for tree attention implementation.
Tests the 2D bias to 4D mask conversion and basic attention computation.
"""

import pytest
import torch
import torch_npu


def test_2d_bias_to_4d_mask():
    """Test converting 2D attention bias to 4D mask."""
    from vllm_ascend.attention.tree_attention_v1 import (
        AscendTreeAttentionMetadataBuilder,
    )

    # Create a simple 2D bias matrix
    # Shape: [num_tokens, num_tokens]
    num_tokens = 5
    attn_bias = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)

    # Simulate a simple causal mask: each token can only attend to itself and previous tokens
    for i in range(num_tokens):
        for j in range(i + 1):
            attn_bias[i, j] = 1.0

    # Convert to 4D mask
    # The builder should convert this to a 4D mask with shape [1, 1, num_tokens, num_tokens]
    mask_4d = AscendTreeAttentionMetadataBuilder._convert_bias_to_4d_mask(
        attn_bias
    )

    # Check shape
    assert mask_4d.shape == (
        1,
        1,
        num_tokens,
        num_tokens,
    ), f"Expected shape (1, 1, {num_tokens}, {num_tokens}), got {mask_4d.shape}"

    # Check values
    # The mask should be 1.0 where attention is allowed, 0.0 otherwise
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j <= i:
                assert mask_4d[0, 0, i, j] == 1.0, f"Expected 1.0 at ({i}, {j}), got {mask_4d[0, 0, i, j]}"
            else:
                assert mask_4d[0, 0, i, j] == 0.0, f"Expected 0.0 at ({i}, {j}), got {mask_4d[0, 0, i, j]}"

    print("test_2d_bias_to_4d_mask passed!")


def test_tree_attention_metadata():
    """Test AscendTreeAttentionMetadata dataclass."""
    from vllm_ascend.attention.tree_attention_v1 import (
        AscendTreeAttentionMetadata,
    )

    metadata = AscendTreeAttentionMetadata(
        attn_bias=torch.zeros((5, 5)),
        tree_attn_mask_4d=torch.zeros((1, 1, 5, 5)),
        use_tree_attention=True,
        tree_context_len=3,
    )

    assert metadata.use_tree_attention == True
    assert metadata.tree_context_len == 3
    assert metadata.tree_attn_mask_4d.shape == (1, 1, 5, 5)

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

    # Create a simple 4D mask (causal mask)
    mask_4d = torch.ones((1, 1, num_tokens, num_tokens), dtype=torch.float32, device="npu")
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j > i:
                mask_4d[0, 0, i, j] = 0.0

    # Reshape to BNSD layout
    query_bnsd = query.view(num_tokens, num_heads, head_size).unsqueeze(0)
    key_bnsd = key.view(num_tokens, num_heads, head_size).unsqueeze(0)
    value_bnsd = value.view(num_tokens, num_heads, head_size).unsqueeze(0)

    # Compute attention using npu_fused_infer_attention_score
    output = torch.randn((1, num_heads, num_tokens, head_size), dtype=torch.float32, device="npu")
    softmax_lse = torch.empty((1, num_heads, num_tokens), dtype=torch.float32, device="npu")

    torch_npu.npu_fused_infer_attention_score.out(
        query=query_bnsd,
        key=key_bnsd,
        value=value_bnsd,
        atten_mask=mask_4d,
        block_table=None,
        input_layout="BNSD",
        block_size=0,
        actual_seq_lengths=[num_tokens],
        actual_seq_lengths_kv=[num_tokens],
        num_key_value_heads=num_heads,
        num_heads=num_heads,
        scale=1.0 / (head_size**0.5),
        sparse_mode=0,
        pre_tokens=65535,
        next_tokens=65535,
        out=[output, softmax_lse],
    )

    # Check output shape
    assert output.shape == (1, num_heads, num_tokens, head_size)

    print("test_tree_attention_computation passed!")


if __name__ == "__main__":
    test_2d_bias_to_4d_mask()
    test_tree_attention_metadata()
    if torch.npu.is_available():
        test_tree_attention_computation()
    print("All tests passed!")
