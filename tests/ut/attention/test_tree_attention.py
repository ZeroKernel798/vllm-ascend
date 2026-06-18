# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Unit tests for tree attention implementation."""

import pytest
import torch

from vllm_ascend.attention.tree_attention_impl import tree_attention_verify
from vllm_ascend.attention.tree_attention_v1 import (
    AscendTreeAttentionMetadata,
    AscendTreeAttentionMetadataBuilder,
)
from vllm_ascend.speculative_token_tree import (
    SpeculativeTokenTreePlan,
    build_speculative_token_tree_plan,
    prepare_speculative_token_tree_attn_bias,
)


def create_test_tree_mask(tree_len: int, device: torch.device) -> torch.Tensor:
    """Create a test tree attention mask (4D, BNSD layout).
    
    Args:
        tree_len: Length of the tree.
        device: Device to create tensor on.
        
    Returns:
        4D tree attention mask, shape [1, 1, tree_len, tree_len].
    """
    # Create a simple linear causal mask as placeholder
    # In real scenario, this should be the actual tree structure mask
    mask = torch.ones((1, 1, tree_len, tree_len), device=device, dtype=torch.float16)
    
    # Set causal mask: each token can only attend to itself and previous tokens
    for i in range(tree_len):
        for j in range(tree_len):
            if j <= i:
                mask[0, 0, i, j] = 0.0  # Visible
            else:
                mask[0, 0, i, j] = float("-inf")  # Masked
    
    return mask


def test_tree_attention_verify_basic():
    """Test basic tree attention verification."""
    # Setup
    num_tokens = 10
    num_heads = 8
    num_kv_heads = 8
    head_size = 64
    device = torch.device("cpu")  # Use CPU for testing
    
    # Create test tensors
    query = torch.randn(num_tokens, num_heads * head_size, device=device, dtype=torch.float16)
    key = torch.randn(num_tokens, num_kv_heads * head_size, device=device, dtype=torch.float16)
    value = torch.randn(num_tokens, num_kv_heads * head_size, device=device, dtype=torch.float16)
    
    # Create tree attention mask
    tree_attn_mask_4d = create_test_tree_mask(num_tokens, device)
    
    # Call tree attention
    output = tree_attention_verify(
        query=query,
        key=key,
        value=value,
        tree_attn_mask_4d=tree_attn_mask_4d,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=1.0 / (head_size ** 0.5),
    )
    
    # Verify output shape
    assert output.shape == (num_tokens, num_heads * head_size), (
        f"Expected output shape ({num_tokens}, {num_heads * head_size}), "
        f"got {output.shape}"
    )


def test_tree_attention_verify_gqa():
    """Test tree attention with GQA (Grouped Query Attention)."""
    # Setup: GQA with 8 query heads and 4 KV heads
    num_tokens = 10
    num_heads = 8
    num_kv_heads = 4
    head_size = 64
    device = torch.device("cpu")
    
    # Create test tensors
    query = torch.randn(num_tokens, num_heads * head_size, device=device, dtype=torch.float16)
    key = torch.randn(num_tokens, num_kv_heads * head_size, device=device, dtype=torch.float16)
    value = torch.randn(num_tokens, num_kv_heads * head_size, device=device, dtype=torch.float16)
    
    # Create tree attention mask
    tree_attn_mask_4d = create_test_tree_mask(num_tokens, device)
    
    # Call tree attention
    output = tree_attention_verify(
        query=query,
        key=key,
        value=value,
        tree_attn_mask_4d=tree_attn_mask_4d,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=1.0 / (head_size ** 0.5),
    )
    
    # Verify output shape
    assert output.shape == (num_tokens, num_heads * head_size), (
        f"Expected output shape ({num_tokens}, {num_heads * head_size}), "
        f"got {output.shape}"
    )


def test_tree_attention_metadata_builder():
    """Test AscendTreeAttentionMetadataBuilder."""
    # Create a mock vllm_config
    class MockSpeculativeConfig:
        def __init__(self):
            self.speculative_token_tree = [
                [0],  # Token 1: parent is root (0)
                [0, 1],  # Token 2: parent is token 1
                [0],  # Token 3: parent is root (0)
                [0, 1, 2],  # Token 4: parent is token 2
            ]
    
    class MockVllmConfig:
        def __init__(self):
            self.speculative_config = MockSpeculativeConfig()
    
    vllm_config = MockVllmConfig()
    
    # Build tree attention metadata
    builder = AscendTreeAttentionMetadataBuilder(vllm_config)
    seq_lens = [20]  # Single request with 20 tokens
    
    metadata = builder.build(seq_lens)
    
    # Verify metadata
    assert metadata is not None, "Tree attention metadata should not be None"
    assert metadata.tree_attn_mask is not None, "Tree attention mask should not be None"
    assert metadata.tree_attn_mask.shape == (1, 1, 4, 4), (
        f"Expected tree mask shape (1, 1, 4, 4), got {metadata.tree_attn_mask.shape}"
    )


def test_prepare_speculative_token_tree_attn_bias():
    """Test prepare_speculative_token_tree_attn_bias function."""
    # Create a simple tree
    tree_choices = [
        [0],  # Token 1: parent is root (0)
        [0, 1],  # Token 2: parent is token 1
        [0],  # Token 3: parent is root (0)
        [0, 1, 2],  # Token 4: parent is token 2
    ]
    
    # Prepare attention bias
    attn_bias = prepare_speculative_token_tree_attn_bias(tree_choices)
    
    # Verify bias shape
    tree_len = len(tree_choices)
    assert attn_bias.shape == (tree_len, tree_len), (
        f"Expected bias shape ({tree_len}, {tree_len}), got {attn_bias.shape}"
    )
    
    # Verify bias values: 0 for visible, -inf for masked
    # Token 0 can attend to itself
    assert attn_bias[0, 0] == 0, "Token 0 should be able to attend to itself"
    
    # Token 1 can attend to itself and token 0
    assert attn_bias[1, 1] == 0, "Token 1 should be able to attend to itself"
    assert attn_bias[1, 0] == 0, "Token 1 should be able to attend to token 0"


if __name__ == "__main__":
    # Run tests
    test_tree_attention_verify_basic()
    print("✓ test_tree_attention_verify_basic passed")
    
    test_tree_attention_verify_gqa()
    print("✓ test_tree_attention_verify_gqa passed")
    
    test_tree_attention_metadata_builder()
    print("✓ test_tree_attention_metadata_builder passed")
    
    test_prepare_speculative_token_tree_attn_bias()
    print("✓ test_prepare_speculative_token_tree_attn_bias passed")
    
    print("\n✅ All tree attention tests passed!")
