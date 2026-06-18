"""Simple test for tree attention 4D mask conversion."""

import torch
from vllm_ascend.attention.tree_attention_v1 import (
    AscendTreeAttentionMetadataBuilder,
)


class MockSpeculativeConfig:
    def __init__(self):
        self.speculative_token_tree = [[0], [0, 1], [0, 1, 2]]


class MockVllmConfig:
    def __init__(self):
        self.speculative_config = MockSpeculativeConfig()


def test_convert_bias_to_4d_mask():
    """Test converting 2D bias to 4D mask."""
    print("Test: Converting 2D bias to 4D mask...")

    # Create builder
    mock_config = MockVllmConfig()
    builder = AscendTreeAttentionMetadataBuilder(mock_config)

    # Create a simple 2D bias matrix
    # 0 = visible, -inf = masked
    num_tokens = 5
    tree_attn_bias = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            tree_attn_bias[i, j] = float("-inf")

    print(f"  Input shape: {tree_attn_bias.shape}")

    # Convert to 4D mask
    mask_4d = builder._convert_bias_to_4d_mask(
        tree_attn_bias, dtype=torch.float16
    )

    print(f"  Output shape: {mask_4d.shape}")

    # Check shape
    expected_shape = (1, 1, num_tokens, num_tokens)
    assert (
        mask_4d.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {mask_4d.shape}"
    print("  Shape correct!")

    # Verify values
    print("  Verifying values...")
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j <= i:
                # Should be 0 (visible)
                assert mask_4d[0, 0, i, j] == 0.0, (
                    f"Expected 0.0 at ({i}, {j}), "
                    f"got {mask_4d[0, 0, i, j]}"
                )
            else:
                # Should be -inf (masked)
                assert torch.isinf(mask_4d[0, 0, i, j]), (
                    f"Expected -inf at ({i}, {j}), "
                    f"got {mask_4d[0, 0, i, j]}"
                )
    print("  Values correct!")

    print("Test passed!\n")


if __name__ == "__main__":
    test_convert_bias_to_4d_mask()
    print("All tests passed!")
