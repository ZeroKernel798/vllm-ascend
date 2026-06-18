"""Test tree attention computation on NPU."""

import torch
import torch_npu


def test_attention_with_4d_mask():
    """Test attention computation with 4D tree mask on NPU."""
    print("Test: Attention computation with 4D tree mask on NPU...")

    # Set NPU device
    torch.npu.set_device(0)

    # Create test tensors
    num_tokens = 5
    num_heads = 4
    head_size = 16
    dtype = torch.float16  # NPU requires float16 or bfloat16

    query = torch.randn(
        (num_tokens, num_heads * head_size), dtype=dtype, device="npu"
    )
    key = torch.randn(
        (num_tokens, num_heads * head_size), dtype=dtype, device="npu"
    )
    value = torch.randn(
        (num_tokens, num_heads * head_size), dtype=dtype, device="npu"
    )

    # Create a simple 4D mask (causal mask)
    # Shape: [1, 1, num_tokens, num_tokens]
    # NPU supports bool, int8, uint8 for mask
    mask_4d = torch.ones(
        (1, 1, num_tokens, num_tokens), dtype=torch.bool, device="npu"
    )
    # Set masked positions to False (will be converted internally)
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j > i:
                mask_4d[0, 0, i, j] = False
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j > i:
                mask_4d[0, 0, i, j] = float("-inf")

    print(f"  Query shape: {query.shape}")
    print(f"  Key shape: {key.shape}")
    print(f"  Mask shape: {mask_4d.shape}")

    # Reshape to BNSD layout
    # [num_tokens, num_heads * head_size] -> [1, num_heads, num_tokens, head_size]
    # Step 1: [num_tokens, num_heads, head_size]
    # Step 2: permute to [num_heads, num_tokens, head_size]
    # Step 3: unsqueeze to [1, num_heads, num_tokens, head_size]
    query_bnsd = query.view(num_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    key_bnsd = key.view(num_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)
    value_bnsd = value.view(num_tokens, num_heads, head_size).permute(1, 0, 2).unsqueeze(0)

    print(f"  Query BNSD shape: {query_bnsd.shape}")
    print(f"  Key BNSD shape: {key_bnsd.shape}")

    # Compute attention using npu_fused_infer_attention_score
    output = torch.randn(
        (1, num_heads, num_tokens, head_size),
        dtype=dtype,
        device="npu",
    )
    softmax_lse = torch.empty(
        (1, num_heads, num_tokens), dtype=dtype, device="npu"
    )

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

    print(f"  Output shape: {output.shape}")

    # Check output shape (BNSD layout)
    expected_shape = (1, num_heads, num_tokens, head_size)
    assert output.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

    print("  Output shape correct!")

    # Reshape output back to TND layout
    # [1, num_heads, num_tokens, head_size] -> [num_tokens, num_heads * head_size]
    # First squeeze batch dim, then permute to [num_tokens, num_heads, head_size], then flatten
    output_tnd = output.squeeze(0).permute(1, 0, 2).contiguous().view(num_tokens, num_heads * head_size)
    print(f"  Output TND shape: {output_tnd.shape}")

    print("Test passed!\n")


if __name__ == "__main__":
    test_attention_with_4d_mask()
    print("All tests passed!")
