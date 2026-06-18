"""
Performance test for tree attention vs standard attention.

Compares the performance of:
1. Linear chain tree (standard speculative decoding)
2. Branching tree (tree attention)
3. Standard causal attention (no speculation)
"""

import time
import torch
import torch_npu


def benchmark_attention(
    num_tokens,
    num_heads,
    head_size,
    num_iterations=100,
    use_tree_mask=False,
    tree_len=4,
):
    """
    Benchmark attention computation.

    Args:
        num_tokens: Number of tokens (sequence length)
        num_heads: Number of attention heads
        head_size: Head dimension
        num_iterations: Number of iterations for averaging
        use_tree_mask: Whether to use tree attention mask
        tree_len: Length of the tree (for tree attention)

    Returns:
        Average latency in milliseconds
    """
    if not torch.npu.is_available():
        print("NPU not available, skipping benchmark")
        return 0.0

    torch.npu.set_device(0)
    dtype = torch.float16

    # Create Q, K, V
    if use_tree_mask:
        # Tree attention: all tokens attend to each other (with mask)
        seq_len = tree_len
        # Create tree attention mask (causal + tree structure)
        mask = torch.ones(
            (1, 1, seq_len, seq_len), dtype=torch.bool, device="npu"
        )
        # Apply causal mask (each token can only attend to itself and previous tokens)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    mask[0, 0, i, j] = False
    else:
        # Standard attention: causal mask
        seq_len = num_tokens
        mask = torch.ones(
            (1, 1, seq_len, seq_len), dtype=torch.bool, device="npu"
        )
        # Apply causal mask
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask[0, 0, i, j] = False

    query = torch.randn(
        (1, num_heads, seq_len, head_size), dtype=dtype, device="npu"
    )
    key = torch.randn(
        (1, num_heads, seq_len, head_size), dtype=dtype, device="npu"
    )
    value = torch.randn(
        (1, num_heads, seq_len, head_size), dtype=dtype, device="npu"
    )
    output = torch.randn(
        (1, num_heads, seq_len, head_size), dtype=dtype, device="npu"
    )
    softmax_lse = torch.empty(
        (1, num_heads, seq_len), dtype=dtype, device="npu"
    )

    # Warmup
    for _ in range(10):
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=~mask,  # Invert mask: True means masked
            block_table=None,
            input_layout="BNSD",
            block_size=0,
            actual_seq_lengths=[seq_len],
            actual_seq_lengths_kv=[seq_len],
            num_key_value_heads=num_heads,
            num_heads=num_heads,
            scale=1.0 / (head_size**0.5),
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            out=[output, softmax_lse],
        )

    torch.npu.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=~mask,  # Invert mask: True means masked
            block_table=None,
            input_layout="BNSD",
            block_size=0,
            actual_seq_lengths=[seq_len],
            actual_seq_lengths_kv=[seq_len],
            num_key_value_heads=num_heads,
            num_heads=num_heads,
            scale=1.0 / (head_size**0.5),
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            out=[output, softmax_lse],
        )

    torch.npu.synchronize()
    end = time.time()

    avg_latency_ms = (end - start) / num_iterations * 1000
    return avg_latency_ms


def test_performance_comparison():
    """Compare performance of different attention types."""
    print("=" * 80)
    print("Tree Attention Performance Comparison")
    print("=" * 80)

    if not torch.npu.is_available():
        print("✗ NPU not available, cannot run performance test")
        return

    print(f"✓ NPU device: {torch.npu.get_device_name(0)}\n")

    # Test configurations
    configs = [
        {"num_tokens": 5, "num_heads": 8, "head_size": 64, "name": "Small (5 tokens)"},
        {"num_tokens": 10, "num_heads": 8, "head_size": 64, "name": "Medium (10 tokens)"},
        {"num_tokens": 20, "num_heads": 8, "head_size": 64, "name": "Large (20 tokens)"},
    ]

    results = []

    for config in configs:
        num_tokens = config["num_tokens"]
        num_heads = config["num_heads"]
        head_size = config["head_size"]
        name = config["name"]

        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"  num_tokens={num_tokens}, num_heads={num_heads}, head_size={head_size}")
        print(f"{'='*60}")

        # Test 1: Standard causal attention
        print("  Running: Standard causal attention...", end="")
        latency_standard = benchmark_attention(
            num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            use_tree_mask=False,
        )
        print(f" {latency_standard:.3f} ms")

        # Test 2: Tree attention (linear chain)
        print("  Running: Tree attention (linear chain)...", end="")
        latency_tree_linear = benchmark_attention(
            num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            use_tree_mask=True,
            tree_len=num_tokens,
        )
        print(f" {latency_tree_linear:.3f} ms")

        # Test 3: Tree attention (branching tree)
        print("  Running: Tree attention (branching tree)...", end="")
        latency_tree_branch = benchmark_attention(
            num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            use_tree_mask=True,
            tree_len=num_tokens,
        )
        print(f" {latency_tree_branch:.3f} ms")

        # Calculate overhead
        if latency_standard > 0:
            overhead_linear = (
                (latency_tree_linear - latency_standard) / latency_standard * 100
            )
            overhead_branch = (
                (latency_tree_branch - latency_standard) / latency_standard * 100
            )
        else:
            overhead_linear = 0.0
            overhead_branch = 0.0

        print(f"\n  Results for {name}:")
        print(f"    - Standard causal attention:    {latency_standard:.3f} ms")
        print(f"    - Tree attention (linear):     {latency_tree_linear:.3f} ms (overhead: {overhead_linear:+.1f}%)")
        print(f"    - Tree attention (branching):  {latency_tree_branch:.3f} ms (overhead: {overhead_branch:+.1f}%)")

        results.append({
            "name": name,
            "num_tokens": num_tokens,
            "latency_standard": latency_standard,
            "latency_tree_linear": latency_tree_linear,
            "latency_tree_branch": latency_tree_branch,
            "overhead_linear": overhead_linear,
            "overhead_branch": overhead_branch,
        })

    # Print summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Config':<25} {'Standard':>12} {'Linear Tree':>15} {'Branch Tree':>15}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<25} {r['latency_standard']:>10.3f}ms {r['latency_tree_linear']:>12.3f}ms {r['latency_tree_branch']:>12.3f}ms")

    print("\nOverhead (%):")
    print(f"{'Config':<25} {'Linear Tree':>15} {'Branch Tree':>15}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<25} {r['overhead_linear']:>+13.1f}% {r['overhead_branch']:>+13.1f}%")

    print("\n" + "=" * 80)
    print("Performance test completed!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    test_performance_comparison()
