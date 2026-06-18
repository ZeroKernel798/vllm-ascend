# Speculative Token Tree Performance Report

## Test Environment

- **Device**: Ascend910B2C
- **Test Date**: 2026-06-18
- **vLLM Ascend Version**: v0.12.0 (with speculative token tree support)

## Test Methodology

We compared the performance of three attention computation methods:

1. **Standard causal attention**: Traditional causal attention without speculation
2. **Tree attention (linear chain)**: Tree attention with linear chain structure (equivalent to standard speculative decoding)
3. **Tree attention (branching tree)**: Tree attention with branching tree structure

Each test ran 100 iterations after 10 warmup iterations. Latency is reported in milliseconds (ms).

## Test Results

### Configuration 1: Small (5 tokens)

| Method | Latency (ms) | Overhead (%) |
|--------|----------------|--------------|
| Standard causal attention | 0.022 | - |
| Tree attention (linear) | 0.022 | -2.1% |
| Tree attention (branching) | 0.022 | -2.0% |

### Configuration 2: Medium (10 tokens)

| Method | Latency (ms) | Overhead (%) |
|--------|----------------|--------------|
| Standard causal attention | 0.023 | - |
| Tree attention (linear) | 0.021 | -5.8% |
| Tree attention (branching) | 0.022 | -5.2% |

### Configuration 3: Large (20 tokens)

| Method | Latency (ms) | Overhead (%) |
|--------|----------------|--------------|
| Standard causal attention | 0.025 | - |
| Tree attention (linear) | 0.024 | -3.0% |
| Tree attention (branching) | 0.024 | -3.5% |

## Performance Summary

```
Config                        Standard     Linear Tree     Branch Tree
--------------------------------------------------------------------------------
Small (5 tokens)               0.022ms        0.022ms        0.022ms
Medium (10 tokens)             0.023ms        0.021ms        0.022ms
Large (20 tokens)              0.025ms        0.024ms        0.024ms

Overhead (%):
Config                        Linear Tree     Branch Tree
--------------------------------------------------------------------------------
Small (5 tokens)                   -2.1%          -2.0%
Medium (10 tokens)                 -5.8%          -5.2%
Large (20 tokens)                  -3.0%          -3.5%
```

## Key Findings

1. **Minimal Performance Overhead**: Tree attention (both linear and branching) has negligible performance overhead compared to standard causal attention. In some cases, tree attention is even slightly faster (likely due to measurement variance).

2. **No Significant Difference Between Linear and Branching Trees**: The performance difference between linear chain trees and branching trees is minimal (> 1ms across all test configurations).

3. **Scalability**: As the number of tokens increases from 5 to 20, the absolute latency increases only slightly (from 0.022ms to 0.025ms for standard attention).

## Conclusion

The Ascend NPU implementation of tree attention is highly optimized and introduces minimal performance overhead. Users can safely use branching tree structures for speculative decoding without significant performance degradation.

## Recommendations

1. **Use branching trees when appropriate**: Since the performance overhead is minimal, users should feel free to use branching tree structures to potentially improve token acceptance rates.

2. **Profile your specific workload**: While the micro-benchmark shows minimal overhead, users should profile their specific models and workloads to understand the end-to-end impact.

3. **Monitor acceptance rates**: The primary benefit of branching trees is higher token acceptance rates, which can improve overall inference throughput even if individual attention computations are slightly slower.

## Test Script

The performance test script is available at:
`tests/ut/attention/test_tree_attention_performance.py`

To run the test on your own NPU device:

```bash
cd /path/to/vllm-ascend
source .venv/bin/activate
python tests/ut/attention/test_tree_attention_performance.py
```

## Limitations

This micro-benchmark only measures the attention computation latency. A complete performance evaluation should also consider:

1. **End-to-end inference latency**: Time from input to final output
2. **Token acceptance rates**: How many speculative tokens are accepted by the verifier
3. **Memory overhead**: Additional memory required for tree attention metadata
4. **Batch size impact**: Performance with multiple concurrent requests

These metrics should be evaluated in future tests.
