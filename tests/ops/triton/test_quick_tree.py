#!/usr/bin/env python3
"""Quick correctness test for 2D gather tree attention."""
import torch
from vllm_ascend.ops.triton.unified_attention import tree_unified_attention_varlen

device = torch.device("npu")
T, H, D, BS = 3, 2, 128, 32
s = D ** -0.5
torch.manual_seed(42)

q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
k = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
v = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
kc = torch.zeros(1, BS, H, D, device=device, dtype=torch.bfloat16)
vc = torch.zeros(1, BS, H, D, device=device, dtype=torch.bfloat16)
for t in range(min(T, BS)):
    kc[0, t] = k[t]
    vc[0, t] = v[t]

bt = torch.zeros(1, 1, dtype=torch.int32, device=device)
cu = torch.tensor([0, T], dtype=torch.int32, device=device)
sl = torch.tensor([T], dtype=torch.int32, device=device)
cl = torch.tensor([1], dtype=torch.int32, device=device)
bias = torch.zeros(T, T, dtype=torch.float32, device=device)

out = tree_unified_attention_varlen(
    q=q, k_cache=kc, v_cache=vc, block_table=bt,
    cu_seqlens_q=cu, seq_lens=sl, context_lens=cl,
    max_query_len=T, qq_bias=bias, scale=s, block_size=BS, num_kv_heads=H,
)

# Reference: einsum attention with causal + prefix mask
qc, kc2, vc2 = q.float().cpu(), k.float().cpu(), v.float().cpu()
scores = torch.einsum("thd,Thd->thT", qc, kc2) * s
for i in range(T):
    for j in range(T):
        if j >= 1 and i < (j - 1):
            scores[i, :, j] = -1e9
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum("thT,Thd->thd", probs, vc2)
diff = (out.float().cpu() - ref).abs().max().item()
print(f"max_abs_diff={diff:.6f}  {'PASS' if diff < 6e-2 else 'FAIL'}")
