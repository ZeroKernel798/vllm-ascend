# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""Consolidated unit tests for speculative token tree attention.

Merged from:
  - test_speculative_token_tree.py
  - test_propose_tree.py
  - test_sparse_mode_routing.py
  - test_tree_attn_correctness.py
"""

import pytest
import torch




from vllm_ascend.spec_decode.speculative_token_tree import (
    attach_speculative_tree_metadata,
    get_speculative_tree_len,
    prepare_speculative_token_tree_attn_bias,
    _compute_drafts_per_level,
    _compute_child_counts,
    _slice_tree_attn_bias,
    parse_speculative_token_tree,
    sort_speculative_token_tree,
)


class MockSpeculativeConfig:
    def __init__(self, num_speculative_tokens=3, speculative_token_tree=None):
        self.num_speculative_tokens = num_speculative_tokens
        self.speculative_token_tree = speculative_token_tree


class MockVllmConfig:
    def __init__(self, num_speculative_tokens=3, speculative_token_tree=None):
        self.speculative_config = MockSpeculativeConfig(num_speculative_tokens, speculative_token_tree)


# =====================================================================
# _compute_drafts_per_level
# =====================================================================

@pytest.mark.parametrize("tree,exp_drafts,exp_cu", [
    ([(0,), (0,0), (0,1)],          [1, 2],    [0, 1, 3]),
    ([(0,), (0,0), (0,0,0)],        [1, 1, 1], [0, 1, 2, 3]),
    ([tuple(0 for _ in range(d)) for d in range(1, 6)],
                                     [1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5]),
])
def test_drafts_per_level(tree, exp_drafts, exp_cu):
    drafts, cu = _compute_drafts_per_level(tree)
    assert drafts == exp_drafts
    assert cu == exp_cu


# =====================================================================
# _compute_child_counts
# =====================================================================

@pytest.mark.parametrize("tree,exp", [
    ([(0,), (0,0), (0,1), (0,0,0)], [(1,), (2,), (1, 0)]),
    ([(0,), (0,0), (0,1)],          [(1,), (2,)]),
])
def test_child_counts(tree, exp):
    assert _compute_child_counts(tree) == exp


# =====================================================================
# prepare_speculative_token_tree_attn_bias
# =====================================================================

TREE_SIBLINGS = [(0,), (0,0), (0,1)]
TREE_ANCESTOR = [(0,), (0,0), (0,0,0)]
TREE_CROSS = [(0,), (0,0), (0,1), (0,0,0)]


def test_prepare_bias_siblings_masked():
    bias = prepare_speculative_token_tree_attn_bias(TREE_SIBLINGS)
    assert bias.shape == (4, 4)
    assert torch.isneginf(bias[2, 3])
    assert torch.isneginf(bias[3, 2])


def test_prepare_bias_ancestor_chain():
    bias = prepare_speculative_token_tree_attn_bias(TREE_ANCESTOR)
    # (0,0,0) at row 3 attends to root(0), (0,)(1), (0,0)(2), self(3)
    assert bias[3, 0] == 0.0
    assert bias[3, 1] == 0.0
    assert bias[3, 2] == 0.0
    assert bias[3, 3] == 0.0
    assert torch.isneginf(bias[0, 1])  # root can't attend to draft


def test_prepare_bias_cross_branch_isolation():
    bias = prepare_speculative_token_tree_attn_bias(TREE_CROSS)
    assert torch.isneginf(bias[2, 3])  # (0,0) ↔ (0,1) blocked
    assert torch.isneginf(bias[4, 3])  # (0,0,0) → (0,1) blocked


# =====================================================================
# _slice_tree_attn_bias
# =====================================================================

@pytest.mark.parametrize("tree,drafts,exp_shapes", [
    (TREE_SIBLINGS,         [1, 2],    [(1, 1), (2, 2)]),
    (TREE_CROSS,            [1, 2, 1], [(1, 1), (2, 2), (1, 1)]),
    ([(0,), (0,0)],         [0, 2],    [(0, 0), (2, 2)]),
])
def test_slice_tree_attn_bias(tree, drafts, exp_shapes):
    bias = prepare_speculative_token_tree_attn_bias(tree)
    slices = _slice_tree_attn_bias(bias, drafts)
    assert len(slices) == len(exp_shapes)
    for s, shape in zip(slices, exp_shapes):
        assert s.shape == shape


# =====================================================================
# parse + sort
# =====================================================================

def test_parse_and_sort_breadth_first():
    tree = [(0, 1), (0,), (0, 0)]
    parsed = parse_speculative_token_tree(tree)
    assert sort_speculative_token_tree(parsed) == [(0,), (0, 0), (0, 1)]


@pytest.mark.parametrize("bad_input,match", [
    ("not-a-list",                     "Invalid"),
    ('{"key": "value"}',               "list/tuple of tuple paths"),
    ([(0, "x")],                       ""),  # TypeError
    ([(0,), (0, -1)],                  "non-negative"),
    ([(0,), ()],                       "non-empty"),
])
def test_parse_rejects_invalid(bad_input, match):
    with pytest.raises((ValueError, TypeError), match=match if match else None):
        parse_speculative_token_tree(bad_input)


# =====================================================================
# get_speculative_tree_len
# =====================================================================

def _make_len_cfg(tree_len, num_tokens):
    sc = MockSpeculativeConfig(num_speculative_tokens=num_tokens)
    object.__setattr__(sc, "tree_len", tree_len)
    return sc


@pytest.mark.parametrize("config,exp", [
    (None,                                                     1),
    (_make_len_cfg(6, 5),                                      6),
    (MockSpeculativeConfig(num_speculative_tokens=3),          4),
    (MockSpeculativeConfig(num_speculative_tokens=0),          1),
    (MockSpeculativeConfig(num_speculative_tokens=None),       1),
])
def test_get_tree_len(config, exp):
    assert get_speculative_tree_len(config) == exp


# =====================================================================
# attach_speculative_tree_metadata
# =====================================================================

def test_attach_required_fields():
    cfg = MockVllmConfig(3, "[(0,), (0,0), (0,1)]")
    attach_speculative_tree_metadata(cfg)
    spec = cfg.speculative_config
    assert spec.tree_len == 4
    assert spec.tree_depth == 2
    assert spec.drafts_per_level == [1, 2]
    assert spec.cu_drafts_per_level == [0, 1, 3]
    assert spec.child_drafts_per_level == [[1], [2]]
    assert spec.tree_attn_bias.shape == (4, 4)
    assert len(spec.tree_attn_bias_slices) == 2


@pytest.mark.parametrize("cfg_factory", [
    lambda: type("V", (), {"speculative_config": None})(),
    lambda: type("V", (), {
        "speculative_config": MockSpeculativeConfig(num_speculative_tokens=None)
    })(),
])
def test_attach_skips(cfg_factory):
    attach_speculative_tree_metadata(cfg_factory())  # must not raise



"""
Tree drafting plan tests — adapted from upstream vLLM's
``tests/v1/spec_decode/test_eagle.py::test_propose_tree`` (snapshot
``be0dcc29d``, removed by PR #42121).

Upstream verifies that ``proposer.propose()`` with a ``speculative_token_tree``
expands draft tokens level-by-level using ``cu_drafts_per_level`` /
``child_drafts_per_level`` and returns ``batch_size × num_speculative_tokens``
draft tokens in the correct order.
"""

import ast
import types

import pytest
import torch

from vllm_ascend.spec_decode.speculative_token_tree import (
    attach_speculative_tree_metadata,
    prepare_speculative_token_tree_attn_bias,
    _compute_drafts_per_level,
    _compute_child_counts,
)


def _make_spec_config(tree_str, num_spec_tokens):
    sc = types.SimpleNamespace()
    sc.speculative_token_tree = tree_str
    sc.num_speculative_tokens = num_spec_tokens
    vllm_config = types.SimpleNamespace(speculative_config=sc)
    return vllm_config, sc


TREES = {
    "single": "[(0,)]",
    "chain": "[(0,), (0, 0), (0, 0, 0)]",
    "parallel": "[(0,), (1,), (2,)]",
    "tree": "[(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]",
}


@pytest.mark.parametrize("name", list(TREES.keys()))
def test_tree_drafting_plan(name):
    tree_str = TREES[name]
    raw = ast.literal_eval(tree_str)
    num_spec = len(raw)

    vllm_config, sc = _make_spec_config(tree_str, num_spec)
    attach_speculative_tree_metadata(vllm_config)

    assert sc.tree_len == num_spec + 1
    assert sum(sc.drafts_per_level) == num_spec

    expected_cu = [0]
    for c in sc.drafts_per_level:
        expected_cu.append(expected_cu[-1] + c)
    assert list(sc.cu_drafts_per_level) == expected_cu

    tree_depth = max(len(p) for p in raw)
    assert len(sc.drafts_per_level) == tree_depth
    assert len(sc.child_drafts_per_level) == tree_depth


@pytest.mark.parametrize("name", list(TREES.keys()))
def test_tree_attn_bias_shape_and_root(name):
    raw = ast.literal_eval(TREES[name])
    bias = prepare_speculative_token_tree_attn_bias(raw)
    tree_len = len(raw) + 1
    assert bias.shape == (tree_len, tree_len)
    assert torch.all(bias[:, 0] == 0)
    for i in range(tree_len):
        assert bias[i, i] == 0


def test_chain_plan_values():
    choices = [(0,), (0, 0), (0, 0, 0)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [1, 1, 1]
    assert cu == [0, 1, 2, 3]


def test_parallel_plan_values():
    choices = [(0,), (1,), (2,)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [3]
    assert cu == [0, 3]
    children = _compute_child_counts(choices)
    assert len(children) == 1


def test_branching_tree_plan_values():
    choices = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [3, 6]
    assert cu == [0, 3, 9]
    assert sum(drafts) == 9



"""Regression tests: sparse_mode routing for tree attention and sinks.

Verifies forward_fused_infer_attention kernel selection:

* ``is_tree_mask=True`` → ``sparse_mode=0`` (tree attention path).
* ``self.sinks is not None`` → V2 API, ``sparse_mode=3/4`` (bypasses tree).
* Source-level checks that documentation comments exist.

See ``attention_v1.py`` for inline comments documenting the sinks-tree
mutual exclusion.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch


@dataclass
class FakeMetadata:
    attn_mask: torch.Tensor | None = None
    is_tree_mask: bool = False
    attn_state: int = 2
    causal: bool = True
    num_actual_tokens: int = 2
    num_decode_tokens: int = 2
    num_prefills: int = 0
    num_decodes: int = 1
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None
    actual_seq_lengths_q: list[int] = None
    max_query_len: int = None
    query_start_loc: torch.Tensor = None
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    model_runner_type: str = "generate"
    kvcomp_metadata: object = None


def _make_metadata(is_tree: bool) -> FakeMetadata:
    tokens = 2
    return FakeMetadata(
        attn_mask=torch.eye(tokens) if is_tree else torch.ones(tokens, tokens),
        is_tree_mask=is_tree,
        seq_lens=torch.tensor([tokens]),
        seq_lens_cpu=torch.tensor([tokens]),
        seq_lens_list=[tokens],
        actual_seq_lengths_q=[tokens],
        max_query_len=tokens,
        query_start_loc=torch.tensor([0, tokens]),
        block_tables=torch.zeros(1, 64, dtype=torch.int32),
        slot_mapping=torch.zeros(tokens, dtype=torch.int64),
    )


class TestSparseModeRouting:
    """Verify is_tree_mask=True routes correctly (Triton or CANN based on flag)."""

    @staticmethod
    def _fake_fia(query, key, value, *, atten_mask=None, block_table=None,
                  input_layout=None, block_size=None, actual_seq_lengths=None,
                  actual_seq_lengths_kv=None, num_key_value_heads=None,
                  num_heads=None, scale=None, sparse_mode=None, **kwargs):
        return (torch.zeros(2, 8, 128), sparse_mode)

    @staticmethod
    def _fake_triton(q, k_cache, v_cache, block_table, seq_lens,
                     qq_bias=None, scale=None, block_size=None,
                     num_kv_heads=None, context_len=None, **kwargs):
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2])

    def _make_impl(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
        impl.num_heads = 8
        impl.head_size = 128
        impl.num_kv_heads = 8
        impl.scale = 1.0
        impl.sliding_window = None
        impl.sinks = None
        impl.attn_type = "DECODER"
        impl.vllm_config = MagicMock()
        impl.vllm_config.speculative_config = None
        impl.vllm_config.parallel_config.data_parallel_size = 1
        impl.vllm_config.parallel_config.enable_expert_parallel = False
        impl.vllm_config.parallel_config.pipeline_parallel_size = 1
        impl.vllm_config.parallel_config.data_parallel_size_local = 1
        impl.vllm_config.kv_transfer_config = None
        impl.vllm_config.compilation_config.cudagraph_mode = None
        impl.enable_hamming_sparse = False
        impl.layerIndex = 0
        # Key/value cache mocks for Triton path
        impl.key_cache = torch.randn(1, 16, 8, 128)   # [B, BS, Hkv, D]
        impl.value_cache = torch.randn(1, 16, 8, 128)
        return impl

    def _call_forward(self, meta, impl, patch_sliding=None):
        query = torch.randn(2, 8, 128)
        key = torch.randn(2, 8, 128)
        value = torch.randn(2, 8, 128)
        output = torch.zeros(2, 8, 128)
        if patch_sliding is not None:
            impl.sliding_window = patch_sliding

        from vllm_ascend.attention.attention_v1 import _USE_TRITON_TREE_ATTENTION
        if _USE_TRITON_TREE_ATTENTION and meta.is_tree_mask and len(meta.seq_lens_list) == 1:
            # Triton path: mock tree_unified_attention at its source module
            with patch.object(impl, "_get_fia_params",
                              return_value=(key, value, 128, meta.block_tables,
                                            torch.tensor([2, 2]))):
                with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX",
                           capturing=False):
                    with patch(
                        "vllm_ascend.ops.triton.unified_attention.tree_unified_attention_varlen",
                        side_effect=self._fake_triton,
                    ) as mock_triton:
                        impl.forward_fused_infer_attention(
                            query, key, value, meta, output,
                        )
            return mock_triton
        else:
            # CANN path
            with patch.object(impl, "_get_fia_params",
                              return_value=(key, value, 128, meta.block_tables,
                                            torch.tensor([2, 2]))):
                with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX",
                           capturing=False):
                    with patch("torch_npu.npu_fused_infer_attention_score",
                               side_effect=self._fake_fia) as mock_fia:
                        impl.forward_fused_infer_attention(
                            query, key, value, meta, output,
                        )
            return mock_fia

    def test_tree_mask_sparse_mode_zero(self):
        """is_tree_mask=True routes to tree attention (Triton or CANN sparse_mode=0)."""
        from vllm_ascend.attention.attention_v1 import _USE_TRITON_TREE_ATTENTION
        meta = _make_metadata(is_tree=True)
        mock_result = self._call_forward(meta, self._make_impl())
        if _USE_TRITON_TREE_ATTENTION:
            # Triton path: tree_unified_attention called with qq_bias
            mock_result.assert_called_once()
            assert mock_result.call_args.kwargs.get("qq_bias") is meta.attn_mask, (
                "Triton path must pass qq_bias=attn_mask"
            )
        else:
            # CANN path: sparse_mode=0
            mock_result.assert_called_once()
            assert mock_result.call_args.kwargs["sparse_mode"] == 0

    def test_causal_no_tree_sparse_mode_three(self):
        """is_tree_mask=False → sparse_mode=3 (always CANN path)."""
        meta = _make_metadata(is_tree=False)
        mock_fia = self._call_forward(meta, self._make_impl())
        mock_fia.assert_called_once()
        assert mock_fia.call_args.kwargs["sparse_mode"] == 3

    def test_non_causal_sparse_mode_zero(self):
        """non-causal + no tree → sparse_mode=0 (always CANN path)."""
        meta = _make_metadata(is_tree=False)
        meta.causal = False
        mock_fia = self._call_forward(meta, self._make_impl())
        mock_fia.assert_called_once()
        assert mock_fia.call_args.kwargs["sparse_mode"] == 0

    def test_tree_mask_beats_sliding_window(self):
        """is_tree_mask=True takes priority over sliding_window.
        With Triton: tree_unified_attention is called.
        With CANN: sparse_mode=0."""
        from vllm_ascend.attention.attention_v1 import _USE_TRITON_TREE_ATTENTION
        meta = _make_metadata(is_tree=True)
        mock_result = self._call_forward(meta, self._make_impl(), patch_sliding=128)
        if _USE_TRITON_TREE_ATTENTION:
            mock_result.assert_called_once()
            assert "qq_bias" in mock_result.call_args.kwargs, (
                "Triton path: tree_unified_attention must be called"
            )
        else:
            mock_result.assert_called_once()
            assert mock_result.call_args.kwargs["sparse_mode"] == 0, (
                "tree mask must take priority over sliding_window"
            )

    def test_sliding_window_no_tree_sparse_mode_four(self):
        """sliding_window=128 + no tree mask → sparse_mode=4."""
        meta = _make_metadata(is_tree=False)
        mock_fia = self._call_forward(meta, self._make_impl(), patch_sliding=128)
        mock_fia.assert_called_once()
        assert mock_fia.call_args.kwargs["sparse_mode"] == 4, (
            f"expected sparse_mode=4 for sliding window, got "
            f"{mock_fia.call_args.kwargs['sparse_mode']}"
        )

    def test_sinks_path_bypasses_tree_mask_check(self):
        """When self.sinks is set, the V2 API path is used — tree mask
        is NOT checked.  See the sinks-branch comment in attention_v1.py."""
        meta = _make_metadata(is_tree=True)
        impl = self._make_impl()
        impl.sinks = torch.ones(1)  # activate sinks path
        query = torch.randn(2, 8, 128)
        key = torch.randn(2, 8, 128)
        value = torch.randn(2, 8, 128)
        output = torch.zeros(2, 8, 128)
        with patch.object(impl, "_get_fia_params",
                          return_value=(key, value, 128, meta.block_tables,
                                        torch.tensor([2, 2]))):
            with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX",
                       capturing=False):
                with patch("torch_npu.npu_fused_infer_attention_score_v2",
                           return_value=(torch.zeros(2, 8, 128), None)) as mock_v2:
                    with patch("torch_npu.npu_fused_infer_attention_score") as mock_v1:
                        impl.forward_fused_infer_attention(
                            query, key, value, meta, output,
                        )
        # Sinks path: V2 is called, V1 is NOT
        mock_v1.assert_not_called()
        mock_v2.assert_called_once()

        # Verify V2 arguments
        kwargs = mock_v2.call_args.kwargs
        assert kwargs["sparse_mode"] == 3, (
            f"sinks path should use sparse_mode=3, got {kwargs['sparse_mode']}"
        )
        assert kwargs["learnable_sink"] is impl.sinks, "learnable_sink must be passed"
        # atten_mask is still passed (could be tree bias) but sparse_mode=3
        # means the kernel treats it as a causal mask — tree semantics lost
        assert kwargs["atten_mask"] is meta.attn_mask, (
            "atten_mask is still forwarded to V2 kernel"
        )

    def test_sinks_with_sliding_window_sparse_mode_four(self):
        """sinks=True + sliding_window → sparse_mode=4 in V2 API."""
        meta = _make_metadata(is_tree=True)
        impl = self._make_impl()
        impl.sinks = torch.ones(1)
        impl.sliding_window = 128
        query = torch.randn(2, 8, 128)
        key = torch.randn(2, 8, 128)
        value = torch.randn(2, 8, 128)
        output = torch.zeros(2, 8, 128)
        with patch.object(impl, "_get_fia_params",
                          return_value=(key, value, 128, meta.block_tables,
                                        torch.tensor([2, 2]))):
            with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX",
                       capturing=False):
                with patch("torch_npu.npu_fused_infer_attention_score_v2",
                           return_value=(torch.zeros(2, 8, 128), None)) as mock_v2:
                    impl.forward_fused_infer_attention(
                        query, key, value, meta, output,
                    )
        assert mock_v2.call_args.kwargs["sparse_mode"] == 4, (
            f"sinks+sliding should use sparse_mode=4, got "
            f"{mock_v2.call_args.kwargs['sparse_mode']}"
        )


# ===================================================================
# Source checks — ordering + inline documentation
# ===================================================================

class TestSourceOrderingAndDocs:
    """Verify branch ordering and that limitation comments exist."""

    def test_tree_check_before_sliding_in_source(self):
        import inspect
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        source = inspect.getsource(
            AscendAttentionBackendImpl.forward_fused_infer_attention
        )
        tree_line = sliding_line = -1
        for i, line in enumerate(source.split("\n")):
            if "is_tree_mask" in line and "if" in line:
                tree_line = i
            if "sliding_window" in line and line.strip().startswith("elif"):
                sliding_line = i
        assert tree_line >= 0, "is_tree_mask check not found"
        assert sliding_line >= 0, "sliding_window check not found"
        assert tree_line < sliding_line, (
            f"is_tree_mask (line {tree_line}) must appear before "
            f"sliding_window (line {sliding_line}) for correct priority"
        )

    def test_sinks_check_before_tree_check_in_source(self):
        """Verify 'self.sinks is not None' appears BEFORE 'is_tree_mask'.
        This means the sinks path completely bypasses tree attention."""
        import inspect
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        source = inspect.getsource(
            AscendAttentionBackendImpl.forward_fused_infer_attention
        )
        sinks_line = tree_line = -1
        for i, line in enumerate(source.split("\n")):
            if "sinks is not None" in line:
                sinks_line = i
            if "is_tree_mask" in line and "if" in line:
                tree_line = i
        assert sinks_line >= 0, "'self.sinks is not None' check not found"
        assert tree_line >= 0, "is_tree_mask check not found"
        assert sinks_line < tree_line, (
            f"sinks check (line {sinks_line}) must appear before "
            f"tree mask check (line {tree_line}) — sinks path bypasses tree attention"
        )

    def test_sinks_branch_has_limitation_comment(self):
        """The sinks branch has an inline comment documenting that tree
        attention is not supported here."""
        import inspect
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        source = inspect.getsource(
            AscendAttentionBackendImpl.forward_fused_infer_attention
        )
        assert "NOTE:" in source and "sinks" in source and "is_tree_mask" in source, (
            "sinks branch must document tree-attention limitation "
            "(expected NOTE comment referencing is_tree_mask)"
        )
        # More specific: "does NOT check is_tree_mask"
        assert "does NOT check is_tree_mask" in source.replace(
            "\n", " "
        ), (
            "expected comment 'does NOT check is_tree_mask' in sinks documentation"
        )

    def test_tree_branch_references_sinks_limitation(self):
        """The tree mask branch has a comment referencing the sinks limitation."""
        import inspect
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        source = inspect.getsource(
            AscendAttentionBackendImpl.forward_fused_infer_attention
        )
        assert "only reached when self.sinks is None" in source, (
            "tree mask branch must document sinks precedence"
        )



"""
Tree attention correctness test — port of upstream vLLM's
``tests/v1/spec_decode/test_tree_attention.py::test_tree_attn_correctness``
(snapshot ``be0dcc29d``, removed by PR #42121) to the Ascend Triton kernel.

Upstream idea (branch-vs-tree equivalence):
  1. Run the whole tree once through tree attention (with qq_bias).
  2. For each query node in the tree, take its visible branch (self + ancestors)
     per the tree mask, run it as a plain causal sequence through a reference
     attention, and assert the tree-attention output for that node matches.

Differences from upstream:
  * device ``npu`` instead of ``cuda``.
  * Tree attention runs through ``tree_unified_attention_multiseq`` (our
    in-house Triton-Ascend kernel) instead of the upstream TREE_ATTN backend.
  * Reference branch is computed with a dense PyTorch SDPA reference instead of
    a second vLLM backend.
  * Covers ``batch_size in [1, 16, 32]`` — exercises the num_seqs>1 path.
"""

import torch
import pytest

from vllm_ascend.ops.triton.unified_attention import (
    tree_unified_attention_multiseq,
)

# Upstream tree masks: row/col 0 is the implicit ROOT. 1 = visible, 0 = masked.
# tree_size_q = mask.shape[0] (includes root).
TREE_ATTN_MASKS = {
    # Chain: [(0,), (0,0), (0,0,0)]  -> 4x4 (root + 3)
    "chain": torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.int32,
    ),
    # Tree: [(0,), (1,), (0,0), (0,1), (1,0), (1,1)]  -> 7x7 (root + 6)
    "tree": torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
        ],
        dtype=torch.int32,
    ),
}


def _mask_to_bias(mask: torch.Tensor) -> torch.Tensor:
    """[T,T] 0/1 visibility -> 0/-inf additive bias (float32)."""
    bias = torch.zeros_like(mask, dtype=torch.float32)
    bias[mask == 0] = float("-inf")
    return bias


def _dense_ref_node(q_node, k_ctx, v_ctx, k_vis, v_vis, scale):
    """Dense SDPA reference for ONE tree query node.

    The node attends to: the full prefix context (all visible) + the set of
    tree tokens visible to it per the tree mask (``k_vis``/``v_vis``, all
    visible — they are exactly this node's ancestors + itself). No extra causal
    masking is needed because the visibility set is already the mask row.

    q_node:      [1, H, D]
    k_ctx/v_ctx: [ctx, Hkv, D]
    k_vis/v_vis: [V, Hkv, D]
    Returns: [1, H, D]
    """
    _, H, D = q_node.shape
    Hkv = k_ctx.shape[1]
    k = torch.cat([k_ctx, k_vis], dim=0)
    v = torch.cat([v_ctx, v_vis], dim=0)
    if Hkv < H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q_node.float(), k.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,khd->qhd", probs, v.float()).to(q_node.dtype)


@pytest.mark.parametrize("mask_name", list(TREE_ATTN_MASKS.keys()))
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(2, 2), (4, 2)])
@pytest.mark.parametrize("sequence_position", [16, 1024, 2048])
def test_tree_attn_correctness(
    mask_name, batch_size, num_heads, num_kv_heads, sequence_position
):
    """Branch-vs-tree equivalence on NPU, mirroring upstream."""
    device = torch.device("npu")
    torch.manual_seed(42)

    tree_mask = TREE_ATTN_MASKS[mask_name].to(device)
    tree_size_q = tree_mask.shape[0]
    dim_per_head = 128
    block_size = 32
    scale = dim_per_head ** (-0.5)
    seqlen_k = sequence_position + tree_size_q
    num_pages = (seqlen_k + block_size - 1) // block_size

    # Random q/k/v for the tree tokens: [B, tree_size_q, H, D]
    q = torch.randn(batch_size, tree_size_q, num_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, tree_size_q, num_kv_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, tree_size_q, num_kv_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)

    # Per-sequence disjoint paged KV cache.
    total_blocks = batch_size * num_pages
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, dim_per_head,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, dim_per_head,
                          device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(batch_size, num_pages, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for p in range(num_pages):
            block_table[b, p] = b * num_pages + p

    # Write the tree tokens' k/v into the cache at positions
    # [sequence_position : seqlen_k] for every sequence.
    for b in range(batch_size):
        for t in range(tree_size_q):
            pos = sequence_position + t
            blk = block_table[b, pos // block_size].item()
            off = pos % block_size
            k_cache[blk, off] = k[b, t]
            v_cache[blk, off] = v[b, t]

    # ---- Whole-tree attention (qq_bias), batched ----
    qq_bias = _mask_to_bias(tree_mask)
    q_flat = q.reshape(batch_size * tree_size_q, num_heads, dim_per_head)
    seq_lens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    query_loc = [i * tree_size_q for i in range(batch_size + 1)]

    tree_out = tree_unified_attention_multiseq(
        q_flat, k_cache, v_cache, block_table, seq_lens, query_loc,
        qq_bias=qq_bias, scale=scale, block_size=block_size,
        num_kv_heads=num_kv_heads,
    ).reshape(batch_size, tree_size_q, num_heads, dim_per_head)

    # Prefix context (positions [0, sequence_position)) per sequence, dense.
    def ctx_kv(b):
        kc = k_cache.new_empty(sequence_position, num_kv_heads, dim_per_head)
        vc = v_cache.new_empty(sequence_position, num_kv_heads, dim_per_head)
        for pos in range(sequence_position):
            blk = block_table[b, pos // block_size].item()
            off = pos % block_size
            kc[pos] = k_cache[blk, off]
            vc[pos] = v_cache[blk, off]
        return kc, vc

    # ---- Verify each branch against dense reference ----
    for q_index in range(tree_size_q):
        branch_idx = torch.nonzero(tree_mask[q_index, :], as_tuple=True)[0]
        for b in range(batch_size):
            kc, vc = ctx_kv(b)
            q_node = q[b, q_index:q_index + 1]      # [1, H, D]
            k_vis = k[b, branch_idx]                 # [V, Hkv, D]
            v_vis = v[b, branch_idx]
            ref = _dense_ref_node(q_node, kc, vc, k_vis, v_vis, scale)
            got = tree_out[b, q_index:q_index + 1]
            assert torch.allclose(got.float(), ref.float(), atol=7.81e-3), (
                f"mismatch mask={mask_name} bs={batch_size} H={num_heads} "
                f"pos={sequence_position} q_index={q_index} b={b} "
                f"max_diff={(got.float()-ref.float()).abs().max().item():.5f}"
            )


