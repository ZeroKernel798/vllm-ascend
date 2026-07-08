# SPDX-License-Identifier: Apache-2.0
"""Regression tests: sparse_mode routing for tree attention and sinks.

Verifies forward_fused_infer_attention kernel selection:

* ``is_tree_mask=True`` → ``sparse_mode=0`` (tree attention path).
* ``self.sinks is not None`` → V2 API, ``sparse_mode=3/4`` (bypasses tree).
* Source-level checks that documentation comments exist.

See ``attention_v1.py`` for inline comments documenting the sinks-tree
mutual exclusion.
"""
from __future__ import annotations

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
                        "vllm_ascend.ops.triton.unified_attention.tree_unified_attention",
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
