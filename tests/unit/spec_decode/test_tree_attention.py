"""Unit tests for tree attention logic — no model loading.

Tests:
- Tree string parsing and validation
- Attention bias matrix correctness for chain/branch trees
- Draft-per-level and child-count computation
- Bias slicing (square and rectangular)
- Top-k branch sampling logic
"""

import pytest
import torch

from vllm_ascend.spec_decode.speculative_token_tree import (
    parse_speculative_token_tree,
    sort_speculative_token_tree,
    prepare_speculative_token_tree_attn_bias,
    _compute_drafts_per_level,
    _compute_child_counts,
    _slice_tree_attn_bias,
    _slice_tree_attn_bias_rectangular,
)


# ---------------------------------------------------------------------------
# parse_speculative_token_tree
# ---------------------------------------------------------------------------

class TestParseTree:
    def test_chain4(self):
        result = parse_speculative_token_tree("[(0,),(0,0),(0,0,0),(0,0,0,0)]")
        assert result == [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]

    def test_branch2(self):
        result = parse_speculative_token_tree("[(0,),(0,0),(0,1)]")
        assert result == [(0,), (0, 0), (0, 1)]

    def test_branch3(self):
        result = parse_speculative_token_tree("[(0,),(0,0),(0,1),(0,2)]")
        assert result == [(0,), (0, 0), (0, 1), (0, 2)]

    def test_none(self):
        assert parse_speculative_token_tree(None) is None

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_speculative_token_tree("[()]")

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            parse_speculative_token_tree("[(0,),(0,-1)]")

    def test_already_list(self):
        result = parse_speculative_token_tree([(0,), (0, 0)])
        assert result == [(0,), (0, 0)]


# ---------------------------------------------------------------------------
# sort_speculative_token_tree
# ---------------------------------------------------------------------------

class TestSortTree:
    def test_already_breadth_first(self):
        tree = [(0,), (0, 0), (0, 1)]
        assert sort_speculative_token_tree(tree) == [(0,), (0, 0), (0, 1)]

    def test_depth_first_input(self):
        out = sort_speculative_token_tree([(0, 0, 0), (0,), (0, 0), (0, 1)])
        assert out == [(0,), (0, 0), (0, 1), (0, 0, 0)]

    def test_empty(self):
        assert sort_speculative_token_tree([]) == []


# ---------------------------------------------------------------------------
# prepare_speculative_token_tree_attn_bias
# ---------------------------------------------------------------------------

class TestAttnBias:
    """Verify attention bias masks for various tree topologies.

    Bias matrix is (N+1) × (N+1), where row/col 0 is the implicit root.
    0 = allowed, -inf = masked.
    """

    def test_chain4_bias(self):
        """Chain: all tokens are ancestors → all visible."""
        bias = prepare_speculative_token_tree_attn_bias(
            [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]
        )
        assert bias.shape == (5, 5)
        # Root self-attention allowed
        assert bias[0, 0] == 0.0
        # All tokens visible to all later tokens (ancestor chain)
        for i in range(5):
            for j in range(1, i + 1):
                assert bias[i, j] == 0.0, f"token {i} should see ancestor {j}"
        # Future tokens masked
        assert bias[1, 4] == float("-inf")

    def test_branch2_sibling_mask(self):
        """Branch2: [(0,),(0,0),(0,1)] — siblings must be mutually invisible."""
        bias = prepare_speculative_token_tree_attn_bias(
            [(0,), (0, 0), (0, 1)]
        )
        assert bias.shape == (4, 4)
        # Root visible to all
        for i in range(1, 4):
            assert bias[i, 0] == 0.0, f"token {i} should see root"
        # Self
        assert bias[1, 1] == 0.0
        assert bias[2, 2] == 0.0
        assert bias[3, 3] == 0.0
        # Token 1 (choice=(0,)) visible to children 2 and 3
        assert bias[2, 1] == 0.0  # child 2 sees parent 1
        assert bias[3, 1] == 0.0  # child 3 sees parent 1
        # SIBLINGS: token 2 and 3 should NOT see each other
        assert bias[2, 3] == float("-inf"), "sibling 2 must NOT see sibling 3"
        assert bias[3, 2] == float("-inf"), "sibling 3 must NOT see sibling 2"

    def test_branch3_sibling_masks(self):
        """Branch3: [(0,),(0,0),(0,1),(0,2)] — three siblings."""
        bias = prepare_speculative_token_tree_attn_bias(
            [(0,), (0, 0), (0, 1), (0, 2)]
        )
        assert bias.shape == (5, 5)
        # All three siblings (idx 2,3,4) mutually invisible
        for i in range(2, 5):
            for j in range(2, 5):
                if i != j:
                    assert bias[i, j] == float("-inf"), (
                        f"sibling {i} must NOT see sibling {j}"
                    )

    def test_complex_tree(self):
        """Multi-level: root→a, root→b, a1→a1a, a2→a2a.

        NOTE: tree choice values must be unique because
        prepare_speculative_token_tree_attn_bias uses a dict {choice: index}
        that cannot disambiguate duplicate paths.
        """
        tree = [(1,), (2,), (1, 0), (2, 0)]
        # Sorted by (len, path): [(1,), (2,), (1,0), (2,0)]
        # → indices: 0=root, 1=(1,), 2=(2,), 3=(1,0), 4=(2,0)
        bias = prepare_speculative_token_tree_attn_bias(tree)
        assert bias.shape == (5, 5)

        # Sibling mask: (1,) → (2,) mutually invisible
        assert bias[1, 2] == float("-inf")
        assert bias[2, 1] == float("-inf")

        # (1,0) sees its parent (1,)
        assert bias[3, 1] == 0.0
        # (1,0) does NOT see (2,) (cross-branch)
        assert bias[3, 2] == float("-inf")

        # (2,0) sees its parent (2,)
        assert bias[4, 2] == 0.0
        # (2,0) does NOT see (1,) (cross-branch)
        assert bias[4, 1] == float("-inf")


# ---------------------------------------------------------------------------
# _compute_drafts_per_level
# ---------------------------------------------------------------------------

class TestDraftsPerLevel:
    def test_chain4(self):
        drafts, cu = _compute_drafts_per_level(
            [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]
        )
        assert drafts == [1, 1, 1, 1]
        assert cu == [0, 1, 2, 3, 4]

    def test_branch2(self):
        drafts, cu = _compute_drafts_per_level([(0,), (0, 0), (0, 1)])
        assert drafts == [1, 2]
        assert cu == [0, 1, 3]

    def test_empty(self):
        drafts, cu = _compute_drafts_per_level([])
        assert drafts == []
        assert cu == [0]


# ---------------------------------------------------------------------------
# _compute_child_counts
# ---------------------------------------------------------------------------

class TestChildCounts:
    def test_chain4(self):
        counts = _compute_child_counts(
            [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]
        )
        assert counts == [(1,), (1,), (1,), (1,)]

    def test_branch2(self):
        counts = _compute_child_counts([(0,), (0, 0), (0, 1)])
        assert counts == [(1,), (2,)]

    def test_uneven(self):
        # Level 1: 3 parents [(0,), (1,), (2,)]
        # Level 2: children: (0,0)∈parent(0,), (0,1)∈parent(0,),
        #          (2,0)∈parent(2,) — so child_counts = (2,0,1)
        tree = [(0,), (1,), (2,), (0, 0), (0, 1), (2, 0)]
        counts = _compute_child_counts(tree)
        assert counts == [(3,), (2, 0, 1)]


# ---------------------------------------------------------------------------
# _slice_tree_attn_bias (square)
# ---------------------------------------------------------------------------

class TestBiasSlicing:
    def test_square_slices_branch2(self):
        bias = prepare_speculative_token_tree_attn_bias(
            [(0,), (0, 0), (0, 1)]
        )
        slices = _slice_tree_attn_bias(bias, [1, 2])
        assert len(slices) == 2
        # Level 0: single token → 1×1
        assert slices[0].shape == (1, 1)
        assert slices[0][0, 0] == 0.0
        # Level 1: two tokens → 2×2 (siblings mutually masked)
        assert slices[1].shape == (2, 2)
        assert slices[1][0, 1] == float("-inf")
        assert slices[1][1, 0] == float("-inf")

    def test_rectangular_slices_branch2(self):
        bias = prepare_speculative_token_tree_attn_bias(
            [(0,), (0, 0), (0, 1)]
        )
        slices = _slice_tree_attn_bias_rectangular(bias, [1, 2])
        assert len(slices) == 2
        # Level 0: 1 row × 3 cols (all drafts, skip root col 0)
        assert slices[0].shape == (1, 3)
        # Level 1: 2 rows × 3 cols
        assert slices[1].shape == (2, 3)


# ---------------------------------------------------------------------------
# Top-k branch sampling logic (extracted from eagle_proposer)
# ---------------------------------------------------------------------------

class TestTopKBranchSampling:
    """Verify the top-k per-token sampling logic used in eagle_proposer.

    With repeat_interleave, all children of the same parent receive identical
    hidden states → identical logits → identical argmax. top-k picks the k most
    likely tokens and assigns them to the k children via a rank index.
    """

    def test_equal_children_same_parent(self):
        """3 children from same parent → top-3 logits assigned by rank."""
        child_counts = [3]  # one parent, 3 children
        logits = torch.tensor([[0.1, 0.5, 0.2, 0.9, 0.3]])  # 5 vocab

        # Repeat logits per child (real code uses repeat_interleave)
        logits_expanded = logits.repeat_interleave(
            torch.tensor(child_counts), dim=0
        )
        max_children = max(child_counts)
        _, topk_ids = torch.topk(logits_expanded, k=max_children, dim=-1)
        ranks = [r for n_ch in child_counts for r in range(n_ch)]
        rank_idx = torch.tensor(ranks, dtype=torch.long)
        selected = topk_ids.gather(1, rank_idx.unsqueeze(1)).squeeze(1)

        # top-3 ids from repeated rows: indices 3(0.9), 1(0.5), 4(0.3)
        assert selected.tolist() == [3, 1, 4]

    def test_two_parents_uneven_children(self):
        """Parent A: 1 child, Parent B: 2 children → rank=[0, 0, 1]."""
        child_counts = [1, 2]
        # Parent A logits (row 0), Parent B logits (row 1)
        logits = torch.tensor([
            [0.1, 0.8, 0.3, 0.5, 0.2],
            [0.7, 0.1, 0.4, 0.9, 0.2],
        ])
        # Parent B has 2 children → repeat_interleave → row appears twice
        logits_expanded = logits.repeat_interleave(
            torch.tensor(child_counts), dim=0
        )
        assert logits_expanded.shape == (3, 5)

        max_children = max(child_counts)
        _, topk_ids = torch.topk(logits_expanded, k=max_children, dim=-1)
        ranks = [r for n_ch in child_counts for r in range(n_ch)]
        rank_idx = torch.tensor(ranks, dtype=torch.long)
        selected = topk_ids.gather(1, rank_idx.unsqueeze(1)).squeeze(1)

        # Child 0 from parent A: rank=0 → top-1 of [0.1,0.8,0.3,0.5,0.2] → idx 1
        # Child 1 from parent B: rank=0 → top-1 of [0.7,0.1,0.4,0.9,0.2] → idx 3
        # Child 2 from parent B: rank=1 → top-2 of same → idx 0 (0.7)
        assert selected.tolist() == [1, 3, 0]

    def test_argmax_fallback_when_single_child(self):
        """When max_children == 1, argmax is used (no top-k)."""
        child_counts = [1, 1]
        logits = torch.tensor([
            [0.1, 0.8, 0.3],
            [0.7, 0.1, 0.4],
        ])
        max_children = max(child_counts)
        if max_children <= 1:
            selected = logits.argmax(dim=-1)
        assert selected.tolist() == [1, 0]
