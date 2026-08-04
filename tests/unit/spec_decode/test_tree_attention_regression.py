"""Regression tests for speculative token-tree pure logic — no model loading.

Each class below guards a bug that actually happened.  The common theme is
that the draft/verify pipeline used to assume a *linear chain* (draft slot i
== sequence position i), which silently produces wrong answers — or crashes
the EngineCore — the moment the tree branches.

Covered functions (all pure, torch is the only heavy import):
- attach_speculative_tree_metadata  — tree config validation
- compute_parent_slots       — BFS parent slot per draft slot
- compute_root_paths      — root-to-leaf paths as BFS slot indices
- _compute_child_counts           — per-level per-parent child counts
- _compute_drafts_per_level         — drafts + cumulative drafts per level
- _slice_tree_attn_bias      — per-level bias slicing
- prepare_speculative_token_tree_attn_bias — (N+1)x(N+1) attn bias
"""

import pytest
import torch

from vllm_ascend.spec_decode.speculative_token_tree import (
    attach_speculative_tree_metadata,
    compute_parent_slots,
    compute_root_paths,
    _compute_child_counts,
    _compute_drafts_per_level,
    _slice_tree_attn_bias,
    prepare_speculative_token_tree_attn_bias,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

class _FakeSpeculativeConfig:
    """Minimal stand-in for vLLM's SpeculativeConfig.

    Only exposes the two attributes attach_speculative_tree_metadata reads on
    the way in (speculative_token_tree, num_speculative_tokens).  It is a plain
    mutable object because attach_* writes derived fields via
    object.__setattr__.
    """

    def __init__(self, speculative_token_tree, num_speculative_tokens):
        self.speculative_token_tree = speculative_token_tree
        self.num_speculative_tokens = num_speculative_tokens


class _FakeVllmConfig:
    """Wraps a speculative_config, mirroring vllm_config.speculative_config."""

    def __init__(self, speculative_config):
        self.speculative_config = speculative_config


def _make_vllm_config(tree_choices, num_speculative_tokens):
    """Build a fake vllm_config whose tree string round-trips through parsing."""
    return _FakeVllmConfig(
        _FakeSpeculativeConfig(str(tree_choices), num_speculative_tokens)
    )


def _node_depth_for_slot(tree_choices, slot):
    """Depth (path length) of the tree node occupying a given BFS slot index."""
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    return len(sorted_choices[slot])


# -------------------------------------------------------------------------
# TestParentSlots
# -------------------------------------------------------------------------

class TestParentSlots:
    """compute_parent_slots must map each slot to its BFS *parent* slot.

    Bug guarded: the verifier assigned linearly increasing target-logits
    indices, which is only correct for a pure chain.  For a branching tree
    sibling slots share a parent, so the second sibling was compared against
    the wrong target position.
    """

    def test_pure_chain_is_identity_shifted(self):
        # Chain: every slot's parent is its immediate predecessor.
        assert compute_parent_slots([(0,), (0, 0), (0, 0, 0)]) == (-1, 0, 1)

    def test_branch_siblings_share_parent(self):
        # slot1=(0,0) and slot2=(0,1) are siblings -> both parented by slot0.
        assert compute_parent_slots([(0,), (0, 0), (0, 1)]) == (-1, 0, 0)

    def test_branch_with_deeper_child(self):
        # (0,0,0) hangs off (0,0)=slot1; (0,1)=slot2 still shares slot0.
        assert compute_parent_slots(
            [(0,), (0, 0), (0, 1), (0, 0, 0)]
        ) == (-1, 0, 0, 1)

    def test_unsorted_input_is_bfs_normalized(self):
        # Same tree as the branch case but supplied out of BFS order.
        assert compute_parent_slots([(0, 1), (0,), (0, 0)]) == (-1, 0, 0)

    def test_chain_parent_plus_one_is_linear_index(self):
        # For a chain, parent_slot+1 must equal the linear sequence index.
        # This is exactly the identity the old linear code relied on.
        choices = [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]
        parents = compute_parent_slots(choices)
        logits_index = tuple(p + 1 for p in parents)
        assert logits_index == (0, 1, 2, 3)


# -------------------------------------------------------------------------
# TestRootPaths
# -------------------------------------------------------------------------

class TestRootPaths:
    """compute_root_paths must enumerate every root-to-leaf path as BFS slots.

    Bug guarded: the sampler assumed draft slot i == position i.  For branches
    that fails; explicit root-to-leaf paths let the caller pick one genuine
    chain where the k-th slot really is position k.
    """

    def test_pure_chain_single_path(self):
        assert compute_root_paths([(0,), (0, 0), (0, 0, 0)]) == ((0, 1, 2),)

    def test_branch_two_paths(self):
        # Two leaves (0,0)=slot1 and (0,1)=slot2, each a length-2 path.
        assert compute_root_paths([(0,), (0, 0), (0, 1)]) == ((0, 1), (0, 2))

    def test_branch_deepest_path_first(self):
        # Leaves: (0,0,0)=slot3 (depth 3) and (0,1)=slot2 (depth 2).
        # Paths are sorted deepest-first so paths[0] is the longest chain.
        assert compute_root_paths(
            [(0,), (0, 0), (0, 1), (0, 0, 0)]
        ) == ((0, 1, 3), (0, 2))

    def test_kth_slot_has_depth_k_plus_1_invariant(self):
        # Invariant: for any tree, the k-th (0-based) slot of every path must
        # correspond to a tree node of depth k+1.  This is what makes a path a
        # valid position-aligned chain for the sampler.
        trees = [
            [(0,), (0, 0), (0, 0, 0)],
            [(0,), (0, 0), (0, 1)],
            [(0,), (0, 0), (0, 1), (0, 0, 0)],
            [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)],
        ]
        for tree in trees:
            for path in compute_root_paths(tree):
                for k, slot in enumerate(path):
                    assert _node_depth_for_slot(tree, slot) == k + 1


# -------------------------------------------------------------------------
# TestConfigValidation
# -------------------------------------------------------------------------

class TestConfigValidation:
    """attach_speculative_tree_metadata must reject trees that would later
    crash the EngineCore or silently truncate draft tokens.

    Bug guarded: invalid trees used to surface as opaque shape errors deep
    inside _draft_tree_per_level; validate up front with actionable messages.
    """

    def test_valid_chain_no_raise(self):
        attach_speculative_tree_metadata(
            _make_vllm_config([(0,), (0, 0), (0, 0, 0)], 3)
        )

    def test_valid_branch_no_raise(self):
        attach_speculative_tree_metadata(
            _make_vllm_config([(0,), (0, 0), (0, 1)], 3)
        )

    def test_valid_branch_with_deeper_child_no_raise(self):
        attach_speculative_tree_metadata(
            _make_vllm_config([(0,), (0, 0), (0, 1), (0, 0, 0)], 4)
        )

    def test_root_fanout_raises(self):
        # Multiple depth-1 nodes: root fan-out is unsupported (seeded with a
        # single first-draft token, next_draft_idx hardcoded to 1).
        with pytest.raises(ValueError, match="root|depth-1"):
            attach_speculative_tree_metadata(
                _make_vllm_config([(0,), (1,), (0, 0), (1, 0)], 4)
            )

    def test_node_count_mismatch_raises(self):
        # 2 nodes but num_speculative_tokens=3 -> slot layout mismatch.
        with pytest.raises(ValueError, match="num_speculative_tokens|node"):
            attach_speculative_tree_metadata(
                _make_vllm_config([(0,), (0, 0)], 3)
            )

    def test_missing_parent_raises(self):
        # (0,0,0) present without its parent (0,0): broken ancestor chain.
        with pytest.raises(ValueError, match="parent|ancestor"):
            attach_speculative_tree_metadata(
                _make_vllm_config([(0,), (0, 0, 0)], 2)
            )


# -------------------------------------------------------------------------
# TestChildCountOrdering
# -------------------------------------------------------------------------

class TestChildCountOrdering:
    """_compute_child_counts must enumerate parents in BFS order.

    Bug guarded: without BFS-sorting, a user tree not already in BFS order
    yields per-parent counts whose order disagrees with the bias matrix and
    the draft-token layout -- a silent wrong answer.
    """

    def test_branch(self):
        # L0: 1 root child.  L1: parent (0,) has 2 children ((0,0),(0,1)).
        assert _compute_child_counts([(0,), (0, 0), (0, 1)]) == [(1,), (2,)]

    def test_unsorted_input_same_result(self):
        assert _compute_child_counts([(0, 1), (0,), (0, 0)]) == [(1,), (2,)]

    def test_branch_with_deeper_child_keeps_zero(self):
        # L2 parents in BFS order are (0,0) then (0,1); only (0,0) has a child.
        # The (0,1) parent must appear as an explicit 0, not be dropped.
        assert _compute_child_counts(
            [(0,), (0, 0), (0, 1), (0, 0, 0)]
        ) == [(1,), (2,), (1, 0)]


# -------------------------------------------------------------------------
# TestZeroChildParent
# -------------------------------------------------------------------------

class TestZeroChildParent:
    """The most important regression: a parent with 0 children.

    Bug guarded: dropping the explicit 0 shrank a level's child-count vector,
    so the drafter's repeat_interleave got a mismatched size and crashed the
    EngineCore.
    """

    def test_explicit_zero_child_is_kept(self):
        # child_counts[2] describes level-2 parents (0,0) and (0,1).
        # (0,1) has no children -> the tuple must still contain the trailing 0.
        counts = _compute_child_counts([(0,), (0, 0), (0, 1), (0, 0, 0)])
        assert counts[2] == (1, 0)

    def test_row_accounting_matches_repeat_interleave(self):
        # Simulate the drafter's per-level repeat_interleave bookkeeping.
        # Starting from cur_rows == batch (level 0 emits one token per request),
        # each level expands rows by the per-parent child counts times batch.
        # If any explicit 0 were dropped, len(counts) != cur_rows and the real
        # repeat_interleave would raise a size-mismatch error.
        tree = [(0,), (0, 0), (0, 1), (0, 0, 0)]
        child_counts = _compute_child_counts(tree)
        for batch in (1, 2, 4):
            cur_rows = batch  # level 0: one seed token per request
            # child_counts[0] is the root level (consumed as the seed);
            # subsequent levels drive the expansion.
            for level in range(1, len(child_counts)):
                counts = list(child_counts[level]) * batch
                assert len(counts) == cur_rows
                cur_rows = sum(counts)

    def test_bias_slicing_tolerates_zero_child_level(self):
        # A level whose parents produce 0 new drafts must not break slicing;
        # we still get exactly one slice per level.
        tree = [(0,), (0, 0), (0, 1), (0, 0, 0)]
        bias = prepare_speculative_token_tree_attn_bias(tree)
        drafts_per_level, _ = _compute_drafts_per_level(tree)
        slices = _slice_tree_attn_bias(bias, drafts_per_level)
        assert len(slices) == len(drafts_per_level)
        # Sanity: every slice is a square torch tensor of the level's width.
        for query_len, sl in zip(drafts_per_level, slices):
            assert isinstance(sl, torch.Tensor)
            assert sl.shape == (query_len, query_len)
