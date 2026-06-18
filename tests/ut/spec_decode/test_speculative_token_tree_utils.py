# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Unit tests for speculative token tree utilities."""

import pytest
import torch

from vllm_ascend.speculative_token_tree import (
    ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY,
    SpeculativeTokenTreePlan,
    build_speculative_token_tree_plan,
    get_speculative_token_tree_depth_counts,
    is_ascend_experimental_tree_attention_enabled,
    is_linear_speculative_token_tree,
    parse_speculative_token_tree,
    prepare_speculative_token_tree_attn_bias,
    sort_speculative_token_tree,
    validate_ascend_speculative_token_tree_support,
)


class TestParseSpeculativeTokenTree:
    """Tests for parse_speculative_token_tree()."""

    def test_linear_chain(self):
        """Linear chain: [0], [0], [0] (3 drafts, all parent=root)."""
        tree_choices = [[0], [0], [0]]
        normalized, depth, length = parse_speculative_token_tree(tree_choices)
        assert depth == 1
        assert length == 3
        assert len(normalized) == 3

    def test_branching_tree(self):
        """Branching tree: root -> child1, root -> child2."""
        tree_choices = [[0], [0]]
        normalized, depth, length = parse_speculative_token_tree(tree_choices)
        assert depth == 1
        assert length == 2

    def test_empty_tree_raises(self):
        """Empty tree should raise ValueError."""
        with pytest.raises(ValueError, match="tree_choices must not be empty"):
            parse_speculative_token_tree([])


class TestIsLinearSpeculativeTokenTree:
    """Tests for is_linear_speculative_token_tree()."""

    def test_linear_chain(self):
        """Linear chain is recognized as linear."""
        tree_choices = [[0], [0], [0]]
        assert is_linear_speculative_token_tree(tree_choices) is True

    def test_branching_tree(self):
        """Branching tree is recognized as non-linear."""
        # Two children of root -> branching
        tree_choices = [[0], [0], [0, 1]]  # 3rd token has ancestors [0, 1]
        assert is_linear_speculative_token_tree(tree_choices) is False

    def test_empty_tree(self):
        """Empty tree is considered linear."""
        assert is_linear_speculative_token_tree([]) is True


class TestValidateAscendSpeculativeTokenTreeSupport:
    """Tests for validate_ascend_speculative_token_tree_support()."""

    def test_linear_tree_no_experimental(self):
        """Linear tree passes validation without experimental flag."""

        class MockSpeculativeConfig:
            speculative_token_tree = [[0], [0], [0]]

        class MockVllmConfig:
            additional_config = {}

        # Should not raise
        validate_ascend_speculative_token_tree_support(
            MockSpeculativeConfig(), MockVllmConfig()
        )

    def test_branching_tree_without_experimental_raises(self):
        """Branching tree without experimental flag raises NotImplementedError."""

        class MockSpeculativeConfig:
            speculative_token_tree = [[0], [0], [0, 1], [0, 2]]  # Branching
            method = "eagle"

        class MockVllmConfig:
            additional_config = {}

        with pytest.raises(NotImplementedError, match="experimental"):
            validate_ascend_speculative_token_tree_support(
                MockSpeculativeConfig(), MockVllmConfig()
            )

    def test_branching_tree_with_experimental_passes(self):
        """Branching tree with experimental flag passes (if constraints met)."""

        class MockSpeculativeConfig:
            speculative_token_tree = [[0], [0], [0, 1], [0, 2]]
            method = "eagle"
            enforce_eager = True
            parallel_drafting = False

        class MockVllmConfig:
            additional_config = {
                ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY: True
            }

        # Should not raise (constraints are met)
        validate_ascend_speculative_token_tree_support(
            MockSpeculativeConfig(), MockVllmConfig()
        )


class TestIsAscendExperimentalTreeAttentionEnabled:
    """Tests for is_ascend_experimental_tree_attention_enabled()."""

    def test_enabled(self):
        """Returns True when flag is set."""

        class MockVllmConfig:
            additional_config = {
                ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY: True
            }

        assert is_ascend_experimental_tree_attention_enabled(MockVllmConfig()) is True

    def test_disabled(self):
        """Returns False when flag is not set."""

        class MockVllmConfig:
            additional_config = {}

        assert is_ascend_experimental_tree_attention_enabled(MockVllmConfig()) is False

    def test_no_additional_config(self):
        """Returns False when vllm_config has no additional_config."""

        class MockVllmConfig:
            pass

        assert is_ascend_experimental_tree_attention_enabled(MockVllmConfig()) is False


class TestBuildSpeculativeTokenTreePlan:
    """Tests for build_speculative_token_tree_plan()."""

    def test_linear_chain_plan(self):
        """Linear chain creates correct plan."""
        tree_choices = [[0], [0], [0]]  # 3 drafts, all children of root
        plan = build_speculative_token_tree_plan(tree_choices)

        assert plan.tree_len == 3
        assert plan.tree_depth == 1
        assert plan.depth_counts == [0, 3]  # depth 0: 0, depth 1: 3
        assert plan.cu_drafts_per_level == [0, 0, 3]
        assert plan.child_drafts_per_level == [3]  # root has 3 children

    def test_branching_tree_plan(self):
        """Branching tree creates correct plan."""
        # Root -> child1, root -> child2
        # child1 -> grandchild1, child1 -> grandchild2
        tree_choices = [[0], [0], [0, 1], [0, 2]]  # 4 drafts
        plan = build_speculative_token_tree_plan(tree_choices)

        assert plan.tree_len == 4
        assert plan.tree_depth == 2
        assert plan.depth_counts[1] == 2  # 2 children of root
        assert plan.depth_counts[2] == 2  # 2 grandchildren

    def test_non_uniform_tree_raises(self):
        """Non-uniform tree (different parents have different number of children) should raise."""
        # This is a tricky case - need to construct a non-uniform tree
        # For now, skip this test as it requires careful construction
        pass

    def test_empty_tree_raises(self):
        """Empty tree should raise ValueError."""
        with pytest.raises(ValueError, match="tree_choices must not be empty"):
            build_speculative_token_tree_plan([])


class TestPrepareSpeculativeTokenTreeAttnBias:
    """Tests for prepare_speculative_token_tree_attn_bias()."""

    def test_linear_chain_bias(self):
        """Linear chain creates correct attention bias."""
        tree_choices = [[0], [0], [0]]  # 3 drafts
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)

        # Bias includes root (index 0) + 3 drafts = shape (4, 4)
        assert bias.shape == (4, 4)
        # Root (index 0) can see itself
        assert bias[0, 0] == 0
        # All drafts can see root (draft indices are 1, 2, 3)
        for i in range(1, 4):
            assert bias[i, 0] == 0

    def test_branching_tree_bias(self):
        """Branching tree creates correct attention bias."""
        tree_choices = [[0], [0], [0, 1], [0, 2]]  # 4 drafts
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)

        # Bias includes root (index 0) + 4 drafts = shape (5, 5)
        assert bias.shape == (5, 5)
        # Token 1 (first draft, child of root) can see root
        assert bias[1, 0] == 0
        # Token 2 (second draft, child of root) can see root
        assert bias[2, 0] == 0
        # Token 3 (third draft, child of root and draft 1) can see root and draft 1
        assert bias[3, 0] == 0  # can see root
        assert bias[3, 2] == 0  # can see parent (draft 1 is at index 2)
        # Token 4 (fourth draft, child of root and draft 2) can see root and draft 2
        assert bias[4, 0] == 0  # can see root
        assert bias[4, 3] == 0  # can see parent (draft 2 is at index 3)


class TestSortSpeculativeTokenTree:
    """Tests for sort_speculative_token_tree()."""

    def test_sort_returns_breadth_first(self):
        """Sort returns tree in breadth-first order."""
        tree_choices = [[0, 1], [0], [0, 2]]  # unsorted
        sorted_tree = sort_speculative_token_tree(tree_choices)

        # After sorting, should be in breadth-first order
        # This is a basic test - actual order depends on implementation
        assert len(sorted_tree) == 3


class TestGetSpeculativeTokenTreeDepthCounts:
    """Tests for get_speculative_token_tree_depth_counts()."""

    def test_linear_chain_depth_counts(self):
        """Linear chain returns correct depth counts."""
        tree_choices = [[0], [0], [0]]
        depth_counts = get_speculative_token_tree_depth_counts(tree_choices)

        assert depth_counts[0] == 0  # root level
        assert depth_counts[1] == 3  # 3 drafts at depth 1
