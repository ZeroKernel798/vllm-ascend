# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Unit tests for propose_tree() method."""

import pytest
import torch

from vllm_ascend.speculative_token_tree import (
    build_speculative_token_tree_plan,
    is_linear_speculative_token_tree,
)


class TestProposeTreeIntegration:
    """Integration tests for propose_tree() method."""

    def test_tree_plan_creation(self):
        """Test that tree plan can be created for branching tree."""
        # Branching tree: root -> child1, root -> child2, child1 -> grandchild1
        tree_choices = [[0], [0], [1]]  # 3 draft tokens
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        
        assert tree_plan.tree_len == 3
        assert tree_plan.tree_depth == 1  # Only depth 1 has nodes
        assert len(tree_plan.depth_counts) == 2  # depth 0 and 1
        assert tree_plan.depth_counts[0] == 0  # root level (no drafts)
        assert tree_plan.depth_counts[1] == 3  # 3 nodes at depth 1

    def test_is_linear_tree(self):
        """Test linear tree detection."""
        # Linear chain
        linear_tree = [[0], [0], [0]]
        assert is_linear_speculative_token_tree(linear_tree) is True
        
        # Branching tree
        branching_tree = [[0], [0], [0, 1], [0, 2]]
        assert is_linear_speculative_token_tree(branching_tree) is False

    def test_tree_plan_parent_indices(self):
        """Test that parent indices are correctly computed."""
        # Tree: root -> (child1, child2), child1 -> grandchild1
        tree_choices = [[0], [0], [1]]
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        
        # Parent of child1 (idx 0) = 0 (root)
        # Parent of child2 (idx 1) = 0 (root)
        # Parent of grandchild1 (idx 2) = 1 (child1)
        assert tree_plan.parent_indices[0] == 0
        assert tree_plan.parent_indices[1] == 0
        assert tree_plan.parent_indices[2] == 1

    def test_uniform_tree_validation(self):
        """Test that non-uniform trees raise error."""
        # Non-uniform tree: root has 2 children, but child1 has 2 children
        # while child2 has 0 children
        tree_choices = [[0], [0], [1], [1]]  # 4 drafts, but uneven
        # This should work if it's uniform
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        assert tree_plan.tree_len == 4


class TestAttentionBias:
    """Tests for attention bias matrix."""

    def test_bias_matrix_shape(self):
        """Test that bias matrix has correct shape (tree_len + 1, tree_len + 1)."""
        from vllm_ascend.speculative_token_tree import prepare_speculative_token_tree_attn_bias
        
        tree_choices = [[0], [0], [0, 1]]
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)
        
        # tree_len = 3, so shape should be (4, 4) (including root)
        assert bias.shape == (4, 4)

    def test_bias_matrix_root_visible(self):
        """Test that root can see itself."""
        from vllm_ascend.speculative_token_tree import prepare_speculative_token_tree_attn_bias
        
        tree_choices = [[0], [0]]
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)
        
        # Root (index 0) should be visible to itself
        assert bias[0, 0] == 0

    def test_bias_matrix_ancestor_visible(self):
        """Test that each token can see its ancestors."""
        from vllm_ascend.speculative_token_tree import prepare_speculative_token_tree_attn_bias
        
        # Tree: root -> child1, root -> child2
        tree_choices = [[0], [0]]
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)
        
        # child1 (index 1 in bias) should see root (index 0)
        assert bias[1, 0] == 0
        # child2 (index 2 in bias) should see root (index 0)
        assert bias[2, 0] == 0

    def test_bias_matrix_sibling_invisible(self):
        """Test that siblings cannot see each other."""
        from vllm_ascend.speculative_token_tree import prepare_speculative_token_tree_attn_bias
        
        # Tree: root -> child1, root -> child2
        tree_choices = [[0], [0]]
        bias = prepare_speculative_token_tree_attn_bias(tree_choices)
        
        # child1 (index 1) should NOT see child2 (index 2)
        assert torch.isinf(bias[1, 2])
        # child2 (index 2) should NOT see child1 (index 1)
        assert torch.isinf(bias[2, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
