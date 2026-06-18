# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Unit tests for speculative token tree utilities."""

import pytest
import torch

from vllm_ascend.speculative_token_tree import (
    ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY,
    SpeculativeTokenTreePlan,
    is_ascend_experimental_tree_attention_enabled,
    is_linear_speculative_token_tree,
    parse_speculative_token_tree,
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
