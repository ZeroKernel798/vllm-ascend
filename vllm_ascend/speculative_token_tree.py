# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Speculative token tree utilities for Ascend backend.

This module provides tree parsing, validation, and metadata preparation
for branching speculative decoding on Ascend NPU devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Configuration key for enabling experimental tree attention
ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY = "enable_ascend_tree_attention_experimental"


@dataclass
class SpeculativeTokenTreePlan:
    """Pre-computed plan for a uniform speculative token tree.

    Attributes:
        tree_choices: List of ancestor indices for each draft token.
        depth_counts: Number of draft tokens at each depth.
        cu_drafts_per_level: Cumulative sum of drafts per level.
        child_drafts_per_level: Number of children per parent at each level.
        parent_indices: Parent index for each draft token.
        tree_depth: Maximum depth of the tree.
        tree_len: Total number of draft tokens (excluding root).
    """
    tree_choices: list[list[int]]
    depth_counts: list[int]
    cu_drafts_per_level: list[int]
    child_drafts_per_level: list[int]
    parent_indices: list[int]
    tree_depth: int
    tree_len: int


def parse_speculative_token_tree(
    tree_choices: list[list[int]],
) -> tuple[list[list[int]], int, int]:
    """Parse speculative token tree definition.

    Args:
        tree_choices: List of ancestor chains for each draft token.
            Each inner list contains ancestor indices ending with the parent.

    Returns:
        Tuple of (normalized_tree_choices, tree_depth, tree_len).

    Raises:
        ValueError: If tree_choices is invalid.
    """
    if not tree_choices:
        raise ValueError("tree_choices must not be empty")

    tree_len = len(tree_choices)
    tree_depth = max(len(chain) for chain in tree_choices)

    # Normalize: ensure all chains have same length (pad with -1)
    normalized = []
    for chain in tree_choices:
        padded = list(chain) + [-1] * (tree_depth - len(chain))
        normalized.append(padded)

    return normalized, tree_depth, tree_len


def is_linear_speculative_token_tree(
    tree_choices: list[list[int]],
) -> bool:
    """Check if the tree is a linear chain (no branching).

    Args:
        tree_choices: List of ancestor chains.

    Returns:
        True if tree is linear (chain), False if branching.
    """
    if not tree_choices:
        return True

    # Linear tree: each node has exactly one child
    # tree_choices[i] = [ancestors...]
    # For linear: len(set(parent for each node)) == 1 (only root has children)
    parents = [chain[-1] for chain in tree_choices]
    unique_parents = set(parents)

    # Root is 0, so linear tree means all draft tokens have parent = 0
    # or form a single chain
    return len(unique_parents) == 1 or _is_single_chain(tree_choices)


def _is_single_chain(tree_choices: list[list[int]]) -> bool:
    """Check if tree forms a single chain."""
    # For a chain: token i's parent = i-1 (if we flatten)
    # Simplified check: no branching means at each depth, only one node
    depth_counts: dict[int, int] = {}
    for chain in tree_choices:
        depth = len(chain)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    # Chain means at most one node per depth
    return all(count <= 1 for count in depth_counts.values())


def validate_ascend_speculative_token_tree_support(
    speculative_config: object,
    vllm_config: object,
    enable_experimental_tree_attention: bool = False,
) -> None:
    """Validate if Ascend backend supports the given speculative token tree.

    Args:
        speculative_config: Speculative decoding configuration.
        vllm_config: vLLM configuration.
        enable_experimental_tree_attention: Whether experimental tree attention is enabled.

    Raises:
        NotImplementedError: If tree is not supported.
        ValueError: If tree configuration is invalid.
    """
    if not hasattr(speculative_config, 'speculative_token_tree'):
        return

    tree = speculative_config.speculative_token_tree
    if tree is None:
        return

    # Check if tree is linear
    if is_linear_speculative_token_tree(tree):
        logger.debug("Linear speculative token tree detected, using existing path")
        return

    # Branching tree detected
    # Check vllm_config if flag not explicitly set
    if not enable_experimental_tree_attention:
        enable_experimental_tree_attention = is_ascend_experimental_tree_attention_enabled(vllm_config)
    if not enable_experimental_tree_attention:
        raise NotImplementedError(
            "Ascend backend currently only supports linear speculative token trees. "
            "Branching trees require experimental tree attention support. "
            f"Set vllm_config.additional_config['{ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY}'] = True "
            "to enable experimental support (not for production use)."
        )

    # Experimental path: validate constraints
    _validate_experimental_tree_constraints(speculative_config, vllm_config)


def _validate_experimental_tree_constraints(
    speculative_config: object,
    vllm_config: object,
) -> None:
    """Validate constraints for experimental tree attention.

    Raises:
        NotImplementedError: If constraints are not met.
    """
    # Check method
    method = getattr(speculative_config, 'method', None)
    if method not in ("eagle", "eagle3", "draft_model"):
        raise NotImplementedError(
            f"Experimental tree attention only supports method in "
            f"('eagle', 'eagle3', 'draft_model'), got '{method}'"
        )

    # Check enforce_eager
    enforce_eager = getattr(speculative_config, 'enforce_eager', False)
    if not enforce_eager:
        raise NotImplementedError(
            "Experimental tree attention requires enforce_eager=True"
        )

    # Check parallel drafting
    parallel_drafting = getattr(speculative_config, 'parallel_drafting', False)
    if parallel_drafting:
        raise NotImplementedError(
            "Experimental tree attention does not support parallel_drafting=True"
        )

    logger.warning(
        "Experimental Ascend tree attention is enabled. "
        "This is not production-ready and may have correctness/performance issues."
    )


def is_ascend_experimental_tree_attention_enabled(vllm_config: object) -> bool:
    """Check if experimental tree attention is enabled.

    Args:
        vllm_config: vLLM configuration.

    Returns:
        True if experimental tree attention is enabled.
    """
    if not hasattr(vllm_config, 'additional_config'):
        return False

    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return False

    return bool(additional_config.get(
        ASCEND_EXPERIMENTAL_TREE_ATTENTION_CONFIG_KEY, False
    ))
