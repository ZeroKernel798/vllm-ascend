# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import ast
from typing import Any

import torch

SpeculativeTokenChoice = tuple[int, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_speculative_token_tree(value: Any) -> list[SpeculativeTokenChoice] | None:
    """Parse a user-supplied tree string into integer paths; None-safe."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid speculative_token_tree: {value!r}") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("speculative_token_tree must be a list/tuple of tuple paths")
    parsed = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            raise ValueError("speculative_token_tree entries must be tuple-like paths")
        path = tuple(int(s) for s in item)
        if not path:
            raise ValueError("speculative_token_tree paths must be non-empty")
        if any(s < 0 for s in path):
            raise ValueError("speculative_token_tree paths must contain non-negative integers")
        parsed.append(path)
    return parsed


def sort_speculative_token_tree(
    tree_choices: list[SpeculativeTokenChoice],
) -> list[SpeculativeTokenChoice]:
    """Sort breadth-first: by length then lexicographically."""
    return sorted(tree_choices, key=lambda p: (len(p), p))


def prepare_speculative_token_tree_attn_bias(
    tree_choices: list[SpeculativeTokenChoice],
) -> torch.Tensor:
    """Build (N+1)×(N+1) attention bias from tree choices.

    0 = allowed, -inf = masked (siblings, cross-branch).
    Row/col 0 represents the implicit root token.
    """
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    n = len(sorted_choices)
    bias = torch.full((n + 1, n + 1), float("-inf"), dtype=torch.float32)
    bias[0, 0] = 0.0
    index = {choice: i + 1 for i, choice in enumerate(sorted_choices)}
    for i, choice in enumerate(sorted_choices, start=1):
        bias[i, i] = 0.0
        bias[i, 0] = 0.0
        for plen in range(1, len(choice)):
            bias[i, index[choice[:plen]]] = 0.0
    return bias


def get_speculative_tree_len(speculative_config: Any) -> int:
    """Return tree positions count (1 root + N draft tokens)."""
    if speculative_config is None:
        return 1
    tree_len = getattr(speculative_config, "tree_len", None)
    if tree_len is not None:
        return int(tree_len)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", 0) or 0
    return 1 + int(num_speculative_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_drafts_per_level(
    tree_choices: list[SpeculativeTokenChoice],
) -> tuple[list[int], list[int]]:
    """Return (drafts_per_level, cu_drafts_per_level)."""
    if not tree_choices:
        return [], [0]
    tree_depth = max(len(p) for p in tree_choices)
    drafts = [sum(1 for c in tree_choices if len(c) == d) for d in range(1, tree_depth + 1)]
    cu = [0]
    for c in drafts:
        cu.append(cu[-1] + c)
    return drafts, cu


def _compute_child_counts(
    tree_choices: list[SpeculativeTokenChoice],
) -> list[tuple[int, ...]]:
    """Per-level per-parent child counts for tree drafting."""
    if not tree_choices:
        return []
    tree_depth = max(len(p) for p in tree_choices)
    counts: list[tuple[int, ...]] = []
    if tree_depth >= 1:
        counts.append((sum(1 for c in tree_choices if len(c) == 1),))
    for depth in range(1, tree_depth):
        parents = [c for c in tree_choices if len(c) == depth]
        children = [c for c in tree_choices if len(c) == depth + 1]
        counts.append(tuple(sum(1 for ch in children if ch[:-1] == p) for p in parents))
    return counts


def _slice_tree_attn_bias(
    tree_attn_bias: torch.Tensor,
    drafts_per_level: list[int],
) -> tuple[torch.Tensor, ...]:
    """Slice tree_attn_bias per drafting level, skipping root row/col."""
    slices: list[torch.Tensor] = []
    offset = 0
    for query_len in drafts_per_level:
        if query_len <= 0:
            slices.append(
                torch.empty(0, 0, device=tree_attn_bias.device, dtype=tree_attn_bias.dtype)
            )
        else:
            start = 1 + offset
            end = start + query_len
            slices.append(tree_attn_bias[start:end, start:end].contiguous())
        offset += query_len
    return tuple(slices)


# ---------------------------------------------------------------------------
# Metadata attachment — called once during config init
# ---------------------------------------------------------------------------

def attach_speculative_tree_metadata(vllm_config: Any) -> None:
    """Compute tree metadata and attach to speculative_config.

    Only attaches fields actually read at runtime (7 out of the original 14).
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return

    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", None)
    if num_speculative_tokens is None:
        return

    raw_tree = getattr(speculative_config, "speculative_token_tree", None)
    if raw_tree:
        tree_choices = parse_speculative_token_tree(raw_tree)
    else:
        tree_choices = [tuple(0 for _ in range(d)) for d in range(1, num_speculative_tokens + 1)]

    assert tree_choices is not None
    tree_attn_bias = prepare_speculative_token_tree_attn_bias(tree_choices)
    drafts_per_level, cu_drafts_per_level = _compute_drafts_per_level(tree_choices)
    tree_depth = max(len(c) for c in tree_choices)
    child_counts = _compute_child_counts(tree_choices)
    bias_slices = _slice_tree_attn_bias(tree_attn_bias, drafts_per_level)

    object.__setattr__(speculative_config, "tree_len", len(tree_choices) + 1)
    object.__setattr__(speculative_config, "tree_depth", tree_depth)
    object.__setattr__(speculative_config, "drafts_per_level", drafts_per_level)
    object.__setattr__(speculative_config, "cu_drafts_per_level", cu_drafts_per_level)
    object.__setattr__(speculative_config, "child_drafts_per_level", [list(l) for l in child_counts])
    object.__setattr__(speculative_config, "tree_attn_bias", tree_attn_bias)
    object.__setattr__(speculative_config, "tree_attn_bias_slices", bias_slices)
