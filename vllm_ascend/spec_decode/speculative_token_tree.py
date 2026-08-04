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
    """Per-level per-parent child counts for tree drafting.

    Parents are enumerated in BFS order to match the slot numbering used by
    ``prepare_speculative_token_tree_attn_bias`` / ``compute_parent_slots`` /
    ``compute_root_paths``.  Without sorting, a user-supplied tree that is not
    already in BFS order would produce per-parent counts whose order disagrees
    with the bias matrix and the draft-token layout -- a silent wrong answer.
    """
    if not tree_choices:
        return []
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    tree_depth = max(len(p) for p in sorted_choices)
    counts: list[tuple[int, ...]] = []
    if tree_depth >= 1:
        counts.append((sum(1 for c in sorted_choices if len(c) == 1),))
    for depth in range(1, tree_depth):
        parents = [c for c in sorted_choices if len(c) == depth]
        children = [c for c in sorted_choices if len(c) == depth + 1]
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


def _slice_tree_attn_bias_rectangular(
    tree_attn_bias: torch.Tensor,
    drafts_per_level: list[int],
) -> tuple[torch.Tensor, ...]:
    """Slice tree_attn_bias per level, rectangular: rows=[level], cols=[all drafts]."""
    slices: list[torch.Tensor] = []
    offset = 0
    total_drafts = sum(drafts_per_level)
    for query_len in drafts_per_level:
        if query_len <= 0:
            slices.append(
                torch.empty(0, total_drafts, device=tree_attn_bias.device, dtype=tree_attn_bias.dtype)
            )
        else:
            start = 1 + offset
            end = start + query_len
            # Include ALL draft columns (skip root col 0)
            slices.append(tree_attn_bias[start:end, 1:].contiguous())
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

    # ── Validate the tree before anything derives metadata from it ──
    # These conditions otherwise surface as shape errors deep inside
    #``_draft_tree_per_level`` (killing the EngineCore) or, worse, as silently
    # truncated draft tokens.
    if len(tree_choices) != num_speculative_tokens:
        raise ValueError(
            f"speculative_token_tree has {len(tree_choices)} nodes but "
            f"num_speculative_tokens={num_speculative_tokens}; they must match "
            f"(每个树节点占用一个 draft 槽位)."
        )
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    choice_set = set(sorted_choices)
    for path in sorted_choices:
        if len(path) > 1 and path[:-1] not in choice_set:
            raise ValueError(
                f"speculative_token_tree node {path} has no parent {path[:-1]}; "
                f"every non-root node needs its ancestor chain present."
            )
    num_roots = sum(1 for c in sorted_choices if len(c) == 1)
    if num_roots != 1:
        # ``_draft_tree_per_level`` seeds level 0 with exactly one token per
        # request (``first_draft_token_ids``) and hardcodes next_draft_idx=1,
        # so a fan-out at the root would mismatch repeat_interleave counts.
        raise ValueError(
            f"speculative_token_tree has {num_roots} depth-1 nodes; only a single "
            f"root child is supported (根节点多分支尚未实现)."
        )

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
    object.__setattr__(speculative_config, "tree_root_paths", compute_root_paths(tree_choices))
    object.__setattr__(speculative_config, "tree_parent_slots", compute_parent_slots(tree_choices))


def compute_parent_slots(
    tree_choices: list[SpeculativeTokenChoice],
) -> tuple[int, ...]:
    """Per draft slot, the BFS slot index of its parent (-1 for root children).

    This is the crux of the verification fix.  The target forward is fed
    ``[last_accepted, d0, d1, ... d_{n-1}]`` and emits logits where index
    ``j`` predicts the token following input``j``.  A draft slot must
    therefore be compared against the logits produced by its **parent**:

        target_logits_index(slot) = parent_slot(slot) + 1

    ``_calc_spec_decode_metadata`` instead assigns linearly increasing
    indices, which is only correct when every slot's parent is its immediate
    predecessor — i.e. a pure chain.  For a branching tree the sibling slots
    share one parent, so the second sibling is silently compared against a
    position it does not belong to.

    Example ``[(0,), (0,0), (0,1)]`` (BFS slots 0,1,2):
        parent_slots = (-1, 0, 0)
        -> logits indices (0, 1, 1);  the linear code produces (0, 1, 2).

    For a chain this returns ``(-1, 0, 1, ..., n-2)``, reproducing the
    linear indices exactly, so nothing changes for non-branching configs.
    """
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    slot_of = {choice: i for i, choice in enumerate(sorted_choices)}
    return tuple(
        -1 if len(choice) == 1 else slot_of[choice[:-1]] for choice in sorted_choices
    )


def compute_root_paths(
    tree_choices: list[SpeculativeTokenChoice],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every root-to-leaf path as BFS draft-slot indices.

    The verification pipeline (``_calc_spec_decode_metadata`` → rejection
    sampler) assumes draft slot ``i`` corresponds to sequence position ``i``.
    That holds only for a linear chain.  In a branching tree the BFS slot
    order interleaves siblings, so a sibling at depth ``d`` lands on position
    ``slot_index`` rather than ``d - 1`` and ends up compared against target
    logits for a position it does not belong to.

    Returning explicit root-to-leaf paths lets the caller select ONE chain
    whose k-th element genuinely is position k, restoring the invariant the
    sampler relies on.

    Slot indices are 0-based into the BFS-sorted draft list (root excluded),
    matching the layout produced by ``_draft_tree_per_level``.

    For a linear chain the only path is ``(0, 1, ..., n-1)`` — an identity
    mapping, so non-branching configs are unaffected.
    """
    sorted_choices = sorted(tree_choices, key=lambda p: (len(p), p))
    slot_of = {choice: i for i, choice in enumerate(sorted_choices)}
    children: dict[SpeculativeTokenChoice, list[SpeculativeTokenChoice]] = {}
    for choice in sorted_choices:
        if len(choice) > 1:
            children.setdefault(choice[:-1], []).append(choice)

    paths = []
    for leaf in sorted_choices:
        if leaf in children:
            continue  # not a leaf
        # Ancestor chain: (0,1) -> slots of [(0,), (0,1)]
        paths.append(tuple(slot_of[leaf[:plen]] for plen in range(1, len(leaf) + 1)))
    # Deepest first, so a caller preferring the longest chain can take paths[0].
    paths.sort(key=lambda p: (-len(p), p))
    return tuple(paths)
