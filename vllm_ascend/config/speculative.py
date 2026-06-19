# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Speculative decoding configuration extensions for Ascend NPU.

This module provides ``AscendSpeculativeConfig``, which extends the upstream
``SpeculativeConfig`` with the ``speculative_token_tree`` field that was removed
from upstream vLLM (PR #42121) but is still needed for Ascend NPU branch-tree
speculative decoding.

Usage
-----
The recommended way to integrate ``speculative_token_tree`` into the vLLM config
is to store it in ``vllm_config.additional_config`` (a free-form ``dict`` that
vLLM-Ascend already uses for NPU-specific settings) **and** monkey-patch it onto
the ``SpeculativeConfig`` instance so that existing code can continue to access it
via ``speculative_config.speculative_token_tree``.

``patch_speculative_config()`` below does exactly that.  Call it once during engine
initialisation (e.g. from ``NPUPlatform`` or the engine core) after the upstream
config has been constructed.

Environment gating
------------------
Tree attention is experimental.  Set the environment variable
``VLLM_ASCEND_EXPERIMENTAL_TREE_ATTENTION=1`` (or pass
``additional_config["enable_ascend_tree_attention_experimental"] = True``) to
enable it.  Without the flag, any non-``None`` ``speculative_token_tree`` value
that would trigger branch-tree behaviour raises ``NotImplementedError``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vllm.config import SpeculativeConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

__all__ = [
    "patch_speculative_config",
    "is_tree_attention_enabled",
    "validate_tree_attention",
]

# ---------------------------------------------------------------------------
# Environment / additional_config keys
# ---------------------------------------------------------------------------

ENV_TREE_ATTENTION = "VLLM_ASCEND_EXPERIMENTAL_TREE_ATTENTION"
CONFIG_KEY_TREE_ATTENTION = "enable_ascend_tree_attention_experimental"
CONFIG_KEY_TOKEN_TREE = "speculative_token_tree"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_tree_attention_enabled(vllm_config: VllmConfig | dict) -> bool:
    """Return ``True`` when experimental tree attention is enabled.

    The flag is read in the following order (first wins):

    1. ``vllm_config.additional_config[CONFIG_KEY_TREE_ATTENTION]``
    2. ``VLLM_ASCEND_EXPERIMENTAL_TREE_ATTENTION`` environment variable
    """
    # Dict / dataclass tolerant access
    additional_config: dict[str, Any] = {}
    if isinstance(vllm_config, dict):
        additional_config = vllm_config.get("additional_config", {})
    else:
        additional_config = getattr(vllm_config, "additional_config", {})

    if isinstance(additional_config, dict):
        if CONFIG_KEY_TREE_ATTENTION in additional_config:
            return bool(additional_config[CONFIG_KEY_TREE_ATTENTION])

    return os.environ.get(ENV_TREE_ATTENTION, "0") == "1"


def _parse_token_tree(value: str | list | None) -> list[tuple[int, ...]] | None:
    """Normalise *value* into a sorted list of ancestor-chain tuples.

    Accepted formats
    -----------------
    * ``None``                  → ``None`` (caller will generate a linear chain)
    * ``"[(0,), (0, 0)]"``  → parsed via ``ast.literal_eval``
    * A ready-made ``list``      → used as-is after sorting

    Returns
    -------
    ``None`` if *value* is ``None``, otherwise a sorted list of tuples.
    """
    if value is None:
        return None
    if isinstance(value, str):
        import ast

        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Cannot parse speculative_token_tree string: {value!r}"
            ) from exc
    if not isinstance(value, list) or not all(isinstance(t, (list, tuple)) for t in value):
        raise ValueError(
            "speculative_token_tree must be a list of lists/tuples, "
            f"got {type(value).__name__}"
        )
    # Sort by (len, chain) – matches upstream behaviour before removal.
    return sorted(
        [tuple(t) for t in value],
        key=lambda t: (len(t), t),
    )


def patch_speculative_config(vllm_config: VllmConfig) -> None:
    """Attach ``speculative_token_tree`` (and helpers) to the live config.

    This function should be called **once** after the upstream ``VllmConfig`` has
    been fully constructed.  It:

    1. Reads ``speculative_token_tree`` from ``additional_config``.
    2. Parses / validates it.
    3. Monkey-patches the parsed value onto
       ``vllm_config.speculative_config.speculative_token_tree`` so that
       later code can access it with
       ``getattr(speculative_config, "speculative_token_tree", None)``.
    4. Attaches ``is_tree_attention_enabled`` as a convenience bound method.

    After this call the ``SpeculativeConfig`` object is **not** a subclass – it
    is the standard upstream object with extra attributes.  This avoids pydantic
    schema issues that would arise from subclassing a ``@config`` model.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return

    additional_config = getattr(vllm_config, "additional_config", {}) or {}

    # -- 1. Read raw value ---------------------------------------------------
    raw = additional_config.get(CONFIG_KEY_TOKEN_TREE, None)

    # -- 2. Parse ------------------------------------------------------------
    parsed = _parse_token_tree(raw)

    # -- 3. Monkey-patch -----------------------------------------------------
    # We deliberately set the attribute directly rather than going through
    # pydantic validators, because SpeculativeConfig is not aware of this field.
    object.__setattr__(spec_config, "speculative_token_tree", parsed)
    object.__setattr__(
        spec_config,
        "speculative_token_tree_raw",
        raw,
    )

    # Attach a helper so callers can test the flag without importing this module.
    object.__setattr__(
        spec_config,
        "is_tree_attention_enabled",
        lambda: is_tree_attention_enabled(vllm_config),
    )


def validate_tree_attention(vllm_config: VllmConfig) -> None:
    """Raise if branch-tree speculative decoding is used without the experimental flag.

    Call this during engine / proposer initialisation.  For a **linear** tree
    (or ``None``) the function returns silently.

    Raises
    ------
    NotImplementedError
        When ``speculative_token_tree`` describes a **branch** tree and the
        experimental flag is not set.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return

    tree = getattr(spec_config, "speculative_token_tree", None)
    if tree is None:
        return

    # A tree is "linear" when every chain has length 1 (i.e. every draft
    # token's only ancestor is the root).
    is_linear = all(len(chain) == 1 for chain in tree)
    if is_linear:
        return

    if not is_tree_attention_enabled(vllm_config):
        raise NotImplementedError(
            "Branch-tree speculative decoding (speculative_token_tree) is "
            "experimental on Ascend NPU.  Enable it with one of:\n"
            f"  export {ENV_TREE_ATTENTION}=1\n"
            f'  or pass additional_config["{CONFIG_KEY_TREE_ATTENTION}"] = True\n'
            "Do NOT use this for production workloads."
        )
