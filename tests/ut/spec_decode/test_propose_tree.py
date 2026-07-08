# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""
Tree drafting plan tests — adapted from upstream vLLM's
``tests/v1/spec_decode/test_eagle.py::test_propose_tree`` (snapshot
``be0dcc29d``, removed by PR #42121).

Upstream verifies that ``proposer.propose()`` with a ``speculative_token_tree``
expands draft tokens level-by-level using ``cu_drafts_per_level`` /
``child_drafts_per_level`` and returns ``batch_size × num_speculative_tokens``
draft tokens in the correct order.
"""

import ast
import types

import pytest
import torch

from vllm_ascend.spec_decode.speculative_token_tree import (
    attach_speculative_tree_metadata,
    prepare_speculative_token_tree_attn_bias,
    _compute_drafts_per_level,
    _compute_child_counts,
)


def _make_spec_config(tree_str, num_spec_tokens):
    sc = types.SimpleNamespace()
    sc.speculative_token_tree = tree_str
    sc.num_speculative_tokens = num_spec_tokens
    vllm_config = types.SimpleNamespace(speculative_config=sc)
    return vllm_config, sc


TREES = {
    "single": "[(0,)]",
    "chain": "[(0,), (0, 0), (0, 0, 0)]",
    "parallel": "[(0,), (1,), (2,)]",
    "tree": "[(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]",
}


@pytest.mark.parametrize("name", list(TREES.keys()))
def test_tree_drafting_plan(name):
    tree_str = TREES[name]
    raw = ast.literal_eval(tree_str)
    num_spec = len(raw)

    vllm_config, sc = _make_spec_config(tree_str, num_spec)
    attach_speculative_tree_metadata(vllm_config)

    assert sc.tree_len == num_spec + 1
    assert sum(sc.drafts_per_level) == num_spec

    expected_cu = [0]
    for c in sc.drafts_per_level:
        expected_cu.append(expected_cu[-1] + c)
    assert list(sc.cu_drafts_per_level) == expected_cu

    tree_depth = max(len(p) for p in raw)
    assert len(sc.drafts_per_level) == tree_depth
    assert len(sc.child_drafts_per_level) == tree_depth


@pytest.mark.parametrize("name", list(TREES.keys()))
def test_tree_attn_bias_shape_and_root(name):
    raw = ast.literal_eval(TREES[name])
    bias = prepare_speculative_token_tree_attn_bias(raw)
    tree_len = len(raw) + 1
    assert bias.shape == (tree_len, tree_len)
    assert torch.all(bias[:, 0] == 0)
    for i in range(tree_len):
        assert bias[i, i] == 0


def test_chain_plan_values():
    choices = [(0,), (0, 0), (0, 0, 0)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [1, 1, 1]
    assert cu == [0, 1, 2, 3]


def test_parallel_plan_values():
    choices = [(0,), (1,), (2,)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [3]
    assert cu == [0, 3]
    children = _compute_child_counts(choices)
    assert len(children) == 1


def test_branching_tree_plan_values():
    choices = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    drafts, cu = _compute_drafts_per_level(choices)
    assert drafts == [3, 6]
    assert cu == [0, 3, 9]
    assert sum(drafts) == 9
