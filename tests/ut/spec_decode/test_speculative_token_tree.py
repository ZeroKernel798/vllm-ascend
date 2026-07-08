import pytest
import torch

from vllm_ascend.spec_decode.speculative_token_tree import (
    attach_speculative_tree_metadata,
    get_speculative_tree_len,
    prepare_speculative_token_tree_attn_bias,
    _compute_drafts_per_level,
    _compute_child_counts,
    _slice_tree_attn_bias,
    parse_speculative_token_tree,
    sort_speculative_token_tree,
)


class MockSpeculativeConfig:
    def __init__(self, num_speculative_tokens=3, speculative_token_tree=None):
        self.num_speculative_tokens = num_speculative_tokens
        self.speculative_token_tree = speculative_token_tree


class MockVllmConfig:
    def __init__(self, num_speculative_tokens=3, speculative_token_tree=None):
        self.speculative_config = MockSpeculativeConfig(num_speculative_tokens, speculative_token_tree)


# =====================================================================
# _compute_drafts_per_level
# =====================================================================

@pytest.mark.parametrize("tree,exp_drafts,exp_cu", [
    ([(0,), (0,0), (0,1)],          [1, 2],    [0, 1, 3]),
    ([(0,), (0,0), (0,0,0)],        [1, 1, 1], [0, 1, 2, 3]),
    ([tuple(0 for _ in range(d)) for d in range(1, 6)],
                                     [1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5]),
])
def test_drafts_per_level(tree, exp_drafts, exp_cu):
    drafts, cu = _compute_drafts_per_level(tree)
    assert drafts == exp_drafts
    assert cu == exp_cu


# =====================================================================
# _compute_child_counts
# =====================================================================

@pytest.mark.parametrize("tree,exp", [
    ([(0,), (0,0), (0,1), (0,0,0)], [(1,), (2,), (1, 0)]),
    ([(0,), (0,0), (0,1)],          [(1,), (2,)]),
])
def test_child_counts(tree, exp):
    assert _compute_child_counts(tree) == exp


# =====================================================================
# prepare_speculative_token_tree_attn_bias
# =====================================================================

TREE_SIBLINGS = [(0,), (0,0), (0,1)]
TREE_ANCESTOR = [(0,), (0,0), (0,0,0)]
TREE_CROSS = [(0,), (0,0), (0,1), (0,0,0)]


def test_prepare_bias_siblings_masked():
    bias = prepare_speculative_token_tree_attn_bias(TREE_SIBLINGS)
    assert bias.shape == (4, 4)
    assert torch.isneginf(bias[2, 3])
    assert torch.isneginf(bias[3, 2])


def test_prepare_bias_ancestor_chain():
    bias = prepare_speculative_token_tree_attn_bias(TREE_ANCESTOR)
    # (0,0,0) at row 3 attends to root(0), (0,)(1), (0,0)(2), self(3)
    assert bias[3, 0] == 0.0
    assert bias[3, 1] == 0.0
    assert bias[3, 2] == 0.0
    assert bias[3, 3] == 0.0
    assert torch.isneginf(bias[0, 1])  # root can't attend to draft


def test_prepare_bias_cross_branch_isolation():
    bias = prepare_speculative_token_tree_attn_bias(TREE_CROSS)
    assert torch.isneginf(bias[2, 3])  # (0,0) ↔ (0,1) blocked
    assert torch.isneginf(bias[4, 3])  # (0,0,0) → (0,1) blocked


# =====================================================================
# _slice_tree_attn_bias
# =====================================================================

@pytest.mark.parametrize("tree,drafts,exp_shapes", [
    (TREE_SIBLINGS,         [1, 2],    [(1, 1), (2, 2)]),
    (TREE_CROSS,            [1, 2, 1], [(1, 1), (2, 2), (1, 1)]),
    ([(0,), (0,0)],         [0, 2],    [(0, 0), (2, 2)]),
])
def test_slice_tree_attn_bias(tree, drafts, exp_shapes):
    bias = prepare_speculative_token_tree_attn_bias(tree)
    slices = _slice_tree_attn_bias(bias, drafts)
    assert len(slices) == len(exp_shapes)
    for s, shape in zip(slices, exp_shapes):
        assert s.shape == shape


# =====================================================================
# parse + sort
# =====================================================================

def test_parse_and_sort_breadth_first():
    tree = [(0, 1), (0,), (0, 0)]
    parsed = parse_speculative_token_tree(tree)
    assert sort_speculative_token_tree(parsed) == [(0,), (0, 0), (0, 1)]


@pytest.mark.parametrize("bad_input,match", [
    ("not-a-list",                     "Invalid"),
    ('{"key": "value"}',               "list/tuple of tuple paths"),
    ([(0, "x")],                       ""),  # TypeError
    ([(0,), (0, -1)],                  "non-negative"),
    ([(0,), ()],                       "non-empty"),
])
def test_parse_rejects_invalid(bad_input, match):
    with pytest.raises((ValueError, TypeError), match=match if match else None):
        parse_speculative_token_tree(bad_input)


# =====================================================================
# get_speculative_tree_len
# =====================================================================

def _make_len_cfg(tree_len, num_tokens):
    sc = MockSpeculativeConfig(num_speculative_tokens=num_tokens)
    object.__setattr__(sc, "tree_len", tree_len)
    return sc


@pytest.mark.parametrize("config,exp", [
    (None,                                                     1),
    (_make_len_cfg(6, 5),                                      6),
    (MockSpeculativeConfig(num_speculative_tokens=3),          4),
    (MockSpeculativeConfig(num_speculative_tokens=0),          1),
    (MockSpeculativeConfig(num_speculative_tokens=None),       1),
])
def test_get_tree_len(config, exp):
    assert get_speculative_tree_len(config) == exp


# =====================================================================
# attach_speculative_tree_metadata
# =====================================================================

def test_attach_required_fields():
    cfg = MockVllmConfig(3, "[(0,), (0,0), (0,1)]")
    attach_speculative_tree_metadata(cfg)
    spec = cfg.speculative_config
    assert spec.tree_len == 4
    assert spec.tree_depth == 2
    assert spec.drafts_per_level == [1, 2]
    assert spec.cu_drafts_per_level == [0, 1, 3]
    assert spec.child_drafts_per_level == [[1], [2]]
    assert spec.tree_attn_bias.shape == (4, 4)
    assert len(spec.tree_attn_bias_slices) == 2


@pytest.mark.parametrize("cfg_factory", [
    lambda: type("V", (), {"speculative_config": None})(),
    lambda: type("V", (), {
        "speculative_config": MockSpeculativeConfig(num_speculative_tokens=None)
    })(),
])
def test_attach_skips(cfg_factory):
    attach_speculative_tree_metadata(cfg_factory())  # must not raise
