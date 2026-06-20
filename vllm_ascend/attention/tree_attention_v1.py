# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Tree attention metadata and builder for Ascend backend.

This module provides tree attention metadata structures and builder
for branching speculative decoding on Ascend NPU devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm_ascend.speculative_token_tree import (
    SpeculativeTokenTreePlan,
    build_speculative_token_tree_plan,
    prepare_speculative_token_tree_attn_bias,
)

logger = logging.getLogger(__name__)


@dataclass
class AscendTreeAttentionMetadata:
    """Tree attention metadata for Ascend backend.
    
    Attributes:
        tree_choices: List of ancestor indices for each draft token.
        tree_depth_counts: Number of draft tokens at each depth.
        tree_attn_bias: Attention bias matrix (tree_len, tree_len).
        tree_attn_mask: Attention mask (tree_len, tree_len), 0=visible, 1=masked.
        experimental_tree_attention_enabled: Whether experimental tree attention is enabled.
        tree_context_len: Context length for tree attention verify.
    """
    tree_choices: list[list[int]] = field(default_factory=list)
    tree_depth_counts: list[int] = field(default_factory=list)
    tree_attn_bias: Optional[torch.Tensor] = None
    tree_attn_mask: Optional[torch.Tensor] = None
    experimental_tree_attention_enabled: bool = False
    tree_context_len: int = 0


class AscendTreeAttentionMetadataBuilder:
    """Builder for AscendTreeAttentionMetadata.
    
    This builder constructs tree attention metadata from speculative config.
    """
    
    def __init__(self, vllm_config: object):
        """Initialize builder.
        
        Args:
            vllm_config: vLLM configuration object.
        """
        self.vllm_config = vllm_config
        self.speculative_config = getattr(vllm_config, 'speculative_config', None)
        
    def build(
        self,
        seq_lens: list[int],
        tree_choices: Optional[list[list[int]]] = None,
    ) -> Optional[AscendTreeAttentionMetadata]:
        """Build tree attention metadata for verify stage.
        
        Args:
            seq_lens: Sequence lengths for each request.
            tree_choices: Tree choices from speculative config.
            
        Returns:
            AscendTreeAttentionMetadata or None if not needed.
        """
        if self.speculative_config is None:
            return None
            
        tree = getattr(self.speculative_config, 'speculative_token_tree', None)
        if tree is None:
            return None
            
        # Check if experimental tree attention is enabled
        from vllm_ascend.speculative_token_tree import (
            is_ascend_experimental_tree_attention_enabled,
        )
        experimental_enabled = is_ascend_experimental_tree_attention_enabled(
            self.vllm_config
        )
        if not experimental_enabled:
            return None
            
        # Build tree attention metadata
        tree_plan = build_speculative_token_tree_plan(tree)
        tree_len = tree_plan.tree_len
        
        # Calculate context length
        # For single request: context_len = seq_len - tree_len
        if len(seq_lens) == 1:
            tree_context_len = seq_lens[0] - tree_len
        else:
            # For batch: use min seq_len as approximation
            tree_context_len = min(seq_lens) - tree_len
            
        # Prepare attention bias
        tree_attn_bias = prepare_speculative_token_tree_attn_bias(tree)
        
        # Convert bias to 2D attention mask (ND layout: [tree_len, tree_len])
        # NPU kernel expects 2D mask (ND layout), not 4D (BNSD layout)
        tree_attn_mask_2d = self._convert_bias_to_2d_mask(tree_attn_bias, dtype=torch.int8)
        
        return AscendTreeAttentionMetadata(
            tree_choices=tree,
            tree_depth_counts=tree_plan.depth_counts,
            tree_attn_bias=tree_attn_bias,
            tree_attn_mask=tree_attn_mask_2d,  # 2D mask for ND layout
            experimental_tree_attention_enabled=True,
            tree_context_len=tree_context_len,
        )
        
    def build_for_drafting(
        self,
        tree_choices: list[list[int]],
        level: int = 0,
    ) -> Optional[AscendTreeAttentionMetadata]:
        """Build tree attention metadata for drafting stage.
        
        Args:
            tree_choices: Tree choices.
            level: Current drafting level.
            
        Returns:
            AscendTreeAttentionMetadata with sliced bias/mask for current level.
        """
        if not tree_choices:
            return None
            
        tree_plan = build_speculative_token_tree_plan(tree_choices)
        
        if level == 0:
            # Root level: no bias needed
            return AscendTreeAttentionMetadata(
                tree_choices=tree_choices,
                tree_depth_counts=tree_plan.depth_counts,
                experimental_tree_attention_enabled=True,
            )
        else:
            # Subsequent levels: slice bias for current level
            tree_attn_bias = prepare_speculative_token_tree_attn_bias(tree_choices)
            # TODO: slice bias for current level
            return AscendTreeAttentionMetadata(
                tree_choices=tree_choices,
                tree_depth_counts=tree_plan.depth_counts,
                tree_attn_bias=tree_attn_bias,
                experimental_tree_attention_enabled=True,
            )
            
    def _convert_bias_to_mask(
        self,
        tree_attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Convert attention bias to Ascend attention mask format.
        
        Args:
            tree_attn_bias: Bias matrix with 0/-inf.
            
        Returns:
            Mask tensor with 0=visible, 1=masked.
        """
        if tree_attn_bias is None:
            return None
            
        # Convert: 0 -> 0 (visible), -inf -> 1 (masked)
        tree_attn_mask = torch.where(
            torch.isinf(tree_attn_bias),
            torch.ones_like(tree_attn_bias, dtype=torch.int8),
            torch.zeros_like(tree_attn_bias, dtype=torch.int8),
        )
        return tree_attn_mask
        
    def _convert_bias_to_2d_mask(
        self,
        tree_attn_bias: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> torch.Tensor:
        """Convert attention bias to 2D attention mask (ND layout).

        NPU kernel expects 2D mask (ND layout), same as standard attention.
        Shape: [tree_len, tree_len]

        Args:
            tree_attn_bias: 2D bias matrix with 0/-inf, shape [tree_len, tree_len].
            dtype: Data type for the mask (default: int8 for NPU compatibility).

        Returns:
            2D attention mask with 0 for visible positions and 1 for masked positions,
            shape [tree_len, tree_len].
        """
        if tree_attn_bias is None:
            return None

        tree_len = tree_attn_bias.shape[0]

        # Create 2D mask: [tree_len, tree_len]
        # NPU expects int8 mask: 0 = visible, 1 = masked
        tree_attn_mask_2d = torch.zeros(
            (tree_len, tree_len), dtype=torch.int8, device=tree_attn_bias.device
        )

        # Apply tree mask: masked positions -> 1
        # tree_attn_bias: 0=visible, -inf=masked
        masked_positions = torch.isinf(tree_attn_bias)
        tree_attn_mask_2d[masked_positions] = 1

        return tree_attn_mask_2d
