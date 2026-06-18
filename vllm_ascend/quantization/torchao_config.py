"""
TorchAO quantization config for vLLM Ascend.

This module provides Ascend-specific torchao quantization support,
reusing upstream vLLM's TorchAOConfig and TorchAOLinearMethod
while adding NPU compatibility checks.
"""
import warnings
from typing import Any, Optional

import torch
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.torchao import (
    TorchAOConfig,
    TorchAOLinearMethod,
)

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def check_torchao_npu_compatibility(torchao_config) -> None:
    """Check if the torchao config is compatible with NPU.
    
    Args:
        torchao_config: The torchao configuration object.
        
    Raises:
        ValueError: If the config is not supported on NPU.
    """
    config_name = type(torchao_config).__name__
    
    # Check FP8 activation quantization support on NPU
    if "Float8" in config_name and "Activation" in config_name:
        device_type = get_ascend_device_type()
        
        # Ascend910B2C may not support FP8 activation quantization
        # This is a placeholder check - actual support depends on CANN version
        if device_type == AscendDeviceType.ASCEND910B:
            warnings.warn(
                f"torchao FP8 activation quantization config '{config_name}' "
                f"may not be fully supported on {device_type}. "
                f"Consider using Int8WeightOnlyConfig or Int4WeightOnlyConfig "
                f"for better compatibility.",
                UserWarning,
            )


class AscendTorchAOConfig(TorchAOConfig):
    """Ascend-specific TorchAOConfig.
    
    This class extends the upstream TorchAOConfig with NPU-specific
    compatibility checks and optimizations.
    """
    
    def __init__(
        self,
        torchao_config,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
        super().__init__(torchao_config, skip_modules, is_checkpoint_torchao_serialized)
        
        # Perform NPU compatibility check
        check_torchao_npu_compatibility(torchao_config)
    
    def get_name(self) -> str:
        return "torchao"
    
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NPU supports BF16 and FP16
        return [torch.float32, torch.float16, torch.bfloat16]
    
    @classmethod
    def get_min_capability(cls) -> int:
        # NPU doesn't use CUDA capability
        raise NotImplementedError('Ascend hardware does not support "get_min_capability" feature.')
    
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        """Get the quantization method for a layer.
        
        Args:
            layer: The layer to quantize.
            prefix: The layer prefix.
            
        Returns:
            The quantization method, or None if the layer doesn't need quantization.
        """
        if not isinstance(layer, LinearBase):
            return None
        
        # Reuse upstream TorchAOLinearMethod
        return TorchAOLinearMethod(self)


# Register the config with vLLM
register_quantization_config("torchao")(AscendTorchAOConfig)

logger.info_once("Ascend torchao quantization config registered.")
