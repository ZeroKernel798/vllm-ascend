"""
单元测试：验证 torchao 量化在 Ascend 上的基本功能
"""
import pytest
import torch
from vllm_ascend.quantization.torchao_config import AscendTorchAOConfig
from torchao.quantization import Int8WeightOnlyConfig


class TestAscendTorchAOConfig:
    """测试 AscendTorchAOConfig"""
    
    def test_import(self):
        """测试导入"""
        assert AscendTorchAOConfig is not None
    
    def test_create_config(self):
        """测试创建配置"""
        config = AscendTorchAOConfig(Int8WeightOnlyConfig())
        assert config.get_name() == "torchao"
    
    def test_supported_dtypes(self):
        """测试支持的 dtype"""
        config = AscendTorchAOConfig(Int8WeightOnlyConfig())
        dtypes = config.get_supported_act_dtypes()
        assert torch.float32 in dtypes
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes
    
    def test_get_quant_method(self):
        """测试获取量化方法"""
        config = AscendTorchAOConfig(Int8WeightOnlyConfig())
        
        # 创建一个 mock layer
        class MockLinear:
            pass
        
        layer = MockLinear()
        method = config.get_quant_method(layer, "test")
        # 对于非 LinearBase 层，应该返回 None
        assert method is None


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
