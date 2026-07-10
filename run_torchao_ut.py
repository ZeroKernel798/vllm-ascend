#!/usr/bin/env python3
"""TorchAO unit test runner — bypasses adapt_patch import errors.

Pre-mocks the vllm-ascend patch modules that trigger API incompatibilities
with the installed vllm version, then runs all torchao tests via unittest.
"""

import sys
import os
from unittest.mock import MagicMock

# ── Pre-mock problematic modules before any vllm-ascend import ──
# modelslim_config needs a proper mock with AscendModelSlimConfig class
_fake_modelslim = MagicMock()
_fake_modelslim.AscendModelSlimConfig = type("AscendModelSlimConfig", (), {
    "override_quantization_method": staticmethod(lambda *a, **kw: None),
})

sys.modules["vllm_ascend.patch"] = MagicMock()
sys.modules["vllm_ascend.patch.platform"] = MagicMock()
sys.modules["vllm_ascend.patch.worker"] = MagicMock()
sys.modules["vllm_ascend.patch.worker.patch_v2"] = MagicMock()
sys.modules["vllm_ascend.patch.platform.patch_dp_device_ids"] = MagicMock()
sys.modules["vllm_ascend.patch.platform.patch_fused_moe"] = MagicMock()
sys.modules["vllm_ascend.patch.platform.patch_use_v2_model_runner"] = MagicMock()
sys.modules["vllm_ascend.quantization.modelslim_config"] = _fake_modelslim
sys.modules["vllm_ascend.quantization.compressed_tensors_config"] = MagicMock()
sys.modules["vllm_ascend.quantization.fp8_config"] = MagicMock()

# ── Now discover and run the test file ──
import pytest

test_file = os.path.join(os.path.dirname(__file__), "tests/ut/quantization/test_torchao.py")
if not os.path.isfile(test_file):
    print(f"ERROR: test file not found at {test_file}")
    sys.exit(1)

sys.exit(pytest.main(["-q", "-v", test_file]))
