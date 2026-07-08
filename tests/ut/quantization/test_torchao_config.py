#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib.util
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

from tests.ut.base import TestBase
from vllm_ascend.quantization.torchao_config import (
    AscendTorchAOConfig,
    AscendTorchAOLinearMethod,
)
from vllm_ascend.utils import TORCHAO_METHOD

torchao_installed = importlib.util.find_spec("torchao") is not None


def _build_cfg(skip_modules=None) -> AscendTorchAOConfig:
    return AscendTorchAOConfig(
        torchao_config=MagicMock(),
        skip_modules=skip_modules,
    )


class TestAscendTorchAOConfig(TestBase):
    # -- registry / class structure ------------------------------------------

    def test_overrides_upstream_registry(self):
        # Importing vllm_ascend.quantization.torchao_config runs
        # @register_quantization_config("torchao"), which must win over the
        # built-in vLLM TorchAOConfig.
        self.assertIs(get_quantization_config(TORCHAO_METHOD), AscendTorchAOConfig)

    def test_is_subclass_of_upstream(self):
        self.assertTrue(issubclass(AscendTorchAOConfig, TorchAOConfig))

    def test_min_capability_is_not_cuda_sm(self):
        # Upstream returns 75 (CUDA SM 7.5); NPU must not be gated by it.
        self.assertEqual(AscendTorchAOConfig.get_min_capability(), -1)

    def test_get_name(self):
        cfg = _build_cfg()
        self.assertEqual(cfg.get_name(), TORCHAO_METHOD)

    def test_supported_act_dtypes(self):
        cfg = _build_cfg()
        self.assertEqual(
            cfg.get_supported_act_dtypes(),
            [torch.float32, torch.float16, torch.bfloat16],
        )

    def test_quant_description_is_instance_attr(self):
        # Layered code reads ``cfg.quant_description`` without a guard;
        # exposing an empty mapping per instance keeps that code happy and
        # avoids cross-instance state leaks.
        cfg_a = _build_cfg()
        cfg_b = _build_cfg()
        self.assertEqual(cfg_a.quant_description, {})
        cfg_a.quant_description["x"] = 1
        self.assertEqual(cfg_b.quant_description, {})

    def test_repr_contains_class_name(self):
        cfg = _build_cfg(skip_modules=["lm_head"])
        rendered = repr(cfg)
        self.assertIn("AscendTorchAOConfig", rendered)
        self.assertIn("lm_head", rendered)

    # -- override_quantization_method (interaction with ModelSlim) ----------

    def test_modelslim_override_does_not_steal_torchao(self):
        # AscendModelSlimConfig historically claimed ``"ascend"`` whenever the
        # model lacked a ``quant_method`` field on NPU, which silently stole
        # ``--quantization torchao``. The contract we rely on is that
        # ModelSlim must back off when the user requested a different method.
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

        self.assertIsNone(AscendModelSlimConfig.override_quantization_method({}, user_quant=TORCHAO_METHOD))

    def test_inherits_default_no_override(self):
        # We deliberately do NOT define ``override_quantization_method`` on
        # AscendTorchAOConfig: ``"torchao"`` is a built-in vLLM method, and
        # vLLM raises if a built-in method's override returns its own name
        # without being listed in the resolver's ``overrides``. Inherit the
        # base ``QuantizationConfig.override_quantization_method`` (returns
        # ``None``) instead.
        self.assertIsNone(
            AscendTorchAOConfig.override_quantization_method({"quant_method": "torchao"}, user_quant="torchao")
        )

    # -- get_quant_method ----------------------------------------------------

    def test_get_quant_method_non_linear_returns_none(self):
        cfg = _build_cfg()
        not_a_linear = MagicMock()  # not a LinearBase
        self.assertIsNone(cfg.get_quant_method(not_a_linear, prefix="model.foo"))

    def test_get_quant_method_skipped_module_is_unquantized(self):
        cfg = _build_cfg(skip_modules=["lm_head"])
        layer = MagicMock(spec=LinearBase)
        method = cfg.get_quant_method(layer, prefix="lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_get_quant_method_linear_returns_ascend_method(self):
        from torchao.quantization import Int8WeightOnlyConfig

        cfg = AscendTorchAOConfig(torchao_config=Int8WeightOnlyConfig())
        layer = MagicMock(spec=LinearBase)
        method = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.gate_proj")
        self.assertIsInstance(method, AscendTorchAOLinearMethod)
        # The selected method should carry an Ascend-typed config so further
        # ``process_weights_after_loading`` calls keep using Ascend behavior.
        self.assertIsInstance(method.quant_config, AscendTorchAOConfig)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_get_quant_method_module_fqn_to_config_default_none(self):
        # ``ModuleFqnToConfig._default = None`` means "do not quantize layers
        # without a more specific match". Result should be the unquantized
        # method, not a torchao one.
        from torchao.quantization import Int8WeightOnlyConfig, ModuleFqnToConfig

        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "model.layers.0.mlp.gate_proj": Int8WeightOnlyConfig(),
                    "_default": None,
                }
            )
        )
        layer = MagicMock(spec=LinearBase)
        explicit = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.gate_proj")
        self.assertIsInstance(explicit, AscendTorchAOLinearMethod)
        default = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.up_proj")
        self.assertIsInstance(default, UnquantizedLinearMethod)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_get_quant_method_module_fqn_to_config_regex(self):
        from torchao.quantization import Int8WeightOnlyConfig, ModuleFqnToConfig

        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "re:.*\\.gate_proj$": Int8WeightOnlyConfig(),
                    "_default": None,
                }
            )
        )
        layer = MagicMock(spec=LinearBase)
        matched = cfg.get_quant_method(layer, prefix="model.layers.7.mlp.gate_proj")
        self.assertIsInstance(matched, AscendTorchAOLinearMethod)
        unmatched = cfg.get_quant_method(layer, prefix="model.layers.7.mlp.up_proj")
        self.assertIsInstance(unmatched, UnquantizedLinearMethod)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_get_quant_method_module_fqn_exact_match_beats_regex(self):
        # Exact FQN match must take priority over a regex that would also
        # match the same prefix. Mirrors upstream's documented lookup order.
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            ModuleFqnToConfig,
        )

        exact_cfg = Int4WeightOnlyConfig()
        regex_cfg = Int8WeightOnlyConfig()
        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "model.layers.0.mlp.gate_proj": exact_cfg,
                    "re:.*\\.gate_proj$": regex_cfg,
                    "_default": None,
                }
            )
        )
        selected = cfg._select_torchao_config_for_module("model.layers.0.mlp.gate_proj")
        self.assertIs(selected, exact_cfg)
        # A different layer with no exact entry falls through to the regex.
        selected_regex = cfg._select_torchao_config_for_module("model.layers.1.mlp.gate_proj")
        self.assertIs(selected_regex, regex_cfg)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_get_quant_method_unsupported_prefix_falls_to_default_none(self):
        # A prefix that matches no entry and no regex falls to ``_default``.
        from torchao.quantization import Int8WeightOnlyConfig, ModuleFqnToConfig

        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "re:.*\\.gate_proj$": Int8WeightOnlyConfig(),
                    "_default": None,
                }
            )
        )
        self.assertIsNone(cfg._select_torchao_config_for_module("model.lm_head"))

    # -- apply_vllm_mapper ---------------------------------------------------

    def test_apply_vllm_mapper_translates_skip_modules(self):
        # ``apply_vllm_mapper`` must rewrite ``skip_modules`` from HF naming
        # to vLLM naming so ``should_skip`` matches the layers the model
        # actually instantiates (e.g. fused ``gate_up_proj``).
        from vllm.model_executor.models.utils import WeightsMapper

        cfg = _build_cfg(skip_modules=["model.lm_head", "model.embed_tokens"])
        mapper = WeightsMapper(
            orig_to_new_substr={"model.": "language_model."},
        )
        cfg.apply_vllm_mapper(mapper)
        self.assertEqual(
            cfg.skip_modules,
            ["language_model.lm_head", "language_model.embed_tokens"],
        )

    def test_apply_vllm_mapper_no_skip_modules_is_noop(self):
        from vllm.model_executor.models.utils import WeightsMapper

        cfg = _build_cfg()  # skip_modules defaults to []
        mapper = WeightsMapper(orig_to_new_substr={"foo": "bar"})
        cfg.apply_vllm_mapper(mapper)
        self.assertEqual(cfg.skip_modules, [])

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_apply_vllm_mapper_translates_module_fqn_to_config(self):
        # ``ModuleFqnToConfig`` keys should be translated, except ``re:``
        # regex patterns and the ``_default`` sentinel.
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            ModuleFqnToConfig,
        )
        from vllm.model_executor.models.utils import WeightsMapper

        c8 = Int8WeightOnlyConfig()
        c4 = Int4WeightOnlyConfig()
        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "model.layers.0.mlp.gate_proj": c8,
                    "model.layers.0.mlp.up_proj": c8,
                    "model.layers.0.self_attn.q_proj": c4,
                    "re:.*\\.k_proj$": c4,
                    "_default": None,
                }
            )
        )
        mapper = WeightsMapper(
            orig_to_new_substr={"model.layers.": "language_model.layers."},
        )
        cfg.apply_vllm_mapper(mapper)

        new_map = cfg.torchao_config.module_fqn_to_config
        # Translated keys
        self.assertIn("language_model.layers.0.mlp.gate_proj", new_map)
        self.assertIn("language_model.layers.0.mlp.up_proj", new_map)
        self.assertIn("language_model.layers.0.self_attn.q_proj", new_map)
        # Regex and _default preserved verbatim
        self.assertIn("re:.*\\.k_proj$", new_map)
        self.assertIn("_default", new_map)
        # Original (HF-style) keys removed
        self.assertNotIn("model.layers.0.mlp.gate_proj", new_map)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_apply_vllm_mapper_only_runs_get_quant_method_after(self):
        # End-to-end: build a config with HF-style FQNs, run
        # ``apply_vllm_mapper``, then verify ``get_quant_method`` selects
        # the right per-layer config using the post-mapper FQN.
        from torchao.quantization import Int8WeightOnlyConfig, ModuleFqnToConfig
        from vllm.model_executor.models.utils import WeightsMapper

        target = Int8WeightOnlyConfig()
        cfg = AscendTorchAOConfig(
            torchao_config=ModuleFqnToConfig(
                {
                    "model.layers.0.mlp.gate_proj": target,
                    "_default": None,
                }
            )
        )
        # vLLM fuses gate_proj + up_proj into gate_up_proj; assume the
        # mapper rewrites that for our skip-config lookup.
        mapper = WeightsMapper(
            orig_to_new_substr={"mlp.gate_proj": "mlp.gate_up_proj"},
        )
        cfg.apply_vllm_mapper(mapper)

        layer = MagicMock(spec=LinearBase)
        # vLLM layer FQN (post-mapper)
        method = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.gate_up_proj")
        self.assertIsInstance(method, AscendTorchAOLinearMethod)
        # Pre-mapper FQN (HF naming) should now miss → unquantized
        skipped = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.gate_proj")
        self.assertIsInstance(skipped, UnquantizedLinearMethod)

    def test_apply_vllm_mapper_with_opt_style_prefix_rewrite(self):
        # Realistic OPT case: HF param names start with ``decoder.`` while
        # vLLM uses ``model.decoder.``. Confirm a prefix-rewrite mapper
        # translates ``skip_modules`` correctly so ``should_skip`` matches
        # the actual vLLM FQN at quant-method resolution time.
        from vllm.model_executor.models.utils import WeightsMapper

        cfg = _build_cfg(skip_modules=["decoder.embed_tokens"])
        mapper = WeightsMapper(orig_to_new_prefix={"decoder.": "model.decoder."})
        cfg.apply_vllm_mapper(mapper)
        self.assertEqual(cfg.skip_modules, ["model.decoder.embed_tokens"])

        layer = MagicMock(spec=LinearBase)
        # The vLLM-style FQN should now be skipped.
        method = cfg.get_quant_method(layer, prefix="model.decoder.embed_tokens")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_apply_vllm_mapper_drops_keys_mapped_to_none(self):
        # ``WeightsMapper`` returning ``None`` for a key means "ignore".
        # ``apply_list`` then drops that entry entirely.
        from vllm.model_executor.models.utils import WeightsMapper

        cfg = _build_cfg(skip_modules=["a.b.c", "skip_me", "x.y"])
        mapper = WeightsMapper(orig_to_new_substr={"skip_me": None})
        cfg.apply_vllm_mapper(mapper)
        # 'skip_me' is dropped; the others pass through unchanged.
        self.assertEqual(cfg.skip_modules, ["a.b.c", "x.y"])

    # -- from_config ---------------------------------------------------------

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_from_config_int8_weight_only(self):
        from torchao.core.config import config_to_dict
        from torchao.quantization import Int8WeightOnlyConfig

        model_config = {
            "quant_type": {"default": config_to_dict(Int8WeightOnlyConfig())},
        }
        cfg = AscendTorchAOConfig.from_config(model_config)
        self.assertIsInstance(cfg, AscendTorchAOConfig)
        self.assertEqual(cfg.get_name(), TORCHAO_METHOD)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_from_config_respects_quant_method_torchao(self):
        # When ``quant_method = "torchao"`` is present in the HF config,
        # ``from_config`` sets ``is_checkpoint_torchao_serialized = True``
        # (aligning with upstream vLLM behavior — no Ascend-specific override).
        from torchao.core.config import config_to_dict
        from torchao.quantization import Int8WeightOnlyConfig

        model_config = {
            "quant_method": "torchao",
            "quant_type": {"default": config_to_dict(Int8WeightOnlyConfig())},
        }
        cfg = AscendTorchAOConfig.from_config(model_config)
        self.assertTrue(cfg.is_checkpoint_torchao_serialized)


class TestAscendTorchAOLinearMethod(TestBase):
    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_apply_uses_f_linear(self):
        # With a plain (unquantized) weight, apply() must behave like F.linear.
        from torchao.quantization import Int8WeightOnlyConfig

        cfg = AscendTorchAOConfig(torchao_config=Int8WeightOnlyConfig())
        method = AscendTorchAOLinearMethod(cfg)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        x = torch.randn(2, 4, dtype=torch.float32)
        out = method.apply(layer, x)
        expected = torch.nn.functional.linear(x, layer.weight)
        self.assertTrue(torch.allclose(out, expected))

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_create_weights_plain_when_not_serialized(self):
        from torchao.quantization import Int8WeightOnlyConfig

        cfg = AscendTorchAOConfig(
            torchao_config=Int8WeightOnlyConfig(),
            is_checkpoint_torchao_serialized=False,
        )
        method = AscendTorchAOLinearMethod(cfg)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=4,
            output_partition_sizes=[8],
            input_size=4,
            output_size=8,
            params_dtype=torch.float32,
        )
        # ``create_weights`` should leave the placeholder as a plain Parameter
        # (no TorchAO subclass) when the checkpoint is not serialized; the
        # subclass is only built later in ``process_weights_after_loading``.
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertIs(type(layer.weight.data), torch.Tensor)
        self.assertEqual(tuple(layer.weight.shape), (8, 4))

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_process_weights_after_loading_online_quantizes(self):
        # On the online-quant path with the native int8 kernel disabled, a
        # plain weight must be converted into a TorchAO tensor subclass after
        # load. (With VLLM_ASCEND_TORCHAO_NATIVE_KERNEL=1 the subclass is
        # further lowered to a plain int8 weight for npu_weight_quant_batchmatmul;
        # that path is covered separately.)
        from torchao.quantization import Int8WeightOnlyConfig

        cfg = AscendTorchAOConfig(
            torchao_config=Int8WeightOnlyConfig(),
            is_checkpoint_torchao_serialized=False,
        )
        method = AscendTorchAOLinearMethod(cfg)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        with patch.dict(
            "os.environ",
            {"VLLM_ASCEND_TORCHAO_NATIVE_KERNEL": "0"},
            clear=False,
        ):
            method.process_weights_after_loading(layer)
        # After in-place online quantize, the weight should be a torchao
        # subclass (i.e. not the plain ``torch.Tensor`` data class).
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertIsNot(type(layer.weight.data), torch.Tensor)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_process_weights_after_loading_calls_hardware_packing_online(self):
        # Both the online and serialized paths must run torchao's
        # hardware-aware packing converter so that a future torchao release
        # adding NPU-specific packing automatically picks up.
        from torchao.quantization import Int8WeightOnlyConfig

        cfg = AscendTorchAOConfig(
            torchao_config=Int8WeightOnlyConfig(),
            is_checkpoint_torchao_serialized=False,
        )
        method = AscendTorchAOLinearMethod(cfg)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        with patch(
            "vllm_ascend.quantization.torchao_config.convert_to_packed_tensor_based_on_current_hardware",
            side_effect=lambda t: t,
        ) as packer:
            method.process_weights_after_loading(layer)
        self.assertEqual(packer.call_count, 1)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_process_weights_after_loading_serialized_preserves_attrs(self):
        # In the serialized path we wrap the loaded subclass in a fresh
        # Parameter (matching upstream); the wrap drops dynamically-added
        # attrs, so ``_restore_weight_attrs`` must put them back.
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        cfg = AscendTorchAOConfig(
            torchao_config=Int8WeightOnlyConfig(),
            is_checkpoint_torchao_serialized=True,
        )
        method = AscendTorchAOLinearMethod(cfg)

        # Build a "loaded" serialized weight (a TorchAO subclass) and
        # decorate it with vLLM-style attrs as the framework loader would.
        dense = torch.nn.Linear(4, 8, bias=False)
        quantize_(dense, Int8WeightOnlyConfig())
        layer = torch.nn.Module()
        layer.weight = dense.weight
        layer.weight.input_dim = 1
        layer.weight.output_dim = 0

        with patch(
            "vllm_ascend.quantization.torchao_config.convert_to_packed_tensor_based_on_current_hardware",
            side_effect=lambda t: t,
        ) as packer:
            method.process_weights_after_loading(layer)

        self.assertEqual(packer.call_count, 1)
        self.assertEqual(getattr(layer.weight, "input_dim", None), 1)
        self.assertEqual(getattr(layer.weight, "output_dim", None), 0)

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_dequant_fallback_replaces_subclass_weight(self):
        # Simulate a weight whose F.linear dispatch fails, and confirm the
        # fallback dequantizes to a plain parameter and still computes linear.
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        cfg = AscendTorchAOConfig(torchao_config=Int8WeightOnlyConfig())
        method = AscendTorchAOLinearMethod(cfg)

        dense = torch.nn.Linear(4, 8, bias=False)
        quantize_(dense, Int8WeightOnlyConfig())

        layer = torch.nn.Module()
        layer.weight = dense.weight  # TorchAO tensor subclass
        self.assertIsNot(type(layer.weight.data), torch.Tensor)

        method._replace_with_dequantized_parameter(layer)
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertIs(type(layer.weight.data), torch.Tensor)
        self.assertEqual(tuple(layer.weight.shape), (8, 4))

        x = torch.randn(2, 4, dtype=layer.weight.dtype)
        out = torch.nn.functional.linear(x, layer.weight)
        self.assertEqual(tuple(out.shape), (2, 8))

    @pytest.mark.skipif(not torchao_installed, reason="torchao is not installed")
    def test_full_lifecycle_create_load_process_apply(self):
        # End-to-end on CPU: simulate create_weights → loader copy_ →
        # process_weights_after_loading → apply, then check the output is
        # close to the unquantized reference (within int8 weight-only error).
        from torchao.quantization import Int8WeightOnlyConfig

        torch.manual_seed(0)
        in_features, out_features = 16, 32
        cfg = AscendTorchAOConfig(
            torchao_config=Int8WeightOnlyConfig(),
            is_checkpoint_torchao_serialized=False,
        )
        method = AscendTorchAOLinearMethod(cfg)

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.float32,
        )

        # Stand-in for vLLM's weight loader: copy real values into the
        # placeholder Parameter as the safetensors loader would.
        reference_weight = torch.randn(out_features, in_features, dtype=torch.float32)
        with torch.no_grad():
            layer.weight.data.copy_(reference_weight)

        # This lifecycle test exercises the portable subclass + F.linear path
        # (CPU). The native int8 kernel path needs an NPU and is validated by
        # the E2E benchmark, so disable it here.
        with patch.dict(
            "os.environ",
            {"VLLM_ASCEND_TORCHAO_NATIVE_KERNEL": "0"},
            clear=False,
        ):
            method.process_weights_after_loading(layer)
            self.assertIsNot(type(layer.weight.data), torch.Tensor)  # subclass

            x = torch.randn(2, in_features, dtype=torch.float32)
            out = method.apply(layer, x)
        ref = torch.nn.functional.linear(x, reference_weight)

        # Int8 weight-only quantization is lossy but should be a close
        # approximation of the reference matmul. Use a generous tolerance:
        # the test guards that the lifecycle wires together correctly, not
        # numerical accuracy of int8wo itself.
        self.assertEqual(tuple(out.shape), (2, out_features))
        self.assertTrue(
            torch.allclose(out, ref, atol=0.5, rtol=0.05),
            f"int8wo output diverged too far from fp32 reference: max_abs={(out - ref).abs().max().item():.4f}",
        )

    def test_dequantize_weight_is_no_op_on_plain_tensor(self):
        # ``_dequantize_weight`` must be tolerant of plain tensors (those have
        # no torchao subclass). It should pass them through unchanged unless
        # dtype/device are requested.
        plain = torch.randn(4, 8, dtype=torch.float32)
        out = AscendTorchAOLinearMethod._dequantize_weight(plain)
        self.assertIs(type(out), torch.Tensor)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (4, 8))

    def test_dequantize_weight_applies_dtype_and_device(self):
        plain = torch.randn(4, 8, dtype=torch.float32)
        out = AscendTorchAOLinearMethod._dequantize_weight(plain, dtype=torch.bfloat16, device=torch.device("cpu"))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.device.type, "cpu")

    def test_apply_fallback_triggers_on_runtime_error(self):
        # Patch F.linear to raise on the first call, then succeed; confirm the
        # method catches the failure, dequantizes, and retries.
        cfg = AscendTorchAOConfig(torchao_config=MagicMock())
        method = AscendTorchAOLinearMethod(cfg)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        x = torch.randn(2, 4, dtype=torch.float32)

        call_count = {"n": 0}
        original_linear = torch.nn.functional.linear

        def flaky_linear(inp, weight, bias=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated NPU dispatch miss")
            return original_linear(inp, weight, bias)

        with patch(
            "vllm_ascend.quantization.torchao_config.F.linear",
            side_effect=flaky_linear,
        ):
            out = method.apply(layer, x)
        # Two calls: one that raised, one after dequant fallback.
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_apply_fallback_disabled_propagates_error(self):
        # When the fallback env flag is 0, a dispatch failure must propagate
        # so the user sees the real kernel error instead of silent dequant.
        cfg = AscendTorchAOConfig(torchao_config=MagicMock())
        method = AscendTorchAOLinearMethod(cfg)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        x = torch.randn(2, 4, dtype=torch.float32)

        with (
            patch.dict(
                "os.environ",
                {"VLLM_ASCEND_TORCHAO_ALLOW_DEQUANT_FALLBACK": "0"},
                clear=False,
            ),
            patch(
                "vllm_ascend.quantization.torchao_config.F.linear",
                side_effect=RuntimeError("simulated NPU dispatch miss"),
            ),
            self.assertRaises(RuntimeError),
        ):
            method.apply(layer, x)


class _FakeTensorImpl:
    """Minimal stand-in for a TorchAO ``PlainAQTTensorImpl``.

    Carries the three attributes ``_extract_int8wo_params`` reads
    (``int_data``, ``scale``, ``zero_point``) so the extraction contract can
    be exercised on CPU without torchao or an NPU.
    """

    def __init__(self, int_data, scale, zero_point=None):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point


class _FakeAQT:
    """Stand-in for an ``AffineQuantizedTensor`` exposing ``tensor_impl``."""

    def __init__(self, tensor_impl):
        self.tensor_impl = tensor_impl


class TestExtractInt8woParams(TestBase):
    """Unit tests for the native int8 weight-only extraction contract.

    These are pure-tensor tests (no torchao, no NPU): they validate which
    quantized layouts are accepted into the native
    ``npu_weight_quant_batchmatmul`` fast path and which must fall back to the
    dequant path by returning ``None``.
    """

    extract = staticmethod(AscendTorchAOLinearMethod._extract_int8wo_params)

    def test_symmetric_int8_per_channel_extracts(self):
        out_f, in_f = 8, 4
        int_data = torch.randint(-128, 127, (out_f, in_f), dtype=torch.int8)
        scale = torch.rand(out_f, dtype=torch.float32)
        zero_point = torch.zeros(out_f, dtype=torch.int64)
        weight = _FakeAQT(_FakeTensorImpl(int_data, scale, zero_point))

        result = self.extract(weight)
        self.assertIsNotNone(result)
        got_int, got_scale = result
        self.assertIs(got_int, int_data)
        self.assertIs(got_scale, scale)

    def test_zero_point_none_treated_as_symmetric(self):
        out_f, in_f = 8, 4
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(out_f, in_f, dtype=torch.int8),
                torch.rand(out_f, dtype=torch.float32),
                zero_point=None,
            )
        )
        self.assertIsNotNone(self.extract(weight))

    def test_plain_tensor_without_tensor_impl_returns_none(self):
        # A plain (non-quantized) tensor has no ``tensor_impl``.
        self.assertIsNone(self.extract(torch.randn(8, 4)))

    def test_non_int8_dtype_returns_none(self):
        out_f, in_f = 8, 4
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(out_f, in_f, dtype=torch.int32),
                torch.rand(out_f, dtype=torch.float32),
            )
        )
        self.assertIsNone(self.extract(weight))

    def test_scale_count_mismatch_returns_none(self):
        # Per-tensor (single scalar) or otherwise non per-output-channel scale
        # does not match the kernel's antiquant contract.
        out_f, in_f = 8, 4
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(out_f, in_f, dtype=torch.int8),
                torch.rand(1, dtype=torch.float32),
            )
        )
        self.assertIsNone(self.extract(weight))

    def test_asymmetric_nonzero_zero_point_returns_none(self):
        out_f, in_f = 8, 4
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(out_f, in_f, dtype=torch.int8),
                torch.rand(out_f, dtype=torch.float32),
                zero_point=torch.ones(out_f, dtype=torch.int64),
            )
        )
        self.assertIsNone(self.extract(weight))

    def test_one_dim_int_data_returns_none(self):
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(8, dtype=torch.int8),
                torch.rand(8, dtype=torch.float32),
            )
        )
        self.assertIsNone(self.extract(weight))


class _FakeLAQT:
    """Stand-in for a ``LinearActivationQuantizedTensor`` (W8A8): the real int8
    weight lives in ``original_weight_tensor`` (an AffineQuantizedTensor)."""

    def __init__(self, original_weight_tensor):
        self.original_weight_tensor = original_weight_tensor


class TestExtractW8a8Params(TestBase):
    """Unit tests for the W8A8 dynamic-activation extraction contract.

    W8A8 wraps the weight one level deeper than int8wo; extraction must reach
    into ``original_weight_tensor`` and reuse the int8wo logic.
    """

    extract = staticmethod(AscendTorchAOLinearMethod._extract_w8a8_params)

    def test_w8a8_nested_int8_extracts(self):
        out_f, in_f = 8, 4
        int_data = torch.randint(-128, 127, (out_f, in_f), dtype=torch.int8)
        scale = torch.rand(out_f, dtype=torch.float32)
        weight = _FakeLAQT(
            _FakeAQT(_FakeTensorImpl(int_data, scale, torch.zeros(out_f, dtype=torch.int64)))
        )
        result = self.extract(weight)
        self.assertIsNotNone(result)
        got_int, got_scale = result
        self.assertIs(got_int, int_data)
        self.assertIs(got_scale, scale)

    def test_w8a8_without_original_weight_returns_none(self):
        # A plain int8wo tensor (no original_weight_tensor) is not W8A8.
        weight = _FakeAQT(
            _FakeTensorImpl(
                torch.zeros(8, 4, dtype=torch.int8),
                torch.rand(8, dtype=torch.float32),
            )
        )
        self.assertIsNone(self.extract(weight))

    def test_w8a8_inner_asymmetric_returns_none(self):
        out_f, in_f = 8, 4
        weight = _FakeLAQT(
            _FakeAQT(_FakeTensorImpl(
                torch.zeros(out_f, in_f, dtype=torch.int8),
                torch.rand(out_f, dtype=torch.float32),
                zero_point=torch.ones(out_f, dtype=torch.int64),
            ))
        )
        self.assertIsNone(self.extract(weight))


class _FakeIntx:
    """Stand-in for an ``IntxUnpackedToInt8Tensor`` (W4A16): int4 stored as
    int8 in ``qdata``, with per-channel ``scale`` and ``zero_point``."""

    def __init__(self, qdata, scale, zero_point=None):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point


class TestExtractW4a16Params(TestBase):
    """Unit tests for the W4A16 (int4 weight-only) extraction contract."""

    extract = staticmethod(AscendTorchAOLinearMethod._extract_w4a16_params)

    def test_symmetric_int4_per_channel_extracts(self):
        out_f, in_f = 8, 4
        qdata = torch.randint(-8, 7, (out_f, in_f), dtype=torch.int8)
        scale = torch.rand(out_f, 1, dtype=torch.float32)
        weight = _FakeIntx(qdata, scale, torch.zeros(out_f, 1, dtype=torch.int8))
        result = self.extract(weight)
        self.assertIsNotNone(result)
        got_q, got_scale = result
        self.assertIs(got_q, qdata)
        self.assertIs(got_scale, scale)

    def test_out_of_int4_range_returns_none(self):
        # int8-ranged values (>7) are not int4 weights.
        out_f, in_f = 8, 4
        weight = _FakeIntx(
            torch.full((out_f, in_f), 50, dtype=torch.int8),
            torch.rand(out_f, 1, dtype=torch.float32),
        )
        self.assertIsNone(self.extract(weight))

    def test_per_group_scale_returns_none(self):
        # More than one scale per output row = per-group, not supported.
        out_f, in_f = 8, 16
        weight = _FakeIntx(
            torch.zeros(out_f, in_f, dtype=torch.int8),
            torch.rand(out_f, 4, dtype=torch.float32),  # 4 groups per row
        )
        self.assertIsNone(self.extract(weight))

    def test_asymmetric_returns_none(self):
        out_f, in_f = 8, 4
        weight = _FakeIntx(
            torch.zeros(out_f, in_f, dtype=torch.int8),
            torch.rand(out_f, 1, dtype=torch.float32),
            zero_point=torch.ones(out_f, 1, dtype=torch.int8),
        )
        self.assertIsNone(self.extract(weight))

    def test_plain_tensor_returns_none(self):
        self.assertIsNone(self.extract(torch.randn(8, 4)))
