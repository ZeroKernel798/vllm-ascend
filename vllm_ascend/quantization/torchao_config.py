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
"""TorchAO quantization support for Ascend NPU (``--quantization torchao``).

The upstream :class:`vllm.model_executor.layers.quantization.torchao.TorchAOConfig`
already does most of the right thing on non-CUDA hardware:
``convert_to_packed_tensor_based_on_current_hardware`` is a defensive no-op
unless the tensor is an ``Int4Tensor`` on CUDA SM>=9.0 with MSLK available.
What the upstream code gets wrong on Ascend is:

1. ``get_min_capability`` returns ``75`` (CUDA SM 7.5), which would gate the
   config off when vLLM compares it to ``current_platform.get_device_capability()``
   on NPU.
2. ``apply()`` calls ``F.linear`` directly. On NPU, some TorchAO tensor
   subclasses do not have a ``__torch_dispatch__`` rule for ``aten.linear`` /
   ``aten.mm`` and the call would crash with a kernel-not-found error.
3. ``from_config`` follows ``quant_method`` blindly to set
   ``is_checkpoint_torchao_serialized``. vLLM's TorchAO-flattened safetensors
   loader (``unflatten_tensor_state_dict``) has not been validated on Ascend
   yet, so we default to the online-quant code path that runs end-to-end on
   NPU today.
4. ``AscendModelSlimConfig.override_quantization_method`` historically claimed
   ``"ascend"`` for any NPU model whose ``config.json`` lacks an explicit
   ``quant_method`` field, silently stealing user-requested
   ``--quantization torchao``. That is fixed in ``modelslim_config.py``
   (companion change) so this module does not need its own override.

This module therefore registers an Ascend-specific subclass that:

* Drops the CUDA capability gate (``get_min_capability`` returns ``-1``).
* Adds a dequant-fallback path in ``apply()`` for TorchAO subclasses that
  have no NPU dispatch.
* Forces the online-quant code path by default (with an env opt-out).
* Implements ``apply_vllm_mapper`` to translate HF parameter names
  (``modules_to_not_convert``, ``ModuleFqnToConfig`` keys) into the vLLM
  layer FQNs that ``get_quant_method`` actually sees. Upstream skips this
  hook, which is a known correctness gap on models that fuse projections
  (LLaMA's ``gate_up_proj``, Qwen's ``qkv_proj`` etc.).
* Carries an empty ``quant_description`` mapping so other vllm-ascend
  modules that read that attribute without a ``getattr`` guard (notably
  ``ops/layernorm.py``) keep working.

Otherwise the implementation is a thin port of upstream — the
``process_weights_after_loading`` lifecycle, the ``ModuleFqnToConfig``
resolver, the ``should_skip`` rules, and the ``torchao_quantize_param_data``
helper are all reused verbatim.

This is the "compatible & runnable" layer of the integration. Ascend-native
fast paths (W8A16 / W8A8) for specific TorchAO configs can be layered on top
without changing this module's public contract.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.torchao import (
    TorchAOConfig,
    _get_weight_attrs,
    _restore_weight_attrs,
    convert_to_packed_tensor_based_on_current_hardware,
    should_skip,
    torchao_quantize_param_data,
)
from vllm.model_executor.utils import set_weight_attrs

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import TORCHAO_METHOD

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

#
# Environment flags
# -----------------
#
# All Ascend-specific torchao env vars are registered in
# ``vllm_ascend/envs.py``; their semantics are documented there. This module
# reads them via ``envs_ascend`` for consistency with the rest of vllm-ascend.
#
#   VLLM_ASCEND_TORCHAO_ALLOW_DEQUANT_FALLBACK  (default 1)
#   VLLM_ASCEND_TORCHAO_NATIVE_KERNEL           (default 1)
#   VLLM_ASCEND_TORCHAO_CONFIG_TYPE             (default "int8wo")
#

# ``vllm-ascend`` modules read ``quant_config.quant_description`` directly
# without a ``getattr`` guard. Known sites:
#   * vllm_ascend/ops/layernorm.py: ``"norm.bias" in name for name in
#     vllm_config.quant_config.quant_description`` (anti-method-m4 detection).
# An empty mapping keeps that init code happy without changing semantics.
_EMPTY_QUANT_DESCRIPTION: dict[str, Any] = {}


_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str, *args: object) -> None:
    """Emit ``message`` through the logger at most once per rendered text."""
    rendered = message % args if args else message
    if rendered not in _WARNED_MESSAGES:
        _WARNED_MESSAGES.add(rendered)
        logger.warning(rendered)


@register_quantization_config(TORCHAO_METHOD)
class AscendTorchAOConfig(TorchAOConfig):
    """Ascend override of vLLM's ``--quantization torchao`` config.

    Inherits :meth:`from_config_file`, :meth:`from_config_dict_json`,
    :meth:`get_supported_act_dtypes`, :meth:`get_config_filenames` and the
    ``ModuleFqnToConfig`` parsing helpers from upstream. Overrides only the
    pieces that hard-code CUDA behavior or that conflict with Ascend's native
    quant-method resolution.
    """

    def __init__(
        self,
        torchao_config: Any = None,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
        # Provide a default torchao_config if None (required for no-arg instantiation
        # by get_quant_config() in weight_utils.py).
        if torchao_config is None:
            import torchao.quantization as tq

            config_type = envs_ascend.VLLM_ASCEND_TORCHAO_CONFIG_TYPE
            if config_type == "int8wo":
                torchao_config = tq.Int8WeightOnlyConfig()
            elif config_type == "w4a8":
                torchao_config = tq.Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=torch.int4,
                )
            elif config_type == "w8a8":
                torchao_config = tq.Int8DynamicActivationInt8WeightConfig()
            elif config_type == "int4wo":
                torchao_config = tq.Int4WeightOnlyConfig()
            elif config_type == "intx4wo":
                torchao_config = tq.IntxWeightOnlyConfig(
                    weight_dtype=torch.int4,
                )
            else:
                torchao_config = tq.Int8WeightOnlyConfig()  # safe default

        super().__init__(
            torchao_config=torchao_config,
            skip_modules=skip_modules,
            is_checkpoint_torchao_serialized=is_checkpoint_torchao_serialized,
        )
        # Per-instance copy so test code that mutates ``cfg.quant_description``
        # does not leak into other instances.
        self.quant_description: dict[str, Any] = dict(_EMPTY_QUANT_DESCRIPTION)

    def __repr__(self) -> str:
        return (
            f"AscendTorchAOConfig({self.torchao_config!r}, "
            f"skip_modules={self.skip_modules!r}, "
            f"is_checkpoint_torchao_serialized="
            f"{self.is_checkpoint_torchao_serialized!r})"
        )

    def get_name(self) -> Any:
        return TORCHAO_METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        # Upstream returns 75 (CUDA SM 7.5). NPU must not be gated by a CUDA
        # capability number; ``-1`` reads as "no minimum".
        return -1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AscendTorchAOConfig:
        """Parse a torchao quant config from an HF model config dict.

        Delegates entirely to upstream :meth:`TorchAOConfig.from_config`, which
        constructs an :class:`AscendTorchAOConfig` instance via ``cls(...)``.
        ``is_checkpoint_torchao_serialized`` is determined by whether the HF
        config declares ``quant_method=torchao`` — same as upstream CUDA
        behavior. No Ascend-specific override is needed.
        """
        return super().from_config(config)  # type: ignore[return-value]


    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Translate HF parameter names to vLLM names in our skip lists.

        vLLM may rename modules during construction (e.g., fusing
        ``gate_proj`` + ``up_proj`` into ``gate_up_proj``, or fusing QKV
        projections into ``qkv_proj``). The HF ``modules_to_not_convert``
        list and ``ModuleFqnToConfig`` keys are in HF naming; without
        translation they would not match the vLLM layer FQNs that
        :meth:`get_quant_method` sees.

        Upstream :class:`vllm.model_executor.layers.quantization.torchao.TorchAOConfig`
        does not implement this hook (a known correctness gap shared with
        CUDA). Ascend implements it because several Ascend-supported
        models — Qwen, LLaMA, GLM, MiniMax — rely on
        ``hf_to_vllm_mapper`` to fuse projections.

        Notes:

        * Called by ``configure_quant_config`` before any
          :meth:`get_quant_method` calls, so all per-layer paths see the
          translated names.
        * Mutates ``self.skip_modules`` and ``self.torchao_config`` in
          place. Safe because each ``from_config`` invocation produces a
          fresh :class:`AscendTorchAOConfig`; configs are never shared
          across LLMs in vLLM.
        * For ``ModuleFqnToConfig``, ``re:`` regex patterns and the
          ``_default`` sentinel are not FQNs and are passed through
          unchanged.
        """
        if self.skip_modules:
            self.skip_modules = hf_to_vllm_mapper.apply_list(self.skip_modules)

        try:
            from torchao.quantization import ModuleFqnToConfig
        except ImportError:
            return

        if isinstance(self.torchao_config, ModuleFqnToConfig):
            module_fqn_map = self.torchao_config.module_fqn_to_config
            translatable = {
                k: v
                for k, v in module_fqn_map.items()
                if not k.startswith("re:") and k != "_default"
            }
            non_translatable = {
                k: v
                for k, v in module_fqn_map.items()
                if k.startswith("re:") or k == "_default"
            }
            translated = hf_to_vllm_mapper.apply_dict(translatable)
            # Mutate in place so the same ``ModuleFqnToConfig`` instance
            # (and its updated map) is visible to every
            # :meth:`get_quant_method` call that re-reads it.
            self.torchao_config.module_fqn_to_config = {
                **translated,
                **non_translatable,
            }

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        **kwargs: Any,
    ) -> QuantizeMethodBase | None:
        # NOTE: ``**kwargs`` intentionally widens the upstream base-class
        # signature ``get_quant_method(self, layer, prefix)``. vllm-ascend's
        # own FusedMoE layer calls ``get_quant_method(self, layer_name,
        # tid2eid=...)`` (see ops/fused_moe/fused_moe.py), and
        # AscendModelSlimConfig follows the same convention. Keep ``**kwargs``
        # when rebasing onto upstream — it is required by the Ascend MoE path,
        # not an accidental divergence.
        if not isinstance(layer, LinearBase):
            # MoE / FusedMoE layers — the container is quantized by a
            # FusedMoEMethodBase, not a linear method. Return the Ascend
            # unquantized MoE method so expert linear layers are still
            # quantized via their own get_quant_method calls.
            if isinstance(layer, FusedMoE):
                from vllm_ascend.ops.fused_moe.fused_moe import (
                    AscendUnquantizedFusedMoEMethod,
                )
                return AscendUnquantizedFusedMoEMethod(
                    getattr(layer, "moe_config", None),
                    tid2eid=kwargs.get("tid2eid"),
                )
            # Embeddings / attention / non-linear: defer to vLLM default.
            return None

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        per_layer_config = self._select_torchao_config_for_module(prefix)
        if per_layer_config is None:
            # ``ModuleFqnToConfig`` explicitly maps this module to ``None``,
            # meaning "do not quantize this layer".
            return UnquantizedLinearMethod()

        if per_layer_config is self.torchao_config:
            return AscendTorchAOLinearMethod(self)

        # ``ModuleFqnToConfig`` resolved to a layer-specific config — wrap it
        # in our own subclass so per-layer ``process_weights_after_loading``
        # / ``apply`` go through Ascend-aware code, not upstream's CUDA path.
        return AscendTorchAOLinearMethod(
            type(self)(
                per_layer_config,
                self.skip_modules,
                self.is_checkpoint_torchao_serialized,
            )
        )

    def _select_torchao_config_for_module(self, prefix: str) -> Any | None:
        """Resolve the TorchAO config that applies to a specific module FQN.

        For non-``ModuleFqnToConfig`` configs this returns ``self.torchao_config``
        unchanged. For ``ModuleFqnToConfig`` it implements upstream's lookup
        order:

        1. exact-FQN match in ``module_fqn_to_config``
        2. first ``re:<pattern>`` whose pattern fully matches ``prefix``
        3. ``_default`` entry
        4. ``None`` (caller treats as "do not quantize this layer")
        """
        from torchao.quantization import ModuleFqnToConfig

        if not isinstance(self.torchao_config, ModuleFqnToConfig):
            return self.torchao_config

        fqn_map = self.torchao_config.module_fqn_to_config
        if prefix in fqn_map:
            assert not prefix.startswith("re:"), (
                "module fqn should not start with `re:`, which is used for specifying regex"
            )
            return fqn_map[prefix]

        for pattern in fqn_map:
            if not pattern.startswith("re:"):
                continue
            if re.fullmatch(pattern[3:], prefix):
                return fqn_map[pattern]
        return fqn_map.get("_default", None)

# 为了解决上游 torchao 缺失，加了非常多的兜底逻辑
class AscendTorchAOLinearMethod(LinearMethodBase):
    """Linear method that runs TorchAO-quantized weights on Ascend NPU.

    Lifecycle::

        create_weights                 # plain Parameter placeholder
            -> weight loader copies the on-disk bf16/fp16 tensor in
        process_weights_after_loading  # online quantize via torchao.quantize_
        apply                          # F.linear, with dequant fallback

    Embedding layers are handled separately by vLLM (we do not implement
    :meth:`embedding`); ``AscendTorchAOConfig.get_quant_method`` returns
    ``None`` for non-Linear layers, so vLLM falls back to
    ``UnquantizedEmbeddingMethod``.
    """

    def __init__(self, quant_config: AscendTorchAOConfig) -> None:
        self.quant_config = quant_config

    # ------------------------------------------------------------------ create

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # When the on-disk checkpoint is genuinely TorchAO-serialized the
        # placeholder must be the same subclass type, otherwise the loader's
        # ``copy_`` / ``narrow`` calls would not flow through the subclass
        # ``__torch_dispatch__`` and the subclass payload would be lost.
        if self.quant_config.is_checkpoint_torchao_serialized:
            weight = torchao_quantize_param_data(weight, self.quant_config.torchao_config)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # ----------------------------------------------------- after-load quantize

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Finalize weights after the framework loader has populated them.

        # This method mirrors upstream :meth:`TorchAOLinearMethod.process_weights_after_loading`
        verbatim except for two Ascend-specific additions:

        1. We do **not** reimplement the upstream
           ``convert_to_packed_tensor_based_on_current_hardware`` skip; that
           function is a defensive no-op for everything except
           ``Int4Tensor`` on CUDA SM>=9.0 with MSLK, so calling it on NPU is
           safe and forwards-compatible (a future torchao release that adds
           NPU-specific packing would automatically apply here).
        2. After online quantization, we attempt to set up an Ascend-native
           fast path (W8A16 / W8A8 / W4A16) that replaces the TorchAO
           subclass weight with plain int8/int4pack weights + scales for
           fused NPU quant GEMM kernels.
        """
        if not hasattr(layer, "weight"):
            return

        # ------------------------------------------------------------------ #
        # Path 1: serialized checkpoint (weights already subclass tensors)
        # ------------------------------------------------------------------ #
        if self.quant_config.is_checkpoint_torchao_serialized:
            recorded_weight_attr = _get_weight_attrs(layer.weight)
            layer.weight = Parameter(
                convert_to_packed_tensor_based_on_current_hardware(layer.weight),
                requires_grad=layer.weight.requires_grad,
            )
            _restore_weight_attrs(layer.weight, recorded_weight_attr)
            return

        # ------------------------------------------------------------------ #
        # Path 2: online quantization (plain weights → quantize now)
        # ------------------------------------------------------------------ #
        # ``torchao_quantize_param_data`` quantizes the local TP shard.
        # For per-row / per-channel scaling (default for
        # ``Int8WeightOnlyConfig`` etc.) this is identical to scaling the
        # unsharded tensor.  For per-tensor scaling configs (uncommon
        # for weights) the per-rank scale would diverge from a globally-
        # quantized checkpoint; users who need that should pre-quantize the
        # model and run via the serialized path.
        recorded_weight_attr = _get_weight_attrs(layer.weight)
        weight = torchao_quantize_param_data(layer.weight, self.quant_config.torchao_config)
        weight = Parameter(
            convert_to_packed_tensor_based_on_current_hardware(weight),
            requires_grad=weight.requires_grad,
        )
        _restore_weight_attrs(weight, recorded_weight_attr)
        layer.register_parameter("weight", weight)

        # Try to set up a native int8 fast path. This replaces the TorchAO
        # subclass weight with a plain int8 ``weight`` + ``weight_scale`` so
        # ``apply`` can call a fused NPU quant GEMM instead of dequantizing to
        # bf16 and running ``F.linear``:
        #   * int8wo (weight-only)  -> npu_weight_quant_batchmatmul (W8A16)
        #   * W8A8 dynamic-act      -> npu_dynamic_quant + npu_quant_matmul
        #   * W4A16 (int4 wo)       -> int4pack + npu_weight_quant_batchmatmul
        # On any failure these are no-ops and the dequant path still works.
        if envs_ascend.VLLM_ASCEND_TORCHAO_NATIVE_KERNEL:
            (self._maybe_setup_native_w8a16(layer)
             or self._maybe_setup_native_w8a8(layer)
             or self._maybe_setup_native_w4a16(layer))

    # ----------------------------------------------- native int8 fast path

    @staticmethod
    def _extract_int8wo_params(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract (int8 weight, per-channel scale) from an int8 weight-only
        TorchAO ``AffineQuantizedTensor``.

        Returns ``None`` (caller keeps the dequant path) unless the tensor is
        a symmetric, per-output-channel int8 quantization — the only shape the
        native ``npu_weight_quant_batchmatmul`` antiquant contract matches:

        * ``tensor_impl.int_data`` is ``int8`` with shape ``[out, in]``
        * ``tensor_impl.scale`` has ``out`` elements (one per output row)
        * ``tensor_impl.zero_point`` is all-zero (symmetric)

        Verified on torchao 0.17.0 / Int8WeightOnlyConfig: dequant ==
        ``int_data * scale`` and the kernel reproduces ``F.linear`` to within
        bf16 rounding (rel err ~6e-3).
        """
        impl = getattr(weight, "tensor_impl", None)
        if impl is None:
            return None
        int_data = getattr(impl, "int_data", None)
        scale = getattr(impl, "scale", None)
        if int_data is None or scale is None:
            return None
        if int_data.dtype != torch.int8 or int_data.dim() != 2:
            return None
        out_features = int_data.shape[0]
        if scale.numel() != out_features:
            return None
        zero_point = getattr(impl, "zero_point", None)
        if zero_point is not None and bool(zero_point.any()):
            # Asymmetric quantization: the kernel needs an antiquant_offset we
            # do not derive here. Fall back to the dequant path.
            return None
        return int_data, scale

    @staticmethod
    def _extract_w8a8_params(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract (int8 weight, per-channel weight scale) from a W8A8
        dynamic-activation TorchAO ``LinearActivationQuantizedTensor``.

        ``Int8DynamicActivationInt8WeightConfig`` wraps the weight as::

            LinearActivationQuantizedTensor
              .original_weight_tensor : AffineQuantizedTensor  (int8 weight)

        i.e. the weight itself is an ordinary symmetric per-output-channel
        int8 tensor (identical layout to int8wo); only the *activation* is
        additionally quantized at runtime. We therefore reuse
        :meth:`_extract_int8wo_params` on the inner tensor. Returns ``None``
        (caller keeps the dequant path) for anything that is not this shape.

        Verified on torchao 0.17.0: ``npu_dynamic_quant(x)`` +
        ``npu_quant_matmul`` reproduces the dequant ``F.linear`` to within int8
        activation rounding (rel err ~6e-3).
        """
        inner = getattr(weight, "original_weight_tensor", None)
        if inner is None:
            return None
        return AscendTorchAOLinearMethod._extract_int8wo_params(inner)

    @staticmethod
    def _extract_w4a16_params(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract (int4 weight, per-channel scale) from a W4A16 weight-only
        TorchAO ``IntxUnpackedToInt8Tensor`` (``IntxWeightOnlyConfig`` with
        ``weight_dtype=torch.int4``).

        Layout (verified on torchao 0.17.0)::

            .qdata      : int8 in [-8, 7], shape [out, in]   (int4 stored in int8)
            .scale      : float32, shape [out, 1]            (per-output-channel)
            .zero_point : int8, all zero                     (symmetric)

        Returns ``None`` (caller keeps the dequant path) unless the tensor is
        symmetric per-output-channel int4 — the only shape the native
        ``npu_convert_weight_to_int4pack`` + ``npu_weight_quant_batchmatmul``
        path matches. Per-group scales (``scale.shape[1] > 1``) are rejected.
        """
        qdata = getattr(weight, "qdata", None)
        scale = getattr(weight, "scale", None)
        if qdata is None or scale is None:
            return None
        if qdata.dtype != torch.int8 or qdata.dim() != 2:
            return None
        if int(qdata.min()) < -8 or int(qdata.max()) > 7:
            return None  # not int4-ranged
        out_features = qdata.shape[0]
        # per-channel only: scale has exactly one group per output row
        if scale.numel() != out_features:
            return None
        zero_point = getattr(weight, "zero_point", None)
        if zero_point is not None and bool(zero_point.any()):
            return None  # asymmetric: would need an offset we do not derive
        return qdata, scale

    @staticmethod
    def _store_native_int8_weight(
        layer: torch.nn.Module,
        int_data: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Replace ``layer.weight`` with a plain int8 weight in ``[in, out]``
        NPU layout + a flattened ``weight_scale``, preserving vLLM weight attrs.

        Shared by the W8A16 and W8A8 fast paths. Layout mirrors
        ``vllm_ascend/quantization/methods/w8a16.py`` / ``w8a8_dynamic.py``.
        """
        from vllm_ascend.utils import maybe_trans_nz

        recorded_weight_attr = _get_weight_attrs(layer.weight)
        int_t = int_data.transpose(0, 1).contiguous()  # [out,in] -> [in,out]
        try:
            weight_t = maybe_trans_nz(int_t)
        except Exception:  # noqa: BLE001 - NZ is best-effort
            weight_t = int_t
        new_weight = Parameter(weight_t, requires_grad=False)
        _restore_weight_attrs(new_weight, recorded_weight_attr)
        layer.register_parameter("weight", new_weight)
        layer.register_parameter(
            "weight_scale", Parameter(scale.flatten().contiguous(), requires_grad=False)
        )

    def _maybe_setup_native_w8a16(self, layer: torch.nn.Module) -> bool:
        """If ``layer.weight`` is int8 weight-only, store plain int8 weight +
        per-channel scale and flag the layer so ``apply`` uses
        ``npu_weight_quant_batchmatmul``. Returns ``True`` if the fast path was
        installed, ``False`` otherwise (caller may try another path). On any
        error this is a no-op (the TorchAO subclass weight is left in place and
        ``apply`` keeps the F.linear + dequant fallback).
        """
        params = self._extract_int8wo_params(layer.weight)
        if params is None:
            return False
        int_data, scale = params
        try:
            import torch_npu  # noqa: F401

            self._store_native_int8_weight(layer, int_data, scale)
            layer._torchao_native_w8a16 = True
            return True
        except Exception as exc:  # noqa: BLE001 - never break weight loading
            _warn_once(
                "TorchAO native int8 (W8A16) fast-path setup failed (layer=%s); "
                "keeping the F.linear + dequant path. Error: %s",
                type(layer).__name__,
                str(exc),
            )
            return False

    def _maybe_setup_native_w8a8(self, layer: torch.nn.Module) -> bool:
        """If ``layer.weight`` is a W8A8 dynamic-activation tensor, store plain
        int8 weight + per-channel scale and flag the layer so ``apply`` uses
        ``npu_dynamic_quant`` + ``npu_quant_matmul`` (both activation and weight
        in int8). Returns ``True`` if installed. No-op on any failure.
        """
        params = self._extract_w8a8_params(layer.weight)
        if params is None:
            return False
        int_data, scale = params
        try:
            import torch_npu  # noqa: F401

            self._store_native_int8_weight(layer, int_data, scale)
            layer._torchao_native_w8a8 = True
            return True
        except Exception as exc:  # noqa: BLE001 - never break weight loading
            _warn_once(
                "TorchAO native int8 (W8A8) fast-path setup failed (layer=%s); "
                "keeping the F.linear + dequant path. Error: %s",
                type(layer).__name__,
                str(exc),
            )
            return False

    def _maybe_setup_native_w4a16(self, layer: torch.nn.Module) -> bool:
        """If ``layer.weight`` is a W4A16 weight-only int4 tensor, pack it into
        the NPU int4 format and flag the layer so ``apply`` uses
        ``npu_weight_quant_batchmatmul`` (int4 weight + bf16 activation, 4x
        weight compression). Returns ``True`` if installed. No-op on failure.

        Layout mirrors ``vllm_ascend/quantization/methods/w4a16.py``: the int4
        weight (stored as int8 in [-8, 7]) is transposed to ``[in, out]``,
        cast to int32, then packed via ``npu_convert_weight_to_int4pack``.
        """
        params = self._extract_w4a16_params(layer.weight)
        if params is None:
            return False
        qdata, scale = params
        try:
            import torch_npu

            recorded_weight_attr = _get_weight_attrs(layer.weight)
            # [out, in] int8(int4) -> [in, out] int32 -> int4pack
            w_io = qdata.transpose(0, 1).contiguous().to(torch.int32)
            packed = torch_npu.npu_convert_weight_to_int4pack(w_io)
            new_weight = Parameter(packed, requires_grad=False)
            _restore_weight_attrs(new_weight, recorded_weight_attr)
            layer.register_parameter("weight", new_weight)
            layer.register_parameter(
                "weight_scale",
                Parameter(scale.flatten().contiguous(), requires_grad=False),
            )
            layer._torchao_native_w4a16 = True
            return True
        except Exception as exc:  # noqa: BLE001 - never break weight loading
            _warn_once(
                "TorchAO native int4 (W4A16) fast-path setup failed (layer=%s); "
                "keeping the F.linear + dequant path. Error: %s",
                type(layer).__name__,
                str(exc),
            )
            return False

    # -------------------------------------------------------------- inference

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_torchao_native_w8a16", False) or getattr(
            layer, "_torchao_native_w4a16", False
        ):
            import torch_npu

            # npu_weight_quant_batchmatmul handles both int8 (W8A16) and int4pack
            # (W4A16) weights with a bf16 activation. It requires a 2D activation
            # [tokens, in_features]; vLLM may pass a 3D [batch, seq, in] tensor,
            # so flatten the leading dims and restore afterwards.
            orig_shape = x.shape
            x2d = x.reshape(-1, orig_shape[-1]) if x.dim() != 2 else x
            # The kernel requires the bias to be float32 (not bf16/fp16).
            bias_f32 = bias.to(torch.float32) if bias is not None else None
            out = torch_npu.npu_weight_quant_batchmatmul(
                x=x2d,
                weight=layer.weight,
                antiquant_scale=layer.weight_scale.to(x.dtype),
                antiquant_offset=None,
                bias=bias_f32,
            )
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
            if x.dim() != 2:
                out = out.reshape(*orig_shape[:-1], out.shape[-1])
            return out
        if getattr(layer, "_torchao_native_w8a8", False):
            import torch_npu

            # W8A8: quantize the activation to int8 per-token at runtime, then
            # run a fully-int8 GEMM. Both the kernel and npu_dynamic_quant need
            # a 2D activation, so flatten leading dims and restore.
            orig_shape = x.shape
            x2d = x.reshape(-1, orig_shape[-1]) if x.dim() != 2 else x
            xq, x_scale = torch_npu.npu_dynamic_quant(x2d)
            bias_f32 = bias.to(torch.float32) if bias is not None else None
            out = torch_npu.npu_quant_matmul(
                xq,
                layer.weight,
                layer.weight_scale.to(torch.float32),
                pertoken_scale=x_scale,
                bias=bias_f32,
                output_dtype=x.dtype,
            )
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
            if x.dim() != 2:
                out = out.reshape(*orig_shape[:-1], out.shape[-1])
            return out
        try:
            return F.linear(x, layer.weight, bias)
        except (RuntimeError, NotImplementedError) as exc:
            if not envs_ascend.VLLM_ASCEND_TORCHAO_ALLOW_DEQUANT_FALLBACK:
                raise
            _warn_once(
                "TorchAO F.linear dispatch failed on Ascend (layer=%s); "
                "falling back to a dequantized weight for this layer "
                "(memory/perf benefit lost). Set "
                "VLLM_ASCEND_TORCHAO_ALLOW_DEQUANT_FALLBACK=0 to fail fast. "
                "Original error: %s",
                type(layer).__name__,
                str(exc),
            )
            self._replace_with_dequantized_parameter(layer, dtype=x.dtype, device=x.device)
            return F.linear(x, layer.weight, bias)

    # -------------------------------------------------------- dequant helpers

    @staticmethod
    def _dequantize_weight(
        weight: torch.Tensor,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Dequantize a (possibly TorchAO subclass) weight to a plain tensor.

        TorchAO subclasses (``AffineQuantizedTensor`` and friends) implement
        ``dequantize`` and return a plain ``torch.Tensor``. For an already
        plain tensor ``Tensor.dequantize`` is a no-op in some torch versions
        and raises in others; we treat both cases as "already dequantized".
        Optional ``dtype`` / ``device`` are applied via ``.to`` after dequant.
        """
        dequantized = False
        if hasattr(weight, "dequantize"):
            # ``Tensor.dequantize`` raises on non-quantized tensors in some
            # torch versions; treat that as "already dequantized" and keep
            # ``weight`` unchanged.
            with contextlib.suppress(RuntimeError, NotImplementedError):
                weight = weight.dequantize()
                dequantized = True
        if not dequantized and hasattr(weight, "original_weight_tensor"):
            # Dynamic-activation configs (e.g.
            # ``Int8DynamicActivationInt8WeightConfig`` → W8A8) wrap the weight
            # in a ``LinearActivationQuantizedTensor`` whose own
            # ``dequantize`` is not implemented on NPU. The real weight lives
            # in ``original_weight_tensor`` (an ``AffineQuantizedTensor``),
            # which does dequantize. Recurse into it so the fallback path works
            # for these configs too.
            inner = weight.original_weight_tensor
            with contextlib.suppress(RuntimeError, NotImplementedError, AttributeError):
                weight = inner.dequantize()
        if dtype is not None or device is not None:
            weight = weight.to(
                dtype=dtype if dtype is not None else weight.dtype,
                device=device if device is not None else weight.device,
            )
        return weight

    def _replace_with_dequantized_parameter(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Replace ``layer.weight`` with a plain Parameter wrapping the
        dequantized tensor, preserving any vLLM weight attrs (``input_dim``,
        ``output_dim``, ``weight_loader`` etc.).
        """
        recorded_weight_attr = _get_weight_attrs(layer.weight)
        dequantized = self._dequantize_weight(layer.weight, dtype=dtype, device=device)
        parameter = Parameter(dequantized, requires_grad=False)
        _restore_weight_attrs(parameter, recorded_weight_attr)
        layer.register_parameter("weight", parameter)
