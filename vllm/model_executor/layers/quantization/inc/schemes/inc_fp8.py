# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.parameter import PerTensorScaleParameter


class INCFp8Linear(QuantizeMethodBase):
    """Per-tensor fp8 weight-only linear for INC mixed-precision checkpoints.

    Checkpoint layout (matches vLLM's Fp8LinearMethod contract):
      `<name>.weight`       float8_e4m3fn [out, in]
      `<name>.weight_scale` float32 [num_shards]

    Unlike Fp8LinearMethod this dequantises to the activation dtype and runs
    a plain bf16 GEMM instead of `cutlass_scaled_mm`: the fp8 w8a8 path dies
    in this Windows build (`cutlass_scaled_mm_sm80_epilogue`, scaled_mm_c2x.cu
    -- empty TORCH_CHECK message) on an sm120 GPU. For weight-only draft
    matrices (10x [1280, 5120]) the dequant bandwidth is negligible next to
    the GEMM, so correctness is bought back at well under 1% step time.
    """

    def __init__(self) -> None:
        super().__init__()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                output_size_per_partition, input_size_per_partition, weight_loader
            ),
        )
        layer.register_parameter(
            "weight_scale",
            create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer.weight_scale is not None and torch.isfinite(
            layer.weight_scale
        ).all(), "fp8 weight_scale was never loaded (still at its fill value?)"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)
