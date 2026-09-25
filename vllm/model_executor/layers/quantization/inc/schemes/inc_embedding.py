# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized embedding lookup for the inc path (DFlash2 selector codebooks).

Layout matches the compressed-tensors embedding packing: rows are token ids,
`PACK_FACTOR = 32 // bits` codes per int32 along hidden, LSB-first, codes
stored offset-binary (`raw = q + 2**(bits-1)`) which the gather kernel
restores with `q = raw - 2**(bits-1)`. Scales are per-row, one per
`group_size` hidden columns. No zero points (symmetric).

Dequantization reuses compressed-tensors' `_dequant_gather_triton`, which
unpacks only the gathered rows.
"""
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (
    _dequant_gather_triton,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)


class INCEmbeddingWNA16Int(QuantizeMethodBase):
    """Group-wise int lookup table (the DFlash2 candidate-selector codebooks)."""

    def __init__(self, num_bits: int, group_size: int):
        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = 32 // num_bits

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
        weight_loader = extra_weight_attrs["weight_loader"]
        # Embedding weight is [vocab, hidden]; vocab (rows) is the partitioned
        # output dim, hidden is the input dim -- same convention as
        # CompressedTensorsEmbeddingWNA16Int.
        vocab_pp = sum(output_partition_sizes)
        hidden = input_size_per_partition
        assert self.group_size > 0 and hidden % self.group_size == 0
        layer.hidden_size = hidden

        qweight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
            data=torch.empty(vocab_pp, hidden // self.pack_factor, dtype=torch.int32),
        )
        scales = GroupQuantScaleParameter(
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader,
            data=torch.empty(vocab_pp, hidden // self.group_size, dtype=params_dtype),
        )
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer.qweight.shape[1] * self.pack_factor == layer.hidden_size, (
            f"packed width {layer.qweight.shape[1]}x{self.pack_factor} "
            f"!= hidden {layer.hidden_size}"
        )
        assert layer.scales.shape[1] == layer.hidden_size // self.group_size

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # ids arrive as local row numbers (VocabParallelEmbedding shifts and
        # masks before calling us), so gather straight against the packed rows.
        ids = input_.reshape(-1).contiguous()
        hidden = layer.hidden_size
        deq = _dequant_gather_triton(
            ids, layer.qweight, layer.scales, hidden, self.num_bits
        )
        return deq.reshape(*input_.shape, hidden)

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "INCEmbeddingWNA16Int supports embedding lookup only"
        )
