# SPDX-License-Identifier: Apache-2.0
"""Reference Torch WNA16 fallback for packed symmetric 3-bit weights.

This is an experiment-only path for environments without a CUDA uint3b4
kernel. It dequantizes one layer on demand and is intentionally slow.

Dequantization is chunked over output rows. Unpacking the full row block at
once materializes several [out_features, in_features] int64 temporaries, which
for a merged projection such as gate_up_proj (34816x5120) peaks near 4.8 GiB of
transient torch memory. That peak is live at the same time as the KV cache and
on a 16 GiB card it blows the KV budget. Chunking keeps the peak bounded by the
chunk size instead of the layer size.
"""

import os

import torch

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from vllm.scalar_type import scalar_types

# Bitstream gather indices depend only on the logical row width and the packed
# word count, so every layer with the same in_features shares one set.
_BIT_INDEX_CACHE: dict[tuple[int, int, int], tuple] = {}

_DEFAULT_CHUNK_ROWS = 2048


def _chunk_rows() -> int:
    raw = os.getenv("VLLM_TORCH_WNA16_CHUNK_ROWS")
    if raw is None:
        return _DEFAULT_CHUNK_ROWS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_CHUNK_ROWS
    return value if value > 0 else _DEFAULT_CHUNK_ROWS


def _bit_index(in_features: int, words: int, device: torch.device) -> tuple:
    key = (in_features, words, device.index if device.index is not None else -1)
    cached = _BIT_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    positions = torch.arange(in_features, device=device, dtype=torch.int64) * 3
    word = positions // 32
    shift = positions.remainder(32)
    crossing = shift > 29
    # 3-bit values that straddle a 32-bit word boundary need the next word.
    next_word = torch.clamp(word + 1, max=words - 1) if bool(crossing.any()) else None
    index = (word, shift, crossing, next_word)
    _BIT_INDEX_CACHE[key] = index
    return index


class TorchWNA16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type != scalar_types.uint3b4:
            return False, "Torch fallback only supports uint3b4"
        if c.zero_points or c.has_g_idx:
            return False, "Torch fallback requires symmetric weights without g_idx"
        if c.group_size <= 0:
            return False, "Torch fallback requires group quantization"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep the packed representation. Dequantization is performed per call
        # so model construction does not allocate a second full-precision copy.
        return None

    def _orient_scales(self, scales: torch.Tensor, out_features: int) -> torch.Tensor:
        if scales.shape[0] == out_features:
            return scales
        if scales.shape[1] == out_features:
            return scales.transpose(0, 1)
        raise ValueError(
            f"Unexpected uint3b4 scale shape {tuple(scales.shape)} for "
            f"weight shape {out_features} output rows"
        )

    def _dequantize_rows(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        group_size: int,
    ) -> torch.Tensor:
        """Dequantize a slice of output rows to float32."""
        word, shift, crossing, next_word = _bit_index(
            in_features, packed.shape[1], packed.device
        )
        # Compressed-tensors uint3b4 is a bitstream: 5120 values occupy exactly
        # 480 int32 words (3 bits/value), so the packed row count is
        # authoritative and the logical in_features selects the live values.
        packed64 = packed.to(torch.int64) & 0xFFFFFFFF
        values = (packed64[:, word] >> shift) & 7
        if next_word is not None:
            crossed = ((packed64[:, next_word] << (32 - shift)) | values) & 7
            values = torch.where(crossing.unsqueeze(0), crossed, values)
        values = (values - 4).to(torch.float32)

        scale_values = scales.to(torch.float32).unsqueeze(-1).expand(
            scales.shape[0], scales.shape[1], group_size
        ).reshape(scales.shape[0], -1)
        return values * scale_values[:, :in_features]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = getattr(layer, self.w_q_name).data
        scales = getattr(layer, self.w_s_name).data
        shape = getattr(layer, "weight_shape").data.to(device=packed.device)
        # Merged projections may keep a per-slice weight_shape, so the packed
        # row count is authoritative for the output dimension.
        out_features = int(packed.shape[0])
        in_features = int(shape[-1])
        scales = self._orient_scales(scales, out_features)

        group_size = self.config.group_size
        step = _chunk_rows()
        out_dtype = x.dtype
        pieces: list[torch.Tensor] = []
        for start in range(0, out_features, step):
            stop = min(start + step, out_features)
            weight = self._dequantize_rows(
                packed[start:stop], scales[start:stop], in_features, group_size
            )
            pieces.append(torch.matmul(x, weight.t().to(out_dtype)))
        output = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        return output if bias is None else output + bias
