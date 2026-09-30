# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 063): KVMem rematerialisation primitive (K3 items 4-6).

A page that the sliding window dropped lives in the host workspace with its K
baked at the *original* positions. Putting it back into the window means giving
it a new position (the fixed-slot layout compresses the window to ``0..B-1``),
and NVFP4 K cannot be re-rotated in place: RoPE is a per-pair rotation, while
the cache's fp8 block scales cover 16 consecutive head dims, so the scale has
to be recomputed for every group the rotation touches.

The design's rule (``docs/vllm-030win-调研-KVMem虚拟化KV工作区.md`` §5.3) is that
every re-entry rebuilds the rotated part **from the pre-RoPE K**, once, at the
target position. Nothing is ever derived from a previously rotated copy, so a
page can be displaced arbitrarily many times without accumulating drift -- the
mechanism behind the reference implementation's >210K cross-talk (#4) simply
does not exist here. :func:`rematerialize_page` is that single rebuild.

What this module owns:

* the byte geometry of one NVFP4 KV page (``[K_data | K_scale | V_data |
  V_scale]``, HND, per-head rows carved as ``[block, data]`` then
  ``[block, scale]`` -- see ``v1/attention/reference_nvfp4.py``);
* the rotation of the rotary prefix, delegating to the engine's own
  ``ApplyRotaryEmb``/``apply_interleaved_rope`` so the convention cannot drift
  from the model's;
* the NVFP4 quantiser restricted to the rotary prefix's scale groups, which is
  byte-identical to the engine's full-row quantiser on those groups (the groups
  are disjoint, so restricting the reduction cannot change the result).

Only the rotary prefix is rewritten. The other 192 of the 256 head dims and the
whole of V are position-independent, so a re-entering page keeps the bytes it
already had -- which is why the workspace only has to store the pre-RoPE rotary
prefix as an authority and can leave the rest in the packed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# NVFP4 stores one fp8 scale per 16 consecutive head dims.
_SCALE_GROUP = 16
# E2M1 magnitudes for the 3-bit magnitude field of a code (bit 3 is the sign).
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class PageGeometry:
    """Byte geometry of one NVFP4 KV page (one layer, one block, K side + V side).

    The physical layout is the engine's HND NVFP4 layout: per block, the K side
    is one contiguous byte range followed by the V side, and each side is
    ``[num_heads][block_size][data_dim]`` followed by
    ``[num_heads][block_size][scale_dim]``. ``page_bytes`` is therefore
    ``2 * num_heads * block_size * full_dim`` and equals the byte count the
    workspace copies per (layer, block).
    """

    head_size: int
    num_heads: int
    block_size: int
    rotary_dim: int

    @property
    def data_dim(self) -> int:
        return self.head_size // 2

    @property
    def scale_dim(self) -> int:
        return self.head_size // 16

    @property
    def full_dim(self) -> int:
        return self.data_dim + self.scale_dim

    @property
    def k_data_bytes(self) -> int:
        """Bytes of the K side's data region (the scale region starts here)."""
        return self.num_heads * self.block_size * self.data_dim

    @property
    def side_bytes(self) -> int:
        return self.num_heads * self.block_size * self.full_dim

    @property
    def page_bytes(self) -> int:
        return 2 * self.side_bytes

    @property
    def rot_groups(self) -> int:
        """Scale groups covered by the rotary prefix."""
        return self.rotary_dim // _SCALE_GROUP

    @property
    def rot_data_bytes(self) -> int:
        """Packed fp4 bytes per head covered by the rotary prefix."""
        return self.rotary_dim // 2

    @property
    def rot_scale_bytes(self) -> int:
        return self.rot_groups

    def __post_init__(self) -> None:
        if self.head_size % 32:
            raise ValueError(f"head_size {self.head_size} must be a multiple of 32")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_size:
            raise ValueError(f"rotary_dim {self.rotary_dim} out of range")
        if self.rotary_dim % _SCALE_GROUP:
            # The rotary prefix has to end on a scale-group boundary, otherwise
            # a group would straddle the rotated and the untouched dims and
            # neither side could be written independently.
            raise ValueError(
                f"rotary_dim {self.rotary_dim} is not a multiple of {_SCALE_GROUP}"
            )


def _check(page: torch.Tensor, geom: PageGeometry) -> None:
    if page.dtype != torch.uint8:
        raise ValueError(f"page must be uint8, got {page.dtype}")
    if page.numel() != geom.page_bytes:
        raise ValueError(
            f"page has {page.numel()} bytes, geometry expects {geom.page_bytes}"
        )


def rotated_byte_offsets(
    geom: PageGeometry, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte offsets of the rotary prefix inside a page, for *tokens*.

    ``tokens`` holds block-local token indices. Returns ``(data, scale)``, both
    shaped ``[T, num_heads, width]`` and relative to the start of the page.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    heads = torch.arange(geom.num_heads, dtype=torch.long)
    t = tokens[:, None, None]
    h = heads[None, :, None]
    data = (
        h * (geom.block_size * geom.data_dim)
        + t * geom.data_dim
        + torch.arange(geom.rot_data_bytes, dtype=torch.long)[None, None, :]
    )
    scale = (
        geom.k_data_bytes
        + h * (geom.block_size * geom.scale_dim)
        + t * geom.scale_dim
        + torch.arange(geom.rot_scale_bytes, dtype=torch.long)[None, None, :]
    )
    return data, scale


def bake_rotated_k(
    raw_rot: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
    mrope_section: list[int] | None = None,
) -> torch.Tensor:
    """Rotate the pre-RoPE rotary prefix to *positions*.

    Mirrors ``MRotaryEmbedding.forward_native``: 1-D ``positions`` are the
    scalar text positions, 2-D ``[3, T]`` positions take the interleaved
    M-RoPE layout (for text-only input the three rows are equal, so the
    permutation is a no-op and the two forms agree exactly).

    ``raw_rot`` is ``[T, num_heads, rotary_dim]``; the result keeps its dtype.
    """
    # cos_sin_cache is [max_position, rotary_dim]: cos in the first half, sin in
    # the second (each rotary_dim // 2 wide), so its width equals rotary_dim.
    if raw_rot.shape[-1] != cos_sin_cache.shape[-1]:
        raise ValueError(
            f"rotary_dim {raw_rot.shape[-1]} does not match cos_sin_cache width "
            f"{cos_sin_cache.shape[-1]}"
        )
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        if mrope_section is None:
            raise ValueError("2-D positions need mrope_section")
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            apply_interleaved_rope,
        )

        cos = apply_interleaved_rope(cos, mrope_section)
        sin = apply_interleaved_rope(sin, mrope_section)

    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    return ApplyRotaryEmb.forward_static(raw_rot, cos, sin, is_neox_style=is_neox_style)


def quantize_rotated(
    post_rope_rot: torch.Tensor, k_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4-quantise the rotary prefix, restricted to its own scale groups.

    Returns ``(packed, sf)`` with ``packed`` uint8 ``[T, heads, rotary_dim // 2]``
    and ``sf`` float8_e4m3fn ``[T, heads, rotary_dim // 16]`` -- the same bytes
    the engine's writer puts in the corresponding slice of the page.
    """
    from vllm.v1.attention.reference_nvfp4 import _e2m1_codes

    num_tokens, num_heads, rotary_dim = post_rope_rot.shape
    if rotary_dim % _SCALE_GROUP:
        raise ValueError(f"rotary_dim {rotary_dim} is not a multiple of {_SCALE_GROUP}")
    groups = rotary_dim // _SCALE_GROUP
    grouped = post_rope_rot.reshape(num_tokens, num_heads, groups, _SCALE_GROUP)

    amax = grouped.abs().amax(dim=-1)
    global_scale = 1.0 / float(k_scale)
    sf = (amax / (6.0 * global_scale)).to(torch.float8_e4m3fn)
    sf_value = sf.float()
    output_scale = torch.where(
        sf_value > 0,
        1.0 / (sf_value * global_scale),
        torch.zeros_like(sf_value),
    )
    codes = _e2m1_codes(grouped * output_scale.unsqueeze(-1))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        packed.reshape(num_tokens, num_heads, rotary_dim // 2).contiguous(),
        sf.view(torch.uint8).contiguous(),
    )


def dequantize_rotated(
    packed: torch.Tensor, sf: torch.Tensor, k_scale: float = 1.0
) -> torch.Tensor:
    """Inverse of :func:`quantize_rotated`, for error accounting and tests."""
    num_tokens, num_heads, packed_dim = packed.shape
    rotary_dim = packed_dim * 2
    groups = rotary_dim // _SCALE_GROUP
    bytes_per_group = _SCALE_GROUP // 2
    codes = torch.empty(
        (num_tokens, num_heads, groups, _SCALE_GROUP),
        dtype=torch.long,
        device=packed.device,
    )
    pk = packed.reshape(num_tokens, num_heads, groups, bytes_per_group).long()
    codes[..., 0::2] = pk & 0x0F
    codes[..., 1::2] = pk >> 4
    magnitudes = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device
    )
    values = magnitudes[codes & 0x07]
    values = torch.where((codes & 0x08) != 0, -values, values)
    sf_value = sf.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    output_scale = torch.where(
        sf_value > 0,
        float(k_scale) / sf_value,
        torch.zeros_like(sf_value),
    )
    return (values / output_scale).reshape(num_tokens, num_heads, rotary_dim)


def write_rotated(
    page: torch.Tensor,
    geom: PageGeometry,
    tokens: torch.Tensor,
    packed: torch.Tensor,
    sf: torch.Tensor,
) -> None:
    """Write a quantised rotary prefix into *page* (in place)."""
    _check(page, geom)
    if packed.shape[0] != tokens.numel() or sf.shape[0] != tokens.numel():
        raise ValueError("packed/sf token count does not match tokens")
    data_off, scale_off = rotated_byte_offsets(geom, tokens)
    flat = page.view(-1)
    flat[data_off.reshape(-1)] = packed.reshape(-1).to(torch.uint8)
    flat[scale_off.reshape(-1)] = sf.reshape(-1).view(torch.uint8)


def read_rotated(
    page: torch.Tensor, geom: PageGeometry, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the quantised rotary prefix back out of a page."""
    _check(page, geom)
    data_off, scale_off = rotated_byte_offsets(geom, tokens)
    flat = page.view(-1)
    num_tokens = torch.as_tensor(tokens).numel()
    packed = flat[data_off.reshape(-1)].reshape(
        num_tokens, geom.num_heads, geom.rot_data_bytes
    )
    sf = (
        flat[scale_off.reshape(-1)]
        .reshape(num_tokens, geom.num_heads, geom.rot_scale_bytes)
        .view(torch.float8_e4m3fn)
    )
    return packed, sf


def rematerialize_page(
    page: torch.Tensor,
    geom: PageGeometry,
    raw_rot: torch.Tensor,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_scale: float = 1.0,
    is_neox_style: bool = True,
    mrope_section: list[int] | None = None,
) -> None:
    """Rebuild *tokens*' rotary prefix at *positions*, in place.

    ``raw_rot`` is the stored pre-RoPE rotary prefix ``[T, heads, rotary_dim]``
    and is the only input -- a page that was rotated before is never read back,
    so repeated displacements cannot drift.
    """
    post = bake_rotated_k(
        raw_rot,
        positions,
        cos_sin_cache,
        is_neox_style=is_neox_style,
        mrope_section=mrope_section,
    )
    packed, sf = quantize_rotated(post, k_scale=k_scale)
    write_rotated(page, geom, tokens, packed, sf)
