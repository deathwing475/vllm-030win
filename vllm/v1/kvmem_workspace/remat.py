# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 063, split in step 097): KVMem rematerialisation.

A page that the sliding window dropped lives in the host workspace with its K
baked at the *original* positions. Putting it back into the window means giving
it a new position (the fixed-slot layout compresses the window to ``0..B-1``),
and a quantised K cannot be re-rotated in place: RoPE is a per-pair rotation,
while the cache's scale groups cover several consecutive head dims, so the
scale has to be recomputed for every group the rotation touches.

The design's rule (``docs/vllm-030win-调研-KVMem虚拟化KV工作区.md`` §5.3) is that
every re-entry rebuilds the rotated part **from the pre-RoPE K**, once, at the
target position. Nothing is ever derived from a previously rotated copy, so a
page can be displaced arbitrarily many times without accumulating drift -- the
mechanism behind the reference implementation's >210K cross-talk (#4) simply
does not exist here. :func:`rematerialize_page` is that single rebuild.

What this module owns is the half that is the same for every KV dtype:

* the rotation of the rotary prefix, delegating to the engine's own
  ``ApplyRotaryEmb``/``apply_interleaved_rope`` so the convention cannot drift
  from the model's;
* the authority's view of a packed K row (which head owns which dims).

How rotated values become *bytes in a page* belongs to the KV packing, and that
lives in :mod:`vllm.v1.kvmem_workspace.codec` -- step 097 moved the NVFP4 byte
geometry and quantiser there unchanged, so the channel is no longer NVFP4-only
by accident.

Only the rotary prefix is rewritten. The other head dims and the whole of V are
position-independent, so a re-entering page keeps the bytes it already had --
which is why the workspace only has to store the pre-RoPE rotary prefix as an
authority and can leave the rest in the packed form.
"""

from __future__ import annotations

import torch

from vllm.v1.kvmem_workspace.codec import PageCodec, PageGeometry


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


def rotated_prefix_from_packed_k(
    k: torch.Tensor, head_size: int, rotary_dim: int
) -> torch.Tensor:
    """Slice the rotary prefix out of a packed ``[T, num_heads * head_size]`` K.

    The capture hands out K exactly as the attention layer held it: heads
    concatenated along the last axis. Taking ``k[:, :rotary_dim]`` would grab
    the first head's prefix only, so the head axis has to be restored first.
    """
    num_tokens, width = k.shape
    num_heads = width // head_size
    if num_heads * head_size != width:
        raise ValueError(f"packed K width {width} is not a multiple of {head_size}")
    return k.reshape(num_tokens, num_heads, head_size)[..., :rotary_dim]


def rematerialize_page(
    page: torch.Tensor,
    codec: PageCodec,
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
    so repeated displacements cannot drift. Turning the rotated values into
    page bytes is the codec's job, which is why this function is the same for
    every KV packing.
    """
    post = bake_rotated_k(
        raw_rot,
        positions,
        cos_sin_cache,
        is_neox_style=is_neox_style,
        mrope_section=mrope_section,
    )
    codec.encode_rotary_prefix(page, geom, tokens, post, k_scale=k_scale)
