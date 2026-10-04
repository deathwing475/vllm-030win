# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 097): KVMem page codecs (design doc §1.2-A).

A page in the workspace is a slice of the engine's own KV cache, so how one
head's values become bytes -- the packing width, the scale groups, where those
bytes sit inside a page -- is a property of ``--kv-cache-dtype``, not of KVMem.
Step 094's census found that knowledge split across two files (the byte
geometry and quantiser in ``remat.py``, the dtype guard in ``worker.py``),
which made the whole rematerialisation channel NVFP4-only. This module owns it
behind one interface:

    ``page_geometry``     -- how a page is laid out
    ``encode_rotary_prefix`` -- rotated K -> the page's rotary bytes
    ``decode_rotary_prefix`` -- page bytes -> values plus the stored codes
    ``supported``         -- may the raw-K channel arm on this dtype?

``select_codec()`` is the registry the worker's guard consults, so the message
a user sees when a dtype is unsupported comes from the registry rather than
from a hard-coded prefix.

NVFP4 is the only implemented codec, and its body is step 063/064's code moved
here unchanged -- the frozen step-083 byte-for-byte behaviour is a property of
that code, not of a re-implementation. bf16 and fp8 are registered
placeholders: they name the dtypes they would serve and nothing more, because
their page semantics have never been exercised on a real target model (design
doc §1.3). ``supported()`` stays False for them, so arming raw-K on an
unsupported dtype still fails fast -- now with the registry's own list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PageGeometry:
    """Byte geometry of one KV page (one layer, one manager block).

    ``data_dim`` and ``scale_dim`` are the bytes one head's ``head_size``
    values occupy in the packed-data and scale regions respectively, and
    ``scale_group`` is how many consecutive dims one scale covers (0 when the
    dtype has no scales). The codec computes all three; everything here is
    derived from them, so the offset arithmetic does not need to know which
    dtype it is laying out.

    Step 064 established the NVFP4 page layout empirically (data probes 15/15
    exact, scale probes 58.84% vs 0.93% under the old assumption): the page is
    *not* one contiguous K side followed by one contiguous V side. It is
    ``block_size // kernel_block_size`` chunks of 18,432 B, one per kernel
    block -- the NVFP4 append kernel writes at 16-token granularity -- and
    each chunk carries the K side first, then the V side:

        chunk(t // kbs) = [ K side | V side ]
        side            = [heads * kbs * data_dim data bytes]
                          [heads * kbs * scale_dim scale bytes]

    ``kernel_block_size`` falls out of the engine's own cache tensor:
    ``spec.block_size // group_kernel_blocks(cache, num_blocks).shape[1]``
    (1424 // 89 = 16 here). With ``kernel_block_size == block_size`` this
    degenerates to "one K side, then one V side" with per-side ``[heads]
    [block_size]`` order, which is exactly what ``side_carve_views``' formula
    assumes for the NHD illusion view it is handed -- and why that view has to
    be taken at kernel-block, not manager-block, granularity.
    """

    head_size: int
    num_heads: int
    block_size: int
    rotary_dim: int
    kernel_block_size: int
    data_dim: int
    scale_dim: int
    scale_group: int

    @property
    def full_dim(self) -> int:
        return self.data_dim + self.scale_dim

    @property
    def chunks_per_page(self) -> int:
        return self.block_size // self.kernel_block_size

    @property
    def chunk_bytes(self) -> int:
        return 2 * self.num_heads * self.kernel_block_size * self.full_dim

    @property
    def side_chunk_bytes(self) -> int:
        """Bytes of one side (K or V) within one chunk."""
        return self.num_heads * self.kernel_block_size * self.full_dim

    @property
    def k_data_bytes(self) -> int:
        """Bytes of the K side's data region within one chunk (the K scale
        region starts here)."""
        return self.num_heads * self.kernel_block_size * self.data_dim

    @property
    def page_bytes(self) -> int:
        return self.chunks_per_page * self.chunk_bytes

    @property
    def rot_groups(self) -> int:
        """Scale groups covered by the rotary prefix."""
        return self.rotary_dim // self.scale_group

    @property
    def rot_data_bytes(self) -> int:
        """Packed bytes per head covered by the rotary prefix."""
        return self.rotary_dim * self.data_dim // self.head_size

    @property
    def rot_scale_bytes(self) -> int:
        return self.rotary_dim * self.scale_dim // self.head_size

    def __post_init__(self) -> None:
        if self.data_dim <= 0:
            raise ValueError(f"data_dim {self.data_dim} must be positive")
        if self.scale_dim < 0 or self.scale_group < 0:
            raise ValueError("scale_dim and scale_group must be non-negative")
        if self.scale_group and self.rotary_dim % self.scale_group:
            # The rotary prefix has to end on a scale-group boundary, otherwise
            # a group would straddle the rotated and the untouched dims and
            # neither side could be written independently.
            raise ValueError(
                f"rotary_dim {self.rotary_dim} is not a multiple of "
                f"{self.scale_group}"
            )
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_size:
            raise ValueError(f"rotary_dim {self.rotary_dim} out of range")
        if self.kernel_block_size <= 0 or self.block_size % self.kernel_block_size:
            raise ValueError(
                f"kernel_block_size {self.kernel_block_size} must divide "
                f"block_size {self.block_size}"
            )


@dataclass(frozen=True)
class RotaryPrefix:
    """The rotary prefix read back out of a page.

    ``values`` is the dequantised prefix ``[T, heads, rotary_dim]`` (what an
    error budget is measured against); ``codes`` and ``scales`` are the bytes
    as stored, kept because the comparison that matters is the byte diff and
    because a codec's scale region is what its step size has to be derived
    from. ``max_step`` is the largest codebook step those scales imply, so a
    caller can report "the rebuild is off by N of a quantisation step" without
    knowing anything about the dtype.
    """

    values: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    max_step: float


class PageCodec:
    """The dtype half of the workspace: page bytes <-> pre-RoPE rotary prefix."""

    #: Name of the packing, for logs and for the unsupported-dtype message.
    name: str = "abstract"
    #: ``--kv-cache-dtype`` strings this codec is registered for.
    dtype_prefixes: tuple[str, ...] = ()
    #: Whether the codec actually computes bytes for those dtypes.
    implemented: bool = False

    def supported(self, kv_cache_dtype: str) -> bool:
        """May the raw-K channel arm on *kv_cache_dtype*?"""
        raise NotImplementedError

    def page_geometry(
        self,
        *,
        head_size: int,
        num_heads: int,
        block_size: int,
        rotary_dim: int,
        kernel_block_size: int,
    ) -> PageGeometry:
        raise NotImplementedError

    def encode_rotary_prefix(
        self,
        page: torch.Tensor,
        geom: PageGeometry,
        tokens: torch.Tensor,
        post_rope_rot: torch.Tensor,
        *,
        k_scale: float = 1.0,
    ) -> None:
        """Quantise *post_rope_rot* and write it into *page* in place."""
        raise NotImplementedError

    def decode_rotary_prefix(
        self,
        page: torch.Tensor,
        geom: PageGeometry,
        tokens: torch.Tensor,
        *,
        k_scale: float = 1.0,
    ) -> RotaryPrefix:
        raise NotImplementedError


def _check_page(page: torch.Tensor, geom: PageGeometry) -> None:
    if page.dtype != torch.uint8:
        raise ValueError(f"page must be uint8, got {page.dtype}")
    if page.numel() != geom.page_bytes:
        raise ValueError(
            f"page has {page.numel()} bytes, geometry expects {geom.page_bytes}"
        )


class Nvfp4PageCodec(PageCodec):
    """NVFP4 KV pages: 4-bit nibbles plus one fp8 scale per 16 dims.

    Everything below is step 063/064's NVFP4 primitive, moved here verbatim --
    the E2M1 code table, the ``head_size // 2`` data and ``head_size // 16``
    scale regions, and the byte placement inside the kernel-block interleaved
    page. The quantiser is restricted to the rotary prefix's own scale groups,
    which is byte-identical to the engine's full-row quantiser on those groups
    because the groups are disjoint, so narrowing the reduction cannot change
    the result.
    """

    name = "nvfp4"
    dtype_prefixes = ("nvfp4",)
    implemented = True

    # NVFP4 stores one fp8 scale per 16 consecutive head dims.
    SCALE_GROUP = 16
    # E2M1 magnitudes for the 3-bit magnitude field of a code (bit 3 is the sign).
    E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def supported(self, kv_cache_dtype: str) -> bool:
        return str(kv_cache_dtype or "").lower().startswith(self.name)

    def page_geometry(
        self,
        *,
        head_size: int,
        num_heads: int,
        block_size: int,
        rotary_dim: int,
        kernel_block_size: int,
    ) -> PageGeometry:
        if head_size % (2 * self.SCALE_GROUP):
            # Two nibbles per byte and one scale per SCALE_GROUP dims both have
            # to divide the head, or a row would not be a whole number of
            # data and scale bytes.
            raise ValueError(
                f"head_size {head_size} must be a multiple of "
                f"{2 * self.SCALE_GROUP}"
            )
        return PageGeometry(
            head_size=head_size,
            num_heads=num_heads,
            block_size=block_size,
            rotary_dim=rotary_dim,
            kernel_block_size=kernel_block_size,
            data_dim=head_size // 2,
            scale_dim=head_size // self.SCALE_GROUP,
            scale_group=self.SCALE_GROUP,
        )

    def rotated_byte_offsets(
        self, geom: PageGeometry, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte offsets of the rotary prefix inside a page, for *tokens*.

        ``tokens`` holds block-local token indices. Returns ``(data, scale)``,
        both shaped ``[T, num_heads, width]`` and relative to the start of the
        page. The rotary prefix lives on the K side (side 0) of each chunk, so
        no V-side term appears; within the K side each head owns ``kbs``
        consecutive rows.
        """
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        heads = torch.arange(geom.num_heads, dtype=torch.long)
        t = tokens[:, None, None]
        h = heads[None, :, None]
        kbs = geom.kernel_block_size
        chunk = (t // kbs) * geom.chunk_bytes
        in_chunk = t % kbs
        data = (
            chunk
            + h * (kbs * geom.data_dim)
            + in_chunk * geom.data_dim
            + torch.arange(geom.rot_data_bytes, dtype=torch.long)[None, None, :]
        )
        scale = (
            chunk
            + geom.k_data_bytes
            + h * (kbs * geom.scale_dim)
            + in_chunk * geom.scale_dim
            + torch.arange(geom.rot_scale_bytes, dtype=torch.long)[None, None, :]
        )
        return data, scale

    def quantize_rotated(
        self, post_rope_rot: torch.Tensor, k_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NVFP4-quantise the rotary prefix, restricted to its own scale groups.

        Returns ``(packed, sf)`` with ``packed`` uint8 ``[T, heads, rotary_dim // 2]``
        and ``sf`` float8_e4m3fn ``[T, heads, rotary_dim // 16]`` -- the same bytes
        the engine's writer puts in the corresponding slice of the page.
        """
        from vllm.v1.attention.reference_nvfp4 import _e2m1_codes

        num_tokens, num_heads, rotary_dim = post_rope_rot.shape
        if rotary_dim % self.SCALE_GROUP:
            raise ValueError(
                f"rotary_dim {rotary_dim} is not a multiple of {self.SCALE_GROUP}"
            )
        groups = rotary_dim // self.SCALE_GROUP
        grouped = post_rope_rot.reshape(
            num_tokens, num_heads, groups, self.SCALE_GROUP
        )

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
        self, packed: torch.Tensor, sf: torch.Tensor, k_scale: float = 1.0
    ) -> torch.Tensor:
        """Inverse of :meth:`quantize_rotated`, for error accounting and tests."""
        num_tokens, num_heads, packed_dim = packed.shape
        rotary_dim = packed_dim * 2
        groups = rotary_dim // self.SCALE_GROUP
        bytes_per_group = self.SCALE_GROUP // 2
        codes = torch.empty(
            (num_tokens, num_heads, groups, self.SCALE_GROUP),
            dtype=torch.long,
            device=packed.device,
        )
        pk = packed.reshape(num_tokens, num_heads, groups, bytes_per_group).long()
        codes[..., 0::2] = pk & 0x0F
        codes[..., 1::2] = pk >> 4
        magnitudes = torch.tensor(
            self.E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device
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
        self,
        page: torch.Tensor,
        geom: PageGeometry,
        tokens: torch.Tensor,
        packed: torch.Tensor,
        sf: torch.Tensor,
    ) -> None:
        """Write a quantised rotary prefix into *page* (in place)."""
        _check_page(page, geom)
        if packed.shape[0] != tokens.numel() or sf.shape[0] != tokens.numel():
            raise ValueError("packed/sf token count does not match tokens")
        data_off, scale_off = self.rotated_byte_offsets(geom, tokens)
        flat = page.view(-1)
        flat[data_off.reshape(-1)] = packed.reshape(-1).to(torch.uint8)
        flat[scale_off.reshape(-1)] = sf.reshape(-1).view(torch.uint8)

    def read_rotated(
        self, page: torch.Tensor, geom: PageGeometry, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the quantised rotary prefix back out of a page."""
        _check_page(page, geom)
        data_off, scale_off = self.rotated_byte_offsets(geom, tokens)
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

    def encode_rotary_prefix(
        self,
        page: torch.Tensor,
        geom: PageGeometry,
        tokens: torch.Tensor,
        post_rope_rot: torch.Tensor,
        *,
        k_scale: float = 1.0,
    ) -> None:
        packed, sf = self.quantize_rotated(post_rope_rot, k_scale=k_scale)
        self.write_rotated(page, geom, tokens, packed, sf)

    def decode_rotary_prefix(
        self,
        page: torch.Tensor,
        geom: PageGeometry,
        tokens: torch.Tensor,
        *,
        k_scale: float = 1.0,
    ) -> RotaryPrefix:
        packed, sf = self.read_rotated(page, geom, tokens)
        # The largest step on the page (the 6->4 magnitude rung), so callers get
        # a scale-free yardstick for "how far off is this rebuild".
        sf_value = sf.view(torch.float8_e4m3fn).float()
        max_step = float(
            torch.where(
                sf_value > 0, 2.0 / sf_value, torch.zeros_like(sf_value)
            ).max()
        )
        return RotaryPrefix(
            values=self.dequantize_rotated(packed, sf, k_scale=k_scale),
            codes=packed,
            scales=sf,
            max_step=max_step,
        )


class UnimplementedPageCodec(PageCodec):
    """A dtype registered but not computed -- the guard's "unsupported" answer.

    Exists so the registry can say out loud which dtypes a real target model
    would still need work for. Nothing here is derived from the engine, so no
    page is laid out or interpreted on faith: :meth:`supported` is False, the
    raw-K channel refuses to arm, and copy-before-free (K1) keeps working on
    any dtype because it never asks a codec a question.
    """

    def __init__(self, name: str, dtype_prefixes: tuple[str, ...], missing: str):
        self.name = name
        self.dtype_prefixes = dtype_prefixes
        self.implemented = False
        self._missing = missing

    def _placeholder_error(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.name} KVMem page codec is a registered placeholder: "
            f"{self._missing} (design doc §1.3 -- it gets implemented when a "
            f"real target model needs it, not before)"
        )

    def supported(self, kv_cache_dtype: str) -> bool:
        return False

    def page_geometry(self, **_kwargs) -> PageGeometry:
        raise self._placeholder_error()

    def encode_rotary_prefix(self, *args, **kwargs) -> None:
        raise self._placeholder_error()

    def decode_rotary_prefix(self, *args, **kwargs) -> RotaryPrefix:
        raise self._placeholder_error()


NVFP4_PAGE_CODEC = Nvfp4PageCodec()

PAGE_CODECS: tuple[PageCodec, ...] = (
    NVFP4_PAGE_CODEC,
    UnimplementedPageCodec(
        "bf16",
        ("bfloat16", "bf16"),
        "the whole-head byte layout and the re-rotated prefix are unverified",
    ),
    UnimplementedPageCodec(
        "fp8",
        ("fp8",),
        "the scale-group layout (e4m3/e5m2, per-tensor vs per-block) is unverified",
    ),
)


def select_codec(kv_cache_dtype: str) -> PageCodec | None:
    """The codec registered for *kv_cache_dtype*, longest prefix winning.

    Returns None when nothing claims the dtype. A placeholder codec can be
    returned (callers must check :meth:`PageCodec.supported`); it is what makes
    "fp8 is known-but-not-implemented" different from "this dtype is not in the
    registry at all" in :func:`describe_registry`.
    """
    dtype = str(kv_cache_dtype or "").lower()
    match: tuple[int, PageCodec] | None = None
    for candidate in PAGE_CODECS:
        for prefix in candidate.dtype_prefixes:
            if dtype.startswith(prefix) and (match is None or len(prefix) > match[0]):
                match = (len(prefix), candidate)
                break
    return match[1] if match else None


def describe_registry() -> str:
    """One-line account of the registry, for the fail-fast message."""
    ready = [c.name for c in PAGE_CODECS if c.implemented]
    stubs = [c.name for c in PAGE_CODECS if not c.implemented]
    parts = [f"ready: {', '.join(ready) or 'none'}"]
    if stubs:
        parts.append(f"placeholder (not implemented): {', '.join(stubs)}")
    return "; ".join(parts)


def codec_for_dtype(kv_cache_dtype: str) -> PageCodec | None:
    """A codec that is *allowed to run* on this dtype, or None."""
    codec = select_codec(kv_cache_dtype)
    if codec is None or not codec.supported(kv_cache_dtype):
        return None
    return codec
