from __future__ import annotations

import torch


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    mag = values.abs()
    code = torch.where(
        mag <= 0.25,
        0,
        torch.where(
            mag < 0.75,
            1,
            torch.where(
                mag <= 1.25,
                2,
                torch.where(
                    mag < 1.75,
                    3,
                    torch.where(
                        mag <= 2.5,
                        4,
                        torch.where(mag < 3.5, 5, torch.where(mag <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    return code | ((values < 0).to(torch.uint8) << 3)


def side_carve_views(side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Carved (data, scale) views for one NVFP4 KV side.

    Only the side's storage_offset (side base), stride(0) (bytes per page block)
    and shape participate -- the 144-byte row strides of the NHD illusion view
    are intentionally ignored.  Both the torch reference writer and FlashInfer's
    nvfp4 append kernel (``kv_layout="HND"``) write through exactly these
    targets, which is why the two produce byte-identical caches.

    Returns (data, scale): uint8 ``[blocks, heads, block, data_dim]`` and
    float8_e4m3fn ``[blocks, heads, block, scale_dim]``.
    """
    num_blocks, block_size, num_heads, full_dim = side.shape
    data_dim = full_dim * 8 // 9
    scale_dim = full_dim - data_dim
    S0 = side.stride(0)
    base = side.storage_offset()
    data = torch.as_strided(
        side,
        (num_blocks, num_heads, block_size, data_dim),
        (S0, block_size * data_dim, data_dim, 1),
        base,
    )
    scale = torch.as_strided(
        side,
        (num_blocks, num_heads, block_size, scale_dim),
        (S0, block_size * scale_dim, scale_dim, 1),
        base + num_heads * block_size * data_dim,
    )
    return data, scale.view(torch.float8_e4m3fn)


def write_nvfp4_cache_flashinfer(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Write the carved NVFP4 KV cache with FlashInfer's CUDA kernel.

    Byte-identical to :func:`write_reference_nvfp4_cache` -- both were run on the
    same inputs by ``research/src/nvfp4_writer_swap_probe.py`` and the real XQA
    kernel read both at cosine 1.0000 -- but this is one fused kernel per layer
    instead of ~40 torch ops, worth ~8.4% of the decode step (E5.3 ablation).
    Negative slot entries are ignored by the kernel, so it stays CUDA-graph safe
    without the clamp workaround.
    """
    from flashinfer.page import (
        nvfp4_quantize_append_paged_kv_cache_with_slot_mapping as _append,
    )

    if key.numel() == 0:
        return

    def _scalar(t):
        # No-op when already a CUDA fp32 scalar (the capture case), so this adds
        # neither a host sync nor a copy under CUDA graph capture.
        if isinstance(t, torch.Tensor):
            return t.to(device=key.device, dtype=torch.float32).reshape(())
        return float(t)

    k_data, k_sf = side_carve_views(key_cache)
    v_data, v_sf = side_carve_views(value_cache)
    _append(
        key,
        value,
        slot_mapping,
        (k_data, v_data),
        (k_sf, v_sf),
        _scalar(k_scale),
        _scalar(v_scale),
        kv_layout="HND",
    )


def write_reference_nvfp4_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Reference SM120 writer for the carved NVFP4 KV cache layout.

    key_cache/value_cache are ``[blocks, page, heads, full_dim]`` uint8 NHD
    views (full_dim = data_dim + scale_dim = 144 for head_size 256), but the
    physical side layout is NOT interleaved per-row ``[data | scale]``: each
    page owns one contiguous byte range per side, itself split into a data
    region (``heads * page * data_dim`` bytes) followed by a scale region
    (``heads * page * scale_dim`` bytes) — the layout PR50288's
    reshape_and_cache_nvfp4 writes (scales at
    ``side_base + heads * page * data_dim``) and nvfp4_split_data_scale plus
    every reader (FA2 fp4 prefill, XQA decode) read back. Writing interleaved
    rows here silently corrupts all NVFP4 reads ('!' storm).

    This intentionally favors correctness and inspectability over throughput.
    """
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("reference NVFP4 writer expects matching 4D cache views")
    num_blocks, block_size, num_heads, full_dim = key_cache.shape
    _, input_heads, head_size = key.shape
    data_dim = full_dim * 8 // 9
    scale_dim = full_dim - data_dim
    if input_heads != num_heads or full_dim != data_dim + scale_dim:
        raise ValueError(
            f"unexpected NVFP4 layout: key={tuple(key.shape)} cache={tuple(key_cache.shape)}"
        )
    if key_cache.dtype != torch.uint8:
        raise ValueError("reference NVFP4 writer expects a uint8 cache view")
    # CUDA-graph capture requires zero host syncs: the eager-era
    # `bool(valid.all())` filter was a D2H sync and aborted capture
    # (cudaErrorStreamCaptureUnsupported). Under the shipped config
    # (cudagraph capture_sizes=[1], max_num_seqs=1) slot_mapping never
    # contains -1 padding; clamp(min=0) keeps indices in range without a
    # sync. A padded row would alias slot 0 — not reachable here.
    slot_mapping = slot_mapping.clamp(min=0)
    if key.numel() == 0:
        return

    def quantize(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grouped = x.reshape(x.shape[0], num_heads, scale_dim, 16)
        amax = grouped.abs().amax(dim=-1)
        global_scale = 1.0 / scale.to(device=x.device, dtype=torch.float32).reshape(())
        sf = (amax / (6.0 * global_scale)).to(torch.float8_e4m3fn)
        sf_value = sf.float()
        output_scale = torch.where(
            sf_value > 0,
            1.0 / (sf_value * global_scale),
            torch.zeros_like(sf_value),
        )
        codes = _e2m1_codes(grouped * output_scale.unsqueeze(-1))
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        return packed.reshape(x.shape[0], num_heads, data_dim), sf.view(torch.uint8)

    blocks = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = slot_mapping.remainder(block_size)

    def side_targets(side: torch.Tensor):
        """Carved (data, scale) write targets for one side view.

        Delegates to :func:`side_carve_views` so this torch path and the
        FlashInfer kernel path can never drift apart -- the two must write the
        identical carve or the readers see a corrupt cache ('!' storm).
        """
        data, scale = side_carve_views(side)
        # This path assigns into the uint8 scale region directly, so undo the
        # fp8 reinterpretation the shared helper applies for the kernel API.
        return data, scale.view(torch.uint8)

    k_data, k_sf = quantize(key, k_scale)
    v_data, v_sf = quantize(value, v_scale)
    k_data_tgt, k_sf_tgt = side_targets(key_cache)
    v_data_tgt, v_sf_tgt = side_targets(value_cache)
    k_data_tgt[blocks, :, offsets] = k_data
    k_sf_tgt[blocks, :, offsets] = k_sf
    v_data_tgt[blocks, :, offsets] = v_data
    v_sf_tgt[blocks, :, offsets] = v_sf
