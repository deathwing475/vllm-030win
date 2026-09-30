"""Offline unit test for the KVMem rematerialisation primitive (step 063).

The design's stage-1 exit criterion for K3's second half is a *unit test*: a
page re-baked at a displaced position and then moved back must come out inside
one quantisation step of the original, and the maximum deviation must be
recorded. A silent bug here (wrong RoPE convention, wrong carve offset, a scale
group that straddles the rotary prefix) would look exactly like "retrieval
works but the model cannot read what came back", so this runs before any engine
wiring.

Every claim is anchored on the engine's own code, never on the primitive:

* the rotation is checked against a line-by-line port of the production
  ``_triton_mrope_forward`` kernel (both the equal-row text-only case and a
  synthetic unequal-row case, so the port itself is not taken on faith);
* the bytes are checked against the engine's own
  ``write_reference_nvfp4_cache`` writer, and read back through the engine's
  own ``side_carve_views`` so the offset arithmetic is derived twice,
  independently;
* the page itself is built in the real physical layout step 064 established
  empirically (kernel-block interleaved chunks of [K side | V side]) and the
  writer is invoked through kernel-block-granular illusion views, the way the
  engine hands the cache over -- a manager-block-sized view would silently
  rebuild the disproven layout and re-create the self-confirmation trap this
  test used to have.

Usage:
    python kvmem_remat_test.py [--device cpu|cuda]

Defaults to CPU: the production server holds ~15.9 of 16.3 GiB, so a CUDA
context is not available while it runs. Pass ``--device cuda`` with the server
stopped to additionally exercise the real Triton kernel.
"""
import argparse
import json
import os

import torch

MODEL_CONFIG = r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\config.json"
# Engine-reported for this arm (attn block size, step 060/061/062).
BLOCK_SIZE = 1424
# Engine kernel block size, step 064: the NVFP4 append kernel writes at
# 16-token granularity, so a page is 89 chunks of 18,432 B. Derived from the
# real cache as block_size // group_kernel_blocks(cache, nb).shape[1].
KERNEL_BLOCK_SIZE = 16
# Fallbacks if the checkpoint config is not readable.
FALLBACK = {
    "head_size": 256,
    "num_kv_heads": 4,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def load_geometry() -> dict:
    try:
        with open(MODEL_CONFIG, "r", encoding="utf-8") as fh:
            text_config = json.load(fh)["text_config"]
        return {
            "head_size": text_config["head_dim"],
            "num_kv_heads": text_config["num_key_value_heads"],
            "partial_rotary_factor": text_config["partial_rotary_factor"],
            "rope_parameters": text_config["rope_parameters"],
        }
    except Exception as exc:  # noqa: BLE001 - fall back, but say so
        print(f"  (model config unreadable: {exc!r}; using literals)")
        return dict(FALLBACK)


# ---------------------------------------------------------------------------
# A line-by-line port of vllm/model_executor/layers/rotary_embedding/mrope.py
# ::_triton_mrope_forward (NeoX branch). This is the kernel the production arm
# runs for these positions, so the primitive is checked against it rather than
# against itself.
# ---------------------------------------------------------------------------


def triton_mrope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    interleaved: bool,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin: [3, T, rotary_dim // 2]. q/k: [T, heads * head_size]."""
    half = rotary_dim // 2
    channels = torch.arange(half, device=q.device)
    if interleaved:
        h_mask = (channels % 3 == 1) & (channels <= 3 * mrope_section[1])
        w_mask = (channels % 3 == 2) & (channels <= 3 * mrope_section[2])
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = channels < mrope_section[0]
        h_mask = (channels >= mrope_section[0]) & (
            channels < mrope_section[0] + mrope_section[1]
        )
        w_mask = channels >= mrope_section[0] + mrope_section[1]

    # The kernel loads each row with `other=0` and sums the three masked rows.
    cos_row = cos[0] * t_mask + cos[1] * h_mask + cos[2] * w_mask
    sin_row = sin[0] * t_mask + sin[1] * h_mask + sin[2] * w_mask

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1, head_size)
        rot = x[..., :rotary_dim]
        if is_neox_style:
            x1, x2 = rot[..., :half], rot[..., half:]
        else:
            x1, x2 = rot[..., 0::2], rot[..., 1::2]
        c = cos_row.unsqueeze(-2)
        s = sin_row.unsqueeze(-2)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        if is_neox_style:
            out = torch.cat((o1, o2), dim=-1)
        else:
            out = torch.stack((o1, o2), dim=-1).flatten(-2)
        return torch.cat((out, x[..., rotary_dim:]), dim=-1).reshape(x.shape)

    return rotate(q), rotate(k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()

    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.v1.attention.reference_nvfp4 import (
        side_carve_views,
        write_reference_nvfp4_cache,
    )
    from vllm.v1.kvmem_workspace.remat import (
        PageGeometry,
        bake_rotated_k,
        dequantize_rotated,
        quantize_rotated,
        read_rotated,
        rematerialize_page,
        rotated_byte_offsets,
        rotated_prefix_from_packed_k,
        write_rotated,
    )

    cfg = load_geometry()
    head_size = cfg["head_size"]
    num_heads = cfg["num_kv_heads"]
    rotary_dim = int(head_size * cfg["partial_rotary_factor"])
    rope_parameters = cfg["rope_parameters"]
    mrope_section = list(rope_parameters["mrope_section"])
    geom = PageGeometry(
        head_size, num_heads, BLOCK_SIZE, rotary_dim, KERNEL_BLOCK_SIZE
    )

    print("KVMem rematerialisation unit test (step 063, real layout per 064)")
    print(
        f"  geometry: head_size={head_size} heads={num_heads} block={BLOCK_SIZE} "
        f"kernel_block={KERNEL_BLOCK_SIZE} rotary_dim={rotary_dim} "
        f"full_dim={geom.full_dim}"
    )
    print(
        f"  page: {geom.page_bytes} B = {geom.chunks_per_page} chunks x "
        f"{geom.chunk_bytes} B (per chunk K side {geom.side_chunk_bytes} B: "
        f"data {geom.k_data_bytes} B + scales; then V side; rot prefix "
        f"{geom.rot_data_bytes} B data + {geom.rot_scale_bytes} B scales per head)"
    )
    print(f"  device: {device}")

    # The model builds its rotary embedding exactly like qwen3_next.py:409.
    # `MRotaryEmbedding` is a CustomOp, so it needs a vLLM config context even
    # though nothing here is compiled or dispatched.
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        rotary_emb = get_rope(
            head_size=head_size,
            max_position=262144,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=None,
        )
    cos_sin_cache = rotary_emb.cos_sin_cache
    assert rotary_emb.rotary_dim == rotary_dim, rotary_emb.rotary_dim
    assert type(rotary_emb).__name__ == "MRotaryEmbedding", type(rotary_emb).__name__
    print(
        f"  rope: {type(rotary_emb).__name__} cache {tuple(cos_sin_cache.shape)} "
        f"{cos_sin_cache.dtype} is_neox={rotary_emb.is_neox_style} "
        f"interleaved={rotary_emb.mrope_interleaved} section={mrope_section}"
    )

    torch.manual_seed(20260930)
    dtype = torch.bfloat16
    cache = cos_sin_cache.to(device=device, dtype=dtype)

    # ---------------------------------------------------------------- T0
    print("\nT0 geometry constants (real layout, step 064)")
    check("page_bytes == chunks * chunk_bytes == 1640448",
          geom.page_bytes == 1640448
          and geom.page_bytes == geom.chunks_per_page * geom.chunk_bytes,
          f"{geom.page_bytes}")
    check("chunk == 89 x 18432 B (kernel block interleaved)",
          geom.chunks_per_page == 89 and geom.chunk_bytes == 18432,
          f"{geom.chunks_per_page} x {geom.chunk_bytes} B")
    check("K-side data region per chunk == heads*kbs*data_dim == 8192",
          geom.k_data_bytes == 8192, f"{geom.k_data_bytes}")
    check("rotary prefix ends on a scale-group boundary", rotary_dim % 16 == 0,
          f"{rotary_dim} = {rotary_dim // 16} groups")

    # ---------------------------------------------------------------- T1
    print("\nT1 rotation vs a port of the production triton_mrope kernel")
    tokens = torch.arange(BLOCK_SIZE, device=device)
    raw = (torch.randn(BLOCK_SIZE, num_heads * head_size, dtype=torch.float32) * 0.05)
    raw = raw.to(dtype).to(device)
    q_dummy = torch.zeros_like(raw)

    pos1d = (1000 + tokens).to(device)
    pos2d = pos1d.unsqueeze(0).expand(3, -1).contiguous()
    cos_sin2d = cache[pos2d]
    cos2, sin2 = cos_sin2d.chunk(2, dim=-1)
    _, k_kernel = triton_mrope_reference(
        q_dummy, raw.clone(), cos2, sin2, mrope_section, head_size, rotary_dim,
        bool(rotary_emb.mrope_interleaved), bool(rotary_emb.is_neox_style),
    )
    ours1d = bake_rotated_k(
        raw.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim].contiguous(),
        pos1d, cache,
    )
    kernel_rot = k_kernel.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim]
    d = (ours1d.float() - kernel_rot.float()).abs().max().item()
    check("text-only 2-D positions == 1-D positions, vs triton port", d == 0.0,
          f"max|d|={d:.3e}")

    ours2d = bake_rotated_k(
        raw.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim].contiguous(),
        pos2d, cache, mrope_section=mrope_section,
    )
    d2 = (ours2d.float() - kernel_rot.float()).abs().max().item()
    check("2-D path with mrope_section == triton port (text-only)", d2 == 0.0,
          f"max|d|={d2:.3e}")

    # Unequal rows: forces the interleaved permutation to actually do work, so
    # the port is validated rather than trivially equal.
    pos_uneq = torch.stack(
        [pos1d, pos1d + 7, pos1d + 13]
    ).contiguous()
    cos_sin_u = cache[pos_uneq]
    cosu, sinu = cos_sin_u.chunk(2, dim=-1)
    _, k_uneq = triton_mrope_reference(
        q_dummy, raw.clone(), cosu, sinu, mrope_section, head_size, rotary_dim,
        bool(rotary_emb.mrope_interleaved), bool(rotary_emb.is_neox_style),
    )
    ours_uneq = bake_rotated_k(
        raw.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim].contiguous(),
        pos_uneq, cache, mrope_section=mrope_section,
    )
    kernel_uneq = k_uneq.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim]
    d3 = (ours_uneq.float() - kernel_uneq.float()).abs().max().item()
    check("2-D path with unequal rows == triton port (permutation live)",
          d3 == 0.0, f"max|d|={d3:.3e}")
    d4 = (ours_uneq.float() - ours1d.float()).abs().max().item()
    check("unequal rows really differ from the text-only result", d4 > 0.0,
          f"max|d|={d4:.3e}")

    # ---------------------------------------------------------------- T2
    print("\nT2 bytes vs the engine's own NVFP4 writer (real layout)")
    num_pages = 2
    # The real physical buffer: a page is a flat byte range whose 18,432 B
    # chunks are [K side | V side]. The writer reaches it through
    # kernel-block-granular NHD illusion views, exactly how the engine hands
    # the cache over -- step 064 established that side_carve_views' formula
    # holds per kernel block, NOT per manager block, so a manager-block-sized
    # view here would rebuild the disproven layout and re-create the
    # self-confirmation trap this test used to have.
    phys = torch.zeros(num_pages * geom.page_bytes, dtype=torch.uint8,
                       device=device)
    kv_shape = (
        num_pages * geom.chunks_per_page, KERNEL_BLOCK_SIZE, num_heads,
        geom.full_dim,
    )
    kv_stride = (geom.chunk_bytes, geom.full_dim,
                 KERNEL_BLOCK_SIZE * geom.full_dim, 1)
    key_cache = phys.as_strided(kv_shape, kv_stride, 0)
    value_cache = phys.as_strided(kv_shape, kv_stride, geom.side_chunk_bytes)

    raw_k = raw.view(BLOCK_SIZE, num_heads, head_size)
    v = (torch.randn(BLOCK_SIZE, num_heads * head_size, dtype=torch.float32) * 0.05)
    v = v.to(dtype).to(device).view(BLOCK_SIZE, num_heads, head_size)
    k_scale = 1.0
    # Engine path: full-row rotation through the model's own module, then the
    # engine writer. `positions` reaches the model as [3, T].
    _, k_engine = rotary_emb(pos2d, q_dummy.view(BLOCK_SIZE, num_heads, head_size),
                             raw_k.clone())
    slot_mapping = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)
    write_reference_nvfp4_cache(
        k_engine.reshape(BLOCK_SIZE, num_heads, head_size).contiguous(),
        v.contiguous(),
        key_cache, value_cache, slot_mapping,
        torch.tensor(k_scale, dtype=torch.float32, device=device),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
    )
    engine_page = phys[: geom.page_bytes].clone()
    check("the writer only touched page 0 (slot -> kernel block mapping)",
          int(phys[geom.page_bytes:].sum()) == 0,
          f"{int(phys[geom.page_bytes:].sum())} stray bytes on page 1")

    ours_page = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    rematerialize_page(
        ours_page, geom,
        raw_k[..., :rotary_dim].contiguous(), tokens, pos2d, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    data_off, scale_off = rotated_byte_offsets(geom, tokens)
    e_rot = torch.cat(
        [engine_page[data_off.reshape(-1)], engine_page[scale_off.reshape(-1)]]
    )
    o_rot = torch.cat(
        [ours_page[data_off.reshape(-1)], ours_page[scale_off.reshape(-1)]]
    )
    mismatch = int((e_rot != o_rot).sum())
    check("rotary bytes == engine writer, bit for bit", mismatch == 0,
          f"{e_rot.numel()} bytes, {mismatch} mismatch")

    touched = torch.zeros(geom.page_bytes, dtype=torch.bool, device=device)
    touched[data_off.reshape(-1)] = True
    touched[scale_off.reshape(-1)] = True
    outside = ours_page[~touched]
    check("nothing outside the rotary prefix was written",
          int(outside.abs().sum()) == 0,
          f"{int(touched.sum())} bytes touched, {int((~touched).sum())} untouched")

    # ---------------------------------------------------------------- T3
    print("\nT3 offsets re-derived through the engine's own carve")
    # Carve the kernel-block-granular view: [chunk, heads, kbs, w] ->
    # [chunk, kbs, heads, w] -> [t = chunk*kbs + r, heads, w].
    carve_data, carve_scale = side_carve_views(key_cache)
    cd = (
        carve_data[: geom.chunks_per_page, :, :, : geom.rot_data_bytes]
        .permute(0, 2, 1, 3)
        .reshape(BLOCK_SIZE, num_heads, geom.rot_data_bytes)
    )
    cs = (
        carve_scale[: geom.chunks_per_page, :, :, : geom.rot_scale_bytes]
        .permute(0, 2, 1, 3)
        .reshape(BLOCK_SIZE, num_heads, geom.rot_scale_bytes)
    )
    ours_packed, ours_sf = read_rotated(ours_page, geom, tokens)
    check("carve data slice == primitive's view",
          bool(torch.equal(cd.contiguous(), ours_packed.contiguous())),
          f"shape {tuple(ours_packed.shape)}")
    check("carve scale slice == primitive's view",
          bool(torch.equal(cs.contiguous().view(torch.uint8),
                           ours_sf.contiguous().view(torch.uint8))),
          f"shape {tuple(ours_sf.shape)}")
    check("engine page's carve slice == primitive's view",
          bool(torch.equal(cd.contiguous(), e_rot[: cd.numel()].reshape(cd.shape))))

    # ---------------------------------------------------------------- T4
    print("\nT4 statelessness: displacement introduces no drift")
    disp = 4096
    pos_disp = (pos2d + disp).contiguous()
    moved = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    rematerialize_page(
        moved, geom, raw_k[..., :rotary_dim].contiguous(), tokens, pos_disp, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    back = moved.clone()
    rematerialize_page(
        back, geom, raw_k[..., :rotary_dim].contiguous(), tokens, pos2d, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    check("moved then moved back == original, bit for bit",
          bool(torch.equal(back, ours_page)),
          f"{int((back != ours_page).sum())} bytes differ")
    check("the displacement really changed the page (test is not vacuous)",
          int((moved != ours_page).sum()) > 0,
          f"{int((moved != ours_page).sum())} bytes differ at d={disp}")

    # Eight displacements, as the reference implementation's >210K cross-talk
    # scenario would have produced under delta re-RoPE.
    walk = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    for step in range(8):
        rematerialize_page(
            walk, geom, raw_k[..., :rotary_dim].contiguous(), tokens,
            (pos2d + disp * (step + 1)).contiguous(), cache, k_scale=k_scale,
            is_neox_style=bool(rotary_emb.is_neox_style),
            mrope_section=mrope_section,
        )
    rematerialize_page(
        walk, geom, raw_k[..., :rotary_dim].contiguous(), tokens, pos2d, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    check("8 displacements then back == original, bit for bit",
          bool(torch.equal(walk, ours_page)),
          f"{int((walk != ours_page).sum())} bytes differ")

    # ---------------------------------------------------------------- T5
    print("\nT5 quantisation error")
    post = bake_rotated_k(
        raw_k[..., :rotary_dim].contiguous(), pos2d, cache,
        is_neox_style=bool(rotary_emb.is_neox_style), mrope_section=mrope_section,
    )
    packed, sf = quantize_rotated(post, k_scale=k_scale)
    deq = dequantize_rotated(packed, sf, k_scale=k_scale)
    err = (deq - post.float()).abs()
    # Per-group step at the top of the E2M1 range, in dequantised units.
    sf_value = sf.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    output_scale = torch.where(sf_value > 0, k_scale / sf_value,
                               torch.zeros_like(sf_value))
    step = (6.0 - 4.0) / output_scale
    max_err = err.max().item()
    max_step = step.max().item()
    ratio = max_err / max_step
    print(f"    max |dequant - post_rope| = {max_err:.6e}")
    print(f"    max E2M1 step             = {max_step:.6e}")
    print(f"    ratio (err / step)        = {ratio:.4f}")
    check("error within one quantisation step", ratio <= 0.6,
          f"ratio {ratio:.4f}")

    # ---------------------------------------------------------------- T6
    print("\nT6 a re-bake leaves the position-independent bytes alone")
    keep = engine_page.clone()
    rematerialize_page(
        keep, geom, raw_k[..., :rotary_dim].contiguous(), tokens, pos_disp, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    same = keep[~touched]
    ref = engine_page[~touched]
    check("non-rotary bytes identical after re-bake", bool(torch.equal(same, ref)),
          f"{int((same != ref).sum())} of {same.numel()} bytes differ")
    check("rotary bytes did change on that same page",
          int((keep[touched] != engine_page[touched]).sum()) > 0)

    # ---------------------------------------------------------------- T7
    print("\nT7 page-level round trip through the engine writer (all 256 dims)")
    # Decompose an engine page into "authority" (pre-RoPE rotary prefix) plus
    # the untouched bytes, then rebuild and compare to the engine's page.
    rebuilt = engine_page.clone()
    rematerialize_page(
        rebuilt, geom, raw_k[..., :rotary_dim].contiguous(), tokens, pos2d, cache,
        k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
        mrope_section=mrope_section,
    )
    check("rebuild from pre-RoPE authority == engine page, bit for bit",
          bool(torch.equal(rebuilt, engine_page)),
          f"{int((rebuilt != engine_page).sum())} bytes differ")

    # ---------------------------------------------------------------- T8
    print("\nT8 packed-K slicing (the authority write path, step 064)")
    # External anchor: every (head, dim) carries a distinct value, so a wrong
    # head/dim mapping cannot pass. `k[:, :rotary_dim]` would take head 0's
    # prefix for every token and look plausible, which is exactly the silent
    # failure this checks against.
    packed = (
        torch.arange(num_heads * head_size, dtype=torch.float32)
        .repeat(BLOCK_SIZE, 1)
        .to(dtype)
    )
    sliced = rotated_prefix_from_packed_k(packed, head_size, rotary_dim)
    expect = (
        torch.arange(num_heads * head_size)
        .reshape(num_heads, head_size)[:, :rotary_dim]
        .repeat(BLOCK_SIZE, 1, 1)
    )
    check("slicing takes each head's first rotary_dim dims",
          bool(torch.equal(sliced, expect)),
          f"shape {tuple(sliced.shape)}")
    check("the naive k[:, :rotary_dim] is a different (wrong) answer",
          not bool(torch.equal(sliced.reshape(BLOCK_SIZE, -1),
                               packed[:, :rotary_dim])),
          "head axis really is restored")

    # ---------------------------------------------------------------- T9
    print("\nT9 authority token indexing (absolute offsets, gaps allowed)")
    import numpy as np

    capacity = 4096
    region = torch.zeros((capacity, num_heads * rotary_dim), dtype=torch.float16)
    # A gap models a prefix-cache hit: the step's rows are not contiguous.
    offsets = np.array([10, 11, 12, 500, 501], dtype=np.int64)
    rows = np.arange(
        offsets.size * num_heads * rotary_dim, dtype=np.float32
    ).reshape(offsets.size, -1).astype(np.float16)
    region.numpy()[offsets] = rows
    check("rows read back by absolute token offset",
          bool(np.array_equal(region.numpy()[offsets], rows)))
    check("offsets nobody wrote stay zero",
          int(region.numpy()[13:500].sum()) == 0)

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
