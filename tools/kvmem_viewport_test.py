"""Offline unit test for the fixed-slot compressed window (step 072).

Two halves of the step 072 mechanism are pinned here, both anchored on engine
code rather than on the primitive under test:

* T0 pins the window layout arithmetic (design §5.1): S = one page, the
  retrieval budget N in whole pages, the recent tail R, and the rewritten
  prefill sequence ``prompt[:S+N] + prompt[L-R:]`` -- whose head keeps the
  original positions and whose tail is the prompt's own last R tokens. This is
  the invariant the whole route rests on: no window position may depend on
  what retrieval later selects.
* T1-T4 pin the slot bake: a stored page's rotary prefix rebuilt *at a slot
  position* must equal what the engine's own ``write_reference_nvfp4_cache``
  writer puts there for the same pre-RoPE K (bit for bit, exactly the anchor
  the step 065 test used), everything outside the rotary prefix must keep the
  stored page's bytes, and re-baking from the same raw rows is bit-identical
  (no accumulation, the design's zero-drift rule).

Usage:
    python kvmem_viewport_test.py [--device cpu|cuda]

Defaults to CPU; the production server holds ~15.9 of 16.3 GiB.
"""
import argparse

import torch

from kvmem_remat_test import (
    BLOCK_SIZE,
    KERNEL_BLOCK_SIZE,
    MODEL_CONFIG,
    check,
    load_geometry,
    triton_mrope_reference,
    _results,
)

# The arm's window configuration (config.py defaults): 55 retrieval pages.
RETRIEVAL_PAGES = 55
RECENT_TOKENS = 16384
GEN_RESERVE = 32768
# The pool the window must fit into (the arm's sliding window).
SLIDING_WINDOW = 163072


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()

    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.v1.attention.reference_nvfp4 import (
        write_reference_nvfp4_cache,
    )
    from vllm.v1.kvmem_workspace.codec import NVFP4_PAGE_CODEC as nvfp4
    from vllm.v1.kvmem_workspace.remat import rematerialize_page

    cfg = load_geometry()
    head_size = cfg["head_size"]
    num_heads = cfg["num_kv_heads"]
    rotary_dim = int(head_size * cfg["partial_rotary_factor"])
    rope_parameters = cfg["rope_parameters"]
    mrope_section = list(rope_parameters["mrope_section"])
    geom = nvfp4.page_geometry(
        head_size=head_size,
        num_heads=num_heads,
        block_size=BLOCK_SIZE,
        rotary_dim=rotary_dim,
        kernel_block_size=KERNEL_BLOCK_SIZE,
    )

    print("KVMem viewport unit test (step 072, fixed-slot compressed window)")
    print(f"  geometry: head_size={head_size} heads={num_heads} "
          f"block={BLOCK_SIZE} kernel_block={KERNEL_BLOCK_SIZE} "
          f"rotary_dim={rotary_dim}")
    print(f"  device: {device}")

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        rotary_emb = get_rope(
            head_size=head_size,
            max_position=262144,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=None,
        )
    cos_sin_cache = rotary_emb.cos_sin_cache
    assert type(rotary_emb).__name__ == "MRotaryEmbedding", type(rotary_emb).__name__

    torch.manual_seed(20261001)
    dtype = torch.bfloat16
    cache = cos_sin_cache.to(device=device, dtype=dtype)
    cache_fp32 = cos_sin_cache.to(device=device)

    # ---------------------------------------------------------------- T0
    print("\nT0 window layout arithmetic (design 5.1, arm defaults)")
    sink = BLOCK_SIZE
    retrieval = RETRIEVAL_PAGES * BLOCK_SIZE
    window = sink + retrieval + RECENT_TOKENS
    prompt_len = 200_000
    head = sink + retrieval
    check("sink is exactly one page", sink == BLOCK_SIZE, f"S={sink}")
    check("retrieval budget is whole pages",
          retrieval % BLOCK_SIZE == 0 and RETRIEVAL_PAGES == 55,
          f"N={RETRIEVAL_PAGES} pages = {retrieval} tokens")
    check("slot rows start page-aligned after the sink",
          head % BLOCK_SIZE == 0, f"S+N={head}")
    check("window leaves room for the generation reserve inside the pool",
          window + GEN_RESERVE <= SLIDING_WINDOW,
          f"{window} + {GEN_RESERVE} = {window + GEN_RESERVE} "
          f"<= {SLIDING_WINDOW}")
    check("a 200K prompt has a mid-section to re-represent",
          prompt_len > window,
          f"L={prompt_len} > B={window}; mid-section "
          f"{prompt_len - window} tokens")
    # The rewrite: head keeps original positions, tail is the prompt's own end.
    prompt = torch.arange(prompt_len, dtype=torch.int64)
    rewritten = torch.cat([prompt[:head], prompt[prompt_len - RECENT_TOKENS:]])
    check("rewritten sequence length == window",
          rewritten.numel() == window, f"{rewritten.numel()}")
    check("head section is the prompt's own leading tokens",
          torch.equal(rewritten[:head], prompt[:head]))
    check("recent section is the prompt's own last R tokens",
          torch.equal(rewritten[head:], prompt[prompt_len - RECENT_TOKENS:]))
    check("placeholder section keeps original positions",
          torch.equal(rewritten[sink:head], prompt[sink:head]),
          "slot KV baked by the engine writer for those tokens is correct "
          "as-is; only the slots need baking afterwards")

    # ------------------------------------------------- T1-T4 shared setup
    # A "stored page" in the workspace: engine-written bytes at the page's
    # original positions (kernel-block layout, the step 065 construction).
    src_page = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    kv_shape = (
        geom.chunks_per_page, KERNEL_BLOCK_SIZE, num_heads, geom.full_dim
    )
    kv_stride = (geom.chunk_bytes, geom.full_dim,
                 KERNEL_BLOCK_SIZE * geom.full_dim, 1)
    key_cache = src_page.as_strided(kv_shape, kv_stride, 0)
    value_cache = src_page.as_strided(kv_shape, kv_stride, geom.side_chunk_bytes)

    tokens = torch.arange(BLOCK_SIZE, device=device)
    raw_k = (torch.randn(BLOCK_SIZE, num_heads, head_size, dtype=torch.float32)
             * 0.05).to(dtype).to(device)
    v = (torch.randn(BLOCK_SIZE, num_heads, head_size, dtype=torch.float32)
         * 0.05).to(dtype).to(device)
    q_dummy = torch.zeros(BLOCK_SIZE, num_heads, head_size, dtype=dtype,
                          device=device)
    k_scale = 1.0

    orig_start = 35_328  # page 25: the first page past the eviction boundary
    _, k_orig = rotary_emb(
        (orig_start + tokens).unsqueeze(0).expand(3, -1).contiguous(),
        q_dummy, raw_k.clone(),
    )
    # The writer derives its block split from the view's block_size (16), so
    # slot_mapping is page-relative; the original *position* only enters
    # through the rotation.
    slot_mapping = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)
    write_reference_nvfp4_cache(
        k_orig.contiguous(), v.contiguous(), key_cache, value_cache,
        slot_mapping,
        torch.tensor(k_scale, dtype=torch.float32, device=device),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
    )
    stored = src_page.clone()

    def bake(page: torch.Tensor, slot_start: int) -> None:
        # Simulate the worker's path: the authority holds fp16 rows (exact for
        # bf16 values), restored to bf16 before the bake.
        raw_rows = raw_k[..., :rotary_dim].reshape(BLOCK_SIZE, -1).to(
            torch.float16
        ).numpy()
        raw = torch.from_numpy(raw_rows.copy()).to(torch.bfloat16).to(device)
        raw = raw.view(BLOCK_SIZE, num_heads, rotary_dim)
        dst_positions = torch.arange(
            slot_start, slot_start + BLOCK_SIZE, device=device
        )
        # The worker reads fp32 cos/sin straight from the engine cache (the
        # step 065 precision contract: fp32 cos/sin, bf16 K); the full cache
        # goes in and rematerialize_page indexes the slot rows itself.
        rematerialize_page(
            page, nvfp4, geom, raw.cpu(), tokens.cpu(), dst_positions.cpu(),
            cache_fp32.cpu(),
            k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
            mrope_section=mrope_section,
        )

    data_off, scale_off = nvfp4.rotated_byte_offsets(geom, tokens.cpu())
    rot_mask = torch.zeros(geom.page_bytes, dtype=torch.bool)
    rot_mask[data_off.reshape(-1)] = True
    rot_mask[scale_off.reshape(-1)] = True

    # ---------------------------------------------------------------- T1
    print("\nT1 slot bake vs the engine's own writer, at the slot position")
    slot_start = 1456 + 3 * BLOCK_SIZE  # retrieval slot j=3 (S + 3 pages)
    slot_page = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    # Engine anchor: the writer itself, at the slot's absolute positions.
    _, k_slot = rotary_emb(
        (slot_start + tokens).unsqueeze(0).expand(3, -1).contiguous(),
        q_dummy, raw_k.clone(),
    )
    slot_page_kv = torch.zeros(
        geom.page_bytes, dtype=torch.uint8, device=device
    )
    key_slot = slot_page_kv.as_strided(kv_shape, kv_stride, 0)
    value_slot = slot_page_kv.as_strided(kv_shape, kv_stride,
                                         geom.side_chunk_bytes)
    # The writer's slot_mapping is page-relative (it indexes the 89 kernel
    # blocks of the one page handed to it); the slot's *position* only enters
    # through the rotation.
    write_reference_nvfp4_cache(
        k_slot.contiguous(), v.contiguous(), key_slot, value_slot,
        torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
    )
    # The bake, starting from the stored page's bytes (V + non-rotary K), at
    # the same slot positions, fp32 cos/sin.
    baked = stored.clone()
    bake(baked, slot_start)
    engine_rot = torch.cat([
        slot_page_kv[data_off.reshape(-1)], slot_page_kv[scale_off.reshape(-1)]
    ])
    baked_rot = torch.cat([
        baked[data_off.reshape(-1)], baked[scale_off.reshape(-1)]
    ])
    mismatch = int((engine_rot != baked_rot).sum())
    check("slot-baked rotary bytes == engine writer at the slot position",
          mismatch == 0,
          f"{engine_rot.numel()} bytes, {mismatch} mismatch")

    # ---------------------------------------------------------------- T2
    print("\nT2 only the rotary prefix moves; V and non-rotary K stay")
    outside_baked = baked[~rot_mask]
    outside_stored = stored[~rot_mask]
    check("non-rotary bytes of the baked page == the stored page's",
          torch.equal(outside_baked, outside_stored),
          f"{int((~rot_mask).sum())} bytes")
    check("the baked page's rotary prefix differs from the stored page's",
          not torch.equal(baked[rot_mask], stored[rot_mask]),
          "the position actually participates (T1 already proved it is "
          "*correctly* different)")

    # ---------------------------------------------------------------- T3
    print("\nT3 two slots, time-ordered, from two pages")
    raw_k2 = (torch.randn(BLOCK_SIZE, num_heads, head_size,
                          dtype=torch.float32) * 0.05).to(dtype).to(device)
    src2 = torch.zeros(geom.page_bytes, dtype=torch.uint8, device=device)
    key2 = src2.as_strided(kv_shape, kv_stride, 0)
    value2 = src2.as_strided(kv_shape, kv_stride, geom.side_chunk_bytes)
    _, k_orig2 = rotary_emb(
        (orig_start + tokens).unsqueeze(0).expand(3, -1).contiguous(),
        q_dummy, raw_k2.clone(),
    )
    write_reference_nvfp4_cache(
        k_orig2.contiguous(), v.contiguous(), key2, value2,
        torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
        torch.tensor(k_scale, dtype=torch.float32, device=device),
    )
    baked1 = stored.clone()
    bake(baked1, 1456 + 0 * BLOCK_SIZE)
    # Second page: re-run the bake with raw_k2 in place of raw_k.
    def bake2(page: torch.Tensor, slot_start: int) -> None:
        raw_rows = raw_k2[..., :rotary_dim].reshape(BLOCK_SIZE, -1).to(
            torch.float16
        ).numpy()
        raw = torch.from_numpy(raw_rows.copy()).to(torch.bfloat16)
        raw = raw.view(BLOCK_SIZE, num_heads, rotary_dim)
        dst_positions = torch.arange(slot_start, slot_start + BLOCK_SIZE)
        rematerialize_page(
            page, nvfp4, geom, raw, tokens.cpu(), dst_positions, cache_fp32.cpu(),
            k_scale=k_scale, is_neox_style=bool(rotary_emb.is_neox_style),
            mrope_section=mrope_section,
        )
    baked2 = src2.clone()
    bake2(baked2, 1456 + 1 * BLOCK_SIZE)
    check("slot 0 and slot 1 rotary bytes differ (distinct pages, "
          "distinct positions)",
          not torch.equal(baked1[rot_mask], baked2[rot_mask]))
    check("slot 1's non-rotary bytes == page 2's stored bytes",
          torch.equal(baked2[~rot_mask], src2[~rot_mask]))

    # ---------------------------------------------------------------- T4
    print("\nT4 re-bake idempotence (zero accumulation from the same raw)")
    again = stored.clone()
    bake(again, slot_start)
    check("baking the same raw rows twice is bit-identical",
          torch.equal(again, baked),
          "no delta re-RoPE is involved anywhere: every bake starts from the "
          "pre-RoPE authority")

    # ---------------------------------------------------------------- T5
    print("\nT5 slot bake vs the line-by-line triton port (dequantised)")
    packed, sf = nvfp4.read_rotated(baked, geom, tokens.cpu())
    deq = nvfp4.dequantize_rotated(packed, sf)
    half = rotary_dim // 2
    cos_sin_t = cache_fp32[torch.arange(
        slot_start, slot_start + BLOCK_SIZE
    )]
    cos_t, sin_t = cos_sin_t.chunk(2, dim=-1)
    cos3 = cos_t.unsqueeze(0).expand(3, -1, -1)
    sin3 = sin_t.unsqueeze(0).expand(3, -1, -1)
    _, k_ref = triton_mrope_reference(
        q_dummy.cpu().view(BLOCK_SIZE, -1), raw_k.cpu().view(BLOCK_SIZE, -1),
        cos3, sin3, mrope_section, head_size, rotary_dim,
        bool(rotary_emb.mrope_interleaved), bool(rotary_emb.is_neox_style),
    )
    ref_rot = k_ref.view(BLOCK_SIZE, num_heads, head_size)[..., :rotary_dim]
    # T1 already pinned the bytes bit-for-bit against the engine writer; this
    # checks the dequantised *values* sit inside one E2M1 step of the port,
    # which is the design's actual acceptance form.
    sf_f = sf.view(torch.float8_e4m3fn).float()
    max_step = float(torch.where(
        sf_f > 0, 2.0 / sf_f, torch.zeros_like(sf_f)
    ).max())
    delta = (deq.float() - ref_rot[..., :rotary_dim].float()).abs()
    over = int((delta > max_step).sum())
    check("dequantised slot bake within one E2M1 step of the triton port",
          over == 0,
          f"max|delta|={float(delta.max()):.3e}, max_step={max_step:.3e}, "
          f"{over} element(s) over the step")

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
