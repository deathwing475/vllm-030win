"""Rail A of the 092 dual-track consistency guard: derive the engine's KV page
geometry from config files alone (no GPU, no model instance, no boot), using the
engine's own functions as the single source of truth.

Verified anchors (2026-10-04, step 092):
  GSQ  prod (L=163,072, pool 3.4e9, G=8, bf16 ssm, gptq3c N=2)
       -> block 1456 / unified page 1,677,312 / bpb 13,418,496 / 253 blocks /
          capacity 163,719 tokens  (engine banner, steps 036/046)
  Orca c2  (L=131,072, same keys) -> capacity 157,910 / concurrency 1.2048
       (engine banner, step 089)

Non-obvious inputs the engine applies silently (each one bit us once):
  * the DFlash2 draft's 5 sliding-window layers keep their KV in the SAME pool
    (one more group; capacity denominator +admission blocks/request) --
    this is the `sw*cdiv(5,G)` term of the step-036 capacity formula;
  * the mamba spec must be rebuilt like MambaBase.get_kv_cache_spec does
    (num_speculative_blocks=2, page_size_padded, align mode) or the capacity
    denominator drops the draft-slot states and reads ~3.6% high;
  * prefix caching flips mamba_cache_mode none -> "align" (models/config.py);
  * VLLM_KV_GROUP_SIZE is read from os.environ inside get_kv_cache_groups,
    so it must be set before that call, not before importing vllm.
"""
from __future__ import annotations

import os
from typing import Any


def _maybe_register_exl3() -> str:
    """orcasaq2 registers the 'exl3' quant method; without it ModelConfig()
    rejects EXL3 checkpoints with 'Unknown quantization method'."""
    try:
        import orcasaq2  # noqa: F401
        orcasaq2.register()
        return "registered"
    except Exception as e:  # pragma: no cover - GSQ path has no orcasaq2
        return f"unavailable ({e})"


def derive(
    model: str,
    draft: str | None = None,
    num_spec: int = 2,
    kv_dtype: str = "nvfp4",
    mamba_ssm_dtype: str | None = None,   # None = engine default (HF override)
    mamba_cache_dtype: str | None = None, # conv-state dtype, None = engine default
    group_size: int | None = 8,           # None -> leave engine heuristic
    mbt: int = 1024,
    max_model_len: int | None = None,
    kv_cache_bytes: int | None = None,
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 1,
) -> dict[str, Any]:
    """Derive the full KV page geometry. Pure CPU; imports torch+vllm."""
    import torch
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                             SchedulerConfig, SpeculativeConfig, VllmConfig)
    from vllm.utils.math_utils import cdiv
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.kv_cache_interface import (FullAttentionSpec,
                                            get_kv_quant_mode)
    from vllm.v1.core import kv_cache_utils
    from vllm.model_executor.models import ModelRegistry

    reg_state = _maybe_register_exl3()

    model_config = ModelConfig(model=model, dtype="auto",
                               max_model_len=max_model_len or 32768)
    cache_config = CacheConfig(block_size=16, cache_dtype=kv_dtype,
                               enable_prefix_caching=enable_prefix_caching,
                               kv_cache_memory_bytes=kv_cache_bytes)
    if mamba_cache_dtype is not None:
        cache_config.mamba_cache_dtype = mamba_cache_dtype
    if mamba_ssm_dtype is not None:
        cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
    sched_config = SchedulerConfig(max_num_batched_tokens=mbt,
                                   max_num_seqs=max_num_seqs,
                                   max_model_len=model_config.max_model_len,
                                   is_encoder_decoder=False)
    paral_config = ParallelConfig(tensor_parallel_size=1)
    spec_config = None
    if draft is not None:
        # dflash carries max_num_new_slots_for_drafting=num_spec (MTP would be 0)
        spec_config = SpeculativeConfig(
            method="dflash", model=draft, num_speculative_tokens=num_spec,
            target_model_config=model_config,
            target_parallel_config=paral_config)
    vllm_config = VllmConfig(model_config=model_config,
                             cache_config=cache_config,
                             scheduler_config=sched_config,
                             parallel_config=paral_config,
                             speculative_config=spec_config)

    # mamba_cache_mode resolution mirror (models/config.py:609-657):
    # prefix caching + hybrid forces "align" and mamba_block_size = block_size.
    # ModelConfig/HF verify hooks run inside VllmConfig.__post_init__ for the
    # ssm dtype override; the mode mirror below matches the engine's outcome.
    if cache_config.mamba_cache_mode is None or cache_config.mamba_cache_mode == "none":
        if enable_prefix_caching and model_config.is_hybrid:
            cache_config.mamba_cache_mode = "align"
            cache_config.mamba_block_size = cache_config.block_size

    # -- 1) per-token attention page, nvfp4 packing via the backend
    attn_1tok = FlashInferBackend.customize_spec(FullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(paral_config),
        head_size=model_config.get_head_size(),
        dtype=torch.uint8,
        kv_quant_mode=get_kv_quant_mode(kv_dtype)))
    page1 = attn_1tok.page_size_bytes

    # -- 2) mamba raw page (platform fallback path, platforms/interface.py)
    arch_cls, _ = ModelRegistry.resolve_model_cls(model_config.architecture,
                                                  model_config=model_config)
    from vllm.v1.kv_cache_interface import MambaSpec
    if hasattr(arch_cls, "get_mamba_specs_from_config"):
        raw_specs = list(arch_cls.get_mamba_specs_from_config(vllm_config))
    else:
        shapes, dtypes = (arch_cls.get_mamba_state_shape_from_config(vllm_config),
                          arch_cls.get_mamba_state_dtype_from_config(vllm_config))
        raw_specs = [MambaSpec(shapes=tuple(tuple(s) for s in shapes),
                               dtypes=tuple(dtypes), block_size=-1)]
    mamba_page_raw = max(s.page_size_bytes for s in raw_specs)

    # -- 3) attention block size alignment (platforms/interface.py:911-958)
    kba = max(min(FlashInferBackend.get_supported_kernel_block_sizes()),
              cache_config.block_size)
    block_size = kba * cdiv(mamba_page_raw, kba * page1)
    attn_page = block_size * page1
    cache_config.block_size = block_size
    cache_config.mamba_block_size = block_size
    cache_config.mamba_page_size_padded = attn_page
    cache_config.kv_cache_layout = "LBHNC"  # nvfp4 head-major (engine resolve)

    attn_spec = FlashInferBackend.customize_spec(FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=model_config.get_num_kv_heads(paral_config),
        head_size=model_config.get_head_size(),
        dtype=torch.uint8,
        kv_quant_mode=get_kv_quant_mode(kv_dtype)))

    def rebuild_mamba(base: MambaSpec, tp_replicated: bool = False) -> MambaSpec:
        # mirror of MambaBase.get_kv_cache_spec (mamba/abstract.py:67-86)
        return MambaSpec(
            shapes=tuple(tuple(s) for s in base.shapes),
            dtypes=tuple(base.dtypes),
            block_size=cache_config.mamba_block_size,
            page_size_padded=cache_config.mamba_page_size_padded,
            mamba_type=base.mamba_type,
            tp_replicated=tp_replicated,
            mamba_cache_mode=cache_config.mamba_cache_mode,
            num_speculative_blocks=0 if cache_config.use_kda_recoverssm
            else vllm_config.num_speculative_tokens)

    gdn_spec = rebuild_mamba(raw_specs[0])
    extra_mamba = [rebuild_mamba(s) for s in raw_specs[1:]]

    # -- 4) assemble the per-layer spec dict
    hf = model_config.hf_text_config
    layer_types = getattr(hf, "layer_types", None)
    interval = getattr(hf, "full_attention_interval", None)
    full_ids: list[int] = []
    gdn_ids: list[int] = []
    if layer_types:
        for i, lt in enumerate(layer_types):
            (full_ids if "full" in str(lt) else gdn_ids).append(i)
    else:
        for i in range(hf.num_hidden_layers):
            (full_ids if interval and (i + 1) % interval == 0 else gdn_ids).append(i)

    kv_cache_spec: dict[str, Any] = {}
    for i in gdn_ids:
        kv_cache_spec[f"model.layers.{i}.linear_attn"] = gdn_spec
    for i in full_ids:
        kv_cache_spec[f"model.layers.{i}.self_attn"] = attn_spec
    for j, s in enumerate(extra_mamba):
        kv_cache_spec[f"model.layers.{gdn_ids[0]}.extra_mamba.{j}"] = s

    draft_info: dict[str, Any] | None = None
    spec_family: dict[str, Any] | None = None
    if draft is not None:
        draft_config = ModelConfig(model=draft, dtype="auto")
        dhf = draft_config.hf_config
        # spec-family usability, all values resolved by engine code paths:
        # fc width is the only runtime guard (qwen3_dflash.py:78-86), mask
        # rides on the shared target embed, aux taps must fit target depth.
        aux_taps = None
        fc_width = None
        try:
            from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
                get_eagle3_aux_layers_from_config)
            aux_taps = get_eagle3_aux_layers_from_config(spec_config)
        except Exception:
            pass
        if aux_taps:
            try:
                from vllm.model_executor.models.qwen3_dflash import (
                    _get_dflash_fc_input_size)
                fc_width = _get_dflash_fc_input_size(vllm_config)
            except Exception:
                target_hidden = (getattr(dhf, "target_hidden_size", None)
                                 or dhf.hidden_size)
                fc_width = target_hidden * len(aux_taps)
        mask_id = (getattr(dhf, "dflash_config", {}) or {}).get("mask_token_id")
        vocab = model_config.hf_text_config.vocab_size
        n_target_layers = model_config.hf_text_config.num_hidden_layers
        supports_eagle3 = hasattr(arch_cls, "set_aux_hidden_state_layers")
        spec_family = {
            "family": "dflash2",
            "aux_taps": list(aux_taps) if aux_taps else None,
            "fc_width": fc_width,
            "mask_token_id": mask_id,
            "mask_token_ok": (mask_id < vocab) if mask_id is not None else None,
            "aux_max_below_target_depth": (max(aux_taps) < n_target_layers
                                           if aux_taps else None),
            "supports_eagle3_interface": supports_eagle3,
            "v2_model_runner_required": True,  # V1 silently downgrades to DFlash1
            "draft_slots": (spec_config.max_num_new_slots_for_drafting
                            if spec_config else None),
        }
        d_layer_types = set(getattr(dhf, "layer_types", []) or [])
        sw = getattr(dhf, "sliding_window", None)
        if d_layer_types and any("sliding" in str(t) for t in d_layer_types) and sw:
            from vllm.v1.kv_cache_interface import SlidingWindowSpec
            draft_spec = FlashInferBackend.customize_spec(SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=dhf.num_key_value_heads,
                head_size=dhf.head_dim,
                sliding_window=sw,
                dtype=torch.uint8,
                kv_quant_mode=get_kv_quant_mode(kv_dtype)))
            for i in range(dhf.num_hidden_layers):
                kv_cache_spec[f"draft.layers.{i}.self_attn"] = draft_spec
            draft_info = {
                "path": draft,
                "architecture": getattr(dhf, "architectures", [None])[0],
                "num_hidden_layers": dhf.num_hidden_layers,
                "sliding_window": sw,
                "spec": "SlidingWindowSpec",
                "per_request_blocks": -(
                    -draft_spec.max_memory_usage_bytes(vllm_config)
                    // draft_spec.page_size_bytes),
            }
        else:
            draft_info = {"path": draft,
                          "architecture": getattr(dhf, "architectures", [None])[0],
                          "num_hidden_layers": dhf.num_hidden_layers,
                          "spec": "none (no sliding-window draft layers)"}

    # -- 5) groups -> bytes_per_block -> capacity
    old_g = os.environ.get("VLLM_KV_GROUP_SIZE")
    if group_size is not None:
        os.environ["VLLM_KV_GROUP_SIZE"] = str(group_size)
    try:
        groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_spec)
    finally:
        if old_g is None:
            os.environ.pop("VLLM_KV_GROUP_SIZE", None)
        else:
            os.environ["VLLM_KV_GROUP_SIZE"] = old_g
    bpb = kv_cache_utils._get_kv_cache_bytes_per_block(groups)
    group_rows = []
    blocks_per_req_by_group = []
    for g in groups:
        page = g.kv_cache_spec.page_size_bytes
        per_req = -(-g.kv_cache_spec.max_memory_usage_bytes(vllm_config) // page)
        group_rows.append({"layers": len(g.layer_names), "page_bytes": page,
                           "example": g.layer_names[0] if g.layer_names else None,
                           "blocks_per_request": per_req})
        blocks_per_req_by_group.append(per_req)

    num_blocks = None
    capacity_tokens = None
    max_concurrency = None
    if kv_cache_bytes is not None:
        num_blocks = kv_cache_bytes // bpb
        # capacity mirror (kv_cache_utils.py:2042-2052)
        denom = sum(blocks_per_req_by_group)
        max_concurrency = num_blocks / denom
        capacity_tokens = int(max_concurrency * model_config.max_model_len)

    # -- 6) need(L) = a + b*L decomposition (per-request pool bytes)
    n_attn_groups = sum(1 for g in group_rows
                        if "self_attn" in (g["example"] or "")
                        and not g["example"].startswith("draft."))
    n_mamba_groups = sum(1 for g in group_rows
                         if "linear_attn" in (g["example"] or ""))
    n_swa_groups = sum(1 for r in (draft_info,) if r and r.get("per_request_blocks"))
    # a = every group whose per-request cost does NOT scale with L
    # (mamba states + draft sliding window); b = the attention linear term.
    a_bytes = 0
    seen: set[str] = set()
    for g, row in zip(groups, group_rows):
        example = row["example"]
        if example is None or example in seen:
            continue
        if "self_attn" in example and not example.startswith("draft."):
            continue  # the linear term
        seen.add(example)
        a_bytes += row["blocks_per_request"] * bpb
    b_bytes_per_token = n_attn_groups * bpb / block_size

    # mbt lower bound: page length + num_spec (iron rule 16(10) / 28)
    mbt_min = block_size + (spec_config.max_num_new_slots_for_drafting
                            if spec_config else 0)
    # single-request L ceiling for the given pool (null block held back)
    l_ceiling = None
    if num_blocks is not None:
        usable = num_blocks - 1 - a_bytes // bpb
        l_ceiling = (usable // n_attn_groups) * block_size if n_attn_groups else None

    return {
        "registration": {"orcasaq2": reg_state},
        "model": {"path": model, "architecture": model_config.architecture,
                  "is_hybrid": model_config.is_hybrid,
                  "quantization": getattr(model_config, "quantization", None)},
        "inputs": {
            "kv_dtype": kv_dtype, "num_spec": num_spec,
            "mamba_ssm_dtype": cache_config.mamba_ssm_cache_dtype,
            "mamba_cache_dtype": cache_config.mamba_cache_dtype,
            "mamba_cache_mode": cache_config.mamba_cache_mode,
            "group_size": group_size, "mbt": mbt,
            "max_model_len": model_config.max_model_len,
            "kv_cache_bytes": kv_cache_bytes,
            "enable_prefix_caching": enable_prefix_caching,
        },
        "page_geometry": {
            "attn_page_bytes_per_token_per_layer": page1,
            "mamba_page_raw_bytes": mamba_page_raw,
            "block_size_tokens": block_size,
            "unified_page_bytes": attn_page,
        },
        "groups": {"count": len(groups), "rows": group_rows,
                   "bytes_per_block": bpb},
        "draft": draft_info,
        "spec_family": spec_family,
        "capacity": {
            "kv_cache_bytes": kv_cache_bytes,
            "num_blocks": num_blocks,
            "blocks_per_request_total": (sum(blocks_per_req_by_group)
                                         if num_blocks else None),
            "max_concurrency": max_concurrency,
            "capacity_tokens": capacity_tokens,
        },
        "need_decomposition": {
            "formula": "need(L) = a + b*L  (per-request pool bytes)",
            "a_bytes": a_bytes,
            "b_bytes_per_token": b_bytes_per_token,
            "mbt_min": mbt_min,
            "single_request_l_ceiling": l_ceiling,
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--num-spec", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--mbt", type=int, default=1024)
    ap.add_argument("--mamba-ssm-dtype", default="bfloat16",
                    help="engine default 'auto' picks up HF mamba_ssm_dtype")
    ap.add_argument("--max-model-len", type=int, default=163072)
    ap.add_argument("--kv-bytes", type=int, default=None)
    args = ap.parse_args()
    out = derive(model=args.model, draft=args.draft,
                 num_spec=args.num_spec, group_size=args.group_size,
                 mbt=args.mbt, mamba_ssm_dtype=args.mamba_ssm_dtype,
                 max_model_len=args.max_model_len,
                 kv_cache_bytes=args.kv_bytes)
    print(json.dumps(out, indent=2, ensure_ascii=False))
