"""Build a model profile card (design doc §3.1): every value either derived
from the checkpoint itself or explicitly marked as debt (`human-copied`).
Zero boot, zero GPU.

Usage:
  python tools/profile_card/build_card.py <model_dir> [--draft <draft_dir>]
      [--name <card_name>] [--kv-bytes N] [--max-model-len L] [--num-spec N]
      [--group-size G] [--mbt N] [--ssm-dtype DT] [--out-dir DIR]

Output: profile/<name>.json  (default out-dir: <repo>/profile)

The launcher generator consumes this card; a value it cannot find as
derived/measured is a missing card field, never a reason to hand-write a
number into a launcher (iron rule 34).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # tools/profile_card -> repo root

sys.path.insert(0, HERE)
from geometry import derive  # noqa: E402
from weight_census import census  # noqa: E402

# Step 096: generation reserve of the §7.1 fixed-slot budget ledger (design
# constant of the layout, not a model measurement) and the snapshot ring's
# trajectory count (step-066 per-trajectory filing). The KVMem workspace
# config resolves its numeric defaults through the card section built here
# (vllm/v1/kvmem_workspace/config.py, three-level chain: env > card > file
# fallback); every `derived` row reproduces the GSQ filing bit-for-bit.
KVMEM_GEN_RESERVE_TOKENS = 32768
KVMEM_SNAPSHOT_TRAJECTORIES = 2


def kvmem_workspace_section(text_cfg: dict, geo: dict) -> dict:
    """Derive the KVMem workspace defaults for one model (step 096).

    Ledger rows: `value` + `formula` + `source`. `derived` rows reproduce
    the GSQ filings (steps 060/066/072); `human-copied` rows are recorded
    debt the config chain resolves through anyway (values unchanged).
    VIEWPORT_PAGES needs the workspace page size, which is a
    sliding-window-spec property only known at engine assembly, so its row
    carries the runtime formula and `value: None` -- the manager passes the
    engine-resolved block size to config.viewport_retrieval_pages().
    """
    l_ceiling = geo["need_decomposition"].get("single_request_l_ceiling")
    num_layers = text_cfg.get("num_hidden_layers") or 0
    interval = text_cfg.get("full_attention_interval")
    linear_layers = (
        num_layers - num_layers // interval if interval else 0)
    mamba_state_bytes = (
        geo["page_geometry"]["mamba_page_raw_bytes"] * linear_layers)
    platform_path = os.path.join(HERE, "platform_card.json")
    host_gib = None
    if os.path.exists(platform_path):
        with open(platform_path, encoding="utf-8") as fh:
            platform = json.load(fh)
        host_row = platform.get("machine", {}).get("host_ram_total_gib")
        host_gib = (host_row.get("value")
                    if isinstance(host_row, dict) else host_row)
    section: dict = {}

    mpe = text_cfg.get("max_position_embeddings")
    section["workspace_tokens"] = (
        {"value": mpe,
         "formula": "max_position_embeddings (design §7.2 capability "
                    "ceiling; a host-budget quantity, not a code limit)",
         "source": "derived"} if mpe else
        {"value": None, "source": "missing (no max_position_embeddings)"})
    section["workspace_mb"] = (
        {"value": math.ceil(host_gib / 8) * 1024,
         "formula": "ceil(host_ram_total_gib / 8) GiB -- pinned host "
                    "budget policy from the platform card",
         "source": "derived"} if host_gib else
        {"value": None, "source": "missing (platform card host_ram_total_gib)"})
    if section["workspace_mb"]["value"] and mamba_state_bytes:
        keep = int(section["workspace_mb"]["value"] * 2**20
                   // (KVMEM_SNAPSHOT_TRAJECTORIES * mamba_state_bytes))
        section["snapshot_keep"] = {
            "value": keep,
            "formula": (f"floor(workspace_mb * 2^20 / (snapshot_traj="
                        f"{KVMEM_SNAPSHOT_TRAJECTORIES} * mamba group state "
                        f"{mamba_state_bytes} B)) -- step-066 ring ledger"),
            "source": "derived"}
    else:
        section["snapshot_keep"] = {
            "value": None,
            "source": "missing (workspace_mb or mamba state bytes)"}
    if l_ceiling:
        recent = 2 ** (int(math.floor(
            math.log2(l_ceiling - KVMEM_GEN_RESERVE_TOKENS))) - 2)
        section["viewport_recent_tokens"] = {
            "value": recent,
            "formula": (f"2^(floor(log2(L - gen)) - 2) with L={l_ceiling}, "
                        f"gen={KVMEM_GEN_RESERVE_TOKENS} -- §7.1 clamp band "
                        "lower end"),
            "source": "derived"}
        section["recent_tokens"] = {
            "value": 2 * recent,
            "formula": ("2 x viewport_recent -- geometric midpoint of the "
                        "§7.1 clamp band [R, 4R]"),
            "source": "derived"}
        section["viewport_pages"] = {
            "value": None,
            "formula": (f"floor((L - gen)/p) - 1 - ceil(R/p) - floor(gen/p) "
                        f"with L={l_ceiling}, gen={KVMEM_GEN_RESERVE_TOKENS} "
                        "-- §7.1 budget cut in pages; p = the engine-resolved "
                        "workspace block size, passed at runtime"),
            "source": "derived-runtime"}
    else:
        section["viewport_recent_tokens"] = {
            "value": None,
            "source": "missing (no single_request_l_ceiling: pool size "
                      "unknown without --kv-bytes)"}
        section["recent_tokens"] = {
            "value": None,
            "source": "missing (no viewport_recent anchor)"}
        section["viewport_pages"] = {
            "value": None,
            "source": "missing (no single_request_l_ceiling)"}
    section["authority_trajectories"] = {
        "value": KVMEM_SNAPSHOT_TRAJECTORIES,
        "formula": ("= snapshot_trajectories -- both host regions are "
                    "provisioned for the same trajectory concurrency"),
        "source": "derived"}
    section["gen_reserve_tokens"] = {
        "value": KVMEM_GEN_RESERVE_TOKENS,
        "source": "human-copied (§7.1 layout design constant)"}
    section["query_span"] = {
        "value": 256,
        "source": "human-copied (step-061 2^8 filing)"}
    section["trajectory_prefix_tokens"] = {
        "value": 512,
        "source": "human-copied (structural constant)"}
    section["index_subblock"] = {
        "value": 128,
        "source": "human-copied (§5.3 R1-mitigation constant; 64 filed "
                  "alternative)"}
    return section


def build(model: str, draft: str | None, name: str | None,
          kv_bytes: int | None, max_model_len: int | None,
          num_spec: int, group_size: int, mbt: int,
          ssm_dtype: str | None, out_dir: str) -> str:
    cfg_path = os.path.join(model, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        hf = json.load(fh)
    text_cfg = hf.get("text_config", hf)
    vision_wrapped = "vision_config" in hf
    quant = hf.get("quantization_config", {})
    quant_method = quant.get("quant_method", "none")

    geo = derive(model=model, draft=draft, num_spec=num_spec,
                 kv_dtype="nvfp4", mamba_ssm_dtype=ssm_dtype,
                 group_size=group_size, mbt=mbt,
                 max_model_len=max_model_len, kv_cache_bytes=kv_bytes,
                 enable_prefix_caching=True)

    card = {
        "schema": "profile-card/1",
        "name": name or os.path.basename(os.path.normpath(model)),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "tool": "tools/profile_card/build_card.py",
        "model": {
            "path": model,
            "architecture": hf.get("architectures", [None])[0],
            "model_type": hf.get("model_type"),
            "hidden_size": text_cfg.get("hidden_size"),
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "vocab_size": text_cfg.get("vocab_size"),
            "head_dim": text_cfg.get("head_dim"),
            "num_key_value_heads": text_cfg.get("num_key_value_heads"),
            "max_position_embeddings": text_cfg.get("max_position_embeddings"),
            "source": "derived",
        },
        "source_flags": {
            "vision_wrapped": vision_wrapped,
            "language_model_only_required": vision_wrapped,
            "_why": ("--language-model-only is a no-op on pure causal-LM "
                     "checkpoints; required when a vision wrapper exists"),
            "quantization_method": quant_method,
            "source": "derived",
        },
        "layer_distribution": {
            "num_layers": text_cfg.get("num_hidden_layers"),
            "full_attention_interval": text_cfg.get("full_attention_interval"),
            "full_attention_layers_count": None,
            "linear_layers_count": None,
            "source": "derived",
        },
        "page_geometry": {**geo["page_geometry"], "source": "derived"},
        "groups": {**geo["groups"], "source": "derived"},
        "capacity": {**geo["capacity"], "source": "derived"},
        "need_decomposition": {**geo["need_decomposition"],
                               "source": "derived"},
        "kvmem_workspace": kvmem_workspace_section(text_cfg, geo),
        "draft": geo["draft"],
        "spec_family": geo["spec_family"],
        "weight_census": census(model),
        "unused_modules": {},   # filled from census engine-drop verdicts
        "candidates": {
            "mbt_min": geo["need_decomposition"]["mbt_min"],
            "single_request_l_ceiling":
                geo["need_decomposition"]["single_request_l_ceiling"],
            "kv_cache_bytes": ("autoprobe decides from the residency-cliff "
                               "search (design doc §4); auto sizing is a "
                               "KNOWN speed trap on this card (step 090)"),
            "kv_offloading_backend": {"value": "native", "size_gib": 8,
                                      "source": "human-copied"},
            "source": "derived+human-copied",
        },
        "platform_ref": "tools/profile_card/platform_card.json",
        "provenance": {
            "rail_a_verified_anchors": [
                "GSQ prod: capacity 163,719 @ L=163,072 pool 3.4e9 (steps 036/046)",
                "Orca c2: capacity 157,910 @ L=131,072 pool 3.4e9 (step 089)",
                "088-arm reject readings: residual 1.0% / 3.8% < 5% (accept §6.2)",
            ],
            "mtp_visual_verdict": ("mtp.* and visual.* never reach VRAM "
                                   "(step 092); no skip patch required"),
        },
    }

    # layer split straight from the config
    gd = card["layer_distribution"]
    lc = text_cfg.get("num_hidden_layers") or 0
    interval = text_cfg.get("full_attention_interval")
    if interval:
        gd["full_attention_layers_count"] = lc // interval
        gd["linear_layers_count"] = lc - lc // interval
    card["unused_modules"] = {
        bucket: {"gib": row["gib"], "bytes": row["bytes"],
                 "verdict": row["engine_drop"]}
        for bucket, row in card["weight_census"]["buckets"].items()
        if "engine_drop" in row
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{card['name']}.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(card, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--kv-bytes", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--num-spec", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--mbt", type=int, default=1024)
    ap.add_argument("--ssm-dtype", default="bfloat16")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "profile"))
    args = ap.parse_args()
    path = build(model=args.model, draft=args.draft, name=args.name,
                 kv_bytes=args.kv_bytes, max_model_len=args.max_model_len,
                 num_spec=args.num_spec, group_size=args.group_size,
                 mbt=args.mbt, ssm_dtype=args.ssm_dtype, out_dir=args.out_dir)
    print(f"[build_card] wrote {path}")


if __name__ == "__main__":
    main()
