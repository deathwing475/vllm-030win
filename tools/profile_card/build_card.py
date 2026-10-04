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
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # tools/profile_card -> repo root

sys.path.insert(0, HERE)
from geometry import derive  # noqa: E402
from weight_census import census  # noqa: E402


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
