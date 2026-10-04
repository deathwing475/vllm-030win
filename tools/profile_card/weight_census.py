"""Weight census straight from safetensors headers (no load, no GPU).
Evolved from _tmp_line_b/o91_weight_census.py (step 091) into the profile-card
building block: exclusive buckets + the engine-drop verdicts of step 092.

Step-092 verdicts (source: vllm source + boot-log reconciliation, see docs):
  * `mtp.*`  - dropped by the target's hf_to_vllm_mapper ("mtp.": None,
    qwen3_5.py:324/:481) BEFORE any module consumes it: never reaches VRAM
    (MTP-method drafts are the only consumer and load it themselves).
  * `visual.*` - with --language-model-only the tower is built on the meta
    device as StageMissingLayer (interfaces.py:341-381): silent drop, 0 bytes.
  => the generic fix "skip unused modules" is NOT needed; the engine already
     does it. The card records the fact so nobody re-investigates.
"""
from __future__ import annotations

import json
import os
import struct

BUCKETS = ("mtp.", "lm_head", "embed_tokens", "visual", "layers.")

ENGINE_DROPPED = {
    "mtp.": ("hf_to_vllm_mapper 'mtp.': None drops before consumption "
             "(step 092 verdict: never reaches VRAM; MTP-method draft is the "
             "only consumer)"),
    "visual": ("--language-model-only builds the tower on meta device "
               "(StageMissingLayer); step 092 verdict: never reaches VRAM"),
}


def bucket_of(key: str) -> str:
    for b in BUCKETS:
        if b in key:
            return b
    return "other"


def header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode())


def census(model_dir: str) -> dict:
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        return {"model_dir": model_dir, "error": "no safetensors"}
    tens: dict[str, int] = {}
    for name in files:
        for key, val in header(os.path.join(model_dir, name)).items():
            if key == "__metadata__" or not isinstance(val, dict):
                continue
            tens[key] = tens.get(key, 0) + (val["data_offsets"][1]
                                            - val["data_offsets"][0])
    gib = 1024 ** 3
    sums: dict[str, list] = {}
    for k, v in tens.items():
        slot = sums.setdefault(bucket_of(k), [0, 0])
        slot[0] += 1
        slot[1] += v
    buckets = {}
    for b, (n, tot) in sorted(sums.items(), key=lambda kv: -kv[1][1]):
        row = {"tensors": n, "bytes": tot, "gib": round(tot / gib, 3)}
        for prefix, verdict in ENGINE_DROPPED.items():
            if b.startswith(prefix.rstrip(".")):
                row["engine_drop"] = verdict
        buckets[b] = row
    return {
        "model_dir": model_dir,
        "files": len(files),
        "tensors": len(tens),
        "payload_gib": round(sum(tens.values()) / gib, 3),
        "buckets": buckets,
        "source": "derived (safetensors headers; drop verdicts = step 092)",
    }


if __name__ == "__main__":
    import sys
    for d in (sys.argv[1:] or [r"G:\qwen3.8model\qwen3.8exl3",
                               r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ"]):
        c = census(d)
        print(f"== {d}: {c.get('payload_gib')} GiB in {c.get('tensors')} tensors")
        for b, row in c.get("buckets", {}).items():
            flag = " [engine-dropped]" if "engine_drop" in row else ""
            print(f"   {b:<14} n={row['tensors']:<6} {row['gib']:.3f} GiB{flag}")
