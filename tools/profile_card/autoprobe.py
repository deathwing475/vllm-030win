"""Autoprobe (design doc §3.3/§4): boot the generated launcher once, read the
engine's OWN numbers out of the log (rail B), write them into the profile
card's `measured` section, and run the dual-track consistency guard (§5).

Rails:
  A (derived)  - profile card page geometry / capacity, from config alone
  B (measured) - engine banners: attention block size, GPU KV cache size,
                 maximum concurrency, model loading GiB
Guard: both rails must agree (block size exactly; capacity tokens exactly -
they are the same int(mc*L) quantity on both sides). On mismatch the guard
FAILS with a non-zero exit and the measured section is still written so the
human can diff - nobody is invited to "pick the plausible number".

Usage:
  python tools/profile_card/autoprobe.py --launch <generated.cmd> \
      --card <profile/<model>.json> [--port 8001] [--health-timeout 900]

The engine is killed at the end (file-version killer). Production state is
only ever recorded, never restored (iron rule 30).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

KILLER = r"G:\qwen3.8model\vllm-030win-git\tools\kill_vllm_orphans.ps1"
LOG_DIR = r"G:\qwen3.8model\_tmp_line_b"

BANNERS = {
    "model_loading_gib": r"Model loading took ([\d.]+) GiB memory",
    "attn_block_size": r"Setting attention block size to (\d+) tokens",
    "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "estimated_max_len": r"estimated maximum model length[:\s]+([\d,]+)",
}


def scrape(log_path: str) -> dict:
    out: dict[str, float | int] = {}
    rx = {k: re.compile(v) for k, v in BANNERS.items()}
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for key, r in rx.items():
                m = r.search(line)
                if m:
                    if key == "max_concurrency":
                        out["concurrency_at_len"] = int(m.group(1).replace(",", ""))
                        out["max_concurrency"] = float(m.group(2))
                    elif key == "kv_cache_tokens":
                        out[key] = int(m.group(1).replace(",", ""))
                    else:
                        out[key] = float(m.group(1).replace(",", "")) \
                            if "." in m.group(1) else int(m.group(1).replace(",", ""))
    return out


def wait_health(port: int, timeout_s: float, proc: subprocess.Popen) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # engine died before health
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_engine() -> None:
    subprocess.run(["powershell", "-NoProfile", "-File", KILLER],
                   capture_output=True, text=True)


def guard(card: dict, measured: dict) -> dict:
    a = card["page_geometry"]["block_size_tokens"]
    cap_a = card["capacity"]["capacity_tokens"]
    checks = []
    if "attn_block_size" in measured:
        checks.append({"name": "attention_block_size",
                       "rail_a": a, "rail_b": measured["attn_block_size"],
                       "ok": a == measured["attn_block_size"]})
    if "kv_cache_tokens" in measured and cap_a is not None:
        checks.append({"name": "capacity_tokens",
                       "rail_a": cap_a, "rail_b": measured["kv_cache_tokens"],
                       "ok": cap_a == measured["kv_cache_tokens"]})
    ok = bool(checks) and all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks,
            "note": "empty checks = engine printed no comparable banner"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--health-timeout", type=int, default=900)
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"autoprobe_{ts}.log")
    print(f"[autoprobe] boot log -> {log_path}")

    proc = subprocess.Popen(["cmd", "/c", args.launch],
                            stdout=open(log_path, "ab"),
                            stderr=subprocess.STDOUT,
                            cwd=r"G:\qwen3.8model")
    try:
        healthy = wait_health(args.port, args.health_timeout, proc)
        time.sleep(2)  # let the capacity banners land
        measured = scrape(log_path)
        with open(args.card, encoding="utf-8") as fh:
            card = json.load(fh)
        result = guard(card, measured)
        card.setdefault("measured", {})
        card["measured"][ts] = {
            "launch": args.launch, "port": args.port,
            "healthy": healthy, "log": log_path,
            "rail_b": measured, "guard": result,
            "source": "measured (autoprobe, engine banners)",
        }
        with open(args.card, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(card, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"[autoprobe] healthy={healthy}")
        print(f"[autoprobe] rail B = {json.dumps(measured)}")
        print(f"[autoprobe] guard = {json.dumps(result)}")
        sys.exit(0 if (healthy and result["ok"]) else 3)
    finally:
        if proc.poll() is None:
            proc.terminate()
        time.sleep(3)
        kill_engine()


if __name__ == "__main__":
    main()
