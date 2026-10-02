# kvmem_ingest_cadence.py — single long ingest as a page-step cadence probe.
#
# Step 077 found the spec'd workspace arm running ~11 s per 1456-token page
# step against ~1.4 s on the step 074 arm, but those two arms differ in more
# than speculation. This tool measures ONE request only (no flush, no serve):
# the ingest phase of the viewport probe prefills natively and the connector
# does its per-step incremental store, which is exactly the regime the 6x sits
# in. Cadence comes from two independent readings:
#   client side  TTFT / ceil(tokens / page)          (what this tool prints)
#   engine side  deltas between "KVMem capture drain" log lines (median)
# The engine-side reading is the one to trust for the distribution; the client
# side is enough to tell "fast" from "slow" and works on arms with no connector.
#
# Usage:
#   python tools\kvmem_ingest_cadence.py run --tokens 60000 --nonce cad078b \
#       --engine-log G:\qwen3.8model\prod029_logs\step078_boot1_xxx.out.log \
#       --out G:\qwen3.8model\prod029_logs\cad078b.json
#   python tools\kvmem_ingest_cadence.py cadence --engine-log <log>

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_ws_probe import post  # noqa: E402
from kvmem_k3_probe import build, tokenizer  # noqa: E402

DRAIN_RE = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[capture\.py:\d+\].*?"
    r"(\d+) layer\(s\), (\d+) token\(s\)")


def drain_timeline(path: str, min_tokens: int = 1000):
    """Median inter-drain delta for full-page steps (second resolution)."""
    stamps = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = DRAIN_RE.search(line)
            if match and int(match.group(3)) >= min_tokens:
                stamps.append(datetime.strptime("2026-" + match.group(1),
                                                "%Y-%m-%d %H:%M:%S").timestamp())
    if len(stamps) < 3:
        return {"drain_lines": len(stamps), "median_s": None,
                "p10_s": None, "p90_s": None, "max_s": None}
    deltas = sorted(stamps[i] - stamps[i - 1] for i in range(1, len(stamps)))
    # drop the tail: the last drain of a request is followed by the finish sweep
    return {
        "drain_lines": len(stamps),
        "median_s": deltas[len(deltas) // 2],
        "p10_s": deltas[max(0, int(0.10 * len(deltas)))],
        "p90_s": deltas[min(len(deltas) - 1, int(0.90 * len(deltas)))],
        "max_s": deltas[-1],
    }


def run(args) -> dict:
    prompt, _ = build(args.tokens, args.depth, args.nonce, False, "focused")
    tok = tokenizer()
    n = len(tok.encode(prompt, add_special_tokens=False))
    steps = math.ceil(n / args.page)
    print(f"[cadence] {n} tokens -> {steps} page steps at {args.page} "
          f"(nonce {args.nonce})")
    t0 = time.time()
    res = post(args.base, prompt, args.max_tokens, ignore_eos=True)
    out = {
        "nonce": args.nonce,
        "tokens": n,
        "page": args.page,
        "page_steps": steps,
        "ok": res.get("ok"),
        "ttft_s": res.get("ttft_s"),
        "total_s": res.get("total_s"),
        "prefill_tok_s": (n / res["ttft_s"]) if res.get("ttft_s") else None,
        "s_per_page_step": (res["ttft_s"] / steps) if res.get("ttft_s") else None,
        "client_wall_s": round(time.time() - t0, 2),
        "finish_reason": res.get("finish_reason"),
        "n_chunks": res.get("n_chunks"),
    }
    if args.engine_log and os.path.exists(args.engine_log):
        out["engine_drain"] = drain_timeline(args.engine_log, args.page)
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=2)
    return out


def cadence(args) -> dict:
    out = {"engine_log": args.engine_log,
           **drain_timeline(args.engine_log, args.min_tokens)}
    print(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--tokens", type=int, default=60000)
    p.add_argument("--depth", type=float, default=0.65,
                   help="unused with --no-needle; kept for shape parity")
    p.add_argument("--nonce", default="cad")
    p.add_argument("--page", type=int, default=1456)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=8)
    p.add_argument("--engine-log", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=run)

    c = sub.add_parser("cadence")
    c.add_argument("--engine-log", required=True)
    c.add_argument("--min-tokens", type=int, default=1000,
                   help="ignore partial-page drains below this token count")
    c.set_defaults(func=cadence)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
