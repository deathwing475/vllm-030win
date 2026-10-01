# kvmem_viewport_probe.py — fixed-slot compressed-window probe (step 072).
#
# Step 066 assembled stored pages back at their ORIGINAL positions, which
# proved the delivery mechanism but left the needle outside the attention
# window unanswerable. Step 072 is the design §5.1 re-bake route: a request
# whose prompt is longer than the window gets REWRITTEN onto the compressed
# window (prompt head through the placeholder section + the prompt's own tail),
# its prefill runs in engine-native window coordinates, and the scored pages
# are baked into the retrieval slots from the raw-K authority afterwards. The
# mid-section the window displaces is exactly what the retrieval slots must
# re-represent.
#
# The probe runs THREE requests to one server (same shape as the step 066
# probe, different judgement):
#   request 1 (ingest)  the full 200K prompt, needle at OUT-OF-WINDOW depth
#                       (the window covers the first ~79.8K and the last 16K
#                       tokens). The sliding window evicts pages into the
#                       workspace; the finish sweep stores the ~114 pages that
#                       were still resident when the request ended.
#   request 2 (flush)   a DIFFERENT 200K prompt that overwrites the pool
#                       blocks request 1 owned. Without it the serve request's
#                       rewritten sequence would native-hit the placeholder
#                       section and the viewport would decline (a hit beyond
#                       the sink section is a decline by design), leaving the
#                       native path to prefill everything.
#   request 3 (serve)   the SAME prompt as request 1. The connector rewrites
#                       it onto the window, the window prefill runs, scoring
#                       picks pages, the bake fills the slots, and decode
#                       answers the question that sits in the recent tail.
#
# Judgement:
#   mechanism  engine log must show "KVMem viewport: request ... rewritten",
#              then "KVMem retrieval (req=...)" with the ranking, then "baked
#              N page(s) into the retrieval slots".
#   recall     NEEDLE_CODE must appear in the serve output with VIEWPORT=1 and
#              must NOT appear in a VIEWPORT=0 boot of the same probe (the
#              needle sits outside the window, so a hit can only come from the
#              retrieval slots - not from the native prefix cache, which the
#              flush invalidated).
#
# Usage (per boot):
#   python tools\kvmem_viewport_probe.py run --tokens 200000 --depth 0.65 \
#       --tag vp1 --out G:\qwen3.8model\prod029_logs\kvmem_k7a\vp1.json

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_ws_probe import NEEDLE_CODE, post  # noqa: E402
from kvmem_k3_probe import build, tokenizer  # noqa: E402


def run(args) -> dict:
    base = args.base
    ingest_prompt, needle_char = build(args.tokens, args.depth, args.nonce,
                                       True, args.question_style)
    flush_prompt, _ = build(args.tokens, args.depth, args.nonce + "-flush",
                            False, args.question_style)
    tok = tokenizer()
    n_ingest = len(tok.encode(ingest_prompt, add_special_tokens=False))
    needle_token = tok.encode(ingest_prompt[:needle_char],
                              add_special_tokens=False)
    needle_depth_tokens = len(needle_token)
    # The arm's window (config.py defaults): first S+N = 79,776 tokens plus the
    # last R = 16,384 tokens are covered verbatim; the rest is what retrieval
    # must re-represent.
    in_window = needle_depth_tokens < args.head_tokens or (
        needle_depth_tokens >= n_ingest - args.recent_tokens
    )

    print(f"[ingest] ~{n_ingest} tokens, needle at token {needle_depth_tokens} "
          f"(depth {args.depth}, in-window={in_window})")
    if args.stages != "serve":
        r1 = post(base, ingest_prompt, args.ingest_max_tokens, ignore_eos=True)
        time.sleep(1.0)
        print(f"[flush ] ~{n_ingest} tokens (different nonce, overwrites the pool)")
        r2 = post(base, flush_prompt, 1, ignore_eos=True)
        time.sleep(1.0)
    else:
        # Re-send only the serve against a trajectory this boot already stored
        # (the ingest + flush pair costs ~8 minutes; a declined or crashed serve
        # attempt should not force the whole sequence again).
        print("[stages=serve] skipping ingest and flush, reusing the stored pages")
        r1 = r2 = {}
    print(f"[serve ] ~{n_ingest} tokens (same prompt; the window path runs)")
    r3 = post(base, ingest_prompt, args.max_tokens,
              ignore_eos=args.serve_ignore_eos)

    out = {
        "tag": args.tag,
        "tokens_ingest": n_ingest,
        "needle_depth": args.depth,
        "needle_token": needle_depth_tokens,
        "needle_in_window": in_window,
        "head_tokens": args.head_tokens,
        "recent_tokens": args.recent_tokens,
        "ingest": {
            "ttft_s": r1.get("ttft_s"),
            "total_s": r1.get("total_s"),
            "ok": r1.get("ok"),
        },
        "flush": {
            "ok": r2.get("ok"),
            "total_s": r2.get("total_s"),
        },
        "serve": {
            "ttft_s": r3.get("ttft_s"),
            "total_s": r3.get("total_s"),
            "ok": r3.get("ok"),
            "finish_reason": r3.get("finish_reason"),
            "n_chunks": r3.get("n_chunks"),
            "text": r3.get("text", ""),
            "sample": r3.get("sample", ""),
            "needle_hit": NEEDLE_CODE in (r3.get("text", "")
                                          or r3.get("sample", "")),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(json.dumps(out["serve"], indent=2))
    print("VIEWPORT VERDICT:",
          "HIT" if out["serve"]["needle_hit"] else "MISS")
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--tokens", type=int, default=200000)
    p.add_argument("--depth", type=float, default=0.65,
                   help="needle depth in the prompt; with the arm defaults "
                        "the window covers the first ~0.40 of a 200K prompt, "
                        "so anything past ~0.45 sits outside it")
    p.add_argument("--head-tokens", type=int, default=79776,
                   help="S+N of the arm's window (one sink page + 55 pages)")
    p.add_argument("--recent-tokens", type=int, default=16384)
    p.add_argument("--nonce", default="vp")
    p.add_argument("--question-style", default="summary",
                   choices=("summary", "focused"),
                   help="focused asks for the number only, so a working window "
                        "can answer inside a short budget instead of writing a "
                        "200-word summary first")
    p.add_argument("--ingest-max-tokens", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--tag", default="run")
    p.add_argument("--stages", default="all", choices=("all", "serve"),
                   help="serve = re-send only the window request against a "
                        "trajectory this boot already stored")
    p.add_argument("--serve-ignore-eos", action="store_true",
                   help="force the serve request to generate past EOS, which "
                        "separates 'the model sampled EOS' from 'the request "
                        "ended without a usable sample'")
    p.add_argument("--out", default=r"G:\qwen3.8model\prod029_logs\kvmem_k7a\vp.json")
    p.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
