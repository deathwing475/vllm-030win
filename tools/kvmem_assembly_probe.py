# kvmem_assembly_probe.py — prefix-assembly probe for the KVMem line (step 066).
#
# Step 065 proved "store raw, rebuild a page" at the byte level but no page was
# ever put back into a window, so a needle outside the sliding window was still
# unanswerable. Step 066 wires the connector-level half of the assembly: a later
# request of the same trajectory gets the stored pages copied into its first
# blocks (original positions, no re-RoPE) and num_computed_tokens jumped past
# them, so only the delta is prefilled; the mamba groups resume from a
# page-aligned snapshot the worker captured during the ingest prefill.
#
# The probe runs THREE requests to one server:
#   request 1 (ingest)  the full prompt; the sliding window evicts pages into
#                       the workspace and the worker snapshots boundaries.
#   request 2 (flush)   a DIFFERENT 200K prompt, sized to overwrite every pool
#                       block request 1 owned. This step exists because the
#                       engine's *native* prefix cache already recovers ~97% of
#                       a same-transcript resend while the blocks are intact
#                       (measured in the first boot of this step: the serve
#                       prefill collapsed to ~8 s without the connector doing
#                       anything). The flush invalidates those hashes, which is
#                       exactly the residual scenario where only the KVMem
#                       workspace still holds the prefix.
#   request 3 (serve)   the SAME prompt as request 1 plus a short tail asking
#                       the question. With VLLM_KVMEM_LOAD=1 the connector
#                       matches the stored prefix, loads it, and only prefills
#                       the tail; with LOAD=0 (the step 065 arm) it prefills
#                       everything again.
#
# Judgement (two boots, LOAD=1 vs LOAD=0, same prompts):
#   mechanism  serve TTFT with LOAD=1 must be far below LOAD=0 (the engine log
#              must show "KVMem assembly: request ... matches").
#   equality   greedy outputs of request 3 must be identical across the two
#              boots: the loaded pages are byte-identical to what a full
#              prefill computes (the K1 round-trip proved it) and the mamba
#              snapshot is the exact boundary state, so the resumed prefill
#              sees the same activations. Any difference means the assembly is
#              wrong, not noisy.
#   note       a needle outside the window is STILL unanswerable in this step:
#              pages go back at their original positions, and the question sits
#              at the far end of the window. Fixed-slot rematerialisation (the
#              design's compressed window, which does need re-RoPE) is the step
#              after. What this probe proves is the delivery mechanism.
#
# Usage (per boot):
#   python tools\kvmem_assembly_probe.py run --tokens 200000 --depth 0.50 \
#       --tag load1 --out asm_load1.json
#   python tools\kvmem_assembly_probe.py compare --a asm_load1.json --b asm_full1.json

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_ws_probe import NEEDLE_CODE, post  # noqa: E402
from kvmem_k3_probe import build, tokenizer  # noqa: E402

TAIL_MARK = "\n\n[UPDATE] The session is about to end. "
TAIL_QUESTION = (
    "Question: What is the secret access code? Answer with the number only.\nAnswer:"
)


def build_serve_prompt(ingest_prompt: str, tail_tokens: int):
    """The ingest prompt + a filler tail that ends with the question.

    The tail must exist: with zero new tokens the request is a pure prefix
    cache hit and the scheduler clamps the match to num_tokens-1, which is not
    page aligned. The filler is repeated harmless text sized by the tokenizer
    so the tail stays above one page but short enough to prefill fast.
    """
    tok = tokenizer()
    block = " Nothing else changes; please keep the details above in mind."
    filler = block
    while len(tok.encode(TAIL_MARK + filler, add_special_tokens=False)) < tail_tokens:
        filler += block
    tail = TAIL_MARK + filler + "\n" + TAIL_QUESTION
    return ingest_prompt + tail


def run(args) -> dict:
    base = args.base
    # The ingest prompt ends on a plain summary instruction; the question the
    # serve request answers sits in its tail.
    ingest_prompt, _ = build(args.tokens, args.depth, args.nonce, True, "summary")
    serve_prompt = build_serve_prompt(ingest_prompt, args.tail_tokens)
    flush_prompt, _ = build(args.tokens, args.depth, args.nonce + "-flush", True,
                            "summary")
    tok = tokenizer()
    n_ingest = len(tok.encode(ingest_prompt, add_special_tokens=False))
    n_serve = len(tok.encode(serve_prompt, add_special_tokens=False))

    print(f"[ingest] ~{n_ingest} tokens, needle at depth {args.depth}")
    r1 = post(base, ingest_prompt, args.ingest_max_tokens, ignore_eos=True)
    time.sleep(1.0)
    print(f"[flush ] ~{n_ingest} tokens (different nonce, overwrites the pool)")
    r2 = post(base, flush_prompt, 1, ignore_eos=True)
    time.sleep(1.0)
    print(f"[serve ] ~{n_serve} tokens (tail ~{n_serve - n_ingest})")
    r3 = post(base, serve_prompt, args.max_tokens)

    out = {
        "tag": args.tag,
        "tokens_ingest": n_ingest,
        "tokens_serve": n_serve,
        "tail_tokens": n_serve - n_ingest,
        "depth": args.depth,
        "ingest": {
            "ttft_s": r1.get("ttft_s"),
            "total_s": r1.get("total_s"),
            "ok": r1.get("ok"),
            "sample": r1.get("sample", "")[:200],
        },
        "flush": {
            "ok": r2.get("ok"),
            "total_s": r2.get("total_s"),
        },
        "serve": {
            "ttft_s": r3.get("ttft_s"),
            "total_s": r3.get("total_s"),
            "ok": r3.get("ok"),
            "sample": r3.get("sample", ""),
            "text": r3.get("text", ""),
            "needle_hit": NEEDLE_CODE in (r3.get("sample") or ""),
        },
    }
    # The engine only creates VLLM_KVMEM_DUMP on first write; on an arm whose
    # requests produced no retrieval dump the directory may not exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(json.dumps(out["serve"], indent=2))
    return out


def compare(args) -> dict:
    a = json.load(open(args.a, encoding="utf-8"))
    b = json.load(open(args.b, encoding="utf-8"))
    sa, sb = a["serve"], b["serve"]
    # Structural expectation for original-position assembly: the workspace only
    # ever holds the pages the sliding window evicted, i.e. the contiguous page
    # run below ``L - W``. That is the whole prefix this step can deliver, so
    # the token share it skips is (L - W) / L. The *time* saved is smaller than
    # that share: prefill cost per token grows with the attention window
    # (design §12.2), and the tokens this step skips are the earliest, cheapest
    # ones. So the token share is an upper bound on the saving, not an equality.
    boundary = (
        (a["tokens_serve"] - args.sliding_window) // args.block_size
    ) * args.block_size
    token_share = boundary / a["tokens_serve"]
    verdict = {
        "ttft_assembly": sa["ttft_s"],
        "ttft_full": sb["ttft_s"],
        "ttft_ratio": (
            (sa["ttft_s"] / sb["ttft_s"]) if sa["ttft_s"] and sb["ttft_s"] else None
        ),
        "boundary_tokens": boundary,
        "skipped_token_share": round(token_share, 4),
        "outputs_identical": sa.get("text", sa["sample"]) == sb.get("text", sb["sample"]),
        "output_chars": len(sa.get("text") or sa["sample"]),
        "needle_hit": {"a": sa["needle_hit"], "b": sb["needle_hit"]},
        "prompt_parity": (
            a["tokens_serve"] == b["tokens_serve"]
            and a["tokens_ingest"] == b["tokens_ingest"]
            and bool(sa["ok"]) and bool(sb["ok"])
        ),
    }
    ratio = verdict["ttft_ratio"]
    # A real gain must be visible, and it cannot exceed the token share (the
    # window-growth effect makes it strictly smaller in practice).
    gain_ok = ratio is not None and ratio <= args.max_ratio
    bounded_ok = ratio is not None and ratio >= 1.0 - token_share - args.tolerance
    verdict["ttft_gain_ok"] = gain_ok
    verdict["ttft_within_structural_bound"] = bounded_ok
    print(json.dumps(verdict, indent=2))
    ok = (
        verdict["prompt_parity"]
        and verdict["outputs_identical"]
        and gain_ok
        and bounded_ok
    )
    print("ASSEMBLY VERDICT:", "GO" if ok else "CHECK LOGS")
    return verdict


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--tokens", type=int, default=200000)
    p.add_argument("--depth", type=float, default=0.50)
    p.add_argument("--tail-tokens", type=int, default=768)
    p.add_argument("--nonce", default="asml")
    p.add_argument("--ingest-max-tokens", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default=r"G:\qwen3.8model\prod029_logs\kvmem_k4a\asm.json")
    p.set_defaults(func=run)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="LOAD=1 run output")
    c.add_argument("--b", required=True, help="LOAD=0 run output")
    c.add_argument("--sliding-window", type=int, default=163072)
    c.add_argument("--block-size", type=int, default=1424)
    c.add_argument(
        "--max-ratio",
        type=float,
        default=0.95,
        help="the assembled TTFT must be at most this fraction of the full one",
    )
    c.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="slack below the structural bound 1 - token_share",
    )
    c.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
