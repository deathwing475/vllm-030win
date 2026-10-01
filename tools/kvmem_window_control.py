# kvmem_window_control.py — is the compressed window itself broken? (step 073)
#
# Step 072's N=55 window request (96,128 tokens) prefilled, scored and baked
# fine, then the FIRST sampled token ended the request (finish_reason=stop,
# zero chunks). Three very different faults produce that same client-visible
# result:
#
#   A  the model genuinely samples EOS from the window's logits
#   B  the scheduler ends the request before/without a real sample
#   C  a token IS sampled but the streaming/stop handling drops it
#
# This tool separates A from B/C without touching the viewport machinery at
# all: it sends the window's token sequence as an ORDINARY prompt (token ids,
# no rewriting, no scoring, no bake, and a prompt short enough that the
# viewport guard declines by construction). If the native path also answers
# with an immediate EOS, the window's content/length is the problem, not the
# connector. If it answers normally, the machinery is the suspect.
#
# Modes (all built from the same haystack the step 072 probe builds):
#   window  ids[:head] + ids[-recent:]        == what the viewport prefills
#   tail    ids[-recent:]                     == the recent section alone
#   head    ids[:head]                        == sink + placeholder section
#   full    ids                               == the original prompt
#
# Judgement aids: the first sampled token id, whether it is in the model's EOS
# set, the top-k logprobs of that step, and the same request with ignore_eos
# (which forces generation past EOS: text appears => A, still empty => B/C).
#
# Usage (against a running arm, no ingest needed):
#   python tools\kvmem_window_control.py run --mode window --depth 0.30
#   python tools\kvmem_window_control.py run --mode window --depth 0.30 --ignore-eos
#   python tools\kvmem_window_control.py run --mode tail --depth 0.30

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_k3_probe import build, tokenizer  # noqa: E402
from kvmem_ws_probe import NEEDLE_CODE  # noqa: E402


def ids_for(mode: str, ids: list[int], head: int, recent: int) -> list[int]:
    if mode == "window":
        return ids[:head] + ids[len(ids) - recent:]
    if mode == "tail":
        return ids[len(ids) - recent:]
    if mode == "head":
        return ids[:head]
    if mode == "full":
        return list(ids)
    raise SystemExit(f"unknown mode {mode}")


def send(args, ids: list[int]) -> dict:
    payload = {
        "model": args.model,
        "prompt": ids,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "logprobs": args.logprobs,
        "return_token_ids": True,
        "stream": False,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    req = urllib.request.Request(
        args.base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as handle:
            body = json.loads(handle.read().decode())
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "error": exc.read().decode(errors="replace")[:600],
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the round
        return {"ok": False, "error": repr(exc)}
    choice = (body.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    first_step = (lp.get("top_logprobs") or [None])[0]
    return {
        "ok": True,
        "total_s": round(time.time() - t0, 3),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "token_ids": choice.get("token_ids"),
        "text": choice.get("text"),
        "first_token_logprob": (lp.get("token_logprobs") or [None])[0],
        "top_logprobs_first_step": first_step,
    }


def run(args) -> dict:
    tok = tokenizer()
    eos_ids = set()
    for attr in ("eos_token_id",):
        value = getattr(tok, attr, None)
        if isinstance(value, (list, tuple)):
            eos_ids.update(int(v) for v in value)
        elif value is not None:
            eos_ids.add(int(value))
    prompt, needle_char = build(args.tokens, args.depth, args.nonce, True,
                                args.question_style)
    ids = tok.encode(prompt, add_special_tokens=False)
    sent = ids_for(args.mode, ids, args.head_tokens, args.recent_tokens)
    needle_token = len(tok.encode(prompt[:needle_char], add_special_tokens=False))
    digest = hashlib.sha256(
        json.dumps(sent).encode()
    ).hexdigest()[:16]

    print(f"[prompt ] native mode={args.mode} sent={len(sent)} token(s) "
          f"sha256={digest} needle_token={needle_token} "
          f"in-sent={needle_token < len(sent)} eos_set={sorted(eos_ids)} "
          f"ignore_eos={args.ignore_eos} max_tokens={args.max_tokens}")
    out = {"mode": args.mode, "sent_tokens": len(sent), "sha256_16": digest,
           "needle_token": needle_token, "ignore_eos": args.ignore_eos,
           "eos_ids": sorted(eos_ids), "response": send(args, sent)}
    r = out["response"]
    if not r.get("ok"):
        print(json.dumps(r, indent=2))
        print("CONTROL VERDICT: REQUEST FAILED")
        return out
    first = (r.get("token_ids") or [None])[0]
    out["first_token_is_eos"] = first in eos_ids if first is not None else None
    print(json.dumps({k: r.get(k) for k in
                      ("total_s", "finish_reason", "prompt_tokens",
                       "completion_tokens", "token_ids", "first_token_logprob")},
                     indent=2))
    print("text[:120]:", repr((r.get("text") or "")[:120]))
    if r.get("top_logprobs_first_step"):
        print("top_logprobs[first]:", json.dumps(
            r["top_logprobs_first_step"], ensure_ascii=False)[:800])
    needle_hit = NEEDLE_CODE in (r.get("text") or "")
    verdict = (
        "EMPTY-AND-STOP"
        if r["completion_tokens"] in (0, 1) and first in eos_ids
        else "GENERATED"
    )
    print(f"first token {first} is_eos={out['first_token_is_eos']} "
          f"needle_hit={needle_hit}")
    print("CONTROL VERDICT:", verdict)
    out["needle_hit"] = needle_hit
    out["verdict"] = verdict
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="qwen3.8-27b-gsq")
    p.add_argument("--mode", default="window",
                   choices=("window", "tail", "head", "full"))
    p.add_argument("--tokens", type=int, default=200000)
    p.add_argument("--depth", type=float, default=0.30)
    p.add_argument("--head-tokens", type=int, default=79744,
                   help="S+N of the window (sink 1424 + 55 pages)")
    p.add_argument("--recent-tokens", type=int, default=16384)
    p.add_argument("--nonce", default="wc")
    p.add_argument("--question-style", default="summary",
                   choices=("summary", "focused"))
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--logprobs", type=int, default=5)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--out", default="")
    p.set_defaults(func=run)
    args = parser.parse_args()
    result = args.func(args)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
