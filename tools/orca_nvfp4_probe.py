# orca_nvfp4_probe.py — chat/needle smoke probe for the Orca O1 arm (step 084).
#
# The O1 acceptance (docs/orcasaq2后续推进计划.md §4 O1) is boot + FlashInfer FA2
# NVFP4 + chat + identity/needle, performance recorded only. This probe covers
# the client side of that: a short identity chat and multi-depth needle
# retrieval over the /v1/completions API, with the served model name passed
# explicitly (the GSQ probes hardcode it).
#
# The needle haystack builder is REUSED from kvmem_ws_probe (same spec-eval
# filler units, same NEEDLE/QUESTION wording) so O1 needle results stay
# comparable with the GSQ line's needle protocol. No local tokenizer is needed:
# token counts come back in the server's usage block.
#
# Usage:
#   python tools\orca_nvfp4_probe.py chat   --base http://127.0.0.1:8001 --out r_chat.json
#   python tools\orca_nvfp4_probe.py needle --base http://127.0.0.1:8001 \
#       --tokens 14000 --depths 0.25,0.5,0.75 --out r_needle.json

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_ws_probe import NEEDLE, NEEDLE_CODE, QUESTION, build_prompt  # noqa: E402

IDENTITY_PROMPTS = [
    ("count", "What is 17 + 25? Answer with the number only.\nAnswer:"),
    ("repeat", "Repeat exactly, with no other text: The quick brown fox.\nAnswer:"),
    ("needle_short", NEEDLE.strip()
     + "\n\nQuestion: What is the secret access code? Answer with the number"
       " only.\nAnswer:"),
]


def post_completion(base: str, model: str, prompt: str, max_tokens: int,
                    timeout: int = 3600) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    chunks = 0
    finish_reason = None
    text: list[str] = []
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("text") or ""
                    if delta and ttft is None:
                        ttft = time.time() - t0
                    text.append(delta)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    chunks += 1
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "body": exc.read().decode(errors="replace")[:500]}
    joined = "".join(text)
    return {
        "text": joined,
        "hit": NEEDLE_CODE in joined,
        "finish_reason": finish_reason,
        "chunks": chunks,
        "ttft_s": None if ttft is None else round(ttft, 2),
        "latency_s": round(time.time() - t0, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def run_chat(args) -> dict:
    results = {}
    for name, prompt in IDENTITY_PROMPTS:
        results[name] = post_completion(args.base, args.model, prompt, args.max_tokens)
        print(f"[chat:{name}] hit={results[name].get('hit')} "
              f"finish={results[name].get('finish_reason')} "
              f"ttft={results[name].get('ttft_s')}s "
              f"text={results[name].get('text', '')[:80]!r}")
    return {"kind": "chat", "base": args.base, "results": results}


def run_needle(args) -> dict:
    depths = [float(d) for d in args.depths.split(",")]
    results = {}
    for depth in depths:
        nonce = f"orca-o1-{args.tag}-{int(depth * 100):03d}"
        prompt = build_prompt(args.tokens, depth, nonce)
        res = post_completion(args.base, args.model, prompt, args.max_tokens)
        results[f"d{depth:g}"] = res
        print(f"[needle d{depth:g}] hit={res.get('hit')} "
              f"prompt_tokens={res.get('prompt_tokens')} "
              f"ttft={res.get('ttft_s')}s latency={res.get('latency_s')}s "
              f"finish={res.get('finish_reason')}")
        time.sleep(1.0)
    hits = sum(1 for r in results.values() if r.get("hit"))
    return {"kind": "needle", "base": args.base, "tokens": args.tokens,
            "hits": hits, "n": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["chat", "needle"])
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="orcasaq2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=14000,
                        help="needle: target prompt size in tokens")
    parser.add_argument("--depths", default="0.25,0.5,0.75")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--tag", default="a")
    args = parser.parse_args()
    report = run_chat(args) if args.command == "chat" else run_needle(args)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
