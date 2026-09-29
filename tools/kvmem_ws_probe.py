# kvmem_ws_probe.py — acceptance probe for the KVMem host KV workspace.
#
# One self-contained tool for the three things the workspace has to be able to
# show (docs/vllm-030win-调研-KVMem虚拟化KV工作区.md §12.4, steps 060/061):
#
#   cold       a cold prompt longer than the KV pool, with a needle inside the
#              window (must be found) and optionally outside it (must be missed,
#              which is the positive control that the window really is bounded).
#              The workspace must spill the pages the window drops, and the
#              engine log must show evicted == stored with dropped == 0.
#   grow       the same prompt twice, the second longer by a delta and sharing
#              the prefix. Both must land on the SAME trajectory key, i.e. the
#              workspace survives across requests (the design's cross-request
#              store). Compare the trajectory keys in the engine log.
#   admission  the K2 guards: prompt over the workspace ceiling, prompt plus
#              requested output over the ceiling, and a control that must pass.
#              A 400 is the expected answer for the first; the second is clamped
#              upstream by the engine, which is what keeps the workspace from
#              being overrun.
#
# The prompt is built from the spec-eval 4k units with a random nonce in the
# haystack, so no earlier request can match it in the prefix cache: a warm TTFT
# would hide the real prefill. (This is why tools/anchor_longctx.py is not used
# here — its warmup poisons the first TTFT.)
#
# Usage:
#   python tools\kvmem_ws_probe.py cold      --tokens 210000 --out r.json
#   python tools\kvmem_ws_probe.py cold      --tokens 210000 --needle-depth 0.10 --out ctrl.json
#   python tools\kvmem_ws_probe.py grow      --tokens 210000 --grow 20000 --out g.json
#   python tools\kvmem_ws_probe.py admission --out a.json
#
# The engine must be the KVMem arm (VLLM_KVMEM_WORKSPACE=1 + VLLM_KVMEM_SW_WINDOW)
# for the workspace assertions to mean anything.

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid

UNIT_JSONS = [
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\doc_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\code_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\reason_4k.json",
]
UNIT_TOKENS = 4070
NEEDLE = "\n\n[IMPORTANT FACT] The secret access code is 77349. Remember this.\n\n"
NEEDLE_CODE = "77349"
QUESTION = (
    "\n\nQuestion: What is the secret access code mentioned earlier "
    "in this document? Answer with the number only, then write a "
    "detailed summary of the document above (at least 200 words).\nAnswer:"
)


def build_prompt(target_tokens: int, depth: float, nonce: str) -> str:
    """Haystack of spec-eval units with the needle at `depth` and a nonce first.

    The nonce is the very first thing in the prompt, so it is also part of the
    trajectory key's prefix: two probes with different nonces are different
    trajectories, which is what `grow` relies on.
    """
    units = [
        json.load(open(u, encoding="utf-8"))["messages"][0]["content"]
        for u in UNIT_JSONS
    ]
    reps = max(1, round(target_tokens / UNIT_TOKENS))
    needle_at = max(1, int(reps * depth))
    parts = [f"[SESSION {nonce}]\n"]
    for i in range(reps):
        parts.append(units[i % len(units)])
        if i == needle_at - 1:
            parts.append(NEEDLE)
    parts.append(QUESTION)
    return "".join(parts)


def post(base: str, prompt: str, max_tokens: int, ignore_eos: bool = False) -> dict:
    payload = {
        "model": "qwen3.8-27b-gsq",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    chunks = 0
    text: list[str] = []
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
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
                for ch in obj.get("choices", []):
                    piece = ch.get("text") or ""
                    if piece and ttft is None:
                        ttft = time.time() - t0
                    if piece:
                        chunks += 1
                        text.append(piece)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"raw": raw[:400]}
        return {"status": exc.code, "ok": False, "error": detail}
    out = "".join(text)
    return {
        "status": 200,
        "ok": True,
        "ttft_s": None if ttft is None else round(ttft, 3),
        "total_s": round(time.time() - t0, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "n_chunks": chunks,
        "needle_hit": NEEDLE_CODE in out,
        "prefill_tok_s": (
            None
            if not usage or ttft is None
            else round(usage.get("prompt_tokens", 0) / ttft, 1)
        ),
        "sample": out[:200],
    }


def cmd_cold(args) -> dict:
    nonce = uuid.uuid4().hex
    prompt = build_prompt(args.tokens, args.needle_depth, nonce)
    res = post(args.base, prompt, args.max_tokens)
    res["nonce"] = nonce
    res["needle_depth"] = args.needle_depth
    return {"cold": res}


def cmd_grow(args) -> dict:
    # One nonce for both requests: same leading tokens => same trajectory key.
    nonce = "kvmemgrowfixed01"
    r1 = post(args.base, build_prompt(args.tokens, args.needle_depth, nonce), args.max_tokens)
    r2 = post(
        args.base,
        build_prompt(args.tokens + args.grow, args.needle_depth, nonce),
        args.max_tokens,
    )
    return {"grow_r1": r1, "grow_r2": r2, "nonce": nonce}


def cmd_admission(args) -> dict:
    out: dict = {}
    out["A_prompt_over_ceiling"] = post(
        args.base, build_prompt(270000, args.needle_depth, "kvmemadmprobea"), 32
    )
    out["B_prompt_plus_output_over_ceiling"] = post(
        args.base, build_prompt(200000, args.needle_depth, "kvmemadmprobeb"), 100000
    )
    out["C_control_inside"] = post(
        args.base, build_prompt(2000, args.needle_depth, "kvmemadmprobec"), 64
    )
    if args.with_ceiling_clamp:
        # Decisive clamp evidence: a ~260K prompt leaves ~2K of the ceiling for
        # output, so asking for 20K with ignore_eos must come back with ~2K and
        # total == max_model_len exactly. Doubles as a cold ~260K spill run.
        out["D_ceiling_clamp_cold260k"] = post(
            args.base, build_prompt(260000, args.needle_depth, "kvmemadmprobed"),
            20000,
            ignore_eos=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cold", "grow", "admission"])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--tokens", type=int, default=210000)
    ap.add_argument("--grow", type=int, default=20000)
    ap.add_argument("--needle-depth", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--with-ceiling-clamp", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {"cold": cmd_cold, "grow": cmd_grow, "admission": cmd_admission}[
        args.mode
    ](args)
    for name, res in results.items():
        print(name, json.dumps(res, ensure_ascii=False)[:400], flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("saved:", args.out, "| exists:", os.path.exists(args.out))


if __name__ == "__main__":
    main()
