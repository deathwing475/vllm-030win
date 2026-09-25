# anchor_longctx.py — 长上下文/offload 锚点采集器（A5）
#
# 口径（迁移计划 v2 §1 + 锚点采集协议 §2）：
#   - 对已启动的 serve（任意 launcher）打真实请求，逐档测：容量(max_model_len)、
#     TTFT（流式首 token，客户端口径含服务端 tokenize）、稳态 tok/s、needle 命中
#   - haystack = doc/code/reason 三类单元轮转拼接（~4070 tok/单元，降病态重复，
#     草稿接受率更贴近自然文本），85% 深度插入 needle（秘钥 77349），末尾提问+长答
#   - 稳态按 usage.completion_tokens 计（vLLM 流式每引擎步一个 delta、可含多 token，
#     chunk 数 ≠ token 数——步骤 006 实测教训）；同时落 n_chunks/steady_window 供审计
#   - 温度 0、stream、max_tokens=256
#
# 用法：
#   python tools\anchor_longctx.py --base http://127.0.0.1:8080 \
#     --lengths 8000 32000 --out <data>.json --arm <name>

import argparse, json, os, re, time
import urllib.request

UNIT_JSONS = [
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\doc_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\code_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\reason_4k.json",
]
NEEDLE = "\n\n[IMPORTANT FACT] The secret access code is 77349. Remember this.\n\n"
QUESTION = ("\n\nQuestion: What is the secret access code mentioned earlier "
            "in this document? Answer with the number only, then write a "
            "detailed summary of the document above (at least 200 words).\nAnswer:")
NEEDLE_TARGET = "77349"
UNIT_TOKENS = 4070  # doc_4k 实测（步骤 005），仅估算重复数，实际以 usage 为准


def http_json(url, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_text(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def build_prompt(target_tokens):
    units = [json.load(open(u, encoding="utf-8"))["messages"][0]["content"]
             for u in UNIT_JSONS]
    reps = max(1, round(target_tokens / UNIT_TOKENS))
    needle_at = max(1, int(reps * 0.85))
    parts = []
    for i in range(reps):
        parts.append(units[i % len(units)])
        if i == needle_at - 1:
            parts.append(NEEDLE)
    parts.append(QUESTION)
    return "".join(parts)


def run_one(base, target):
    prompt = build_prompt(target)
    payload = {
        "model": "qwen3.8-27b-gsq",
        "prompt": prompt,
        "max_tokens": 256, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    first_dt = None
    n_chunks = 0
    text = []
    usage = None
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            d = json.loads(body)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = ch.get("text") or ""
                if piece:
                    if first_dt is None:
                        first_dt = time.time() - t0
                    n_chunks += 1
                    text.append(piece)
    total_dt = time.time() - t0
    out_text = "".join(text)
    steady_s = max(1e-6, total_dt - (first_dt or 0))
    n_tok = (usage or {}).get("completion_tokens") or n_chunks
    return {
        "target_tokens": target,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": n_tok,
        "ttft_s": round(first_dt or -1, 3),
        "steady_tok_s": round(max(0, n_tok - 1) / steady_s, 2),
        "n_chunks": n_chunks,
        "steady_window_s": round(steady_s, 3),
        "total_dt_s": round(total_dt, 3),
        "needle_hit": NEEDLE_TARGET in out_text,
        "sample": out_text[:120],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    models = http_json(args.base + "/v1/models")
    max_len = models["data"][0]["max_model_len"]
    metrics_before = http_text(args.base + "/metrics")

    results = []
    # 热身：弃 1 条（runner v2 口径），防首请求 JIT/时钟爬坡污染
    for w in range(args.warmup):
        try:
            run_one(args.base, args.lengths[0])
            print(f"[warmup {w + 1}/{args.warmup}] discarded", flush=True)
        except Exception as e:
            print(f"[warmup {w + 1}] {type(e).__name__}: {e}", flush=True)

    for L in args.lengths:
        print(f"== {args.arm} target={L}", flush=True)
        reps = []
        for i in range(args.repeats):
            try:
                r = run_one(args.base, L)
            except Exception as e:  # 单档失败不弃全盘，照实记录
                r = {"target_tokens": L, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            reps.append(r)

        def med(key):
            vals = sorted(x[key] for x in reps if key in x and isinstance(x[key], (int, float)))
            return vals[len(vals) // 2] if vals else None

        results.append({
            "target_tokens": L,
            "repeats": reps,
            "ttft_s": med("ttft_s"),
            "steady_tok_s": med("steady_tok_s"),
            "needle_hit": all(x.get("needle_hit") for x in reps),
            "prompt_tokens": med("prompt_tokens"),
        })

    metrics_after = http_text(args.base + "/metrics")
    keep = re.compile(r"accept|spec_|num_.*tokens|prefix|offload", re.I)

    def slim(m):
        return [ln for ln in m.splitlines() if keep.search(ln) and not ln.startswith("#")]

    doc = {
        "arm": args.arm,
        "base": args.base,
        "max_model_len": max_len,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "metrics_before": slim(metrics_before),
        "metrics_after": slim(metrics_after),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
