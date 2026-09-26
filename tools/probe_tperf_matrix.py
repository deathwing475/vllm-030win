# -*- coding: utf-8 -*-
r"""T-perf 三疑点分离探针（阶段 5 优化战役，2026-09-26）。

背景（步骤 018 步长解剖异常）：
  同一 0.29 栈上，anchor_longctx（completions/英文haystack/mt=256）引擎步长
  33/50/98ms（O(L)），stage0_ttft（chat/中文wiki/mt=1024）却 31/31/33ms 全平；
  0.27 上两工具一致（~31/66/133）。两侧数据均自洽（tok/delta≈2.1-2.6、tail≈0），
  差异真在引擎行为。三疑点 = 会话位置 / 请求路径(chat vs completions) / 内容。

本探针在单 serve 会话内逐一拨动三因子（一次一因子），每请求记录：
  - 逐 delta 时间戳 + 文本长度（chunk 粒度审计）
  - usage.completion_tokens（真 token 计，chunk≠token 教训）
  - /metrics spec_decode 差分（接受率三件套）
  - t_last_token vs t_done（尾部拖累显式测量）
判读：step p50/p90/mean + steady_all + tok/delta + tail。

用法（在任意非 vllm 源码目录运行）：
  G:\qwen3.8model\vllm-win029\Scripts\python.exe probe_tperf_matrix.py \
    --base http://127.0.0.1:8080 --out <result.json>
"""
import argparse
import importlib.util
import json
import os
import time
import urllib.request

ANCHOR_TOOL = r"G:\qwen3.8model\vllm-030win-git\tools\anchor_longctx.py"
STAGE0_PROMPT_DIR = (r"G:\qwen3.8model\nvfp4-win-experiment"
                     r"\_bench_results\stage0\stage0_prompts")
MODEL_DIR = r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ"
SPEC_METRICS = ["spec_decode_num_accepted_tokens_total",
                "spec_decode_num_draft_tokens_total"]


def load_anchor_tool():
    spec = importlib.util.spec_from_file_location("anchor_longctx", ANCHOR_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wiki_content(length):
    """stage0 口径内容：中文维基百科长文 + 尾部口令问题。"""
    p = os.path.join(STAGE0_PROMPT_DIR, "prompt_%dk_v1.json" % (length // 1000))
    with open(p, encoding="utf-8") as f:
        return json.load(f)["messages"][0]["content"]


def metrics_snapshot(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": type(e).__name__}
    out = {}
    for line in txt.splitlines():
        for k in SPEC_METRICS:
            if line.startswith("vllm:" + k) and " " in line:
                try:
                    out[k] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def run_request(base, arm, content, path, max_tokens):
    """发一条流式请求并记录逐 delta 时间线。path ∈ {chat, completions}。"""
    if path == "chat":
        payload = {"model": "qwen3.8-27b-gsq",
                   "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}}
        url = base + "/v1/chat/completions"
    else:
        payload = {"model": "qwen3.8-27b-gsq", "prompt": content,
                   "max_tokens": max_tokens, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}}
        url = base + "/v1/completions"

    m0 = metrics_snapshot(base)
    t0 = time.perf_counter()
    deltas = []  # (t, n_chars)
    usage = None
    finish_reason = None
    error = None
    t_last = None
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
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
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                if path == "chat":
                    delta = ch.get("delta") or {}
                    visible = (delta.get("content") or "") + \
                              (delta.get("reasoning") or "")
                else:
                    visible = ch.get("text") or ""
                if visible:
                    t = time.perf_counter() - t0
                    deltas.append((t, len(visible)))
                    t_last = t
    except Exception as e:
        error = "%s: %s" % (type(e).__name__, e)
    t_done = time.perf_counter() - t0
    m1 = metrics_snapshot(base)

    n_chunks = len(deltas)
    n_tok = (usage or {}).get("completion_tokens") or 0
    ttft = deltas[0][0] if deltas else None
    gaps = sorted((deltas[i][0] - deltas[i - 1][0]) * 1000.0
                  for i in range(1, n_chunks))
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0
    steady_all = ((n_tok - 1) / (t_last - ttft)
                  if n_tok > 1 and t_last and ttft and t_last > ttft else 0.0)
    # 稳态（跳过前 20 个 chunk，防首段爬坡）：时间窗只到最后一个 token
    if n_chunks > 22:
        t_s = deltas[20][0]
        n_after = n_tok * (n_chunks - 21) / n_chunks  # 按 chunk 占比折算 token
        steady_skip = (n_chunks - 21) / (t_last - t_s) * (n_tok / n_chunks) \
            if t_last > t_s else 0.0
    else:
        steady_skip = 0.0
    tail = (t_done - t_last) if t_last else None

    def diff(a, b):
        return (b - a) if isinstance(a, float) and isinstance(b, float) else None

    acc = diff(m0.get("spec_decode_num_accepted_tokens_total"),
               m1.get("spec_decode_num_accepted_tokens_total"))
    draft = diff(m0.get("spec_decode_num_draft_tokens_total"),
                 m1.get("spec_decode_num_draft_tokens_total"))

    rec = {
        "arm": arm, "path": path, "max_tokens": max_tokens,
        "content_head": content[:80], "content_len_chars": len(content),
        "ttft_s": round(ttft, 3) if ttft else None,
        "t_last_s": round(t_last, 3) if t_last else None,
        "t_done_s": round(t_done, 3),
        "tail_s": round(tail, 3) if tail is not None else None,
        "n_chunks": n_chunks, "n_tokens": n_tok,
        "tok_per_delta": round(n_tok / n_chunks, 3) if n_chunks else None,
        "step_p50_ms": round(pct(gaps, 0.5), 2),
        "step_p90_ms": round(pct(gaps, 0.9), 2),
        "step_mean_ms": round(gap_mean, 2),
        "steady_all_tok_s": round(steady_all, 2),
        "steady_skip20_tok_s": round(steady_skip, 2),
        "accept_rate": round(acc / draft, 4) if acc is not None and draft else None,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "finish_reason": finish_reason, "error": error,
        "delta_ts_s": [round(t, 4) for t, _ in deltas],
        "delta_chars": [n for _, n in deltas],
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="32000")
    ap.add_argument("--cooldown", type=float, default=8.0)
    args = ap.parse_args()

    anchor = load_anchor_tool()
    lengths = [int(x) for x in args.lengths.split(",")]

    # 臂定义：一次拨一个因子。base 双极 = A(anchor 复刻) / S(stage0 复刻)。
    # A: completions + haystack + 256；S: chat + wiki + 1024。
    # P: chat + haystack + 256（路径因子@anchor 内容）
    # C: completions + wiki + 256（路径因子@stage0 内容）
    # M: chat + wiki + 256（max_tokens 因子@stage0 形态）
    out = {"created": time.strftime("%F %T"), "base": args.base,
           "lengths": lengths, "records": []}

    for L in lengths:
        hay = anchor.build_prompt(L)
        wiki = wiki_content(L)
        arms = [("A", "completions", hay, 256),
                ("S", "chat", wiki, 1024),
                ("P", "chat", hay, 256),
                ("C", "completions", wiki, 256),
                ("M", "chat", wiki, 256),
                ("A_rep", "completions", hay, 256)]
        for name, path, content, mt in arms:
            t0 = time.time()
            rec = run_request(args.base, name, content, path, mt)
            rec["target_tokens"] = L
            rec["wall_s"] = round(time.time() - t0, 1)
            out["records"].append(rec)
            print("[L=%d %s/%s/mt=%d] p50=%.1fms mean=%.1fms tok/delta=%s "
                  "steady_all=%.1f tail=%.2fs acc=%s tok=%s/%s err=%s" % (
                      L, name, path, mt, rec["step_p50_ms"], rec["step_mean_ms"],
                      rec["tok_per_delta"], rec["steady_all_tok_s"],
                      rec["tail_s"] or -1, rec["accept_rate"],
                      rec["n_tokens"], rec["n_chunks"], rec["error"]),
                  flush=True)
            time.sleep(args.cooldown)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
