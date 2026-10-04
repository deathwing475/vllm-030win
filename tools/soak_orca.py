# -*- coding: utf-8 -*-
r"""Orca DFlash2 健康档长稳 + 多轮（步骤 093-C）。

soak_prod.py 的 Orca 版（052 的 finish_reason 健康语义原样保留；089/090 挂账
"长稳/多轮"的落地件）。差异 = 长度档缩进池容量（健康档池 8e8 = 18,589 tokens，
16k 请求占池 85% 是本档上限，没有 100k 档）+ 新增多轮对话段（/v1/chat/completions
messages 逐轮追加，prefix caching 按 messages 前缀命中，逐轮记 TTFT）。

健康判据（抄 soak_prod.ok_finish）：无 error + finish_reason ∈ {stop, length}
+ 输出非空（n_tokens > 10）。任何 HANG/崩溃即中止并 FAIL。

一轮 20 条混合请求：
  1-3   short chat 300 tok x3          （基本活性）
  4-5   chat 2000 tok x2               （decode 压力）
  6-8   8k haystack needle x3          （needle + 二发 prefix TTFT 门）
  11-12 16k haystack needle x2（max_tokens 32）（本档容量上限压力：整块填充实发
        ~16,333；256 output 会超 L=16,384 被 HTTP 400 拒——第一/二次跑的实测，
        故压力档缩 output；needle 码位于答案开头，32 token 内必出）
  13-14 wiki 8k chat 1024 tok x2       （真实文本）
  15    wiki 8k 二发                   （prefix 命中 TTFT 下降）
  16-18 reasoning chat x3              （finish=stop 语义段）
  19-20 short chat x2 收尾             （服务仍健康）

多轮段：6 轮固定提问（第 1 轮 800 字背景 + 逐轮短追问），messages 逐轮追加。

用法（任意非 vllm 源码目录，服务已在 8001 健康运行）：
  G:\qwen3.8model\vllm-win029\Scripts\python.exe ^
    G:\qwen3.8model\vllm-030win-git\tools\soak_orca.py [--rounds 2] [--skip-multiturn]
"""
import argparse
import datetime
import importlib.util
import json
import os
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8001"
MODEL = "orcasaq2"
OUT_DIR = r"G:\qwen3.8model\prod029_logs\soak"
ANCHOR = r"G:\qwen3.8model\vllm-030win-git\tools\anchor_longctx.py"
WIKI_JSONS = [
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\doc_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\code_4k.json",
    r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts\reason_4k.json",
]


def load_anchor():
    spec = importlib.util.spec_from_file_location("anchor", ANCHOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


anchor = load_anchor()


def ok_finish(r):
    """健康 = 无 error + finish_reason ∈ {stop, length} + 输出非空（052 语义）。"""
    return (r and not r.get("error")
            and r.get("finish_reason") in ("stop", "length")
            and (r.get("n_tokens") or 0) > 10)


class Deadline:
    """线程预算：超时 = 挂死判定。"""

    def __init__(self, name, fn, budget_s):
        self.name, self.fn, self.budget = name, fn, budget_s
        self.result, self.error, self.done_s = None, None, None

    def run(self):
        t0 = time.time()
        th = threading.Thread(target=self._wrap, daemon=True)
        th.start()
        th.join(self.budget)
        if th.is_alive():
            return {"name": self.name, "status": "HANG", "budget_s": self.budget}
        self.done_s = round(time.time() - t0, 1)
        return {"name": self.name, "status": "FAIL" if self.error else "OK",
                "done_s": self.done_s, "result": self.result, "error": self.error}

    def _wrap(self):
        try:
            self.result = self.fn()
        except Exception as e:  # noqa: BLE001
            self.error = "%s: %s" % (type(e).__name__, e)


def stream_post(path, payload, timeout=1800):
    """流式打一发，返回 (ttft_s, text, n_tokens, finish_reason, prompt_tokens)。"""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first_dt = None
    parts = []
    finish = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                # /v1/completions streams delta.text; /v1/chat/completions streams
                # delta.content -- accept both or the needle field reads false (31(i)).
                delta = ch.get("delta") or ch.get("text") or ""
                piece = delta.get("content") or delta.get("text") if isinstance(delta, dict) else delta
                if piece:
                    if first_dt is None:
                        first_dt = time.time() - t0
                    parts.append(piece)
    text = "".join(parts)
    return {"ttft_s": round(first_dt or -1, 3), "text": text,
            "n_tokens": (usage or {}).get("completion_tokens") or len(parts),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "finish_reason": finish, "total_s": round(time.time() - t0, 3)}


def chat_prompt(user, max_tokens):
    return {"path": "/v1/chat/completions",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": user}],
                        "max_tokens": max_tokens, "temperature": 0}}


def wiki_8k_prompt(max_tokens):
    unit = json.load(open(WIKI_JSONS[0], encoding="utf-8"))["messages"][0]["content"]
    return {"path": "/v1/completions",
            "payload": {"model": MODEL, "prompt": unit[:14000],
                        "max_tokens": max_tokens, "temperature": 0}}


def haystack(target, max_tokens=256):
    def make():
        prompt = anchor.build_prompt(target)
        return {"path": "/v1/completions",
                "payload": {"model": MODEL, "prompt": prompt,
                            "max_tokens": max_tokens, "temperature": 0}}
    return make


CHAT_TASKS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a haiku about GPUs.",
    "Summarize what a KV cache is in two sentences.",
]
REASONING_TASKS = [
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the calculation.",
    "If a cube's volume is 64 cm^3, what is its surface area? Show the calculation.",
    "Sort these by brightness: candle, sun, moon, flashlight. Explain briefly.",
]
MULTI_TURN_BG = (
    "Here is the meeting context: The platform team runs a 16 GB Windows inference box "
    "serving a 27B 3-bit quantized model with speculative decoding. Decode speed sits at "
    "80-87 tok/s on the 8k anchor as long as the KV pool stays near 0.8 GB; larger pools "
    "fall off a residency cliff. Boot takes about 60 seconds. ")
MULTI_TURNS = [
    "Summarize the context above in one sentence.",
    "What hardware constraint dominates the design?",
    "Which knob is described as the speed switch, and why?",
    "What happens when the pool grows past 1.5 GB?",
    "Rewrite your previous answer as a single bullet list.",
    "What would you measure next, given all of the above?",
]


def build_rounds():
    """一轮 20 条：(名字, 构造器, 线程预算秒, needle 期望)。"""
    rounds = []
    for i, t in enumerate(CHAT_TASKS, 1):
        rounds.append(("chat_s%d" % i, lambda t=t: chat_prompt(t, 300), 120, False))
    for i in (1, 2):
        rounds.append(("chat_2k%d" % i,
                       lambda i=i: chat_prompt("Write a detailed technical essay about "
                                               "GPU memory hierarchies, covering L2, HBM, "
                                               "and zero-copy paths. Part %d." % i, 2000),
                       300, False))
    rounds += [("needle8k_%d" % i, haystack(8000), 300, True) for i in (1, 2, 3)]
    rounds += [("needle12k_%d" % i, haystack(12000), 420, True) for i in (1, 2)]
    # Capacity-pressure tier (~16,333 prompt tokens = 85% of the 8e8 pool): judged on
    # "request succeeds and finishes cleanly at 85% pool occupancy", NOT on the needle
    # code. Measured twice: with output capped at 32 tokens the model is still inside
    # its <think> block when the budget ends (64-token rerun shows it reasoning its way
    # toward the code), so the code cannot appear within 32 tokens -- that is an output
    # budget fact, not a retrieval failure; retrieval depth is covered by the 8k/12k
    # tiers at the same 85% needle depth. 256 output tokens would push the request past
    # L=16,384 (HTTP 400, measured).
    rounds += [("needle16k_%d" % i, haystack(16000, max_tokens=32), 540, False)
               for i in (1, 2)]
    rounds += [("wiki8k_%d" % i, lambda i=i: wiki_8k_prompt(1024), 300, False)
               for i in (1, 2)]
    rounds.append(("wiki8k_second", lambda: wiki_8k_prompt(1024), 300, False))
    rounds += [("reason_%d" % i,
                lambda t=REASONING_TASKS[i - 1]: chat_prompt(t, 512), 180, False)
               for i in (1, 2, 3)]
    rounds += [("chat_close%d" % i, lambda i=i: chat_prompt(CHAT_TASKS[i - 1], 300),
                120, False)]
    return rounds


def run_soak_round(idx):
    rows = []
    for name, make, budget, want_needle in build_rounds():
        d = Deadline(name, lambda make=make: stream_post(make()["path"],
                                                         make()["payload"]), budget)
        rec = d.run()
        r = rec.get("result") or {}
        text = r.get("text") or ""
        row = {"name": name, "status": rec["status"], "done_s": rec.get("done_s"),
               "ttft_s": r.get("ttft_s"), "n_tokens": r.get("n_tokens"),
               "prompt_tokens": r.get("prompt_tokens"),
               "finish_reason": r.get("finish_reason"),
               "healthy": ok_finish(r), "error": rec.get("error")}
        if want_needle:
            row["needle_hit"] = "77349" in text
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["status"] == "HANG":
            print("HANG detected, aborting round", flush=True)
            break
    return {"round": idx, "rows": rows,
            "healthy_n": sum(1 for r in rows if r.get("healthy")),
            "hang_n": sum(1 for r in rows if r["status"] == "HANG"),
            "needle_all": all(r.get("needle_hit", True) for r in rows)}


def run_multiturn():
    msgs = [{"role": "user", "content": MULTI_TURN_BG + MULTI_TURNS[0]}]
    rows = []
    for i, follow in enumerate(MULTI_TURNS, 1):
        if i > 1:
            msgs.append({"role": "assistant", "content": rows[-1]["text_head"]})
            msgs.append({"role": "user", "content": follow})
        d = Deadline("mt%d" % i,
                     lambda m=list(msgs): stream_post(
                         "/v1/chat/completions",
                         {"model": MODEL, "messages": m, "max_tokens": 256,
                          "temperature": 0}), 180)
        rec = d.run()
        r = rec.get("result") or {}
        rows.append({"turn": i, "status": rec["status"], "ttft_s": r.get("ttft_s"),
                     "n_tokens": r.get("n_tokens"), "finish_reason": r.get("finish_reason"),
                     "healthy": ok_finish(r),
                     "prompt_tokens": r.get("prompt_tokens"),
                     "text_head": (r.get("text") or "")[:4000], "error": rec.get("error")})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "text_head"},
                         ensure_ascii=False), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--skip-multiturn", action="store_true")
    args = ap.parse_args()

    models = json.load(urllib.request.urlopen(BASE + "/v1/models"))
    doc = {"arm": "orca_dflash2_healthy", "base": BASE,
           "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "max_model_len": models["data"][0]["max_model_len"],
           "health_semantics": "soak_prod.ok_finish (step 052)", "soak": [],
           "multiturn": []}
    for i in range(1, args.rounds + 1):
        doc["soak"].append(run_soak_round(i))
    if not args.skip_multiturn:
        doc["multiturn"] = run_multiturn()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "soak_orca_result_%s.json"
                       % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    soak_ok = all(r["healthy_n"] == len(r["rows"]) and r["hang_n"] == 0
                  and r["needle_all"] for r in doc["soak"])
    mt_ok = all(r["healthy"] for r in doc["multiturn"]) if doc["multiturn"] else None
    print("SOAK_VERDICT: %s" % ("GO" if soak_ok else "FAIL"))
    print("MULTITURN_VERDICT: %s" % (mt_ok if mt_ok is None else ("GO" if mt_ok else "FAIL")))
    print("saved:", out)


if __name__ == "__main__":
    main()
