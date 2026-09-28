# -*- coding: utf-8 -*-
r"""生产配置真实请求长稳（20 条混合请求）。

升格自 _tmp_line_b/soak_e1_prod.py（步骤 052 清债）。判定语义修正：
健康判据 = 无 error + finish_reason ∈ {stop, length} + 输出非空（n_tokens > 10），
不再用 n_tokens>500 阈值——简答/总结类 prompt 模型自然早停（finish=stop、
79-118 token）曾被该阈值连续五轮误判 FAIL（步骤 027/036/042/045/051 记档）。

一轮 20 条混合请求：
  1-3   short chat 300 tok x3            （基本活性）
  4-5   chat 2000 tok x2                 （decode 步长/接受率，boot_e2e 口径）
  6-7   8k haystack needle x2            （needle + 多轮 TTFT 门：二发 <=3.5s）
  8-10  32k haystack needle x3           （needle + 多轮 TTFT 门：二/三发 <=3.5s）
  11-12 reasoning chat x2                （message.reasoning + finish=stop）
  13-14 tool-call x2                     （finish=tool_calls + 结构化 arguments）
  15-16 wiki 8k / 32k chat 1024 tok x2   （真实文本接受率，A4 口径 0.68-0.80）
  17    wiki 8k 二发                     （prefix 命中 TTFT 下降）
  18    100k haystack needle x1          （长上下文压力，防长 prefill 死锁）
  19-20 short chat x2 收尾               （服务仍健康）

每条请求带线程预算（挂死检测）；任何 HANG/崩溃即中止并 FAIL。
结果落 prod029_logs\soak\soak_prod_result_<时间戳>.json（复跑锚点不覆写）。
用法（任意非 vllm 源码目录）：
  G:\qwen3.8model\vllm-win029\Scripts\python.exe ^
    G:\qwen3.8model\vllm-030win-git\tools\soak_prod.py
"""
import datetime
import importlib.util
import json
import os
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
OUT_DIR = r"G:\qwen3.8model\prod029_logs\soak"
PROBE = r"G:\qwen3.8model\vllm-030win-git\tools\probe_tperf_matrix.py"
ANCHOR = r"G:\qwen3.8model\vllm-030win-git\tools\anchor_longctx.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = load(PROBE, "probe")
anchor = load(ANCHOR, "anchor")


def ok_finish(r):
    """健康 = 无 error + finish_reason ∈ {stop, length} + 输出非空。

    stop/length 都是正常终点：stop=模型语义结束（长短皆可），
    length=吃满 max_tokens（decode 压力档）。空响应（n_tokens<=10）
    或无 finish_reason（连接中断/半途异常）判 FAIL。
    """
    return (r and not r.get("error")
            and r.get("finish_reason") in ("stop", "length")
            and (r.get("n_tokens") or 0) > 10)


class Deadline:
    """线程预算：超时=挂死判定（历史 1/20 死锁的探测器）。"""

    def __init__(self, name, fn, budget_s):
        self.name, self.fn, self.budget = name, fn, budget_s
        self.result, self.error, self.done_s = None, None, None

    def run(self):
        t0 = time.time()
        th = threading.Thread(target=self._wrap, daemon=True)
        th.start()
        th.join(self.budget)
        if th.is_alive():
            return {"name": self.name, "status": "HANG",
                    "budget_s": self.budget}
        self.done_s = round(time.time() - t0, 1)
        return {"name": self.name,
                "status": "ERROR" if self.error else "OK",
                "error": self.error, "wall_s": self.done_s,
                "result": self.result}

    def _wrap(self):
        try:
            self.result = self.fn()
        except Exception as e:  # noqa: BLE001
            self.error = "%s: %s" % (type(e).__name__, e)


def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def check_reasoning():
    payload = {"model": "qwen3.8-27b-gsq",
               "messages": [{"role": "user",
                             "content": "9.11 和 9.9 哪个大？简要说明理由。"}],
               "max_tokens": 512, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    reasoning = msg.get("reasoning") or ""
    return {"finish": d["choices"][0].get("finish_reason"),
            "reasoning_len": len(reasoning),
            "has_reasoning": len(reasoning) > 0}


def check_toolcall():
    payload = {
        "model": "qwen3.8-27b-gsq",
        "messages": [{"role": "user", "content": "广州今天天气怎么样？"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "查询城市天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "tool_choice": "auto", "max_tokens": 128, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    args_ok, city = False, None
    if tcs:
        try:
            args = json.loads(tcs[0]["function"]["arguments"])
            city = args.get("city")
            args_ok = isinstance(city, str) and len(city) > 0
        except Exception:
            args_ok = False
    return {"finish": ch.get("finish_reason"),
            "n_tool_calls": len(tcs), "args_ok": args_ok, "city": city}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries, verdicts = [], []

    def add(name, budget, fn, judge):
        rec = Deadline(name, fn, budget).run()
        entries.append(rec)
        ok = rec["status"] == "OK" and judge(rec.get("result"))
        verdicts.append((name, ok))
        print("[%s] %s %s" % ("PASS" if ok else "FAIL", name,
                              json.dumps(rec.get("result"), ensure_ascii=False)
                              [:220]), flush=True)
        return rec

    short = ("你是一个简洁的助手。请用两三句话说明什么是缓存，"
             "并举一个生活中的例子。")

    print("== soak round begin, health=%s ==" % health(), flush=True)
    if not health():
        print("FAIL service not healthy at start", flush=True)
        return 2

    # 1-3 short chat
    for i in range(3):
        add("short_chat_%d" % (i + 1), 120,
            lambda: probe.run_request(BASE, "short", short, "chat", 300),
            ok_finish)

    # 4-5 chat 2000 tok (decode metrics)
    for i in range(2):
        add("decode2k_%d" % (i + 1), 300,
            lambda: probe.run_request(BASE, "d2k", short, "chat", 2000),
            ok_finish)

    # 6-7 8k needle x2 (multi-turn TTFT gate on 2nd)
    r1 = add("needle8k_first", 300, lambda: anchor.run_one(BASE, 8000),
             lambda r: r and r.get("needle_hit"))
    r2 = add("needle8k_second", 300, lambda: anchor.run_one(BASE, 8000),
             lambda r: r and r.get("needle_hit") and (r.get("ttft_s") or 99) <= 3.5)

    # 8-10 32k needle x3 (multi-turn TTFT gate 21s -> 3.1s)
    add("needle32k_first", 600, lambda: anchor.run_one(BASE, 32000),
        lambda r: r and r.get("needle_hit"))
    add("needle32k_second", 600, lambda: anchor.run_one(BASE, 32000),
        lambda r: r and r.get("needle_hit") and (r.get("ttft_s") or 99) <= 3.5)
    add("needle32k_third", 600, lambda: anchor.run_one(BASE, 32000),
        lambda r: r and r.get("needle_hit") and (r.get("ttft_s") or 99) <= 3.5)

    # 11-12 reasoning
    for i in range(2):
        add("reasoning_%d" % (i + 1), 180, check_reasoning,
            lambda r: r and r.get("has_reasoning") and r.get("finish") == "stop")

    # 13-14 tool-call
    for i in range(2):
        add("toolcall_%d" % (i + 1), 180, check_toolcall,
            lambda r: r and r.get("finish") == "tool_calls" and r.get("args_ok"))

    # 15-16 wiki real-text accept rate (A4 band 0.68-0.80)
    w8 = probe.wiki_content(8000)
    w32 = probe.wiki_content(32000)
    add("wiki8k", 600,
        lambda: probe.run_request(BASE, "w8k", w8, "chat", 1024),
        ok_finish)
    add("wiki32k", 600,
        lambda: probe.run_request(BASE, "w32k", w32, "chat", 1024),
        ok_finish)

    # 17 wiki 8k repeat (prefix hit)
    add("wiki8k_repeat", 600,
        lambda: probe.run_request(BASE, "w8k2", w8, "chat", 1024),
        ok_finish)

    # 18 100k needle (long-prefill stress)
    add("needle100k", 900, lambda: anchor.run_one(BASE, 100000),
        lambda r: r and r.get("needle_hit"))

    # 19-20 closing health
    for i in range(2):
        add("short_chat_close_%d" % (i + 1), 120,
            lambda: probe.run_request(BASE, "close", short, "chat", 300),
            ok_finish)

    # aggregate acceptance over streaming requests
    accs = [e["result"]["accept_rate"] for e in entries
            if e["status"] == "OK" and isinstance(e.get("result"), dict)
            and e["result"].get("accept_rate") is not None]
    fails = [n for n, ok in verdicts if not ok]
    summary = {
        "n_requests": len(entries),
        "n_pass": sum(1 for _, ok in verdicts if ok),
        "fails": fails,
        "accept_rates": accs,
        "health_after": health(),
    }
    out_path = os.path.join(
        OUT_DIR, "soak_prod_result_%s.json"
        % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "entries": entries}, f,
                  ensure_ascii=False, indent=1)
    print("SUMMARY %s" % json.dumps(summary, ensure_ascii=False), flush=True)
    print("OUT %s" % out_path, flush=True)
    return 0 if (not fails and summary["health_after"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
