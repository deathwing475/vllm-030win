# step085_orca_boot.py -- one-boot driver for step 085 (Orca O2: checkpoint-native MTP).
#
# 协议（docs/orcasaq2后续推进计划.md §4 O2）
# ------------------------------------------
# 两臂逐字同参，只动一个变量 = 投机：
#   arm=on   -> tools/serve_orcasaq2_029_mtp.cmd（059 + method mtp + num_spec 2）
#   arm=off  -> tools/serve_orcasaq2_029_nomtp_12k.cmd（059，无投机）
# 两支都是 --max-model-len 12000（不是 059/O1 的 16384）、gpu-memory-utilization 0.88：
# 带 MTP 头时单个 16K 请求要 1.71 GiB KV，而 0.88 只剩 1.44 GiB，引擎自估最大长度 12,000；
# 0.95 在这张卡上直接不可达（启动空闲 14.68/15.89 GiB < 0.95 要的 15.1 GiB）。
# 其余 = auto KV / eager / max-num-seqs 1。
# 一支 boot 内做完：起服 -> 健康 -> 投机计数器取基线 -> chat identity 三连 ->
# needle 三深度（12K token，与 084 boot5 同口径）-> 长解码三连（接受率样本）->
# 计数器取末值 -> 解析 boot 日志（KV 池 / 并发 / 权重与 KV 显存 / attention block）
# -> 落台账 prod029_logs/step085_boots.json。
#
# 接受率读的是引擎自己的 prometheus 计数器（vllm:spec_decode_*），取 boot 内
# 两次快照之差，所以它是"本 boot 本探针集"的样本，不是全局累计。
#
# 用法（venv python；cwd 必须在记录仓之外 = 必守 2）：
#   cd G:\qwen3.8model
#   G:\qwen3.8model\vllm-win029\Scripts\python.exe ^
#       G:\qwen3.8model\vllm-030win-git\tools\step085_orca_boot.py ^
#       next --boot b1 --arm on
#   ... next --boot b1o --arm off
#   tools/step085_orca_boot.py show
#   tools/step085_orca_boot.py stop

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
EVID = os.path.join(LOGS, "orca_o2")
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
KILL_PS1 = os.path.join(TOOLS, "kill_vllm_orphans.ps1")
PROBE = os.path.join(TOOLS, "orca_nvfp4_probe.py")
STATE = os.path.join(LOGS, "step085_boots.json")

ARMS = {
    "on": os.path.join(TOOLS, "serve_orcasaq2_029_mtp.cmd"),
    "off": os.path.join(TOOLS, "serve_orcasaq2_029_nomtp_12k.cmd"),
}
BASE = "http://127.0.0.1:8001"

# The needle haystack size and depths follow 084 boot5, shrunk for this arm's 12,000-token
# context: build_prompt overshoots its target (asking 11,000 delivered >= 11,905 and the
# engine answered HTTP 400), so 9,000 is the size that actually fits with 96 output tokens.
NEEDLE_TOKENS = 9000
NEEDLE_DEPTHS = "0.25,0.5,0.75"
DECODE_PROMPT = (
    "Count down from 500 to 400, one number per line, nothing else.\nAnswer:"
)
DECODE_TOKENS = 220


def say(msg: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gpu_used() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0].split(",")[0]) if out else -1.0


def get(path: str, timeout: float = 20.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return -1, repr(exc)[:200]


def health() -> bool:
    return get("/health", 10.0)[0] == 200


# ------------------------------------------------------------- metrics -------
SPEC_RE = re.compile(
    r'^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens'
    r'|num_accepted_tokens_per_pos)(?:_total)?\{([^}]*)\}\s+([0-9.eE+-]+)')


def spec_counters() -> dict:
    """Parse the engine's spec-decode prometheus counters into a flat dict."""
    rc, body = get("/metrics", timeout=30.0)
    if rc != 200:
        return {"metrics_error": body}
    out: dict[str, float] = {}
    for line in body.splitlines():
        m = SPEC_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), float(m.group(3))
        pos = re.search(r'position="([^"]+)"', labels)
        key = name + (f"_pos{pos.group(1)}" if pos else "")
        out[key] = out.get(key, 0.0) + val
    return out


def acceptance(before: dict, after: dict) -> dict:
    def delta(k: str) -> float:
        return after.get(k, 0.0) - before.get(k, 0.0)

    drafts = delta("num_drafts")
    dtok = delta("num_draft_tokens")
    acc = delta("num_accepted_tokens")
    rec: dict[str, object] = {
        "drafts": int(drafts), "draft_tokens": int(dtok), "accepted_tokens": int(acc),
    }
    if dtok > 0:
        rec["acceptance_rate"] = round(acc / dtok, 4)
    if drafts > 0:
        rec["mean_acceptance_length"] = round(1.0 + acc / drafts, 4)
        rec["per_pos_rate"] = {
            p: round(delta(f"num_accepted_tokens_per_pos_pos{p}") / drafts, 4)
            for p in ("0", "1") if f"num_accepted_tokens_per_pos_pos{p}" in after
            or f"num_accepted_tokens_per_pos_pos{p}" in before
        }
    return rec


# --------------------------------------------------------------- logparse ----
PATTERNS = {
    "attention_block_tokens": r"Setting attention block size to (\d+) tokens",
    "kv_pool_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "kv_cache_avail_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "model_load_gib": r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
    "weight_gib": r"Actual usage is ([\d.]+) GiB for consumed memory",
    "resolved_arch": r"Resolved architecture: (\S+)",
    "draft_arch": r"draft model.*?architectures?=.?\[?'?(\w*MTP\w*)",
    "spec_tokens": r"num_speculative_tokens'?[:=]\s*(\d+)",
    "mamba_cache_mode": r"Mamba cache mode is set to '(\w+)'",
}


def parse_boot_log(out_log: str, err_log: str) -> dict:
    blob = ""
    for path in (out_log, err_log):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                blob += fh.read()
    rec: dict[str, object] = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, blob)
        if m:
            rec[key] = m.group(1).replace(",", "") if key.endswith(
                ("tokens", "gib", "concurrency")) or key in (
                "kv_pool_tokens", "attention_block_tokens", "spec_tokens") else m.group(1)
    rec["error_lines"] = len(re.findall(r"\bERROR\b", blob))
    rec["traceback_lines"] = len(re.findall(r"Traceback \(most recent call", blob))
    rec["engine_failed"] = "EngineCore failed to start" in blob
    rec["draft_model_log"] = [ln.strip()[:200] for ln in blob.splitlines()
                             if "speculative" in ln.lower()][:4]
    return rec


# --------------------------------------------------------------- probes ------
def run_probe(kind: str, out_json: str, extra: list[str]) -> dict:
    cmd = [PY, PROBE, kind, "--base", BASE, "--out", out_json, "--tag",
           os.path.splitext(os.path.basename(out_json))[0][:24]] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRATCH,
                          timeout=1800)
    tail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-400:]
    print(tail, flush=True)
    rec = {"json": out_json, "rc": proc.returncode}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            data = json.load(fh)
        rec.update(data)
    return rec


def decode_rates(needle: dict, chat: dict, dec: dict) -> dict:
    """tok/s from the streaming usage block: completion_tokens / (latency - ttft).

    Recorded, not a gate (O2's acceptance criteria are acceptance rate,
    correctness, KV capacity and memory). Short completions inflate the figure
    because the rate excludes TTFT only, not the tail of the stream.
    """
    rates = []
    for src in (chat, needle, dec):
        for name, r in (src.get("results") or {}).items():
            ct, ttft, lat = r.get("completion_tokens"), r.get("ttft_s"), r.get("latency_s")
            if ct and ttft and lat and lat > ttft and ct >= 16:
                rates.append({"name": name, "tokens": ct,
                              "tok_s": round(ct / (lat - ttft), 2)})
    vals = sorted(x["tok_s"] for x in rates)
    return {"per_request": rates,
            "median_tok_s": vals[len(vals) // 2] if vals else None,
            "n": len(vals)}


# ----------------------------------------------------------------- boot ------
def boot_and_probe(boot: str, arm: str, keep_up: bool, timeout: float,
                   wait_mib: float, no_down: bool) -> dict:
    os.makedirs(EVID, exist_ok=True)
    out_log = os.path.join(EVID, "%s_%s.out.log" % (boot, arm))
    err_log = os.path.join(EVID, "%s_%s.err.log" % (boot, arm))
    rec: dict[str, object] = {"boot": boot, "arm": arm, "arm_path": ARMS[arm],
                              "out_log": out_log, "err_log": err_log,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not no_down:
        rec["down"] = down(wait_mib, 240.0, boot)
    say("boot %s arm=%s: launching %s" % (boot, arm, os.path.basename(ARMS[arm])))
    with open(out_log, "w", encoding="utf-8", errors="replace") as fo, \
            open(err_log, "w", encoding="utf-8", errors="replace") as fe:
        subprocess.Popen([PY, BOOT_KEEP, ARMS[arm], out_log, err_log], cwd=SCRATCH)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(5)
            if health():
                break
            blob = ""
            for path in (err_log, out_log):
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        fh.seek(max(0, os.path.getsize(path) - 200_000))
                        blob += fh.read()
            if ("EngineCore failed to start" in blob
                    or "Traceback (most recent call last)" in blob):
                rec["fatal"] = True
                break
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["health"] = health()
    say("boot %s: boot_s=%s health=%s fatal=%s" % (
        boot, rec["boot_s"], rec["health"], rec.get("fatal")))
    rec.update(parse_boot_log(out_log, err_log))

    if rec["health"]:
        m0 = spec_counters()
        rec["chat"] = run_probe("chat", os.path.join(
            EVID, "%s_chat.json" % boot), ["--max-tokens", "48"])
        rec["needle"] = run_probe("needle", os.path.join(
            EVID, "%s_needle.json" % boot),
            ["--tokens", str(NEEDLE_TOKENS), "--depths", NEEDLE_DEPTHS,
             "--max-tokens", "96"])
        rec["decode"] = run_probe("needle", os.path.join(
            EVID, "%s_decode.json" % boot),
            ["--tokens", "600", "--depths", "0.5", "--max-tokens",
             str(DECODE_TOKENS)])
        m1 = spec_counters()
        rec["metrics_before"] = m0
        rec["metrics_after"] = m1
        rec["acceptance"] = acceptance(m0, m1)
        rec["decode_rate"] = decode_rates(rec["chat"], rec["needle"], rec["decode"])
        say("boot %s: acceptance=%s decode_median=%s" % (
            boot, json.dumps(rec["acceptance"], ensure_ascii=False),
            rec["decode_rate"].get("median_tok_s")))
        rec["gpu_used_mib"] = gpu_used()
    if not keep_up:
        rec["down_after"] = down(wait_mib, 240.0, boot + "-after")
    merge(boot, rec)
    return rec


def down(wait_mib: float, timeout: float, label: str) -> dict:
    proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy",
                           "Bypass", "-File", KILL_PS1], capture_output=True,
                          text=True)
    killed = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    t0 = time.time()
    used = gpu_used()
    while used > wait_mib and time.time() - t0 < timeout:
        time.sleep(5)
        used = gpu_used()
    say("down(%s): killed=%s gpu_used=%.0f MiB after %.0fs" % (
        label, killed, used, time.time() - t0))
    return {"killed": killed, "gpu_used_mib_after": used,
            "down_wait_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- state ------
def load() -> list:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save(rows: list) -> None:
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def merge(boot: str, patch: dict) -> None:
    rows = load()
    row = next((r for r in rows if r.get("boot") == boot), None)
    if row is None:
        row = {"boot": boot}
        rows.append(row)
    row.update(patch)
    rows.sort(key=lambda r: r.get("boot", ""))
    save(rows)


def cmd_next(args) -> int:
    rec = boot_and_probe(args.boot, args.arm, args.keep_up, args.timeout,
                         args.wait_mib, args.no_down)
    say("STATE %s" % json.dumps({k: rec.get(k) for k in (
        "boot", "arm", "boot_s", "health", "fatal", "attention_block_tokens",
        "kv_pool_tokens", "max_concurrency", "kv_cache_avail_gib", "model_load_gib",
        "error_lines", "traceback_lines")}, ensure_ascii=False))
    return 0 if rec.get("health") else 1


def cmd_probe_only(args) -> int:
    """Attach probes to a boot that is already serving (crash-diagnosis path)."""
    rows = load()
    row = next((r for r in rows if r.get("boot") == args.boot), None)
    if row is None:
        raise SystemExit("unknown boot %s" % args.boot)
    if not health():
        raise SystemExit("server on %s is not healthy" % BASE)
    m0 = spec_counters()
    rec: dict[str, object] = {"chat": run_probe("chat", os.path.join(
        EVID, "%s_chat.json" % args.boot), ["--max-tokens", "48"])}
    rec["needle"] = run_probe("needle", os.path.join(
        EVID, "%s_needle.json" % args.boot),
        ["--tokens", str(NEEDLE_TOKENS), "--depths", NEEDLE_DEPTHS])
    rec["decode"] = run_probe("needle", os.path.join(
        EVID, "%s_decode.json" % args.boot),
        ["--tokens", "600", "--depths", "0.5", "--max-tokens", str(DECODE_TOKENS)])
    m1 = spec_counters()
    rec["acceptance"] = acceptance(m0, m1)
    rec["decode_rate"] = decode_rates(rec["chat"], rec["needle"], rec["decode"])
    say("acceptance=%s" % json.dumps(rec["acceptance"], ensure_ascii=False))
    merge(args.boot, rec)
    return 0


def cmd_stop(args) -> int:
    merge("STOP", {"stopped": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "down": down(args.wait_mib, 240.0, args.label or "stop")})
    return 0


def cmd_show(args) -> int:
    rows = [r for r in load() if r.get("boot") != "STOP"]
    print("%-5s %-4s %6s %6s %8s %8s %7s %7s %6s %6s %5s %6s %6s" % (
        "boot", "arm", "boot_s", "health", "block", "kvpool", "conc", "kvgib",
        "wGiB", "errs", "acc", "accTok", "med_tps"))
    for r in rows:
        acc = r.get("acceptance") or {}
        dr = r.get("decode_rate") or {}
        print("%-5s %-4s %6s %6s %8s %8s %7s %7s %6s %6s %5s %6s %6s" % (
            r.get("boot", ""), r.get("arm", ""), r.get("boot_s", ""),
            str(r.get("health")), r.get("attention_block_tokens", ""),
            r.get("kv_pool_tokens", ""), r.get("max_concurrency", ""),
            r.get("kv_cache_avail_gib", ""), r.get("model_load_gib", ""),
            r.get("error_lines", ""), acc.get("acceptance_rate", ""),
            acc.get("mean_acceptance_length", ""), dr.get("median_tok_s", "")))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next")
    n.add_argument("--boot", required=True)
    n.add_argument("--arm", required=True, choices=("on", "off"))
    n.add_argument("--timeout", type=float, default=600.0)
    n.add_argument("--wait-mib", type=float, default=800.0)
    n.add_argument("--keep-up", action="store_true")
    n.add_argument("--no-down", action="store_true")
    n.set_defaults(func=cmd_next)
    p = sub.add_parser("probe")
    p.add_argument("--boot", required=True)
    p.set_defaults(func=cmd_probe_only)
    s = sub.add_parser("stop")
    s.add_argument("--label", default="")
    s.add_argument("--wait-mib", type=float, default=800.0)
    s.set_defaults(func=cmd_stop)
    h = sub.add_parser("show")
    h.set_defaults(func=cmd_show)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
