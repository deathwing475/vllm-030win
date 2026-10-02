# step082_boot.py -- one-boot driver for step 082 (KVMem 移植验收收口).
#
# 与 tools/step081_boot.py 的关系
# -------------------------------
# 逐字承袭 081 的协议（window 免费判态读数 / sweep 弃 1 取中位 / main cadence
# 30 页步 / prof 窗三明治 + 夹逼 / vp 探针），只改三件事：
#   * 臂 = serve_gsq_kvmem_viewport082_spec.cmd（= 081 臂 + S082_KVMB/S082_WSMB
#     两个驻留旋钮，其余逐字）；
#   * 状态/日志前缀换 step082_（每步一commit、证据不混档）;
#   * 新增 --pdh 1：boot 全程开一条 PDH 采样线程（tools/pdh_gpu_engine.py，
#     082-C 的仪器），每 2 s 一行 jsonl，记录独显 LUID 上每个进程的
#     GPU Engine 利用率（pid x engtype）与 GPU Process Memory 的
#     local/non_local（non_local = 被 WDDM 降到系统内存的字节 = 权重页驻留
#     假设的直接读数）。
#
# 用法（venv python、cwd 必须不在记录仓，铁律 2）：
#   cd G:\qwen3.8model
#   G:\qwen3.8model\vllm-win029\Scripts\python.exe ^
#       G:\qwen3.8model\vllm-030win-git\tools\step082_boot.py next --boot b1 --tag kvmem_k16a
#
# State: G:\qwen3.8model\prod029_logs\step082_boots.json (append-merge per boot).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
ARM = os.path.join(TOOLS, "serve_gsq_kvmem_viewport082_spec.cmd")
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
KILL_PS1 = os.path.join(TOOLS, "kill_vllm_orphans.ps1")
CADENCE = os.path.join(TOOLS, "kvmem_ingest_cadence.py")
VPROBE = os.path.join(TOOLS, "kvmem_viewport_probe.py")
WINDOW = os.path.join(TOOLS, "kvmem_boot_window.py")
STATE = os.path.join(LOGS, "step082_boots.json")
BASE = "http://127.0.0.1:8080"

sys.path.insert(0, TOOLS)
import step080_boot as s80  # noqa: E402  (reuse down/verify/health/headroom)
from kvmem_boot_fingerprint import Sampler  # noqa: E402

# 独显 LUID（2026-10-02 实测：11746 的 dedicated ≈ nvidia-smi 读数；
# F28A 是核显桌面，116FB 是空适配器）。
PDH_LUID = "0x00000000_0x00011746"

# Page-step request shape: 1456 tokens == exactly one page step, and the arm's
# budget is mbt(1458) - draft_slots(2) = 1456 (iron rule 16-10).
PAGE = 1456


def say(msg: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def post_json(path: str, timeout: float = 300.0) -> tuple[int, str]:
    req = urllib.request.Request(BASE + path, method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, ""
    except Exception as exc:  # noqa: BLE001
        return -1, repr(exc)[:300]


# ----------------------------------------------------------------- pdh -------
class PdhThread:
    """082-C 采样线程：boot 全程每 interval 秒一行 PDH jsonl（只记非零值）。"""

    def __init__(self, boot: str, interval: float = 2.0) -> None:
        self.path = os.path.join(LOGS, "step082_%s_pdh.jsonl" % boot)
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.rows = 0

    def _loop(self) -> None:
        from pdh_gpu_engine import EngineUtilQuery, sample_once
        engq = EngineUtilQuery(None, PDH_LUID)  # 速率计数器须持久句柄
        fh = open(self.path, "a", encoding="utf-8")
        while not self._stop.is_set():
            try:
                row = sample_once(None, PDH_LUID, engq)
            except Exception as exc:  # noqa: BLE001
                row = {"pdh_err": repr(exc)[:200]}
            row["ts"] = time.strftime("%H:%M:%S")
            try:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                self.rows += 1
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)
        fh.close()
        engq._close()

    def __enter__(self) -> "PdhThread":
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        say("pdh thread on -> %s (luid=%s)" % (self.path, PDH_LUID))
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=15)
        say("pdh thread off (%d rows)" % self.rows)


# ---------------------------------------------------------------- sweep ------
class Sweeper:
    """N single-page requests, each with its own nonce (no prefix-cache hit).

    Iron rule 22 (ii): a repeated identical prompt would native-hit the prefix
    cache and print a fake cadence, so nonces must differ. The tokenizer is
    loaded once per process, which is why this lives in the driver instead of
    calling kvmem_ingest_cadence.py per request.
    """

    def __init__(self, tokens: int = PAGE) -> None:
        from kvmem_k3_probe import build, tokenizer  # heavy imports, late
        from kvmem_ws_probe import post
        self._build, self._tok, self._post = build, tokenizer, post
        self.tokens = tokens

    def one(self, nonce: str) -> dict | None:
        prompt, _ = self._build(self.tokens, 0.65, nonce, False, "focused")
        n = len(self._tok().encode(prompt, add_special_tokens=False))
        res = self._post(BASE, prompt, 8, ignore_eos=True)
        ttft = res.get("ttft_s")
        return None if not ttft else {"ttft_s": ttft, "tokens": n,
                                      "steps": max(1, -(-n // PAGE))}

    def run(self, count: int, tag: str) -> dict:
        vals = []
        for i in range(count):
            rec = self.one("%s-s%02d" % (tag, i))
            if rec is None:
                say("sweep %s #%d FAILED (no ttft)" % (tag, i))
                continue
            per = rec["ttft_s"] / rec["steps"]
            vals.append(round(per, 3))
            say("sweep %s #%d tokens=%d steps=%d s/step=%.3f"
                % (tag, i, rec["tokens"], rec["steps"], per))
        out = {"tag": tag, "raw": vals, "n": len(vals)}
        if len(vals) >= 2:
            out["median_drop1"] = round(_median(vals[1:]), 3)
            out["median_all"] = round(_median(vals), 3)
        return out


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


# ---------------------------------------------------------------- window -----
def read_window(err_log: str) -> dict:
    out_json = os.path.join(
        LOGS, "step082_%s_window.json" % os.path.basename(err_log)
        .replace(".err.log", "").replace(".out.log", ""))
    proc = subprocess.run([PY, WINDOW, "--files", err_log, "--out", out_json],
                          capture_output=True, text=True, cwd=SCRATCH)
    rec = {}
    try:
        with open(out_json, encoding="utf-8") as fh:
            rows = json.load(fh).get("rows") or []
        if rows:
            r = rows[-1]
            rec = {"window_s": r.get("window_s"), "window_state": r.get("state")
                   or r.get("window_state"), "mbt": r.get("mbt")}
    except Exception as exc:  # noqa: BLE001
        rec = {"window_error": repr(exc)[:200]}
    say("window: %s" % json.dumps(rec, ensure_ascii=False))
    return rec


def cadence(tokens: int, nonce: str, out_log: str, js: str) -> dict:
    proc = subprocess.run(
        [PY, CADENCE, "run", "--tokens", str(tokens), "--nonce", nonce,
         "--engine-log", out_log, "--out", js],
        capture_output=True, text=True, cwd=SCRATCH, timeout=3000)
    if proc.returncode != 0:
        return {"cadence_rc": proc.returncode,
                "cadence_err": (proc.stderr or proc.stdout)[-300:]}
    with open(js, encoding="utf-8") as fh:
        d = json.load(fh)
    return {"cadence_json": js,
            "s_per_page_step": d.get("s_per_page_step"),
            "engine_drain_median_s": (d.get("engine_drain") or {}).get("median_s"),
            "cadence_tokens": d.get("tokens")}


# ------------------------------------------------------------------ boot -----
def up(tag: str, boot: str, every: int, extra: dict, prof: bool,
       pick: str, timeout: float) -> dict:
    out_log = os.path.join(LOGS, "step082_%s_arm.out.log" % boot)
    err_log = os.path.join(LOGS, "step082_%s_arm.err.log" % boot)
    env = dict(os.environ)
    env["S079_TAG"] = tag
    env["S079_EVERY"] = str(every)
    env["S079_TIMING"] = "1"
    env["S081_PROF"] = "1" if prof else "0"
    env["S081_PICK"] = pick
    for key, val in extra.items():
        env[key] = str(val)
    with open(out_log, "w", encoding="utf-8", errors="replace") as fo, \
            open(err_log, "w", encoding="utf-8", errors="replace") as fe:
        subprocess.Popen([PY, BOOT_KEEP, ARM, out_log, err_log], cwd=SCRATCH,
                         env=env)
        t0 = time.time()
        rec = {"boot": boot, "tag": tag, "prof": int(prof), "pick": pick,
               "out_log": out_log, "err_log": err_log,
               "up_at": time.strftime("%H:%M:%S")}
        while time.time() - t0 < timeout:
            time.sleep(5)
            if s80.health():
                break
            blob = "\n".join(
                ln for ln in (s80.read_tail(err_log, 200_000)
                              + s80.read_tail(out_log, 400_000)).splitlines()
                if "[rank0]:W" not in ln)
            if ("EngineCore failed to start" in blob
                    or "Traceback (most recent call" in blob):
                rec["fatal"] = True
                break
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["health"] = s80.health()
    rec.update(s80.verify(out_log, err_log))
    say("up %s/%s: boot_s=%s health=%s load_n=%s compiling_n=%s block=%s "
        "slots=%s kvtime=%s err=%s tb=%s" % (
            boot, tag, rec["boot_s"], rec["health"], rec.get("directly_load_n"),
            rec.get("compiling_n"), rec.get("attention_block"),
            rec.get("host_slots"), rec.get("kvtime_loaded"),
            rec.get("error_lines"), rec.get("traceback_lines")))
    return rec


def state_of(s: float | None) -> str:
    if s is None:
        return "UNKNOWN"
    return ("fast" if s <= s80.FAST_MAX else
            "slow" if s >= s80.SLOW_MIN else "mid")


def cmd_next(args) -> int:
    dump_dir = os.path.join(LOGS, args.tag)
    if os.path.exists(dump_dir):
        say("REFUSING: dump dir %s already exists (iron rule 16-viii)" % dump_dir)
        return 2
    extra = {}
    for item in args.env or []:
        key, _, val = item.partition("=")
        extra[key] = val
    rec = {"boot": args.boot, "tag": args.tag, "started":
           time.strftime("%Y-%m-%d %H:%M:%S"), "extra_env": extra,
           "condition": args.condition}
    if not args.no_down:
        rec["down"] = s80.down(args.wait_mib, 240, args.boot)
    pdh = PdhThread(args.boot) if args.pdh else None
    if pdh is not None:
        pdh.__enter__()
    try:
        rec.update(up(args.tag, args.boot, args.every, extra, bool(args.prof),
                      args.pick, args.timeout))
        if not rec.get("health"):
            say("boot %s never reached health -- record and stop" % args.boot)
            merge(args.boot, rec)
            return 1
        # 1) the free readout: one model-only 1458-token forward at boot
        rec.update(read_window(rec["err_log"]))
        # 2) request-side state: drop #1, median of >=3 (iron rule 25 companion-2)
        sweeper = Sweeper()
        rec["sweep_pre"] = sweeper.run(args.sweep, "s082%s_pre" % args.boot)
        rec["sweep_state"] = state_of(rec["sweep_pre"].get("median_drop1"))
        say("boot %s: window=%s (%s) sweep_drop1_median=%s (%s)" % (
            args.boot, rec.get("window_s"), rec.get("window_state"),
            rec["sweep_pre"].get("median_drop1"), rec["sweep_state"]))
        # 3) the 079/080/081-comparable main cadence (30 page steps of one ingest)
        if not args.no_main:
            rec["main"] = cadence(args.main_tokens, "s082%s_main" % args.boot,
                                  rec["out_log"],
                                  os.path.join(LOGS, "step082_%s_cadence.json"
                                               % args.boot))
            rec["main_state"] = state_of(rec["main"].get("s_per_page_step"))
            say("main cadence: %s s/page step (%s)" % (rec["main"].get(
                "s_per_page_step"), rec["main_state"]))
    finally:
        if pdh is not None:
            pdh.__exit__()
            rec["pdh_rows"] = pdh.rows
            rec["pdh_jsonl"] = pdh.path
    merge(args.boot, rec)
    say("STATE: %s" % json.dumps({k: rec.get(k) for k in (
        "boot", "tag", "condition", "pick", "window_s", "window_state",
        "sweep_state", "main_state", "compiling_n", "error_lines",
        "traceback_lines")}, ensure_ascii=False))
    return 0


def cmd_prof(args) -> int:
    """One profile window: single-page requests + a short ingest, then stop."""
    rec = current(args.boot)
    out_log = rec["out_log"]
    prof_dir = os.path.join(LOGS, rec["tag"], "prof")
    before = set(os.listdir(prof_dir)) if os.path.isdir(prof_dir) else set()
    t_start = time.time()
    rc, err = post_json("/start_profile")
    say("start_profile rc=%s %s" % (rc, err))
    if rc != 200:
        merge(args.boot, {"prof_error": "start rc=%s %s" % (rc, err)})
        return 1
    fp_csv = os.path.join(LOGS, "step082_%s_engine_fp.csv" % args.boot)
    sampler = Sampler(1.0, out_csv=fp_csv)
    sampler.start()
    pdh = PdhThread(args.boot) if args.pdh else None
    if pdh is not None:
        pdh.__enter__()
    try:
        sweeper = Sweeper()
        rec["sweep_prof"] = sweeper.run(args.sweep, "s082%s_prof" % args.boot)
        rec["ingest_prof"] = cadence(args.pages * PAGE, "s082%s_ing" % args.boot,
                                     out_log,
                                     os.path.join(LOGS, "step082_%s_ingestprof.json"
                                                  % args.boot))
    finally:
        rc2, err2 = post_json("/stop_profile", timeout=900)
        fp_rows = sampler.stop()
        if pdh is not None:
            pdh.__exit__()
        say("stop_profile rc=%s %s (window %.0f s; fingerprint rows=%d "
            "coverage=%.2f)" % (rc2, err2, time.time() - t_start, len(fp_rows),
                                sampler.coverage()))
    rec["fp_rows"] = len(fp_rows)
    rec["fp_coverage"] = round(sampler.coverage(), 3)
    # the trace + the key_averages table land asynchronously; wait, then stamp
    # names so a second window in the same boot cannot collide (080 lesson:
    # profiler_out_<rank>.txt is a fixed name).
    new = []
    t0 = time.time()
    while time.time() - t0 < args.wait_trace:
        time.sleep(3)
        if os.path.isdir(prof_dir):
            new = sorted(set(os.listdir(prof_dir)) - before)
            if new:
                break
    stamped = {}
    win_index = 1 + (len(rec.get("prof_files") or {}))
    for idx, name in enumerate(sorted(new)):
        # 同一支 boot 的第二个窗口会撞死名文件（`profiler_out_<rank>.txt` 是固定
        # 名，trace 名里只有毫秒时间戳），所以一律加 boot + 窗口序号再落盘。
        dst = os.path.join(prof_dir, "%s_w%d_%s" % (args.boot, win_index, name))
        try:
            os.replace(os.path.join(prof_dir, name), dst)
            stamped[name] = os.path.basename(dst)
        except Exception as exc:  # noqa: BLE001
            stamped[name] = "rename failed: %r" % (exc,)
    rec["prof_window_s"] = round(time.time() - t_start, 1)
    rec["prof_files"] = stamped
    rec["prof_fp_csv"] = fp_csv
    # intrusion: same-shape sweep right after the window
    rec["sweep_post"] = Sweeper().run(3, "s082%s_post" % args.boot)
    say("prof traces: %s" % json.dumps(stamped, ensure_ascii=False))
    merge(args.boot, rec)
    return 0


def current(boot: str) -> dict:
    for row in load():
        if row.get("boot") == boot:
            return row
    raise SystemExit("no recorded boot %s in %s" % (boot, STATE))


def load() -> list:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save(rows: list) -> None:
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def merge(boot: str, patch: dict) -> None:
    """把 patch 并进这一支 boot 的那一行，绝不丢已有字段（081 教训）。"""
    rows = load()
    row = next((r for r in rows if r.get("boot") == boot), None)
    if row is None:
        row = {"boot": boot}
        rows.append(row)
    row.update(patch)
    rows.sort(key=lambda r: r.get("boot", ""))
    save(rows)


def cmd_vp(args) -> int:
    rec = current(args.boot)
    js = os.path.join(LOGS, rec["tag"], "vp082_%s_d%02d%s.json" % (
        args.nonce, int(args.depth * 100), "_ignoreeos" if args.ignore_eos else ""))
    cmd = [PY, VPROBE, "run", "--tokens", str(args.tokens),
           "--depth", str(args.depth), "--nonce", args.nonce,
           "--question-style", "focused", "--out", js]
    if args.serve_only:
        cmd += ["--stages", "serve"]
    if args.ignore_eos:
        cmd += ["--serve-ignore-eos", "--max-tokens", str(args.max_tokens)]
    say("viewport probe: %s" % " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRATCH,
                          timeout=args.timeout)
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-400:]
    print(tail)
    merge(args.boot, {"vp_%s" % args.nonce: {
        "json": js, "rc": proc.returncode,
        "verdict": "HIT" if "VIEWPORT VERDICT: HIT" in tail else
                   ("MISS" if "VIEWPORT VERDICT: MISS" in tail else "ERR"),
        "tokens": args.tokens, "depth": args.depth,
        "serve_only": bool(args.serve_only), "ignore_eos": bool(args.ignore_eos),
    }})
    return 0


def cmd_stop(args) -> int:
    rec = {"stop_at": time.strftime("%H:%M:%S"), "label": args.label}
    s80.down(800.0, 240, "stop-" + (args.label or ""))
    rows = [r for r in load() if r.get("boot") != "STOP"]
    rows.append({"boot": "STOP", **rec})
    save(rows)
    return 0


def cmd_show(args) -> int:
    rows = load()
    print(f"{'boot':5} {'tag':11} {'cond':9} {'prof':4} {'pick':5} "
          f"{'win_s':>7} {'winSt':>6} {'sweepPre':>8} {'St':>5} {'main':>7} "
          f"{'St':>5} {'cmp':>4} {'err':>4} {'tb':>3} {'up_at':>9}")
    for r in sorted(rows, key=lambda x: x.get("boot", "")):
        if r.get("boot") == "STOP":
            continue
        print(f"{r.get('boot',''):5} {str(r.get('tag'))[:11]:11} "
              f"{str(r.get('condition','-'))[:9]:9} "
              f"{str(r.get('prof','-')):4} {str(r.get('pick','-')):5} "
              f"{str(r.get('window_s')):>7} {str(r.get('window_state'))[:6]:>6} "
              f"{str((r.get('sweep_pre') or {}).get('median_drop1')):>8} "
              f"{str(r.get('sweep_state'))[:5]:>5} "
              f"{str((r.get('main') or {}).get('s_per_page_step')):>7} "
              f"{str(r.get('main_state'))[:5]:>5} "
              f"{str(r.get('compiling_n')):>4} {str(r.get('error_lines')):>4} "
              f"{str(r.get('traceback_lines')):>3} {str(r.get('up_at')):>9}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next")
    n.add_argument("--boot", required=True)
    n.add_argument("--tag", required=True)
    n.add_argument("--condition", default="")
    n.add_argument("--every", type=int, default=10)
    n.add_argument("--timeout", type=float, default=420)
    n.add_argument("--wait-mib", type=float, default=800)
    n.add_argument("--env", action="append")
    n.add_argument("--prof", type=int, default=1)
    n.add_argument("--pdh", type=int, default=1)
    n.add_argument("--pick", default="time", choices=("time", "score"))
    n.add_argument("--sweep", type=int, default=5)
    n.add_argument("--main-tokens", type=int, default=43680)
    n.add_argument("--no-main", action="store_true")
    n.add_argument("--no-down", action="store_true")
    n.set_defaults(func=cmd_next)
    p = sub.add_parser("prof")
    p.add_argument("--boot", required=True)
    p.add_argument("--sweep", type=int, default=6)
    p.add_argument("--pages", type=int, default=20)
    p.add_argument("--wait-trace", type=float, default=240)
    p.add_argument("--pdh", type=int, default=1)
    p.set_defaults(func=cmd_prof)
    v = sub.add_parser("vp")
    v.add_argument("--boot", required=True)
    v.add_argument("--depth", type=float, required=True)
    v.add_argument("--nonce", required=True)
    v.add_argument("--tokens", type=int, default=200000)
    v.add_argument("--serve-only", action="store_true")
    v.add_argument("--ignore-eos", action="store_true")
    v.add_argument("--max-tokens", type=int, default=96)
    v.add_argument("--timeout", type=float, default=2400)
    v.set_defaults(func=cmd_vp)
    s = sub.add_parser("stop")
    s.add_argument("--label", default="")
    s.set_defaults(func=cmd_stop)
    h = sub.add_parser("show")
    h.set_defaults(func=cmd_show)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
