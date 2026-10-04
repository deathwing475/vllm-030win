"""step090_orca_speed.py -- speed-tier driver for the Orca DFlash2 arm (step 090).

Why: step 088/089 produced decode readings from single boots (c1 median 34.81 tok/s
with a 223 s needle TTFT, c2 median 3.63 tok/s at L=131,072). Iron rule 7 says a
speed claim needs >=3 boots and must report the spread; iron rule 25(ii)/26 says the
classification must pair utilization.gpu with utilization.memory, power.draw,
temperature, clocks and pcie link, otherwise "slow" gets misattributed (081 killed the
"submit-completion handshake" theory exactly this way).

Per boot it does: kill + wait for the dGPU to drop -> launch the prodcap arm (env
S089_L / S089_POOL / S089_SPEC / S089_MBT / S089_UTIL come from the caller's
environment) -> wait /health -> parse the capacity banners and the DFlash2
self-attestation -> run the campaign's own 8k anchor probe (anchor_longctx.py,
warmup 0, repeats 3) while sampling nvidia-smi out-of-band once a second -> append a
row to prod029_logs/step090_boots.json.

Usage (venv python, cwd OUTSIDE the repo -- iron rule 2):
  cd G:\\qwen3.8model
  G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe ^
      G:\\qwen3.8model\\vllm-030win-git\\tools\\step090_orca_speed.py next --boot s1
  ... next --boot s2 / s3 / s4 (s4 with S089_L=131072 S089_POOL=3400000000)
  tools/step090_orca_speed.py show
  tools/step090_orca_speed.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.request

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
EVID = os.environ.get("S090_EVID_DIR", os.path.join(LOGS, "orca_o3"))
STATE = os.environ.get("S090_STATE_FILE", os.path.join(LOGS, "step090_boots.json"))
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
KILL_PS1 = os.path.join(TOOLS, "kill_vllm_orphans.ps1")
ANCHOR = os.path.join(TOOLS, "anchor_longctx.py")
ARM = os.path.join(TOOLS, "serve_orcasaq2_029_nvfp4_dflash2_prodcap.cmd")
BASE = "http://127.0.0.1:8001"
MODEL = "orcasaq2"

SMI = ("--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu,"
       "memory.used,clocks.sm,clocks.max.sm")
SAMI = ["nvidia-smi", "--query-gpu=" + SMI.split("=", 1)[1], "--format=csv,noheader,nounits"]
PCIE = ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader,nounits"]


def say(msg: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def smi_row() -> dict:
    out = subprocess.run(SAMI, capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return {}
    g, m, p, t, mem, clk, cmax = [x.strip() for x in out[0].split(",")]
    return {"util_gpu": float(g), "util_mem": float(m), "power_w": float(p),
            "temp_c": float(t), "vram_mib": float(mem), "clk": float(clk),
            "clk_max": float(cmax)}


def pcie_row() -> dict:
    out = subprocess.run(PCIE, capture_output=True, text=True).stdout.strip()
    try:
        a, b, c, d = [float(x.strip()) for x in out.split(",")]
        return {"pcie_gen": a, "pcie_gen_max": b, "pcie_w": c, "pcie_w_max": d}
    except Exception:  # noqa: BLE001
        return {"pcie_raw": out[:80]}


class Sampler(threading.Thread):
    """Out-of-band 1 Hz GPU sampling; never touches the engine process (必守 26)."""

    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True)
        self.period = period
        self.rows: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            r = smi_row()
            if r:
                self.rows.append(r)
            self._stop.wait(self.period)

    def stop(self) -> list[dict]:
        self._stop.set()
        return self.rows

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {}
        def q(key, fn):
            return round(fn(r[key] for r in rows if key in r), 2)
        return {
            "n": len(rows),
            "util_gpu_p50": q("util_gpu", statistics.median),
            "util_gpu_max": q("util_gpu", max),
            "util_mem_p50": q("util_mem", statistics.median),
            "util_mem_max": q("util_mem", max),
            "power_p50": q("power_w", statistics.median),
            "power_max": q("power_w", max),
            "temp_p50": q("temp_c", statistics.median),
            "clk_p50": q("clk", statistics.median),
            "clk_max_cap": q("clk_max", max),
            "vram_p50": q("vram_mib", statistics.median),
        }


def gpu_used() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0]) if out else -1.0


def get(path: str, timeout: float = 20.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return -1, repr(exc)[:200]


def health() -> bool:
    return get("/health", 10.0)[0] == 200


PATTERNS = {
    "attention_block_tokens": r"Setting attention block size to (\d+) tokens",
    "kv_pool_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "kv_cache_avail_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "model_load_gib": r"Model loading took ([\d.]+) GiB memory",
    "weight_gib": r"Actual usage is ([\d.]+) GiB for consumed memory",
    "aux_layers_from_config": r"Using Eagle3 auxiliary layers from config: \(([^)]*)\)",
    "v2_runner": r"(Using V2 Model Runner)",
    "draft_resolved_arch": r"Resolved architecture: (DFlash2DraftModel)",
    "kv_offload": r"(Offloading KV cache[^\n]{0,80})",
    "graph_mode": r"(Capturing CUDA graph|cudagraph_mode[^,\n]{0,40})",
}


def parse_logs(out_log: str, err_log: str) -> dict:
    blob = ""
    for path in (out_log, err_log):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                blob += fh.read()
    rec: dict[str, object] = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, blob)
        if m:
            rec[key] = m.group(1).replace(",", "")
    rec["error_lines"] = len(re.findall(r"\bERROR\b", blob))
    rec["traceback_lines"] = len(re.findall(r"Traceback \(most recent call", blob))
    rec["empty_shared_tolerated"] = len(re.findall(r"s085.*\b(empty|tolerat)\w*\b", blob, re.I))
    rec["cuda_graph_lines"] = len(re.findall(r"[Cc]aptur\w+ cuda graph|CUDA graph capture", blob))
    rec["kv_offload_bytes"] = sorted({
        m for m in re.findall(r"vllm:kv_offload_store_bytes\s+([0-9.eE+-]+)", blob)})[-1:]
    return rec


def down(wait_mib: float, timeout: float, label: str) -> dict:
    proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", KILL_PS1], capture_output=True, text=True)
    killed = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    t0 = time.time()
    used = gpu_used()
    while used > wait_mib and time.time() - t0 < timeout:
        time.sleep(5)
        used = gpu_used()
    say("down(%s): killed=%s gpu_used=%.0f MiB after %.0fs" % (label, killed, used, time.time() - t0))
    return {"killed": killed, "gpu_after_mib": used, "down_wait_s": round(time.time() - t0, 1)}


def run_anchor(boot: str, lengths: list[int], repeats: int) -> dict:
    out_json = os.path.join(EVID, "%s_anchor.json" % boot)
    cmd = [PY, ANCHOR, "--base", BASE, "--model", MODEL, "--lengths",
           *[str(x) for x in lengths], "--warmup", "0", "--repeats", str(repeats),
           "--arm", f"orca_dflash2_{boot}", "--out", out_json]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRATCH, timeout=3600)
    tail = (proc.stdout or "")[-2500:] + (proc.stderr or "")[-500:]
    print(tail, flush=True)
    rec: dict[str, object] = {"json": out_json, "rc": proc.returncode}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            data = json.load(fh)
        # anchor_longctx writes {"results": [{"target_tokens": L, "repeats": [row, ...],
        # "steady_tok_s": median, "needle_hit": all(...)}, ...]}
        entries = data.get("results") or []
        rec["rows"] = [row for e in entries for row in (e.get("repeats") or [])]
        rec["agg"] = [{"target_tokens": e.get("target_tokens"),
                       "steady_tok_s": e.get("steady_tok_s"),
                       "ttft_s": e.get("ttft_s"),
                       "prompt_tokens": e.get("prompt_tokens"),
                       "needle_hit": e.get("needle_hit")} for e in entries]
    return rec


def boot_speed(boot: str, lengths: list[int], repeats: int, timeout: float,
               wait_mib: float, no_down: bool, keep_up: bool) -> dict:
    os.makedirs(EVID, exist_ok=True)
    out_log = os.path.join(EVID, "%s_d2p.out.log" % boot)
    err_log = os.path.join(EVID, "%s_d2p.err.log" % boot)
    rec: dict[str, object] = {
        "boot": boot, "arm": ARM, "out_log": out_log, "err_log": err_log,
        "env": {k: v for k, v in os.environ.items() if k.startswith("S089_")},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not no_down:
        rec["down"] = down(wait_mib, 240.0, boot)
    say("boot %s: launching %s (L=%s pool=%s spec=%s mbt=%s util=%s eager=%s offload=%s)" % (
        boot, os.path.basename(ARM), os.environ.get("S089_L", "16384"),
        os.environ.get("S089_POOL", "auto"), os.environ.get("S089_SPEC", "2"),
        os.environ.get("S089_MBT", "1024"), os.environ.get("S089_UTIL", "0.922"),
        os.environ.get("S089_NOEAGER", ""), os.environ.get("S089_OFFLOAD", "8")))
    with open(out_log, "w", encoding="utf-8", errors="replace") as fo, \
            open(err_log, "w", encoding="utf-8", errors="replace") as fe:
        subprocess.Popen([PY, BOOT_KEEP, ARM, out_log, err_log], cwd=SCRATCH)
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
            if "EngineCore failed to start" in blob or "Traceback (most recent call last)" in blob:
                rec["fatal"] = True
                break
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["health"] = health()
    rec.update(parse_logs(out_log, err_log))
    if rec["health"]:
        rec["idle"] = smi_row()
        sampler = Sampler()
        sampler.start()
        rec["anchor"] = run_anchor(boot, lengths, repeats)
        rec["during_probe"] = Sampler.summarize(sampler.stop())
        rec["pcie"] = pcie_row()
        rec["vram_after_mib"] = gpu_used()
        rates = [r.get("steady_tok_s") for r in (rec["anchor"].get("rows") or [])
                 if r.get("steady_tok_s")]
        rec["needle_hits"] = [bool(r.get("needle_hit")) for r in (rec["anchor"].get("rows") or [])]
        rec["median_tok_s"] = statistics.median(rates) if rates else None
        rec["spread"] = (round(min(rates), 2), round(max(rates), 2)) if rates else None
        say("boot %s: median_tok_s=%s spread=%s util(g/m)=%s/%s power=%sW clk=%s" % (
            boot, rec.get("median_tok_s"), rec.get("spread"),
            rec["during_probe"].get("util_gpu_p50"), rec["during_probe"].get("util_mem_p50"),
            rec["during_probe"].get("power_p50"), rec["during_probe"].get("clk_p50")))
    if not keep_up:
        rec["down_after"] = down(wait_mib, 240.0, boot + "-after")
    merge(boot, rec)
    return rec


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
    rec = boot_speed(args.boot, args.lengths, args.repeats, args.timeout,
                     args.wait_mib, args.no_down, args.keep_up)
    say("STATE %s" % json.dumps({k: rec.get(k) for k in (
        "boot", "boot_s", "health", "fatal", "attention_block_tokens", "kv_pool_tokens",
        "max_concurrency", "model_load_gib", "error_lines", "aux_layers_from_config",
        "v2_runner", "graph_mode", "median_tok_s", "spread", "during_probe", "pcie")},
        ensure_ascii=False))
    return 0 if rec.get("health") else 1


def cmd_show(args) -> int:
    rows = load()
    print("%-4s %6s %6s %6s %8s %6s %6s | %7s %6s %6s %7s %6s" % (
        "boot", "boot_s", "health", "block", "pool", "conc", "errs", "med_tps",
        "min-mx", "utilG", "utilM", "powerW"))
    for r in rows:
        d = r.get("during_probe") or {}
        print("%-4s %6s %6s %6s %8s %6s %6s | %7s %6s %6s %7s %6s" % (
            r.get("boot", ""), r.get("boot_s", ""), str(r.get("health")),
            r.get("attention_block_tokens", ""), r.get("kv_pool_tokens", ""),
            r.get("max_concurrency", ""), r.get("error_lines", ""),
            r.get("median_tok_s", ""),
            "-".join(str(x) for x in (r.get("spread") or [])[:2]) or "",
            d.get("util_gpu_p50", ""), d.get("util_mem_p50", ""),
            d.get("power_p50", ""), d.get("clk_p50", "")))
    return 0


def cmd_stop(args) -> int:
    merge("STOP", {"stopped": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "down": down(args.wait_mib, 240.0, "stop")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next")
    n.add_argument("--boot", required=True)
    n.add_argument("--lengths", type=int, nargs="+", default=[8000])
    n.add_argument("--repeats", type=int, default=3)
    n.add_argument("--timeout", type=float, default=420.0)
    n.add_argument("--wait-mib", type=float, default=800.0)
    n.add_argument("--keep-up", action="store_true")
    n.add_argument("--no-down", action="store_true")
    n.set_defaults(func=cmd_next)
    s = sub.add_parser("stop")
    s.add_argument("--wait-mib", type=float, default=800.0)
    s.set_defaults(func=cmd_stop)
    h = sub.add_parser("show")
    h.set_defaults(func=cmd_show)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
