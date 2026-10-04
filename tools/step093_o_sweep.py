"""step093_o_sweep.py -- O-line debt-sweep driver (step 093).

Three booked debts from 089/090/092, one driver:
  A) same-gauge comparison of the two Orca speculation tiers -- 086 MTP (nvfp4 KV,
     spec=1) vs 090 DFlash2 healthy tier -- on the production 8k anchor gauge
     (anchor_longctx.py, the gauge behind GSQ production's 122.58);
  B) the L-speed knee of the DFlash2 tier: pool-byte sweep at fixed L, then an L
     sweep with the pool scaled as need(L)*1.2 (092 profile-card derivation);
  C) soak + multi-turn on the healthy tier (driven by tools/soak_orca.py, which
     keeps the 052 finish_reason health semantics).

Per boot: kill + wait for the dGPU to drop -> launch the launcher named by
S093_ARM (default = the prodcap DFlash2 arm, knobs S089_* passed through) ->
wait /health -> parse capacity banners and self-attestation -> run the 8k anchor
probe while sampling nvidia-smi at 1 Hz out-of-band -> append a row to
prod029_logs/step093_boots.json. Speed claims still need >=3 boots per tier
(iron rule 7); single boots are recorded as "that boot's state".

Usage (venv python, cwd OUTSIDE the repo -- iron rule 2):
  cd G:\\qwen3.8model
  G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe ^
      G:\\qwen3.8model\\vllm-030win-git\\tools\\step093_o_sweep.py next --boot d1
  ... next --boot m1 (with S093_ARM=...nvfp4_mtp.cmd) / p1 (S089_POOL=1400000000) ...
  tools/step093_o_sweep.py show
  tools/step093_o_sweep.py stop
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import time

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
EVID = os.environ.get("S093_EVID_DIR", os.path.join(LOGS, "orca_o3", "step093"))
STATE = os.environ.get("S093_STATE_FILE", os.path.join(LOGS, "step093_boots.json"))
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
BASE = "http://127.0.0.1:8001"
MODEL = "orcasaq2"
DEFAULT_ARM = os.path.join(TOOLS, "serve_orcasaq2_029_nvfp4_dflash2_prodcap.cmd")


def _load_step090():
    spec = importlib.util.spec_from_file_location(
        "step090_orca_speed", os.path.join(TOOLS, "step090_orca_speed.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s090 = _load_step090()
say = s090.say
smi_row = s090.smi_row
pcie_row = s090.pcie_row
Sampler = s090.Sampler
gpu_used = s090.gpu_used
down = s090.down
merge = None  # replaced below; STATE differs


def load() -> list:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save(rows: list) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def merge(boot: str, patch: dict) -> None:
    rows = load()
    row = next((r for r in rows if r.get("boot") == boot), None)
    if row is None:
        row = {"boot": boot}
        rows.append(row)
    row.update(patch)
    save(rows)


def health() -> bool:
    # s090.get hardcodes the same BASE (127.0.0.1:8001), so it is directly reusable.
    status, _ = s090.get("/health", 10.0)
    return status == 200


def run_anchor(boot: str, lengths: list, repeats: int) -> dict:
    out_json = os.path.join(EVID, "%s_anchor.json" % boot)
    cmd = [PY, os.path.join(TOOLS, "anchor_longctx.py"), "--base", BASE,
           "--model", MODEL, "--lengths", *[str(x) for x in lengths],
           "--warmup", "0", "--repeats", str(repeats),
           "--arm", "orca_s93_%s" % boot, "--out", out_json]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRATCH,
                          timeout=5400)
    tail = (proc.stdout or "")[-2500:] + (proc.stderr or "")[-500:]
    print(tail, flush=True)
    rec: dict = {"json": out_json, "rc": proc.returncode}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("results") or []
        rec["rows"] = [row for e in entries for row in (e.get("repeats") or [])]
        rec["agg"] = [{"target_tokens": e.get("target_tokens"),
                       "steady_tok_s": e.get("steady_tok_s"),
                       "ttft_s": e.get("ttft_s"),
                       "prompt_tokens": e.get("prompt_tokens"),
                       "needle_hit": e.get("needle_hit")} for e in entries]
    return rec


def boot_sweep(boot: str, lengths: list, repeats: int, timeout: float,
               wait_mib: float, no_down: bool, keep_up: bool) -> dict:
    arm = os.environ.get("S093_ARM", DEFAULT_ARM)
    os.makedirs(EVID, exist_ok=True)
    out_log = os.path.join(EVID, "%s.out.log" % boot)
    err_log = os.path.join(EVID, "%s.err.log" % boot)
    knobs = {k: v for k, v in os.environ.items()
             if k.startswith(("S089_", "S086_", "S093_"))}
    rec: dict = {"boot": boot, "arm": arm, "out_log": out_log,
                 "err_log": err_log, "env": knobs,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not no_down:
        rec["down"] = down(wait_mib, 240.0, boot)
    say("boot %s: launching %s (%s)" % (boot, os.path.basename(arm),
                                        " ".join("%s=%s" % kv for kv in sorted(knobs.items()))))
    with open(out_log, "w", encoding="utf-8", errors="replace") as fo, \
            open(err_log, "w", encoding="utf-8", errors="replace") as fe:
        subprocess.Popen([PY, BOOT_KEEP, arm, out_log, err_log], cwd=SCRATCH)
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
            if "EngineCore failed to start" in blob or \
                    "Traceback (most recent call last)" in blob:
                rec["fatal"] = True
                break
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["health"] = health()
    rec.update(s090.parse_logs(out_log, err_log))
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
        rec["needle_hits"] = [bool(r.get("needle_hit"))
                              for r in (rec["anchor"].get("rows") or [])]
        rec["median_tok_s"] = statistics.median(rates) if rates else None
        rec["spread"] = (round(min(rates), 2), round(max(rates), 2)) if rates else None
        say("boot %s: median_tok_s=%s spread=%s util(g/m)=%s/%s power=%sW" % (
            boot, rec.get("median_tok_s"), rec.get("spread"),
            rec["during_probe"].get("util_gpu_p50"),
            rec["during_probe"].get("util_mem_p50"),
            rec["during_probe"].get("power_p50")))
    if not keep_up:
        rec["down_after"] = down(wait_mib, 240.0, boot + "-after")
    merge(boot, rec)
    return rec


def cmd_next(args) -> int:
    rec = boot_sweep(args.boot, args.lengths, args.repeats, args.timeout,
                     args.wait_mib, args.no_down, args.keep_up)
    say("STATE %s" % json.dumps(
        {k: rec.get(k) for k in ("boot", "boot_s", "health", "fatal",
                                 "attention_block_tokens", "kv_pool_tokens",
                                 "max_concurrency", "model_load_gib", "error_lines",
                                 "aux_layers_from_config", "v2_runner",
                                 "draft_resolved_arch", "graph_mode",
                                 "median_tok_s", "spread", "during_probe")},
        ensure_ascii=False))
    return 0 if rec.get("health") else 1


def cmd_show(args) -> int:
    rows = load()
    print("%-6s %6s %6s %6s %9s %6s %5s | %8s %11s %6s %6s %7s" % (
        "boot", "boot_s", "health", "block", "pool", "conc", "errs",
        "med_tps", "min-max", "utilG", "utilM", "powerW"))
    for r in rows:
        d = r.get("during_probe") or {}
        print("%-6s %6s %6s %6s %9s %6s %5s | %8s %11s %6s %6s %7s" % (
            r.get("boot", ""), r.get("boot_s", ""), str(r.get("health")),
            r.get("attention_block_tokens", ""), r.get("kv_pool_tokens", ""),
            r.get("max_concurrency", ""), r.get("error_lines", ""),
            r.get("median_tok_s", ""),
            "-".join(str(x) for x in (r.get("spread") or [])[:2]) or "",
            d.get("util_gpu_p50", ""), d.get("util_mem_p50", ""),
            d.get("power_p50", "")))
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
