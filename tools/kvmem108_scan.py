#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 108 KVMem pool x context scan driver.

The step-108 plan's batch A fixed W=131,072 / L=200,704 while lowering the
pool. That is not bootable below ~3.0e9: the engine's startup capacity check
needs at least one request at the configured sliding window, and
3.4e9/W=131,072 measured 1.17x (234,000 tokens). This driver walks the joint
(pool, W) frontier instead:

  gsq  main   VIEWPORT=0 LOAD=1  assembly + speed
  gsq  recall VIEWPORT=1 LOAD=0  step 074/078 out-of-window viewport protocol
  orca main   VIEWPORT=0 LOAD=1  assembly + speed (eager, no spec)
  orca recall VIEWPORT=1 LOAD=0  viewport protocol (unverified on Orca;
                                 record INCOMPLETE if the mechanism lines
                                 do not appear)

Two facts this driver is built around:

  * The page size is 16*cdiv(mamba_raw, 16*1152); it depends on spec N and
    the model geometry, not on SW_WINDOW. GSQ +spec N=2 -> page 1456 and
    mbt >= 1458 at every W. Orca no-spec -> page 1424.
  * The assembly path cannot answer an out-of-window needle (step 066 header
    says so). Recall beyond the window must use the viewport arm.

This tool only orchestrates; it does not import vllm and never starts Python
inside the record repo. Run it from G:\\qwen3.8model (or anywhere outside the
record repo):

  G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe \\
      G:\\qwen3.8model\\vllm-030win-git\\tools\\kvmem108_scan.py plan

  ... kvmem108_scan.py boot --arm gsq --tag s108_a1_base --pool 3400000000 \\
      --w 131072 --mbt 1458
  ... kvmem108_scan.py anchor --arm gsq --tag s108_a1_base --lengths 8000 180633
  ... kvmem108_scan.py recall --arm gsq --tag s108_a1_base --mode viewport \\
      --depths 0.45,0.55,0.65 --engine-log ...
  ... kvmem108_scan.py collect --tag s108_a1_base
  ... kvmem108_scan.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(r"G:\qwen3.8model")
PY = Path(r"G:\qwen3.8model\vllm-win029\Scripts\python.exe")
BOOT_KEEP = ROOT / "_tmp_line_b" / "boot_keep.py"
KILL_PS = REPO / "tools" / "kill_vllm_orphans.ps1"
HEADROOM_PS = REPO / "tools" / "prod_headroom_check.ps1"
ANCHOR = REPO / "tools" / "anchor_longctx.py"
VIEWPORT = REPO / "tools" / "kvmem_viewport_probe.py"
ASSEMBLY = REPO / "tools" / "kvmem_assembly_probe.py"
OUT_ROOT = ROOT / "prod029_logs" / "step108"

ARMS = {
    "gsq": REPO / "tools" / "serve_gsq_kvmem108_frontier.cmd",
    "orca": REPO / "tools" / "serve_orcasaq2_029_kvmem108.cmd",
    "gsq_serve": REPO / "tools" / "serve_gsq_serve108.cmd",
}
BASES = {
    "gsq": "http://127.0.0.1:8080",
    "orca": "http://127.0.0.1:8001",
    "gsq_serve": "http://127.0.0.1:8080",
}
MODELS = {
    "gsq": "qwen3.8-27b-gsq",
    "orca": "orcasaq2",
    "gsq_serve": "qwen3.8-27b-gsq",
}
PORTS = {"gsq": 8080, "orca": 8001, "gsq_serve": 8080}

# Paper frontier (step 108 plan section 2.1). Estimates are formula-derived and
# must be confirmed by the boot banner; the driver never boots a point whose
# estimated concurrency is below this floor.
MIN_EST_CONCURRENCY = 1.05

FRONTIER = {
    "gsq": [
        dict(pool=3_400_000_000, w=131_072, ml=200_704, mbt=1458, est=1.17, measured=True),
        dict(pool=2_600_000_000, w=98_304, ml=200_704, mbt=1458, est=1.16),
        dict(pool=2_000_000_000, w=65_536, ml=200_704, mbt=1458, est=1.21),
        dict(pool=1_600_000_000, w=49_152, ml=200_704, mbt=1458, est=1.20),
        dict(pool=1_200_000_000, w=32_768, ml=200_704, mbt=1458, est=1.16),
    ],
    "gsq_serve": [
        dict(pool=3_400_000_000, ml=163_072, est=1.00, measured=True),
        dict(pool=2_600_000_000, ml=117_000, est=1.01),
        dict(pool=2_000_000_000, ml=85_000, est=1.01),
        dict(pool=1_600_000_000, ml=63_000, est=1.02),
    ],
    "orca": [
        dict(pool=3_100_000_000, w=65_536, ml=131_072, mbt=1456, est=2.11, measured=True),
        dict(pool=2_000_000_000, w=65_536, ml=131_072, mbt=1456, est=1.22),
        dict(pool=1_800_000_000, w=65_536, ml=131_072, mbt=1456, est=1.10),
        dict(pool=1_400_000_000, w=49_152, ml=98_304, mbt=1456, est=1.05, optional=True),
    ],
}


def outdir(tag: str) -> Path:
    d = OUT_ROOT / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_logs(paths: list[Path]) -> str:
    text = ""
    for p in paths:
        try:
            text += p.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            pass
    return text


def parse_banner(text: str) -> dict:
    rec: dict = {}
    m = re.search(
        r"GPU KV cache size:\s*([\d,]+)\s*tokens,\s*Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x",
        text,
    )
    if m:
        rec["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        rec["max_concurrency_at"] = int(m.group(2).replace(",", ""))
        rec["max_concurrency"] = float(m.group(3))
    m = re.search(r"Setting attention block size to (\d+) tokens", text)
    if m:
        rec["page_tokens"] = int(m.group(1))
    m = re.search(r"cudagraph_mode':\s*<CUDAGraphMode\.(\w+)", text)
    if m:
        rec["cudagraph_mode"] = m.group(1)
    rec["aot_direct_loads"] = len(re.findall(r"Directly load AOT compilation", text))
    rec["aot_compiles"] = len(re.findall(r"Compiling a graph for compile range", text))
    rec["aot_cache_hit"] = rec["aot_compiles"] == 0
    rec["errors"] = [
        ln.strip() for ln in text.splitlines() if re.search(r"\bERROR\b", ln)
    ][:8]
    m = re.search(r"estimated maximum model length is (\d+)", text)
    if m:
        rec["engine_max_len"] = int(m.group(1))
    return rec


def parse_mechanism(text: str) -> dict:
    rec: dict = {}
    m = re.search(
        r"KVMem assembly: request .*?matches at ([\d,]+) tokens \((\d+) stored pages\)",
        text,
    )
    if m:
        rec["asm_matches_tokens"] = int(m.group(1).replace(",", ""))
        rec["asm_stored_pages"] = int(m.group(2))
    m = re.search(r"issued \((\d+) pages? \+ (\d+) mamba state blocks?\)", text)
    if m:
        rec["asm_issued_pages"] = int(m.group(1))
        rec["asm_mamba_blocks"] = int(m.group(2))
    rec["asm_prefix_landed"] = "prefix landed" in text
    rec["asm_mamba_registered"] = "snapshot region" in text
    rec["asm_miss_lines"] = [
        ln.strip() for ln in text.splitlines() if "asm-miss" in ln
    ][:8]
    rec["viewport_rewritten"] = bool(
        re.search(r"KVMem viewport: request .*?rewritten", text)
    )
    m = re.search(r"baked (\d+) page\(s\) into the retrieval slots", text)
    if m:
        rec["viewport_baked_pages"] = int(m.group(1))
    return rec


def nvidia_smi(query: str) -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"smi-fail:{exc}"


def gpu_used_mib() -> int | None:
    out = nvidia_smi("memory.used")
    m = re.match(r"\s*(\d+)", out)
    return int(m.group(1)) if m else None


def health(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def post_reset(base: str) -> bool:
    try:
        req = urllib.request.Request(base + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception as exc:  # noqa: BLE001
        print(f"[reset ] POST /reset_prefix_cache failed: {exc}")
        return False


def kill_and_wait(wait_mib: int = 800, timeout_s: int = 180) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-File", str(KILL_PS)],
        capture_output=True,
        text=True,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = gpu_used_mib()
        if used is not None and used < wait_mib:
            return
        time.sleep(3)
    used = gpu_used_mib()
    raise SystemExit(f"GPU did not fall below {wait_mib} MiB (last used={used})")


def boot(args: argparse.Namespace) -> dict:
    tag = args.tag
    d = outdir(tag)
    outlog = d / "engine.out.log"
    errlog = d / "engine.err.log"
    pages = args.pages if args.pages is not None else (40 if args.arm == "orca" else 55)
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("S108_"):
            del env[k]
    env.update(
        {
            "S108_TAG": tag,
            "S108_KVMB": str(args.pool),
            "S108_W": str(args.w),
            "S108_ML": str(args.ml),
            "S108_MBT": str(args.mbt),
            "S108_WSMB": str(args.wsmb),
            "S108_LOAD": str(args.load),
            "S108_VIEWPORT": str(args.viewport),
            "S108_PAGES": str(pages),
            "S108_PICK": args.pick,
            "S108_SPEC": str(args.spec),
            "S108_CAP": str(args.cap),
        }
    )
    cmd = [str(PY), str(BOOT_KEEP), str(ARMS[args.arm]), str(outlog), str(errlog)]
    if args.dry_run:
        print("[dry-run] env S108_*:", {k: v for k, v in env.items() if k.startswith("S108_")})
        print("[dry-run] cmd:", cmd)
        return {"dry_run": True}

    if not args.no_kill:
        kill_and_wait(args.wait_mib)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"boot_keep failed rc={r.returncode}: {r.stderr[-400:]}")

    base = BASES[args.arm]
    deadline = t0 + args.timeout
    up = False
    while time.time() < deadline:
        time.sleep(5)
        if health(base):
            up = True
            break
        text = read_logs([errlog, outlog])
        if "EngineCore failed to start" in text or "ValueError: To serve at least one request" in text:
            break
    boot_s = round(time.time() - t0, 1)
    text = read_logs([errlog, outlog])
    rec = {
        "tag": tag,
        "arm": args.arm,
        "boot_s": boot_s,
        "health": up,
        "pool": args.pool,
        "w": args.w,
        "ml": args.ml,
        "mbt": args.mbt,
        "load": args.load,
        "viewport": args.viewport,
        "pages": pages,
        "pick": args.pick,
        "spec": args.spec,
        "cap": args.cap,
        "engine_out_log": str(outlog),
        "engine_err_log": str(errlog),
        "smi_after_boot": nvidia_smi("memory.used,memory.total"),
        **parse_banner(text),
        **parse_mechanism(text),
    }
    rec["s108_env"] = {k: v for k, v in env.items() if k.startswith("S108_")}
    rec["shape_ok"] = (args.load == 1) == bool(rec.get("asm_mamba_registered"))
    (d / "boot.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if not up:
        raise SystemExit(1)
    return rec


def anchor(args: argparse.Namespace) -> dict:
    d = outdir(args.tag)
    out = d / "anchor.json"
    cmd = [
        str(PY),
        str(ANCHOR),
        "--base",
        BASES[args.arm],
        "--model",
        MODELS[args.arm],
        "--lengths",
        *[str(x) for x in args.lengths],
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--arm",
        args.tag,
        "--out",
        str(out),
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rec = {"tag": args.tag, "anchor_rc": r.returncode, "out": str(out)}
    if out.exists():
        rec["anchor"] = json.loads(out.read_text(encoding="utf-8"))
    else:
        rec["stderr_tail"] = (r.stderr or "")[-800:]
    (d / "anchor_summary.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return rec


def recall(args: argparse.Namespace) -> dict:
    if args.arm == "gsq_serve":
        raise SystemExit("the pure-serve control arm has no KVMem recall path")
    if args.head_tokens is None:
        if args.arm == "orca":
            raise SystemExit(
                "Orca viewport head is not calibrated (W=65,536, pages=40 would "
                "put head+recent above the window); pass --head-tokens and --w"
            )
        args.head_tokens = 81_536  # GSQ W=131072 + 55 pages at page 1456
    if args.w and args.head_tokens + args.recent_tokens > args.w:
        raise SystemExit(
            f"head {args.head_tokens} + recent {args.recent_tokens} exceeds "
            f"window {args.w}; the viewport probe would not be in-window"
        )
    d = outdir(args.tag)
    base = BASES[args.arm]
    engine_text = read_logs([d / "engine.out.log", d / "engine.err.log"])
    rec: dict = {
        "tag": args.tag,
        "arm": args.arm,
        "mode": args.mode,
        "reset_ok": None,
        "legs": [],
        "before": parse_mechanism(engine_text),
    }
    if args.mode == "viewport":
        for depth in args.depths:
            if args.reset:
                rec["reset_ok"] = post_reset(base)
                if rec["reset_ok"] is False:
                    raise SystemExit("prefix-cache reset failed (dev mode must be on)")
            out = d / f"viewport_d{depth:.2f}.json"
            cmd = [
                str(PY),
                str(VIEWPORT),
                "run",
                "--base",
                base,
                "--tokens",
                str(args.tokens),
                "--depth",
                str(depth),
                "--head-tokens",
                str(args.head_tokens),
                "--recent-tokens",
                str(args.recent_tokens),
                "--tag",
                f"{args.tag}_d{depth:.2f}",
                "--out",
                str(out),
            ]
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            leg = {"depth": depth, "rc": r.returncode, "out": str(out)}
            if out.exists():
                leg["probe"] = json.loads(out.read_text(encoding="utf-8"))
            else:
                leg["stderr_tail"] = (r.stderr or "")[-800:]
            rec["legs"].append(leg)
    elif args.mode == "assembly":
        out = d / "assembly.json"
        cmd = [
            str(PY),
            str(ASSEMBLY),
            "run",
            "--base",
            base,
            "--tokens",
            str(args.tokens),
            "--depth",
            str(args.depths[0]),
            "--tail-tokens",
            str(args.tail_tokens),
            "--tag",
            args.tag,
            "--out",
            str(out),
        ]
        if args.reset:
            cmd.append("--reset-prefix-cache")
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        leg = {"depth": args.depths[0], "rc": r.returncode, "out": str(out)}
        if out.exists():
            leg["probe"] = json.loads(out.read_text(encoding="utf-8"))
        else:
            leg["stderr_tail"] = (r.stderr or "")[-800:]
        rec["legs"].append(leg)
    else:
        raise SystemExit("--mode must be viewport or assembly")

    engine_text = read_logs([d / "engine.out.log", d / "engine.err.log"])
    rec["after"] = parse_mechanism(engine_text)
    (d / f"recall_{args.mode}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def collect(args: argparse.Namespace) -> dict:
    d = outdir(args.tag)
    rec: dict = {
        "tag": args.tag,
        "smi": nvidia_smi(
            "memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,"
            "temperature.gpu,pcie.link.gen.current,pcie.link.width.current"
        ),
    }
    if args.headroom:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(HEADROOM_PS)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        rec["headroom_rc"] = r.returncode
        rec["headroom_out"] = r.stdout.strip()
    (d / "collect.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def sample(args: argparse.Namespace) -> dict:
    d = outdir(args.tag)
    rows = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        rows.append(
            {
                "t": round(time.time() - (deadline - args.seconds), 2),
                "smi": nvidia_smi(
                    "memory.used,utilization.gpu,utilization.memory,power.draw,"
                    "temperature.gpu,pcie.link.gen.current,pcie.link.width.current"
                ),
            }
        )
        time.sleep(1.0)
    rec = {"tag": args.tag, "rows": rows}
    (d / "sample.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def status(_: argparse.Namespace) -> dict:
    rec = {
        "gsq_health": health(BASES["gsq"]),
        "orca_health": health(BASES["orca"]),
        "smi": nvidia_smi("memory.used,memory.total,utilization.gpu"),
    }
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def stop(args: argparse.Namespace) -> dict:
    kill_and_wait(args.wait_mib)
    rec = {"stopped": True, "smi": nvidia_smi("memory.used,memory.total")}
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def plan(_: argparse.Namespace) -> dict:
    print("paper frontier (boot banner is authoritative; est < %.2f is forbidden)"
          % MIN_EST_CONCURRENCY)
    for name, rows in FRONTIER.items():
        print(f"\n[{name}]")
        for r in rows:
            flag = "MEASURED" if r.get("measured") else ("OPTIONAL" if r.get("optional") else "EST")
            if name == "gsq_serve":
                flag += " SERVE-CTRL"
            elif r["est"] < MIN_EST_CONCURRENCY:
                flag += " BELOW-FLOOR"
            print(
                f"  pool={r['pool']:>12,}  w={r.get('w', '-'):>7}  "
                f"ml={r['ml']:>7,}  est={r['est']:.2f}x  {flag}"
            )
    return {"frontier": FRONTIER}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.set_defaults(func=plan)

    p = sub.add_parser("boot")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--pool", type=int, required=True)
    p.add_argument("--w", type=int, default=0,
                   help="sliding window; 0 for the pure-serve control arm")
    p.add_argument("--ml", type=int, required=True)
    p.add_argument("--mbt", type=int, required=True)
    p.add_argument("--wsmb", type=int, default=5120)
    p.add_argument("--load", type=int, default=1)
    p.add_argument("--viewport", type=int, default=0)
    p.add_argument("--pages", type=int, default=None,
                   help="default 55 (GSQ) / 40 (Orca)")
    p.add_argument("--pick", default="score")
    p.add_argument("--spec", type=int, default=1)
    p.add_argument("--cap", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--wait-mib", type=int, default=800)
    p.add_argument("--no-kill", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=boot)

    p = sub.add_parser("anchor")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--lengths", type=int, nargs="+", required=True)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(func=anchor)

    p = sub.add_parser("recall")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=("viewport", "assembly"), required=True)
    p.add_argument("--depths", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    p.add_argument("--tokens", type=int, default=200_000)
    p.add_argument("--w", type=int, default=0,
                   help="sliding window, used to check head+recent <= W")
    p.add_argument("--head-tokens", type=int, default=None,
                   help="GSQ default 81,536 = (1+55)*1456; Orca requires an explicit calibrated value")
    p.add_argument("--recent-tokens", type=int, default=16_384)
    p.add_argument("--tail-tokens", type=int, default=768)
    p.add_argument("--reset", action="store_true")
    p.set_defaults(func=recall)

    p = sub.add_parser("collect")
    p.add_argument("--tag", required=True)
    p.add_argument("--headroom", action="store_true")
    p.set_defaults(func=collect)

    p = sub.add_parser("sample")
    p.add_argument("--tag", required=True)
    p.add_argument("--seconds", type=int, default=60)
    p.set_defaults(func=sample)

    p = sub.add_parser("status")
    p.set_defaults(func=status)

    p = sub.add_parser("stop")
    p.add_argument("--wait-mib", type=int, default=800)
    p.set_defaults(func=stop)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
