# -*- coding: utf-8 -*-
"""Context-lever arm driver: boot one launcher variant, measure, shut down.

Per arm:
  1. derive the launcher from the production one via mk_arm.py
  2. kill leftovers, boot, poll /health
  3. snapshot nvidia-smi (used + shared) right after health
  4. scrape the engine's own "GPU KV cache size" line
  5. run anchor_longctx on the requested lengths (ttft / steady / needle)
  6. shut down

Usage (run from anywhere; the launcher cwd is forced to the model dir):
  python ctx_arm.py --arms "G8L140000:8:140000, G1L144432:1:144432" \
      --lengths 8000 --out-tag sweep1
Arm spec: name:g:len:n:mbt  (empty field = leave production value)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = r"G:\qwen3.8model\_tmp_line_b"
BLOG = os.path.join(ROOT, "bootlogs")
WORKDIR = r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
ANCHOR = r"G:\qwen3.8model\vllm-030win-git\tools\anchor_longctx.py"
MK = os.path.join(ROOT, "mk_arm.py")
BASE = "http://127.0.0.1:8080"

KILL_PS = (
    "$me = {pid}; Get-CimInstance Win32_Process | Where-Object {{ "
    "$_.ProcessId -ne $me -and $_.CommandLine -match "
    "'multiprocessing\\.spawn|vllm\\.entrypoints|serve_arm_|serve_kvgroup' "
    "}} | ForEach-Object {{ try {{ Stop-Process -Id $_.ProcessId "
    "-Force -ErrorAction Stop }} catch {{}} }}"
)


def log(path, msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kill():
    # Exclude this process: the --launcher argument can itself contain one of
    # the match strings (e.g. serve_gsq_prod029_n2.cmd), which would otherwise
    # make the driver kill itself.
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    KILL_PS.format(pid=os.getpid())], capture_output=True)


def health_up(timeout=3):
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception as e:
        return "smi-fail:%s" % e


def parse_arm(spec):
    parts = spec.split(":")
    while len(parts) < 5:
        parts.append("")
    name = parts[0]
    kw = {}
    if parts[1].strip():
        kw["--g"] = parts[1].strip()
    if parts[2].strip():
        kw["--len"] = parts[2].strip()
    if parts[3].strip():
        kw["--n"] = parts[3].strip()
    if parts[4].strip():
        kw["--mbt"] = parts[4].strip()
    return name, kw


def run_arm(spec, lengths, tag, logf, timeout_s, skip_probe, repeats,
            launcher_override=None):
    name, kw = parse_arm(spec)
    if launcher_override:
        launcher = launcher_override
    else:
        launcher = os.path.join(ROOT, "serve_arm_%s_%s.cmd" % (name, tag))
        cmd = [PY, MK, launcher] + [x for kv in kw.items() for x in kv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(logf, "arm=%s MK FAIL: %s" % (name, r.stderr.strip()[:300]))
            return None

    kill()
    time.sleep(6)
    outl = os.path.join(BLOG, "arm_%s_%s.out.log" % (name, tag))
    errl = os.path.join(BLOG, "arm_%s_%s.err.log" % (name, tag))
    fo = open(outl, "w", encoding="utf-8", errors="replace")
    fe = open(errl, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(["cmd.exe", "/d", "/c", launcher], cwd=WORKDIR,
                            stdout=fo, stderr=fe,
                            creationflags=subprocess.CREATE_NO_WINDOW)

    up = False
    deadline = t0 + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        if health_up():
            up = True
            break
        # Fast-fail: a rejected config writes its fatal error and exits, so
        # waiting out the full timeout would waste ~25 min per bad arm. The
        # marker lands in stdout ("EngineCore failed to start") while the
        # ValueError traceback lands in stderr -- check both.
        try:
            _t = (open(errl, encoding="utf-8", errors="replace").read()
                  + open(outl, encoding="utf-8", errors="replace").read())
        except Exception:
            _t = ""
        if ("EngineCore failed to start" in _t
                or "ValueError: To serve at least one request" in _t):
            break
    boot_s = round(time.time() - t0, 1)
    rec = {"arm": name, "spec": spec, "boot_s": boot_s, "health": up}

    if not up:
        try:
            full = (open(errl, encoding="utf-8", errors="replace").read()
                    + open(outl, encoding="utf-8", errors="replace").read())
        except Exception:
            full = ""
        m = re.search(r"estimated maximum model length is (\d+)", full)
        rec["engine_max_len"] = int(m.group(1)) if m else None
        errs = [ln.strip() for ln in full.splitlines() if "ValueError" in ln]
        rec["value_error"] = errs[-1][:400] if errs else None
        rec["err_tail"] = full[-1200:]
        log(logf, "arm=%s health FAIL boot_s=%s engine_max_len=%s" % (
            name, boot_s, rec["engine_max_len"]))
        if rec["value_error"]:
            log(logf, "  %s" % rec["value_error"])
        fo.close()
        fe.close()
        kill()
        time.sleep(6)
        return rec

    txt = ""
    for p in (errl, outl):
        try:
            txt += open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
    rec["kv_cache_size"] = int(m[-1].replace(",", "")) if m else None
    m2 = re.findall(r"layers-per-group (\d+) -> (\d+)", txt)
    rec["group_size"] = m2[-1] if m2 else None
    m3 = re.findall(r"Maximum concurrency for ([\d,]+) tokens", txt)
    rec["max_concurrency"] = m3[-1] if m3 else None
    warns = [ln.strip() for ln in txt.splitlines()
             if re.search(r"max_num_scheduled_tokens|suboptimal|WARNING", ln)]
    rec["warnings"] = warns[:6]
    rec["smi_after_boot"] = smi()
    log(logf, "arm=%s boot_s=%s kv=%s group=%s smi=%s" % (
        name, boot_s, rec["kv_cache_size"], rec["group_size"],
        rec["smi_after_boot"]))

    if not skip_probe:
        outp = os.path.join(BLOG, "arm_%s_%s.anchor.json" % (name, tag))
        r = subprocess.run(
            [PY, ANCHOR, "--base", BASE, "--lengths"] + [str(x) for x in lengths]
            + ["--out", outp, "--arm", name, "--warmup", "0",
               "--repeats", str(repeats)],
            capture_output=True, text=True, cwd=ROOT)
        log(logf, "arm=%s anchor rc=%s" % (name, r.returncode))
        if r.returncode != 0:
            log(logf, "  anchor err: %s" % (r.stderr or "")[-400:])
        rec["smi_after_probe"] = smi()
        if os.path.exists(outp):
            d = json.load(open(outp, encoding="utf-8"))
            rec["probe"] = d.get("results")
            for res in d.get("results", []):
                for rep in res.get("repeats", []):
                    log(logf, "  len=%s prompt=%s ttft=%.3f steady=%.2f "
                        "hit=%s" % (res["target_tokens"],
                                    rep.get("prompt_tokens"),
                                    rep.get("ttft_s", -1),
                                    rep.get("steady_tok_s", -1),
                                    rep.get("needle_hit")))

    fo.close()
    fe.close()
    kill()
    time.sleep(6)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--lengths", default="8000")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--launcher",
                    help="boot this launcher verbatim instead of deriving one")
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()

    os.makedirs(BLOG, exist_ok=True)
    logf = os.path.join(BLOG, "ctx_arm_%s.log" % a.out_tag)
    lengths = [int(x) for x in a.lengths.split(",")]
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    log(logf, "=== ctx_arm begin tag=%s arms=%s lengths=%s ===" % (
        a.out_tag, arms, lengths))

    results = []
    for spec in arms:
        rec = run_arm(spec, lengths, a.out_tag, logf, a.timeout,
                      a.skip_probe, a.repeats, a.launcher)
        if rec:
            results.append(rec)
        json.dump(results, open(os.path.join(BLOG, "ctx_arm_%s.json" % a.out_tag),
                                "w"), ensure_ascii=False, indent=1)
    log(logf, "=== ctx_arm done tag=%s ===" % a.out_tag)


if __name__ == "__main__":
    main()
