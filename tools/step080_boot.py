# step080_boot.py -- one-boot driver for the step 080 slow-state timing hunt.
#
# What it automates (and why each piece is there):
#   down      kill via the FILE-BASED ps1 (iron rule 10: chained bash -> inline
#             PowerShell crashes), then poll nvidia-smi until the discrete GPU
#             falls below --wait-mib (iron rule 18: killing and booting 5 s
#             apart got four consecutive boots evicted).
#   up        spawn the 079 arm verbatim through _tmp_line_b/boot_keep.py (iron
#             rule 2: never start python with the record repo as cwd), poll
#             /health, fast-fail on a fatal banner, then CHECK THE BANNERS
#             (Directly load / block size 1456 / 200 host slots / [KVTIME] patch
#             loaded / GPU KV cache size / ERROR+Traceback counts) -- iron rule
#             24 (i): a gated log must prove it is running.
#   classify  a SHORT cadence probe (16k = 11 page steps). Fast ~16 s, slow
#             ~100 s, so triage is cheap and one hour of wall clock buys ~12
#             boots instead of 079's 6. Bands are pre-declared: fast <= 2.0
#             s/page step, slow >= 5.0, anything between is written as `mid`
#             (iron rule 7: this is a continuous spectrum, do not force-fit).
#   measure   on a slow boot: the out-of-band fingerprint sampler + the main
#             cadence + the 8k decode anchor (the pairing 078/079 need for the
#             prefill/decode anti-correlation).
#   vp        one viewport probe (step 080 item 3, depth x nonce grid).
#
# Every boot MUST get a fresh dump dir (iron rule 16 (viii): kvmem_retrieval_%03d
# restarts at 001 each boot, so reusing a directory silently overwrites
# evidence). `--tag` is that directory name; `next` refuses a tag that already
# exists on disk.
#
# Run it with the venv python from a scratch cwd, e.g.
#   cd G:\qwen3.8model
#   G:\qwen3.8model\vllm-win029\Scripts\python.exe ^
#       G:\qwen3.8model\vllm-030win-git\tools\step080_boot.py next --boot b1 --tag kvmem_k14a
#
# State file: G:\qwen3.8model\prod029_logs\step080_boots.json (append-only per
# boot, so the final table can be rebuilt offline).

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
ARM = os.path.join(TOOLS, "serve_gsq_kvmem_viewport079_spec.cmd")
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
KILL_PS1 = os.path.join(TOOLS, "kill_vllm_orphans.ps1")
CADENCE = os.path.join(TOOLS, "kvmem_ingest_cadence.py")
ANCHOR = os.path.join(TOOLS, "anchor_longctx.py")
VPROBE = os.path.join(TOOLS, "kvmem_viewport_probe.py")
STATE = os.path.join(LOGS, "step080_boots.json")
BASE = "http://127.0.0.1:8080"
FAST_MAX = 2.0     # s per 1456-token page step
SLOW_MIN = 5.0

sys.path.insert(0, TOOLS)
from kvmem_boot_fingerprint import Sampler  # noqa: E402


def say(msg: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gpu_used() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True,
                             timeout=20).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return -1.0


def health() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def read_tail(path: str, limit: int = 400_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:]
    except Exception:
        return ""


def down(wait_mib: float = 800.0, timeout: float = 240.0, label: str = "") -> dict:
    rec = {"down_at": time.strftime("%H:%M:%S")}
    subprocess.run(["powershell", "-NoProfile", "-File", KILL_PS1],
                   capture_output=True, text=True)
    t0 = time.time()
    used = gpu_used()
    # gpu_used() returns -1 when nvidia-smi is unreadable; treating that as
    # "already settled" would boot straight into the eviction window (rule 18),
    # so an unknown reading keeps polling.
    while (used < 0 or used > wait_mib) and time.time() - t0 < timeout:
        time.sleep(5)
        used = gpu_used()
    rec["gpu_used_mib"] = used
    rec["settled_s"] = round(time.time() - t0, 1)
    rec["label"] = label
    say("down%s: gpu_used=%.0f MiB after %.0f s" % (label, used, time.time() - t0))
    return rec


BANNERS = [
    ("attention_block", r"Setting attention block size to (\d+) tokens"),
    ("directly_load", r"Directly load the compiled graph\(s\).*?took ([\d.]+) s"),
    ("compiling", r"Compiling a graph for compile range \(([^)]*)\) takes ([\d.]+) s"),
    ("kv_cache", r"GPU KV cache size: ([\d,]+) tokens"),
    ("host_slots", r"(\d+) host slots \(([\d.]+) GiB budget"),
    ("kvtime_loaded", r"\[KVTIME\] patch loaded: (.*?)\n"),
    ("model_load", r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    # Host RAM at weight-load time: 049 measured this once for the DECODE band
    # ("Available RAM 与档位不相关"), and 077-079 slow/fast boots show it does
    # not separate them either (13.57/13.34 slow vs 11.72/12.41 fast). Kept as
    # a per-boot control, not as a hypothesis.
    ("avail_ram_gb", r"Checkpoint size: [\d.]+ GiB\. Available RAM: ([\d.]+) GiB"),
    ("gpu_initial_free_gb", r"Initial free memory ([\d.]+) GiB"),
]


def _val(found, index: int = 0, field: int = 0):
    """re.findall gives str for one group and tuple for several."""
    if not found:
        return None
    item = found[index]
    return item if isinstance(item, str) else item[field]


def verify(out_log: str, err_log: str) -> dict:
    text = read_tail(out_log, limit=2_000_000) + read_tail(err_log, limit=500_000)
    rec: dict = {}
    for name, pattern in BANNERS:
        found = re.findall(pattern, text)
        if name == "directly_load":
            rec["directly_load_n"] = len(found)
            rec["directly_load_first_s"] = _val(found, 0, 1)
        elif name == "compiling":
            # non-zero = the AOT key moved (iron rule 17): such a boot sits in a
            # degraded state and must never carry a speed verdict.
            rec["compiling_n"] = len(found)
        elif name == "host_slots":
            rec["host_slots"] = _val(found, 0, 0)
            rec["host_slots_gib"] = _val(found, 0, 1)
        elif name == "kv_cache":
            rec["kv_cache_tokens"] = _val(found, -1, 0)
        elif name == "attention_block":
            rec["attention_block"] = _val(found, 0, 0)
        elif name == "model_load":
            rec["model_load_gib"] = _val(found, -1, 0)
            rec["model_load_s"] = _val(found, -1, 1)
        elif name == "kvtime_loaded":
            rec["kvtime_loaded"] = bool(found)
            rec["kvtime_head"] = found[0][:220] if found else None
        elif name == "avail_ram_gb":
            # host RAM at weight-load time, one line per checkpoint shard; kept
            # as a per-boot control (049 found it did NOT separate the bands)
            rec["avail_ram_gb"] = list(found)[:4]
        else:
            rec[name] = _val(found, 0, 0)
    # torch's cpp_extension probe prints a "[rank0]:W ... Traceback (most recent
    # call last)" banner on EVERY boot of this arm. 077-079 counted clean logs by
    # reading the engine stream, so exclude that benign warning prefix rather
    # than calling every boot dirty -- but keep the count visible in
    # warn_traceback_lines so a real change cannot hide behind the filter.
    lines = text.splitlines()
    hard = [ln for ln in lines if "[rank0]:W" not in ln]
    rec["error_lines"] = sum(1 for ln in hard if re.search(r"\bERROR\b", ln))
    rec["traceback_lines"] = sum(
        1 for ln in hard if "Traceback (most recent call" in ln)
    rec["warn_traceback_lines"] = sum(
        1 for ln in lines if "Traceback (most recent call" in ln) - rec["traceback_lines"]
    rec["error_samples"] = [ln.strip()[:200] for ln in hard
                            if re.search(r"\bERROR\b", ln)][:5]
    rec["pageable_fallback"] = bool(re.search(r"falling back to pageable", text))
    return rec


def up(tag: str, boot: str, every: int, extra: dict, timeout: float) -> dict:
    out_log = os.path.join(LOGS, "step080_%s_arm.out.log" % boot)
    err_log = os.path.join(LOGS, "step080_%s_arm.err.log" % boot)
    env = dict(os.environ)
    env["S079_TAG"] = tag
    env["S079_EVERY"] = str(every)
    env["S079_TIMING"] = "1"
    for key, val in extra.items():
        env[key] = str(val)
    fo = open(out_log, "w", encoding="utf-8", errors="replace")
    fe = open(err_log, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    subprocess.Popen([PY, BOOT_KEEP, ARM, out_log, err_log], cwd=SCRATCH,
                     env=env)
    rec = {"boot": boot, "tag": tag, "out_log": out_log, "err_log": err_log,
           "up_at": time.strftime("%H:%M:%S")}
    while time.time() - t0 < timeout:
        time.sleep(5)
        if health():
            break
        # Fast-fail on a real fatal, but NOT on torch's benign
        # "[rank0]:W ... Traceback (most recent call last)" cpp_extension probe
        # banner -- the first boot of this driver tripped on it and abandoned a
        # perfectly healthy boot at 49 s.
        blob = "\n".join(
            ln for ln in (read_tail(err_log, 200_000)
                          + read_tail(out_log, 400_000)).splitlines()
            if "[rank0]:W" not in ln)
        if ("EngineCore failed to start" in blob
                or "Traceback (most recent call" in blob):
            rec["fatal"] = True
            break
    rec["boot_s"] = round(time.time() - t0, 1)
    rec["health"] = health()
    rec.update(verify(out_log, err_log))
    say("up %s/%s: boot_s=%s health=%s load_n=%s compiling_n=%s block=%s slots=%s "
        "kvtime=%s err=%s tb=%s" % (
            boot, tag, rec["boot_s"], rec["health"], rec.get("directly_load_n"),
            rec.get("compiling_n"), rec.get("attention_block"),
            rec.get("host_slots"), rec.get("kvtime_loaded"),
            rec.get("error_lines"), rec.get("traceback_lines")))
    fo.close()
    fe.close()
    return rec


def load_state() -> list:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as handle:
            return json.load(handle)
    return []


def save_state(rows: list) -> None:
    with open(STATE, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def merge_boot(boot: str, patch: dict) -> dict:
    """Merge patch into the row for this boot (never drop earlier fields)."""
    rows = load_state()
    row = next((r for r in rows if r.get("boot") == boot), None)
    if row is None:
        row = {"boot": boot}
        rows.append(row)
    row.update(patch)
    rows.sort(key=lambda r: r.get("boot", ""))
    save_state(rows)
    return row


def classify(boot: str, tokens: int, out_log: str, nonce: str = "") -> dict:
    # A repeated identical prompt would native-hit the prefix cache and print a
    # fake cadence, so every triage run gets its own nonce (iron rule 22 (ii)).
    nonce = nonce or "s080cls%s" % boot
    js = os.path.join(LOGS, "step080_%s_classify_%s.json" % (boot, nonce))
    rec = {"classify_json": js, "classify_nonce": nonce}
    try:
        proc = subprocess.run(
            [PY, CADENCE, "run", "--tokens", str(tokens),
             "--nonce", nonce, "--engine-log", out_log, "--out", js],
            capture_output=True, text=True, cwd=SCRATCH, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        rec["classify_rc"] = 124
        rec["classify_err"] = "timeout " + str(exc)[-200:]
        rec["state"] = "PROBE-TIMEOUT"
        say("classify %s: TIMEOUT (record kept, not lost)" % boot)
        return rec
    rec["classify_rc"] = proc.returncode
    if proc.returncode != 0:
        rec["classify_err"] = (proc.stderr or proc.stdout)[-400:]
        rec["state"] = "PROBE-FAIL"
        say("classify %s: rc=%s %s" % (boot, proc.returncode,
                                       rec["classify_err"][:200]))
        return rec
    with open(js, encoding="utf-8") as handle:
        data = json.load(handle)
    spd = data.get("s_per_page_step")
    rec["classify_tokens"] = data.get("tokens")
    rec["classify_page_steps"] = data.get("page_steps")
    rec["classify_ttft_s"] = data.get("ttft_s")
    rec["classify_s_per_page_step"] = spd
    rec["classify_prefill_tok_s"] = data.get("prefill_tok_s")
    rec["engine_drain_median_s"] = (data.get("engine_drain") or {}).get("median_s")
    rec["state"] = ("fast" if spd is not None and spd <= FAST_MAX else
                    "slow" if spd is not None and spd >= SLOW_MIN else
                    "mid" if spd is not None else "UNKNOWN")
    say("classify %s: %s s/page step -> %s (ttft=%s, drain med=%s)" % (
        boot, spd, rec["state"], data.get("ttft_s"),
        rec["engine_drain_median_s"]))
    return rec


def cmd_next(args) -> int:
    dump_dir = os.path.join(LOGS, args.tag)
    if os.path.exists(dump_dir):
        say("REFUSING: dump dir %s already exists (iron rule 16 (viii))" % dump_dir)
        return 2
    extra = {}
    for item in args.env or []:
        key, _, val = item.partition("=")
        extra[key] = val
    rows = load_state()
    rec = {"boot": args.boot, "tag": args.tag, "every": args.every,
           "extra_env": extra, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.no_down:
        rec["down"] = down(args.wait_mib, 240, args.boot)
    rec.update(up(args.tag, args.boot, args.every, extra, args.timeout))
    if not rec.get("health"):
        say("boot %s never reached health -- record and stop" % args.boot)
    elif not args.no_classify:
        rec.update(classify(args.boot, args.classify_tokens, rec["out_log"]))
    rows = [r for r in load_state() if r.get("boot") != args.boot]
    rows.append(rec)
    rows.sort(key=lambda r: r.get("boot", ""))
    save_state(rows)
    say("STATE: %s" % json.dumps({k: rec.get(k) for k in (
        "boot", "tag", "state", "classify_s_per_page_step", "boot_s",
        "directly_load_first_s", "error_lines", "traceback_lines")},
        ensure_ascii=False))
    return 0


def current_boot(boot: str) -> dict:
    for row in load_state():
        if row.get("boot") == boot:
            return row
    raise SystemExit("no recorded boot %s in %s" % (boot, STATE))


HEADROOM_PS1 = os.path.join(TOOLS, "prod_headroom_check.ps1")
HEADROOM_RE = re.compile(
    r"engine pid=(\d+) on adapter (\S+): dedicated=([\d.]+) MiB "
    r"shared=([\d.]+) MiB")
DISCRETE_RE = re.compile(r"discrete \(engine\)=([\d.]+) MiB")


def headroom_once() -> dict:
    """One read of the Windows GPU adapter counters for the engine process.

    NOTE (iron rule 19): the script's own `shared - 8298` verdict is only valid
    on the production config with the 8 GiB offload mmap. The KVMem arm has no
    such mmap, so only the ABSOLUTE dedicated/shared numbers are usable here --
    and only as a slow-vs-fast pair, never against the production constant.
    """
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-File", HEADROOM_PS1],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:
        return {"hms": time.strftime("%H:%M:%S"), "err": str(exc)[:120]}
    m = HEADROOM_RE.search(out)
    rec = {"hms": time.strftime("%H:%M:%S")}
    if m:
        rec.update({"pid": int(m.group(1)), "dedicated_mib": float(m.group(3)),
                    "shared_mib": float(m.group(4))})
    d = DISCRETE_RE.search(out)
    if d:
        rec["adapter_ded_mib"] = float(d.group(1))
    rec["gpu_used_mib"] = gpu_used()
    return rec


class HeadroomPoller(threading.Thread):
    """Sample dedicated/shared VRAM while a probe runs (eviction witness)."""

    def __init__(self, every: float = 45.0):
        super().__init__(daemon=True)
        self.every = every
        self.samples = []
        self._stopev = threading.Event()

    def run(self) -> None:
        while not self._stopev.is_set():
            self.samples.append(headroom_once())
            self._stopev.wait(self.every)

    def stop(self) -> list:
        self._stopev.set()
        self.join(timeout=150)
        return self.samples


ENGINE_PID_RE = re.compile(r"\(EngineCore pid=(\d+)\)")
PYSPY = r"G:\qwen3.8model\vllm-win029\Scripts\py-spy.exe"


def engine_pid(out_log: str) -> int | None:
    """The engine pid from the log prefix.

    py-spy needs it, and 079's ledger never covered the capture path: this
    process runs the model forward AND the connector (uniproc executor), which
    is exactly why a custom op inside the forward shows up as `outside` in a
    ledger that only times wait_for_save.
    """
    text = read_tail(out_log, limit=200_000)
    m = ENGINE_PID_RE.search(text)
    return int(m.group(1)) if m else None


def cmd_measure(args) -> int:
    rec = current_boot(args.boot)
    out_log = rec["out_log"]
    csv_path = os.path.join(LOGS, "step080_%s_fingerprint%s.csv"
                            % (args.boot, args.suffix))
    # rows hit the disk as they are sampled: a slow boot is a once-a-day
    # specimen and must not die with an in-memory buffer
    sampler = Sampler(1.0, out_csv=csv_path)
    sampler.start()
    t0 = time.time()
    # a repeat measurement MUST use a fresh nonce or the native prefix cache
    # answers instantly and the "cadence" is fiction (iron rule 22 (ii))
    js = os.path.join(LOGS, "step080_%s_cadence%s.json" % (args.boot, args.suffix))
    poller = (HeadroomPoller(args.headroom_every)
              if args.headroom_every > 0 else None)
    if poller:
        poller.start()
    cadence_rc = -1
    spy = None
    spy_path = os.path.join(LOGS, "step080_%s_pyspy%s.raw" % (args.boot,
                                                              args.suffix))
    spy_err = os.path.join(LOGS, "step080_%s_pyspy%s.err" % (args.boot,
                                                             args.suffix))
    spy_fe = None
    if args.pyspy:
        pid = engine_pid(out_log)
        # size the recording to this boot's own cadence so it ends by itself
        # (py-spy flushes on exit, and terminate() on Windows would not)
        per = float(rec.get("classify_s_per_page_step") or 0) or 1.6
        steps = max(1, int(args.tokens / 1456) + 1)
        dur = int(steps * per + 45)
        rec["pyspy_pid"] = pid
        rec["pyspy_duration_s"] = dur
        if pid:
            spy_fe = open(spy_err, "w", encoding="utf-8", errors="replace")
            spy_cmd = [PYSPY, "record", "-p", str(pid), "-d", str(dur),
                       "-r", str(args.pyspy_rate), "-f", "raw", "-o", spy_path]
            if args.pyspy_idle:
                # Default py-spy drops threads that are not holding the GIL, so
                # a boot blocked inside a C call (cudaMemcpy / stream sync) is
                # invisible: b6 recorded only 3424 of the expected ~6600
                # samples. --idle records them, which is where "who is waiting
                # for whom" actually lives.
                spy_cmd += ["--idle", "--threads"]
            spy = subprocess.Popen(spy_cmd, stdout=spy_fe, stderr=spy_fe)
        else:
            say("pyspy requested but no EngineCore pid found in %s" % out_log)
    try:
        proc = subprocess.run(
            [PY, CADENCE, "run", "--tokens", str(args.tokens),
             "--nonce", "s080m%s%s" % (args.boot, args.suffix),
             "--engine-log", out_log, "--out", js],
            capture_output=True, text=True, cwd=SCRATCH, timeout=args.timeout)
        cadence_rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        # Never lose the sampler: stop it, flush, record the timeout, THEN fail.
        cadence_rc = 124
        rec["cadence_timeout_s"] = args.timeout
        rec["cadence_tail"] = str(exc)[-300:]
        say("measure %s: cadence timed out after %s s" % (args.boot, args.timeout))
    if poller:
        rec["headroom_samples"] = poller.stop()
        rec["headroom_json"] = os.path.join(
            LOGS, "step080_%s_headroom%s.json" % (args.boot, args.suffix))
        with open(rec["headroom_json"], "w", encoding="utf-8") as handle:
            json.dump({"boot": args.boot, "suffix": args.suffix,
                       "note": "iron rule 19: only the absolute dedicated/shared "
                               "numbers are comparable here, not the script's "
                               "shared-8298 verdict",
                       "samples": rec["headroom_samples"]},
                      handle, ensure_ascii=False, indent=2)
    rows_ = sampler.stop()
    if spy is not None:
        try:
            rec["pyspy_rc"] = spy.wait(
                timeout=int(rec.get("pyspy_duration_s", 400)) + 240)
        except subprocess.TimeoutExpired:
            rec["pyspy_rc"] = "TIMEOUT"
            spy.kill()
        if spy_fe:
            spy_fe.close()
        rec["pyspy_raw"] = spy_path if os.path.exists(spy_path) else None
        say("pyspy: rc=%s raw=%s" % (rec["pyspy_rc"], rec["pyspy_raw"]))
    from kvmem_boot_fingerprint import summarise
    fp = {"boot": args.boot, "suffix": args.suffix,
          "window": "ingest cadence only (the 8k anchor is a separate window)",
          "seconds": round(time.time() - t0, 1),
          "cadence_rc": cadence_rc,
          "gpu_coverage": sampler.coverage(),
          "fit_for_verdict": bool(sampler.coverage() >= 0.8 and len(rows_) >= 10),
          **summarise(rows_)}
    js_path = os.path.join(LOGS, "step080_%s_fingerprint%s.json"
                           % (args.boot, args.suffix))
    with open(js_path, "w", encoding="utf-8") as handle:
        json.dump(fp, handle, ensure_ascii=False, indent=2)
    rec["fingerprint_json"] = js_path
    rec["fingerprint_csv"] = csv_path
    if args.anchor:
        # The anchor is a different step shape, so it gets its own sampler
        # window instead of being folded into the ingest fingerprint.
        ajs = os.path.join(LOGS, "step080_%s_8k%s.json" % (args.boot, args.suffix))
        ap = subprocess.run(
            [PY, ANCHOR, "--lengths", "8000", "--warmup", "0", "--repeats", "3",
             "--out", ajs, "--arm", "kvmem080%s%s_timing" % (args.boot,
                                                             args.suffix)],
            capture_output=True, text=True, cwd=SCRATCH, timeout=900)
        rec["anchor_rc"] = ap.returncode
        rec["anchor_json"] = ajs
        if ap.returncode == 0 and os.path.exists(ajs):
            with open(ajs, encoding="utf-8") as handle:
                data = json.load(handle)
            reps = (data.get("results") or [{}])[0].get("repeats") or []
            vals = sorted(r.get("steady_tok_s") for r in reps
                          if r.get("steady_tok_s"))
            rec["anchor_median_tok_s"] = vals[len(vals) // 2] if vals else None
            rec["anchor_all"] = vals
    if cadence_rc == 0 and os.path.exists(js):
        with open(js, encoding="utf-8") as handle:
            data = json.load(handle)
        rec["main_ttft_s"] = data.get("ttft_s")
        rec["main_s_per_page_step"] = data.get("s_per_page_step")
        rec["main_prefill_tok_s"] = data.get("prefill_tok_s")
        rec["main_page_steps"] = data.get("page_steps")
        # NOT this request's cadence: drain_timeline() reads the whole engine
        # log, so on a boot with two probes this is their mixture. Use the
        # client s_per_page_step above and the [KVTIME] windows for anything
        # that has to be attributed to a single request.
        rec["engine_drain_wholelog"] = data.get("engine_drain")
        rec["cadence_json"] = js
    rec["fingerprint_json"] = js_path
    rec["fingerprint_csv"] = csv_path
    rec["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    merge_boot(args.boot, rec)
    say("measure %s: cadence rc=%s ttft=%s s/page=%s anchor=%s fp_s=%s" % (
        args.boot, cadence_rc, rec.get("main_ttft_s"),
        rec.get("main_s_per_page_step"), rec.get("anchor_median_tok_s"),
        fp.get("window_s")))
    say("fingerprint: power med=%s mem_util med=%s py_cpu med=%s "
        "ram_avail min=%s disk_read p90=%s waiting_share=%s" % (
            (fp.get("power_w") or {}).get("median"),
            (fp.get("gpu_mem_util") or {}).get("median"),
            (fp.get("py_cpu_pct") or {}).get("median"),
            (fp.get("ram_avail_gb") or {}).get("min"),
            (fp.get("disk_read_mb_s") or {}).get("p90"),
            fp.get("waiting_fingerprint_share")))
    return 0


def cmd_classify(args) -> int:
    """Classify a boot that is already up (used after an aborted `next`)."""
    rec = current_boot(args.boot)
    patch = {"health": health(), "out_log": rec["out_log"],
             "err_log": rec["err_log"]}
    patch.update(verify(rec["out_log"], rec["err_log"]))
    if not patch["health"]:
        say("%s: /health is down, nothing to classify" % args.boot)
    else:
        patch.update(classify(args.boot, args.tokens, rec["out_log"],
                              nonce="s080cls%s%s" % (args.boot, args.suffix)))
    merge_boot(args.boot, patch)
    say("STATE: %s" % json.dumps({k: patch.get(k) for k in (
        "boot", "health", "state", "classify_s_per_page_step",
        "directly_load_first_s", "compiling_n", "error_lines",
        "traceback_lines")}, ensure_ascii=False))
    return 0


def cmd_vp(args) -> int:
    rec = current_boot(args.boot)
    js = args.out or os.path.join(
        LOGS, "step080_%s_vp_d%s_n%s.json" % (args.boot, str(args.depth).replace(".", ""),
                                              args.nonce))
    proc = subprocess.run(
        [PY, VPROBE, "run", "--tokens", str(args.tokens), "--depth",
         str(args.depth), "--nonce", args.nonce, "--tag",
         "s080_%s_d%s_%s" % (args.boot, args.depth, args.nonce),
         "--out", js],
        capture_output=True, text=True, cwd=SCRATCH, timeout=args.timeout)
    say("vp %s depth=%s nonce=%s rc=%s" % (args.boot, args.depth, args.nonce,
                                           proc.returncode))
    tail = (proc.stdout or "").splitlines()[-6:]
    for line in tail:
        say("  | " + line)
    if proc.returncode != 0:
        say("  stderr tail: " + (proc.stderr or "")[-300:])
        return 1
    with open(js, encoding="utf-8") as handle:
        data = json.load(handle)
    serve = data.get("serve") or {}
    say("  verdict: hit=%s ttft=%s finish=%s chunks=%s needle_token=%s in_window=%s" % (
        serve.get("needle_hit"), serve.get("ttft_s"), serve.get("finish_reason"),
        serve.get("n_chunks"), data.get("needle_token"), data.get("needle_in_window")))
    return 0


def cmd_stop(args) -> int:
    down(800.0, 240, args.label)
    return 0


def cmd_show(args) -> int:
    for row in load_state():
        if args.boot and row.get("boot") != args.boot:
            continue
        print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("next")
    n.add_argument("--boot", required=True)
    n.add_argument("--tag", required=True)
    n.add_argument("--every", type=int, default=10)
    n.add_argument("--classify-tokens", type=int, default=16000)
    n.add_argument("--timeout", type=float, default=420)
    n.add_argument("--wait-mib", type=float, default=800)
    n.add_argument("--env", action="append")
    n.add_argument("--no-down", action="store_true")
    n.add_argument("--no-classify", action="store_true")
    n.set_defaults(func=cmd_next)

    c = sub.add_parser("classify")
    c.add_argument("--boot", required=True)
    c.add_argument("--tokens", type=int, default=16000)
    c.add_argument("--suffix", default="",
                   help="nonce suffix; a re-triage of the same boot MUST use a "
                        "new one or the prefix cache fakes the cadence")
    c.set_defaults(func=cmd_classify)

    m = sub.add_parser("measure")
    m.add_argument("--boot", required=True)
    m.add_argument("--tokens", type=int, default=43680)
    m.add_argument("--anchor", action="store_true")
    m.add_argument("--suffix", default="",
                   help="nonce/evidence suffix; a repeat run MUST carry a new "
                        "one or the prefix cache fakes the cadence")
    m.add_argument("--headroom-every", type=float, default=45.0,
                   help="seconds between GPU-adapter dedicated/shared reads "
                        "while the ingest runs (0 = off)")
    m.add_argument("--pyspy", action="store_true",
                   help="record the engine's Python stacks for the cadence "
                        "window (the instrument that found 079's unmeasured "
                        "segment); output = step080_<boot>_pyspy<suffix>.raw")
    m.add_argument("--pyspy-rate", type=int, default=25)
    m.add_argument("--pyspy-idle", action="store_true",
                   help="also sample threads that released the GIL")
    m.add_argument("--timeout", type=float, default=1500)
    m.set_defaults(func=cmd_measure)

    v = sub.add_parser("vp")
    v.add_argument("--boot", required=True)
    v.add_argument("--depth", type=float, required=True)
    v.add_argument("--nonce", required=True)
    v.add_argument("--tokens", type=int, default=200000)
    v.add_argument("--out", default=None)
    v.add_argument("--timeout", type=float, default=2400)
    v.set_defaults(func=cmd_vp)

    s = sub.add_parser("stop")
    s.add_argument("--label", default="")
    s.set_defaults(func=cmd_stop)

    w = sub.add_parser("verify")
    w.add_argument("--boot", required=True)
    w.set_defaults(func=lambda a: (print(json.dumps(
        verify(current_boot(a.boot)["out_log"], current_boot(a.boot)["err_log"]),
        ensure_ascii=False, indent=2)), 0)[1])

    h = sub.add_parser("show")
    h.add_argument("--boot", default=None)
    h.set_defaults(func=cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
