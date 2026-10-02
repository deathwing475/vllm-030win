# kvmem_boot_fingerprint.py -- out-of-band 1 Hz sampler for boot-state attribution.
#
# Why this exists (step 080): step 078 found the SAME launcher swinging between
# ~1.4 s and 8-11 s per 1456-token page step across boots, and characterised the
# slow state only by what it was NOT (pageable fallback, PCIe downtrain, evicted
# VRAM, the fast/slow pin) -- its fingerprint was "utilization.gpu 100% but
# utilization.memory 1%, 82-105 W". Step 079 then showed the connector itself is
# only ~12% of a fast page step, so the slow state lives in `outside`, which the
# [KVTIME] ledger cannot split any further without patching files that are in the
# AOT cache key.
#
# This tool splits it WITHOUT touching the engine: it samples GPU and host state
# out of band while a probe runs, so a slow page step can be classified as
#   (i)  GPU really computing   -> power.draw high, utilization.memory high
#   (ii) host busy launching    -> python CPU seconds tick ~1 core per wall sec
#   (iii) host blocked waiting  -> python CPU ~0 while wall time runs
#   (iv) host memory pressure   -> RAM available low, pagefile use rising,
#                                  disk read bytes/s spiking (this box only has
#                                  23.1 GB usable RAM and the arm pins 5 GiB for
#                                  the workspace while loading 11.35 GiB of
#                                  weights), which no earlier step ever measured.
#
# Read-only: nvidia-smi + psutil. It never imports vllm, never touches the GPU,
# never syncs a stream -- it cannot perturb what it measures (iron rule 24).
#
# Usage:
#   python tools/kvmem_boot_fingerprint.py run --seconds 420 --out <csv> [--label b3]
#   python tools/kvmem_boot_fingerprint.py summarize --csv <csv> [--out json]

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # keep the module importable on machines without psutil
    psutil = None

FIELDS = [
    "t", "hms",
    "gpu_util", "gpu_mem_util", "power_w", "power_limit_w", "sm_clock",
    "mem_clock", "pstate", "temperature_c", "clock_reasons",
    # iron rule 7: PCIe current-vs-max is part of the required typing kit
    "pcie_gen_cur", "pcie_gen_max", "pcie_width_cur", "pcie_width_max",
    "gpu_used_mib", "gpu_apps_n", "gpu_apps_mib", "gpu_apps_pids",
    "ram_avail_gb", "ram_used_pct", "pagefile_used_gb",
    "py_n", "py_cpu_pct", "py_breakdown", "self_pid", "py_rss_gb",
    "cpu_pct", "disk_read_mb_s", "disk_write_mb_s", "host_err",
]

SMI_CSV = ("memory.used,utilization.gpu,utilization.memory,power.draw,"
           "power.limit,clocks.sm,clocks.mem,pstate,temperature.gpu,"
           "pcie.link.gen.current,pcie.link.gen.max,"
           "pcie.link.width.current,pcie.link.width.max")
# clocks_event_reasons.active can itself contain commas ("350 W Power Limit,
# Active Thread..."), so it gets its own query instead of a positional split.
SMI_REASONS = "clocks_event_reasons.active"


def _f(value: str) -> float | None:
    try:
        return float(value.strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


def sample_gpu() -> dict:
    """Instantaneous GPU state; {} when nvidia-smi is unavailable.

    Keys are assigned BY NAME from the query order above, so a field added in
    the middle cannot silently shift the whole row.
    """
    out: dict = {}
    names = ["gpu_used_mib", "gpu_util", "gpu_mem_util", "power_w",
             "power_limit_w", "sm_clock", "mem_clock", "pstate",
             "temperature_c", "pcie_gen_cur", "pcie_gen_max",
             "pcie_width_cur", "pcie_width_max"]
    try:
        raw = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_CSV}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True,
                             timeout=8).stdout.strip()
    except Exception:
        return out
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < len(names):
        return out
    for key, value in zip(names, parts):
        out[key] = value if key == "pstate" else _f(value)
    try:
        reasons = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_REASONS}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        out["clock_reasons"] = reasons[:80]
    except Exception:
        pass
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout
        rows = [r for r in apps.splitlines() if r.strip()]
        out["gpu_apps_n"] = len(rows)
        # "used_memory" can come back as "N/A" or with a trailing unit, and the
        # list is who-holds-VRAM -- exactly the witness for a co-tenant theory,
        # so keep the pids rather than only a sum.
        mib = 0
        pids = []
        for r in rows:
            cols = r.split(",")
            if not cols:
                continue
            pids.append(cols[0].strip())
            if len(cols) > 1:
                digits = "".join(ch if ch.isdigit() else " "
                                 for ch in cols[1]).split()
                if digits:
                    mib += int(digits[0])
        out["gpu_apps_mib"] = mib
        out["gpu_apps_pids"] = ",".join(pids)[:120]
    except Exception:
        pass
    return out


def python_breakdown(limit: int = 5) -> tuple[str, float, int, int]:
    """Per-PID cpu/rss for python processes, biggest-CPU first.

    An aggregate over all python processes is worthless here: the driver and
    its probe child are python too, so "py_cpu = 101%" cannot tell a busy
    engine from its own observer. Reporting the breakdown (plus this process's
    own pid, sampled by the caller) lets the engine be identified and the
    observer subtracted offline.

    (page_faults is intentionally NOT collected: psutil does not fill it on
    Windows, and a field that is always 0 reads as "not the bottleneck" --
    iron rule 24 (i).)
    """
    if psutil is None:
        return "", 0.0, 0, 0
    rows = []
    rss = 0
    for p in psutil.process_iter(["name", "pid"]):
        try:
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            mi = p.memory_info()
            rss += mi.rss
            rows.append((p.pid, p.cpu_percent(interval=None), mi.rss))
        except Exception:
            continue
    rows.sort(key=lambda r: -r[1])
    top = rows[:limit]
    return ("|".join("%d:%.1f:%.0f" % (pid, cpu, r / 2 ** 20)
                     for pid, cpu, r in top),
            sum(r[1] for r in rows), rss, len(rows))


def sample_host(prev: dict | None, t_prev: float | None) -> dict:
    """Host RAM / commit / python CPU / paging rates (psutil)."""
    out: dict = {}
    if psutil is None:
        return out
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    out["ram_avail_gb"] = round(vm.available / 2 ** 30, 2)
    out["ram_used_pct"] = vm.percent
    out["pagefile_used_gb"] = round(sm.used / 2 ** 30, 3)
    out["cpu_pct"] = psutil.cpu_percent(interval=None)
    breakdown, cpu_sum, rss, n = python_breakdown()
    out["py_cpu_pct"] = round(cpu_sum, 1)
    out["py_breakdown"] = breakdown
    out["py_n"] = n
    out["py_rss_gb"] = round(rss / 2 ** 30, 2)
    out["self_pid"] = os.getpid()
    if prev is not None and t_prev:
        dt = max(time.time() - t_prev, 1e-6)
        try:
            io = psutil.disk_io_counters()
            for key, cur, name in (("read", io.read_bytes, "disk_read_mb_s"),
                                   ("write", io.write_bytes, "disk_write_mb_s")):
                pd = prev.get(key)
                if pd is not None:
                    out[name] = round((cur - pd) / 2 ** 20 / dt, 1)
                prev[key] = cur
        except Exception:
            pass
    return out


class Sampler(threading.Thread):
    """1 Hz out-of-band sampler.

    Writes each row as it goes (out_csv) so a crash or a Ctrl-C in the middle of
    a once-a-day slow boot cannot take the only copy of the evidence with it --
    the first version kept rows in memory and wrote them at the end.
    """

    def __init__(self, interval: float = 1.0, out_csv: str | None = None):
        super().__init__(daemon=True)
        self.interval = interval
        self.out_csv = out_csv
        self.rows: list[dict] = []
        self.gpu_missing = 0
        # NOT self._stop: threading.Thread already owns a _stop method.
        self._stopev = threading.Event()
        self._writer = None
        self._handle = None

    def _open(self) -> None:
        if not self.out_csv:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.out_csv)), exist_ok=True)
        self._handle = open(self.out_csv, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=FIELDS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._handle.flush()

    def _emit(self, row: dict) -> None:
        self.rows.append(row)
        if self._writer:
            try:
                self._writer.writerow(row)
                self._handle.flush()
            except Exception:
                pass

    def run(self) -> None:
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # prime the counters
            except Exception:
                pass
        prev: dict = {}
        try:
            io = psutil.disk_io_counters() if psutil else None
            if io is not None:
                prev = {"read": io.read_bytes, "write": io.write_bytes}
        except Exception:
            prev = {}
        self._open()
        t_prev = time.time()
        first = True
        while not self._stopev.is_set():
            row = {"t": round(time.time(), 2),
                   "hms": time.strftime("%H:%M:%S")}
            gpu = sample_gpu()
            if "gpu_util" not in gpu:
                self.gpu_missing += 1
            row.update(gpu)
            try:
                # first row has no baseline -> no rates (a dt~0 divisor would
                # otherwise invent a huge disk number in row 0)
                row.update(sample_host(None if first else prev, t_prev))
                first = False
            except Exception as exc:
                row["host_err"] = str(exc)[:80]
            t_prev = time.time()
            self._emit(row)
            self._stopev.wait(self.interval)
        if self._handle:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = self._writer = None

    def stop(self) -> list[dict]:
        """Join long enough for the slowest in-flight row (4 nvidia-smi calls
        at 8 s each), then hand back a snapshot."""
        self._stopev.set()
        self.join(timeout=90)
        return list(self.rows)

    def coverage(self) -> float:
        """Fraction of rows that carry GPU numbers; <0.8 = not attributable."""
        return 1.0 if not self.rows else round(
            1.0 - self.gpu_missing / len(self.rows), 3)

    def rows_snapshot(self) -> list[dict]:
        return list(self.rows)


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stat(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    idx = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"n": len(values), "median": round(statistics.median(values), 3),
            "p10": round(idx(0.10), 3), "p90": round(idx(0.90), 3),
            "min": round(values[0], 3), "max": round(values[-1], 3)}


def _mode(values: list[str]) -> dict:
    if not values:
        return {"n": 0}
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    return {"n": len(values), "top": ["%s x%d" % (k, v) for k, v in top]}


def summarise(rows: list[dict]) -> dict:
    out: dict = {"samples": len(rows),
                 "window_s": (round(rows[-1]["t"] - rows[0]["t"], 1)
                              if rows else 0.0)}
    for key in ("gpu_util", "gpu_mem_util", "power_w", "power_limit_w",
                "sm_clock", "mem_clock", "temperature_c", "gpu_used_mib",
                "gpu_apps_mib", "pcie_gen_cur", "pcie_width_cur",
                "ram_avail_gb", "pagefile_used_gb", "py_cpu_pct",
                "py_rss_gb", "cpu_pct", "disk_read_mb_s", "disk_write_mb_s"):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = _stat(vals)
    out["pstate"] = _mode([r["pstate"] for r in rows if r.get("pstate")])
    out["clock_reasons"] = _mode([r["clock_reasons"] for r in rows
                                  if r.get("clock_reasons")])
    # who held the VRAM during the window (co-tenant witness)
    out["gpu_apps_pids"] = _mode([r["gpu_apps_pids"] for r in rows
                                  if r.get("gpu_apps_pids")])
    busy = [r for r in rows if (r.get("gpu_util") or 0) >= 90]
    if busy:
        out["gpu_busy_share"] = round(len(busy) / len(rows), 3)
        out["waiting_fingerprint_share"] = round(
            sum(1 for r in busy if (r.get("power_w") or 999) < 130
                and (r.get("gpu_mem_util") or 99) < 10) / len(busy), 3)
        out["host_python_cpu_while_busy_median"] = _stat(
            [r.get("py_cpu_pct", 0) for r in busy])
        out["ram_avail_while_busy_gb"] = _stat(
            [r["ram_avail_gb"] for r in busy
             if isinstance(r.get("ram_avail_gb"), (int, float))])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--seconds", type=float, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument("--label", default=None, help="only a tag, written to the json")
    r.add_argument("--summary", default=None)

    s = sub.add_parser("summarize")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "summarize":
        with open(args.csv, newline="", encoding="utf-8") as handle:
            rows = []
            for rec in csv.DictReader(handle):
                row = {"t": float(rec["t"]) if rec.get("t") else 0.0}
                for key in FIELDS:
                    if key in ("t", "hms") or rec.get(key) in (None, ""):
                        continue
                    if key in ("pstate", "clock_reasons", "py_breakdown"):
                        row[key] = rec[key]
                        continue
                    try:
                        row[key] = float(rec[key])
                    except ValueError:
                        pass
                rows.append(row)
        out = {"csv": args.csv, **summarise(rows)}
        text = json.dumps(out, ensure_ascii=False, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
        return 0

    sampler = Sampler(args.interval, out_csv=args.out)
    sampler.start()
    try:
        time.sleep(args.seconds)
    finally:
        rows = sampler.stop()
    if not sampler.out_csv:
        write_csv(rows, args.out)
    out = {"out_csv": args.out, "label": args.label,
           "gpu_coverage": sampler.coverage(), **summarise(rows)}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    main()
