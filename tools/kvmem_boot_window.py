# step081_boot 用的窗口读数工具（纯离线，零 GPU）。
#
# 为什么这一步会多出这么一件工具（081 的实测发现，勿当巧合）
# ---------------------------------------------------------------
# vLLM 在 boot 期跑 flashinfer autotune 时会打两行毫秒时间戳：
#
#   ... - INFO - autotuner.py:972 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
#   ... - INFO - autotuner.py:995 - flashinfer.jit: [Autotuner]: Autotuning process ends
#
# 这两行之间夹着的正是 `kernel_warmup.py` 里的
# `_run_flashinfer_autotune_dummy_runs()` —— 一次 `--max-num-batched-tokens`
# 大小（本臂 = 1458）的 **模型自身前向**，`_dummy_run(is_profile=True)`。
# 所以这个差值 = **一支 boot 里唯一一个"没有连接器入库、没有投机、没有 KV 搬运、
# 没有长上下文注意力"的整页前向墙钟**，而且它**免费**（已经在每支 boot 的 err.log 里）。
#
# 080 的 11 支 boot 用它读数：快态 0.986-1.150 s / 慢态 2.482-2.621 s，与 080
# 自己按请求打出的快慢标签 **10/11 同判**（唯一不同判 = b9：窗口 1.150 而请求页步
# 7.76 s）。⇒ 慢态在请求到来之前、在连接器不在场的时候就已经成立（幅度 ~2.5×），
# 而请求路径上的 7-8× 是它之上的另一次放大。
#
# 用法
# ----
#   python tools/kvmem_boot_window.py --logs-dir G:\qwen3.8model\prod029_logs
#   python tools/kvmem_boot_window.py --glob "step08*_arm.err.log" --logs-dir ...
#   python tools/kvmem_boot_window.py --files a.log b.log --out window_table.json
#
# 注意
# ----
# * 只读日志，不 import vllm、不碰 GPU（必守 2）。
# * `mbt` 从 `Running FlashInfer autotune with N tokens.` 取；不同臂的 N 不同，
#   **跨配置比较只比同 mbt 的窗口**（生产 mbt 与本臂不同，不能直接同列比大小）。
# * 若日志里没有这两行（老 flashinfer 版本 / 关掉 autotune），返回 `no-window`，
#   不许把缺读当 0。

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
from datetime import datetime

TS_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[,.]\d{3})")
MBT_RE = re.compile(r"Running FlashInfer autotune with (\d+) tokens")

# 同一段 boot 期里另外几条"逐字同形才算启动侧无差异"的读数，
# 080 已用它们否掉盘/编译/图捕获，这里一并带上，免得下次又手工 grep。
SIDE = [
    ("model_load_s", r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", 1),
    ("torch_compile_s", r"torch\.compile took ([\d.]+) s in total", 0),
]
PIN_RE = re.compile(r"pin[ _]shim.*?moved[= ](\d+)", re.I)


def _ts(line: str) -> float | None:
    m = TS_RE.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1).replace(".", ","),
                             "%Y-%m-%d %H:%M:%S,%f").timestamp()


def read_tail(path: str, limit: int = 3_000_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()[-limit:]
    except Exception:  # noqa: BLE001
        return ""


def companion(path: str) -> str | None:
    """flashinfer 的 logger 写 stderr、vLLM 的写 stdout ⇒ 两个流要配对读。"""
    if path.endswith(".err.log"):
        alt = path[: -len(".err.log")] + ".out.log"
    elif path.endswith(".out.log"):
        alt = path[: -len(".out.log")] + ".err.log"
    else:
        return None
    return alt if os.path.isfile(alt) else None


def read_log(path: str) -> dict:
    """One arm log -> {mbt, window_s, starts/ends stamp, side timings}."""
    rec: dict = {"file": os.path.basename(path), "mbt": None, "window_s": None,
                 "starts": None, "ends": None, "window_from": None}
    paths = [path]
    comp = companion(path)
    if comp:
        paths.append(comp)
        rec["window_from"] = os.path.basename(path)
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if rec["mbt"] is None:
                    m = MBT_RE.search(line)
                    if m:
                        rec["mbt"] = int(m.group(1))
                if rec["starts"] is None and "Autotuning process starts" in line:
                    rec["starts"] = _ts(line)
                    continue
                if (rec["starts"] is not None and rec["ends"] is None
                        and "Autotuning process ends" in line):
                    rec["ends"] = _ts(line)
                    rec["window_s"] = round(rec["ends"] - rec["starts"], 3)
    text = "\n".join(read_tail(p) for p in paths)
    for name, pattern, field in SIDE:
        found = re.findall(pattern, text)
        if found:
            item = found[-1]
            rec[name] = float(item if isinstance(item, str) else item[field])
    mv = PIN_RE.findall(text)
    if mv:
        rec["pin_moved_lines"] = len(mv)
    return rec


def classify_window(w: float | None, fast: float, slow: float,
                    ref_fast: float | None = None) -> str:
    """按同 mbt 的两簇给标签；中间值如实写 mid（必守 7：不许硬凑成两档）。"""
    if w is None:
        return "no-window"
    if w <= fast:
        return "fast"
    if w >= slow:
        return "slow"
    return "mid"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=r"G:\qwen3.8model\prod029_logs")
    ap.add_argument("--glob", action="append",
                    help="相对 --logs-dir 的 glob，可多次；默认全部 *.log")
    ap.add_argument("--files", nargs="*", default=[],
                    help="显式文件路径（优先于 --glob）")
    ap.add_argument("--fast", type=float, default=1.25,
                    help="窗口 <= 此值 = fast（本臂 1458-token 实测快簇 0.986-1.150）")
    ap.add_argument("--slow", type=float, default=2.4,
                    help="窗口 >= 此值 = slow（实测慢簇 2.482-2.621）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths: list[str] = list(args.files or [])
    if not paths:
        pats = args.glob or ["*.log"]
        for pat in pats:
            paths += _glob.glob(os.path.join(args.logs_dir, pat))
    rows = []
    for p in sorted(set(paths)):
        if not os.path.isfile(p):
            continue
        rec = read_log(p)
        rec["window_state"] = classify_window(rec["window_s"], args.fast, args.slow)
        if rec["window_s"] is not None:
            rows.append(rec)

    print(f"{'win_s':>8} {'state':>6} {'mbt':>5} "
          f"{'load_s':>7} {'compile_s':>9}  file")
    for r in sorted(rows, key=lambda x: (x["file"])):
        print(f"{r['window_s']:8.3f} {r['window_state']:>6} "
              f"{str(r['mbt']):>5} {str(r.get('model_load_s')):>7} "
              f"{str(r.get('torch_compile_s')):>9}  {r['file']}")

    # 按 mbt 分组出簇，避免跨配置比大小
    by_mbt: dict[int, list[float]] = {}
    for r in rows:
        by_mbt.setdefault(r["mbt"] or -1, []).append(r["window_s"])
    print("\n簇（按 mbt 分组；n / min / 中位 / max）:")
    for mbt, vals in sorted(by_mbt.items()):
        vals = sorted(vals)
        med = vals[len(vals) // 2]
        print(f"  mbt={mbt:>6}  n={len(vals):>4}  min={vals[0]:7.3f}  "
              f"med={med:7.3f}  max={vals[-1]:7.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows,
                       "clusters_by_mbt": {str(k): sorted(v)
                                           for k, v in by_mbt.items()}},
                      fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
