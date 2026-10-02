# -*- coding: utf-8 -*-
"""082-C：PDH GPU Engine / GPU Process Memory 取样器（工具落正式 tools/）。

081 四个探针的真败因：`EnumObjectItems("", "", obj, detail)` 把 machine 传成
空串（被当成远程机器名）⇒ buffer size 错。本机对象名就是英文 "GPU Engine"，
machine 必须传 None。

读数三件套（对 082-A 驻留假设的直接证据）：
  * GPU Engine(pid_..._engtype_X)\\Utilization Percentage —— 按 pid × engtype
    的引擎利用率（否证/坐实"他占客户端"必守 26④）；
  * GPU Process Memory(pid_...)\\{Local,Non Local,Dedicated,Shared,Total Committed}
    Usage —— **Non Local Usage = 该进程被 WDDM 降到系统内存的显存字节数**，
    这正是"权重页驻留被破坏"的直接读数；
  * GPU Adapter Memory(luid_...)\\{Dedicated,Shared,Total Committed} —— 卡级
    对照（WDDM 下卡级 memory.used 是唯一可靠外置口径，必守 10，这里多一条
    独立来源）。

用法（cwd 必须不在记录仓，铁律 2）：
  python tools/pdh_gpu_engine.py sample --seconds 600 --interval 1.0 \
      --out G:\\qwen3.8model\\prod029_logs\\step082_x_pdh.jsonl [--pid 123]

输出 jsonl：每 tick 一行 {"ts":..., "eng": {pid: {engtype: pct}}, "pmem":
{pid: {"local":..,"non_local":..,...}}, "adapter": {luid: {...}}}，只记非零。
实例集每 tick 重建（引擎进程是后出现的，固定查询会漏）。
"""
from __future__ import annotations

import argparse
import json
import time

import win32pdh

OBJ_ENGINE = "GPU Engine"
OBJ_PMEM = "GPU Process Memory"
OBJ_ADAPTER = "GPU Adapter Memory"


def _collect(paths: dict[str, int]) -> dict[str, float]:
    """一次 Collect + 读全部计数器；坏路径静默跳过（进程消失是常态）。"""
    if not paths:
        return {}
    q = win32pdh.OpenQuery(None, 0)
    handles = []
    try:
        for path in paths:
            try:
                handles.append((path, win32pdh.AddCounter(q, path)))
            except Exception:  # noqa: BLE001
                pass
        win32pdh.CollectQueryData(q)
        out = {}
        for path, h in handles:
            try:
                out[path] = win32pdh.GetFormattedCounterValue(
                    h, win32pdh.PDH_FMT_DOUBLE)[1]
            except Exception:  # noqa: BLE001
                pass
        return out
    finally:
        for _, h in handles:
            try:
                win32pdh.RemoveCounter(h)
            except Exception:  # noqa: BLE001
                pass
        win32pdh.CloseQuery(q)


def _util_paths(pid_filter: str | None,
                luid_tag: str | None = None) -> dict[str, str]:
    """GPU Engine 利用率路径：path -> 实例名。"""
    try:
        _, instances = win32pdh.EnumObjectItems(
            None, None, OBJ_ENGINE, win32pdh.PERF_DETAIL_WIZARD)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for inst in instances:
        if pid_filter and ("pid_%s_" % pid_filter) not in inst:
            continue
        if luid_tag and luid_tag not in inst:
            continue
        out["\\%s(%s)\\Utilization Percentage" % (OBJ_ENGINE, inst)] = inst
    return out


def _adapter_snapshot() -> dict[str, dict[str, float]]:
    """卡级对照 + 用来识别独显 LUID（dedicated 最大的那个）。"""
    out: dict[str, dict[str, float]] = {}
    try:
        _, insts = win32pdh.EnumObjectItems(
            None, None, OBJ_ADAPTER, win32pdh.PERF_DETAIL_WIZARD)
        paths = {}
        for inst in insts:
            for ctr in ("Dedicated Usage", "Shared Usage", "Total Committed"):
                paths["\\%s(%s)\\%s" % (OBJ_ADAPTER, inst, ctr)] = (inst, ctr)
        for path, val in _collect(paths).items():
            if not val:
                continue
            inst, ctr = paths[path]
            luid = inst.split("_phys_")[0].replace("luid_", "", 1)
            key = {"Dedicated Usage": "dedicated", "Shared Usage": "shared",
                   "Total Committed": "total"}[ctr]
            out.setdefault(luid, {})[key] = round(
                out.get(luid, {}).get(key, 0.0) + val, 0)
    except Exception:  # noqa: BLE001
        pass
    return out


def sample_once(pid_filter: str | None,
                luid: str | None = None) -> dict:
    """luid=None 时全收；给 luid 时只收该卡（引擎实例数从 ~379 掉到几十）。"""
    luid_tag = ("luid_%s_phys_0" % luid) if luid else None
    eng = _collect(_util_paths(pid_filter, luid_tag))
    # eng: path -> 值；折叠成 {pid: {engtype: pct}}
    eng_fold: dict[str, dict[str, float]] = {}
    inst_of = _util_paths(pid_filter, luid_tag)
    for path, val in eng.items():
        inst = inst_of.get(path, "")
        pid = inst.split("_")[1] if inst.startswith("pid_") else inst[:20]
        engtype = inst.split("engtype_")[-1] if "engtype_" in inst else "?"
        if val:
            eng_fold.setdefault(pid, {})[engtype] = round(
                eng_fold.get(pid, {}).get(engtype, 0.0) + val, 2)
    # 进程显存（Non Local = 被降到系统内存）
    pmem_fold: dict[str, dict[str, float]] = {}
    try:
        _, insts = win32pdh.EnumObjectItems(
            None, None, OBJ_PMEM, win32pdh.PERF_DETAIL_WIZARD)
        paths = {}
        for inst in insts:
            if pid_filter and ("pid_%s_" % pid_filter) not in inst:
                continue
            if luid_tag and luid_tag not in inst:
                continue
            pid = inst.split("_")[1]
            for ctr in ("Local Usage", "Non Local Usage", "Shared Usage",
                        "Total Committed"):
                paths["\\%s(%s)\\%s" % (OBJ_PMEM, inst, ctr)] = (pid, ctr)
        vals = _collect(paths)
        for path, val in vals.items():
            if not val:
                continue
            pid, ctr = paths[path]
            key = {"Local Usage": "local", "Non Local Usage": "non_local",
                   "Shared Usage": "shared",
                   "Total Committed": "total"}[ctr]
            pmem_fold.setdefault(pid, {})[key] = round(
                pmem_fold.get(pid, {}).get(key, 0.0) + val, 0)
    except Exception:  # noqa: BLE001
        pass
    adapter = _adapter_snapshot()
    return {"eng": eng_fold, "pmem": pmem_fold, "adapter": adapter}


def cmd_sample(args) -> int:
    fh = open(args.out, "a", encoding="utf-8")
    luid = args.luid
    if luid is None and not args.no_autolid:
        snap = _adapter_snapshot()
        best = sorted(snap.items(), key=lambda kv: -kv[1].get("dedicated", 0))
        if best:
            luid = best[0][0]
    print("sampling %ss every %ss -> %s (luid=%s)"
          % (args.seconds, args.interval, args.out, luid or "ALL"), flush=True)
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        row = sample_once(args.pid, luid)
        row["ts"] = time.strftime("%H:%M:%S")
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        n += 1
        time.sleep(args.interval)
    fh.close()
    print("done, %d rows" % n)
    return 0


def cmd_oneshot(args) -> int:
    row = sample_once(args.pid, args.luid)
    print(json.dumps(row, ensure_ascii=False, indent=1)[:4000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seconds", type=float, default=600)
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--out", required=True)
    s.add_argument("--pid", default=None)
    s.add_argument("--luid", default=None)
    s.add_argument("--no-autolid", action="store_true")
    s.set_defaults(func=cmd_sample)
    o = sub.add_parser("oneshot")
    o.add_argument("--pid", default=None)
    o.add_argument("--luid", default=None)
    o.set_defaults(func=cmd_oneshot)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
