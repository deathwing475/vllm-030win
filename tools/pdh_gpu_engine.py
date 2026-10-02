# -*- coding: utf-8 -*-
"""082-C：PDH GPU Engine / GPU Process Memory 取样器（工具落正式 tools/）。

081 四个探针的真败因：`EnumObjectItems("", "", obj, detail)` 把 machine 传成
空串（被当成远程机器名）⇒ buffer size 错。本机对象名就是英文 "GPU Engine"，
machine 必须传 None。

读数三件套（对 082-A 驻留假设的直接证据）：
  * GPU Engine(pid_..._engtype_X)\\Utilization Percentage —— 按 pid x engtype
    的引擎利用率（否证/坐实"他占客户端"必守 26④）。**速率计数器**：同一
    查询句柄要 Collect >=2 次才有真值 ⇒ 引擎查询必须持久化（081 踩过的另一
    坑的镜像：每 tick 重建查询 = 永远只有首采，读数恒 0）。
  * GPU Process Memory(pid_...)\\{Local,Non Local,Dedicated,Shared,Total Committed}
    Usage —— **Non Local Usage = 该进程被 WDDM 放到系统内存的显存字节数**。
    注意：本臂 non_local 里含显式的 8 GiB kv-offloading 主机池（WDDM 把它记
    成该进程的 shared/non-local 面），判"权重页被降级"要看**动态增量**与
    跨条件差，不能裸读绝对值。
  * GPU Adapter Memory(luid_...)\\{Dedicated,Shared,Total Committed} —— 卡级
    对照（WDDM 下卡级 memory.used 是唯一可靠外置口径，必守 10）。

用法（cwd 必须不在记录仓，铁律 2）：
  python tools/pdh_gpu_engine.py sample --seconds 600 --interval 2.0 \
      --out G:\\qwen3.8model\\prod029_logs\\x_pdh.jsonl [--luid 0x..._0x...]

输出 jsonl：每 tick 一行 {"ts", "eng": {pid: {engtype: pct}},
"pmem": {pid: {...}}, "adapter": {luid: {...}}}，只记非零。
实例集每 REBUILD_TICKS 重建一次以捕捉新进程；重建后首个 tick 只做首采。
"""
from __future__ import annotations

import argparse
import json
import time

import win32pdh

OBJ_ENGINE = "GPU Engine"
OBJ_PMEM = "GPU Process Memory"
OBJ_ADAPTER = "GPU Adapter Memory"
REBUILD_TICKS = 30  # 每 30 个 tick 重建一次引擎实例集


def _collect_once(q: int, paths: dict[str, int]) -> dict[str, float]:
    out = {}
    for path, h in paths.items():
        try:
            out[path] = win32pdh.GetFormattedCounterValue(
                h, win32pdh.PDH_FMT_DOUBLE)[1]
        except Exception:  # noqa: BLE001  (进程消失/无数据是常态)
            pass
    return out


def _add_all(q: int, paths: list[str]) -> dict[str, int]:
    keep = {}
    for path in paths:
        try:
            keep[path] = win32pdh.AddCounter(q, path)
        except Exception:  # noqa: BLE001
            pass
    return keep


def _util_paths(pid_filter: str | None,
                luid_tag: str | None = None) -> list[str]:
    try:
        _, instances = win32pdh.EnumObjectItems(
            None, None, OBJ_ENGINE, win32pdh.PERF_DETAIL_WIZARD)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for inst in instances:
        if pid_filter and ("pid_%s_" % pid_filter) not in inst:
            continue
        if luid_tag and luid_tag not in inst:
            continue
        out.append("\\%s(%s)\\Utilization Percentage" % (OBJ_ENGINE, inst))
    return out


def _inst_of(paths: list[str]) -> dict[str, str]:
    return {p: p[p.find("(") + 1:p.find(")")] for p in paths}


class EngineUtilQuery:
    """持久化的 GPU Engine 利用率查询（速率计数器须同一句柄 Collect 两次）。"""

    def __init__(self, pid_filter: str | None = None,
                 luid: str | None = None) -> None:
        self.pid_filter = pid_filter
        self.luid_tag = ("luid_%s_phys_0" % luid) if luid else None
        self._q = None
        self._paths: list[str] = []
        self._handles: dict[str, int] = {}
        self._inst: dict[str, str] = {}
        self._tick = REBUILD_TICKS  # 立刻触发首次构建

    def _rebuild(self) -> None:
        self._close()
        self._paths = _util_paths(self.pid_filter, self.luid_tag)
        self._inst = _inst_of(self._paths)
        try:
            self._q = win32pdh.OpenQuery(None, 0)
            self._handles = _add_all(self._q, self._paths)
            win32pdh.CollectQueryData(self._q)  # 首采只做首律，不读值
        except Exception:  # noqa: BLE001
            self._q = None

    def _close(self) -> None:
        if self._q is not None:
            for h in self._handles.values():
                try:
                    win32pdh.RemoveCounter(h)
                except Exception:  # noqa: BLE001
                    pass
            try:
                win32pdh.CloseQuery(self._q)
            except Exception:  # noqa: BLE001
                pass
        self._q, self._handles = None, {}

    def sample(self) -> dict[str, dict[str, float]]:
        if self._tick >= REBUILD_TICKS:
            self._rebuild()
            self._tick = 0
        self._tick += 1
        if self._q is None:
            return {}
        try:
            win32pdh.CollectQueryData(self._q)
        except Exception:  # noqa: BLE001
            return {}
        fold: dict[str, dict[str, float]] = {}
        for path, val in _collect_once(self._q, self._handles).items():
            if not val:
                continue
            inst = self._inst.get(path, "")
            pid = inst.split("_")[1] if inst.startswith("pid_") else inst[:20]
            engtype = inst.split("engtype_")[-1] if "engtype_" in inst else "?"
            fold.setdefault(pid, {})[engtype] = round(
                fold.get(pid, {}).get(engtype, 0.0) + val, 2)
        return fold


def _adapter_snapshot() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    try:
        _, insts = win32pdh.EnumObjectItems(
            None, None, OBJ_ADAPTER, win32pdh.PERF_DETAIL_WIZARD)
        paths = {}
        for inst in insts:
            for ctr in ("Dedicated Usage", "Shared Usage", "Total Committed"):
                paths["\\%s(%s)\\%s" % (OBJ_ADAPTER, inst, ctr)] = (inst, ctr)
        vals = _collect_paths(paths)
        for path, val in vals.items():
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


def _collect_paths(paths: dict[str, tuple]) -> dict[str, float]:
    """一次性计数器（绝对量）：临时查询 + 单次 Collect 即可。"""
    if not paths:
        return {}
    q = None
    handles = {}
    try:
        q = win32pdh.OpenQuery(None, 0)
        handles = _add_all(q, list(paths))
        win32pdh.CollectQueryData(q)
        return _collect_once(q, handles)
    except Exception:  # noqa: BLE001
        return {}
    finally:
        for h in handles.values():
            try:
                win32pdh.RemoveCounter(h)
            except Exception:  # noqa: BLE001
                pass
        if q is not None:
            try:
                win32pdh.CloseQuery(q)
            except Exception:  # noqa: BLE001
                pass


def sample_once(pmem_pid: str | None, luid: str | None,
                eng: EngineUtilQuery | None = None) -> dict:
    """eng 传持久查询（推荐）；pmem/adapter 是绝对量，临时查询无碍。"""
    eng_fold = eng.sample() if eng is not None else {}
    luid_tag = ("luid_%s_phys_0" % luid) if luid else None
    # 进程显存（Non Local = WDDM 放在系统内存里的部分）
    pmem_fold: dict[str, dict[str, float]] = {}
    try:
        _, insts = win32pdh.EnumObjectItems(
            None, None, OBJ_PMEM, win32pdh.PERF_DETAIL_WIZARD)
        paths = {}
        for inst in insts:
            if pmem_pid and ("pid_%s_" % pmem_pid) not in inst:
                continue
            if luid_tag and luid_tag not in inst:
                continue
            pid = inst.split("_")[1]
            for ctr in ("Local Usage", "Non Local Usage", "Shared Usage",
                        "Total Committed"):
                paths["\\%s(%s)\\%s" % (OBJ_PMEM, inst, ctr)] = (pid, ctr)
        vals = _collect_paths(paths)
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
    return {"eng": eng_fold, "pmem": pmem_fold,
            "adapter": _adapter_snapshot()}


def cmd_sample(args) -> int:
    fh = open(args.out, "a", encoding="utf-8")
    luid = args.luid
    if luid is None and not args.no_autolid:
        snap = _adapter_snapshot()
        best = sorted(snap.items(), key=lambda kv: -kv[1].get("dedicated", 0))
        if best:
            luid = best[0][0]
    engq = EngineUtilQuery(None, luid)
    print("sampling %ss every %ss -> %s (luid=%s)"
          % (args.seconds, args.interval, args.out, luid or "ALL"), flush=True)
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        row = sample_once(args.pid, luid, engq)
        row["ts"] = time.strftime("%H:%M:%S")
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        n += 1
        time.sleep(args.interval)
    fh.close()
    engq._close()
    print("done, %d rows" % n)
    return 0


def cmd_oneshot(args) -> int:
    engq = EngineUtilQuery(args.pid, args.luid)
    engq.sample()  # 首采
    time.sleep(1.2)
    row = sample_once(args.pid, args.luid, engq)
    engq._close()
    print(json.dumps(row, ensure_ascii=False, indent=1)[:4000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seconds", type=float, default=600)
    s.add_argument("--interval", type=float, default=2.0)
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
