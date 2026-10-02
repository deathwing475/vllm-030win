# -*- coding: utf-8 -*-
"""步骤 081：把 torch profiler 的 chrome trace 折成"整页前向的钱在 GPU 上还是
在提交空隙里"的读数（纯离线，不 import vllm、不碰 GPU）。

为什么这一步需要它
------------------
079 的 ``[KVTIME]`` 台账只能拆到"连接器哪一段"（13/14 段），080 的 py-spy 只能
拆到"哪个 Python 行阻塞"，两者都**看不见 C/驱动帧**，所以 080 只能说到
"整页前向的提交-完成握手被拖住 ×7-8"而定不到对象。torch.profiler（Kineto+CUPTI）
在 Windows + cu13 上实测能给出带时间戳的 ``cat=="kernel"`` 事件（081 前提闸门），
于是第一次拿到：

  * **每个 kernel 的时长**（执行侧）与 **kernel 之间的空隙**（提交/完成侧）；
  * **host 侧 API 的墙钟**（``cudaLaunchKernel`` / ``cudaStreamSynchronize`` /
    ``cudaMemcpyAsync`` …）以及 **launch→kernel start 的提交延迟**；
  * **kernel 名单**本身 —— 快态 boot 与慢态 boot 的名单差 = 044 挂账的
    "tactic 非确定"的直接判据（080 明确说缓存取证排除不了它）。

口径（三条，别读错）
--------------------
① **busy = 区间并集，不是求和**。多流（本引擎 8 个 PID/多流）下 Σdur 会重复计数，
   所以 device busy 用合并区间；"空隙" = 时间跨度 − busy（只在单流时等于相邻差）。
② **CUDA graph 里的 kernel 仍然逐个带时间戳**，图边界**不是**空隙；prefill 不进图
   （``CudagraphDispatcher`` 对 >max_cudagraph_capture_size 返回 NONE），decode 进图。
   所以分块时不能把"图回放"当成提交停顿。
③ **切块靠大空隙**：一个请求结束到下一个请求开始之间会有整段空闲，用
   ``--split-gap``（默认 0.5 s）切成"每请求一块"，再按块内 device busy 排序；
   块序 ≠ 请求序时以 ``--cadence`` 的客户端节拍对齐。

用法
----
    python tools/kvmem_trace_budget.py --trace <文件或目录> [--out json]
        [--split-gap 0.5] [--min-chunk-ms 50] [--top 15] [--label b4-slow]
    python tools/kvmem_trace_budget.py --compare A.json.gz B.json.gz [--top 20]

``--compare`` 只回答一件事：**两次的 kernel 名单/同名 kernel 的平均时长差多少**，
用来把"同 kernel 变慢"（执行/等待侧）与"换了 kernel"（tactic 侧）分开。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import re
import statistics as st
from collections import defaultdict

DEV_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
API_SPLIT = [
    ("launch", re.compile(r"cudaLaunchKernel|cuLaunchKernel")),
    ("memcpy_async", re.compile(r"cudaMemcpyAsync|cuMemcpyAsync")),
    ("memset_async", re.compile(r"cudaMemsetAsync|cuMemsetAsync")),
    ("sync", re.compile(r"cudaStreamSynchronize|cudaDeviceSynchronize|"
                        r"cudaEventSynchronize|cuStreamSynchronize")),
    ("event_query", re.compile(r"cudaEventQuery|cudaStreamQuery")),
    ("graph", re.compile(r"cudaGraphLaunch|cuGraphLaunch")),
    ("alloc", re.compile(r"cudaMalloc|cudaFree|cudaMallocAsync")),
]


def load_trace(path: str) -> dict:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = json.loads(io.BytesIO(raw).read().decode("utf-8", "replace"))
    if isinstance(data, dict):
        ev = data.get("traceEvents", [])
    else:
        ev = data
    return {"events": ev, "path": path}


def find_traces(spec: str) -> list[str]:
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "*.trace.json"))
                      + glob.glob(os.path.join(spec, "*.trace.json.gz"))
                      + glob.glob(os.path.join(spec, "*.pt.trace.json*")))
    return sorted(glob.glob(spec))


def bucket_api(name: str) -> str:
    for label, rx in API_SPLIT:
        if rx.search(name or ""):
            return label
    return "other_api"


def merge_intervals(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def coverage(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in merge_intervals(iv))


def analyze(path: str, args) -> dict:
    t = load_trace(path)
    dev, api = [], []
    corr_api: dict[int, tuple[float, str]] = {}
    for e in t["events"]:
        cat = e.get("cat")
        if cat in DEV_CATS and "dur" in e:
            stream = (e.get("args") or {}).get("stream")
            corr = (e.get("args") or {}).get("correlation")
            dev.append({"name": e.get("name", ""), "ts": e["ts"] / 1e6,
                        "dur": e["dur"] / 1e6, "cat": cat, "stream": stream,
                        "corr": corr,
                        "bytes": (e.get("args") or {}).get("bytes")})
        elif cat in ("cuda_runtime", "cuda_driver") and "dur" in e:
            a = e.get("args") or {}
            rec = {"name": e.get("name", ""), "ts": e["ts"] / 1e6,
                   "dur": e["dur"] / 1e6, "corr": a.get("correlation"),
                   "bucket": bucket_api(e.get("name", ""))}
            api.append(rec)
            if rec["corr"] is not None:
                prev = corr_api.get(rec["corr"])
                if prev is None or rec["ts"] < prev[0]:
                    corr_api[rec["corr"]] = (rec["ts"], rec["name"])

    if not dev:
        return {"trace": os.path.basename(path), "error": "no device events"}

    dev.sort(key=lambda d: d["ts"])
    span0, span1 = dev[0]["ts"], max(d["ts"] + d["dur"] for d in dev)
    busy = coverage([(d["ts"], d["ts"] + d["dur"]) for d in dev])

    # host API 分桶
    api_by: dict[str, list[float]] = defaultdict(list)
    for a in api:
        if a["bucket"] in ("launch", "sync", "memcpy_async", "graph",
                           "event_query", "alloc"):
            api_by[a["bucket"]].append(a["dur"])
    api_sum = {k: round(sum(v), 3) for k, v in api_by.items()}
    api_n = {k: len(v) for k, v in api_by.items()}

    # 提交延迟：kernel.ts − 对应 launch 的 host ts
    sub_lat = []
    for d in dev:
        if d["cat"] != "kernel":
            continue
        hit = corr_api.get(d["corr"])
        if hit:
            sub_lat.append(d["ts"] - hit[0])
    sub_stats = {}
    if sub_lat:
        s = sorted(sub_lat)
        sub_stats = {
            "n": len(s),
            "median_ms": round(st.median(s) * 1e3, 3),
            "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))] * 1e3, 3),
            "max_ms": round(s[-1] * 1e3, 3),
            "neg": sum(1 for x in s if x < 0),
        }

    # 切块（每请求一块）
    chunks, cur, last_end = [], [], span0
    for d in dev:
        if cur and d["ts"] - last_end > args.split_gap:
            if (last_end - cur[0]["ts"]) * 1e3 >= args.min_chunk_ms:
                chunks.append(cur)
            cur = []
        cur.append(d)
        last_end = max(last_end, d["ts"] + d["dur"])
    if cur and (last_end - cur[0]["ts"]) * 1e3 >= args.min_chunk_ms:
        chunks.append(cur)

    def chunk_stats(c: list[dict]) -> dict:
        t0 = c[0]["ts"]
        t1 = max(x["ts"] + x["dur"] for x in c)
        busy_c = coverage([(x["ts"], x["ts"] + x["dur"]) for x in c])
        per: dict[str, float] = defaultdict(float)
        for x in c:
            per[x["name"][:70]] += x["dur"]
        top = sorted(per.items(), key=lambda kv: -kv[1])[:args.top]
        return {
            "wall_s": round(t1 - t0, 3), "busy_s": round(busy_c, 3),
            "gap_s": round((t1 - t0) - busy_c, 3),
            "busy_pct": round(100.0 * busy_c / (t1 - t0), 1) if t1 > t0 else 0,
            "kernels": sum(1 for x in c if x["cat"] == "kernel"),
            "memcpy": sum(1 for x in c if x["cat"] == "gpu_memcpy"),
            "memcpy_bytes": sum(x.get("bytes") or 0 for x in c
                                if x["cat"] == "gpu_memcpy"),
            "top": [(round(v, 4), k) for k, v in top],
        }

    cs = [chunk_stats(c) for c in chunks]
    cs.sort(key=lambda x: -x["wall_s"])

    # 全 trace 的 kernel 汇总（名单 + 平均时长 = 比对的单位）
    per_all: dict[str, list[float]] = defaultdict(list)
    for d in dev:
        if d["cat"] == "kernel":
            # 内部一律用秒；这里换算成毫秒再存，别让字段名骗人
            per_all[d["name"][:90]].append(d["dur"] * 1e3)
    kernels = [{"name": k, "n": len(v), "sum_s": round(sum(v) / 1e3, 3),
                "mean_ms": round(st.median(v), 4),
                "max_ms": round(max(v), 4)}
               for k, v in sorted(per_all.items(),
                                  key=lambda kv: -sum(kv[1]))]

    busy_ratio = busy / (span1 - span0) if span1 > span0 else 0.0
    if busy_ratio >= 0.85:
        verdict = ("执行型：device 覆盖率 %.1f%% ⇒ 时间几乎都在 kernel 里；"
                   "要问的是『同一 kernel 为什么更慢』" % (100 * busy_ratio))
    elif busy_ratio <= 0.55:
        verdict = ("空隙型：device 覆盖率 %.1f%% ⇒ 时间几乎不在 kernel 里，"
                   "而在提交/完成之间；看 launch 的 host 墙钟与 sub_lat"
                   % (100 * busy_ratio))
    else:
        verdict = ("混合型：device 覆盖率 %.1f%% ⇒ 两块都有份，必须分桶报数，"
                   "不强行定罪" % (100 * busy_ratio))

    return {
        "trace": os.path.basename(path),
        "events": len(t["events"]),
        "device_events": len(dev),
        "streams": sorted({str(d["stream"]) for d in dev}),
        "span_s": round(span1 - span0, 3),
        "busy_s": round(busy, 3),
        "gap_s": round((span1 - span0) - busy, 3),
        "busy_pct": round(100 * busy_ratio, 1),
        "api_sum_s": api_sum, "api_n": api_n,
        "submit_latency": sub_stats,
        "chunks": cs[:10], "n_chunks": len(cs),
        "kernels": kernels[:40],
        "verdict": verdict,
    }


def compare(paths: list[str], args) -> dict:
    """同名 kernel 的中位时长比 = tactic 与执行速度的分岔判据。"""
    sets = {}
    for p in paths:
        r = analyze(p, args)
        sets[p] = r
    names = []
    for p, r in sets.items():
        names.append({k["name"]: k for k in r["kernels"]})
    if len(names) != 2:
        print("compare 需要恰好两个 trace")
        return {}
    a, b = names
    common = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"A={list(sets)[0]}\nB={list(sets)[1]}")
    print(f"共同 kernel {len(common)} / 只在 A {len(only_a)} / 只在 B {len(only_b)}")
    if only_a or only_b:
        print("  只在 A:", [n[:60] for n in only_a[:8]])
        print("  只在 B:", [n[:60] for n in only_b[:8]])
    rows = []
    for n in common:
        ma, mb = a[n]["mean_ms"], b[n]["mean_ms"]
        rows.append((mb / ma if ma else 0, n, ma, mb, a[n]["n"], b[n]["n"]))
    rows.sort(key=lambda r: -abs(r[0] - 1.0) * max(r[2], r[3]))
    print(f"\n{'B/A mean':>9} {'meanA_ms':>9} {'meanB_ms':>9} {'nA':>6} {'nB':>6}  kernel")
    for ratio, n, ma, mb, na, nb in rows[:args.top]:
        print(f"{ratio:9.2f} {ma:9.4f} {mb:9.4f} {na:6d} {nb:6d}  {n[:64]}")
    return {"common": len(common), "only_a": only_a, "only_b": only_b,
            "rows": [{"kernel": n, "mean_a_ms": ma, "mean_b_ms": mb,
                      "ratio_b_over_a": round(r, 3)}
                     for r, n, ma, mb, _, _ in rows]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="文件或目录或 glob")
    ap.add_argument("--compare", action="store_true",
                    help="与 --trace-b 比对同名 kernel")
    ap.add_argument("--trace-b", default=None)
    ap.add_argument("--split-gap", type=float, default=0.5,
                    help="大于此空闲（秒）切成下一块；一个请求的两块之间就是这个")
    ap.add_argument("--min-chunk-ms", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = find_traces(args.trace)
    if not paths:
        print("没有找到 trace 文件：", args.trace)
        return 2
    if args.compare:
        pb = find_traces(args.trace_b or "")
        if not pb:
            print("缺少 --trace-b")
            return 2
        compare([paths[0], pb[0]], args)
        return 0

    reports = []
    for p in paths:
        r = analyze(p, args)
        reports.append(r)
        print("=" * 78)
        print("trace %s%s" % (args.label or "", os.path.basename(p)))
        if r.get("error"):
            print("  ！", r["error"])
            continue
        print(f"  事件 {r['events']} / device {r['device_events']} / "
              f"流 {r['streams']}")
        print(f"  跨度 {r['span_s']} s  device busy {r['busy_s']} s "
              f"({r['busy_pct']}%)  空隙 {r['gap_s']} s")
        print(f"  host API 墙钟合计: {r['api_sum_s']}  次数: {r['api_n']}")
        print(f"  提交延迟 launch→kernel start: {r['submit_latency']}")
        print(f"  切块 {r['n_chunks']} 个（按 wall_s 降序前 {len(r['chunks'])}）")
        for i, c in enumerate(r["chunks"]):
            print(f"    #{i + 1} wall={c['wall_s']}s busy={c['busy_s']}s "
                  f"gap={c['gap_s']}s({100 - c['busy_pct']:.0f}%) "
                  f"kernels={c['kernels']} memcpy={c['memcpy']} "
                  f"({c['memcpy_bytes'] / 2**20:.1f} MiB)")
            for v, k in c["top"][:5]:
                print(f"        {v:8.4f} s  {k[:66]}")
        print("  top kernel（Σdur 降序）：")
        for k in r["kernels"][:args.top]:
            print(f"    {k['sum_s']:8.3f} s  n={k['n']:6d} "
                  f"med={k['mean_ms']:8.4f} ms  max={k['max_ms']:8.4f} ms  "
                  f"{k['name'][:64]}")
        print("  判定:", r["verdict"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"label": args.label, "reports": reports}, fh,
                      ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
