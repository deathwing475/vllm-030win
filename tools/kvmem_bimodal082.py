# -*- coding: utf-8 -*-
"""082-A 分析器：锚定单一 launch 配置的调用时长双峰统计（跨条件比较）。

锚 = 081 定过的那一支 humming WNA16 GEMM：
  名含 Shape<0,34816,5120>（gate/up 族）、grid [70, 1, 1]、block [384, 1, 1]。
  081 实测同一配置内快簇 ~5.15 ms / 慢簇 ~77.67 ms（15.1x），每次前向
  （= 64 层 x 1 次）稳定 19-23 个慢调用且位次固定。

082-A 的判据（三值结论的原料，逐条件报）：
  n / 快簇中位 / 慢簇中位 / 比值 / P50,P90,P95 / 慢调用占比（计数与时间）/
  每前向（64 调用一组）慢个数分布 / 慢位次直方图与位次稳定度。

用法（cwd 不在记录仓）：
  python tools/kvmem_bimodal082.py --traces "G:\\...\\prof\\*.trace.json.gz" \
      --label A0_b1 [--out xxx.json] [--json-only]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict

ANCHOR_NAME_KEY = "34816"
ANCHOR_GRID = "[70, 1, 1]"
ANCHOR_BLOCK = "[384, 1, 1]"
SLOW_MS = 30.0  # 081 的分簇门槛（快簇 5.15 / 慢簇 77.67，隔着量级，门槛稳健）
GROUP = 64      # 每次前向的锚调用数（64 层 x 1）


def load_events(path: str) -> list[dict]:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8", "replace")).get("traceEvents") or []


def pick_anchor(events: list[dict]) -> tuple[str, str, str] | None:
    """优先取 081 锚配置；缺席时取同族最大 n 配置并在输出里标 fallback。"""
    cfg: dict[tuple[str, str, str], int] = Counter()
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        if ANCHOR_NAME_KEY not in name:
            continue
        a = e.get("args") or {}
        cfg[(name[:90], str(a.get("grid")), str(a.get("block")))] += 1
    if not cfg:
        return None
    want = next((k for k in cfg
                 if k[1] == ANCHOR_GRID and k[2] == ANCHOR_BLOCK), None)
    if want is None:
        want = max(cfg, key=lambda k: cfg[k])
    return want


def pct(sorted_v: list[float], q: float) -> float:
    return sorted_v[min(len(sorted_v) - 1, int(q * len(sorted_v)))]


def analyze_trace(path: str) -> dict:
    events = load_events(path)
    anchor = pick_anchor(events)
    if anchor is None:
        return {"trace": os.path.basename(path), "error": "anchor family absent"}
    name, grid, block = anchor
    corr_map = {}
    for e in events:
        if e.get("cat") == "kernel" and "args" in e:
            a = e["args"]
            corr_map[a.get("correlation")] = (str(a.get("grid")),
                                              str(a.get("block")))
    sel = sorted((e for e in events
                  if e.get("cat") == "kernel"
                  and e.get("name", "")[:90] == name
                  and corr_map.get((e.get("args") or {}).get("correlation"))
                  == (grid, block)),
                 key=lambda e: e["ts"])
    durs = [e["dur"] / 1e3 for e in sel]  # ms
    if not durs:
        return {"trace": os.path.basename(path), "error": "anchor config empty",
                "anchor": [name[:60], grid, block]}
    fast = sorted(x for x in durs if x < SLOW_MS)
    slow = sorted(x for x in durs if x >= SLOW_MS)
    sd = sorted(durs)
    fallback = not (grid == ANCHOR_GRID and block == ANCHOR_BLOCK)

    # 每前向（GROUP 个一组）的慢个数 + 位次
    grp_slow_n, pos_hist = [], defaultdict(int)
    for i in range(0, len(durs) - len(durs) % GROUP, GROUP):
        g = durs[i:i + GROUP]
        nslow = sum(1 for x in g if x >= SLOW_MS)
        grp_slow_n.append(nslow)
        for j, x in enumerate(g):
            if x >= SLOW_MS:
                pos_hist[j] += 1
    n_grp = len(grp_slow_n)
    modal = Counter(grp_slow_n).most_common(1)[0][0] if grp_slow_n else None
    stable = (sum(1 for x in grp_slow_n if x == modal) / n_grp
              if grp_slow_n else None)
    top_pos = Counter(pos_hist).most_common(8)

    return {
        "trace": os.path.basename(path),
        "anchor": {"name": name[:70], "grid": grid, "block": block,
                   "fallback": fallback},
        "n_calls": len(durs),
        "n_forwards": n_grp,
        "fast_cluster": {"n": len(fast),
                         "median_ms": round(fast[len(fast) // 2], 3)
                         if fast else None},
        "slow_cluster": {"n": len(slow),
                         "median_ms": round(slow[len(slow) // 2], 3)
                         if slow else None,
                         "sum_s": round(sum(slow) / 1e3, 2)},
        "ratio": round(slow[len(slow) // 2] / fast[len(fast) // 2], 2)
        if fast and slow else None,
        "p50_ms": round(pct(sd, 0.5), 3), "p90_ms": round(pct(sd, 0.9), 3),
        "p95_ms": round(pct(sd, 0.95), 3), "max_ms": round(sd[-1], 2),
        "slow_share_count": round(len(slow) / len(durs), 4),
        "slow_share_time": round(sum(slow) / sum(durs), 4) if sum(durs) else 0,
        "bands": {"lt10": sum(1 for x in durs if x < 10),
                  "b10_30": sum(1 for x in durs if 10 <= x < 30),
                  "b30_100": sum(1 for x in durs if 30 <= x < 100),
                  "ge100": sum(1 for x in durs if x >= 100)},
        "per_forward_slow": {"n_groups": n_grp, "modal_count": modal,
                             "stable_frac": round(stable, 3)
                             if stable is not None else None,
                             "min": min(grp_slow_n) if grp_slow_n else None,
                             "max": max(grp_slow_n) if grp_slow_n else None},
        "top_slow_positions": [[p, c, round(c / n_grp, 3)]
                               for p, c in top_pos] if n_grp else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True,
                    help="文件/目录/glob，多个用逗号分隔")
    ap.add_argument("--label", default="", help="本次批次标签（写进汇总）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    paths: list[str] = []
    for spec in args.traces.split(","):
        spec = spec.strip()
        if os.path.isdir(spec):
            paths += sorted(glob.glob(os.path.join(spec, "*.trace.json.gz")))
            paths += sorted(glob.glob(os.path.join(spec, "*.trace.json")))
        elif any(ch in spec for ch in "*?["):
            paths += sorted(glob.glob(spec))
        else:
            paths.append(spec)
    reports = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            reports.append(analyze_trace(p))
        except Exception as exc:  # noqa: BLE001
            reports.append({"trace": os.path.basename(p),
                            "error": repr(exc)[:200]})
    blob = {"label": args.label, "reports": reports}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
    if not args.json_only:
        for r in reports:
            if "error" in r:
                print("%-46s ERROR %s" % (r["trace"], r["error"]))
                continue
            a = r["anchor"]
            print("%-46s anchor(fb=%s) grid=%s block=%s" %
                  (r["trace"], a["fallback"], a["grid"], a["block"]))
            print("   n=%d fwd=%d  fast_med=%s ms x%d  slow_med=%s ms x%d "
                  "ratio=%s" % (
                      r["n_calls"], r["n_forwards"],
                      r["fast_cluster"]["median_ms"], r["fast_cluster"]["n"],
                      r["slow_cluster"]["median_ms"], r["slow_cluster"]["n"],
                      r["ratio"]))
            print("   P50=%.2f P90=%.2f P95=%.2f max=%.1f  slow_share cnt=%.1f%% "
                  "time=%.1f%%" % (
                      r["p50_ms"], r["p90_ms"], r["p95_ms"], r["max_ms"],
                      100 * r["slow_share_count"], 100 * r["slow_share_time"]))
            pf = r["per_forward_slow"]
            print("   per-fwd slow: modal=%s x%s(min %s, max %s)  bands=%s"
                  % (pf["modal_count"], pf["stable_frac"], pf["min"],
                     pf["max"], r["bands"]))
            print("   top slow pos: %s"
                  % r["top_slow_positions"][:5])
    print("TOTAL %d traces" % len(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
