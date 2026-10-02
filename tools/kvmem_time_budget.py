# -*- coding: utf-8 -*-
"""步骤 079：把 ``[KVTIME]`` 台账折算成"一步的时间到底在哪"的四分表（纯离线）。

口径
----
一条 ``[KVTIME]`` 是一个 ``VLLM_KVMEM_TIMING_EVERY`` 次 ``wait_for_save`` 的窗口，
带 ``dt``（该窗口的墙钟）与各段的**绝对累加 + 本窗口增量**。本工具把 ``dt`` 拆成五块：

  gpu_wait   = d_sync      drain() 里**第一次** ``.cpu()`` 返回之前的时间
                          = 等本步已经异步排上的算子跑完（模型自己的计算）
  gpu_copy   = d_copies    drain() 里其余 47 次 D2H 的净搬运
  host_py    = d_fold + d_storing - d_selftest - d_copy
                          连接器里的 Python：位置掩码 / 索引折叠 / 组装 copy 条目
  selftest   = d_selftest   roundtrip 自测（内含 ``torch.cuda.synchronize()``）
  copy_issue = d_copy       ``ops.swap_blocks_batch`` 的发射（不含传输）
  record     = d_rec       ⭐``capture.record`` custom op 本体 —— 它在 **execute_model
                          内部**跑（每个 full-attention 层一次），``wait_for_save``
                          窗口天生看不见它。079 把它记进了 ``outside`` 并据此说
                          "连接器之外"，是必守 24①那个坑（未计量的段读起来像无罪）；
                          步骤 080 补了这一段，py-spy 同时独立指到同一行。
  outside    = dt - accounted - record
                          剩下才真是连接器之外：调度、图重放、模型自身计算

两个必须记住的陷阱
------------------
* ``gpu_wait`` 不是"拷贝慢"。它是"等 GPU 把这一页算完"。b1 之前只有一个 ``drain``
  数，谁都可能把它读成"D2H 搬运贵" —— 所以才在 capture 里拆了 sync/copies。
* 只有 ``dsteps >= --min-steps`` 且 ``dt > 0`` 的窗口参与统计；首窗 ``dt=0`` 是设计如
  此（窗口尚未建立），不许进任何均值。

用法
----
    python tools/kvmem_time_budget.py --engine-log G:\\qwen3.8model\\prod029_logs\\step079_b1_timing.out.log
    python tools/kvmem_time_budget.py --engine-log ...log --cadence ...json --out ...json
"""
import argparse
import json
import os
import re
import statistics as st
import sys

# 一条台账段的形状：<key>=<绝对值> d<key>=<本窗口增量> n<计数> dn<计数增量>。
# 反向引用 \\1 是关键：段名 drain 与它的增量 ddrain 用前缀切会互相串味。
# tools/kvmem_timing_unit_test.py 用的就是这个正则（单测保证两者不漂）。
SEG_RE = re.compile(r"(?:^|\s)([a-z_]+)=(-?\d+\.\d+) d\1=(-?\d+\.\d+) n(\d+) dn(\d+)")
HEAD_RE = re.compile(
    r"wall=(-?\d+\.\d+) dt=(-?\d+\.\d+) dsteps=(\d+).*?"
    r"acc=(-?\d+\.\d+).*?bytes=([\d.]+)MiB.*?"
    r"copy_calls=(\d+) entries=(\d+) busy=(\d+)"
)
LOADED_RE = re.compile(r"\[KVTIME\] patch loaded: (.*?)\s*$")


def parse(path: str) -> tuple[list[dict], list[str]]:
    windows = []
    loaded = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "patch loaded" in line:
                loaded.append(LOADED_RE.search(line).group(1))
                continue
            if "[KVTIME] wall=" not in line:
                continue
            head = HEAD_RE.search(line)
            if not head:
                continue
            segs = {m.group(1): float(m.group(3)) for m in SEG_RE.finditer(line)}
            windows.append({
                "wall": float(head.group(1)),
                "dt": float(head.group(2)),
                "dsteps": int(head.group(3)),
                "acc_logged": float(head.group(4)),
                "bytes_mib": float(head.group(5)),
                "copy_calls": int(head.group(6)),
                "entries": int(head.group(7)),
                "busy": int(head.group(8)),
                "d": segs,
            })
    return windows, loaded


def split_of(w: dict) -> dict:
    d = w["d"]
    dt = w["dt"]
    storing = d.get("storing", 0.0)
    selftest = d.get("selftest", 0.0)
    copy_issue = d.get("copy", 0.0)
    out = {
        "dt": dt,
        "steps": w["dsteps"],
        "gpu_wait": d.get("sync", 0.0),
        "gpu_copy": d.get("copies", 0.0),
        "selftest": selftest,
        "copy_issue": copy_issue,
        "host_py": d.get("fold", 0.0) + max(0.0, storing - selftest - copy_issue),
        "score_pure": max(0.0, d.get("score", 0.0) - d.get("bake", 0.0)),
        "bake": d.get("bake", 0.0),
        "snap": d.get("snap", 0.0),
        "load": d.get("load", 0.0),
        "save": d.get("save", 0.0),
        # step 080: NOT part of `save` (record runs in the forward), so it is
        # deliberately outside `accounted` -- accounted/save stays the
        # self-consistency check on the wait_for_save window.
        "record": d.get("rec", 0.0),
    }
    out["accounted"] = (out["gpu_wait"] + out["gpu_copy"] + out["selftest"]
                        + out["copy_issue"] + out["host_py"] + out["score_pure"]
                        + out["bake"] + out["snap"] + out["load"])
    out["outside"] = dt - out["accounted"] - out["record"]
    out["residual"] = out["save"] - out["accounted"]  # 记账自洽性检查，应≈0
    return out


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-log", required=True)
    ap.add_argument("--cadence", default=None,
                    help="kvmem_ingest_cadence.py 的 --out JSON，用来交叉校验 dt/步")
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", default=None, help="只作标签")
    args = ap.parse_args()

    windows, loaded = parse(args.engine_log)
    name = args.boot or os.path.basename(args.engine_log)
    print("=" * 78)
    print(f"boot {name}")
    if not loaded:
        print("  ！日志里没有 [KVTIME] patch loaded 行 —— 补丁没在跑，别用这份日志")
    else:
        print(f"  loaded: {loaded[0]}")
    print(f"  台账窗口 {len(windows)} 条（含 dt=0 的首窗）")
    usable = [w for w in windows if w["dt"] > 0 and w["dsteps"] >= args.min_steps]
    if not usable:
        print(f"  ！没有可用窗口（需 dt>0 且 dsteps>={args.min_steps}）"
              f"：把 VLLM_KVMEM_TIMING_EVERY 调小或多发几条请求")
        return 2
    # 两类窗口必须分开统计：连接器在 decode 步里根本不在场（wait_for_save 首行
    # 就 return），把两堆混在一起取中位会假得出"时间全在连接器之外"的结论。
    active = [w for w in usable if w["d"].get("save", 0.0) > 0.05]
    idle = [w for w in usable if w["d"].get("save", 0.0) <= 0.05]
    cad = None
    if args.cadence:
        with open(args.cadence, encoding="utf-8") as fh:
            cad = json.load(fh)
        print(f"  客户端节拍: {cad.get('s_per_page_step', 0):.3f} s/页步 "
              f"ttft={cad.get('ttft_s')} "
              f"engine_drain中位={cad.get('engine_drain', {}).get('median_s')}")
    keys = ("gpu_wait", "gpu_copy", "host_py", "score_pure", "selftest",
            "copy_issue", "bake", "snap", "load", "record", "outside")
    ref = (cad or {}).get("s_per_page_step") if args.cadence else None
    if ref:
        slow_w = [w for w in active if w["dt"] / w["dsteps"] > 2.0 * ref]
        active = [w for w in active if w["dt"] / w["dsteps"] <= 2.0 * ref]
        if slow_w:
            print(f"  剔除 {len(slow_w)} 个跨空闲窗口（dt/步 = "
                  + ", ".join(f"{w['dt'] / w['dsteps']:.2f}" for w in slow_w)
                  + f" s > 2x 客户端 {ref:.3f} s）")
        # 整页步与短请求步（8k 探针那种 < 1456 收尾块）必须分开：前者 dt/步 是
        # 1.3-1.5 s、打分占比≈0，后者 0.3-0.4 s、打分占比 27-38%。混着取中位会
        # 同时污染 outside% 和与客户端节拍的交叉校验。
        long_w = [w for w in active if w["dt"] / w["dsteps"] >= 0.8 * ref]
        short_w = [w for w in active if w["dt"] / w["dsteps"] < 0.8 * ref]
        active = long_w
    else:
        short_w = []
        print("  ！未给 --cadence ⇒ 无法剔除跨空闲窗口，outside 只能当上界看")
    print(f"  窗口分类：整页步 {len(active)} 个 / 短请求步 {len(short_w)} 个 / "
          f"纯 decode（或空闲）{len(idle)} 个")

    def report(group: list[dict], label: str, judge: bool) -> dict:
        if not group:
            print(f"\n  === [{label}] === 无窗口")
            return {}
        rows = [split_of(w) for w in group]
        print(f"\n  === [{label}] ===")
        print(f"  {'窗口':<6s}{'步':>4s}{'dt':>8s}{'dt/步':>8s} "
              + " ".join(f"{k:>10s}" for k in keys))
        for i, s in enumerate(rows):
            per = s["dt"] / s["steps"] if s["steps"] else 0.0
            print(f"  {i + 1:<6d}{s['steps']:>4d}{s['dt']:>8.2f}{per:>8.3f} "
                  + " ".join(f"{pct(s[k], s['dt']):>9.1f}%" for k in keys))
        med = {k: st.median([pct(r[k], r["dt"]) for r in rows]) for k in keys}
        per_step = {k: st.median([r[k] / r["steps"] for r in rows]) for k in keys}
        per_step["dt"] = st.median([r["dt"] / r["steps"] for r in rows])
        print("  占比中位: " + "  ".join(f"{k}={med[k]:.1f}%" for k in keys))
        print("  每步秒数: " + "  ".join(
            f"{k}={per_step[k]:.3f}s" for k in list(keys) + ["dt"]))
        if cad and judge and cad.get("s_per_page_step"):
            print(f"  交叉校验: 台账 dt/步 {per_step['dt']:.3f} s vs 客户端 "
                  f"{cad['s_per_page_step']:.3f} s（同量级才可用；背离先解释）")
        resid = max(abs(r["residual"]) for r in rows)
        tol = 0.05 * max(r["dt"] for r in rows)
        print(f"  记账自洽性: |save - accounted| 最大 {resid:.3f} s "
              f"{'（可接受）' if resid <= tol else '（偏大：有段没计量，先修口径再定罪）'}")
        if not judge:
            return {"median_pct": med, "median_seconds_per_step": per_step,
                    "max_residual_s": resid, "windows": len(rows)}
        gw, gc, hp = med["gpu_wait"], med["gpu_copy"], med["host_py"]
        oth, sp = med["outside"], med["score_pure"]
        rec = med.get("record", 0.0)
        if rec >= 50.0:
            v = ("(d) capture.record 为主 —— 连接器在 **前向内部** 的那一段（raw-K "
                 "暂存 / M-RoPE 同轴检查）吃掉了这一步；它不在 wait_for_save 窗口里，"
                 "079 的口径把它记成了 outside")
        elif oth >= 50.0:
            v = ("(c) 连接器之外为主 —— 时间在这一页的模型前向/调度/图侧，"
                 "连接器内没有可归因的大头")
        elif gw >= 50.0:
            v = ("(a) 等本步算子完成为主 —— GPU 真在算这一页，"
                 "修拷贝形态（合批 / GPU 化烘焙）不会有用")
        elif hp >= 25.0:
            v = "(b) 连接器 Python（折叠/建条目）为主 —— 先降 dfold 与 dstoring"
        elif sp >= 25.0:
            v = "(b) 打分为主 —— 目标在 index.score 的每步开销"
        elif gc >= 25.0:
            v = "(b-1) drain 的净 D2H 搬运为主 —— 合并大拷贝 + pinned 双缓冲"
        elif med["selftest"] >= 25.0:
            v = "(b) roundtrip 自测的同步为主 —— 自测降频即可"
        else:
            v = "混合，无单一主导项 —— 报数字，不强行定罪"
        print(f"  判定: {v}")
        return {"median_pct": med, "median_seconds_per_step": per_step,
                "max_residual_s": resid, "verdict": v, "windows": len(rows)}

    pre = report(active, "整页步（连接器在场）", True)
    shr = report(short_w, "短请求步（8k 探针那种收尾块）", False)
    dec = report(idle, "decode 步（连接器不在场）", False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "boot": name,
                "loaded": loaded[:1],
                "cadence": cad,
                "prefill": pre,
                "short_steps": shr,
                "decode": dec,
            }, fh, ensure_ascii=False, indent=2)
        print(f"  wrote {args.out}")
    return 0
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "boot": name,
                "loaded": loaded[:1],
                "windows": [split_of(w) for w in usable],
                "median_pct": med,
                "median_seconds_per_step": per_step,
                "max_residual_s": resid,
                "verdict": verdict,
            }, fh, ensure_ascii=False, indent=2)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
