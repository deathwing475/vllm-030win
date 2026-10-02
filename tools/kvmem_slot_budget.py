# -*- coding: utf-8 -*-
"""步骤 079：把检索槽预算的口径一次算清（纯离线，不碰 GPU）。

为什么要这个
------------
078 记的"针页名次从 074 的第 9 掉到第 39/40，距 55 槽还剩 ~15 名"有两处错：

1. **页号是旧的**。针的绝对 token 位由探针自己记在日志里（078 两 boot 都是
   ``needle at token 125370``）。074 那批臂的页长是 1424 ⇒ ``125370//1424 = 88``；
   075 把 ``SW_WINDOW`` 降到 131072 之后页长变 1456 ⇒ ``125370//1456 = 86``。
   078 文档去 top_pages 里查"页 88"，查到的早就不是针页了。
2. **名次有两个互不相同的量，混用了**。引擎选页用的是 ``_summarize`` 的
   **跨层 softmax 求和分** ``page_scores``（``index.py:501-521``），而仓内既有分析器
   ``kvmem_k3_probe.rank_of`` 的名次字段其实叫 **``rank_by_logit``**，按**跨层均值
   logit** ``page_logits`` 排（``index.py:517-519``）。同一份 dump 两个口径给不同名次。
   而且真正决定"针进不进槽"的 neither：槽是按**时间序**填的
   （``worker.py:953 selected = sorted(top_pages)[:len(stage.slots)]``），
   所以判据是"针页在 ``sorted(top_pages)`` 里排第几"，跟 55 比。

本工具把三个量并列输出，并扫 ``TOPN × VIEWPORT_PAGES × VIEWPORT_RECENT``。

扫描的可算性边界（重要，别把近似当实测）
----------------------------------------
- ``TOPN`` / ``VIEWPORT_PAGES``：**精确**。名次序不随截断变，改的只是取前多少个。
- ``VIEWPORT_RECENT``：**换的是 eligible 集合本身**（``index.py:524-536``
  ``end = num_pages - recent//block_size``），而 ``page_scores`` 是"对 eligible 做
  逐层 softmax 再求和"的结果 ⇒ 换了 eligible 就不能从 dump 里的分数精确重算。
  换 recent 的那一行只用 **logit 口径**给近似，并在输出里标 ``approx``。

用法
----
    python tools/kvmem_slot_budget.py --dump kvmem_k12c --dump kvmem_k12d \
        --needle-token 125370
    python tools/kvmem_slot_budget.py --dump kvmem_k13a --needle-token 125370 \
        --out step079_slotbudget.json
"""
import argparse
import json
import os
import sys

LOGS = r"G:\qwen3.8model\prod029_logs"
# 必守 16⑦：分支指纹只认 recent_tokens。16384 = 视窗，32768 = 原生。
VIEWPORT_FINGERPRINT = 16384
NATIVE_FINGERPRINT = 32768


def load_reports(dump_dir: str) -> list[tuple[str, dict]]:
    out = []
    for name in sorted(os.listdir(dump_dir)):
        if name.startswith("kvmem_retrieval_") and name.endswith(".json"):
            with open(os.path.join(dump_dir, name), encoding="utf-8") as fh:
                out.append((name, json.load(fh)))
    return out


def eligible_for(num_pages: int, block: int, sink: int, recent: int) -> list[int]:
    """index.py:_eligible 的原式（start 至少跳过 sink 那一页）。"""
    import math
    start = max(1, math.ceil(sink / block))
    end = num_pages - max(0, recent // block)
    end = max(start, end)
    return [p for p in range(start, end)]


def rank_order(pages: list[int], values: list) -> list[int]:
    """按值降序；None 不参与；并列按页号升序（np.argsort 的实际效果）。"""
    scored = [(p, values[p]) for p in pages if values[p] is not None]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [p for p, _ in scored]


def rank_in(order: list[int], page: int):
    return order.index(page) + 1 if page in order else None


def analyse(report: dict, needle_token: int, topns: list[int],
            budgets: list[int], recents: list[int]) -> dict:
    block = report["block_size"]
    num_pages = report["num_pages"]
    sink = report.get("sink_tokens") or block
    recent0 = report["recent_tokens"]
    sub = report.get("subblock") or block
    needle_page = needle_token // block
    needle_snapped = ((needle_token // sub) * sub) // block
    variants = report.get("variants") or {}
    primary = f"{report['score_modes'][0]}@{report['granularities'][0]}"
    rows = {}
    for name, v in variants.items():
        scores = v.get("page_scores")
        logits = v.get("page_logits")
        el = report["eligible"]
        exact_top = rank_order(el, scores)
        logit_top = rank_order(el, logits)
        engine_top = v.get("top_pages") or []
        # 自检：用 dump 里的 page_scores 重排，前 topn 个应当就是引擎给的 top_pages
        replay = sorted(exact_top[:len(engine_top)]) if engine_top else []
        rows[name] = {
            "is_primary": name == primary,
            "engine_order_reproduced": (not engine_top) or replay == sorted(engine_top),
            "rank_by_score_order": rank_in(exact_top, needle_page),
            "rank_by_score_position_in_top_pages": (
                engine_top.index(needle_page) + 1 if needle_page in engine_top else None
            ),
            "rank_by_logit": rank_in(logit_top, needle_page),
            "in_top_pages": needle_page in engine_top,
            "time_order_slot": (
                sorted(engine_top).index(needle_page) + 1
                if needle_page in engine_top else None
            ),
            "of_slots": min(report["topn"], len(engine_top)),
            "needle_logit": logits[needle_page] if logits else None,
            "median_logit_eligible": (
                sorted(x for x in (logits[i] for i in el) if x is not None)[
                    len(el) // 2] if el and logits else None),
        }
    sweep = []
    for topn in topns:
        for pages in budgets:
            for recent in recents:
                exact = recent == recent0
                row = {"topn": topn, "pages": pages, "recent": recent,
                       "exact": exact}
                for name, v in variants.items():
                    values = v["page_scores"] if exact else v["page_logits"]
                    order = rank_order(
                        eligible_for(num_pages, block, sink, recent), values)
                    chosen = sorted(order[:topn])
                    row[name] = {
                        "in_slots": needle_page in chosen[:pages],
                        "slot": (chosen.index(needle_page) + 1
                                 if needle_page in chosen else None),
                        "slots": min(pages, len(chosen)),
                    }
                sweep.append(row)
    return {
        "block_size": block,
        "num_pages": num_pages,
        "num_tokens": report.get("num_tokens"),
        "recent_tokens_in_dump": recent0,
        "branch": ("viewport" if recent0 == VIEWPORT_FINGERPRINT else
                   "native" if recent0 == NATIVE_FINGERPRINT else "unknown"),
        "eligible": len(report["eligible"]),
        "topn_in_dump": report["topn"],
        "needle_token": needle_token,
        "needle_page": needle_page,
        "needle_page_snapped": needle_snapped,
        "variants": rows,
        "sweep": sweep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    help="dump 目录名（相对 prod029_logs）或绝对路径，可多次")
    ap.add_argument("--needle-token", type=int, default=125370,
                    help="针的绝对 token 位，取探针日志的 \"needle at token N\"")
    ap.add_argument("--topn", default="16,32,64,96")
    ap.add_argument("--pages", default="30,45,55,64,80")
    ap.add_argument("--recent", default="8192,16384,32768")
    ap.add_argument("--only", default="viewport",
                    choices=("viewport", "native", "all"),
                    help="按分支指纹过滤（必守 16⑦），默认只看视窗")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default=None,
                    help="只报某个 (mode@granularity) 变体")
    args = ap.parse_args()

    topns = [int(x) for x in args.topn.split(",")]
    budgets = [int(x) for x in args.pages.split(",")]
    recents = [int(x) for x in args.recent.split(",")]
    out = {}
    for dump in args.dump:
        d = dump if os.path.isabs(dump) else os.path.join(LOGS, dump)
        if not os.path.isdir(d):
            print(f"MISSING {d}", file=sys.stderr)
            continue
        for name, report in load_reports(d):
            if (args.only != "all"
                    and report.get("recent_tokens")
                    != (VIEWPORT_FINGERPRINT if args.only == "viewport"
                        else NATIVE_FINGERPRINT)):
                continue
            if "error" in report:
                print(f"{os.path.basename(d)}/{name}: report has error "
                      f"{report['error']}", file=sys.stderr)
                continue
            res = analyse(report, args.needle_token, topns, budgets, recents)
            key = f"{os.path.basename(d)}/{name}"
            out[key] = res
            print("=" * 78)
            print(f"{key}  [{res['branch']}]  block={res['block_size']} "
                  f"pages={res['num_pages']} eligible={res['eligible']} "
                  f"dump_topn={res['topn_in_dump']} "
                  f"needle_token={res['needle_token']} -> page {res['needle_page']}"
                  f" (snapped {res['needle_page_snapped']})")
            for vname, r in sorted(res["variants"].items()):
                if args.variant and vname != args.variant:
                    continue
                med = r["median_logit_eligible"]
                print(f"  {vname:12s}{'*' if r['is_primary'] else ' '} "
                      f"score序 {str(r['rank_by_score_order']):>4s} | "
                      f"top_pages位 {str(r['rank_by_score_position_in_top_pages']):>4s} | "
                      f"logit序(旧口径) {str(r['rank_by_logit']):>4s} | "
                      f"时间序槽 {str(r['time_order_slot']):>4s}/{r['of_slots']} "
                      f"⇒ 55 槽余量 "
                      f"{55 - r['time_order_slot'] if r['time_order_slot'] else '-'} | "
                      f"replay_ok={r['engine_order_reproduced']} "
                      f"needle_logit {r['needle_logit']} med {med}")
            # 摘要：视窗分支 + 主变体下，(pages, recent) 里针进槽的格点
            prim = next((v for v, r in res["variants"].items()
                         if res["variants"][v]["is_primary"]), None)
            if prim:
                hits = [f"{s['pages']}/{s['recent']}"
                        + ("" if s["exact"] else "~")
                        for s in res["sweep"]
                        if s["topn"] == max(topns) and s[prim]["in_slots"]]
                misses = [f"{s['pages']}/{s['recent']}"
                          + ("" if s["exact"] else "~")
                          for s in res["sweep"]
                          if s["topn"] == max(topns) and not s[prim]["in_slots"]]
                print(f"  针进槽格点(topn={max(topns)}, pages/recent"
                      f"{'，~ = 换 recent 属 logit 近似口径' if len(recents) > 1 else ''}): "
                      f"{', '.join(hits) or '无'}")
                print(f"  打不进槽的格点: {', '.join(misses) or '无'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")
    if not out:
        print("没有可分析的报告（检查 --only 指纹过滤与目录名）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
