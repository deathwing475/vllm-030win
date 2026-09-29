"""Offline re-analysis of the K3 retrieval dumps (step 061).

The probe's --dump-dir must equal the engine's VLLM_KVMEM_DUMP; when it does
not, the run still produced a report, so this recomputes the needle offset from
the saved nonce and judges every variant.
"""
import glob
import json
import os
import sys

sys.path.insert(0, r"G:\qwen3.8model\vllm-030win-git\tools")
from kvmem_k3_probe import build, tokenizer  # noqa: E402

LOGS = r"G:\qwen3.8model\prod029_logs"
DUMP = os.path.join(LOGS, "kvmem_k3")


def report_after(stamp):
    # The engine writes the dump, then the probe exits, so this probe's report
    # is the newest one written at or before the probe JSON was saved.
    best = None
    for path in glob.glob(os.path.join(DUMP, "kvmem_retrieval_*.json")):
        m = os.path.getmtime(path)
        if m <= stamp + 2 and (best is None or m > best[0]):
            best = (m, path)
    return None if best is None else json.load(open(best[1], encoding="utf-8")), best[1]


def rank_of(report, page):
    out = {}
    for name, v in report["variants"].items():
        lg = v["page_logits"]
        el = report["eligible"]
        sc = sorted([(i, lg[i]) for i in el if lg[i] is not None], key=lambda t: -t[1])
        order = [i for i, _ in sc]
        out[name] = {
            "rank": (order.index(page) + 1) if page in order else None,
            "of": len(order),
            "in_top": page in v["top_pages"],
            "needle_logit": lg[page],
            "best": sc[0][1] if sc else None,
            "top6": v["top_pages"][:6],
        }
    return out


for name in sys.argv[1:]:
    probe = json.load(open(os.path.join(LOGS, name), encoding="utf-8"))
    report, path = report_after(os.path.getmtime(os.path.join(LOGS, name)))
    print("=" * 78)
    print(name, "->", os.path.basename(path))
    print("  depth", probe.get("needle_depth"), "nonce", probe.get("nonce"),
          "tokens", report["num_tokens"], "pages", report["num_pages"],
          "eligible", len(report["eligible"]), "subblock", report["subblock"],
          "grans", report["granularities"], "modes", report["score_modes"])
    if not probe.get("with_needle"):
        v = next(iter(report["variants"].values()))
        lg = sorted([x for x in v["page_logits"] if x is not None], reverse=True)
        print("  CONTROL top8 logits", [round(x, 2) for x in lg[:8]],
              "median", round(lg[len(lg) // 2], 2))
        print("  CONTROL top6 pages", v["top_pages"][:6])
        continue
    prompt, needle_char = build(
        probe["prompt_tokens"], probe["needle_depth"], probe["nonce"],
        True, probe["question_style"], probe["question_repeat"],
    )
    off = len(tokenizer().encode(prompt[:needle_char], add_special_tokens=False))
    sb, bs = report["subblock"], report["block_size"]
    page = ((off // sb) * sb) // bs
    print(f"  needle offset {off} (page {page}, by token {off // bs}), "
          f"evicted={off < report['num_tokens'] - 163072}")
    for variant, r in rank_of(report, page).items():
        print(f"    {variant:12s} rank {r['rank']:>4}/{r['of']}  in_top={str(r['in_top']):5s} "
              f"needle_logit {r['needle_logit']:7.3f}  best {r['best']:7.3f}  top6 {r['top6']}")
