# kvmem_k3c_analyze.py — offline analysis of the K3 diagnostic sidecar (step 062).
#
# The engine's report says *what* it ranked; the sidecar .npz says *why*, because
# it carries the vectors the ranking was made of (page mean-K per layer and KV
# head, the per-head-group query, the per-sub-block and per-page logits). One
# 200K ingest costs ~4.5 minutes, so every hypothesis about the step 061
# residual bias is answered here instead of with another engine run.
#
#   python tools\kvmem_k3c_analyze.py <report.json> [<report2.json> ...]
#
# Single report: where the page logit comes from (per layer, per dimension, and
# max-reduce vs page-mean-reduce). Two or more: what changes when the query
# changes, which is the question step 061 could not answer because its three
# runs all used the same question.

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path: str) -> tuple[dict, dict, dict]:
    """Accept either the probe's output or the engine's own report.

    The probe's file carries the question text but not the per-page variants;
    the engine's file is the other way round. Both name the sidecar, so either
    entry point is enough to analyse a run.
    """
    with open(path, encoding="utf-8") as handle:
        outer = json.load(handle)
    context = {
        "question": outer.get("question_text") or outer.get("question_style"),
        "with_needle": outer.get("with_needle"),
        "needle_page": outer.get("needle_page"),
        "probe": outer,
    }
    if "variants" in outer:
        report, report_path = outer, path
    else:
        report_path = outer.get("report_path")
        if not report_path or not os.path.exists(report_path):
            raise SystemExit(f"{path} does not name a readable engine report")
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
    npz_path = report_path.replace(".json", "_kbar.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(
            f"{npz_path} is missing - was VLLM_KVMEM_DUMP_KBAR=1 set for that run?"
        )
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    context["npz_path"] = npz_path
    return report, arrays, context


def page_logits_of(report: dict, variant: str) -> np.ndarray:
    raw = report["variants"][variant]["page_logits"]
    return np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)


def eligible_set(report: dict) -> np.ndarray:
    return np.array(report["eligible"], dtype=np.int64)


def describe(path: str) -> dict:
    report, arrays, context = load(path)
    logits = page_logits_of(report, f"{report['score_modes'][0]}@{report['granularities'][0]}")
    eligible = eligible_set(report)
    print("=" * 78)
    print(os.path.basename(path))
    print(
        f"  tokens {report['num_tokens']} pages {report['num_pages']} "
        f"eligible {len(eligible)} layers {report['num_layers_scored']} "
        f"query_span {report['query_span']} subblock {report['subblock']}"
    )
    print(f"  question: {context['question']}  needle_page {context['needle_page']}")
    print(f"  top_pages {report['top_pages'][:10]}")
    order = eligible[np.argsort(-logits[eligible])]
    print(f"  top-10 eligible by logit: {[(int(p), round(float(logits[p]), 2)) for p in order[:10]]}")

    pmk = arrays["page_mean_k"].astype(np.float32)  # [P, L, KV, D]
    qg = arrays["q_group_mean"].astype(np.float32)  # [S, L, KV, D]
    pml = arrays["page_mean_logits"].astype(np.float32)  # [L, P]
    sbl = arrays["subblock_logits"].astype(np.float32)  # [L, sub]
    num_pages, num_layers = pmk.shape[0], pmk.shape[1]
    head_dim = pmk.shape[-1]

    # --- max reduce vs page-mean reduce -----------------------------------
    # The design picked max so that a page with one exact hit beats a page of
    # mediocre sub-blocks. The side effect is that a page whose sub-blocks vary
    # more has more chances to produce a high max. Comparing the two reductions
    # on the same ingest says how much of the ranking is that effect.
    mean_reduce = pml.sum(axis=0) / num_layers
    rank_max = {int(p): i for i, p in enumerate(order)}
    order_mean = eligible[np.argsort(-mean_reduce[eligible])]
    rank_mean = {int(p): i for i, p in enumerate(order_mean)}
    print("  --- max reduce vs page-mean reduce (eligible top-8) ---")
    print(f"    max      : {[(int(p), round(float(logits[p]), 2)) for p in order[:8]]}")
    print(f"    page-mean: {[(int(p), round(float(mean_reduce[p]), 2)) for p in order_mean[:8]]}")
    common = [p for p in order[:8] if int(p) in rank_mean]
    print(f"    pages in both top-8: {len(common)}/8; "
          f"max-minus-mean logit std {np.nanstd(logits[eligible] - mean_reduce[eligible]):.2f}")

    # --- per-layer attribution -------------------------------------------
    # If the bias lives in one or two layers, the fix belongs there; if it is
    # spread evenly, it is a property of the hidden states themselves.
    print("  --- per-layer page-mean logit spread over eligible pages ---")
    spreads = [
        (li, float(np.nanstd(pml[li][eligible])), float(np.nanmean(pml[li][eligible])))
        for li in range(num_layers)
    ]
    for li, std, mean in spreads:
        print(f"    layer {int(arrays['layers'][li]):>3}: std {std:6.2f} mean {mean:7.2f}")

    # --- which dimensions carry it ---------------------------------------
    # q . kbar summed over (layer, kv head) is a sum over the 256 dims of the
    # head; if a handful of dims carry the spread, the index can be fixed by
    # dropping or rescaling them instead of redesigning the score.
    q_bar = qg.mean(axis=0)  # [L, KV, D]
    dim_contrib = np.einsum("lkd,plkd->pd", q_bar, pmk) / np.sqrt(head_dim) / num_layers
    dim_spread = np.nanstd(dim_contrib[eligible], axis=0)
    top_dims = np.argsort(-dim_spread)[:8]
    print("  --- dims carrying the eligible-page logit spread ---")
    print(f"    top-8 dims {top_dims.tolist()} carry "
          f"{100.0 * dim_spread[top_dims].sum() / dim_spread.sum():.1f}% of the total")
    for d in top_dims:
        print(f"      dim {int(d):>3}: spread {dim_spread[d]:7.3f}  "
              f"mean {dim_contrib[eligible, d].mean():+8.3f}  "
              f"q_bar|dim| {np.abs(q_bar[..., d]).mean():7.3f}")

    # --- page mean-K geometry --------------------------------------------
    # Step 061 ruled out the norm. The remaining candidates are "the mean
    # vector's deviation from the document-wide mean grows towards the front"
    # and "the query simply matches some pages' content".
    flat = pmk.reshape(num_pages, -1)
    mu = np.nanmean(flat, axis=0)
    norm = np.linalg.norm(flat, axis=1)
    dev_norm = np.linalg.norm(flat - mu, axis=1)
    offset = np.arange(num_pages) * report["block_size"]
    print("  --- page mean-K geometry ---")
    print(f"    ||kbar||      mean {np.nanmean(norm):.2f} std {np.nanstd(norm):.2f}")
    print(f"    ||kbar - mu|| mean {np.nanmean(dev_norm):.2f} std {np.nanstd(dev_norm):.2f}")
    print(f"    pearson(||kbar-mu||, logit) over eligible = "
          f"{np.corrcoef(dev_norm[eligible], logits[eligible])[0, 1]:+.3f}")
    print(f"    pearson(page offset, logit)  over eligible = "
          f"{np.corrcoef(offset[eligible], logits[eligible])[0, 1]:+.3f}")
    mitigations(report, arrays, logits, eligible, context)
    return {
        "path": path,
        "report": report,
        "context": context,
        "logits": logits,
        "mean_reduce": mean_reduce,
        "eligible": eligible,
        "dim_contrib": dim_contrib,
        "order": order,
    }


def mitigations(
    report: dict, arrays: dict, logits: np.ndarray, eligible: np.ndarray, context: dict
) -> None:
    """Re-score the same ingest under candidate de-biasing transforms.

    The step 062 finding is that the page logit is dominated by a component of
    the page mean-K that varies with the page's *position* in the document, and
    that the query happens to be aligned with it. The candidates below attack
    that component directly; the one that wins has to kill the position
    correlation *and* keep the needle's page at the top, which is why this only
    becomes conclusive on a run that has a needle.
    """
    pmk = arrays["page_mean_k"].astype(np.float32)
    qg = arrays["q_group_mean"].astype(np.float32)
    num_pages, num_layers, num_kv_heads, head_dim = pmk.shape
    flat_dim = num_layers * num_kv_heads * head_dim
    offset = np.arange(num_pages) * report["block_size"]
    q_bar = qg.mean(axis=0)
    needle_page = context.get("needle_page")

    def score(q: np.ndarray, k: np.ndarray) -> np.ndarray:
        return np.einsum("lkd,plkd->p", q, k) / np.sqrt(head_dim) / num_layers

    def report_one(name: str, scores: np.ndarray) -> None:
        order = eligible[np.argsort(-scores[eligible])]
        corr = np.corrcoef(offset[eligible], scores[eligible])[0, 1]
        line = f"    {name:<22} corr(offset,logit) {corr:+.3f}  top8 {[int(p) for p in order[:8]]}"
        if needle_page is not None and needle_page < num_pages:
            rank = int(np.where(order == needle_page)[0][0]) + 1 if needle_page in order else None
            line += f"  needle page {needle_page} rank {rank}/{eligible.size}"
        print(line)

    print("  --- de-biasing candidates (page-mean reduce, dot mode) ---")
    report_one("baseline", score(q_bar, pmk))

    # High-pass: subtract the mean of a +/-w page neighbourhood. The position
    # component is slow (PC2 tracks the offset at +0.80), a needle is local, so
    # this is the transform that separates them without any SVD.
    flat = pmk.reshape(num_pages, flat_dim).astype(np.float64)
    cum = np.vstack([np.zeros((1, flat_dim)), np.cumsum(flat, axis=0)])
    for window in (2, 4, 8):
        lo = np.maximum(np.arange(num_pages) - window, 0)
        hi = np.minimum(np.arange(num_pages) + window + 1, num_pages)
        local = (cum[hi] - cum[lo]) / (hi - lo)[:, None]
        report_one(f"high-pass w={window}", score(q_bar, (flat - local).reshape(pmk.shape).astype(np.float32)))

    # Linear detrend per dimension, and dropping the top principal components
    # of the centred page mean-K from both sides.
    t = (offset - offset.mean()) / offset.std()
    beta = (t[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / (t**2).sum()
    report_one("linear detrend", score(q_bar, (flat - t[:, None] * beta).reshape(pmk.shape).astype(np.float32)))

    centred = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    q_flat = q_bar.reshape(flat_dim)
    for keep in (2, 4):
        basis = vt[:keep].T
        k_drop = (centred - (centred @ basis) @ basis.T).reshape(pmk.shape).astype(np.float32)
        q_drop = (q_flat - (q_flat @ basis) @ basis.T).reshape(q_bar.shape).astype(np.float32)
        report_one(f"drop top-{keep} PCs", score(q_drop, k_drop))
    print(f"    (cos(q_bar, PC2) = "
          f"{float(q_flat @ vt[1] / np.linalg.norm(q_flat)):+.3f}, random baseline "
          f"~{0.6745 / np.sqrt(flat_dim):.4f})")


def compare(a: dict, b: dict) -> None:
    """Two ingests of the same haystack under different queries."""
    print("=" * 78)
    print("QUERY COMPARISON")
    print(f"  A: {os.path.basename(a['path'])}  q={a['context']['question']}")
    print(f"  B: {os.path.basename(b['path'])}  q={b['context']['question']}")
    common = np.intersect1d(a["eligible"], b["eligible"])
    da = a["logits"][common]
    db = b["logits"][common]
    print(f"  pearson(logit_A, logit_B) over {common.size} shared eligible pages = "
          f"{np.corrcoef(da, db)[0, 1]:+.3f}")
    print(f"  mean |delta| {np.nanmean(np.abs(da - db)):.2f}   "
          f"delta std {np.nanstd(da - db):.2f}")
    top_a = [int(p) for p in a["order"][:10]]
    top_b = [int(p) for p in b["order"][:10]]
    print(f"  top-10 A {top_a}")
    print(f"  top-10 B {top_b}")
    print(f"  shared in both top-10: {sorted(set(top_a) & set(top_b))}")
    # Per-page logits are a projection of the *page* onto the query, so if the
    # bias is a property of the page it survives a query swap; a page whose
    # rank collapses under the new query was matching that query's content.
    for p in top_a[:6]:
        print(f"    page {p:>3}: A {da[common == p][0] if p in common else float('nan'):7.2f}  "
              f"B {db[common == p][0] if p in common else float('nan'):7.2f}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results = [describe(path) for path in sys.argv[1:]]
    for i in range(len(results) - 1):
        compare(results[i], results[i + 1])


if __name__ == "__main__":
    main()
