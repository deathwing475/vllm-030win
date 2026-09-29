# kvmem_k3_replay.py — replay the engine's own ranking offline, then try to fix it.
#
# The engine's page score is not the page mean-K dotted with the query: it is a
# **max over the page's sub-blocks** of that dot product, applied per layer
# before the layers are summed. A 15-token needle in a 1424-token page survives
# that reduction and is diluted away by the mean, which is why the diagnostic
# sidecar carries sub-block means: with them the ranking can be reproduced
# offline (that reproduction is itself the check that the sidecar holds the same
# arithmetic the engine ran), and a de-biasing transform can be judged on the
# two things that matter - does it flatten the position trend, and does it keep
# the needle.
#
#   python tools\kvmem_k3_replay.py <control.json> <needle.json>
#
# Both files are the probe's output; the engine report and the .npz are found
# through the report_path they carry.
#
# Why the position trend matters (step 062): the page logit is dominated by a
# component of the page mean-K that varies with the page's position in the
# document, and the query is aligned with it at ~50x the random baseline. Two
# unrelated questions produced per-page logits correlated at +0.986, so the
# ranking barely depends on what is asked. The practical cost is that the front
# pages win by default: a needle placed in the back half of a 200K document
# lands outside the top-16 on roughly half of all pages.

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_k3c_analyze import load  # noqa: E402


def page_starts(num_subblocks: int, block_size: int, num_pages: int,
                granularity: int) -> np.ndarray:
    """First sub-block of each page, on the real page boundaries."""
    page_of = (np.arange(num_subblocks, dtype=np.int64) * granularity) // block_size
    return np.searchsorted(page_of, np.arange(num_pages), side="left")


def subblock_scores(q: np.ndarray, sub: np.ndarray, head_dim: int) -> np.ndarray:
    """[span, L, S] per-query-token, per-layer dot of the query with each sub-block."""
    return np.einsum("tlkd,slkd->tls", q, sub) / np.sqrt(head_dim)


def page_scores(per: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """[span, L, P] max over each page's sub-blocks, i.e. the engine's reduction."""
    span, num_layers = per.shape[0], per.shape[1]
    out = np.empty((span, num_layers, starts.size), dtype=np.float32)
    for layer in range(num_layers):
        out[:, layer] = np.maximum.reduceat(per[:, layer], starts, axis=-1)
    return out


def to_page_logits(pag: np.ndarray) -> np.ndarray:
    """[P] the engine reports: sum over layers, mean over the query span."""
    return (pag.sum(axis=1) / pag.shape[1]).mean(axis=0)


def position_direction(pmk: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The direction in page mean-K space that tracks the page's position.

    Least squares per dimension against the page offset. This is the component
    step 062 implicated: the second principal component of the page mean-K
    correlates with the offset at +0.80 and carries 22% of the variance.
    """
    t = offsets.astype(np.float64)
    t = (t - t.mean()) / t.std()
    flat = pmk.reshape(pmk.shape[0], -1).astype(np.float64)
    beta = (t[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / (t**2).sum()
    return beta.reshape(pmk.shape[1:])


def project_out(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove ``v``'s direction from every query token (v is [L, KV, D])."""
    coef = (q * v).sum(-1, keepdims=True) / ((v * v).sum(-1, keepdims=True) + 1e-12)
    return q - coef * v


def detrend(sub: np.ndarray, granularity: int) -> np.ndarray:
    """Subtract each sub-block dimension's linear trend in token position."""
    num_sub = sub.shape[0]
    t = (np.arange(num_sub) * granularity).astype(np.float64)
    t = (t - t.mean()) / t.std()
    flat = sub.reshape(num_sub, -1).astype(np.float64)
    beta = (t[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / (t**2).sum()
    return (flat - t[:, None] * beta).reshape(sub.shape).astype(np.float32)


def high_pass(sub: np.ndarray, window: int) -> np.ndarray:
    """Subtract a +/-window sub-block neighbourhood mean (slow component out)."""
    num_sub = sub.shape[0]
    flat = sub.reshape(num_sub, -1).astype(np.float64)
    cum = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    lo = np.maximum(np.arange(num_sub) - window, 0)
    hi = np.minimum(np.arange(num_sub) + window + 1, num_sub)
    local = (cum[hi] - cum[lo]) / (hi - lo)[:, None]
    return (flat - local).reshape(sub.shape).astype(np.float32)


def needle_gain(q: np.ndarray, direction: np.ndarray, head_dim: int) -> float:
    """Page-logit lift produced by adding one unit-norm sub-block of ``direction``."""
    per_layer = np.einsum("tlkd,lkd->tl", q, direction) / np.sqrt(head_dim)
    return float(per_layer.mean(axis=0).sum() / q.shape[1])


class Run:
    """One ingest's captured vectors, ready to be re-scored."""

    def __init__(self, probe_path: str):
        self.report, arrays, self.context = load(probe_path)
        self.q = arrays["q_group_mean"].astype(np.float32)
        self.sub = arrays["subblock_mean_k"].astype(np.float32)
        self.counts = arrays["subblock_counts"]
        self.pmk = arrays["page_mean_k"].astype(np.float32)
        self.block_size = self.report["block_size"]
        self.granularity = self.report["subblock"]
        self.num_pages = self.report["num_pages"]
        self.head_dim = self.q.shape[-1]
        self.num_layers = self.q.shape[1]
        self.num_kv_heads = self.q.shape[2]
        self.eligible = np.array(self.report["eligible"], dtype=np.int64)
        self.offsets = np.arange(self.num_pages) * self.block_size
        self.starts = page_starts(self.sub.shape[0], self.block_size,
                                 self.num_pages, self.granularity)
        self.valid = self.counts > 0

    def engine_logits(self) -> np.ndarray:
        key = f"dot@{self.granularity}"
        return np.array(
            [v if v is not None else np.nan
             for v in self.report["variants"][key]["page_logits"]]
        )

    def score(self, q: np.ndarray | None = None, sub: np.ndarray | None = None):
        q = self.q if q is None else q
        sub = self.sub if sub is None else sub
        per = subblock_scores(q, sub, self.head_dim)
        per[:, :, ~self.valid] = -np.inf
        pag = page_scores(per, self.starts)
        return per, pag, to_page_logits(pag)


def rank_of(logits: np.ndarray, eligible: np.ndarray, page: int) -> int | None:
    order = eligible[np.argsort(-logits[eligible])]
    return int(np.where(order == page)[0][0]) + 1 if page in order else None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    control = Run(sys.argv[1])
    needle = Run(sys.argv[2])
    needle_page = needle.context.get("needle_page")

    print("=" * 78)
    print("REPLAY CHECK — offline max-reduce against the engine's own report")
    for name, run in (("control", control), ("needle", needle)):
        _, _, got = run.score()
        diff = float(np.nanmax(np.abs(got - run.engine_logits())))
        print(f"  {name:<8} max |replay - engine| = {diff:.3e}  "
              f"({'MATCH' if diff < 1e-3 else 'MISMATCH'})")

    # --- the transforms under test ----------------------------------------
    beta = position_direction(control.pmk, control.offsets).astype(np.float32)
    centred = control.pmk.reshape(control.num_pages, -1)
    centred = centred - centred.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pc_basis = vt[:2].T  # [F, 2]

    def drop_pcs(q: np.ndarray) -> np.ndarray:
        """Remove the top-2 principal directions of the page mean-K from q."""
        flat = q.reshape(-1, pc_basis.shape[0])
        return (flat - (flat @ pc_basis) @ pc_basis.T).reshape(q.shape).astype(np.float32)

    transforms = {
        "baseline": (control.q, control.sub, needle.q, needle.sub),
        "q - beta (page slope)": (project_out(control.q, beta), control.sub,
                                  project_out(needle.q, beta), needle.sub),
        "q - PC1..2": (drop_pcs(control.q), control.sub,
                       drop_pcs(needle.q), needle.sub),
        "sub detrend": (control.q, detrend(control.sub, control.granularity),
                        needle.q, detrend(needle.sub, needle.granularity)),
        "sub high-pass w=8": (control.q, high_pass(control.sub, 8),
                              needle.q, high_pass(needle.sub, 8)),
    }

    # --- the real needle --------------------------------------------------
    _, _, base_logits = control.score()
    lift = None
    if needle_page is not None:
        _, _, ndl_logits = needle.score()
        lift = float(ndl_logits[needle_page] - base_logits[needle_page])
        print()
        print("REAL NEEDLE (page %d)" % needle_page)
        print(f"  control logit {base_logits[needle_page]:.2f} -> needle "
              f"{ndl_logits[needle_page]:.2f}   lift {lift:+.2f}")
        print(f"  rank {rank_of(base_logits, control.eligible, needle_page)} "
              f"-> {rank_of(ndl_logits, control.eligible, needle_page)} of "
              f"{control.eligible.size} eligible")

    # --- a needle of the measured size, placed on every page in turn ------
    # This is the decision criterion. The front pages win by position, so the
    # question is not whether the needle survives on page 19 but whether it
    # survives anywhere.
    if lift is None:
        return
    direction = control.q.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    gain = needle_gain(control.q, direction, control.head_dim)
    delta = direction * (lift / gain)
    sub_of_page = (np.arange(control.sub.shape[0]) * control.granularity) // control.block_size

    print()
    print(f"SYNTHETIC NEEDLE ({lift:+.2f}) ON EVERY ELIGIBLE PAGE IN TURN")
    print(f"  {'transform':<24} {'corr':>6}  {'real':>5}  {'med':>4} {'worst':>6} "
          f"{'>16':>5}   top-8 pages")
    for name, (qc, sc, qn, sn) in transforms.items():
        per_c, pag_c, logits_c = control.score(qc, sc)
        corr = np.corrcoef(control.offsets[control.eligible],
                           logits_c[control.eligible])[0, 1]
        _, _, logits_n = needle.score(qn, sn)
        real = rank_of(logits_n, needle.eligible, needle_page)

        ranks = []
        for page in control.eligible:
            subs = np.where(sub_of_page == page)[0]
            target = subs[len(subs) // 2]
            bump = np.einsum("tlkd,lkd->tl", qc, delta) / np.sqrt(control.head_dim)
            bumped = pag_c.copy()
            bumped[:, :, page] = np.maximum(pag_c[:, :, page],
                                            per_c[:, :, target] + bump)
            ranks.append(rank_of(to_page_logits(bumped), control.eligible, int(page)))
        ranks = np.array(ranks)
        order = control.eligible[np.argsort(-logits_c[control.eligible])]
        print(f"  {name:<24} {corr:+.3f}  {real:>5}  {int(np.median(ranks)):>4} "
              f"{int(ranks.max()):>6} {int((ranks > 16).sum()):>5}   "
              f"{[int(p) for p in order[:8]]}")


if __name__ == "__main__":
    main()
