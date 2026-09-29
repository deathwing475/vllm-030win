# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 061): KVMem Mean-K index and page scoring (K3 2-3).

Host-resident, keyed by trajectory. The index is a running fp32 sum of the
pre-RoPE K of every stored sub-block, finalised to an fp16 mean on demand; the
design's 128-token sub-block over a 262,144-token workspace and 16 layers of 4
KV heads x 256 dims costs 64 MiB, which is why it can live in RAM permanently.

Scoring follows design doc §5.3: ``logit = q . kbar / sqrt(d)`` per (layer,
head), page score = **max** over the page's sub-blocks (a page of merely
mediocre sub-blocks must not outrank a page with one exact hit), then a softmax
over the candidate pages, summed over the query span and averaged over (layer,
head). Sink and recent pages are excluded from the candidates.

Nothing here feeds attention yet: this step measures whether the retrieval
decision is any good at the 1424-token page granularity the block size forces
on us (design risk R1), which is the precondition for building the
rematerialisation path at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kvmem_workspace import config

logger = init_logger(__name__)


@dataclass
class _TrajectoryIndex:
    subblock: int
    num_subblocks: int
    # layer_idx -> fp32 [num_subblocks, num_kv_heads * head_dim]
    sums: dict[int, np.ndarray] = field(default_factory=dict)
    counts: np.ndarray = None  # int32 [num_subblocks]
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = np.zeros(self.num_subblocks, dtype=np.int32)


def _page_reduce(
    values: np.ndarray, granularity: int, block_size: int, num_pages: int
) -> np.ndarray:
    """Max of ``values`` per page, on the *true* page boundaries.

    A page is ``block_size`` tokens, and ``block_size`` is 1424 here — which no
    power-of-two sub-block size divides (1424 = 16 x 89). Grouping the last
    axis by ``block_size // granularity`` therefore walks off the real page
    boundaries, by 16 tokens per page at a 128-token sub-block: by page 19 the
    engine's "page 19" is 304 tokens away from the page the block table and the
    client mean by page 19. A sub-block can also straddle a boundary, so it is
    attributed to the page holding its first token; that keeps the engine's
    page index equal to ``token_offset // block_size``.
    """
    if granularity > block_size:
        raise ValueError(
            f"sub-block {granularity} is coarser than a page {block_size}"
        )
    page_of = (np.arange(values.shape[-1]) * granularity) // block_size
    keep = page_of < num_pages
    values = values[..., keep]
    page_of = page_of[keep]
    starts = np.searchsorted(page_of, np.arange(num_pages), side="left")
    return np.maximum.reduceat(values, starts, axis=-1)


def _reduce_counts(counts: np.ndarray, divisor: int) -> np.ndarray:
    """Sum per-sub-block token counts up to a coarser granularity."""
    if divisor == 1:
        return counts
    trimmed = counts[: (counts.size // divisor) * divisor]
    return trimmed.reshape(-1, divisor).sum(axis=1)


def _reduce_sums(sums: np.ndarray, divisor: int) -> np.ndarray:
    """Sum the stored per-sub-block K sums up to a coarser granularity.

    Summing is what makes a coarser mean exact: the mean over 128 tokens is the
    sum of the four 32-token sums over the sum of the four counts.
    """
    if divisor == 1:
        return sums
    trimmed = sums[: (sums.shape[0] // divisor) * divisor]
    return trimmed.reshape(-1, divisor, sums.shape[1]).sum(axis=1)


class KVMemMeanKIndex:
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_group = num_heads // num_kv_heads
        self.subblock = config.index_subblock()
        self.modes = config.score_modes()
        self.granularities = config.score_granularities()
        self.num_subblocks = (
            config.max_workspace_tokens() + self.subblock - 1
        ) // self.subblock
        self._by_trajectory: dict[bytes, _TrajectoryIndex] = {}
        self.added_tokens = 0

    def _state(self, trajectory: bytes) -> _TrajectoryIndex:
        state = self._by_trajectory.get(trajectory)
        if state is None:
            state = _TrajectoryIndex(
                subblock=self.subblock, num_subblocks=self.num_subblocks
            )
            self._by_trajectory[trajectory] = state
        return state

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    def add(
        self,
        trajectory: bytes,
        positions: np.ndarray,
        k_by_layer: dict[int, np.ndarray],
    ) -> None:
        """Fold one prefill step's pre-RoPE K into the index."""
        if not k_by_layer or positions.size == 0:
            return
        state = self._state(trajectory)
        sub_ids = np.asarray(positions, dtype=np.int64) // self.subblock
        in_range = sub_ids < self.num_subblocks
        if not in_range.all():
            logger.warning(
                "vllm-030win KVMem index: %d token(s) beyond the workspace "
                "ceiling (%d sub-blocks) were not indexed",
                int((~in_range).sum()),
                self.num_subblocks,
            )
            sub_ids = sub_ids[in_range]
        np.add.at(state.counts, sub_ids, 1)
        for layer_idx, k in k_by_layer.items():
            rows = k
            if not in_range.all():
                rows = k[in_range]
            sums = state.sums.get(layer_idx)
            if sums is None:
                sums = np.zeros(
                    (self.num_subblocks, self.num_kv_heads * self.head_dim),
                    dtype=np.float32,
                )
                state.sums[layer_idx] = sums
            np.add.at(sums, sub_ids, rows.astype(np.float32))
        state.tokens = max(state.tokens, int(np.max(positions)) + 1)
        self.added_tokens += int(sub_ids.size)

    def has(self, trajectory: bytes) -> bool:
        return trajectory in self._by_trajectory

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(
        self,
        trajectory: bytes,
        q_by_layer: dict[int, np.ndarray],
        block_size: int,
        num_tokens: int,
        sink_tokens: int,
        recent_tokens: int,
        topn: int,
    ) -> dict:
        """Rank the workspace pages for one query span.

        Returns a policy-free artifact: the per-page mean logit over every
        page, plus the eligible set and the top-N the engine would use. The
        per-page logits are dumped so that the *quality* of the retrieval
        decision can be judged against the needle's real page, which only the
        client-side probe knows.

        Every (mode, granularity) pair is reported, because they all come from
        the same index and the same prefill: a 200K ingest costs ~4 minutes and
        this is the only way to see which knob, if any, carries a signal.
        """
        state = self._by_trajectory.get(trajectory)
        if state is None or not state.sums or not q_by_layer:
            return {"error": "no index for trajectory"}

        layers = sorted(set(state.sums) & set(q_by_layer))
        if not layers:
            return {"error": "no layer captured both a stored K and a query q"}

        num_pages = (num_tokens + block_size - 1) // block_size
        span = min(len(q_by_layer[layers[0]]), min(len(q) for q in q_by_layer.values()))
        eligible = self._eligible(num_pages, block_size, sink_tokens, recent_tokens)

        variants: dict[str, dict] = {}
        diag = (
            self._new_diag(len(layers), num_pages, block_size, span)
            if config.dump_kbar()
            else None
        )
        for granularity in self.granularities:
            for mode in self.modes:
                # The diagnostic wants the vectors behind the finest dot-product
                # ranking, and that is exactly one of the variants being
                # computed anyway, so it rides along instead of being recomputed.
                diag_out = (
                    diag
                    if diag is not None
                    and granularity == self.subblock
                    and mode == self.modes[0]
                    else None
                )
                logits, norms = self._page_logits(
                    state,
                    layers,
                    q_by_layer,
                    num_pages,
                    block_size,
                    granularity,
                    mode,
                    diag_out,
                )
                variants[f"{mode}@{granularity}"] = self._summarize(
                    logits, norms, eligible, num_pages, topn
                )

        primary = variants[f"{self.modes[0]}@{self.granularities[0]}"]
        report = {
            "trajectory": trajectory.hex()[:12],
            "num_tokens": num_tokens,
            "block_size": block_size,
            "subblock": self.subblock,
            "score_modes": self.modes,
            "granularities": self.granularities,
            "num_pages": num_pages,
            "num_layers_scored": len(layers),
            "query_span": int(span),
            "sink_tokens": sink_tokens,
            "recent_tokens": recent_tokens,
            "topn": topn,
            "eligible": eligible.tolist(),
            "stored_subblocks": int((state.counts > 0).sum()),
            "variants": variants,
            **primary,
        }
        if diag is not None:
            report["_diag"] = self._finish_diag(diag, layers, span)
        return report

    def _new_diag(
        self, num_layers: int, num_pages: int, block_size: int, span: int
    ) -> dict:
        """Buffers for the sidecar ``.npz`` (see :func:`config.dump_kbar`)."""
        sub_keep = -(-(num_pages * block_size) // self.subblock)
        return {
            "layers": np.zeros(num_layers, dtype=np.int32),
            "num_layers": num_layers,
            "block_size": block_size,
            "subblock_logits": np.full(
                (num_layers, sub_keep), np.nan, dtype=np.float32
            ),
            "page_mean_logits": np.full(
                (num_layers, num_pages), np.nan, dtype=np.float32
            ),
            "q_group_mean": np.full(
                (span, num_layers, self.num_kv_heads, self.head_dim),
                np.nan,
                dtype=np.float16,
            ),
            "page_mean_k": np.full(
                (num_pages, num_layers, self.num_kv_heads, self.head_dim),
                np.nan,
                dtype=np.float16,
            ),
            # Sub-block resolution too: the ranking the engine uses takes a max
            # over a page's sub-blocks, and a needle is 15 tokens in 1424, so it
            # only shows up at that resolution (page means dilute it by 100x).
            # Without this the sidecar could not replay the engine's own
            # reduction, and a de-biasing transform could not be judged on
            # whether it keeps the needle.
            "subblock_mean_k": np.full(
                (sub_keep, num_layers, self.num_kv_heads, self.head_dim),
                np.nan,
                dtype=np.float16,
            ),
            "subblock_counts": np.zeros(sub_keep, dtype=np.int32),
            "page_offsets": None,
            "filled": 0,
        }

    @staticmethod
    def _finish_diag(diag: dict, layers: list[int], span: int) -> dict:
        """Keep the populated buffers and label them, dropping the bookkeeping."""
        diag["layers"][: diag["filled"]] = np.asarray(layers, dtype=np.int32)
        out = {
            key: diag[key]
            for key in (
                "layers",
                "subblock_logits",
                "page_mean_logits",
                "q_group_mean",
                "page_mean_k",
                "subblock_mean_k",
                "subblock_counts",
                "page_offsets",
            )
            if diag[key] is not None
        }
        out["layers"] = out["layers"][: diag["filled"]]
        out["query_span"] = np.int32(span)
        return out

    def _page_logits(
        self,
        state: "_TrajectoryIndex",
        layers: list[int],
        q_by_layer: dict[int, np.ndarray],
        num_pages: int,
        block_size: int,
        granularity: int,
        mode: str,
        diag_out: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-query-token page logits [span, num_pages] plus the mean-K norms.

        Coarser granularities are summed up from the stored ones, so the
        storage cost is paid once at the finest setting.

        ``diag_out`` (see :func:`config.dump_kbar`) additionally receives the
        vectors the ranking is made of, for the one variant it is passed to.
        """
        divisor = granularity // self.subblock
        keep = num_pages * block_size  # tokens the reported pages cover
        sub_keep = -(-keep // granularity)  # ceil: sub-blocks inside them
        span = len(q_by_layer[layers[0]])
        logits = np.zeros((span, num_pages), dtype=np.float32)
        norms = np.zeros(num_pages, dtype=np.float32)
        # A cosine is already scale-free; the design's 1/sqrt(d) only belongs on
        # the raw dot product.
        scale = 1.0 if mode == "cosine" else 1.0 / np.sqrt(self.head_dim)

        # The counts do not depend on the layer, so the page boundaries are
        # resolved once. A sub-block belongs to the page holding its first
        # token, which is what keeps the engine's page index equal to
        # ``token_offset // block_size`` (see _page_reduce).
        if diag_out is not None:
            diag_counts = _reduce_counts(
                state.counts[: sub_keep * divisor], divisor
            )
            page_of = (
                np.arange(diag_counts.size, dtype=np.int64) * granularity
            ) // block_size
            page_starts = np.searchsorted(
                page_of, np.arange(num_pages), side="left"
            )
            page_ends = np.searchsorted(
                page_of, np.arange(1, num_pages + 1), side="left"
            )
            page_ends[-1] = np.searchsorted(page_of, num_pages, side="left")

        for layer_idx in layers:
            counts = _reduce_counts(state.counts[: sub_keep * divisor], divisor)
            valid = counts > 0
            if not valid.any():
                continue
            sums = _reduce_sums(state.sums[layer_idx][: sub_keep * divisor], divisor)
            mean_k = sums / np.maximum(counts[:, None], 1)
            mean_k = mean_k.reshape(-1, self.num_kv_heads, self.head_dim)
            q = q_by_layer[layer_idx][:span].reshape(
                span, self.num_heads, self.head_dim
            )
            # Reported from the raw mean-K so the diagnostic means the same
            # thing in every mode: it is the "early pages have a bigger mean
            # vector" hypothesis, which the step 061 control run implicated.
            norms = np.maximum(
                norms,
                _page_reduce(
                    np.linalg.norm(mean_k, axis=-1).max(axis=-1),
                    granularity,
                    block_size,
                    num_pages,
                ),
            )
            if mode == "cosine":
                mean_k = mean_k / (
                    np.linalg.norm(mean_k, axis=-1, keepdims=True) + 1e-6
                )
                q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
            # The page-level mean K the diagnostic reports, resolved before the
            # per-sub-block reduction so the sidecar carries both the reduction
            # input (mean_k) and the page mean the ranking is compared against.
            diag_row = diag_out["filled"] if diag_out is not None else -1
            page_mean = (
                self._layer_page_mean(sums, counts, page_starts, page_ends)
                if diag_out is not None
                else None
            )
            if page_mean is not None:
                if mode == "cosine":
                    page_mean = page_mean / (
                        np.linalg.norm(page_mean, axis=-1, keepdims=True) + 1e-6
                    )
                diag_out["page_mean_k"][:, diag_row] = page_mean.astype(np.float16)
                per_layer_page = np.zeros((span, page_mean.shape[0]), np.float32)
            # GQA: the `head_group` query heads of a group all read the same KV
            # head, so the mean over heads is the mean query dotted with that
            # head's mean K.
            per_layer = np.zeros((span, counts.size), dtype=np.float32)
            for kv_head in range(self.num_kv_heads):
                q_group = q[:, kv_head * self.head_group : (kv_head + 1) * self.head_group]
                q_mean = q_group.astype(np.float32).mean(axis=1)  # [span, D]
                per_layer += q_mean @ mean_k[:, kv_head, :].T  # [span, sub]
                if page_mean is not None:
                    per_layer_page += q_mean @ page_mean[:, kv_head, :].T
                    diag_out["q_group_mean"][:, diag_row, kv_head, :] = (
                        q_mean.astype(np.float16)
                    )
            per_layer *= scale
            per_layer[:, ~valid] = -np.inf
            logits += _page_reduce(per_layer, granularity, block_size, num_pages)

            if diag_out is not None:
                self._fill_layer_diag(
                    diag_out,
                    diag_row,
                    mean_k,
                    counts,
                    per_layer,
                    valid,
                    per_layer_page * scale,
                )

        logits /= len(layers)
        return logits, norms

    def _layer_page_mean(
        self,
        sums: np.ndarray,
        counts: np.ndarray,
        page_starts: np.ndarray,
        page_ends: np.ndarray,
    ) -> np.ndarray:
        """Mean K per page, summed exactly from the per-sub-block sums.

        A page mean must not be a mean of means: the sub-blocks hold different
        token counts (a sub-block can straddle a page boundary, and the tail
        sub-blocks are partial), so the sums and counts are accumulated
        separately and divided once.
        """
        flat = sums.reshape(sums.shape[0], self.num_kv_heads * self.head_dim)
        cum_sum = np.concatenate(
            [
                np.zeros((1, flat.shape[1]), dtype=np.float64),
                np.cumsum(flat.astype(np.float64), axis=0),
            ]
        )
        cum_cnt = np.concatenate([[0.0], np.cumsum(counts.astype(np.float64))])
        page_sum = cum_sum[page_ends] - cum_sum[page_starts]
        page_cnt = (cum_cnt[page_ends] - cum_cnt[page_starts])[:, None]
        return (page_sum / np.maximum(page_cnt, 1)).reshape(
            -1, self.num_kv_heads, self.head_dim
        )

    def _fill_layer_diag(
        self,
        diag_out: dict,
        row: int,
        mean_k: np.ndarray,
        counts: np.ndarray,
        per_layer: np.ndarray,
        valid: np.ndarray,
        per_layer_page: np.ndarray,
    ) -> None:
        """Record one layer's per-sub-block and per-page logits."""
        if diag_out["page_offsets"] is None:
            diag_out["page_offsets"] = (
                np.arange(diag_out["page_mean_k"].shape[0], dtype=np.int64)
                * diag_out["block_size"]
            )
        diag_out["filled"] = row + 1
        diag_out["subblock_mean_k"][:, row] = mean_k.astype(np.float16)
        if row == 0:
            diag_out["subblock_counts"] = counts.astype(np.int32)
        diag_out["subblock_logits"][row, :] = np.where(
            valid, per_layer, np.nan
        ).mean(axis=0)
        diag_out["page_mean_logits"][row, :] = per_layer_page.mean(axis=0)

    @staticmethod
    def _summarize(
        logits: np.ndarray,
        norms: np.ndarray,
        eligible: np.ndarray,
        num_pages: int,
        topn: int,
    ) -> dict:
        masked = np.full_like(logits, -np.inf)
        masked[:, eligible] = logits[:, eligible]
        finite = np.isfinite(masked)
        row_has = finite.any(axis=1)
        scores = np.zeros(num_pages, dtype=np.float64)
        if row_has.any():
            rows = masked[row_has]
            rows = rows - rows.max(axis=1, keepdims=True)
            np.exp(rows, out=rows)
            rows /= rows.sum(axis=1, keepdims=True)
            scores = rows.sum(axis=0)

        order = np.argsort(-scores)
        return {
            "top_pages": [int(p) for p in order[:topn]],
            "top_scores": [float(scores[p]) for p in order[:topn]],
            "page_logits": [
                float(v) if np.isfinite(v) else None for v in logits.mean(axis=0)
            ],
            "page_kbar_norm": [float(v) for v in norms],
            "page_scores": [float(v) for v in scores],
        }

    def _eligible(
        self, num_pages: int, block_size: int, sink_tokens: int, recent_tokens: int
    ) -> np.ndarray:
        """Pages retrieval may pick: outside the sink page and the recent tail.

        The sink is the first page (its own slot in the fixed layout) and the
        recent tail is what the window already holds verbatim, so scoring them
        would only waste retrieval slots.
        """
        pages = np.arange(num_pages, dtype=np.int64)
        start = max(1, int(np.ceil(sink_tokens / block_size)))
        end = num_pages - max(0, recent_tokens // block_size)
        return pages[(pages >= start) & (pages < max(start, end))]

    def stats(self) -> dict:
        return {
            "subblock": self.subblock,
            "score_modes": self.modes,
            "granularities": self.granularities,
            "num_subblocks": self.num_subblocks,
            "trajectories": len(self._by_trajectory),
            "added_tokens": self.added_tokens,
            "layers_per_trajectory": {
                t.hex()[:12]: len(s.sums) for t, s in self._by_trajectory.items()
            },
        }

    def reset(self) -> None:
        self._by_trajectory.clear()
        self.added_tokens = 0
