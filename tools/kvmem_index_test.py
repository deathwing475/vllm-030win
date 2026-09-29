"""Offline unit test for the KVMem Mean-K index / page scoring math (step 061).

Synthetic data with a known needle: if the scoring pipeline (GQA head pairing,
page-level max reduce, softmax over eligible pages) is right, the needle's page
must come out on top. Run before spending 250 s on a real 210K prefill, because
a silent index bug would look exactly like "retrieval does not work".

Usage: python kvmem_index_test.py
"""
import os
import sys

import numpy as np

NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
SUBBLOCK = 128
BLOCK_SIZE = 1424
NUM_PAGES = 120
NUM_TOKENS = NUM_PAGES * BLOCK_SIZE
SUBBLOCKS_PER_PAGE = BLOCK_SIZE // SUBBLOCK

os.environ.setdefault("VLLM_KVMEM_INDEX_SUBBLOCK", str(SUBBLOCK))

from vllm.v1.kvmem_workspace.index import KVMemMeanKIndex  # noqa: E402

rng = np.random.default_rng(20260929)
LAYERS = [3, 7, 11]
TRAJ = b"unit-test-traj"
NEEDLE_SUBBLOCK = 700
# The engine attributes a sub-block to the page holding its FIRST token, and
# 1424 is not a multiple of 128 (11.125 sub-blocks per page), so this is
# (700*128)//1424 = 62 - not 700//11 = 63, which is what the buggy
# block_size // granularity grouping implied.
NEEDLE_PAGE = (NEEDLE_SUBBLOCK * SUBBLOCK) // BLOCK_SIZE
print(f"needle sub-block {NEEDLE_SUBBLOCK} -> page {NEEDLE_PAGE}")

needle_v = rng.normal(size=(NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
needle_v /= np.linalg.norm(needle_v, axis=1, keepdims=True)

index = KVMemMeanKIndex(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
positions = np.arange(NUM_TOKENS, dtype=np.int64)

# Feed the workspace one page-chunk at a time, exactly like the prefill does.
for chunk_start in range(0, NUM_TOKENS, BLOCK_SIZE):
    chunk_pos = positions[chunk_start : chunk_start + BLOCK_SIZE]
    k_by_layer = {}
    for layer in LAYERS:
        k = rng.normal(size=(chunk_pos.size, NUM_KV_HEADS * HEAD_DIM)).astype(
            np.float16
        ) * np.float16(0.05)
        sub_ids = chunk_pos // SUBBLOCK
        hit = sub_ids == NEEDLE_SUBBLOCK
        if hit.any():
            k[hit] = np.tile(
                needle_v.reshape(-1).astype(np.float16), (int(hit.sum()), 1)
            )
        k_by_layer[layer] = k
    index.add(TRAJ, chunk_pos, k_by_layer)

# Query = the needle's content, placed on the matching query heads of every
# layer. RoPE-free frame on both sides, which is the whole point of capturing
# pre-RoPE q and k.
span = 64
q = np.zeros((span, NUM_HEADS * HEAD_DIM), dtype=np.float16)
group = NUM_HEADS // NUM_KV_HEADS
for h in range(NUM_HEADS):
    q[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = needle_v[h // group].astype(np.float16)
q_by_layer = {layer: q for layer in LAYERS}

report = index.score(
    TRAJ,
    q_by_layer,
    block_size=BLOCK_SIZE,
    num_tokens=NUM_TOKENS,
    sink_tokens=BLOCK_SIZE,
    recent_tokens=32768,
    topn=8,
)

failures = []

# --- direct numeric check of the scatter-add itself ------------------------
# If the sub-block mapping were off by anything, the stored mean would not
# equal the arithmetic mean of the rows that were fed in.
state = index._by_trajectory[TRAJ]
got_rows = int(state.counts[NEEDLE_SUBBLOCK])
print(f"sub-block {NEEDLE_SUBBLOCK} counted {got_rows} rows (expect {SUBBLOCK})")
if got_rows != SUBBLOCK:
    failures.append(
        f"sub-block {NEEDLE_SUBBLOCK} counted {got_rows} rows, expected {SUBBLOCK}"
    )
stored_mean = state.sums[3][NEEDLE_SUBBLOCK] / max(got_rows, 1)
want_mean = needle_v.reshape(-1).astype(np.float32)
max_diff = float(np.abs(stored_mean - want_mean).max())
print(f"sub-block mean check: max |stored - fed| = {max_diff:.3e}")
if max_diff > 1e-3:
    failures.append(f"stored sub-block mean differs from the fed rows ({max_diff})")
# A sub-block with no rows must stay empty rather than pick up neighbours.
if int(state.counts[NEEDLE_SUBBLOCK + 1]) != SUBBLOCK:
    failures.append("neighbouring sub-block count is wrong")
untouched = (NUM_TOKENS // SUBBLOCK) + 10
if int(state.counts[untouched]) != 0:
    failures.append(f"untouched sub-block {untouched} has a non-zero count")
if float(np.abs(state.sums[3][untouched]).max()) != 0.0:
    failures.append("an untouched sub-block has non-zero sums")

if "error" in report:
    failures.append(f"score() returned an error: {report['error']}")
else:
    print(f"scored {report['num_layers_scored']} layer(s), "
          f"{report['num_pages']} pages, {len(report['eligible'])} eligible")
    print(f"top-{report['topn']} pages: {report['top_pages']}")
    print(f"needle page {NEEDLE_PAGE} in top-{report['topn']}: "
          f"{NEEDLE_PAGE in report['top_pages']}")

    if NEEDLE_PAGE not in report["top_pages"]:
        failures.append(
            f"needle page {NEEDLE_PAGE} not in top-{report['topn']} "
            f"{report['top_pages']}"
        )
    if report["top_pages"][0] != NEEDLE_PAGE:
        failures.append(
            f"needle page {NEEDLE_PAGE} is not rank 1 (got {report['top_pages'][0]})"
        )
    if NEEDLE_PAGE not in report["eligible"]:
        failures.append("needle page is not eligible")

    # Page 0 (the sink page) must be excluded from the eligible set.
    if 0 in report["eligible"]:
        failures.append("sink page 0 is eligible")
    last_page = NUM_PAGES - 1
    if last_page in report["eligible"]:
        failures.append(f"recent page {last_page} is eligible")

    # Page-level max reduce: a page with exactly one strong sub-block must beat
    # a page with many mediocre ones. Check the needle sub-block's own page is
    # the max over its sub-blocks, i.e. its score dominates its page.
    logits = np.array(
        [v if v is not None else -np.inf for v in report["page_logits"]]
    )
    order = np.argsort(-logits)
    if order[0] != NEEDLE_PAGE:
        failures.append(
            f"needle page is not rank 1 by raw page logit (got {order[0]})"
        )

    # Determinism: the same query scored twice must give the same answer.
    again = index.score(
        TRAJ, q_by_layer, BLOCK_SIZE, NUM_TOKENS, BLOCK_SIZE, 32768, 8
    )
    if again["top_pages"] != report["top_pages"]:
        failures.append("scoring is not deterministic across two calls")

print()
if failures:
    for failure in failures:
        print(f"FAIL: {failure}")
    sys.exit(1)

# --- the normalisation modes must keep the needle on top too ---------------
# They exist because the step 061 control run showed the raw dot product's
# ranking is carried by per-page magnitude, not content; if a mode cannot even
# pass this synthetic test it is not worth an engine run.
def feed(index, rng_seed):
    local = np.random.default_rng(rng_seed)
    for chunk_start in range(0, NUM_TOKENS, BLOCK_SIZE):
        chunk_pos = positions[chunk_start : chunk_start + BLOCK_SIZE]
        k_by_layer = {}
        for layer in LAYERS:
            k = local.normal(size=(chunk_pos.size, NUM_KV_HEADS * HEAD_DIM)).astype(
                np.float16
            ) * np.float16(0.05)
            hit = (chunk_pos // SUBBLOCK) == NEEDLE_SUBBLOCK
            if hit.any():
                k[hit] = np.tile(
                    needle_v.reshape(-1).astype(np.float16), (int(hit.sum()), 1)
                )
            k_by_layer[layer] = k
        index.add(TRAJ, chunk_pos, k_by_layer)


for mode in ("cosine",):
    os.environ["VLLM_KVMEM_SCORE_MODES"] = mode
    alt = KVMemMeanKIndex(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    feed(alt, 20260929)
    alt_report = alt.score(
        TRAJ, q_by_layer, BLOCK_SIZE, NUM_TOKENS, BLOCK_SIZE, 32768, 8
    )
    top = alt_report["variants"][f"{mode}@{SUBBLOCK}"]["top_pages"][0]
    ok = top == NEEDLE_PAGE
    print(f"mode {mode}: needle page rank-1 = {ok} (top={top})")
    if not ok:
        print(f"FAIL: mode {mode} did not rank the needle page first")
        sys.exit(1)

# --- coarse granularities must be exactly the sums of the fine ones ---------
# The multi-variant report is only meaningful if dot@128 derived from a 32-token
# index equals dot@128 measured on a 128-token index; otherwise the ablation
# would be comparing different quantities.
os.environ["VLLM_KVMEM_SCORE_MODES"] = "dot"
os.environ["VLLM_KVMEM_SCORE_GRANULARITIES"] = "32,64,128"
os.environ["VLLM_KVMEM_INDEX_SUBBLOCK"] = "32"
fine = KVMemMeanKIndex(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
feed(fine, 20260929)
fine_report = fine.score(
    TRAJ, q_by_layer, BLOCK_SIZE, NUM_TOKENS, BLOCK_SIZE, 32768, 8
)

os.environ["VLLM_KVMEM_SCORE_GRANULARITIES"] = "128"
os.environ["VLLM_KVMEM_INDEX_SUBBLOCK"] = "128"
coarse = KVMemMeanKIndex(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
feed(coarse, 20260929)
coarse_report = coarse.score(
    TRAJ, q_by_layer, BLOCK_SIZE, NUM_TOKENS, BLOCK_SIZE, 32768, 8
)

for gran in ("32", "64", "128"):
    top = fine_report["variants"][f"dot@{gran}"]["top_pages"][0]
    print(f"fine index, dot@{gran}: needle page rank-1 = {top == NEEDLE_PAGE} (top={top})")
    if top != NEEDLE_PAGE:
        print(f"FAIL: dot@{gran} did not rank the needle page first")
        sys.exit(1)

a = fine_report["variants"]["dot@128"]["page_logits"]
b = coarse_report["variants"]["dot@128"]["page_logits"]
max_diff = max(
    abs(x - y) for x, y in zip(a, b) if x is not None and y is not None
)
print(f"dot@128 from a 32-index vs from a 128-index: max |diff| = {max_diff:.3e}")
if max_diff > 1e-4:
    print("FAIL: coarse granularity is not the exact sum of the fine one")
    sys.exit(1)

# --- page boundaries must be the real ones ---------------------------------
# A marker token at a known offset must make THAT page (offset // block_size)
# the top page, at every granularity. This is the check that catches a
# sub-block-to-page reduce walking off the true boundaries: 1424 = 16 x 89, so
# no power-of-two sub-block size divides it, and grouping by
# block_size // granularity drifts by 16 tokens per page at a 128-token
# sub-block (304 tokens by page 19).
os.environ["VLLM_KVMEM_SCORE_MODES"] = "dot"
os.environ["VLLM_KVMEM_INDEX_SUBBLOCK"] = "32"
os.environ["VLLM_KVMEM_SCORE_GRANULARITIES"] = "32,64,128"
for marker_offset in (2112, 27776, 100384):
    want_page = marker_offset // BLOCK_SIZE
    idx = KVMemMeanKIndex(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    local = np.random.default_rng(11)
    for cs in range(0, NUM_TOKENS, BLOCK_SIZE):
        cp = positions[cs : cs + BLOCK_SIZE]
        kbl = {}
        for layer in LAYERS:
            k = (
                local.normal(size=(cp.size, NUM_KV_HEADS * HEAD_DIM)) * 0.01
            ).astype(np.float16)
            hit = cp == marker_offset
            if hit.any():
                k[hit] = (needle_v.reshape(-1) * 16).astype(np.float16)
            kbl[layer] = k
        idx.add(TRAJ, cp, kbl)
    rep = idx.score(TRAJ, q_by_layer, BLOCK_SIZE, NUM_TOKENS, BLOCK_SIZE, 32768, 8)
    for gran in ("32", "64", "128"):
        top = rep["variants"][f"dot@{gran}"]["top_pages"][0]
        ok = top == want_page
        print(f"marker at {marker_offset} (page {want_page}), dot@{gran}: "
              f"top page = {top} -> {ok}")
        if not ok:
            print("FAIL: page reduce is not aligned to the real page boundaries")
            sys.exit(1)

print("PASS: KVMem Mean-K index / page scoring unit test")
