# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem workspace configuration.

Every knob is read from the environment, matching the ``VLLM_KVMEM_SW_WINDOW``
convention the step 057 patch established. Unset values keep the KVMem arm on
its documented defaults; the whole subsystem stays inert unless
``VLLM_KVMEM_WORKSPACE`` is armed, which is checked in the core KV cache
manager.
"""

import os

DEFAULT_WORKSPACE_MB = 3072
DEFAULT_TRAJECTORY_PREFIX_TOKENS = 512
DEFAULT_MAX_WORKSPACE_TOKENS = 262144
DEFAULT_INDEX_SUBBLOCK = 128
DEFAULT_QUERY_SPAN = 256
DEFAULT_RETRIEVAL_TOPN = 16
DEFAULT_RECENT_TOKENS = 32768


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def workspace_host_bytes() -> int:
    """Host bytes the workspace may occupy (the pinned store's budget)."""
    return _env_int("VLLM_KVMEM_WORKSPACE_MB", DEFAULT_WORKSPACE_MB) << 20


def trajectory_prefix_tokens() -> int:
    """How many leading prompt tokens identify a trajectory.

    The client resends the whole conversation, so the first tokens are the root
    task message: they stay fixed while the conversation grows, and differ
    between conversations. That is the design's "hash(root task message
    identity)" without touching the client.
    """
    return _env_int(
        "VLLM_KVMEM_TRAJ_PREFIX", DEFAULT_TRAJECTORY_PREFIX_TOKENS
    )


def max_workspace_tokens() -> int:
    """Workspace capacity ceiling in tokens (design §7.2: 262,144)."""
    return _env_int("VLLM_KVMEM_WORKSPACE_TOKENS", DEFAULT_MAX_WORKSPACE_TOKENS)


def roundtrip_selftest() -> bool:
    """Copy each stored page back and compare it byte for byte (K1 exit)."""
    return bool(int(os.environ.get("VLLM_KVMEM_SELFTEST", "0")))


def dump_dir() -> str | None:
    """Directory for the page-table / occupancy dump, or None."""
    raw = os.environ.get("VLLM_KVMEM_DUMP", "").strip()
    return raw or None


def dump_kbar() -> bool:
    """Also dump the vectors behind the ranking, as a sidecar ``.npz``.

    Step 061 left one open question: with no needle in the prompt, pages 1, 2,
    5, 8 and 11 already score 22.7-26.2, so retrieval slots are spent before the
    needle is considered. Page-mean norms are nearly constant, so the answer is
    in the vectors, not in their magnitude - and a 200K prefill costs ~4.5
    minutes, which makes "one ingest, unlimited offline analysis" worth a
    sidecar. The JSON report stays the policy-free artifact; the arrays land
    next to it so the ranking can be re-derived under another reduction, mode or
    centring without re-running the engine.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_DUMP_KBAR", "0")))


def rawk_enabled() -> bool:
    """Capture the pre-RoPE q/k of the full-attention layers (K3 item 1).

    Arming this forces the full-attention layers onto the eager
    ``q_norm -> k_norm -> RoPE`` path: the production fused kernel
    (``fused_qk_rmsnorm_rope_gate``) computes the norm and RoPE in one pass and
    exposes no intermediate, so the raw K cannot be read out of it. The eager
    path is numerically equivalent but not bit-identical, which is why the
    design compares the KVMem arm only against itself (see design doc §5.2).
    """
    return bool(int(os.environ.get("VLLM_KVMEM_RAWK", "0")))


def index_subblock() -> int:
    """Mean-K index granularity in tokens (design §5.3: 128).

    This is the *storage* granularity, i.e. the finest one: the index keeps one
    fp32 running sum per sub-block and coarser score granularities are obtained
    by summing groups of these, so one run can report several granularities at
    once. One 1424-token page holds 11 of the design's sub-blocks.
    """
    return _env_int("VLLM_KVMEM_INDEX_SUBBLOCK", DEFAULT_INDEX_SUBBLOCK)


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    try:
        values = [int(part) for part in raw.replace(" ", "").split(",") if part]
    except ValueError:
        raise ValueError(f"{name} must be comma-separated integers, got {raw!r}") from None
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must be positive integers, got {raw!r}")
    return values


def _env_str_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [part for part in raw.replace(" ", "").split(",") if part]


def score_granularities() -> list[int]:
    """Score granularities to report, coarsest multiples of the storage one.

    Risk R1 says the page size the block layout forces on us (1424 tokens) is
    far coarser than the paper's 32-token blocks, and that the sub-block index
    is the mitigation. Reporting several granularities from one prefill is what
    turns "retrieval does not work" into "retrieval does not work *at this
    granularity*".
    """
    base = index_subblock()
    values = _env_int_list("VLLM_KVMEM_SCORE_GRANULARITIES", [base])
    for value in values:
        if value % base:
            raise ValueError(
                f"VLLM_KVMEM_SCORE_GRANULARITIES entries must be multiples of "
                f"VLLM_KVMEM_INDEX_SUBBLOCK ({base}), got {value}"
            )
    return sorted(set(values))


def score_modes() -> list[str]:
    """Which page-logit variants to report (see :func:`score_mode`)."""
    values = _env_str_list("VLLM_KVMEM_SCORE_MODES", ["dot"])
    for value in values:
        if value not in SCORE_MODES:
            raise ValueError(
                f"VLLM_KVMEM_SCORE_MODES entries must be in {SCORE_MODES}, "
                f"got {value!r}"
            )
    return values


def query_span() -> int:
    """How many trailing prompt tokens act as the retrieval query.

    The design's "current query span" is the delta of the incoming prompt; for
    an agent turn that is the tail, which is where the question sits. Only the
    tail is captured because the full pre-RoPE q of a 262K prompt would be
    3.2 GiB.
    """
    return _env_int("VLLM_KVMEM_QUERY_SPAN", DEFAULT_QUERY_SPAN)


def retrieval_topn() -> int:
    """How many pages retrieval would put in the retrieval slots."""
    return _env_int("VLLM_KVMEM_TOPN", DEFAULT_RETRIEVAL_TOPN)


def recent_tokens() -> int:
    """Size of the recent tail that retrieval must not compete with.

    Design §7.1 wants this to track the last tool output, clamped to
    [16K, 64K]; until that distribution is measured, the midpoint is used and
    the value is reported with every score dump so a ranking can be re-judged
    under another policy offline.
    """
    return _env_int("VLLM_KVMEM_RECENT", DEFAULT_RECENT_TOKENS)


SCORE_MODES = ("dot", "cosine")


def score_mode() -> str:
    """How the page logit is formed (step 061 measurement knob).

    ``dot`` is the design's ``q . kbar / sqrt(d)``. ``cosine`` normalises both
    sides first, which removes the per-sub-block magnitude differences that the
    step 061 control run showed to be carrying the whole ranking.

    (A "subtract the mean vector" mode was tried and dropped: centring kbar
    over sub-blocks is a constant offset in logit space, so it cannot reorder
    anything, and centring q by its own span mean zeroes the query whenever the
    span is homogeneous — exactly the focused-query case it was meant for.)
    """
    raw = os.environ.get("VLLM_KVMEM_SCORE_MODE", "").strip() or "dot"
    if raw not in SCORE_MODES:
        raise ValueError(
            f"VLLM_KVMEM_SCORE_MODE must be one of {SCORE_MODES}, got {raw!r}"
        )
    return raw
