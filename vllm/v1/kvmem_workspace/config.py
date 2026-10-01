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
DEFAULT_AUTHORITY_TRAJECTORIES = 2


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


def authority_enabled() -> bool:
    """Keep the pre-RoPE rotary prefix as the rematerialisation authority.

    A stored page's K is baked at the positions it was written at. Putting it
    back into the window means giving it *new* positions (the fixed-slot layout
    compresses the window to ``0..B-1``), and NVFP4 cannot be re-rotated in
    place: the fp8 block scale covers 16 consecutive head dims, so every group
    the rotation touches needs a new scale. The design's answer is to keep the
    pre-RoPE rotary prefix (64 of the 256 head dims, fp16) as the authority and
    rebuild from it, once, per re-entry -- never from a previously rotated copy,
    which is what makes displacement drift-free (design §5.3, step 063).

    Arming this costs ``num_kv_heads * rotary_dim * 2`` bytes per token per
    layer of host memory (512 B here, so 2.0 GiB over a full 262,144-token
    workspace) and nothing on the device. Requires ``VLLM_KVMEM_RAWK``, which
    is where the pre-RoPE K comes from. It changes no attention behaviour: the
    authority is written and read back for verification only until the assembly
    path lands.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_AUTHORITY", "0")))


def load_enabled() -> bool:
    """Assemble stored pages back into a later request's prefix (step 066).

    When armed, a request whose leading tokens match a trajectory that already
    has a contiguous run of stored pages *and* a mamba snapshot at the run's
    end gets ``num_computed_tokens`` jumped to that run's token boundary: the
    worker copies the pages (and the snapshot) into the request's freshly
    allocated blocks, and only the remainder is prefilled. The pages go back at
    their *original* positions, so no re-RoPE is involved here -- this is the
    connector-level assembly half of the design's step 5.1 flow; compressing
    the window to fixed slots (which does need re-RoPE) is the step after it.

    Requires ``VLLM_KVMEM_WORKSPACE`` (the store must exist to load from it)
    and the ``MambaManager`` external-allocation patch, without which the
    mamba groups would allocate one state slot per assembled page.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_LOAD", "0")))


def snapshot_keep() -> int:
    """How many page-aligned mamba snapshots to keep per trajectory (066).

    One snapshot is one full state of every mamba group (80.4 MiB here: 48
    layers x (conv 102,400 B + bf16 ssm 1,572,864 B)). A snapshot can only be
    taken while its block is inside the CoW window (two blocks), so captures
    ride the prefill steps: each step that ends on a page boundary grabs the
    state of the running slot. The ring must hold a boundary until the
    workspace's page prefix reaches it, which takes ``sliding window /
    max_num_batched_tokens`` steps -- with a sparse capture interval (see
    :func:`snapshot_every_pages`) that turns into ``interval_pages x block
    size / max_num_batched_tokens + 2`` rows.
    """
    return _env_int("VLLM_KVMEM_SNAPSHOT_KEEP", 20)


def snapshot_trajectories() -> int:
    """How many trajectories may hold a snapshot ring at once (step 066).

    The rings are per-trajectory (a global FIFO lets one trajectory's prefill
    evict another's snapshots and silently disable its assembly -- measured in
    the step 066 second boot). Each ring is ``snapshot_keep`` rows of the full
    mamba state, so this is a host-memory bound: past it the least recently
    started ring is dropped whole and reported.
    """
    return _env_int("VLLM_KVMEM_SNAPSHOT_TRAJ", 2)


def snapshot_every_pages() -> int:
    """Capture one snapshot every N page boundaries (step 066).

    The assembled boundary must carry an exact recurrent state, so it can only
    sit on a *captured* boundary; a sparser capture simply caps the assemblable
    prefix at the newest sparse boundary below the page run (at most N-1 pages
    of the run go unassembled). Each row costs ``snapshot_keep``-independent
    80.4 MiB of host, and the ring row count scales with the interval, so
    denser is not free.
    """
    return _env_int("VLLM_KVMEM_SNAPSHOT_EVERY_PAGES", 8)


def authority_tokens() -> int:
    """Token capacity of one trajectory's authority region."""
    return _env_int("VLLM_KVMEM_AUTHORITY_TOKENS", max_workspace_tokens())


def authority_trajectories() -> int:
    """How many trajectories may hold an authority region at once.

    One region is 2.0 GiB at full workspace length, so this is a host-memory
    bound rather than a semantic one; past it the pages are still stored and
    simply cannot be rematerialised, which is counted and reported.
    """
    return _env_int("VLLM_KVMEM_AUTHORITY_TRAJ", DEFAULT_AUTHORITY_TRAJECTORIES)


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


def debug_enabled() -> bool:
    """One-shot observation of the window request's engine-side state.

    Step 073 diagnostic only (the N=55 empty-output fault): it prints what the
    scheduler actually did to a rewritten request -- how many tokens it adopted
    as computed, how many block hashes it carries, what its status and sampled
    tokens were at finish. Nothing about behaviour changes.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_DEBUG", "0")))


def bake_verify() -> int:
    """Layer-pages read back after the retrieval-slot bake (step 074).

    The bake log can only prove a copy was *issued*; this proves the bytes landed
    in the physical block the decode step will read, per stored kv cache group.
    ``VLLM_KVMEM_BAKE_VERIFY=n`` checks the first ``n`` layers of every slot of
    every group (0 = off, the default: each check is a blocking device->host copy
    of one page, so 55 slots x 2 groups x n layers is real time).
    """
    return _env_int("VLLM_KVMEM_BAKE_VERIFY", 0)


def viewport_enabled() -> bool:
    """Rewrite a long request onto the fixed-slot compressed window (step 072).

    This is the design §5.1 *re-bake* route, and it is single-coordinate: the
    request's prefill token sequence *is* the compressed window
    (``prompt[:S+N] + prompt[L-R:]``), so positions/slot_mapping/block table all
    stay in engine-native window coordinates. No frame translation, no
    scheduler jump, no mamba snapshot restore -- the window's own prefill runs
    the GDN recurrence over real history tokens, and the only post-processing
    is baking the scored pages into the retrieval slots from the raw-K
    authority.

    Requires ``VLLM_KVMEM_RAWK`` (the capture) and ``VLLM_KVMEM_AUTHORITY``
    (the pre-RoPE rotary prefix) -- without them the slots have nothing to
    rebuild from. Independent of ``VLLM_KVMEM_LOAD`` (the step 066 in-place
    assembly): the two routes never mix inside one request.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_VIEWPORT", "0")))


def viewport_recent_tokens() -> int:
    """Tail length R of the compressed window (design §7.1: [16K, 64K]).

    The window is ``S + N + R`` tokens: one sink page, N retrieval pages, then
    the prompt's last R tokens. R is what keeps the question (and the nearest
    context) verbatim; the mid-section it displaces is what retrieval has to
    re-represent.
    """
    return _env_int("VLLM_KVMEM_VIEWPORT_RECENT", 16384)


def viewport_retrieval_pages() -> int:
    """Number N of retrieval slots, in whole pages (design §5.1: time-ordered).

    55 pages x 1424 = 78,320 tokens of retrieval budget: with S=1456 and
    R=16384 the window is 96,160 tokens, which alongside a 32,768 generation
    reserve fits the 163,072 pool with room to spare. Every slot is rewritten
    from the authority on every scored request, so N is also the per-request
    bake volume.
    """
    return _env_int("VLLM_KVMEM_VIEWPORT_PAGES", 55)


def sweep_enabled() -> bool:
    """Store the pages a finished request still holds (step 072).

    K1 copies pages the sliding window *evicted*, so a 200K ingest ends with
    only the ~25 out-of-window pages stored; the ~114 in-window pages (the
    mid-section the retrieval slots need V/non-rotary bytes from) are freed
    with the request. The sweep stores those remaining pages at
    ``request_finished`` -- the blocks are kept alive until the copies land
    (the connector claims them, and the scheduler frees them once the request
    id comes back through ``get_finished``), which is the same
    copy-before-free discipline as K1.
    """
    return bool(int(os.environ.get("VLLM_KVMEM_SWEEP", "0")))


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
