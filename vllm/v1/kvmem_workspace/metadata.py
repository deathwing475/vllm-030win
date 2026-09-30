# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem workspace transfer metadata."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


@dataclass
class KVMemPageTransfer:
    """One page moving between a GPU block and a workspace slot.

    ``group_id`` is the KV cache group (only KVMem sliding-window groups are
    stored). ``page_index`` is the page's position inside the trajectory — the
    logical block index in the request's block table — which together with the
    trajectory key forms the workspace identity, deliberately *not* the
    prefix-cache block hash.
    """

    group_id: int
    block_id: int
    page_index: int
    slot: int


@dataclass
class KVMemStoreJob:
    job_id: int
    trajectory: bytes
    pages: list[KVMemPageTransfer] = field(default_factory=list)
    # Step 066: while storing evicted page k (whose run through the window just
    # ended), also copy the mamba groups' state block at position k -- its CoW
    # slot holds the exact state after token (k+1)*block_size, which is the
    # boundary an assembled prefix would end at. Entries are
    # (group_id, gpu_block_id, page_index); the host destination is the
    # worker's snapshot region, keyed by (trajectory, boundary).
    mamba_snapshots: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class KVMemPageLoad:
    """One workspace page copied back into a request block (step 066).

    The inverse of :class:`KVMemPageTransfer`: the request's freshly allocated
    GPU block receives the page the sliding window once evicted. The block ids
    come from ``update_state_after_alloc``, i.e. they are the real physical
    blocks of this request's block-table rows 0..E-1.
    """

    group_id: int
    block_id: int
    page_index: int
    slot: int


@dataclass
class KVMemLoadJob:
    """Prefix assembly for one request (step 066).

    ``num_tokens`` is the token boundary the scheduler jumped
    ``num_computed_tokens`` to; the worker copies every entry of ``pages``
    (workspace page -> request block) and of ``mamba_snapshots`` (snapshot
    region -> the request's single real mamba state block at position E-1,
    carried as (group_id, gpu_block_id, page_index) with page_index only
    identifying the snapshot's boundary) before the request may run. The
    scheduler is told the request is ready through ``finished_recving``, which
    is what promotes it out of ``WAITING_FOR_REMOTE_KVS``.
    """

    job_id: int
    req_id: str
    trajectory: bytes
    num_tokens: int
    pages: list[KVMemPageLoad] = field(default_factory=list)
    mamba_snapshots: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class KVMemSnapshotRequest:
    """Capture the mamba state at a page-aligned boundary (step 066).

    An assembled prefix needs the recurrent state *exactly at* its token
    boundary, and the sliding window keeps the workspace's contiguous page
    prefix ~W tokens behind the live state, so the state cannot be fetched
    when a page is evicted -- it must be captured while its block is still in
    the CoW window. ``build_connector_meta`` picks the block one page behind
    this step's end (``position c//block_size - 1``, alive and holding the
    exact state after ``c//block_size*block_size`` tokens) and the worker
    copies its slot into the host snapshot region, keyed by
    (trajectory, boundary).
    """

    trajectory: bytes
    boundary: int
    # (group_id, gpu_block_id) per mamba group; the layer dim is expanded
    # worker-side from the group's registered layer views.
    blocks: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class KVMemStepSpan:
    """The token range one request computes in this step (K3 item 1-2).

    The retrieval index is position-keyed, and the model forward only knows the
    absolute positions it was handed, not which trajectory they belong to. This
    span is the bridge: the scheduler knows both, and ``build_connector_meta``
    runs before ``_update_after_schedule`` bumps ``num_computed_tokens``, so
    ``start`` here is the true first position of the step.
    """

    trajectory: bytes
    start: int
    num_tokens: int


@dataclass
class KVMemScoreRequest:
    """A prompt whose prefill just finished: score its workspace pages."""

    trajectory: bytes
    request_id: str
    num_tokens: int
    block_size: int
    sink_tokens: int
    recent_tokens: int


@dataclass
class KVMemConnectorMetadata(KVConnectorMetadata):
    store_jobs: list[KVMemStoreJob] = field(default_factory=list)
    load_jobs: list[KVMemLoadJob] = field(default_factory=list)
    snapshot_requests: list[KVMemSnapshotRequest] = field(default_factory=list)
    spans: list[KVMemStepSpan] = field(default_factory=list)
    score_requests: list[KVMemScoreRequest] = field(default_factory=list)


@dataclass
class KVMemWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler: which transfers have finished their DMA."""

    completed_store_jobs: list[int] = field(default_factory=list)
    # (trajectory, boundary) pairs whose mamba snapshot has landed (step 066).
    completed_snapshots: list[tuple[bytes, int]] = field(default_factory=list)
    # (trajectory, boundary) pairs evicted from the snapshot ring to make room
    # (step 066); the scheduler must stop matching against them.
    removed_snapshots: list[tuple[bytes, int]] = field(default_factory=list)
    # Request ids whose prefix assembly has landed (step 066): the scheduler
    # promotes these out of WAITING_FOR_REMOTE_KVS.
    finished_load_reqs: list[str] = field(default_factory=list)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, KVMemWorkerMetadata)
        self.completed_store_jobs.extend(other.completed_store_jobs)
        self.completed_snapshots.extend(other.completed_snapshots)
        self.removed_snapshots.extend(other.removed_snapshots)
        self.finished_load_reqs.extend(other.finished_load_reqs)
        return self
