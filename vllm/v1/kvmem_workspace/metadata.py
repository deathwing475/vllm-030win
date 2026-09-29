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
    spans: list[KVMemStepSpan] = field(default_factory=list)
    score_requests: list[KVMemScoreRequest] = field(default_factory=list)


@dataclass
class KVMemWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler: which store jobs have finished their DMA."""

    completed_store_jobs: list[int] = field(default_factory=list)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, KVMemWorkerMetadata)
        self.completed_store_jobs.extend(other.completed_store_jobs)
        return self
