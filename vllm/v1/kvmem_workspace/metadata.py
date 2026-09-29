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
class KVMemConnectorMetadata(KVConnectorMetadata):
    store_jobs: list[KVMemStoreJob] = field(default_factory=list)


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
