# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMemConnector.

The connector half of the KVMem host KV workspace
(``docs/vllm-030win-调研-KVMem虚拟化KV工作区.md``). It exists to answer one
question the production stack could not: where do the pages that the bounded
attention window drops actually *go*?

Stage 1a (step 057) made a prompt longer than the KV pool prefillable by giving
the full-attention layers a per-layer sliding window. That window, though, threw
the scrolled-out history away, so a later step had nothing to retrieve from. This
connector takes those pages, copies them into a pinned host workspace keyed by
``(trajectory, page_index)``, and keeps them out of the block pool until the copy
has completed — i.e. the copy-before-free half of the design's stage 1a.

Selection: ``--kv-transfer-config '{"kv_connector":"KVMemConnector",
"kv_role":"kv_both"}'`` together with ``VLLM_KVMEM_SW_WINDOW`` and
``VLLM_KVMEM_WORKSPACE=1``. Nothing here runs otherwise, and the store is a
best-effort cache (a dropped page is a future retrieval miss, not a KV transfer
the engine must wait for).
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.kvmem_workspace.manager import KVMemWorkspaceScheduler
from vllm.v1.kvmem_workspace.metadata import KVMemConnectorMetadata
from vllm.v1.kvmem_workspace.worker import KVMemWorkspaceWorker

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class KVMemConnector(KVConnectorBase_V1, SupportsHMA):
    """Host KV workspace store for the KVMem arm."""

    @property
    def requires_kv_delivery(self) -> bool:
        # A best-effort store: a dropped page is a future retrieval miss, not a
        # transfer the engine has to wait for.
        return False

    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self.scheduler_manager: KVMemWorkspaceScheduler | None = None
        self.worker_handler: KVMemWorkspaceWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = KVMemWorkspaceScheduler(
                vllm_config, kv_cache_config, role
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = KVMemWorkspaceWorker(vllm_config, kv_cache_config)

    # --- Worker-side methods ---

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, KVMemConnectorMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        # Stage 1 K1 stored pages only; step 066 issues the prefix-assembly
        # copies here (the assembling request's step schedules zero tokens, and
        # the no-forward worker path calls only this hook).
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    # --- Scheduler-side methods ---

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def register_workspace_retained_blocks(self, blocks: list) -> bool:
        if self.scheduler_manager is None:
            return False
        return self.scheduler_manager.register_workspace_retained_blocks(blocks)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return KVMemConnectorMetadata()

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    # NOTE: only for KVMemConnector (mirrors SimpleCPUOffloadConnector).
    def has_pending_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_stores()
        return False

    def reset_cache(self) -> bool | None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.reset()
        return None

    def workspace_stats(self) -> dict:
        stats: dict = {}
        if self.scheduler_manager is not None:
            stats["scheduler"] = self.scheduler_manager.stats()
        if self.worker_handler is not None:
            stats["worker"] = self.worker_handler.stats()
        return stats
