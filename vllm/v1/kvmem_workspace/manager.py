# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem workspace, scheduler side.

Owns the workspace page table (``(trajectory, page_index) -> host slot``), the
host slot allocator, and the lifetime of the GPU pages that were evicted from a
KVMem sliding window. A page that the core KV cache manager handed over here has
*not* been returned to the block pool; it is freed exactly once, when the worker
reports that its device->host copy has completed.

This is the "copy-before-free" half of the design's stage 1a
(``docs/vllm-030win-调研-KVMem虚拟化KV工作区.md`` §12.4 item 2): the sliding
window alone threw the evicted history away.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
from vllm.v1.kvmem_workspace import config
from vllm.v1.kvmem_workspace.groups import workspace_group_ids
from vllm.v1.kvmem_workspace.metadata import (
    KVMemConnectorMetadata,
    KVMemPageTransfer,
    KVMemStoreJob,
)
from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)


@dataclass
class _JobStatus:
    job_id: int
    trajectory: bytes
    # Pages still held out of the block pool, keyed by block id.
    retained: list = field(default_factory=list)
    # (group_id, page_index) pairs this job made resident.
    pages: list[tuple[int, int]] = field(default_factory=list)


class KVMemWorkspaceScheduler:
    def __init__(self, vllm_config, kv_cache_config, role: KVConnectorRole):
        del role
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.group_ids = workspace_group_ids(kv_cache_config)
        self.page_bytes = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.page_size_bytes
            for group_id in self.group_ids
        }
        self.block_size = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size
            for group_id in self.group_ids
        }

        self.block_pool = None
        self.trajectory_prefix_tokens = config.trajectory_prefix_tokens()

        # Host slot budget: one slot holds one page of every stored group.
        per_slot_bytes = sum(
            self.page_bytes[group_id]
            * len(kv_cache_config.kv_cache_groups[group_id].layer_names)
            for group_id in self.group_ids
        )
        budget = config.workspace_host_bytes()
        self.num_slots = max(1, budget // per_slot_bytes) if per_slot_bytes else 0

        self._free_slots: list[int] = list(range(self.num_slots - 1, -1, -1))
        # (trajectory, token offset) -> slot. Keyed by token offset rather than
        # by page index so that groups sharing a block size share one slot: the
        # two 8-layer attention groups of this model cover the same token
        # ranges, and giving each its own slot would waste half the region.
        # Groups with different block sizes fall out to distinct slots.
        self._page_table: dict[tuple[bytes, int], int] = {}
        self._jobs: dict[int, _JobStatus] = {}
        self._next_job_id = 0

        # Blocks retained by the core manager this step, keyed by block id.
        self._retained_by_block_id: dict[int, object] = {}
        # request_id -> trajectory key (the request is still in flight).
        self._req_trajectory: dict[str, bytes] = {}
        # request_id -> (evicted, stored, dropped) when the request was first
        # seen, so the per-request summary is a delta and not a session total.
        self._req_baseline: dict[str, tuple[int, int, int]] = {}

        self.pages_evicted = 0
        self.pages_stored = 0
        self.pages_dropped = 0
        self.slots_used = 0

        logger.info(
            "vllm-030win patch (step 060): KVMem workspace scheduler ready: "
            "groups=%s, %d host slots (%.2f GiB budget, %.2f MiB per slot)",
            self.group_ids,
            self.num_slots,
            budget / (1024**3),
            per_slot_bytes / (1024**2),
        )
        if not self.group_ids:
            logger.warning(
                "vllm-030win patch (step 060): KVMem workspace armed but no "
                "sliding-window group matched; evicted pages will be dropped "
                "as before"
            )

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self.block_pool = gpu_block_pool
        self._assert_viewport_fits(gpu_block_pool)

    def _assert_viewport_fits(self, gpu_block_pool) -> None:
        """K2: fail loudly when the viewport cannot fit in the pool.

        The per-request viewport is bounded by the sliding window
        (``max_admission_blocks_per_request``), so a pool smaller than that
        bound can never admit a request — every one of them would sit in the
        waiting queue forever, which is precisely the silent-stall failure the
        design warns about (the reference's guard only checked its context
        window and killed the server instead). Sizing the pool by hand
        (``--kv-cache-memory-bytes``) skips the profiling that would otherwise
        catch this, so check it explicitly and refuse to start.
        """
        if not self.group_ids:
            return
        vllm_config = self.vllm_config
        max_model_len = vllm_config.model_config.max_model_len
        max_in_flight = vllm_config.max_in_flight_tokens
        needed = 0
        for group_id in self.group_ids:
            spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            needed += spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight, max_model_len=max_model_len
            )
        # The pool also carries one null block, and every other kv cache group
        # (the 48 GDN layers here) needs its own viewport.
        available = len(gpu_block_pool.blocks)
        other_groups = len(self.kv_cache_config.kv_cache_groups) - len(self.group_ids)
        logger.info(
            "vllm-030win patch (step 060): KVMem viewport needs %d block(s) for "
            "the %d stored group(s); pool has %d block(s), %d other group(s)",
            needed,
            len(self.group_ids),
            available,
            other_groups,
        )
        if needed + 1 > available:
            raise RuntimeError(
                f"vllm-030win KVMem: the attention viewport needs {needed} "
                f"blocks (sliding window {[self.kv_cache_config.kv_cache_groups[g].kv_cache_spec.sliding_window for g in self.group_ids]}, "
                f"block size {[self.block_size[g] for g in self.group_ids]}) but "
                f"the KV pool only has {available}. Every request would stall in "
                f"the waiting queue. Lower VLLM_KVMEM_SW_WINDOW or raise "
                f"--kv-cache-memory-bytes."
            )

    @staticmethod
    def trajectory_key(prompt_token_ids, prefix_tokens: int) -> bytes:
        tokens = np.asarray(prompt_token_ids[:prefix_tokens], dtype=np.int64)
        return hashlib.blake2b(tokens.tobytes(), digest_size=16).digest()

    def update_state_after_alloc(self, request, blocks, num_external_tokens) -> None:
        del blocks, num_external_tokens
        if request.request_id in self._req_trajectory:
            return
        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = getattr(request, "all_token_ids", None)
        if not token_ids:
            return
        self._req_trajectory[request.request_id] = self.trajectory_key(
            token_ids, self.trajectory_prefix_tokens
        )
        self._req_baseline[request.request_id] = (
            self.pages_evicted,
            self.pages_stored,
            self.pages_dropped,
        )

    # ------------------------------------------------------------------
    # eviction hand-off
    # ------------------------------------------------------------------

    def register_workspace_retained_blocks(self, blocks: list) -> bool:
        """Accept ownership of pages the core manager held out of the pool."""
        if not self.group_ids:
            return False
        for block in blocks:
            self._retained_by_block_id[block.block_id] = block
        return True

    def _allocate_slot(self, group_id: int, trajectory: bytes, page_index: int) -> int | None:
        key = (trajectory, page_index * self.block_size[group_id])
        slot = self._page_table.get(key)
        if slot is not None:
            return slot
        if not self._free_slots:
            return None
        slot = self._free_slots.pop()
        self._page_table[key] = slot
        self.slots_used += 1
        return slot

    def build_connector_meta(self, scheduler_output) -> KVMemConnectorMetadata:
        meta = KVMemConnectorMetadata()
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        evictions = block_state.workspace_evictions if block_state else None
        if evictions:
            for req_id, entries in evictions.items():
                trajectory = self._req_trajectory.get(req_id)
                pages: list[KVMemPageTransfer] = []
                retained = []
                for group_id, block_id, page_index in entries:
                    self.pages_evicted += 1
                    block = self._retained_by_block_id.pop(block_id, None)
                    if block is None:
                        # Not ours (or already claimed): leave it to the pool.
                        logger.error(
                            "vllm-030win KVMem workspace: evicted block %d "
                            "(req=%s page=%d) was not registered as retained; "
                            "dropping it back to the pool",
                            block_id,
                            req_id,
                            page_index,
                        )
                        continue
                    retained.append(block)
                    if trajectory is None:
                        self.pages_dropped += 1
                        continue
                    slot = self._allocate_slot(group_id, trajectory, page_index)
                    if slot is None:
                        self.pages_dropped += 1
                        logger.warning(
                            "vllm-030win KVMem workspace is full: dropping "
                            "page %d of trajectory %s (group %d)",
                            page_index,
                            trajectory.hex()[:12],
                            group_id,
                        )
                        continue
                    pages.append(
                        KVMemPageTransfer(
                            group_id=group_id,
                            block_id=block_id,
                            page_index=page_index,
                            slot=slot,
                        )
                    )
                if not retained:
                    continue
                if not pages:
                    # Nothing to copy (full, or unknown trajectory): the pages
                    # go back to the pool now, i.e. the pre-KVMem behaviour.
                    if self.block_pool is not None:
                        self.block_pool.free_blocks(retained)
                    continue
                job_id = self._next_job_id
                self._next_job_id += 1
                self._jobs[job_id] = _JobStatus(
                    job_id=job_id,
                    trajectory=trajectory,
                    retained=retained,
                    pages=[(p.group_id, p.page_index) for p in pages],
                )
                meta.store_jobs.append(
                    KVMemStoreJob(
                        job_id=job_id, trajectory=trajectory, pages=pages
                    )
                )

        # Any block handed to us but not claimed by an eviction entry would
        # otherwise stay out of the pool forever.
        if self._retained_by_block_id:
            stray = list(self._retained_by_block_id.values())
            self._retained_by_block_id.clear()
            logger.error(
                "vllm-030win KVMem workspace: %d retained block(s) had no "
                "eviction entry; freeing them",
                len(stray),
            )
            if self.block_pool is not None:
                self.block_pool.free_blocks(stray)
        return meta

    # ------------------------------------------------------------------
    # completion
    # ------------------------------------------------------------------

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        if worker_meta is None:
            return
        for job_id in list(getattr(worker_meta, "completed_store_jobs", ())):
            status = self._jobs.pop(job_id, None)
            if status is None:
                continue
            self.pages_stored += len(status.pages)
            if self.block_pool is not None and status.retained:
                # Exactly one free per retained page: their ref count was never
                # decremented when the window evicted them.
                self.block_pool.free_blocks(status.retained)
            logger.info(
                "vllm-030win KVMem workspace: job %d stored %d entry(ies) of "
                "trajectory %s, released %d block(s); cumulative evicted=%d "
                "stored=%d dropped=%d slots=%d/%d",
                job_id,
                len(status.pages),
                status.trajectory.hex()[:12] if status.trajectory else "-",
                len(status.retained),
                self.pages_evicted,
                self.pages_stored,
                self.pages_dropped,
                self.slots_used,
                self.num_slots,
            )

    # ------------------------------------------------------------------
    # connector surface
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        del request, num_computed_tokens
        # Stage 1 K1 stores pages only; retrieval/rematerialisation is K3.
        return 0, False

    def request_finished(self, request, block_ids):
        del block_ids
        self._req_trajectory.pop(request.request_id, None)
        base = self._req_baseline.pop(request.request_id, None)
        if base is None:
            base = (0, 0, 0)
        # Independent expectation for the window-driven eviction volume: pages
        # fully outside the window at the final sequence length. It should match
        # the evicted count (per group), which is what makes the workspace
        # identity checkable from one log line.
        expected = []
        for group_id in self.group_ids:
            spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            outside = max(0, request.num_tokens - spec.sliding_window)
            expected.append(outside // self.block_size[group_id])
        logger.info(
            "vllm-030win KVMem workspace summary (req=%s): this request "
            "evicted=%d stored=%d dropped=%d (window expects %s per group); "
            "session evicted=%d stored=%d dropped=%d; logical slots=%d/%d, "
            "jobs in flight=%d",
            request.request_id,
            self.pages_evicted - base[0],
            self.pages_stored - base[1],
            self.pages_dropped - base[2],
            expected,
            self.pages_evicted,
            self.pages_stored,
            self.pages_dropped,
            self.slots_used,
            self.num_slots,
            len(self._jobs),
        )
        return False, None

    def has_pending_stores(self) -> bool:
        return bool(self._jobs)

    def stats(self) -> dict:
        return {
            "groups": self.group_ids,
            "num_slots": self.num_slots,
            "slots_used": self.slots_used,
            "pages_evicted": self.pages_evicted,
            "pages_stored": self.pages_stored,
            "pages_dropped": self.pages_dropped,
            "jobs_in_flight": len(self._jobs),
        }

    def reset(self) -> None:
        self._free_slots = list(range(self.num_slots - 1, -1, -1))
        self._page_table.clear()
        self._jobs.clear()
        self._retained_by_block_id.clear()
        self._req_trajectory.clear()
        self.slots_used = 0
