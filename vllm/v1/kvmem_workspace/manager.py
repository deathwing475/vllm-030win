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
    KVMemLoadJob,
    KVMemPageLoad,
    KVMemPageTransfer,
    KVMemScoreRequest,
    KVMemSnapshotRequest,
    KVMemStageInRequest,
    KVMemStepSpan,
    KVMemStoreJob,
)
from vllm.v1.kv_cache_interface import MambaSpec
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
        # request_id -> the live request object, for its token accounting.
        self._req_object: dict[str, object] = {}
        # request_id -> prompt length, which `request.num_tokens` stops being
        # once decode starts appending to it.
        self._req_prompt_len: dict[str, int] = {}
        self._req_scored: set[str] = set()
        # request_id -> (evicted, stored, dropped) when the request was first
        # seen, so the per-request summary is a delta and not a session total.
        self._req_baseline: dict[str, tuple[int, int, int]] = {}

        self.rawk = config.rawk_enabled()
        self.recent_tokens = config.recent_tokens()
        self.scores_emitted = 0

        # Step 066: prefix assembly. The mamba groups are not stored page-wise
        # (their blocks are recurrent state slots), but an assembled prefix is
        # only valid if the recurrent state at its boundary is restored too, so
        # the worker snapshots those slots at page-aligned boundaries.
        self.load_enabled = config.load_enabled() and bool(self.group_ids)
        # Step 072: the fixed-slot compressed window (design §5.1 re-bake).
        self.viewport_enabled = (
            config.viewport_enabled()
            and config.rawk_enabled()
            and config.authority_enabled()
            and bool(self.group_ids)
        )
        self.viewport_recent_tokens = config.viewport_recent_tokens()
        self.viewport_retrieval_pages = config.viewport_retrieval_pages()
        # Step 073: read-only observation of what the scheduler does to a
        # rewritten request (see config.debug_enabled). Off by default.
        self.debug = config.debug_enabled()
        # The window plus the request's own generation must fit the sliding
        # window the pool was sized for: the rewritten request is an ordinary
        # request as far as the engine is concerned, and it must never need
        # eviction (its pages are not the trajectory's pages at those
        # positions).
        self.max_window_tokens = min(
            (
                self.kv_cache_config.kv_cache_groups[g].kv_cache_spec.sliding_window
                for g in self.group_ids
            ),
            default=0,
        )
        self.sweep_enabled = config.sweep_enabled() and bool(self.group_ids)
        # request_id -> window plan for a rewritten request: the layout is
        # fixed before the request first runs (design §5.1's core invariant --
        # the query position never depends on which pages get selected).
        self._req_viewport: dict[str, dict] = {}
        # step 073 debug only: forward steps counted per window request.
        self._debug_window_steps: dict[str, int] = {}
        self.viewports_rewritten = 0
        self.viewports_declined = 0
        self.stage_ins_emitted = 0
        self.roll_pages_stored = 0
        self.roll_pages_skipped = 0
        self.snapshot_every_pages = config.snapshot_every_pages()
        self.mamba_group_ids = [
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        self.mamba_page_bytes = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.page_size_bytes
            for group_id in self.mamba_group_ids
        }
        # trajectory -> page-aligned boundaries with a completed snapshot.
        self._snapshots: dict[bytes, set[int]] = {}
        # (trajectory, boundary) pairs already handed to the worker.
        self._snapshot_sent: set[tuple[bytes, int]] = set()
        # (trajectory, page offset) -> page token hash, so an assembled prefix
        # is provably the *same tokens* this request would prefill (the
        # trajectory key only pins the leading VLLM_KVMEM_TRAJ_PREFIX tokens).
        self._page_hashes: dict[tuple[bytes, int], bytes] = {}
        # request_id -> (KVCacheBlocks, matched tokens) awaiting the load job.
        self._pending_loads: dict[str, tuple] = {}
        self.loads_requested = 0
        self.loads_completed = 0
        self.snapshots_requested = 0
        self.snapshots_completed = 0

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
        if request.request_id in self._req_viewport:
            # A rewritten (window) request falls through to the common
            # bookkeeping: its trajectory key is unchanged (the rewrite keeps
            # the leading tokens) and its prompt length is the window length
            # the score trigger needs.
            if request.request_id in self._req_trajectory:
                return
        elif request.request_id in self._req_trajectory:
            # A repeated call for an in-flight request (preemption replay):
            # only the assembly plan needs refreshing, and only for a fresh
            # external allocation.
            if num_external_tokens > 0 and request.request_id in self._pending_loads:
                self._pending_loads[request.request_id] = (
                    blocks,
                    num_external_tokens,
                    self._req_trajectory[request.request_id],
                )
            return
        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = getattr(request, "all_token_ids", None)
        if not token_ids:
            return
        self._req_trajectory[request.request_id] = self.trajectory_key(
            token_ids, self.trajectory_prefix_tokens
        )
        self._req_object[request.request_id] = request
        prompt_len = len(token_ids) if token_ids else None
        if prompt_len:
            self._req_prompt_len[request.request_id] = prompt_len
        self._req_baseline[request.request_id] = (
            self.pages_evicted,
            self.pages_stored,
            self.pages_dropped,
        )
        if num_external_tokens > 0 and self.load_enabled:
            self._pending_loads[request.request_id] = (
                blocks,
                num_external_tokens,
                self._req_trajectory[request.request_id],
            )
        if request.request_id in self._req_viewport:
            # The state the window request really starts from: how much of the
            # rewritten sequence the engine adopted as already computed.
            group_id = self.group_ids[0]
            block_ids = blocks.get_block_ids()
            rows = (
                len(block_ids[group_id])
                if block_ids and group_id < len(block_ids)
                else -1
            )
            self._debug_window(
                "adopted",
                request,
                external=num_external_tokens,
                rows=rows,
                window=self._req_viewport[request.request_id]["window"],
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

    def _emit_spans(
        self, meta: KVMemConnectorMetadata, scheduler_output, block_state
    ) -> None:
        """Tell the worker which token range of which trajectory this step is.

        Only needed once the raw-K capture is armed, or (step 066) once prefix
        assembly is: the snapshot capture also needs to know where each step
        ends so it can grab the boundary state while its block is still in the
        CoW window. Without both, the workspace is a write-only store and no
        position bookkeeping is required.
        """
        if not (self.rawk or self.load_enabled) or not self.group_ids:
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        group_id = self.group_ids[0]
        block_size = self.block_size[group_id]
        for req_id, num_tokens in scheduled.items():
            trajectory = self._req_trajectory.get(req_id)
            request = self._req_object.get(req_id)
            if trajectory is None or request is None or num_tokens <= 0:
                continue
            start = request.num_computed_tokens
            plan = self._req_viewport.get(req_id)
            if plan is not None and self.debug:
                step = self._debug_window_steps.get(req_id, 0) + 1
                self._debug_window_steps[req_id] = step
                self._debug_window(
                    f"forward{step}",
                    request,
                    num=num_tokens,
                    end=start + num_tokens,
                    window=plan["window"],
                )
            is_prefill = True
            prompt_now = len(getattr(request, "prompt_token_ids", ()) or ())
            if prompt_now:
                is_prefill = start < prompt_now
            meta.spans.append(
                KVMemStepSpan(
                    trajectory=trajectory,
                    start=start,
                    num_tokens=num_tokens,
                    viewport=plan is not None,
                    prefill=is_prefill,
                )
            )
            # The step that completes the prompt is the one where retrieval
            # would run: the layout is fixed from there on and decode only
            # appends to the generation reserve. `request.num_tokens` grows
            # during decode, so the fixed prompt length is what decides.
            prompt_len = self._req_prompt_len.get(req_id)
            if (
                prompt_len is not None
                and req_id not in self._req_scored
                and start < prompt_len
                and start + num_tokens >= prompt_len
            ):
                self._req_scored.add(req_id)
                if plan is not None:
                    # A window request scores against its *original* prompt
                    # length: the stored pages are keyed by the original
                    # trajectory's page indices, so the eligible range
                    # (everything strictly between the original sink and the
                    # original recent tail) is what retrieval may pick from.
                    meta.score_requests.append(
                        KVMemScoreRequest(
                            trajectory=trajectory,
                            request_id=req_id,
                            num_tokens=plan["orig_len"],
                            block_size=block_size,
                            sink_tokens=plan["sink"],
                            recent_tokens=plan["recent"],
                        )
                    )
                    self._emit_stage_request(
                        meta, req_id, plan, block_state, request
                    )
                else:
                    meta.score_requests.append(
                        KVMemScoreRequest(
                            trajectory=trajectory,
                            request_id=req_id,
                            num_tokens=prompt_len,
                            block_size=block_size,
                            sink_tokens=block_size,
                            recent_tokens=self.recent_tokens,
                        )
                    )
                self.scores_emitted += 1

    def _emit_stage_request(
        self,
        meta: KVMemConnectorMetadata,
        req_id: str,
        plan: dict,
        block_state,
        request,
    ) -> None:
        """Hand the worker everything the bake needs except the page choice.

        The scoring report picks the pages; this carries the fixed side: the
        retrieval slots' physical blocks (rows S//block_size .. of the request's
        block table), the slot start, and the trajectory's page table so the
        worker can find the V / non-rotary bytes of whichever pages won.
        """
        # The scheduler-local snapshot is the authoritative, up-to-date block
        # table; an admission-time copy would miss the blocks chunked prefill
        # allocated after the request was admitted.
        tables = block_state.block_ids.get(req_id) if block_state else None
        if not tables:
            logger.error(
                "vllm-030win KVMem viewport (req=%s): no block table in this "
                "step's scheduler output; the retrieval slots cannot be baked",
                req_id,
            )
            return
        group_id = self.group_ids[0]
        block_size = self.block_size[group_id]
        first_row = plan["sink"] // block_size
        num_rows = self.viewport_retrieval_pages
        slot_blocks: list[list[tuple[int, int]]] = []
        # Every stored group owns a physical block for the same logical row; a
        # slot is only offered once *all* of them are allocated, otherwise the
        # bake would half-fill it (step 074: reading only group_ids[0] left the
        # second group's layers on the placeholder prefill).
        short_by_group: dict[int, int] = {}
        for j in range(num_rows):
            row = first_row + j
            entries: list[tuple[int, int]] = []
            for gid in self.group_ids:
                rows = tables[gid] if gid < len(tables) else ()
                if self.block_size[gid] != block_size:
                    logger.error(
                        "vllm-030win KVMem viewport (req=%s): stored group %d "
                        "has block size %d, group %d has %d; the workspace page "
                        "table is keyed in one page size, so retrieval slots "
                        "cannot be baked across these groups",
                        req_id,
                        gid,
                        self.block_size[gid],
                        group_id,
                        block_size,
                    )
                    return
                if row >= len(rows) or not rows[row]:
                    short_by_group[gid] = short_by_group.get(gid, 0) + 1
                    entries = []
                    break
                entries.append((gid, rows[row]))
            if not entries:
                logger.error(
                    "vllm-030win KVMem viewport (req=%s): retrieval slot row "
                    "%d (logical page %d) has no real block in group(s) %s; "
                    "slots beyond it stay placeholder",
                    req_id,
                    row,
                    row,
                    sorted(short_by_group) or self.group_ids,
                )
                break
            slot_blocks.append(entries)
        if not slot_blocks:
            return
        trajectory = plan["trajectory"]
        page_table = {
            # Keyed by page index; the token offset form of the table is
            # (trajectory, offset) with offset = page_index * block_size.
            offset // block_size: slot
            for (traj, offset), slot in self._page_table.items()
            if traj == trajectory
        }
        # Coverage is what step 073 could not read off the old line: "N slot
        # block(s)" counted rows, not groups, so a half-covered bake looked
        # complete. This states slots x groups and the distinct blocks.
        covered = {gid for entries in slot_blocks for gid, _ in entries}
        logger.info(
            "vllm-030win KVMem viewport (req=%s): stage-in plan: %d slot(s) x "
            "%d group(s) = %d block(s), from row %d, stored group(s) %s "
            "(covered %s), page table carries %d page(s) of this trajectory "
            "(%d in store overall)",
            req_id,
            len(slot_blocks),
            len(self.group_ids),
            sum(len(entries) for entries in slot_blocks),
            first_row,
            self.group_ids,
            sorted(covered),
            len(page_table),
            len(self._page_table),
        )
        self._debug_window(
            "stage",
            request=request,
            slots=len(slot_blocks),
            groups=self.group_ids,
            slot0=slot_blocks[0],
            slotN=slot_blocks[-1],
        )
        meta.stage_requests.append(
            KVMemStageInRequest(
                trajectory=trajectory,
                request_id=req_id,
                slots=slot_blocks,
                slot_start=plan["sink"],
                page_size=block_size,
                pages=page_table,
            )
        )
        self.stage_ins_emitted += 1

    def _emit_incremental_stores(self, meta, scheduler_output, block_state) -> None:
        """Store each page as soon as its prefill completes (step 072).

        K1 stores pages only when the sliding window evicts them, so a 200K
        ingest used to finish with just the ~25 out-of-window pages on the
        host while the ~115 mid-section pages -- the ones the retrieval slots
        need V / non-rotary bytes from -- were freed with the request. This
        emits a store job for every page some step *completes* (its last token
        is inside this step's schedule), so the copy rides the same step's
        forward: the page's KV is freshly written, the request still owns the
        block, and nothing is held out of the pool -- the two boot-1/boot-2
        failure modes (a finish-time sweep pinning blocks until no request can
        be admitted, hence no forward step hence no copy) cannot happen.

        Blocks come from the scheduler-local block-table snapshot
        (``kv_connector_block_state.block_ids``), which is authoritative and
        per-step; page-token hashes are recorded here too, so an assembled
        prefix stays provably the same tokens (the step 066 property).
        """
        if not (self.sweep_enabled and self.group_ids):
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        if not scheduled:
            return
        group_id = self.group_ids[0]
        block_size = self.block_size[group_id]
        for req_id, num_tokens in scheduled.items():
            if num_tokens <= 0 or req_id in self._req_viewport:
                continue
            trajectory = self._req_trajectory.get(req_id)
            if trajectory is None:
                continue
            request = self._req_object.get(req_id)
            if request is None:
                continue
            token_ids = getattr(request, "prompt_token_ids", None)
            if not token_ids:
                continue
            prompt_len = len(token_ids)
            start = request.num_computed_tokens
            end = start + num_tokens
            tables = block_state.block_ids.get(req_id)
            if not tables:
                continue
            first_page = start // block_size
            last_page = end // block_size  # pages strictly below this are done
            for page_index in range(first_page, last_page):
                offset = page_index * block_size
                if offset + block_size > prompt_len:
                    # The page holds generation-tail or padding tokens; it is
                    # not (entirely) trajectory content.
                    continue
                if (trajectory, offset) in self._page_table:
                    self.roll_pages_skipped += 1
                    continue
                # _allocate_slot keys on the page index, not the token offset.
                slot = self._allocate_slot(group_id, trajectory, page_index)
                if slot is None:
                    self.pages_dropped += 1
                    logger.warning(
                        "vllm-030win KVMem incremental store is full: page %d "
                        "of trajectory %s not stored",
                        page_index,
                        trajectory.hex()[:12],
                    )
                    continue
                self._page_hashes.setdefault(
                    (trajectory, offset),
                    self._page_token_hash(token_ids, offset, block_size),
                )
                transfers = []
                for gid in self.group_ids:
                    rows = tables[gid] if gid < len(tables) else ()
                    # 0 is the shared null placeholder in the block ids.
                    if page_index >= len(rows) or not rows[page_index]:
                        logger.error(
                            "vllm-030win KVMem incremental store: request %s "
                            "group %d has no block row for page %d; skipping",
                            req_id,
                            gid,
                            page_index,
                        )
                        continue
                    transfers.append(
                        KVMemPageTransfer(
                            group_id=gid,
                            block_id=rows[page_index],
                            page_index=page_index,
                            slot=slot,
                        )
                    )
                if not transfers:
                    continue
                self.roll_pages_stored += len(transfers)
                meta.store_jobs.append(
                    KVMemStoreJob(
                        job_id=self._next_job_id,
                        trajectory=trajectory,
                        pages=transfers,
                    )
                )
                self._jobs[self._next_job_id] = _JobStatus(
                    job_id=self._next_job_id,
                    trajectory=trajectory,
                    retained=[],
                    pages=[(p.group_id, p.page_index) for p in transfers],
                )
                self._next_job_id += 1

    @staticmethod
    def _page_token_hash(token_ids, start: int, size: int) -> bytes:
        tokens = np.asarray(token_ids[start:start + size], dtype=np.int64)
        return hashlib.blake2b(tokens.tobytes(), digest_size=8).digest()

    def _emit_snapshots(
        self, meta: KVMemConnectorMetadata, scheduler_output
    ) -> None:
        """Request the mamba state at a page boundary this step just crossed.

        The sliding window keeps the workspace's contiguous page prefix ~W
        tokens behind the live sequence, so by the time page k is evicted the
        recurrent state at boundary (k+1)*block_size is long gone from its
        slot (the align-mode CoW window is two blocks). The only moment that
        state is capturable is right after the step that completes the page:
        the running slot then holds the exact state after ``end`` tokens --
        the engine's own invariant ("slot p holds the state after exactly
        (p + 1) * block_size tokens; state is written at chunk ends, so chunk
        ends must be block aligned", scheduler._mamba_block_aligned_split).

        Snapshots are taken every ``VLLM_KVMEM_SNAPSHOT_EVERY_PAGES`` pages:
        each row is 80.4 MiB of host and the ring must outlive a boundary for
        W/chunk_tokens steps before the page prefix reaches it, so denser
        snapshots would need a proportionally larger ring for no assembly
        gain -- a sparser ring just caps how much of the page run is
        assemblable (the boundary falls back to the newest sparse one).
        """
        if not self.load_enabled or not self.mamba_group_ids:
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        # Authoritative per-request block table (per group list of block ids),
        # filled by the scheduler right before build_connector_meta; a block
        # id of 0 is the shared null placeholder.
        req_blocks_map = block_state.block_ids if block_state is not None else None
        if not req_blocks_map:
            return
        group_id = self.group_ids[0]
        block_size = self.block_size[group_id]
        for req_id, num_tokens in scheduled.items():
            trajectory = self._req_trajectory.get(req_id)
            request = self._req_object.get(req_id)
            group_tables = req_blocks_map.get(req_id)
            if (
                trajectory is None
                or request is None
                or num_tokens <= 0
                or group_tables is None
            ):
                continue
            start = request.num_computed_tokens
            end = start + num_tokens
            if end % block_size or end <= start:
                # The mamba slot holds the state as of the *step end* (the
                # chunked forward writes its recurrence straight into the
                # running slot), so only a step that ends exactly on a page
                # boundary yields the exact boundary state. This requires
                # --max-num-batched-tokens to be a multiple of the page size;
                # a step ending mid-page would snapshot a state that is ahead
                # of the boundary by up to a chunk, and the assembled request
                # would resume its recurrence from the wrong state.
                continue
            if (end // block_size) % max(1, self.snapshot_every_pages):
                continue  # sparse ring: only every Nth page boundary
            boundary = end
            if (trajectory, boundary) in self._snapshot_sent:
                continue
            position = end // block_size - 1
            blocks: list[tuple[int, int]] = []
            null_groups = 0
            for mamba_group in self.mamba_group_ids:
                if mamba_group >= len(group_tables):
                    null_groups += 1
                    continue
                group_blocks = group_tables[mamba_group]
                if position >= len(group_blocks) or group_blocks[position] == 0:
                    # Should not happen (the block is one behind the running
                    # slot), but a null slot must never be snapshotted: the
                    # copy would read recycled memory and silently poison the
                    # workspace. A snapshot row is only usable if EVERY mamba
                    # group's state landed (the load fills one row per group),
                    # so any null group voids the whole boundary.
                    null_groups += 1
                    continue
                blocks.append((mamba_group, group_blocks[position]))
            self._snapshot_sent.add((trajectory, boundary))
            if null_groups:
                logger.warning(
                    "vllm-030win KVMem snapshot (req=%s boundary=%d): %d mamba "
                    "group(s) had a null slot at the boundary position; the "
                    "boundary is not capturable",
                    req_id,
                    boundary,
                    null_groups,
                )
                continue
            self.snapshots_requested += 1
            meta.snapshot_requests.append(
                KVMemSnapshotRequest(
                    trajectory=trajectory, boundary=boundary, blocks=blocks
                )
            )

    def build_connector_meta(self, scheduler_output) -> KVMemConnectorMetadata:
        meta = KVMemConnectorMetadata()
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        self._emit_spans(meta, scheduler_output, block_state)
        self._emit_snapshots(meta, scheduler_output)
        self._emit_load_jobs(meta)
        if block_state is not None:
            self._emit_incremental_stores(meta, scheduler_output, block_state)
        evictions = block_state.workspace_evictions if block_state else None
        if evictions:
            for req_id, entries in evictions.items():
                trajectory = self._req_trajectory.get(req_id)
                request = self._req_object.get(req_id)
                # Step 066: pin each stored page's tokens. An assembled prefix
                # must be the *same tokens* the assembling request carries; the
                # trajectory key alone only pins the leading prefix tokens.
                if self.load_enabled and trajectory and request is not None:
                    token_ids = getattr(request, "prompt_token_ids", None)
                    if token_ids:
                        prompt_len = len(token_ids)
                        page_size = self.block_size[self.group_ids[0]]
                        for _, _, page_index in entries:
                            start = page_index * page_size
                            if start + page_size <= prompt_len:
                                self._page_hashes.setdefault(
                                    (trajectory, start),
                                    self._page_token_hash(
                                        token_ids, start, page_size
                                    ),
                                )
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
        # Step 066: snapshots and loads are DMA-verified before they count, so
        # the assembly match can only ever see boundaries that really landed.
        for trajectory, boundary in list(
            getattr(worker_meta, "completed_snapshots", ())
        ):
            self._snapshots.setdefault(trajectory, set()).add(boundary)
            self.snapshots_completed += 1
            logger.info(
                "vllm-030win KVMem snapshot: boundary %d of trajectory %s "
                "captured (%d snapshot(s) available)",
                boundary,
                trajectory.hex()[:12],
                len(self._snapshots[trajectory]),
            )
        for trajectory, boundary in list(
            getattr(worker_meta, "removed_snapshots", ())
        ):
            boundaries = self._snapshots.get(trajectory)
            if boundaries is not None:
                boundaries.discard(boundary)
        for req_id in list(getattr(worker_meta, "finished_load_reqs", ())):
            self.loads_completed += 1
            logger.info(
                "vllm-030win KVMem assembly: request %s prefix landed; "
                "cumulative requested=%d completed=%d",
                req_id,
                self.loads_requested,
                self.loads_completed,
            )

    # ------------------------------------------------------------------
    # connector surface
    # ------------------------------------------------------------------

    def _emit_load_jobs(self, meta: KVMemConnectorMetadata) -> None:
        """Turn the assembly plan of this step's admissions into load jobs.

        Runs in the same schedule() step as ``update_state_after_alloc``, so
        the ``KVCacheBlocks`` handed over there still describe this request's
        block table: attention rows 0..E-1 are the freshly allocated blocks the
        workspace pages go into, and the mamba rows carry exactly one real
        block (position E-1, from the MambaManager external-allocation patch)
        that receives the boundary snapshot.
        """
        if not self._pending_loads:
            return
        for req_id, (blocks, matched, trajectory) in list(
            self._pending_loads.items()
        ):
            self._pending_loads.pop(req_id, None)
            group_id = self.group_ids[0]
            block_size = self.block_size[group_id]
            num_pages = matched // block_size
            pages: list[KVMemPageLoad] = []
            missing_slot = False
            for gid in self.group_ids:
                group_blocks = blocks.blocks[gid]
                for page_index in range(num_pages):
                    if page_index >= len(group_blocks) or group_blocks[
                        page_index
                    ].is_null:
                        logger.error(
                            "vllm-030win KVMem assembly (req=%s): group %d has "
                            "no real block at page %d; skipping the load",
                            req_id,
                            gid,
                            page_index,
                        )
                        missing_slot = True
                        break
                    slot = self._page_table.get((trajectory, page_index * block_size))
                    if slot is None:
                        logger.error(
                            "vllm-030win KVMem assembly (req=%s): page %d of "
                            "trajectory %s vanished from the page table; "
                            "skipping the load",
                            req_id,
                            page_index,
                            trajectory.hex()[:12],
                        )
                        missing_slot = True
                        break
                    pages.append(
                        KVMemPageLoad(
                            group_id=gid,
                            block_id=group_blocks[page_index].block_id,
                            page_index=page_index,
                            slot=slot,
                        )
                    )
                if missing_slot:
                    break
            if missing_slot:
                continue
            snapshots: list[tuple[int, int, int]] = []
            for mamba_group in self.mamba_group_ids:
                group_blocks = blocks.blocks[mamba_group]
                position = num_pages - 1
                if position >= len(group_blocks) or group_blocks[position].is_null:
                    logger.error(
                        "vllm-030win KVMem assembly (req=%s): mamba group %d "
                        "has no real state block at position %d; skipping the "
                        "load (the assembled prefix would resume from a zeroed "
                        "recurrent state)",
                        req_id,
                        mamba_group,
                        position,
                    )
                    snapshots = []
                    break
                snapshots.append(
                    (mamba_group, group_blocks[position].block_id, matched)
                )
            if not snapshots:
                continue
            self.loads_requested += 1
            meta.load_jobs.append(
                KVMemLoadJob(
                    job_id=self._next_job_id,
                    req_id=req_id,
                    trajectory=trajectory,
                    num_tokens=matched,
                    pages=pages,
                    mamba_snapshots=snapshots,
                )
            )
            self._next_job_id += 1

    def _assembly_match(self, request, num_computed_tokens: int) -> int:
        """Token boundary this request can have assembled, or 0.

        The boundary must simultaneously be (a) the end of a contiguous run of
        stored pages, (b) a boundary whose mamba snapshot has landed, and (c)
        built from pages whose tokens are provably identical to this request's
        prompt at the same offsets.
        """
        if num_computed_tokens > 0:
            # Assembly must own the whole prefix; a local prefix-cache hit
            # would interleave blocks this connector does not manage.
            return 0
        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = getattr(request, "all_token_ids", None)
        if not token_ids:
            return 0
        trajectory = self.trajectory_key(token_ids, self.trajectory_prefix_tokens)
        group_id = self.group_ids[0]
        block_size = self.block_size[group_id]
        prompt_len = len(token_ids)
        num_pages = 0
        while (
            num_pages * block_size + block_size <= prompt_len
            and (trajectory, num_pages * block_size) in self._page_table
        ):
            num_pages += 1
        if num_pages == 0:
            return 0
        page_boundary = num_pages * block_size
        available = self._snapshots.get(trajectory)
        if not available:
            return 0
        # The recurrent state must be exact at the boundary we jump to, so the
        # assembly boundary is a *snapshot* boundary, capped by the page run.
        candidates = [b for b in available if b <= page_boundary]
        if not candidates:
            return 0
        boundary = max(candidates)
        if boundary < block_size:
            return 0
        pages = boundary // block_size
        for page_index in range(pages):
            key = (trajectory, page_index * block_size)
            recorded = self._page_hashes.get(key)
            if recorded is None or recorded != self._page_token_hash(
                token_ids, page_index * block_size, block_size
            ):
                if boundary > page_index * block_size:
                    logger.info(
                        "vllm-030win KVMem assembly: trajectory %s page %d "
                        "token hash mismatch (prompt diverged from the stored "
                        "pages); capping the boundary at %d tokens",
                        trajectory.hex()[:12],
                        page_index,
                        page_index * block_size,
                    )
                boundary = min(boundary, page_index * block_size)
                break
        if boundary < block_size or boundary >= prompt_len:
            # Never claim the whole prompt: the scheduler clamps a full hit
            # back to num_tokens - 1, which is not page aligned.
            return 0
        return boundary

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # Step 072: the compressed-window route. Only the first scheduling
        # look at a request reaches this with the *original* prompt; the
        # rewrite below changes the prompt and defers the request one step, so
        # the scheduler re-runs its prefix lookup on the rewritten sequence.
        if request.request_id in self._req_viewport:
            return 0, False
        if self.viewport_enabled and self._viewport_rewrite(
            request, num_computed_tokens
        ):
            # Returning None parks the request back on the waiting queue for
            # this step (scheduler.py: the request "cannot be scheduled
            # because the KVConnector couldn't determine the number of matched
            # tokens"); the next pass sees the rewritten prompt.
            return None, False
        if not self.load_enabled:
            # Stage 1 K1 stores pages only; retrieval/rematerialisation is K3.
            return 0, False
        boundary = self._assembly_match(request, num_computed_tokens)
        if not boundary:
            return 0, False
        logger.info(
            "vllm-030win KVMem assembly: request %s matches at %d tokens "
            "(%d stored page(s)); async load",
            request.request_id,
            boundary,
            boundary // self.block_size[self.group_ids[0]],
        )
        return boundary, True

    # ------------------------------------------------------------------
    # step 072: the fixed-slot compressed window
    # ------------------------------------------------------------------

    def _debug_window(self, where: str, request, **fields) -> None:
        """One compact observation line (VLLM_KVMEM_DEBUG, step 073).

        Records what the engine really did to a window request, which is the
        only way to tell "the model sampled EOS" from "the request ended before
        a sample" and "a sample happened but was dropped downstream".
        """
        if not self.debug:
            return
        hashes = getattr(request, "block_hashes", None)
        outputs = getattr(request, "_output_token_ids", ()) or ()
        extra = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(
            "vllm-030win KVMem window-debug (%s): req=%s computed=%d "
            "num_tokens=%d prompt_len=%d hashes=%d outputs=%d first_ids=%s "
            "status=%s %s",
            where,
            request.request_id,
            request.num_computed_tokens,
            request.num_tokens,
            request.num_prompt_tokens,
            len(hashes) if hashes is not None else -1,
            len(outputs),
            list(outputs[:3]),
            getattr(request.status, "name", request.status),
            extra,
        )

    def _decline_window(self, request, num_computed_tokens: int, reason: str):
        """Count a window decline, and (under debug) say which guard fired."""
        self.viewports_declined += 1
        if self.debug:
            logger.info(
                "vllm-030win KVMem window-debug (decline:%s): req=%s "
                "computed=%d prompt_len=%d hashes=%d",
                reason,
                request.request_id,
                num_computed_tokens,
                len(getattr(request, "prompt_token_ids", ()) or ()),
                len(getattr(request, "block_hashes", ()) or ()),
            )
        return False

    def _viewport_layout(self, prompt_len: int) -> tuple[int, int, int, int] | None:
        """(S, N tokens, R, B) for a prompt of this length, or None.

        S is one page (the sink), N is the fixed retrieval budget, R the
        recent tail. The layout is a pure function of the configuration and
        the prompt length -- never of what retrieval later selects, which is
        what keeps every window position independent of the scoring (design
        §5.1's key invariant).
        """
        block_size = self.block_size[self.group_ids[0]]
        sink = block_size
        retrieval = self.viewport_retrieval_pages * block_size
        recent = self.viewport_recent_tokens
        window = sink + retrieval + recent
        if prompt_len <= window:
            # No mid-section to re-represent: the window would be a pure
            # truncation, and identity (native prefill) is both simpler and
            # exactly correct.
            return None
        return sink, retrieval, recent, window

    def _trajectory_page_count(self, trajectory: bytes) -> int:
        return sum(1 for traj, _ in self._page_table if traj == trajectory)

    def _viewport_rewrite(self, request, num_computed_tokens: int) -> bool:
        """Rewrite the request's prompt onto the compressed window, or not.

        The rewrite is the whole trick: the window's prefill token sequence is

            prompt[:S+N] + prompt[L-R:]

        i.e. the head (sink + the placeholder section that the retrieval slots
        will overwrite) followed by the prompt's own tail. Three properties
        fall out of that shape:

        * the sink and placeholder sections keep their *original* positions,
          so their prefilled KV is correct as-is and a native prefix-cache hit
          inside them is genuinely theirs;
        * the recent tail is prefilled at window positions, which is what the
          attention needs -- no re-bake;
        * the GDN recurrence runs over real history tokens (head + tail), so
          no recurrent state needs restoring (hard constraint §3.2 is bypassed,
          exactly as §5.1 promised).

        Only the retrieval slots need baking afterwards.

        Guard on the local prefix hit: a hit beyond the sink section means the
        first request's blocks are still alive, i.e. the native prefix cache is
        already rescuing the prompt and the window has nothing to add -- and a
        hit reaching into the recent tail would adopt original-position blocks
        that the window must not use. Both are declined: the request runs
        natively. When the hit is within the sink, the block-hash chain breaks
        right after it (the rewritten tail cannot hash to the original blocks),
        so an adopted block can never reach the retrieval slots or the recent
        tail.
        """
        if num_computed_tokens > self.block_size[self.group_ids[0]]:
            return self._decline_window(
                request, num_computed_tokens, "hit-above-sink"
            )
        token_ids = getattr(request, "prompt_token_ids", None)
        if not token_ids:
            return self._decline_window(request, num_computed_tokens, "no-tokens")
        prompt_len = len(token_ids)
        layout = self._viewport_layout(prompt_len)
        if layout is None:
            return self._decline_window(request, num_computed_tokens, "no-mid")
        trajectory = self.trajectory_key(token_ids, self.trajectory_prefix_tokens)
        if self._trajectory_page_count(trajectory) == 0:
            # Nothing stored for this trajectory: the slots would stay
            # placeholders, which is a pure truncation of the prompt.
            return self._decline_window(request, num_computed_tokens, "no-store")
        if len({self.block_size[g] for g in self.group_ids}) != 1:
            # The page table, the slot rows and the bake are all keyed in one
            # page size; a mixed set would silently half-bake the slots.
            return self._decline_window(
                request, num_computed_tokens, "mixed-page-size"
            )
        sink, retrieval, recent, window = layout
        new_len = sink + retrieval + recent
        if new_len + request.max_tokens > self.max_window_tokens:
            return self._decline_window(request, num_computed_tokens, "too-wide")
        # The rewrite keeps the leading 512 tokens verbatim, so the trajectory
        # key computed by update_state_after_alloc stays the same key.
        token_ids[:] = token_ids[:sink + retrieval] + token_ids[prompt_len - recent:]
        request._all_token_ids[:] = token_ids
        request.num_prompt_tokens = len(token_ids)
        # The block hashes the engine computed describe the *original* prompt, so
        # they are stale for everything past the sink page: left in place, a
        # later prefix match could adopt an original-position block for a window
        # position holding different tokens (step 073 measured hashes=139 against
        # a 68-block rewritten prompt). Recompute the chain for the rewritten
        # sequence; the leading sink blocks hash to the same value either way,
        # because hashing is chained from token 0 over `all_token_ids`.
        if getattr(request, "block_hashes", None):
            request.block_hashes = []
            request.update_block_hashes()
        self._req_viewport[request.request_id] = {
            "trajectory": trajectory,
            "sink": sink,
            "retrieval": retrieval,
            "recent": recent,
            "window": new_len,
            "orig_len": prompt_len,
        }
        self.viewports_rewritten += 1
        logger.info(
            "vllm-030win KVMem viewport: request %s rewritten onto the "
            "compressed window: %d -> %d token(s) (sink %d + %d retrieval "
            "page(s) + recent %d; orig prompt %d, stored page(s) %d); "
            "deferring one scheduling pass",
            request.request_id,
            prompt_len,
            len(token_ids),
            sink,
            self.viewport_retrieval_pages,
            recent,
            prompt_len,
            self._trajectory_page_count(trajectory),
        )
        digest = hashlib.blake2b(
            np.asarray(token_ids, dtype=np.int64).tobytes(), digest_size=8
        ).hexdigest()
        self._debug_window(
            "rewrite",
            request,
            orig=prompt_len,
            window=new_len,
            max_tokens=request.max_tokens,
            sha=digest,
        )
        return True

    def request_finished(self, request, block_ids):
        del block_ids
        plan = self._req_viewport.get(request.request_id)
        if plan is not None:
            # How the window request really left: sampled ids, status, and the
            # computed total. Empty outputs with a computed total short of the
            # window is a scheduler fault, not a model one.
            self._debug_window(
                "finish",
                request,
                window=plan["window"],
                orig=plan["orig_len"],
                scored=request.request_id in self._req_scored,
            )
        self._req_trajectory.pop(request.request_id, None)
        self._req_object.pop(request.request_id, None)
        self._req_prompt_len.pop(request.request_id, None)
        self._req_scored.discard(request.request_id)
        self._pending_loads.pop(request.request_id, None)
        self._req_viewport.pop(request.request_id, None)
        self._debug_window_steps.pop(request.request_id, None)
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
            "rawk": self.rawk,
            "scores_emitted": self.scores_emitted,
            "viewports_rewritten": self.viewports_rewritten,
            "viewports_declined": self.viewports_declined,
            "stage_ins_emitted": self.stage_ins_emitted,
            "roll_pages_stored": self.roll_pages_stored,
        }

    def reset(self) -> None:
        self._free_slots = list(range(self.num_slots - 1, -1, -1))
        self._page_table.clear()
        self._jobs.clear()
        self._retained_by_block_id.clear()
        self._req_trajectory.clear()
        self._req_object.clear()
        self._req_prompt_len.clear()
        self._req_scored.clear()
        self._snapshots.clear()
        self._snapshot_sent.clear()
        self._page_hashes.clear()
        self._pending_loads.clear()
        self._req_viewport.clear()
        self._debug_window_steps.clear()
        self.slots_used = 0
