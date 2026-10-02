# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem workspace, worker side.

Holds the pinned host store and performs the device->host page copies. One
workspace *slot* is one page of every layer of every stored group, so a slot
index addresses a row in each host layer tensor.

Copies go through ``ops.swap_blocks_batch``, which on this platform routes to
the ``cuMemcpyAsync`` loop that the Windows port already needs
(``_WIN_BATCH_MEMCPY_BROKEN``). They are issued on the current stream — i.e.
after the model forward that wrote the page — and their completion is reported
back to the scheduler through a CUDA event, which is what keeps the source page
out of the block pool until the bytes are safely on the host.
"""

import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import MambaSpec, group_kernel_blocks
from vllm.v1.kvmem_workspace import capture, config, remat
from vllm.v1.kvmem_workspace.groups import workspace_group_ids
from vllm.v1.kvmem_workspace.index import KVMemMeanKIndex
from vllm.v1.kvmem_workspace.metadata import (
    KVMemConnectorMetadata,
    KVMemWorkerMetadata,
)

logger = init_logger(__name__)

SELFTEST_PAGES = 8
REMAT_SELFTEST_PAGES = 4


def _alloc_host(num_slots: int, page: int) -> torch.Tensor:
    """Allocate one host layer region.

    Pinned memory is what makes the page copy a real DMA, but it is a bounded
    OS resource; the production offload region gets its 8 GiB through
    ``cudaHostRegister`` on an mmap, which is not the same pool as
    ``cudaMallocHost``. Fall back to pageable memory rather than failing the
    boot, and say so loudly — the copies stay correct, just slower.
    """
    try:
        return torch.zeros(
            (num_slots, page), dtype=torch.int8, device="cpu", pin_memory=True
        )
    except RuntimeError as exc:
        logger.warning(
            "vllm-030win patch (step 060): pinned host allocation of %d x %d B "
            "failed (%s); falling back to pageable memory (copies still "
            "correct, just slower)",
            num_slots,
            page,
            exc,
        )
        return torch.zeros(
            (num_slots, page), dtype=torch.int8, device="cpu", pin_memory=False
        )


class KVMemWorkspaceWorker:
    def __init__(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.group_ids = workspace_group_ids(kv_cache_config)
        self.page_bytes = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.page_size_bytes
            for group_id in self.group_ids
        }

        per_slot_bytes = sum(
            self.page_bytes[group_id]
            * len(kv_cache_config.kv_cache_groups[group_id].layer_names)
            for group_id in self.group_ids
        )
        budget = config.workspace_host_bytes()
        self.num_slots = max(1, budget // per_slot_bytes) if per_slot_bytes else 0

        # (group_id, layer_name) -> (num_blocks, page_bytes) int8 view
        self._gpu_views: dict[tuple[int, str], torch.Tensor] = {}
        # (group_id, layer_name) -> (num_slots, page_bytes) pinned int8
        self._host: dict[tuple[int, str], torch.Tensor] = {}
        self._layers_per_group: dict[int, list[str]] = {}
        self._group_of_layer: dict[str, int] = {}
        self._layer_name_by_index: dict[int, str] = {}
        self._block_size: dict[int, int] = {}
        self._geometry: dict[int, remat.PageGeometry] = {}
        # scratch buffers for the round-trip self test
        self._scratch_gpu: dict[int, torch.Tensor] = {}
        self._scratch_host: dict[int, torch.Tensor] = {}
        self._scratch_host2: dict[int, torch.Tensor] = {}

        self._pending: KVMemConnectorMetadata | None = None
        self._loads_issued = False
        self._events: dict[int, torch.Event] = {}
        self._completed: list[int] = []
        self._selftest_done = 0
        self._selftest_pages = 0
        self._selftest_mismatch = 0
        self._selftest_max_byte_diff = 0
        self.bytes_stored = 0
        self.store_seconds = 0.0

        # vllm-030win step 079 timing instrumentation (env-gated).
        # ``storing`` is the store window as it already stands; ``copy``
        # and ``selftest`` are cut out of it so the entry-building Python
        # (storing - copy - selftest) and the selftest's device sync are
        # visible separately, and ``load`` splits the assembly side out
        # of store_seconds (078: both directions share that accumulator).
        self._kvtime: dict[str, float] = {}
        self._kvtime_n: dict[str, int] = {}
        self._kvtime_prev: dict[str, float] = {}
        self._kvtime_prev_n: dict[str, int] = {}
        self._kvtime_wall = 0.0
        self._kvtime_enabled = config.timing_enabled()
        self._kvtime_every = max(1, config.timing_every())

        # K3: retrieval index (host-resident) and the trailing query span of
        # the most recent prefill step of each trajectory.
        self._index: KVMemMeanKIndex | None = None
        self._last_query: dict[bytes, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
        self._retrieval_reports: list[dict] = []
        self._score_seconds = 0.0

        # Step 072: stage-in (bake the scored pages into the retrieval slots).
        self.stage_ins_served = 0
        self.stage_in_pages = 0
        self.stage_in_seconds = 0.0

        # K3 second half (step 064): the pre-RoPE rotary prefix kept as the
        # rematerialisation authority. trajectory -> layer -> fp16
        # (authority_tokens, num_kv_heads * rotary_dim).
        self._authority: dict[bytes, dict[str, torch.Tensor]] = {}
        self._authority_bytes = 0
        self._authority_missing = 0
        self._authority_unmapped = 0
        self._authority_rows_written = 0
        self._rotary = None
        self._rotary_dim: int | None = None
        self._remat_done = 0
        self._remat_missing = 0
        self._remat_mismatch = 0
        self._remat_max_byte_diff = 0
        self._remat_max_abs_delta = 0.0
        self._remat_reports: list[dict] = []

        # Step 066: mamba state snapshots and prefix loads. The mamba groups
        # are not part of the page store, but an assembled prefix needs the
        # recurrent state at its boundary, so the worker keeps a small ring of
        # page-aligned state snapshots per (group, layer) and copies them back
        # into the assembling request's state block.
        self.mamba_group_ids = [
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        self.mamba_page_bytes = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.page_size_bytes
            for group_id in self.mamba_group_ids
        }
        self.snapshot_keep = config.snapshot_keep()
        self.snapshot_traj = config.snapshot_trajectories()
        # (group_id, layer_name) -> (traj_slots * keep, page_bytes) pinned region
        self._snapshot_host: dict[tuple[int, str], torch.Tensor] = {}
        # (group_id, layer_name) -> gpu state-slot views of the mamba groups
        self._mamba_views: dict[tuple[int, str], torch.Tensor] = {}
        self._mamba_layers_per_group: dict[int, list[str]] = {}
        # trajectory -> {boundary: row}, FIFO per trajectory. The ring is
        # per-trajectory on purpose: a global FIFO lets a second trajectory's
        # prefill evict the first one's snapshots, which silently disables
        # assembly for the first (measured in the step 066 second boot: the
        # flush request wiped the ingest trajectory's ring and the serve
        # request fell back to a full prefill).
        self._snapshot_rings: dict[bytes, OrderedDict[int, int]] = {}
        # trajectory -> base row of its ring inside the shared region
        self._snapshot_bases: dict[bytes, int] = {}
        self._snapshot_events: dict[tuple[bytes, int], torch.Event] = {}
        self._load_events: dict[str, torch.Event] = {}
        self._completed_snapshots: list[tuple[bytes, int]] = []
        self._removed_snapshots: list[tuple[bytes, int]] = []
        self._finished_loads: set[str] = set()
        self.snapshots_taken = 0
        self.snapshots_evicted = 0
        self.loads_served = 0
        self.bytes_loaded = 0

    def _snapshot_row(self, trajectory: bytes, boundary: int) -> int | None:
        """Row of a captured boundary, or None if it is not resident."""
        ring = self._snapshot_rings.get(trajectory)
        if not ring:
            return None
        return ring.get(boundary)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self.group_ids:
            logger.warning(
                "vllm-030win patch (step 060): KVMem workspace worker has no "
                "sliding-window group; the store stays inert"
            )
            return
        num_blocks = self.kv_cache_config.num_blocks
        for group_id in self.group_ids:
            group = self.kv_cache_config.kv_cache_groups[group_id]
            spec = group.kv_cache_spec
            page = self.page_bytes[group_id]
            self._block_size[group_id] = spec.block_size
            # Step 064: the physical page is kernel-block interleaved, so the
            # geometry needs the kernel block size. It falls out of the real
            # cache tensor -- the append kernel's blocks are the leading dim
            # before ``group_kernel_blocks`` unflattens them into manager
            # blocks (num_blocks * ratio); when the cache is not split the
            # ratio degenerates to 1.
            first_raw = kv_caches[group.layer_names[0]]
            ratio = first_raw.shape[0] // num_blocks
            if num_blocks * ratio != first_raw.shape[0]:
                raise ValueError(
                    f"cache blocks {first_raw.shape[0]} are not a multiple of "
                    f"num_blocks {num_blocks}"
                )
            self._geometry[group_id] = remat.PageGeometry(
                head_size=spec.head_size,
                num_heads=spec.num_kv_heads,
                block_size=spec.block_size,
                rotary_dim=self._rotary_prefix_width(spec.head_size),
                kernel_block_size=spec.block_size // ratio,
            )
            layer_names: list[str] = []
            for layer_name in group.layer_names:
                ref = group_kernel_blocks(kv_caches[layer_name], num_blocks)
                elem_size = ref.element_size()
                byte_offset = ref.storage_offset() * elem_size
                block_stride_bytes = ref.stride(0) * elem_size
                view = torch.tensor(
                    [], dtype=torch.int8, device=ref.device
                ).set_(
                    ref.untyped_storage(),
                    byte_offset,
                    (num_blocks, page),
                    (block_stride_bytes, 1),
                )
                self._gpu_views[(group_id, layer_name)] = view
                self._host[(group_id, layer_name)] = _alloc_host(
                    self.num_slots, page
                )
                layer_names.append(layer_name)
                self._group_of_layer[layer_name] = group_id
                # The capture and the retrieval index key layers by *index*
                # (``extract_layer_index``), the workspace keys them by name.
                # The authority crosses the two spaces, so the bridge is built
                # here once; a missing entry is counted and warned about rather
                # than silently dropping the rows (step 064 hit exactly that).
                try:
                    index = extract_layer_index(layer_name)
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    logger.warning(
                        "vllm-030win KVMem authority: cannot derive a layer "
                        "index from %r (%r); its pages will not be "
                        "rematerialisable",
                        layer_name,
                        exc,
                    )
                    continue
                existing = self._layer_name_by_index.setdefault(index, layer_name)
                if existing != layer_name:
                    logger.warning(
                        "vllm-030win KVMem authority: layer index %d is shared "
                        "by %r and %r; the authority will use the first",
                        index,
                        existing,
                        layer_name,
                    )
            self._layers_per_group[group_id] = layer_names
            self._scratch_gpu[group_id] = torch.zeros(
                (page,), dtype=torch.int8, device="cuda"
            )
            self._scratch_host[group_id] = torch.zeros(
                (page,), dtype=torch.int8, device="cpu", pin_memory=False
            )
            self._scratch_host2[group_id] = torch.zeros(
                (page,), dtype=torch.int8, device="cpu", pin_memory=False
            )
            logger.info(
                "vllm-030win patch (step 060): KVMem workspace group %d: %d "
                "layers, page %d B (%d chunks x %d B, kernel block %d), %d "
                "host slots (%.2f GiB pinned), block_stride %d B",
                group_id,
                len(layer_names),
                page,
                self._geometry[group_id].chunks_per_page,
                self._geometry[group_id].chunk_bytes,
                self._geometry[group_id].kernel_block_size,
                self.num_slots,
                self.num_slots * page * len(layer_names) / (1024**3),
                block_stride_bytes,
            )
        # Step 066: register the mamba groups' state-slot views and the
        # snapshot ring. One snapshot row is one full state of every mamba
        # group (page_bytes per layer), so a row index addresses the same
        # boundary across all of them.
        if self.mamba_group_ids and config.load_enabled():
            for group_id in self.mamba_group_ids:
                group = self.kv_cache_config.kv_cache_groups[group_id]
                page = self.mamba_page_bytes[group_id]
                layer_names: list[str] = []
                for layer_name in group.layer_names:
                    ref = group_kernel_blocks(kv_caches[layer_name], num_blocks)
                    elem_size = ref.element_size()
                    byte_offset = ref.storage_offset() * elem_size
                    stride_bytes = ref.stride(0) * elem_size
                    view = torch.tensor(
                        [], dtype=torch.int8, device=ref.device
                    ).set_(
                        ref.untyped_storage(),
                        byte_offset,
                        (num_blocks, page),
                        (stride_bytes, 1),
                    )
                    self._mamba_views[(group_id, layer_name)] = view
                    self._snapshot_host[(group_id, layer_name)] = _alloc_host(
                        self.snapshot_traj * self.snapshot_keep, page
                    )
                    layer_names.append(layer_name)
                self._mamba_layers_per_group[group_id] = layer_names
            total_bytes = self.snapshot_traj * self.snapshot_keep * sum(
                self.mamba_page_bytes[group_id]
                * len(self._mamba_layers_per_group[group_id])
                for group_id in self.mamba_group_ids
            )
            logger.info(
                "vllm-030win KVMem assembly (step 066): mamba groups %s "
                "registered, %d layer(s), state slot %s B per layer, snapshot "
                "region %d x %d row(s) (%.2f GiB host)",
                self.mamba_group_ids,
                sum(len(v) for v in self._mamba_layers_per_group.values()),
                next(iter(self.mamba_page_bytes.values()), 0),
                self.snapshot_traj,
                self.snapshot_keep,
                total_bytes / (1024**3),
            )

    # ------------------------------------------------------------------
    # K3 second half: rematerialisation authority (step 064)
    # ------------------------------------------------------------------

    def _text_config(self):
        hf = self.vllm_config.model_config.hf_config
        getter = getattr(hf, "get_text_config", None)
        return getter() if callable(getter) else hf

    def _rotary_prefix_width(self, head_size: int) -> int:
        """How many head dims the model rotates (partial rotary factor)."""
        if self._rotary_dim is None:
            rope = getattr(self._text_config(), "rope_parameters", None) or {}
            factor = float(rope.get("partial_rotary_factor", 1.0))
            self._rotary_dim = max(1, int(head_size * factor))
        return self._rotary_dim

    def _rotary_embedding(self):
        """The model's own rotary embedding, so the convention cannot drift.

        ``get_rope`` is memoised on its arguments, so this returns the very
        instance ``Qwen3NextAttention`` built and therefore the very
        ``cos_sin_cache`` the forward pass rotates with. The config context is
        re-entered because the object is a ``CustomOp``.
        """
        if self._rotary is not None:
            return self._rotary
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope

        text = self._text_config()
        head_size = next(iter(self._geometry.values())).head_size
        max_position = int(getattr(text, "max_position_embeddings", 0) or 0)
        if not max_position:
            max_position = self.vllm_config.model_config.max_model_len
        with set_current_vllm_config(self.vllm_config):
            self._rotary = get_rope(
                head_size=head_size,
                max_position=max_position,
                rope_parameters=getattr(text, "rope_parameters", None),
                dual_chunk_attention_config=None,
            )
        cache = self._rotary.cos_sin_cache
        if cache.shape[-1] != self._rotary_prefix_width(head_size):
            raise RuntimeError(
                "vllm-030win KVMem authority: cos_sin_cache width "
                f"{cache.shape[-1]} does not match the rotary prefix "
                f"{self._rotary_prefix_width(head_size)}"
            )
        logger.info(
            "vllm-030win patch (step 064): KVMem authority rotary = %s, cache "
            "%s %s, rotary_dim %d, is_neox=%s, interleaved=%s",
            type(self._rotary).__name__,
            tuple(cache.shape),
            cache.dtype,
            self._rotary.rotary_dim,
            self._rotary.is_neox_style,
            getattr(self._rotary, "mrope_interleaved", None),
        )
        return self._rotary

    def _authority_width(self) -> int:
        geom = next(iter(self._geometry.values()))
        return geom.num_heads * geom.rotary_dim

    def _authority_region(
        self, trajectory: bytes, layer_name: str
    ) -> torch.Tensor | None:
        """This trajectory's authority for one layer, allocated on first use."""
        per_trajectory = self._authority.get(trajectory)
        if per_trajectory is None:
            limit = config.authority_trajectories()
            if len(self._authority) >= limit:
                self._authority_missing += 1
                return None
            per_trajectory = {}
            self._authority[trajectory] = per_trajectory
            logger.info(
                "vllm-030win patch (step 064): KVMem authority region for "
                "trajectory %s (%d/%d), %d tokens x %d B per layer",
                trajectory.hex()[:12],
                len(self._authority),
                limit,
                config.authority_tokens(),
                self._authority_width() * 2,
            )
        region = per_trajectory.get(layer_name)
        if region is None:
            region = torch.zeros(
                (config.authority_tokens(), self._authority_width()),
                dtype=torch.float16,
            )
            per_trajectory[layer_name] = region
            self._authority_bytes += region.numel() * region.element_size()
        return region

    def _authority_store(
        self, trajectory: bytes, positions: np.ndarray, k_by_layer: dict
    ) -> None:
        """Keep this step's pre-RoPE rotary prefix for future rematerialisation.

        The rows come out of the same capture the retrieval index consumes, so
        this adds a host-side slice and a copy, not a second device read. The
        keys here are capture *layer indices*, so they are bridged to the
        workspace's layer names first; an unmapped index is counted and warned
        about, never dropped in silence.
        """
        for layer_index, k in k_by_layer.items():
            layer_name = self._layer_name_by_index.get(layer_index)
            if layer_name is None:
                self._authority_unmapped += 1
                if self._authority_unmapped <= 3:
                    logger.warning(
                        "vllm-030win KVMem authority: captured layer index %r "
                        "has no workspace layer name (%d known); its rows are "
                        "not kept",
                        layer_index,
                        len(self._layer_name_by_index),
                    )
                continue
            region = self._authority_region(trajectory, layer_name)
            if region is None:
                continue
            group_id = self._group_of_layer.get(layer_name)
            if group_id is None:
                continue
            geom = self._geometry[group_id]
            rows = remat.rotated_prefix_from_packed_k(
                k, geom.head_size, geom.rotary_dim
            ).reshape(k.shape[0], -1)
            valid = positions < region.shape[0]
            if not valid.any():
                self._authority_missing += 1
                continue
            region.numpy()[positions[valid]] = rows[valid]
            self._authority_rows_written += int(valid.sum())

    def _authority_rows(
        self, trajectory: bytes, layer_name: str, positions: np.ndarray
    ) -> np.ndarray | None:
        per_trajectory = self._authority.get(trajectory)
        if per_trajectory is None:
            return None
        region = per_trajectory.get(layer_name)
        if region is None or positions[-1] >= region.shape[0]:
            return None
        return region.numpy()[positions]

    def _run_remat_selftest(self, job) -> None:
        """Rebuild each stored page's rotary prefix from the authority.

        The stored page is the engine's own NVFP4 output, baked at the page's
        original positions. Rematerialising *those same positions* from the
        pre-RoPE authority has to reproduce them: the offline round trip of
        tools/kvmem_remat_test.py runs here on real pages, and it also exercises
        the authority's token indexing end to end. What is compared is the
        quantised bytes and, more usefully, the dequantised values against the
        page's largest E2M1 step. Expected shape: the rebuild is not
        bit-identical because the engine's triton kernel and the host-side
        bake differ in fp32 rounding order, but the differing codes are
        neighbour rungs (~0.5% of bytes) and every delta stays inside one
        quantisation step; a non-finite delta or a delta over one step means
        the layout or the precision contract is broken, not noise.
        """
        if self._remat_done >= REMAT_SELFTEST_PAGES:
            return
        try:
            rotary = self._rotary_embedding()
        except Exception as exc:  # noqa: BLE001 - report, never break the run
            self._remat_missing += 1
            logger.warning(
                "vllm-030win patch (step 064): KVMem rematerialisation selftest "
                "skipped, could not obtain the model's rotary embedding (%r)",
                exc,
            )
            return
        for page in job.pages:
            if self._remat_done >= REMAT_SELFTEST_PAGES:
                break
            group_id = page.group_id
            geom = self._geometry.get(group_id)
            block = self._block_size.get(group_id)
            if geom is None or not block:
                continue
            start = page.page_index * block
            positions = np.arange(start, start + block, dtype=np.int64)
            for layer_name in self._layers_per_group.get(group_id, ()):
                rows = self._authority_rows(job.trajectory, layer_name, positions)
                if rows is None:
                    self._remat_missing += 1
                    continue
                stored = self._host[(group_id, layer_name)][page.slot]
                stored_u8 = stored.view(torch.uint8)
                rebuilt = stored_u8.clone()
                # The authority keeps the pre-RoPE rows as fp16, which is exact
                # for bf16 values. The engine rotates in bf16 against fp32
                # cos/sin (the triton kernel's input precision), so the rebuild
                # has to restore both: feeding the fp16 rows (and truncating
                # cos/sin to match) double-rounds the inputs and flips ~2% of
                # the codes outside the quantisation step.
                raw = torch.from_numpy(
                    np.ascontiguousarray(rows).reshape(
                        block, geom.num_heads, geom.rotary_dim
                    )
                ).to(torch.bfloat16)
                device = rotary.cos_sin_cache.device
                cos_sin = rotary.cos_sin_cache[
                    torch.from_numpy(positions).to(device)
                ].to(device="cpu")
                tokens = torch.arange(block, dtype=torch.long)
                remat.rematerialize_page(
                    rebuilt,
                    geom,
                    raw,
                    tokens,
                    tokens,
                    cos_sin,
                    is_neox_style=bool(rotary.is_neox_style),
                    mrope_section=getattr(rotary, "mrope_section", None),
                )
                n_diff = int((rebuilt != stored_u8).sum())
                packed_old, sf_old = remat.read_rotated(stored_u8, geom, tokens)
                packed_new, sf_new = remat.read_rotated(rebuilt, geom, tokens)
                delta = (
                    remat.dequantize_rotated(packed_old, sf_old)
                    - remat.dequantize_rotated(packed_new, sf_new)
                ).abs()
                nan_count = int(torch.isnan(delta).sum())
                max_delta = float(torch.nan_to_num(delta, nan=0.0, posinf=0.0).max())
                # Largest E2M1 step on the page (the 6->4 magnitude rung), so
                # the report carries a scale-free error measure: a
                # neighbour-code flip must stay <= 1 step.
                sf_value = sf_old.view(torch.float8_e4m3fn).float()
                max_step = float(
                    torch.where(
                        sf_value > 0, 2.0 / sf_value, torch.zeros_like(sf_value)
                    ).max()
                )
                if n_diff:
                    self._remat_mismatch += 1
                    self._remat_max_byte_diff = max(
                        self._remat_max_byte_diff,
                        int(
                            (
                                rebuilt.to(torch.int16) - stored_u8.to(torch.int16)
                            ).abs().max()
                        ),
                    )
                self._remat_max_abs_delta = max(self._remat_max_abs_delta, max_delta)
                self._remat_done += 1
                self._remat_reports.append(
                    {
                        "layer": layer_name,
                        "page_index": page.page_index,
                        "token_start": int(start),
                        "rotated_bytes": int(
                            block * geom.num_heads * (geom.rot_data_bytes + geom.rot_scale_bytes)
                        ),
                        "bytes_differing": n_diff,
                        "max_byte_diff": int(
                            (rebuilt.to(torch.int16) - stored_u8.to(torch.int16))
                            .abs()
                            .max()
                        ),
                        "max_abs_delta_dequantised": max_delta,
                        "max_e2m1_step": max_step,
                        "delta_over_step": (
                            max_delta / max_step if max_step > 0 else 0.0
                        ),
                        "nan_elements": nan_count,
                    }
                )
                if len(self._remat_reports) == 1:
                    self._dump_remat_arrays(
                        layer_name,
                        stored_u8,
                        rebuilt,
                        packed_old,
                        sf_old,
                        packed_new,
                        sf_new,
                        raw,
                        cos_sin,
                        geom,
                        tokens,
                        positions,
                    )
        if self._remat_reports:
            logger.info(
                "vllm-030win patch (step 064): KVMem rematerialisation round "
                "trip: %d page(s) rebuilt from the pre-RoPE authority, %d with "
                "differing bytes, max byte diff %d, max |delta| dequantised "
                "%.6e",
                self._remat_done,
                self._remat_mismatch,
                self._remat_max_byte_diff,
                self._remat_max_abs_delta,
            )
            self._write_remat_report()

    def _dump_remat_arrays(
        self,
        layer_name,
        stored_u8,
        rebuilt,
        packed_old,
        sf_old,
        packed_new,
        sf_new,
        raw,
        cos_sin,
        geom,
        tokens,
        positions,
    ) -> None:
        """One-shot diagnostic dump of the first compared page.

        Kept because the first live run of the round trip disagreed with the
        offline one and the failure mode (differing bytes, non-finite
        dequantised delta) does not say *which* side is wrong; the arrays let
        that be settled offline instead of by re-running a 200K ingest.
        """
        directory = config.dump_dir()
        if not directory:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            baked = remat.bake_rotated_k(
                raw, torch.arange(raw.shape[0], dtype=torch.long), cos_sin
            )
            # numpy has no bfloat16, and the rebuilt rows are bf16 by the
            # precision contract -- dump them as fp32 (exact widening).
            np.savez(
                os.path.join(directory, "kvmem_remat_arrays.npz"),
                layer=np.array([layer_name]),
                positions=positions,
                authority_rows=raw.float().numpy(),
                baked_from_authority=baked.float().numpy(),
                stored_packed=packed_old.numpy(),
                stored_sf=sf_old.view(torch.uint8).numpy(),
                rebuilt_packed=packed_new.numpy(),
                rebuilt_sf=sf_new.view(torch.uint8).numpy(),
                stored_page=stored_u8.numpy(),
                rebuilt_page=rebuilt.numpy(),
                geom=np.array(
                    [
                        geom.head_size,
                        geom.num_heads,
                        geom.block_size,
                        geom.rotary_dim,
                    ]
                ),
            )
            logger.info(
                "vllm-030win patch (step 064): KVMem rematerialisation "
                "diagnostic arrays written to %s",
                os.path.join(directory, "kvmem_remat_arrays.npz"),
            )
        except OSError as exc:
            logger.warning(
                "vllm-030win KVMem rematerialisation diagnostics: could not "
                "write to %s (%s)",
                directory,
                exc,
            )

    def _write_remat_report(self) -> None:
        directory = config.dump_dir()
        if not directory:
            return
        payload = {
            "rotary": type(self._rotary).__name__ if self._rotary else None,
            "rotary_dim": self._rotary_prefix_width(
                next(iter(self._geometry.values())).head_size
            ),
            "pages_rebuilt": self._remat_done,
            "pages_with_differing_bytes": self._remat_mismatch,
            "max_byte_diff": self._remat_max_byte_diff,
            "max_abs_delta_dequantised": self._remat_max_abs_delta,
            "authority_bytes": self._authority_bytes,
            "authority_tokens": config.authority_tokens(),
            "authority_layers": sorted(
                layer
                for per_trajectory in self._authority.values()
                for layer in per_trajectory
            ),
            "layers_mapped": sorted(self._layer_name_by_index),
            "rows_written": self._authority_rows_written,
            "unmapped_layers": self._authority_unmapped,
            "missing": self._authority_missing,
            "authority_nonzero_rows": (
                {
                    layer: int(np.count_nonzero(region.numpy().any(axis=1)))
                    for layer, region in next(iter(self._authority.values())).items()
                }
                if self._authority
                else {}
            ),
            "per_page": self._remat_reports,
        }
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, "kvmem_remat_selftest.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1)
            logger.info(
                "vllm-030win patch (step 064): KVMem rematerialisation report "
                "written to %s",
                path,
            )
        except OSError as exc:
            logger.warning(
                "vllm-030win KVMem rematerialisation report: could not write "
                "to %s (%s)",
                directory,
                exc,
            )

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------

    def bind_connector_metadata(self, metadata) -> None:
        if isinstance(metadata, KVMemConnectorMetadata):
            self._pending = metadata
            self._loads_issued = False
            # Step 075: tell the raw-K capture which token counts belong to this
            # step's prefill spans *before* the forward runs. Speculative decode
            # makes a verify step 1 + num_spec_tokens tokens, so "more than one
            # token" no longer means prefill -- and a verify step is graphed,
            # where the capture body's clone plus M-RoPE torch.equal would
            # invalidate the CUDA graph capture.
            if capture.enabled():
                capture.arm(
                    {
                        span.num_tokens
                        for span in metadata.spans
                        if span.prefill
                    }
                )

    def clear_connector_metadata(self) -> None:
        self._pending = None
        capture.disarm()

    def start_load_kv(self) -> None:
        """Issue this step's prefix-assembly copies.

        This runs from ``start_load_kv`` rather than ``wait_for_save`` because
        the assembling request sits in ``WAITING_FOR_REMOTE_KVS``: its step
        schedules zero tokens, and the no-forward worker path calls only
        ``start_load_kv`` (``wait_for_save`` is skipped there). Issuing the copy
        from the save path would never fire and the request would wait forever.
        """
        metadata = self._pending
        if metadata is None or self._loads_issued:
            return
        self._loads_issued = True
        if getattr(metadata, "load_jobs", None):
            self._run_loads(metadata.load_jobs)

    @staticmethod
    def _copy(entries: list[tuple[int, int, int]]) -> None:
        if not entries:
            return
        src = torch.tensor([e[0] for e in entries], dtype=torch.int64)
        dst = torch.tensor([e[1] for e in entries], dtype=torch.int64)
        sizes = torch.tensor([e[2] for e in entries], dtype=torch.int64)
        ops.swap_blocks_batch(src, dst, sizes)

    # ------------------------------------------------------------------
    # K3: retrieval index
    # ------------------------------------------------------------------

    def _ingest(self, metadata: KVMemConnectorMetadata) -> None:
        """Fold this step's captured pre-RoPE K into the trajectory's index.

        The step is attributed by absolute position, which is the same key the
        workspace itself uses, so a prefix-cache hit (a step whose positions do
        not start where the previous one ended) needs no special case: the rows
        that fall inside a span are exactly that span's.

        A *viewport* span (step 072) is excluded: a rewritten window request
        re-uses the trajectory's leading positions for tokens the window put
        there (sink, placeholder section) and displaces the recent tail, so
        its rows are not the trajectory's rows at those positions -- they must
        not reach the index or the authority. Only the query tail is kept, for
        scoring.
        """
        _s79 = time.monotonic()
        step = capture.drain()
        self._kvtime_add("drain", _s79)
        if not step:
            return
        if self._index is None:
            geometry = capture.stats().get("geometry")
            if geometry is None:
                return
            self._index = KVMemMeanKIndex(*geometry)
        positions = next(iter(step.values()))[0]
        for span in metadata.spans:
            mask = (positions >= span.start) & (
                positions < span.start + span.num_tokens
            )
            if not mask.any():
                continue
            span_positions = positions[mask]
            q_by_layer = {
                layer: q for layer, (_, q, _) in step.items() if q.size
            }
            if q_by_layer:
                width = min(len(q) for q in q_by_layer.values())
                if len(span_positions) >= width:
                    self._last_query[span.trajectory] = (
                        span_positions[-width:],
                        {layer: q[-width:] for layer, q in q_by_layer.items()},
                    )
            if span.viewport:
                continue
            k_by_layer = {layer: k[mask] for layer, (_, _, k) in step.items()}
            self._index.add(span.trajectory, span_positions, k_by_layer)
            if config.authority_enabled():
                self._authority_store(span.trajectory, span_positions, k_by_layer)

    def _score(self, metadata: KVMemConnectorMetadata) -> None:
        if self._index is None:
            return
        started = time.monotonic()
        stage_by_req = {
            stage.request_id: stage for stage in metadata.stage_requests
        }
        for request in metadata.score_requests:
            entry = self._last_query.get(request.trajectory)
            if entry is None:
                logger.warning(
                    "vllm-030win KVMem retrieval: no captured query span for "
                    "trajectory %s (request %s); nothing to score",
                    request.trajectory.hex()[:12],
                    request.request_id,
                )
                continue
            _, q_by_layer = entry
            report = self._index.score(
                request.trajectory,
                q_by_layer,
                block_size=request.block_size,
                num_tokens=request.num_tokens,
                sink_tokens=request.sink_tokens,
                recent_tokens=request.recent_tokens,
                topn=config.retrieval_topn(),
            )
            report["request_id"] = request.request_id
            self._retrieval_reports.append(report)
            self._write_report(report)
            if "error" in report:
                logger.error(
                    "vllm-030win KVMem retrieval (req=%s): %s",
                    request.request_id,
                    report["error"],
                )
                continue
            logger.info(
                "vllm-030win KVMem retrieval (req=%s): %d tokens / %d pages "
                "(block %d, sub-block %d), %d layer(s) scored, eligible %d, "
                "top-%d = %s",
                request.request_id,
                report["num_tokens"],
                report["num_pages"],
                report["block_size"],
                report["subblock"],
                report["num_layers_scored"],
                len(report["eligible"]),
                report["topn"],
                report["top_pages"],
            )
            stage = stage_by_req.get(request.request_id)
            if stage is not None:
                _s79 = time.monotonic()
                self._stage_in(stage, report)
                self._kvtime_add("bake", _s79)
        self._score_seconds += time.monotonic() - started

    def _stage_in(self, stage, report: dict) -> None:
        """Bake the scored pages into the request's retrieval slots (step 072).

        For each slot, in time order, the stored page's bytes are copied to a
        host rebuild buffer (V and the non-rotary K prefix are
        position-independent and keep their bytes), and the rotary prefix is
        rebuilt *at the slot's window position* from the raw-K authority -- one
        rebuild, from the pre-RoPE rows, never from a previously rotated copy
        (design §5.3; the precision contract is the step 065 one: bf16 restore,
        fp32 cos/sin).

        The copies are issued on the current stream and the method returns only
        after issuing all of them; the decode step that follows observes the
        baked slots because it runs on the same stream. Runs synchronously
        inside ``wait_for_save``, so there is no race with the first decode.

        Step 074: each slot carries one physical block *per stored kv cache
        group*, and every layer of every one of those groups is baked. The 16
        full-attention layers sit in two groups under ``VLLM_KV_GROUP_SIZE=8``,
        and a slot whose second group still holds the placeholder prefill reads
        as half-rebuilt KV to the attention of those 8 layers -- which is what
        made a baked needle page unreadable.
        """
        if not stage.slots:
            return
        top_pages = report.get("top_pages") or []
        if not top_pages:
            logger.warning(
                "vllm-030win KVMem viewport (req=%s): scoring produced no "
                "pages; every retrieval slot stays placeholder",
                stage.request_id,
            )
            return
        # Slots fill in time order (design §5.3: the model sees a time-ordered
        # window); the highest-scoring pages win the earliest slots.
        selected = sorted(top_pages)[: len(stage.slots)]
        started = time.monotonic()
        try:
            rotary = self._rotary_embedding()
        except Exception as exc:  # noqa: BLE001 - report, never break the run
            logger.error(
                "vllm-030win KVMem viewport (req=%s): could not obtain the "
                "model's rotary embedding (%r); retrieval slots stay placeholder",
                stage.request_id,
                exc,
            )
            return
        missing_rows = 0
        missing_pages = 0
        baked_slots = 0
        copies = 0
        bytes_written = 0
        groups_baked: set[int] = set()
        # Step 074 read-back: the MiB line only proves the copy was *issued*;
        # this proves it landed in the physical block the decode step reads.
        # VLLM_KVMEM_BAKE_VERIFY = layer-pages checked per slot per group
        # (0 = off; a check is a blocking device->host copy of one page).
        verify_per_slot = config.bake_verify()
        verified = 0
        verify_mismatch = 0
        cos_sin_cache_cpu = rotary.cos_sin_cache.to(device="cpu")
        for slot_j, group_entries in enumerate(stage.slots):
            if slot_j >= len(selected):
                break
            page_index = selected[slot_j]
            slot = stage.pages.get(page_index)
            if slot is None:
                missing_pages += 1
                continue
            slot_copies = 0
            for group_id, gpu_block_id in group_entries:
                geom = self._geometry.get(group_id)
                block = self._block_size.get(group_id)
                if geom is None or not block:
                    continue
                src_positions = np.arange(
                    page_index * stage.page_size,
                    page_index * stage.page_size + block,
                    dtype=np.int64,
                )
                slot_start = stage.slot_start + slot_j * stage.page_size
                dst_positions = torch.arange(
                    slot_start, slot_start + block, dtype=torch.long
                )
                tokens = torch.arange(block, dtype=torch.long)
                checked_this = 0
                for layer_name in self._layers_per_group.get(group_id, ()):
                    rows = self._authority_rows(stage.trajectory, layer_name, src_positions)
                    if rows is None:
                        missing_rows += 1
                        continue
                    stored = self._host[(group_id, layer_name)][slot]
                    rebuilt = stored.view(torch.uint8).clone()
                    # Step 065 precision contract: the authority is fp16 (exact for
                    # bf16 values) and the engine rotates bf16 K against fp32
                    # cos/sin -- restore both before the bake.
                    raw = torch.from_numpy(
                        np.ascontiguousarray(rows).reshape(
                            block, geom.num_heads, geom.rotary_dim
                        )
                    ).to(torch.bfloat16)
                    remat.rematerialize_page(
                        rebuilt,
                        geom,
                        raw,
                        tokens,
                        dst_positions,
                        cos_sin_cache_cpu,
                        is_neox_style=bool(rotary.is_neox_style),
                        mrope_section=getattr(rotary, "mrope_section", None),
                    )
                    dst = self._gpu_views[(group_id, layer_name)][gpu_block_id]
                    dst.copy_(rebuilt.view(torch.int8))
                    slot_copies += 1
                    groups_baked.add(group_id)
                    bytes_written += rebuilt.numel()
                    if checked_this < verify_per_slot:
                        checked_this += 1
                        back = dst.detach().to(device="cpu").view(torch.uint8)
                        verified += 1
                        if not torch.equal(back, rebuilt):
                            verify_mismatch += 1
                            logger.error(
                                "vllm-030win KVMem viewport (req=%s): read-back "
                                "mismatch at slot %d group %d layer %s block %d "
                                "(page %d): %d/%d byte(s) differ from the bake",
                                stage.request_id,
                                slot_j,
                                group_id,
                                layer_name,
                                gpu_block_id,
                                page_index,
                                int((back != rebuilt).sum()),
                                rebuilt.numel(),
                            )
            if slot_copies:
                baked_slots += 1
                copies += slot_copies
        self.stage_in_seconds += time.monotonic() - started
        self.stage_ins_served += 1
        self.stage_in_pages += baked_slots
        logger.info(
            "vllm-030win KVMem viewport (req=%s): baked %d slot(s) x group(s) "
            "%s = %d layer-page copie(s) into the retrieval slots in %.2f s "
            "(%d slot(s) offered, %d without a stored page, %d layer-row(s) "
            "without authority rows); %.1f MiB written, read-back %d checked "
            "%d mismatch",
            stage.request_id,
            baked_slots,
            sorted(groups_baked),
            copies,
            self.stage_in_seconds,
            len(stage.slots),
            missing_pages,
            missing_rows,
            bytes_written / (1024 * 1024),
            verified,
            verify_mismatch,
        )

    def _write_report(self, report: dict) -> None:
        directory = config.dump_dir()
        diag = report.pop("_diag", None)
        if not directory:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            stem = f"kvmem_retrieval_{len(self._retrieval_reports):03d}"
            path = os.path.join(directory, f"{stem}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)
            report["path"] = path
            if diag is not None:
                # The vectors behind the ranking, so the 200K ingest that
                # produced them can be re-scored offline under another
                # reduction, mode or centring (config.dump_kbar).
                diag_path = os.path.join(directory, f"{stem}_kbar.npz")
                np.savez(diag_path, **diag)
                report["diag_path"] = diag_path
        except OSError as exc:
            logger.warning(
                "vllm-030win KVMem retrieval: could not write %s (%s)",
                directory,
                exc,
            )

    def retrieval_reports(self) -> list[dict]:
        return self._retrieval_reports

    # ------------------------------------------------------------------
    # vllm-030win step 079 timing instrumentation (VLLM_KVMEM_TIMING, off by default).
    # ------------------------------------------------------------------

    _KVTIME_SEGMENTS = (
        "drain",
        "sync",
        "copies",
        "ingest",
        "fold",
        "storing",
        "selftest",
        "copy",
        "score",
        "bake",
        "snap",
        "load",
        "save",
    )

    def _kvtime_add(self, key: str, started: float) -> None:
        """Accumulate one segment. Host monotonic only, never syncs."""
        if not self._kvtime_enabled:
            return
        now = time.monotonic()
        self._kvtime[key] = self._kvtime.get(key, 0.0) + (now - started)
        self._kvtime_n[key] = self._kvtime_n.get(key, 0) + 1

    def _kvtime_bump(self, key: str, amount: int = 1) -> None:
        """Count something that has no duration of its own."""
        if not self._kvtime_enabled:
            return
        self._kvtime_n[key] = self._kvtime_n.get(key, 0) + amount

    def _kvtime_step(self) -> None:
        """Emit one [KVTIME] line every VLLM_KVMEM_TIMING_EVERY steps.

        Absolute accumulators plus this window's increment, so the reader
        gets ``d<segment>/dt`` without offline differencing. The line
        reads Python floats only -- no tensor, no stream query, no device
        sync -- because a timer that changes the thing it measures is
        worth nothing (step 078's boot-state spread is exactly what is
        being sampled here).
        """
        if not self._kvtime_enabled:
            return
        steps = self._kvtime_n.get("save", 0)
        if steps % self._kvtime_every:
            return
        now = time.time()
        first = self._kvtime_wall == 0.0
        dt = 0.0 if first else now - self._kvtime_wall
        acc = dict(self._kvtime)
        acc["fold"] = acc.get("ingest", 0.0) - acc.get("drain", 0.0)
        _cap0 = capture.stats()
        acc["sync"] = float(_cap0.get("drain_sync_seconds") or 0.0)
        acc["copies"] = float(_cap0.get("drain_copy_seconds") or 0.0)
        acc["score"] = self._score_seconds  # existing: score + bake
        counts = self._kvtime_n
        if first:
            # Once per boot: prove the patched code is the code running.
            logger.info(
                "vllm-030win patch (step 079): [KVTIME] patch loaded: "
                "every=%d selftest=%d bake_verify=%d pid=%d "
                "page_bytes=%s layers_per_group=%s store_groups=%s "
                "armed=%s geometry=%s",
                self._kvtime_every,
                int(config.roundtrip_selftest()),
                int(config.bake_verify()),
                os.getpid(),
                {g: int(b) for g, b in self.page_bytes.items()},
                {
                    g: len(v)
                    for g, v in self._layers_per_group.items()
                },
                sorted({group for group, _ in self._host}),
                capture.stats().get("armed_counts"),
                capture.stats().get("geometry"),
            )
        parts = [
            f"wall={now:.3f} dt={dt:.3f} "
            f"dsteps={steps - self._kvtime_prev_n.get('save', 0)}"
        ]
        for key in self._KVTIME_SEGMENTS:
            value = acc.get(key, 0.0)
            count = counts.get(key, 0)
            parts.append(
                f"{key}={value:.3f} "
                f"d{key}="
                f"{value - self._kvtime_prev.get(key, 0.0):.3f} "
                f"n{count} "
                f"dn{count - self._kvtime_prev_n.get(key, 0)}"
            )
        ratio = (acc.get("save", 0.0) / dt) if dt > 0 else 0.0
        parts.append(f"acc={ratio:.3f}")
        parts.append(f"bytes={self.bytes_stored / 1048576.0:.1f}MiB")
        parts.append(
            f"copy_calls={counts.get('copy', 0)} "
            f"entries={counts.get('entry', 0)} "
            f"busy={counts.get('busy', 0)}"
        )
        cap = capture.stats()
        parts.append(
            "cap(unarmed=%d decode=%d)"
            % (
                cap.get("skipped_unarmed_steps") or 0,
                cap.get("skipped_decode_steps") or 0,
            )
        )
        self._kvtime_prev = acc
        self._kvtime_prev_n = dict(counts)
        self._kvtime_wall = now
        logger.info(
            "vllm-030win patch (step 079): [KVTIME] " + " | ".join(parts)
        )

    def wait_for_save(self) -> None:
        _s79_save = time.monotonic()
        metadata = self._pending
        self._pending = None
        if metadata is None:
            self._kvtime_add("save", _s79_save)
            self._kvtime_step()
            return
        self._kvtime_bump("busy")
        if getattr(metadata, "spans", None):
            _s79 = time.monotonic()
            self._ingest(metadata)
            self._kvtime_add("ingest", _s79)
        if getattr(metadata, "score_requests", None):
            self._score(metadata)
        if not metadata.store_jobs:
            pass
        else:
            started = time.monotonic()
            for job in metadata.store_jobs:
                entries: list[tuple[int, int, int]] = []
                for page in job.pages:
                    for layer_name in self._layers_per_group.get(page.group_id, ()):
                        src_view = self._gpu_views[(page.group_id, layer_name)]
                        host = self._host[(page.group_id, layer_name)]
                        entries.append(
                            (
                                src_view[page.block_id].data_ptr(),
                                host[page.slot].data_ptr(),
                                self.page_bytes[page.group_id],
                            )
                        )
                _s79 = time.monotonic()
                self._copy(entries)
                self._kvtime_add("copy", _s79)
                self._kvtime_bump("entry", len(entries))
                _s79 = time.monotonic()
                if config.roundtrip_selftest():
                    self._run_selftest(job)
                if config.authority_enabled() and config.roundtrip_selftest():
                    self._run_remat_selftest(job)
                self._kvtime_add("selftest", _s79)
                event = torch.cuda.Event()
                event.record()
                self._events[job.job_id] = event
                self.bytes_stored += sum(
                    self.page_bytes[p.group_id]
                    * len(self._layers_per_group.get(p.group_id, ()))
                    for p in job.pages
                )
            self.store_seconds += time.monotonic() - started
            self._kvtime_add("storing", started)
        # Step 066: loads are issued from ``start_load_kv`` (see there), so a
        # load job can never race the snapshot ring evicting the boundary it
        # reads -- loads run at the start of a step, snapshots at its end.
        if getattr(metadata, "snapshot_requests", None):
            _s79 = time.monotonic()
            self._take_snapshots(metadata.snapshot_requests)
            self._kvtime_add("snap", _s79)
        self._kvtime_add("save", _s79_save)
        self._kvtime_step()

    def _run_loads(self, load_jobs) -> None:
        """Copy workspace pages and mamba snapshots into request blocks.

        The inverse of the store path: the host slot (or snapshot row) is the
        source, the request's freshly allocated block is the destination. The
        copies are issued on the current stream and tracked per request with a
        CUDA event; ``get_finished`` reports the request back to the scheduler
        only once its event has fired, which is what keeps
        ``WAITING_FOR_REMOTE_KVS`` honest.
        """
        started = time.monotonic()
        for job in load_jobs:
            entries: list[tuple[int, int, int]] = []
            missing_snapshot = False
            snapshot_slot = self._snapshot_row(job.trajectory, job.num_tokens)
            if job.mamba_snapshots and snapshot_slot is None:
                logger.error(
                    "vllm-030win KVMem assembly: trajectory %s boundary %d "
                    "has no snapshot row; refusing to load (the request would "
                    "resume from a zeroed recurrent state)",
                    job.trajectory.hex()[:12],
                    job.num_tokens,
                )
                missing_snapshot = True
            if missing_snapshot:
                continue
            for page in job.pages:
                for layer_name in self._layers_per_group.get(page.group_id, ()):
                    src = self._host[(page.group_id, layer_name)][page.slot]
                    dst = self._gpu_views[(page.group_id, layer_name)][
                        page.block_id
                    ]
                    entries.append(
                        (
                            src.data_ptr(),
                            dst.data_ptr(),
                            self.page_bytes[page.group_id],
                        )
                    )
            for group_id, block_id, _ in job.mamba_snapshots:
                for layer_name in self._mamba_layers_per_group.get(group_id, ()):
                    src = self._snapshot_host[(group_id, layer_name)][
                        snapshot_slot
                    ]
                    dst = self._mamba_views[(group_id, layer_name)][block_id]
                    entries.append(
                        (
                            src.data_ptr(),
                            dst.data_ptr(),
                            self.mamba_page_bytes[group_id],
                        )
                    )
            self._copy(entries)
            event = torch.cuda.Event()
            event.record()
            self._load_events[job.req_id] = event
            self.loads_served += 1
            self.bytes_loaded += sum(
                self.page_bytes[p.group_id]
                * len(self._layers_per_group.get(p.group_id, ()))
                for p in job.pages
            ) + sum(
                self.mamba_page_bytes[g]
                * len(self._mamba_layers_per_group.get(g, ()))
                for g, _, _ in job.mamba_snapshots
            )
            logger.info(
                "vllm-030win KVMem assembly: job for request %s issued (%d "
                "page(s) + %d mamba state block(s), boundary %d)",
                job.req_id,
                len(job.pages),
                len(job.mamba_snapshots),
                job.num_tokens,
            )
        self.store_seconds += time.monotonic() - started
        self._kvtime_add("load", started)

    def _take_snapshots(self, snapshot_requests) -> None:
        """Copy the mamba state slots at page-aligned boundaries to the host.

        Each request names the block one page behind the step's end, which the
        align-mode CoW keeps intact (it only advances its running slot, and
        frees the previous one two steps later). Each trajectory owns a FIFO
        ring of ``snapshot_keep`` rows; the evicted boundaries are reported
        back so the scheduler stops matching against them.
        """
        for snapshot in snapshot_requests:
            trajectory = snapshot.trajectory
            ring = self._snapshot_rings.get(trajectory)
            if ring is None:
                if len(self._snapshot_bases) >= self.snapshot_traj:
                    # All trajectory slots taken: evict the least recently
                    # started ring whole. Its boundaries are reported as
                    # removed so the scheduler stops matching them.
                    oldest = next(iter(self._snapshot_bases))
                    old_ring = self._snapshot_rings.pop(oldest)
                    for boundary in old_ring:
                        self._removed_snapshots.append((oldest, boundary))
                    logger.warning(
                        "vllm-030win KVMem snapshot: evicting the whole ring "
                        "of trajectory %s (%d boundary(ies)) to make room for "
                        "%s; raise VLLM_KVMEM_SNAPSHOT_TRAJ to keep more",
                        oldest.hex()[:12],
                        len(old_ring),
                        trajectory.hex()[:12],
                    )
                    base = self._snapshot_bases.pop(oldest)
                    self._snapshot_bases[trajectory] = base
                else:
                    base = len(self._snapshot_bases) * self.snapshot_keep
                    self._snapshot_bases[trajectory] = base
                ring = OrderedDict()
                self._snapshot_rings[trajectory] = ring
            if snapshot.boundary in ring:
                continue
            base = self._snapshot_bases[trajectory]
            free_row = None
            for offset in range(self.snapshot_keep):
                if offset not in ring.values():
                    free_row = base + offset
                    break
            if free_row is None:
                evicted_boundary, evicted_row = ring.popitem(last=False)
                self._removed_snapshots.append((trajectory, evicted_boundary))
                self.snapshots_evicted += 1
                free_row = evicted_row
            entries: list[tuple[int, int, int]] = []
            for group_id, block_id in snapshot.blocks:
                for layer_name in self._mamba_layers_per_group.get(group_id, ()):
                    src = self._mamba_views[(group_id, layer_name)][block_id]
                    dst = self._snapshot_host[(group_id, layer_name)][free_row]
                    entries.append(
                        (
                            src.data_ptr(),
                            dst.data_ptr(),
                            self.mamba_page_bytes[group_id],
                        )
                    )
            self._copy(entries)
            event = torch.cuda.Event()
            event.record()
            ring[snapshot.boundary] = free_row
            self._snapshot_events[(trajectory, snapshot.boundary)] = event
            self.snapshots_taken += 1

    def _run_selftest(self, job) -> None:
        """Copy each stored page back and compare it byte for byte.

        Two directions are checked per page: a fresh device->host copy of the
        source page must equal the stored bytes, and the stored bytes must
        survive host->device->host unchanged. This is the K1 "the page can be
        reloaded" exit, exercised on the real nvfp4 pages.
        """
        if self._selftest_done >= SELFTEST_PAGES:
            return
        for page in job.pages:
            if self._selftest_done >= SELFTEST_PAGES:
                break
            page_bytes = self.page_bytes[page.group_id]
            scratch_gpu = self._scratch_gpu[page.group_id]
            fresh_host = self._scratch_host[page.group_id]
            back_host = self._scratch_host2[page.group_id]
            for layer_name in self._layers_per_group.get(page.group_id, ()):
                src_view = self._gpu_views[(page.group_id, layer_name)]
                host = self._host[(page.group_id, layer_name)]
                stored = host[page.slot]

                # forward: a fresh device->host copy of the source page must
                # equal what the store wrote.
                self._copy(
                    [
                        (
                            src_view[page.block_id].data_ptr(),
                            fresh_host.data_ptr(),
                            page_bytes,
                        )
                    ]
                )
                # reverse: the stored bytes must survive host->device->host.
                self._copy([(stored.data_ptr(), scratch_gpu.data_ptr(), page_bytes)])
                self._copy(
                    [(scratch_gpu.data_ptr(), back_host.data_ptr(), page_bytes)]
                )
                torch.cuda.synchronize()

                stored_np = stored.numpy()
                forward_ok = bool(np.array_equal(stored_np, fresh_host.numpy()))
                reverse_ok = bool(np.array_equal(stored_np, back_host.numpy()))
                if not (forward_ok and reverse_ok):
                    self._selftest_mismatch += 1
                    diff = int(
                        np.abs(
                            stored_np.astype(np.int16)
                            - fresh_host.numpy().astype(np.int16)
                        ).max()
                    )
                    self._selftest_max_byte_diff = max(
                        self._selftest_max_byte_diff, diff
                    )
                    logger.error(
                        "vllm-030win KVMem workspace selftest MISMATCH: group %d "
                        "layer %s page %d slot %d (forward_ok=%s reverse_ok=%s, "
                        "max byte diff %d)",
                        page.group_id,
                        layer_name,
                        page.page_index,
                        page.slot,
                        forward_ok,
                        reverse_ok,
                        diff,
                    )
                self._selftest_pages += 1
            self._selftest_done += 1
        if self._selftest_done:
            logger.info(
                "vllm-030win patch (step 060): KVMem workspace round-trip selftest: "
                "%d page(s) x %d layers byte-identical, %d mismatch",
                self._selftest_done,
                sum(len(v) for v in self._layers_per_group.values()),
                self._selftest_mismatch,
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        del finished_req_ids
        for job_id, event in list(self._events.items()):
            if event.query():
                del self._events[job_id]
                self._completed.append(job_id)
        # Step 066: a request whose assembly copy has fired is reported as
        # finished *recving*, which is what promotes it out of
        # WAITING_FOR_REMOTE_KVS. The set is kept until
        # build_connector_worker_meta has also carried it (the mixin calls the
        # two in that order), since the scheduler's bookkeeping consumes the
        # meta too; re-reporting is idempotent there.
        for req_id, event in list(self._load_events.items()):
            if event.query():
                del self._load_events[req_id]
                self._finished_loads.add(req_id)
        for key, event in list(self._snapshot_events.items()):
            if event.query():
                del self._snapshot_events[key]
                self._completed_snapshots.append(key)
        return set(), set(self._finished_loads)

    def build_connector_worker_meta(self) -> KVMemWorkerMetadata | None:
        if not (
            self._completed
            or self._completed_snapshots
            or self._removed_snapshots
            or self._finished_loads
        ):
            return None
        meta = KVMemWorkerMetadata(
            completed_store_jobs=self._completed,
            completed_snapshots=self._completed_snapshots,
            removed_snapshots=self._removed_snapshots,
            finished_load_reqs=sorted(self._finished_loads),
        )
        self._completed = []
        self._completed_snapshots = []
        self._removed_snapshots = []
        self._finished_loads = set()
        return meta

    def stats(self) -> dict:
        return {
            "num_slots": self.num_slots,
            "bytes_stored": self.bytes_stored,
            "store_seconds": round(self.store_seconds, 3),
            "jobs_in_flight": len(self._events),
            "selftest_pages": self._selftest_pages,
            "selftest_mismatch": self._selftest_mismatch,
            "selftest_max_byte_diff": self._selftest_max_byte_diff,
            "capture": capture.stats(),
            "index": self._index.stats() if self._index is not None else None,
            "score_seconds": round(self._score_seconds, 3),
            "retrieval_reports": len(self._retrieval_reports),
            "kvtime": {
                "enabled": self._kvtime_enabled,
                "every": self._kvtime_every,
                "seconds": {
                    k: round(v, 3) for k, v in self._kvtime.items()
                },
                "counts": dict(self._kvtime_n),
            },
            "authority": {
                "enabled": config.authority_enabled(),
                "trajectories": len(self._authority),
                "bytes": self._authority_bytes,
                "tokens": config.authority_tokens(),
                "layers_mapped": len(self._layer_name_by_index),
                "rows_written": self._authority_rows_written,
                "missing": self._authority_missing,
                "unmapped_layers": self._authority_unmapped,
                "remat_pages": self._remat_done,
                "remat_mismatch": self._remat_mismatch,
                "remat_max_byte_diff": self._remat_max_byte_diff,
                "remat_max_abs_delta": self._remat_max_abs_delta,
            },
        }

    def remat_reports(self) -> list[dict]:
        return self._remat_reports
