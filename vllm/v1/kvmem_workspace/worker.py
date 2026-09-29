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

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import group_kernel_blocks
from vllm.v1.kvmem_workspace import capture, config
from vllm.v1.kvmem_workspace.groups import workspace_group_ids
from vllm.v1.kvmem_workspace.index import KVMemMeanKIndex
from vllm.v1.kvmem_workspace.metadata import (
    KVMemConnectorMetadata,
    KVMemWorkerMetadata,
)

logger = init_logger(__name__)

SELFTEST_PAGES = 8


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
        # scratch buffers for the round-trip self test
        self._scratch_gpu: dict[int, torch.Tensor] = {}
        self._scratch_host: dict[int, torch.Tensor] = {}
        self._scratch_host2: dict[int, torch.Tensor] = {}

        self._pending: KVMemConnectorMetadata | None = None
        self._events: dict[int, torch.Event] = {}
        self._completed: list[int] = []
        self._selftest_done = 0
        self._selftest_pages = 0
        self._selftest_mismatch = 0
        self._selftest_max_byte_diff = 0
        self.bytes_stored = 0
        self.store_seconds = 0.0

        # K3: retrieval index (host-resident) and the trailing query span of
        # the most recent prefill step of each trajectory.
        self._index: KVMemMeanKIndex | None = None
        self._last_query: dict[bytes, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
        self._retrieval_reports: list[dict] = []
        self._score_seconds = 0.0

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
                "layers, page %d B, %d host slots (%.2f GiB pinned), "
                "block_stride %d B",
                group_id,
                len(layer_names),
                page,
                self.num_slots,
                self.num_slots * page * len(layer_names) / (1024**3),
                block_stride_bytes,
            )

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------

    def bind_connector_metadata(self, metadata) -> None:
        if isinstance(metadata, KVMemConnectorMetadata):
            self._pending = metadata

    def clear_connector_metadata(self) -> None:
        self._pending = None

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
        """
        step = capture.drain()
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
            self._index.add(
                span.trajectory,
                span_positions,
                {layer: k[mask] for layer, (_, _, k) in step.items()},
            )
            # The query span is the tail of the step, and the last prefill step
            # is the one that holds the tail of the prompt, so overwriting here
            # leaves exactly the right query behind. Rows are ordered, so the
            # last `width` of the masked positions line up with the last
            # `width` rows of q.
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

    def _score(self, metadata: KVMemConnectorMetadata) -> None:
        if self._index is None:
            return
        started = time.monotonic()
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
        self._score_seconds += time.monotonic() - started

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

    def wait_for_save(self) -> None:
        metadata = self._pending
        self._pending = None
        if metadata is None:
            return
        if getattr(metadata, "spans", None):
            self._ingest(metadata)
        if getattr(metadata, "score_requests", None):
            self._score(metadata)
        if not metadata.store_jobs:
            return
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
            self._copy(entries)
            if config.roundtrip_selftest():
                self._run_selftest(job)
            event = torch.cuda.Event()
            event.record()
            self._events[job.job_id] = event
            self.bytes_stored += sum(
                self.page_bytes[p.group_id]
                * len(self._layers_per_group.get(p.group_id, ()))
                for p in job.pages
            )
        self.store_seconds += time.monotonic() - started

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
        return set(), set()

    def build_connector_worker_meta(self) -> KVMemWorkerMetadata | None:
        if not self._completed:
            return None
        meta = KVMemWorkerMetadata(completed_store_jobs=self._completed)
        self._completed = []
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
        }
