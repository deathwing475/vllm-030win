# -*- coding: utf-8 -*-
"""步骤 079 计时补丁：给 KVMem 工作区 worker 加分段墙钟台账（``VLLM_KVMEM_TIMING`` 门控）。

背景
----
078 实测同一份 launcher 的 boot 之间 1456-token 页步在 1.42 ↔ 11.0 s 之间跳（7.8×）、
8k decode 在 62 ↔ 111 tok/s 之间跳且与 prefill 反相关，并用外置指标否掉了七类解释，
但**没有任何计时证据**：`KVMemConnector.workspace_stats()` 全仓零调用者，而现成的
``store_seconds`` 又同时被"入库"（worker.py:1146）与"装配取侧"（worker.py:1227）两条
路径累加，且它的窗口里含 roundtrip 自测的 ``torch.cuda.synchronize()``（worker.py:1335）
⇒ 裸读必假。另一方面 ``_ingest``（内含 ``capture.drain()`` 的每页步 48 次阻塞 ``.cpu()``）
与 ``_take_snapshots`` 完全没有累加器。本补丁把这些段各自切开，并每 N 步出一条
``[KVTIME]`` 台账，用来回答"慢态的时间到底在哪"。

用法
----
    python tools/apply_kvmem_timing_step079.py status --target both
    python tools/apply_kvmem_timing_step079.py apply  --target both
    python tools/apply_kvmem_timing_step079.py revert --target both   # 必守 14

门控
----
``VLLM_KVMEM_TIMING`` 不设（默认 0）时，三个 ``_kvtime_*`` 辅助函数全部首行 return，
唯一的行为差别是 ``wait_for_save`` 里多了几个 ``time.monotonic()`` 局部赋值。补丁只落在
``vllm/v1/kvmem_workspace/``（062 实测该目录不改 AOT 编译缓存键 ⇒ 不会引入重编译 boot）。

纪律
----
台账只读 Python float：不碰张量、不查流、不做 device→host 同步（必守 21 / 16⑥ —— 计时
器不能改变它要测的东西）。``store_seconds`` 与 ``_score_seconds`` 的既有语义**一字不动**
（只增键不改名），历史日志与离线分析器零破坏；新段 ``storing``/``copy``/``selftest``/
``load`` 另立，"发射端 Python 开销" = ``storing − copy − selftest``。
"""
import argparse
import os
import sys

REPO_ROOT = r"G:\qwen3.8model\vllm-030win-git\vllm"
VENV_ROOT = r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm"
FILES = {
    "worker": r"v1\kvmem_workspace\worker.py",
    "config": r"v1\kvmem_workspace\config.py",
    "capture": r"v1\kvmem_workspace\capture.py",
}
MARK = "step 079 timing instrumentation"


def T(*lines: str) -> str:
    return "\n".join(lines)


# (file_key, name, old, new) — 全部用 LF 书写，读写时按目标文件自身行尾转换。
EDITS = [
    # ------------------------------------------------------------------ config
    (
        "config",
        "config_timing_knobs",
        T(
            "def sweep_enabled() -> bool:",
        ),
        T(
            "def timing_enabled() -> bool:",
            '    """vllm-030win step 079 timing instrumentation: the [KVTIME] ledger.',
            "",
            "    Off by default: with the gate off every ``_kvtime_*`` helper in",
            "    worker.py returns on its first statement, so the only cost left in",
            "    the hot path is a few ``time.monotonic()`` locals. Revert with",
            "    tools/apply_kvmem_timing_step079.py revert.",
            '    """',
            '    return bool(int(os.environ.get("VLLM_KVMEM_TIMING", "0")))',
            "",
            "",
            "def timing_every() -> int:",
            '    """``wait_for_save`` calls between two [KVTIME] lines.',
            "",
            "    ``_env_int`` rejects <= 0, so the smallest interval is one step. A",
            "    60K-token ingest is ~42 page steps, so 50 would print nothing at all",
            '    -- the step 079 arm sets 25.',
            '    """',
            '    return _env_int("VLLM_KVMEM_TIMING_EVERY", 50)',
            "",
            "",
            "def sweep_enabled() -> bool:",
        ),
    ),
    # ------------------------------------------------------------------ capture
    # b1 readout: a single "drain" number is uninterpretable, because the first
    # ``.cpu()`` in that loop also waits for everything the step queued on the
    # stream -- so the model's own compute gets billed to the copy path. Split
    # it: ``sync`` = until the first host copy returns, ``copies`` = the rest.
    (
        "capture",
        "capture_import_time",
        T(
            "import numpy as np",
            "import torch",
        ),
        T(
            "import numpy as np",
            "import time",
            "import torch",
        ),
    ),
    (
        "capture",
        "capture_drain_acc",
        T(
            "_SKIPPED_DECODE = 0",
            "_SKIPPED_UNARMED = 0",
            "_MROPE_AXES_DIFFER = 0",
        ),
        T(
            "_SKIPPED_DECODE = 0",
            "_SKIPPED_UNARMED = 0",
            "_MROPE_AXES_DIFFER = 0",
            "# vllm-030win step 079 timing instrumentation: drain(), split.",
            "# A dict so the loop below needs no ``global`` statements.",
            '_DRAIN_ACC: dict[str, float] = {"sync": 0.0, "copies": 0.0}',
        ),
    ),
    (
        "capture",
        "capture_drain_split",
        T(
            "    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}",
            "    for layer_idx, (positions, q_tail, k) in stash.items():",
            "        out[layer_idx] = (",
            "            positions.cpu().numpy(),",
            "            q_tail.cpu().to(torch.float16).numpy(),",
            "            k.cpu().to(torch.float16).numpy(),",
            "        )",
        ),
        T(
            "    _t079 = time.monotonic()",
            "    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}",
            "    _split = False",
            "    for layer_idx, (positions, q_tail, k) in stash.items():",
            "        _pos = positions.cpu().numpy()",
            "        if not _split:",
            '            _DRAIN_ACC["sync"] += time.monotonic() - _t079',
            "            _split = True",
            "            _t079 = time.monotonic()",
            "        out[layer_idx] = (",
            "            _pos,",
            "            q_tail.cpu().to(torch.float16).numpy(),",
            "            k.cpu().to(torch.float16).numpy(),",
            "        )",
            '    _DRAIN_ACC["copies"] += time.monotonic() - _t079',
        ),
    ),
    (
        "capture",
        "capture_stats",
        T(
            '        "geometry": _GEOMETRY,',
        ),
        T(
            '        "geometry": _GEOMETRY,',
            '        "drain_sync_seconds": round(_DRAIN_ACC["sync"], 3),',
            '        "drain_copy_seconds": round(_DRAIN_ACC["copies"], 3),',
        ),
    ),
    # ------------------------------------------------------------------ worker
    (
        "worker",
        "worker_state",
        T(
            "        self.bytes_stored = 0",
            "        self.store_seconds = 0.0",
        ),
        T(
            "        self.bytes_stored = 0",
            "        self.store_seconds = 0.0",
            "",
            "        # vllm-030win step 079 timing instrumentation (env-gated).",
            "        # ``storing`` is the store window as it already stands; ``copy``",
            "        # and ``selftest`` are cut out of it so the entry-building Python",
            "        # (storing - copy - selftest) and the selftest's device sync are",
            "        # visible separately, and ``load`` splits the assembly side out",
            "        # of store_seconds (078: both directions share that accumulator).",
            "        self._kvtime: dict[str, float] = {}",
            "        self._kvtime_n: dict[str, int] = {}",
            "        self._kvtime_prev: dict[str, float] = {}",
            "        self._kvtime_prev_n: dict[str, int] = {}",
            "        self._kvtime_wall = 0.0",
            "        self._kvtime_enabled = config.timing_enabled()",
            "        self._kvtime_every = max(1, config.timing_every())",
        ),
    ),
    (
        "worker",
        "worker_helpers_and_wait_for_save",
        T(
            "    def wait_for_save(self) -> None:",
            "        metadata = self._pending",
            "        self._pending = None",
            "        if metadata is None:",
            "            return",
            '        if getattr(metadata, "spans", None):',
            "            self._ingest(metadata)",
            '        if getattr(metadata, "score_requests", None):',
            "            self._score(metadata)",
        ),
        T(
            "    # ------------------------------------------------------------------",
            "    # vllm-030win %s (VLLM_KVMEM_TIMING, off by default)." % MARK,
            "    # ------------------------------------------------------------------",
            "",
            "    _KVTIME_SEGMENTS = (",
            '        "drain",',
            '        "sync",',
            '        "copies",',
            '        "ingest",',
            '        "fold",',
            '        "storing",',
            '        "selftest",',
            '        "copy",',
            '        "score",',
            '        "bake",',
            '        "snap",',
            '        "load",',
            '        "save",',
            "    )",
            "",
            "    def _kvtime_add(self, key: str, started: float) -> None:",
            '        """Accumulate one segment. Host monotonic only, never syncs."""',
            "        if not self._kvtime_enabled:",
            "            return",
            "        now = time.monotonic()",
            "        self._kvtime[key] = self._kvtime.get(key, 0.0) + (now - started)",
            "        self._kvtime_n[key] = self._kvtime_n.get(key, 0) + 1",
            "",
            "    def _kvtime_bump(self, key: str, amount: int = 1) -> None:",
            '        """Count something that has no duration of its own."""',
            "        if not self._kvtime_enabled:",
            "            return",
            "        self._kvtime_n[key] = self._kvtime_n.get(key, 0) + amount",
            "",
            "    def _kvtime_step(self) -> None:",
            '        """Emit one [KVTIME] line every VLLM_KVMEM_TIMING_EVERY steps.',
            "",
            "        Absolute accumulators plus this window's increment, so the reader",
            "        gets ``d<segment>/dt`` without offline differencing. The line",
            "        reads Python floats only -- no tensor, no stream query, no device",
            "        sync -- because a timer that changes the thing it measures is",
            "        worth nothing (step 078's boot-state spread is exactly what is",
            "        being sampled here).",
            '        """',
            "        if not self._kvtime_enabled:",
            "            return",
            '        steps = self._kvtime_n.get("save", 0)',
            "        if steps % self._kvtime_every:",
            "            return",
            "        now = time.time()",
            "        first = self._kvtime_wall == 0.0",
            "        dt = 0.0 if first else now - self._kvtime_wall",
            "        acc = dict(self._kvtime)",
            '        acc["fold"] = acc.get("ingest", 0.0) - acc.get("drain", 0.0)',
            "        _cap0 = capture.stats()",
            '        acc["sync"] = float(_cap0.get("drain_sync_seconds") or 0.0)',
            '        acc["copies"] = float(_cap0.get("drain_copy_seconds") or 0.0)',
            '        acc["score"] = self._score_seconds  # existing: score + bake',
            "        counts = self._kvtime_n",
            "        if first:",
            "            # Once per boot: prove the patched code is the code running.",
            "            logger.info(",
            '                "vllm-030win patch (step 079): [KVTIME] patch loaded: "',
            '                "every=%d selftest=%d bake_verify=%d pid=%d "',
            '                "page_bytes=%s layers_per_group=%s store_groups=%s "',
            '                "armed=%s geometry=%s",',
            "                self._kvtime_every,",
            "                int(config.roundtrip_selftest()),",
            "                int(config.bake_verify()),",
            "                os.getpid(),",
            "                {g: int(b) for g, b in self.page_bytes.items()},",
            "                {",
            "                    g: len(v)",
            "                    for g, v in self._layers_per_group.items()",
            "                },",
            "                sorted({group for group, _ in self._host}),",
            "                capture.stats().get(\"armed_counts\"),",
            "                capture.stats().get(\"geometry\"),",
            "            )",
            "        parts = [",
            '            f"wall={now:.3f} dt={dt:.3f} "',
            "            f\"dsteps={steps - self._kvtime_prev_n.get('save', 0)}\"",
            "        ]",
            "        for key in self._KVTIME_SEGMENTS:",
            "            value = acc.get(key, 0.0)",
            "            count = counts.get(key, 0)",
            "            parts.append(",
            '                f"{key}={value:.3f} "',
            "                f\"d{key}=\"",
            "                f\"{value - self._kvtime_prev.get(key, 0.0):.3f} \"",
            '                f"n{count} "',
            "                f\"dn{count - self._kvtime_prev_n.get(key, 0)}\"",
            "            )",
            "        ratio = (acc.get(\"save\", 0.0) / dt) if dt > 0 else 0.0",
            '        parts.append(f"acc={ratio:.3f}")',
            '        parts.append(f"bytes={self.bytes_stored / 1048576.0:.1f}MiB")',
            "        parts.append(",
            '            f"copy_calls={counts.get(\'copy\', 0)} "',
            '            f"entries={counts.get(\'entry\', 0)} "',
            '            f"busy={counts.get(\'busy\', 0)}"',
            "        )",
            "        cap = capture.stats()",
            "        parts.append(",
            '            "cap(unarmed=%d decode=%d)"',
            "            % (",
            '                cap.get("skipped_unarmed_steps") or 0,',
            '                cap.get("skipped_decode_steps") or 0,',
            "            )",
            "        )",
            "        self._kvtime_prev = acc",
            "        self._kvtime_prev_n = dict(counts)",
            "        self._kvtime_wall = now",
            "        logger.info(",
            '            "vllm-030win patch (step 079): [KVTIME] " + " | ".join(parts)',
            "        )",
            "",
            "    def wait_for_save(self) -> None:",
            "        _s79_save = time.monotonic()",
            "        metadata = self._pending",
            "        self._pending = None",
            "        if metadata is None:",
            "            self._kvtime_add(\"save\", _s79_save)",
            "            self._kvtime_step()",
            "            return",
            '        self._kvtime_bump("busy")',
            '        if getattr(metadata, "spans", None):',
            "            _s79 = time.monotonic()",
            "            self._ingest(metadata)",
            "            self._kvtime_add(\"ingest\", _s79)",
            '        if getattr(metadata, "score_requests", None):',
            "            self._score(metadata)",
        ),
    ),
    (
        "worker",
        "worker_drain",
        T(
            "        step = capture.drain()",
            "        if not step:",
            "            return",
        ),
        T(
            "        _s79 = time.monotonic()",
            "        step = capture.drain()",
            "        self._kvtime_add(\"drain\", _s79)",
            "        if not step:",
            "            return",
        ),
    ),
    (
        "worker",
        "worker_bake",
        T(
            "            stage = stage_by_req.get(request.request_id)",
            "            if stage is not None:",
            "                self._stage_in(stage, report)",
            "        self._score_seconds += time.monotonic() - started",
        ),
        T(
            "            stage = stage_by_req.get(request.request_id)",
            "            if stage is not None:",
            "                _s79 = time.monotonic()",
            "                self._stage_in(stage, report)",
            "                self._kvtime_add(\"bake\", _s79)",
            "        self._score_seconds += time.monotonic() - started",
        ),
    ),
    (
        "worker",
        "worker_copy_and_selftest",
        T(
            "                self._copy(entries)",
            "                if config.roundtrip_selftest():",
            "                    self._run_selftest(job)",
            "                if config.authority_enabled() and config.roundtrip_selftest():",
            "                    self._run_remat_selftest(job)",
        ),
        T(
            "                _s79 = time.monotonic()",
            "                self._copy(entries)",
            "                self._kvtime_add(\"copy\", _s79)",
            '                self._kvtime_bump("entry", len(entries))',
            "                _s79 = time.monotonic()",
            "                if config.roundtrip_selftest():",
            "                    self._run_selftest(job)",
            "                if config.authority_enabled() and config.roundtrip_selftest():",
            "                    self._run_remat_selftest(job)",
            "                self._kvtime_add(\"selftest\", _s79)",
        ),
    ),
    (
        "worker",
        "worker_store_window",
        T(
            "            self.store_seconds += time.monotonic() - started",
            "        # Step 066: loads are issued from ``start_load_kv`` (see there), so a",
        ),
        T(
            "            self.store_seconds += time.monotonic() - started",
            "            self._kvtime_add(\"storing\", started)",
            "        # Step 066: loads are issued from ``start_load_kv`` (see there), so a",
        ),
    ),
    (
        "worker",
        "worker_snapshots_and_emit",
        T(
            '        if getattr(metadata, "snapshot_requests", None):',
            "            self._take_snapshots(metadata.snapshot_requests)",
            "",
            "    def _run_loads(self, load_jobs) -> None:",
        ),
        T(
            '        if getattr(metadata, "snapshot_requests", None):',
            "            _s79 = time.monotonic()",
            "            self._take_snapshots(metadata.snapshot_requests)",
            "            self._kvtime_add(\"snap\", _s79)",
            '        self._kvtime_add("save", _s79_save)',
            "        self._kvtime_step()",
            "",
            "    def _run_loads(self, load_jobs) -> None:",
        ),
    ),
    (
        "worker",
        "worker_load_window",
        T(
            "        self.store_seconds += time.monotonic() - started",
            "",
            "    def _take_snapshots(self, snapshot_requests) -> None:",
        ),
        T(
            "        self.store_seconds += time.monotonic() - started",
            "        self._kvtime_add(\"load\", started)",
            "",
            "    def _take_snapshots(self, snapshot_requests) -> None:",
        ),
    ),
    (
        "worker",
        "worker_stats",
        T(
            '            "retrieval_reports": len(self._retrieval_reports),',
        ),
        T(
            '            "retrieval_reports": len(self._retrieval_reports),',
            '            "kvtime": {',
            '                "enabled": self._kvtime_enabled,',
            '                "every": self._kvtime_every,',
            '                "seconds": {',
            "                    k: round(v, 3) for k, v in self._kvtime.items()",
            "                },",
            '                "counts": dict(self._kvtime_n),',
            "            },",
        ),
    ),
]


def _paths(target: str) -> list[str]:
    roots = []
    if target in ("venv", "both"):
        roots.append(VENV_ROOT)
    if target in ("repo", "both"):
        roots.append(REPO_ROOT)
    return [os.path.join(root, rel) for rel in FILES.values() for root in roots]


def _read(path: str) -> tuple[str, str]:
    with open(path, "rb") as fh:
        data = fh.read()
    eol = "crlf" if data.count(b"\r\n") and not data.count(b"\n") - data.count(
        b"\r\n"
    ) else "lf"
    return data.replace(b"\r\n", b"\n").decode("utf-8"), eol


def _write(path: str, text: str, eol: str) -> None:
    data = text.encode("utf-8")
    if eol == "crlf":
        data = data.replace(b"\n", b"\r\n")
    with open(path, "wb") as fh:
        fh.write(data)


def _state(path: str) -> str:
    text, _eol = _read(path)
    return "patched" if MARK in text else "clean"


def _edits_for(path: str) -> list[tuple[str, str, str]]:
    for key, rel in FILES.items():
        if path.endswith(rel):
            return [(n, o, x) for k, n, o, x in EDITS if k == key]
    raise AssertionError(f"no edit set for {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["apply", "revert", "status"])
    ap.add_argument("--target", choices=["venv", "repo", "both"], default="venv")
    args = ap.parse_args()

    if args.mode == "status":
        for p in _paths(args.target):
            print(f"{_state(p):8s} {p}")
        return 0

    for path in _paths(args.target):
        edits = _edits_for(path)
        text, eol = _read(path)
        if args.mode == "apply":
            if MARK in text:
                print(f"already patched ({eol}): {path}")
                continue
            for name, old, new in edits:
                n = text.count(old)
                assert n == 1, f"anchor {name} count={n} in {path}"
            for name, old, new in edits:
                text = text.replace(old, new, 1)
            compile(text, path, "exec")
            _write(path, text, eol)
            print(f"patch applied ({eol}, {len(edits)} edits): {path}")
        else:
            if MARK not in text:
                print(f"already clean: {path}")
                continue
            for name, old, new in reversed(edits):
                n = text.count(new)
                assert n == 1, f"new block {name} count={n} in {path}"
            for name, old, new in reversed(edits):
                text = text.replace(new, old, 1)
            assert MARK not in text
            compile(text, path, "exec")
            _write(path, text, eol)
            print(f"patch reverted ({eol}): {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
