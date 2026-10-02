# -*- coding: utf-8 -*-
"""步骤 080：把 ``capture._record_impl`` 里的 host 同步挪出前向，并给台账补上 ``record`` 段。

背景（py-spy 实测，不是推演）
--------------------------
078/079 两支都回答不了"慢态那 8-11 s/页乘在哪"。079 的四分表说 ``outside`` 占
76-84%（快态）/96.3%（慢态），而台账只计 ``wait_for_save`` —— 于是**在前向内部跑的
``capture.record`` custom op 从来没被计量过**，正好落进 ``outside`` 这个"连接器之外"
的筐里（必守 24①的坑：没计量的段读起来像无罪）。

慢态 boot b4（9.80 s/页步，30 页步 × 4 次复测同值）的 py-spy 采样：
**7640 个样本里 94.62% 的栈顶停在 ``kvmem_workspace/capture.py:166 _record_impl``**，
调用链 = ``execute_model → qwen3_next.forward → piecewise_backend → record → _record_impl``。
第 166 行是 ``torch.equal(positions[0], positions[1])`` —— CUDA 张量上的 ``torch.equal``
**会阻塞到流排空**，而它在每个 full-attention 层各跑一次 ⇒ **每个 1456-token 前向付 16 个
同步点**。带外指纹同时给出：功耗 84.8 W / limit 350、``utilization.memory`` 1%、SM 3060
= 满频、PCIe gen5 x16 = max、pagefile 与磁盘读平直、engine 进程精确 100% 单核 ⇒
"GPU 不在算，host 在等"。这正是 078 那张"等待型"画像的机制。

本补丁做两件事
--------------
1. **修**（门控 ``VLLM_KVMEM_RECORD_NOSYNC``，默认 0 = 旧行为）：门控打开时把
   ``positions[:2]`` 两行一起暂存，同轴判定挪到 ``drain()`` 里做 —— 那里这些数据
   本来就要 ``.cpu()`` 回主机，判定变成纯 numpy，**前向里零同步**，canary 语义与
   计数粒度（每层一次）不变。
2. **计量**（门控 ``VLLM_KVMEM_TIMING``，沿用 079 的开关）：``capture`` 侧累计
   ``record`` 的秒数/调用数/同步调用数，由 ``worker`` 镜像进 ``[KVTIME]`` 台账成为
   第 14 段 ``rec=... drec=... n... dn...``，分析器 ``kvmem_time_budget.py`` 把它
   从 ``outside`` 里剥出来 ⇒ 四分表从此对连接器前向内部负责。

纪律
----
* 只落 ``vllm/v1/kvmem_workspace/``（062/079 实测该目录**不改 AOT 编译缓存键** ⇒
  不引入重编译 boot，必守 17）；``worker.py`` 的台账只读 Python float，不碰张量、
  不查流、不做 device→host 同步（必守 21/16⑥）。
* 默认关 ⇒ 生产不动（生产压根不设 ``VLLM_KVMEM_*``，连接器不在场）。
* 必守 14：``apply|revert|status`` 三态 + 每条锚点 ``assert count==1`` + 写前
  ``compile()`` + revert 后 ``assert MARK not in text``，**apply/revert 往返实测**。
* ``store_seconds``/``_score_seconds``/既有段名**一字不动**（只增键不改名）。

用法
----
    python tools/apply_kvmem_record_sync_step080.py status --target both
    python tools/apply_kvmem_record_sync_step080.py apply  --target both
    python tools/apply_kvmem_record_sync_step080.py revert --target both
    python tools/kvmem_timing_unit_test.py                 # 跑臂前的必需闸门
"""
import argparse
import os
import sys

REPO_ROOT = r"G:\qwen3.8model\vllm-030win-git\vllm"
VENV_ROOT = r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm"
FILES = {
    "config": r"v1\kvmem_workspace\config.py",
    "capture": r"v1\kvmem_workspace\capture.py",
    "worker": r"v1\kvmem_workspace\worker.py",
}
MARK = "step 080 capture-record fix"


def T(*lines: str) -> str:
    return "\n".join(lines)


# (file_key, name, old, new) — LF 书写，读写按目标文件自身行尾转换。
EDITS = [
    # ------------------------------------------------------------------ config
    (
        "config",
        "config_record_nosync",
        T(
            "def sweep_enabled() -> bool:",
        ),
        T(
            "def record_nosync() -> bool:",
            '    """vllm-030win step 080 capture-record fix: keep the host sync',
            "    out of the capture op.",
            "",
            "    ``capture._record_impl`` used to call ``torch.equal(positions[0],",
            "    positions[1])`` once per full-attention layer, i.e. 16 stream-",
            "    draining syncs inside every prefill forward. Off by default = the",
            "    old behaviour, byte for byte; the step 080 arm sets it to 1.",
            "    Revert with tools/apply_kvmem_record_sync_step080.py revert.",
            '    """',
            '    return bool(int(os.environ.get("VLLM_KVMEM_RECORD_NOSYNC", "0")))',
            "",
            "",
            "def sweep_enabled() -> bool:",
        ),
    ),
    # ---------------------------------------------------------------- capture
    (
        "capture",
        "capture_rec_acc",
        T(
            '_DRAIN_ACC: dict[str, float] = {"sync": 0.0, "copies": 0.0}',
        ),
        T(
            '_DRAIN_ACC: dict[str, float] = {"sync": 0.0, "copies": 0.0}',
            "# vllm-030win step 080 capture-record fix: the cost of the record()",
            "# custom op, which runs INSIDE execute_model and therefore was never",
            "# inside any window the wait_for_save-based [KVTIME] ledger measured.",
            '#   seconds    = monotonic time spent in the armed body (per call)',
            '#   calls      = armed calls (16 per page step on this model)',
            '#   sync_calls = of those, how many paid the torch.equal stream sync',
            '_REC_ACC: dict[str, float] = {"seconds": 0.0, "calls": 0.0,',
            '                              "sync_calls": 0.0}',
            "# Resolved once: _record_impl runs 16 times per page step.",
            "_NOSYNC: bool | None = None",
            "",
            "",
            "def _nosync() -> bool:",
            '    """``VLLM_KVMEM_RECORD_NOSYNC``, cached (this is a hot path)."""',
            "    global _NOSYNC",
            "    if _NOSYNC is None:",
            "        from vllm.v1.kvmem_workspace import config",
            "",
            "        _NOSYNC = bool(config.record_nosync())",
            "    return _NOSYNC",
        ),
    ),
    (
        "capture",
        "capture_record_impl",
        T(
            "    # M-RoPE hands the layer a [3, T] position tensor; text-only input repeats",
            "    # the same row three times, which is the scalar position the index wants.",
            "    # Vision would make the axes differ, and a single scalar cannot express",
            "    # H/W — the design keeps vision out of stage 1 for exactly that reason.",
            "    if positions.ndim == 2:",
            "        if not torch.equal(positions[0], positions[1]):",
            "            _MROPE_AXES_DIFFER += 1",
            "        positions = positions[0]",
            "",
            "    span = min(_query_span(), num_tokens)",
            "    _STASH[layer_idx] = (",
            "        positions.detach().clone(),",
            "        q.detach()[-span:].clone(),",
            "        k.detach().clone(),",
            "    )",
        ),
        T(
            "    # M-RoPE hands the layer a [3, T] position tensor; text-only input repeats",
            "    # the same row three times, which is the scalar position the index wants.",
            "    # Vision would make the axes differ, and a single scalar cannot express",
            "    # H/W — the design keeps vision out of stage 1 for exactly that reason.",
            "    #",
            "    # vllm-030win step 080 capture-record fix. The canary below was written",
            "    # as ``torch.equal(positions[0], positions[1])``, and on a CUDA tensor",
            "    # torch.equal BLOCKS until the stream drains. It runs once per",
            "    # full-attention layer => 16 synchronisation points inside every",
            "    # 1456-token forward, which py-spy measured at 94.6% of the wall clock",
            "    # of a slow-state boot (94.62% of 7640 samples, leaf",
            "    # capture.py:_record_impl). Iron rule 16 (vi) already says device->host",
            "    # reads belong in drain(), not in the forward; this honours it: rows 0",
            "    # and 1 are stashed and compared where the data is already headed to",
            "    # the host, so the canary and its per-layer count survive with zero",
            "    # syncs in the forward. Gate off (default) = the old code path.",
            "    _t080 = time.monotonic()",
            "    axes = None",
            "    if positions.ndim == 2:",
            "        if _nosync():",
            "            axes = positions[:2]",
            "        else:",
            "            if not torch.equal(positions[0], positions[1]):",
            "                _MROPE_AXES_DIFFER += 1",
            "            _REC_ACC[\"sync_calls\"] += 1.0",
            "            positions = positions[0]",
            "",
            "    span = min(_query_span(), num_tokens)",
            "    _STASH[layer_idx] = (",
            "        (axes if axes is not None else positions).detach().clone(),",
            "        q.detach()[-span:].clone(),",
            "        k.detach().clone(),",
            "    )",
            "    _REC_ACC[\"seconds\"] += time.monotonic() - _t080",
            "    _REC_ACC[\"calls\"] += 1.0",
        ),
    ),
    (
        "capture",
        "capture_drain_globals",
        T(
            "    global _STASH, _LOGGED, _SKIPPED_DECODE, _SKIPPED_UNARMED",
        ),
        T(
            "    global _STASH, _LOGGED, _SKIPPED_DECODE, _SKIPPED_UNARMED",
            "    global _MROPE_AXES_DIFFER",
        ),
    ),
    (
        "capture",
        "capture_drain_axis",
        T(
            "    for layer_idx, (positions, q_tail, k) in stash.items():",
            "        _pos = positions.cpu().numpy()",
        ),
        T(
            "    for layer_idx, (positions, q_tail, k) in stash.items():",
            "        _pos = positions.cpu().numpy()",
            "        if _pos.ndim == 2:",
            "            # vllm-030win step 080 capture-record fix: the M-RoPE axis",
            "            # canary moved here from the forward, where it cost a stream",
            "            # sync per layer. Same read, same per-layer granularity, done",
            "            # with numpy on data drain() copies to the host anyway.",
            "            if bool((_pos[0] != _pos[1]).any()):",
            "                _MROPE_AXES_DIFFER += 1",
            "            _pos = _pos[0]",
        ),
    ),
    (
        "capture",
        "capture_stats",
        T(
            '        "drain_sync_seconds": round(_DRAIN_ACC["sync"], 3),',
            '        "drain_copy_seconds": round(_DRAIN_ACC["copies"], 3),',
        ),
        T(
            '        "drain_sync_seconds": round(_DRAIN_ACC["sync"], 3),',
            '        "drain_copy_seconds": round(_DRAIN_ACC["copies"], 3),',
            '        # vllm-030win step 080 capture-record fix',
            '        "record_seconds": round(_REC_ACC["seconds"], 3),',
            '        "record_calls": int(_REC_ACC["calls"]),',
            '        "record_sync_calls": int(_REC_ACC["sync_calls"]),',
            '        "record_nosync": int(_nosync()),',
        ),
    ),
    (
        "capture",
        "capture_reset",
        T(
            "    _SKIPPED_DECODE = 0",
            "    _SKIPPED_UNARMED = 0",
            "    _MROPE_AXES_DIFFER = 0",
        ),
        T(
            "    _SKIPPED_DECODE = 0",
            "    _SKIPPED_UNARMED = 0",
            "    _MROPE_AXES_DIFFER = 0",
            "    # vllm-030win step 080: the accumulators, not the gate -- the env",
            "    # does not change within a boot, and re-reading it 16x per step was",
            "    # the very cost being removed.",
            '    _REC_ACC["seconds"] = 0.0',
            '    _REC_ACC["calls"] = 0.0',
            '    _REC_ACC["sync_calls"] = 0.0',
        ),
    ),
    # ------------------------------------------------------------------ worker
    (
        "worker",
        "worker_seg_rec",
        T(
            '        "load",',
            '        "save",',
            "    )",
        ),
        T(
            '        "load",',
            '        "save",',
            '        # vllm-030win step 080: mirrored from capture.stats(), because',
            '        # record() runs inside the forward and no wait_for_save window',
            '        # can see it (that blind spot is how "outside" grew to 9.5 s).',
            '        "rec",',
            "    )",
        ),
    ),
    (
        "worker",
        "worker_mirror_rec",
        T(
            '        acc["copies"] = float(_cap0.get("drain_copy_seconds") or 0.0)',
        ),
        T(
            '        acc["copies"] = float(_cap0.get("drain_copy_seconds") or 0.0)',
            "        # vllm-030win step 080 capture-record fix: the record() segment,",
            "        # mirrored (not accumulated here) so the emit loop below prints",
            "        # it in the same shape the analyser already parses.",
            '        acc["rec"] = float(_cap0.get("record_seconds") or 0.0)',
            '        self._kvtime_n["rec"] = int(_cap0.get("record_calls") or 0)',
        ),
    ),
    (
        "worker",
        "worker_cap_part",
        T(
            '            "cap(unarmed=%d decode=%d)"',
            "            % (",
            '                cap.get("skipped_unarmed_steps") or 0,',
            '                cap.get("skipped_decode_steps") or 0,',
            "            )",
        ),
        T(
            '            "cap(unarmed=%d decode=%d rec_sync=%d rec_calls=%d nosync=%d)"',
            "            % (",
            '                cap.get("skipped_unarmed_steps") or 0,',
            '                cap.get("skipped_decode_steps") or 0,',
            '                int(cap.get("record_sync_calls") or 0),',
            '                int(cap.get("record_calls") or 0),',
            '                int(cap.get("record_nosync") or 0),',
            "            )",
        ),
    ),
    (
        "worker",
        "worker_loaded_banner",
        T(
            '                "armed=%s geometry=%s",',
        ),
        T(
            '                "armed=%s geometry=%s rec_nosync=%d",',
        ),
    ),
    (
        "worker",
        "worker_loaded_args",
        T(
            '                capture.stats().get("armed_counts"),',
            '                capture.stats().get("geometry"),',
            "            )",
        ),
        T(
            '                capture.stats().get("armed_counts"),',
            '                capture.stats().get("geometry"),',
            "                # vllm-030win step 080 capture-record fix: prove which",
            "                # record path is live, because outside-without-this is",
            "                # how the 9.5 s blind spot got read as 'not the connector'.",
            "                int(config.record_nosync()),",
            "            )",
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
    eol = "crlf" if data.count(b"\r\n") and not data.count(
        b"\n") - data.count(b"\r\n") else "lf"
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
            assert MARK in text, f"no marker landed in {path}"
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
