# -*- coding: utf-8 -*-
"""步骤 104 的 CPU 单测：守卫 (a) 交插 ``VLLM_KVMEM_ASM_INTERLEAVE``。

跑法（纯 CPU、不起 GPU、不起引擎）::

    cd G:\\qwen3.8model
    python vllm-030win-git\\tools\\kvmem_asm_interleave_test.py  # 退出码 0 = 全过

（cwd 必须不在记录仓内——环境契约：仓内 vllm/ 源码树会被 cwd 捕获；
venv python 走 venv 的 vllm（含编译产物），系统 python 回退 repo 源码树。）

它证的是 15 件事（全是"跑臂之前就能证的"，不花显存）：

1.  **``asm_interleave()`` 默认关**：env 不设 = 守卫 (a) 逐字节，生产零暴露。
2.  **env=1 开**。
3.  **关态守卫 (a) 逐字节**：num_computed>0 一律 0（asm-miss local-prefix-hit）。
4.  **开态 + 命中在快照边界之间**：run 扫描从命中页起、候选 = 严格深于命中的
    快照边界、boundary = 其中最大者（101 现场的修复语义）。
5.  **返回值 = 增量**：``get_num_new_matched_tokens`` 交插时返回
    ``boundary - hit``，且写入 ``_assembly_plan``（全局边界 + 起始页）。
6.  **开态 + 命中深于全部快照边界**：候选空 → 0（native 已盖过装配边界，
    装配无增益；引擎自然走本地命中继续）。
7.  **开态 + 页表在命中后断裂**：run 在断点封顶（与现状 cap 语义同型）。
8.  **开态 + 哈希 mismatch 只校验装配段**：[0, start_page) 的哈希不参与
    （那段由 native 哈希链负责），mismatch 把 boundary 封顶到坏页。
9.  **开态 + hit=0 退化 = 现状路径**：start_page=0 时与 103 及以前逐字同值。
10. **update_state_after_alloc 合并 plan**：交插增量进来后 pending =
    (blocks, 全局 boundary, traj, start_page)。
11. **update_state_after_alloc legacy 形状不变**：无 plan 键 = (0, ext)，
    boundary = ext 本身（现状 3 元组语义的 4 元组等价形式）。
12. **ext=0 不记 pending**（调度器 partial_tail 砍掉外部的场景）：
    plan 残键无害，由 finish 清理（T14）。
13. **_emit_load_jobs 偏移**：页循环 ``range(start_page, num_pages)``、
    mamba 快照 position = num_pages-1（全局边界）、job.num_tokens = 全局
    boundary（快照行寻址不变）。
14. **_emit_load_jobs legacy 逐字**：start_page=0 与 103 及以前同形。
15. **清理点在位**（源码级断言，manager 太重无法离线实例化）：finish pop
    plan 键、reset clear plan。
"""
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Append (not insert): the venv's vllm (with the compiled _C pieces) wins
# when present; a bare system python falls back to the repo source tree.
sys.path.append(str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

BS = 1424          # the arm's page size (tokens per page)
SW = 131072        # the arm's sliding window (tokens)
PAGES = 92         # evicted-page run the ingest stored ([0, 92))
SNAP_EVERY = 12    # snapshot boundaries every 12 pages


class FakeRequest:
    """Just what _assembly_match / _debug_window(off) / trajectory_key touch."""

    def __init__(self, token_ids, req_id="r1"):
        self.request_id = req_id
        self.prompt_token_ids = list(token_ids)
        self.all_token_ids = list(token_ids)
        self.num_computed_tokens = 0
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.block_hashes = None
        self._output_token_ids = ()
        self.status = SimpleNamespace(name="WAITING")


class FakeSpec:
    sliding_window = SW


class FakeGroup:
    kv_cache_spec = FakeSpec()


class FakeKVConfig:
    kv_cache_groups = [FakeGroup()]


class FakeBlocks:
    def __init__(self, per_group):
        self.blocks = per_group


def make_manager(interleave):
    """A manager with just the assembly fields -- no engine, no CUDA.

    ``_page_token_hash`` and ``trajectory_key`` are the real (pure blake2b)
    methods; everything else is the minimal bookkeeping the assembly path
    reads.
    """
    from vllm.v1.kvmem_workspace.manager import KVMemWorkspaceScheduler

    mgr = object.__new__(KVMemWorkspaceScheduler)
    mgr.asm_interleave = interleave
    mgr.load_enabled = True
    mgr.viewport_enabled = False
    mgr.debug = False
    mgr.trajectory_prefix_tokens = 512
    mgr.group_ids = [0]
    mgr.mamba_group_ids = [1]
    mgr.block_size = {0: BS, 1: BS}
    mgr.kv_cache_config = FakeKVConfig()
    mgr._page_table = {}
    mgr._page_hashes = {}
    mgr._snapshots = {}
    mgr._req_viewport = {}
    mgr._req_trajectory = {}
    mgr._req_object = {}
    mgr._req_prompt_len = {}
    mgr._req_baseline = {}
    mgr._pending_loads = {}
    mgr._assembly_plan = {}
    mgr._next_job_id = 0
    mgr.loads_requested = 0
    mgr.pages_evicted = 0
    mgr.pages_stored = 0
    mgr.pages_dropped = 0
    return mgr


def fill_pages(mgr, token_ids, first=0, last=PAGES, bad_hash_pages=()):
    """Page table + hashes for ``[first, last)`` of this prompt's trajectory.

    ``bad_hash_pages`` get a deliberately wrong hash (mismatch probes).
    """
    traj = mgr.trajectory_key(token_ids, mgr.trajectory_prefix_tokens)
    for page in range(first, last):
        offset = page * BS
        mgr._page_table[(traj, offset)] = page  # slot value, never read here
        if page in bad_hash_pages:
            mgr._page_hashes[(traj, offset)] = b"\xde\xad\xbe\xef"
        else:
            mgr._page_hashes[(traj, offset)] = mgr._page_token_hash(
                token_ids, offset, BS
            )
    return traj


def fill_snapshots(mgr, traj, max_page):
    boundaries = {
        p * BS for p in range(SNAP_EVERY, max_page + 1, SNAP_EVERY)
        if p * BS <= max_page * BS
    }
    mgr._snapshots[traj] = set(boundaries)
    return boundaries


def set_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main():
    from vllm.v1.kvmem_workspace import config
    from vllm.v1.kvmem_workspace.manager import KVMemWorkspaceScheduler

    # T1: default off (guard (a) stands, zero production exposure).
    set_env("VLLM_KVMEM_ASM_INTERLEAVE", None)
    check("T1 asm_interleave default = off", config.asm_interleave() is False)

    # T2: env=1 turns it on.
    set_env("VLLM_KVMEM_ASM_INTERLEAVE", "1")
    ok = config.asm_interleave()
    set_env("VLLM_KVMEM_ASM_INTERLEAVE", None)
    check("T2 asm_interleave env=1", ok is True)

    # The prompt: 200,768 tokens (ingest 200,000 + the 768-token tail), the
    # 102/103 probe shape. evict_edge = 200,768 - 131,072 = 69,696 -> the run
    # condition caps the scan at page 48.
    n_prompt = 200768
    token_ids = list(range(1000, 1000 + n_prompt))

    # T3: guard (a) byte-for-byte when off.
    mgr = make_manager(interleave=False)
    fill_pages(mgr, token_ids)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    check(
        "T3 off: local hit vetoes the assembly",
        mgr._assembly_match(req, 34176) == 0,
    )

    # T4: on, hit between snapshot boundaries -> the deepest boundary past
    # the hit. Hit = 24 pages (34,176 tokens); snapshots {12..48} pages; run
    # [24, 48) pages -> boundary = 48 pages = 68,352.
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    traj = fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    boundary = mgr._assembly_match(req, 24 * BS)
    check(
        "T4 on: boundary = deepest snapshot past the hit",
        boundary == 48 * BS,
        f"got {boundary}",
    )

    # T5: get_num_new_matched_tokens returns the increment and records the
    # plan (global boundary + start page).
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    ext, load = mgr.get_num_new_matched_tokens(req, 24 * BS)
    plan = mgr._assembly_plan.get(req.request_id)
    check(
        "T5 increment + plan recorded",
        ext == (48 * BS - 24 * BS) and load is True
        and plan == (24, 48 * BS),
        f"ext={ext} plan={plan}",
    )

    # T6: on, hit deeper than every snapshot boundary -> no candidate, 0.
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    check(
        "T6 on: hit past all boundaries = no gain, 0",
        mgr._assembly_match(req, 49 * BS) == 0,
    )

    # T7: on, the page table breaks after the hit -> run capped at the gap.
    # Pages [0, 50) + [60, 92) stored, hit = 40 pages -> run [40, 50) -> the
    # deepest snapshot inside is 48 pages.
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids, first=0, last=50)
    fill_pages(mgr, token_ids, first=60, last=PAGES)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    check(
        "T7 on: page-table gap caps the run",
        mgr._assembly_match(req, 40 * BS) == 48 * BS,
    )

    # T8: on, hash mismatch inside the assembled segment only. Pages [0, 24)
    # carry junk hashes (the native hit's segment -- must NOT be consulted);
    # page 30 is corrupted -> boundary capped at 30 pages.
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids, bad_hash_pages=set(range(0, 24)) | {30})
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    check(
        "T8 on: hash check covers the assembled segment only",
        mgr._assembly_match(req, 24 * BS) == 30 * BS,
    )

    # T9: on, hit = 0 degrades to the exact pre-104 path (num_computed==0
    # never took the guard branch; the scan/candidate/hash code is shared).
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    check(
        "T9 on: hit=0 degrades to the legacy boundary",
        mgr._assembly_match(req, 0) == 48 * BS,
    )

    # T10: update_state_after_alloc merges the plan into the pending load
    # (blocks, GLOBAL boundary, trajectory, start_page).
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    fill_snapshots(mgr, mgr.trajectory_key(token_ids, 512), 48)
    req = FakeRequest(token_ids)
    ext, _ = mgr.get_num_new_matched_tokens(req, 24 * BS)
    blocks = FakeBlocks([[f"att{i}" for i in range(48)], [f"mam{i}" for i in range(48)]])
    mgr.update_state_after_alloc(req, blocks, ext)
    pending = mgr._pending_loads.get(req.request_id)
    traj = mgr.trajectory_key(token_ids, 512)
    check(
        "T10 pending = (blocks, global boundary, traj, start_page)",
        pending is not None
        and pending[1] == 48 * BS and pending[2] == traj and pending[3] == 24,
        f"pending={pending!r}",
    )

    # T11: legacy shape (no plan key) -- boundary = the external count itself,
    # start_page 0. This is what the pre-104 callers produce. The request is
    # NOT yet in _req_trajectory (first alloc), so update takes the
    # full-bookkeeping path, exactly like the engine's first scheduling.
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    req = FakeRequest(token_ids)
    blocks = FakeBlocks([[f"att{i}" for i in range(36)], [f"mam{i}" for i in range(36)]])
    mgr.update_state_after_alloc(req, blocks, 36 * BS)
    pending = mgr._pending_loads.get(req.request_id)
    check(
        "T11 legacy pending unchanged",
        pending is not None and pending[1] == 36 * BS and pending[3] == 0,
        f"pending={pending!r}",
    )

    # T12: ext = 0 (the scheduler's partial_tail reconciliation dropped the
    # external load) -> nothing pending; the stale plan key is inert and
    # cleared at finish (T15).
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    req = FakeRequest(token_ids)
    mgr._assembly_plan[req.request_id] = (24, 48 * BS)
    mgr._req_trajectory[req.request_id] = mgr.trajectory_key(token_ids, 512)
    mgr.update_state_after_alloc(req, FakeBlocks([[], []]), 0)
    check(
        "T12 ext=0 records nothing",
        req.request_id not in mgr._pending_loads,
    )

    # T13: _emit_load_jobs writes pages starting at start_page and anchors
    # the mamba snapshot at num_pages-1 (global boundary addressing).
    mgr = make_manager(interleave=True)
    fill_pages(mgr, token_ids)
    traj = mgr.trajectory_key(token_ids, 512)
    att_blocks = [SimpleNamespace(is_null=False, block_id=100 + i) for i in range(48)]
    mam_blocks = [SimpleNamespace(is_null=True, block_id=-1) for _ in range(47)]
    mam_blocks.append(SimpleNamespace(is_null=False, block_id=999))
    mgr._pending_loads["r1"] = (
        FakeBlocks([att_blocks, mam_blocks]), 48 * BS, traj, 24,
    )
    mgr._page_table.clear()
    for page in range(PAGES):
        mgr._page_table[(traj, page * BS)] = 5000 + page
    meta = SimpleNamespace(load_jobs=[])
    mgr._emit_load_jobs(meta)
    check(
        "T13 one job emitted",
        len(meta.load_jobs) == 1 and mgr.loads_requested == 1,
        f"jobs={len(meta.load_jobs)}",
    )
    if meta.load_jobs:
        job = meta.load_jobs[0]
        pages = job.pages
        att_pages = [p for p in pages if p.group_id == 0]
        mam_pages = [p for p in pages if p.group_id == 1]
        check(
            "T13 pages run [start_page, num_pages)",
            [p.page_index for p in att_pages] == list(range(24, 48))
            and att_pages[0].block_id == 100 + 24
            and len(att_pages) == 24,
            f"first={att_pages[0].page_index if att_pages else None} "
            f"n={len(att_pages)}",
        )
        check(
            "T13 mamba snapshot at num_pages-1, job boundary = global",
            len(mam_pages) == 0
            and job.mamba_snapshots == [(1, 999, 48 * BS)]
            and job.num_tokens == 48 * BS,
            f"snapshots={job.mamba_snapshots}",
        )

    # T14: _emit_load_jobs with start_page 0 = the pre-104 shape verbatim
    # (pages 0..E-1, mamba at E-1).
    mgr = make_manager(interleave=False)
    fill_pages(mgr, token_ids)
    traj = mgr.trajectory_key(token_ids, 512)
    att_blocks = [SimpleNamespace(is_null=False, block_id=200 + i) for i in range(36)]
    mam_blocks = [SimpleNamespace(is_null=True, block_id=-1) for _ in range(35)]
    mam_blocks.append(SimpleNamespace(is_null=False, block_id=888))
    mgr._pending_loads["r2"] = (
        FakeBlocks([att_blocks, mam_blocks]), 36 * BS, traj, 0,
    )
    mgr._page_table.clear()
    for page in range(PAGES):
        mgr._page_table[(traj, page * BS)] = 6000 + page
    meta = SimpleNamespace(load_jobs=[])
    mgr._emit_load_jobs(meta)
    ok = False
    if len(meta.load_jobs) == 1:
        job = meta.load_jobs[0]
        att_pages = [p for p in job.pages if p.group_id == 0]
        ok = (
            [p.page_index for p in att_pages] == list(range(36))
            and job.mamba_snapshots == [(1, 888, 36 * BS)]
            and job.num_tokens == 36 * BS
        )
    check("T14 legacy emit shape verbatim", ok)

    # T15: cleanup points in place (the manager is too heavy to build
    # offline -- source-level assertions, same style as 103's T10).
    src = inspect.getsource(KVMemWorkspaceScheduler)
    check(
        "T15 finish pops the plan key",
        "_assembly_plan.pop(request.request_id, None)" in src,
    )
    check(
        "T15 reset clears the plan",
        "_assembly_plan.clear()" in src,
    )

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        return 1
    print("ALL PASS (15 checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
