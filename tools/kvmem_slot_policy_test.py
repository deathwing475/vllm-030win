# -*- coding: utf-8 -*-
"""步骤 103 的 CPU 单测：轨迹槽替换语义 ``VLLM_KVMEM_SLOT_POLICY``。

跑法（纯 CPU、不起 GPU、不起引擎）::

    cd G:\\qwen3.8model
    python vllm-030win-git\\tools\\kvmem_slot_policy_test.py   # 退出码 0 = 全过

（cwd 必须不在记录仓内——环境契约：仓内 vllm/ 源码树会被 cwd 捕获；
venv python 走 venv 的 vllm（含编译产物），系统 python 回退 repo 源码树。）

它证的是 13 件事（全是"跑臂之前就能证的"，不花显存）：

1. **``slot_policy()`` 默认 legacy**：env 不设 = 引擎现状，生产零暴露。
2. **非法取值 raise 且列出合法值**（与 ``score_mode()`` 同口径）。
3. **legacy 驱逐 = FIFO-by-start**：``_evictable_victim`` 在 legacy 下返回
   最早开始的持有者，与步骤 066-102 的 ``next(iter(self._snapshot_bases))``
   逐字同语义（102 e 支的驱逐就是它干的）。
4. **lru 驱逐 = touch 最老**：装配 touch（步骤 102 e 支里 A 腿 00:37:07 用 P
   装配过）把 P 抬成"最新"，被挤的是更早停手的 flush 轨迹——e 支场景的修复
   就在这一条里。
5. **lru 的活跃保护**：本步活跃（spans/snapshot/load）、有未决快照拷贝
   event、来者自身，三者都不是驱逐候选。
6. **候选空 = 拒绝**：``_take_snapshots`` 计数 ``snapshots_refused``、不落
   环、不驱逐任何持有者（回退 C 语义，设计 §5.3）。
7. **_evict_ring legacy 不摘 authority**：102 观察到的两侧分叉在 legacy 下
   原样保留（默认行为逐字节不变的可信度）。
8. **_evict_ring lru 摘 authority**：环 + authority + touch 一起走，
   ``_authority_bytes`` 记账回收（host 内存真归还），removed 边界逐条上报。
9. **touch 在 legacy 下不记账**：``_slot_touch`` 恒空 = legacy 引擎零新增
   状态。
10. **manager 侧 removed 空键清理在位**（源码级断言）：``discard`` 后空集
    被删，``len(self._snapshots)``（asm-miss 诊断行的 ``snapshot_traj``）
    回到"活跃快照轨迹数"口径。
11. **e 支时序端到端**：同一份三轨迹快照流（P 装配 → flush 停手 → P' 到达）
    在 legacy 下驱逐 P（= 102 e 支 A2 miss 的根因链），在 lru 下驱逐
    P-flush（P 的环与 authority 都活下来 ⇒ A2 可装配）。
"""
import inspect
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path

# Append (not insert): the venv's vllm (with the compiled _C pieces) wins
# when present; a bare system python falls back to the repo source tree.
sys.path.append(str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


class FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return True


class FakeSpan:
    def __init__(self, trajectory):
        self.trajectory = trajectory


class FakeSnapshot:
    def __init__(self, trajectory, boundary):
        self.trajectory = trajectory
        self.boundary = boundary
        self.blocks = []


class FakeMetadata:
    def __init__(self, spans=(), snapshot_requests=(), load_jobs=()):
        self.spans = list(spans)
        self.snapshot_requests = list(snapshot_requests)
        self.load_jobs = list(load_jobs)


def make_worker(slot_policy):
    """A worker with just the slot-bookkeeping fields, no engine, no CUDA."""
    from vllm.v1.kvmem_workspace.worker import KVMemWorkspaceWorker

    worker = object.__new__(KVMemWorkspaceWorker)
    worker.slot_policy = slot_policy
    worker.snapshot_keep = 10
    worker.snapshot_traj = 2
    worker._snapshot_rings = {}
    worker._snapshot_bases = {}
    worker._snapshot_events = {}
    worker._removed_snapshots = []
    worker._authority = {}
    worker._authority_bytes = 0
    worker._slot_touch = {}
    worker._step_active = set()
    worker.snapshots_refused = 0
    worker.snapshots_evicted = 0
    worker.snapshots_taken = 0
    # get_finished() bookkeeping (the real engine polls it every step, which
    # is what clears the snapshot-copy events that guard inflight rows).
    worker._events = {}
    worker._completed = []
    worker._completed_snapshots = []
    worker._load_events = {}
    worker._finished_loads = set()
    return worker


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
    from vllm.v1.kvmem_workspace.worker import KVMemWorkspaceWorker

    # T1: default is legacy (engine behaviour unchanged when env is absent).
    set_env("VLLM_KVMEM_SLOT_POLICY", None)
    check("T1 slot_policy default = legacy", config.slot_policy() == "legacy")

    # T2: illegal values raise, listing the legal ones.
    set_env("VLLM_KVMEM_SLOT_POLICY", "fifo")
    try:
        config.slot_policy()
        check("T2 illegal value raises", False, "no exception")
    except ValueError as exc:
        check(
            "T2 illegal value raises",
            "legacy" in str(exc) and "lru" in str(exc),
            str(exc),
        )
    set_env("VLLM_KVMEM_SLOT_POLICY", None)

    # Patch the DMA + CUDA-event surface out (offline: no ops, no CUDA).
    KVMemWorkspaceWorker._copy = staticmethod(lambda entries: None)
    import torch

    torch.cuda.Event = FakeEvent

    P = b"\x01" * 6
    PF = b"\x02" * 6
    PB = b"\x03" * 6

    # T3: legacy pick = earliest-started holder.
    w = make_worker("legacy")
    w._snapshot_bases = {P: 0, PF: 10}
    w._snapshot_rings = {P: OrderedDict(), PF: OrderedDict()}
    w._slot_touch = {P: 999.0, PF: 1.0}
    check("T3 legacy pick = FIFO-by-start", w._evictable_victim(PB) == P)
    check("T3 legacy ignores touch", w._evictable_victim(PB) is not PF)

    # T4: lru pick = least recently touched idle holder.
    w = make_worker("lru")
    w._snapshot_bases = {P: 0, PF: 10}
    w._snapshot_rings = {P: OrderedDict(), PF: OrderedDict()}
    w._slot_touch = {P: 100.0, PF: 50.0}
    check("T4 lru pick = least recent touch", w._evictable_victim(PB) == PF)
    # ...and the assembly touch (P just matched) protects P: the e-branch fix.
    w._slot_touch[P] = 1.0
    w._slot_touch[PF] = 50.0
    check("T4 assembly touch protects P", w._evictable_victim(PB) == P)

    # T5: lru protection sets -- active / inflight-event / incoming.
    w = make_worker("lru")
    w._snapshot_bases = {P: 0}
    w._snapshot_rings = {P: OrderedDict()}
    w._slot_touch = {P: 1.0}
    w._step_active = {P}
    check("T5 active holder protected", w._evictable_victim(PB) is None)
    w._step_active = set()
    w._snapshot_events = {(P, 17088): FakeEvent()}
    check("T5 inflight event protected", w._evictable_victim(PB) is None)
    w._snapshot_events = {}
    w._snapshot_bases = {P: 0, PB: 10}
    w._snapshot_rings = {P: OrderedDict(), PB: OrderedDict()}
    check("T5 incoming never a victim", w._evictable_victim(PB) == P)

    # T6: refuse when no candidate; nothing evicted, snapshot not landed.
    # Both slots must be full for the eviction branch to run at all.
    w = make_worker("lru")
    w._snapshot_bases = {P: 0, PF: 10}
    w._snapshot_rings = {P: OrderedDict(), PF: OrderedDict()}
    w._slot_touch = {P: 1.0, PF: 50.0}
    w._step_active = {P, PF}
    w._take_snapshots([FakeSnapshot(PB, 17088)])
    check(
        "T6 refusal counted, no eviction",
        w.snapshots_refused == 1
        and PB not in w._snapshot_rings
        and set(w._snapshot_bases) == {P, PF}
        and not w._removed_snapshots,
    )

    # T7: legacy eviction leaves the authority region alone (byte-for-byte
    # the pre-103 behaviour, fork included).
    w = make_worker("legacy")
    w._snapshot_bases = {P: 0}
    w._snapshot_rings = {P: OrderedDict({17088: 0, 34176: 1})}
    w._authority = {P: {"L0": torch.zeros(4, 4)}}
    w._authority_bytes = 4 * 4 * 4  # fp32 zeros (default dtype)
    w._slot_touch = {P: 1.0}
    base = w._evict_ring(P, PB)
    check(
        "T7 legacy: ring gone, authority kept",
        base == 0
        and P not in w._snapshot_rings
        and P in w._authority
        and w._authority_bytes == 64
        and P in w._slot_touch
        and sorted(w._removed_snapshots) == [(P, 17088), (P, 34176)],
    )

    # T8: lru eviction frees authority + bookkeeping together.
    w = make_worker("lru")
    w._snapshot_bases = {P: 0}
    w._snapshot_rings = {P: OrderedDict({17088: 0})}
    w._authority = {P: {"L0": torch.zeros(8, 8), "L1": torch.zeros(4, 8)}}
    w._authority_bytes = 8 * 8 * 4 + 4 * 8 * 4  # fp32 zeros (default dtype)
    w._slot_touch = {P: 1.0}
    base = w._evict_ring(P, PB)
    expected = 8 * 8 * 4 + 4 * 8 * 4
    check(
        "T8 lru: ring+authority+touch gone, bytes returned",
        base == 0
        and P not in w._snapshot_rings
        and P not in w._authority
        and w._authority_bytes == 0
        and P not in w._slot_touch
        and w._removed_snapshots == [(P, 17088)]
        and expected == 384,
        f"authority_bytes={w._authority_bytes}",
    )

    # T9: touch is inert under legacy (zero new engine state).
    w = make_worker("legacy")
    w._touch(P)
    check("T9 legacy touch inert", w._slot_touch == {})

    # T10: manager drops emptied _snapshots keys (source-level, the manager
    # is too heavy to build offline).
    from vllm.v1.kvmem_workspace import manager as manager_mod

    src = inspect.getsource(manager_mod)
    check(
        "T10 manager empty-key cleanup in place",
        "del self._snapshots[trajectory]" in src,
    )

    # T11: the k102e timeline end-to-end. P ingests (ring+authority), PF
    # (the flush request) does the same, A assembles from P (touch), then
    # P' arrives wanting a third slot.
    def run_timeline(policy):
        w = make_worker(policy)
        # ingest P: two boundaries captured
        w._take_snapshots([FakeSnapshot(P, 17088), FakeSnapshot(P, 34176)])
        w.get_finished(set())  # the engine's per-step event poll
        # flush PF: two boundaries captured (slots now full)
        w._take_snapshots([FakeSnapshot(PF, 17088), FakeSnapshot(PF, 34176)])
        w.get_finished(set())
        # both trajectories captured raw-K rows, so both own authority
        # regions (what _authority_store would have built).
        w._authority[P] = {"L0": torch.zeros(2, 2)}
        w._authority[PF] = {"L0": torch.zeros(2, 2)}
        w._authority_bytes = 2 * 2 * 4 * 2
        # leg A assembles from P -> touch P after PF's last activity (the
        # real k102e gap was ~19 s; monotonic cannot resolve same-tick
        # touches, so stamp the assembly touch explicitly later).
        w._touch(P)
        if P in w._slot_touch and PF in w._slot_touch:
            w._slot_touch[P] = w._slot_touch[PF] + 19.0
        # leg B (PB) prefill captures its first boundary
        w._take_snapshots([FakeSnapshot(PB, 17088)])
        return w

    w_legacy = run_timeline("legacy")
    check(
        "T11 legacy: P evicted (k102e root cause reproduced)",
        P not in w_legacy._snapshot_rings
        and PB in w_legacy._snapshot_rings
        and (P, 17088) in w_legacy._removed_snapshots,
    )

    w_lru = run_timeline("lru")
    check(
        "T11 lru: PF evicted, P survives (A2 can assemble)",
        P in w_lru._snapshot_rings
        and PF not in w_lru._snapshot_rings
        and PB in w_lru._snapshot_rings
        and PF not in w_lru._authority
        and P in w_lru._authority,
    )
    # ...and P's boundaries are still resident (the A2 match precondition).
    check(
        "T11 lru: P boundaries resident",
        w_lru._snapshot_rings[P].get(17088) is not None
        and w_lru._snapshot_rings[P].get(34176) is not None,
    )

    # T12: the k103b boot crash -- capture precedes the first snapshot
    # boundary, so a full AUTHORITY table evicts first and hands the base
    # row range to the incoming trajectory; its first snapshot must claim
    # that base, not re-enter the eviction branch (assert would fire).
    w = make_worker("lru")
    w._snapshot_bases = {P: 0, PF: 10}
    w._snapshot_rings = {
        P: OrderedDict({17088: 0, 34176: 1}),
        PF: OrderedDict({17088: 10, 34176: 11}),
    }
    w._authority = {P: {"L0": torch.zeros(2, 2)},
                    PF: {"L0": torch.zeros(2, 2)}}
    w._authority_bytes = 2 * 2 * 4 * 2
    w._slot_touch = {P: 100.0, PF: 50.0}
    w._evict_ring(PF, PB)  # what _authority_region's lru branch does
    w._take_snapshots([FakeSnapshot(PB, 17088)])
    check(
        "T12 authority-evict-first: snapshot claims the handed base",
        PB in w._snapshot_rings
        and PB in w._snapshot_bases
        and w._snapshot_bases[PB] == 10
        and set(w._snapshot_bases) == {P, PB}
        and w.snapshots_refused == 0,
    )

    # T13: ghost base -- a trajectory that got its base from the authority
    # side but ended its prefill before any snapshot boundary; evicting it
    # frees the slot with no ring and nothing reported.
    w = make_worker("lru")
    w._snapshot_bases = {PF: 10, PB: 0}
    w._snapshot_rings = {PF: OrderedDict({17088: 10})}  # PB = ghost
    w._slot_touch = {PF: 1.0, PB: 2.0}
    base = w._evict_ring(PB, b"\x0c" * 6)
    check(
        "T13 ghost base evicted cleanly",
        base == 0
        and b"\x0c" * 6 in w._snapshot_bases
        and PB not in w._snapshot_bases
        and not w._removed_snapshots,
    )

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        return 1
    print("ALL PASS (13 checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
