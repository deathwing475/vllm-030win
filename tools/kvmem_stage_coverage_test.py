"""Retrieval-slot coverage test for the compressed window (step 074).

Step 073 left the out-of-window needle unreadable even though its page was
scored, selected and "baked". The bake log said ``688.4 MiB written`` for 55
slots -- which is exactly ``55 x 1,640,448 B x 8 layers``: with
``VLLM_KV_GROUP_SIZE=8`` the 16 full-attention layers sit in *two* kv cache
groups (6 and 7), and ``_emit_stage_request`` only read
``tables[group_ids[0]]``, so the second group's 8 layers were never written and
half the attention kept reading the placeholder prefill.

These cases pin the emission contract that the fix restores, on the scheduler
side, without a GPU:

* T1 every offered slot carries one physical block *per stored group*, taken
  from that group's own block-table row (the groups use different blocks for
  the same logical row -- a single-group list cannot express the slot at all).
* T2 a row that some group has not allocated truncates the offered slots, and
  no slot is ever emitted half-covered.
* T3 mixed page sizes across stored groups refuse to emit (the page table and
  the bake are keyed in one page size).
* T4 the worker-facing dataclass carries the nested shape ``_stage_in`` reads.

Usage (from the repo root, with the arm's venv):
    python tools\\kvmem_stage_coverage_test.py
"""
import sys

from vllm.v1.kvmem_workspace.manager import KVMemWorkspaceScheduler
from vllm.v1.kvmem_workspace.metadata import (
    KVMemConnectorMetadata,
    KVMemStageInRequest,
)

BLOCK_SIZE = 1424
SINK = BLOCK_SIZE
RETRIEVAL_PAGES = 55
GROUPS = [6, 7]

_results: list[str] = []
_failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        _results.append(f"  PASS  {name}")
        return
    _failures += 1
    _results.append(f"  FAIL  {name}  {detail}")


class _BlockState:
    """Stand-in for ``kv_connector_block_state``.

    ``block_ids`` is keyed by request id, and each value is the request's row
    list **per kv cache group** -- exactly what the scheduler snapshot carries.
    """

    def __init__(self, block_ids: dict):
        self.block_ids = block_ids


class _Request:
    request_id = "req-1"
    num_computed_tokens = 0
    num_tokens = SINK + RETRIEVAL_PAGES * BLOCK_SIZE
    num_prompt_tokens = num_tokens
    block_hashes: list = []
    status = "WAITING"

    @staticmethod
    def update_block_hashes() -> None:
        return None


def make_scheduler(group_ids, block_size, num_blocks_per_group: dict):
    """A scheduler object with only what ``_emit_stage_request`` reads.

    Built with ``__new__`` on purpose: the full constructor needs a live
    ``KVCacheConfig`` and a gpu block pool, and the emission logic under test
    touches none of that.
    """
    sched = object.__new__(KVMemWorkspaceScheduler)
    sched.group_ids = list(group_ids)
    sched.block_size = dict(block_size)
    sched.viewport_retrieval_pages = RETRIEVAL_PAGES
    sched.debug = False
    sched._page_table = {}
    sched.stage_ins_emitted = 0
    # The scheduler snapshot indexes block tables by group id, so the container
    # is a sequence long enough for the highest stored group. Row 0 is the shared
    # null block, and the two stored groups keep *distinct* physical ids for the
    # same logical row, exactly as the pool does.
    tables: list = [[] for _ in range(max(group_ids) + 1)]
    base = 1
    for gid in group_ids:
        rows = [0]
        for _row in range(num_blocks_per_group[gid]):
            rows.append(base)
            base += 1
        tables[gid] = rows
    return sched, tables


def emit(sched, tables, trajectory=b"\x01" * 16):
    meta = KVMemConnectorMetadata()
    plan = {
        "trajectory": trajectory,
        "sink": SINK,
        "retrieval": RETRIEVAL_PAGES * BLOCK_SIZE,
        "recent": 16384,
        "window": SINK + RETRIEVAL_PAGES * BLOCK_SIZE + 16384,
        "orig_len": 198181,
    }
    sched._emit_stage_request(
        meta,
        _Request.request_id,
        plan,
        _BlockState({_Request.request_id: tables}),
        _Request(),
    )
    return meta


def main() -> int:
    # ------------------------------------------------------------------ T1
    # Both groups hold all 56 rows (sink + 55 slots): every slot must carry a
    # pair, one block per group, and the two groups must be *different* blocks.
    sched, tables = make_scheduler(
        GROUPS, {g: BLOCK_SIZE for g in GROUPS}, {g: 60 for g in GROUPS}
    )
    meta = emit(sched, tables)
    stage = meta.stage_requests[0] if meta.stage_requests else None
    check("T1 emission exists", stage is not None, "no stage request")
    if stage is not None:
        check(
            "T1 one entry per stored group",
            all(len(entries) == len(GROUPS) for entries in stage.slots),
            str([len(e) for e in stage.slots]),
        )
        check(
            "T1 slot count = retrieval budget",
            len(stage.slots) == RETRIEVAL_PAGES,
            str(len(stage.slots)),
        )
        covered = {gid for entries in stage.slots for gid, _ in entries}
        check("T1 both groups covered", covered == set(GROUPS), str(sorted(covered)))
        g6 = [b for entries in stage.slots for gid, b in entries if gid == 6]
        g7 = [b for entries in stage.slots for gid, b in entries if gid == 7]
        check(
            "T1 each group keeps its own physical blocks",
            len(set(g6)) == RETRIEVAL_PAGES and len(set(g7)) == RETRIEVAL_PAGES
            and not (set(g6) & set(g7)),
            f"g6={g6[:3]} g7={g7[:3]}",
        )
        first_row = SINK // BLOCK_SIZE
        check(
            "T1 rows start at the sink page",
            g6[0] == tables[6][first_row] and g7[0] == tables[7][first_row],
            f"g6[0]={g6[0]} expected={tables[6][first_row]}",
        )
        check(
            "T1 slot order is row order in both groups",
            g6 == tables[6][first_row : first_row + RETRIEVAL_PAGES]
            and g7 == tables[7][first_row : first_row + RETRIEVAL_PAGES],
            "rows are not consecutive per group",
        )

    # ------------------------------------------------------------------ T2
    # Group 7 stops allocating after its 30th block (rows 1..30): the offered
    # slots must truncate there and never emit a half-covered slot.
    sched, tables = make_scheduler(
        GROUPS, {g: BLOCK_SIZE for g in GROUPS}, {6: 60, 7: 30}
    )
    meta = emit(sched, tables)
    stage = meta.stage_requests[0] if meta.stage_requests else None
    check("T2 emission exists", stage is not None, "no stage request")
    if stage is not None:
        check(
            "T2 truncates at the last row both groups hold",
            len(stage.slots) == 30,
            str(len(stage.slots)),
        )
        check(
            "T2 no half-covered slot",
            all(len(entries) == len(GROUPS) for entries in stage.slots),
            str([len(e) for e in stage.slots]),
        )

    # ------------------------------------------------------------------ T3
    # Mixed page sizes: refuse rather than half-bake.
    sched, tables = make_scheduler(
        GROUPS, {6: BLOCK_SIZE, 7: BLOCK_SIZE * 2}, {g: 60 for g in GROUPS}
    )
    meta = emit(sched, tables)
    check("T3 mixed page size emits nothing", not meta.stage_requests, str(len(meta.stage_requests)))

    # ------------------------------------------------------------------ T4
    # The worker reads ``stage.slots`` as slot -> [(group, block), ...]; pin the
    # shape so a flat list can never be passed off as coverage again.
    stage = KVMemStageInRequest(
        trajectory=b"\x02" * 16,
        request_id="req-2",
        slots=[[(6, 11), (7, 12)], [(6, 13), (7, 14)]],
        slot_start=SINK,
        page_size=BLOCK_SIZE,
        pages={3: 40},
    )
    check(
        "T4 worker-facing shape is nested per slot",
        all(isinstance(e, list) and len(e) == 2 for e in stage.slots)
        and not hasattr(stage, "blocks"),
        str(stage.slots),
    )

    print("kvmem stage-in coverage (step 074)")
    print("\n".join(_results))
    print(f"{len(_results) - _failures}/{len(_results)} passed")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
