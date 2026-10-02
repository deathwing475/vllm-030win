# -*- coding: utf-8 -*-
"""步骤 081 的 CPU 单测：``worker._stage_in`` 的取槽门控 ``VLLM_KVMEM_SLOT_PICK``。

跑法（纯 CPU、不起 GPU、不起引擎；venv python 与系统 python 都能跑）::

    cd G:\\qwen3.8model\\vllm-030win-git
    python tools/kvmem_slot_pick_test.py            # 退出码 0 = 全过

它证的是九件事（全是"跑臂之前就能证的"，不花显存）：

1. **``time`` 档与旧实现逐字相同**：同一份 ``top_pages``（分数降序，即
   ``index._summarize`` 的交出口径）驱动**真实 ``_stage_in``**，落进槽的页必须等于
   ``sorted(top_pages)[:len(slots)]``，也必须等于手算清单 ``1..52 + [86, 87, 88]``。
2. **``score`` 档取的是分数前 N，最终仍按页号升序**：槽内容 = ``sorted(top_pages[:N])``；
   针页从"时间序第 53/64（余量 2）"变成"分数序第 9 ⇒ 排版后第 48/55（余量 7）"。
3. **两档的槽数 / 每槽 ``(group, block_id)`` 覆盖 / 拷贝条数逐项相等**（仓库铁律）：
   ``VLLM_KV_GROUP_SIZE=8`` 把 16 个 full_attention 层切成组 6 与组 7，任何一槽少
   一组就是"半覆盖伪装成全成功"（073 的老坑）。这里把每次 ``dst.copy_`` 记下来逐
   槽比，不是纸面推演。
4. **默认（env 不设）= ``time``**：生产与 079/080 各 boot 的行为不破。
5. **非法取值 raise 且列出合法值**（大小写敏感，与 ``score_mode()`` 同口径）。
6. **自证日志在位**：每次取槽恰好一条 ``vllm-030win patch (step 081): slot pick=``，
   字段含 pick 模式、``len(top_pages)``、槽数、选中清单、针页在不在 selected 里。
7. **它复现 079/080 的实测倒退**：``TOPN`` 64→96 时 ``time`` 档把针页挤出槽，
   ``score`` 档免疫（落槽与 ``TOPN=64`` 逐字相同）。
8. **边界不变**：页不够槽数时两档恒等；没有检索槽时仍旧在取槽前返回。
9. **取槽路径零 device→host 同步**（源码级断言：新增段里不许出现 ``.item(`` /
   ``.cpu(`` / ``torch.equal`` / ``synchronize`` / ``.numpy()``），也不许提到
   capture 或图 —— 本改动全在连接器主机侧，CUDA graph 兼容面为零。

fixture 的来历（不是随手编的数）：针 token 125,370、臂上页长 1456 ⇒ 页 86；079 的
离线口径 ``tools/kvmem_slot_budget.py`` 实测它 ``dot@32`` 分数序第 8-11、时间序第
53/64 ⇒ 55 槽余量 2，``TOPN=96`` 直接出局。下面把这三个读数钉成断言。
"""
import inspect
import os
import re
import sys
import types

import numpy as np
import torch

# vllm 从哪儿来：优先用已装好的那份（venv，sync_venv 保证与仓内逐文件内容一致）；
# 没有就把记录仓根目录塞进 sys.path（同一份补丁代码，只是行尾可能不同）。
try:
    import vllm  # noqa: F401
    VLLM_ORIGIN = "installed (venv)"
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    VLLM_ORIGIN = "repo source tree"

from vllm.v1.kvmem_workspace import config as kvmem_config  # noqa: E402
from vllm.v1.kvmem_workspace import worker as worker_mod  # noqa: E402

W = worker_mod.KVMemWorkspaceWorker
FAILS: list[str] = []

# ---------------------------------------------------------------- fixture
NEEDLE_PAGE = 86            # 125,370 // 1,456
NEEDLE_TOKEN = 125370
BLOCK = 1456                # 页长 = 视窗一槽一页（075 之后是 1456，不是 1424）
SLOTS = 55                  # VLLM_KVMEM_VIEWPORT_PAGES
GROUPS = (6, 7)             # VLLM_KV_GROUP_SIZE=8 下的两个 full_attention 组
LAYERS_PER_GROUP = 8        # 每组 8 层 => 每槽 16 次 layer-page 拷贝

# 分数降序的 top_pages（= index._summarize 的交出形态：order = argsort(-scores)）。
# 构造成臂上读到的形状：
#   第 1-8 名 = 6 个早页 + 页 90/95（晚页）
#   第 9 名   = 针页 86
#   第 10-55 名 = 41 个早页 + 5 个晚页
#   第 56-64 名 = 5 个早页 + 4 个晚页（这 9 页是"分数最低"的那批）
TOP64 = (
    [2, 7, 15, 23, 44, 52, 90, 95]
    + [NEEDLE_PAGE]
    + [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25,
       27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 48, 49,
       50, 51, 87, 88, 92, 98, 110]
    + [9, 26, 33, 41, 47, 101, 105, 118, 125]
)
# TOPN 64 -> 96：多出的 32 名全是页号 < 86 的早页（53..84）⇒ "时间序取前 55"
# 把针页推到第 85 名，直接出局 —— 079/080 各复现过一次的那个倒退。
TOP96 = TOP64 + list(range(53, 85))

# 手算期望（写死在这里，不用被测代码算）
EXPECT_TIME64 = list(range(1, 53)) + [86, 87, 88]
EXPECT_SCORE64 = sorted(
    [p for p in range(1, 53) if p not in (9, 26, 33, 41, 47)]
    + [NEEDLE_PAGE, 87, 88, 90, 92, 95, 98, 110]
)
EXPECT_TIME96 = list(range(1, 56))          # 1..55，针页 86 不在里面
SCORE_LOSERS = [9, 26, 33, 41, 47, 101, 105, 118, 125]


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------- 驱动真实 _stage_in 的假件
class _Log:
    """换掉模块 logger（被测代码经由它打日志），把行收进列表。"""

    def __init__(self):
        self.lines: list[str] = []

    def __enter__(self):
        self._real = worker_mod.logger
        worker_mod.logger = self  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        worker_mod.logger = self._real

    def _fmt(self, msg, args):
        return str(msg % args if args else msg)

    def info(self, msg, *args):
        self.lines.append(self._fmt(msg, args))

    def warning(self, msg, *args):
        self.lines.append("WARN " + self._fmt(msg, args))

    def error(self, msg, *args):
        self.lines.append("ERROR " + self._fmt(msg, args))


class _Dst:
    """一块页视图的替身：只记录谁被写过。CPU 小张量，没有任何流可同步。"""

    def __init__(self, rec, group, layer, block_id):
        self.rec, self.group, self.layer, self.block_id = (
            rec, group, layer, block_id)

    def copy_(self, src):
        self.rec["copies"].append((self.group, self.layer, self.block_id))
        return self

    def detach(self):  # pragma: no cover - BAKE_VERIFY 关着就走不到
        raise AssertionError("the slot-pick path must not read a device tensor")


class _Views:
    def __init__(self, rec, group, layer):
        self.rec, self.group, self.layer = rec, group, layer

    def __getitem__(self, block_id):
        return _Dst(self.rec, self.group, self.layer, block_id)


def drive(top_pages, slots=SLOTS, pick=None, needle=None, drop_pages=()):
    """Run the real ``KVMemWorkspaceWorker._stage_in`` on host-side fakes.

    ``pick=None`` leaves VLLM_KVMEM_SLOT_PICK unset (= production default).
    ``drop_pages`` removes pages from the store the way a lost page table row
    would, to check the per-slot group expansion is a property of the slot plan.
    """
    if pick is None:
        os.environ.pop("VLLM_KVMEM_SLOT_PICK", None)
    else:
        os.environ["VLLM_KVMEM_SLOT_PICK"] = pick
    if needle is None:
        os.environ.pop("VLLM_KVMEM_SLOT_PICK_NEEDLE", None)
    else:
        os.environ["VLLM_KVMEM_SLOT_PICK_NEEDLE"] = str(needle)
    os.environ.pop("VLLM_KVMEM_BAKE_VERIFY", None)

    layers = {g: [f"L{g}.{i}" for i in range(LAYERS_PER_GROUP)] for g in GROUPS}
    pages = sorted(set(top_pages))
    stored = [p for p in pages if p not in set(drop_pages)]
    stage = types.SimpleNamespace(
        request_id="req-081",
        trajectory=b"\x00" * 32,
        slot_start=79776,          # 视窗头部 sink 之后的第一个检索槽
        page_size=BLOCK,
        # slot j -> 每存组一个物理块（manager 交来的形态，逐槽 x 逐组）
        slots=[[(g, 1000 + 2 * j + i) for i, g in enumerate(GROUPS)]
               for j in range(slots)],
        pages={p: i for i, p in enumerate(stored)},
    )
    report = {"top_pages": list(top_pages), "block_size": BLOCK,
              "num_pages": 138, "topn": len(top_pages)}
    rec = {"copies": [], "rows": [], "remat": []}

    host = {(g, layer): torch.zeros(len(pages) + 8, 64, dtype=torch.int8)
            for g in GROUPS for layer in layers[g]}
    views = {(g, layer): _Views(rec, g, layer)
             for g in GROUPS for layer in layers[g]}
    geom = {g: types.SimpleNamespace(num_heads=4, head_size=256, rotary_dim=64,
                                     page_bytes=64) for g in GROUPS}

    def _authority_rows(trajectory, layer_name, src_positions):
        rec["rows"].append((layer_name, int(src_positions[0]) // BLOCK))
        return np.zeros(BLOCK * 4 * 64, dtype=np.float16)

    def _rematerialize_page(rebuilt, g, raw, tokens, dst_positions, cache, **kw):
        rec["remat"].append(int(dst_positions[0]))

    s = types.SimpleNamespace(
        _geometry=geom,
        _block_size={g: BLOCK for g in GROUPS},
        _layers_per_group=layers,
        _host=host,
        _gpu_views=views,
        _rotary_embedding=lambda: types.SimpleNamespace(
            cos_sin_cache=torch.zeros(16, 4), is_neox_style=True),
        _authority_rows=_authority_rows,
        stage_in_seconds=0.0,
        stage_ins_served=0,
        stage_in_pages=0,
    )
    real_remat = worker_mod.remat
    worker_mod.remat = types.SimpleNamespace(rematerialize_page=_rematerialize_page)
    log = _Log()
    try:
        with log:
            W._stage_in(s, stage, report)
    finally:
        worker_mod.remat = real_remat

    # 内层循环按 slot 升序 -> 组 -> 层，故 rows 去重后的顺序就是 selected
    seen, chosen = set(), []
    for _layer, page in rec["rows"]:
        if page not in seen:
            seen.add(page)
            chosen.append(page)
    per_slot: dict[int, set] = {}
    for g, layer, block_id in rec["copies"]:
        per_slot.setdefault(block_id // 2, set()).add((g, layer))
    return {
        "chosen": chosen,
        "copies": rec["copies"],
        "n_copies": len(rec["copies"]),
        "coverage": sorted({(g, b) for g, _l, b in rec["copies"]}),
        "groups_written": sorted({g for g, _l, _b in rec["copies"]}),
        "per_slot": per_slot,
        "full_slots": sum(
            1 for v in per_slot.values() if len(v) == len(GROUPS) * LAYERS_PER_GROUP
        ),
        "log": log.lines,
        "bake_line": next((x for x in log.lines if "baked" in x), ""),
        "pick_lines": [x for x in log.lines
                       if "vllm-030win patch (step 081): slot pick=" in x],
    }


def bake_counts(line: str) -> tuple:
    m = re.search(r"baked (\d+) slot\(s\) x group\(s\) \[([^\]]*)\] = (\d+) "
                  r"layer-page copie", line)
    if not m:
        return ()
    return int(m.group(1)), m.group(2).replace(" ", ""), int(m.group(3))


def main() -> int:
    print(f"=== 0. 被测代码 = {VLLM_ORIGIN}: {worker_mod.__file__} ===")
    check(hasattr(kvmem_config, "slot_pick"), "config.slot_pick() 在位（补丁已 apply）")
    if not hasattr(kvmem_config, "slot_pick"):
        # 负控：补丁没 apply（或 revert 后）时必须在这里就判死，别让后面的
        # AttributeError 冒充"测试通过/测试写错了"。
        print("  ⇒ 先跑 python tools/apply_kvmem_slot_pick_step081.py apply "
              "--target both，再跑本单测")
        print("FAIL: 1 项不合格（门控不在场）")
        return 1
    check(kvmem_config.SLOT_PICKS == ("time", "score"),
          f"SLOT_PICKS == ('time', 'score')，实际 {kvmem_config.SLOT_PICKS}")

    print("=== 1. fixture 自身的读数 = 079/080 的实测口径 ===")
    check(len(TOP64) == 64 and len(set(TOP64)) == 64,
          f"TOP64 = 64 个互不相同的页（{len(TOP64)}）")
    check(TOP64.index(NEEDLE_PAGE) == 8,
          f"针页分数序第 {TOP64.index(NEEDLE_PAGE) + 1} 名（臂上 8-11）")
    check(sorted(TOP64).index(NEEDLE_PAGE) == 52,
          f"针页时间序第 {sorted(TOP64).index(NEEDLE_PAGE) + 1}/64 名 "
          f"⇒ 55 槽余量 {SLOTS - 53}")
    check(sorted(TOP96).index(NEEDLE_PAGE) == 84,
          f"TOPN 抬到 96 ⇒ 时间序第 {sorted(TOP96).index(NEEDLE_PAGE) + 1} 名 > "
          f"{SLOTS}（079/080 各复现一次的倒退）")

    print("=== 2. time 档 = 旧实现逐字相同（驱动真实 _stage_in）===")
    t64 = drive(TOP64, pick="time", needle=NEEDLE_TOKEN)
    legacy = sorted(TOP64)[:SLOTS]                # 补丁前那一行，逐字
    check(t64["chosen"] == legacy,
          "time 档落槽页序列 == 旧表达式 sorted(top_pages)[:len(slots)]")
    check(t64["chosen"] == EXPECT_TIME64,
          f"time 档 == 手算 1..52 + [86,87,88]（{len(t64['chosen'])} 槽）")
    check(t64["chosen"][52] == NEEDLE_PAGE,
          "针页落在第 53 槽（0 基 52）⇒ 余量 2，与臂上读数一致")
    check(len(t64["pick_lines"]) == 1,
          f"取槽自证日志恰好一条（{len(t64['pick_lines'])}）")
    print("    " + t64["pick_lines"][0])

    print("=== 3. score 档 = 分数前 N，再按页号升序排版 ===")
    s64 = drive(TOP64, pick="score", needle=NEEDLE_TOKEN)
    check(s64["chosen"] == sorted(TOP64[:SLOTS]),
          "score 档落槽页序列 == sorted(top_pages[:len(slots)])")
    check(s64["chosen"] == EXPECT_SCORE64,
          f"score 档 == 手算清单（{len(s64['chosen'])} 槽）")
    check(s64["chosen"] == sorted(s64["chosen"]),
          "交出去的仍是页号升序 ⇒ §5.3 '模型看到时间序窗口' 不变式不破")
    check(NEEDLE_PAGE in s64["chosen"],
          f"针页在槽内，排第 {s64['chosen'].index(NEEDLE_PAGE) + 1}/"
          f"{len(s64['chosen'])} ⇒ 余量 {SLOTS - s64['chosen'].index(NEEDLE_PAGE) - 1}")
    check(s64["chosen"].index(NEEDLE_PAGE) == 47, "针页落在第 48 槽（余量 7）")
    check(not (set(SCORE_LOSERS) & set(s64["chosen"])),
          f"出局的是分数最低那 9 页 {SCORE_LOSERS}，不再是页号最大的高分页")

    print("=== 4. 铁律：槽数 / (group, block_id) 覆盖 / 拷贝条数两档逐项相等 ===")
    check(len(t64["chosen"]) == len(s64["chosen"]) == SLOTS,
          f"槽数相等：time={len(t64['chosen'])} score={len(s64['chosen'])} == {SLOTS}")
    check(t64["coverage"] == s64["coverage"],
          f"每槽 (group, block_id) 覆盖集合逐字相等（{len(t64['coverage'])} 项）")
    check(t64["n_copies"] == s64["n_copies"]
          == SLOTS * len(GROUPS) * LAYERS_PER_GROUP,
          f"拷贝条数相等：{t64['n_copies']} == {s64['n_copies']} == "
          f"{SLOTS}x{len(GROUPS)}x{LAYERS_PER_GROUP}")
    check(t64["groups_written"] == s64["groups_written"] == list(GROUPS),
          f"两档都写到组 6 **和** 组 7（{s64['groups_written']}）—— "
          f"少一组就是半覆盖伪装成全成功")
    check(t64["full_slots"] == s64["full_slots"] == SLOTS,
          f"逐槽核对：两档都是 {SLOTS} 槽 x {len(GROUPS)} 组 x "
          f"{LAYERS_PER_GROUP} 层齐全，无一槽半覆盖")
    full_shape = frozenset((g, f"L{g}.{i}") for g in GROUPS
                           for i in range(LAYERS_PER_GROUP))
    check(len(t64["per_slot"]) == len(s64["per_slot"]) == SLOTS
          and all(frozenset(v) == full_shape
                  for v in t64["per_slot"].values())
          and all(frozenset(v) == full_shape
                  for v in s64["per_slot"].values()),
          "每一槽的覆盖形状与所选页无关（是槽计划的函数，不是页的函数）")
    check(len({b for _g, _l, b in s64["copies"]}) == SLOTS * len(GROUPS),
          f"score 档同样铺满 {SLOTS}x{len(GROUPS)} 个物理块")
    check(set(t64["chosen"]) != set(s64["chosen"]),
          "变的只有**页**：两档选中的页集合确实不同（这才是修法）")
    check(bake_counts(t64["bake_line"]) == bake_counts(s64["bake_line"])
          == (SLOTS, "6,7", SLOTS * len(GROUPS) * LAYERS_PER_GROUP),
          f"烘焙台账两档同形：{bake_counts(s64['bake_line'])}")

    print("=== 5. 默认（env 不设）= time，生产路径不动 ===")
    os.environ.pop("VLLM_KVMEM_SLOT_PICK", None)
    check(kvmem_config.slot_pick() == "time",
          f"不设 VLLM_KVMEM_SLOT_PICK ⇒ slot_pick() == 'time'，"
          f"实际 {kvmem_config.slot_pick()!r}")
    d64 = drive(TOP64, pick=None, needle=None)
    check(d64["chosen"] == t64["chosen"],
          "env 不设时真实 _stage_in 的落槽页与显式 time 档逐字相同")
    check(d64["coverage"] == t64["coverage"]
          and d64["n_copies"] == t64["n_copies"],
          "env 不设时覆盖与拷贝条数也与 time 档相同")
    check("pick=time" in d64["pick_lines"][0],
          "默认档的自证日志照打（门控日志必须能自证在跑）")
    check("needle=n/a" in d64["pick_lines"][0] and "in_selected=-" in d64["pick_lines"][0],
          "不设针 token ⇒ needle=n/a / in_selected=-，选择逻辑一字未改")

    print("=== 6. 非法取值 raise 且列出合法值（大小写敏感，同 score_mode 口径）===")
    for bad in ("latest", "SCORE", "1", "tim", "Score", "time,score"):
        os.environ["VLLM_KVMEM_SLOT_PICK"] = bad
        try:
            got = kvmem_config.slot_pick()
            check(False, f"非法值 {bad!r} 竟然没 raise（返回 {got!r}）")
        except ValueError as exc:
            msg = str(exc)
            check("VLLM_KVMEM_SLOT_PICK" in msg and "time" in msg and "score" in msg,
                  f"非法值 {bad!r} raise 并列出合法值：{msg}")
    os.environ["VLLM_KVMEM_SLOT_PICK"] = "  score  "
    check(kvmem_config.slot_pick() == "score", "两侧空白按既有 _env_* 口径 strip")
    os.environ["VLLM_KVMEM_SLOT_PICK"] = ""
    check(kvmem_config.slot_pick() == "time", "空串 = 未设 ⇒ 默认 time")
    os.environ.pop("VLLM_KVMEM_SLOT_PICK", None)
    os.environ["VLLM_KVMEM_SLOT_PICK_NEEDLE"] = "-1"
    try:
        kvmem_config.slot_pick_needle_token()
        check(False, "负针 token 竟然没 raise")
    except ValueError as exc:
        check("VLLM_KVMEM_SLOT_PICK_NEEDLE" in str(exc),
              f"负针 token raise：{exc}")
    os.environ.pop("VLLM_KVMEM_SLOT_PICK_NEEDLE", None)

    print("=== 7. 自证日志：TOPN 64->96 时 time 丢针、score 免疫 ===")
    t96 = drive(TOP96, pick="time", needle=NEEDLE_TOKEN)
    s96 = drive(TOP96, pick="score", needle=NEEDLE_TOKEN)
    check(t96["chosen"] == EXPECT_TIME96,
          f"time@96 落槽 = 手算 1..55（页号最小的 {SLOTS} 个）")
    check(NEEDLE_PAGE not in t96["chosen"],
          "time@96 把针页挤出槽 —— 复现 079/080 的那个倒退")
    check("in_selected=False" in t96["pick_lines"][0]
          and "page 86 (token 125370)" in t96["pick_lines"][0],
          "日志在针页出局时自证 in_selected=False")
    check(s96["chosen"] == s64["chosen"],
          "score@96 与 score@64 落槽逐字相同 ⇒ 取槽不再被抬高 TOPN 反噬")
    check("in_selected=True" in s96["pick_lines"][0],
          "日志自证 pick=score 下针页在槽内")
    for label, line in (("time@96", t96["pick_lines"][0]),
                        ("score@96", s96["pick_lines"][0])):
        check(f"pick=" in line and "top_pages=96" in line and "slots=55" in line
              and "selected=55" in line,
              f"{label} 字段齐全（pick/top_pages/slots/selected）")
    check(t96["n_copies"] == s96["n_copies"] == t64["n_copies"],
          f"96 页时两档拷贝条数仍相等（{s96['n_copies']}）⇒ 只换页不换形状")
    print("    " + s96["pick_lines"][0][:240])

    print("=== 8. 边界：页少于槽 / 无检索槽 / 页表丢页 ===")
    few_score = drive([5, 1, 3], slots=SLOTS, pick="score")
    few_time = drive([5, 1, 3], slots=SLOTS, pick="time")
    check(few_score["chosen"] == few_time["chosen"] == [1, 3, 5],
          f"页不够时两档逐字相同（{few_score['chosen']}）⇒ 短请求不受影响")
    check(few_score["n_copies"] == few_time["n_copies"]
          == 3 * len(GROUPS) * LAYERS_PER_GROUP,
          f"页不够时拷贝条数也相同（{few_score['n_copies']}）")
    zero = drive(TOP64, slots=0, pick="score")
    check(zero["chosen"] == [] and zero["pick_lines"] == []
          and zero["n_copies"] == 0,
          "没有检索槽 ⇒ 仍在取槽前返回（不打日志、不拷贝，行为不变）")
    miss = drive(TOP64, pick="score", drop_pages=(NEEDLE_PAGE,))
    check(miss["chosen"] == [p for p in s64["chosen"] if p != NEEDLE_PAGE],
          "页表丢页时该槽仍被跳过（missing_pages 台账在旧行里，未被本改动掩盖）")
    check(miss["full_slots"] == len(miss["per_slot"]) == SLOTS - 1,
          f"丢一页只少一槽：其余 {SLOTS - 1} 槽仍是整组覆盖"
          f"（半覆盖不会伪装成全成功）")
    check(miss["n_copies"] == (SLOTS - 1) * len(GROUPS) * LAYERS_PER_GROUP,
          f"丢一页的拷贝条数 = {miss['n_copies']}（差额恰是一槽的 16 层）")

    print("=== 9. 取槽路径零 device->host 同步（源码级断言）===")
    src = inspect.getsource(W._stage_in)
    start = src.index("pick = config.slot_pick()")
    end = src.index("started = time.monotonic()", start)
    added = src[start:end]
    for forbidden in (".item(", ".cpu(", "torch.equal", "synchronize", ".numpy()",
                      "to(device"):
        check(forbidden not in added,
              f"取槽+日志段（{len(added.splitlines())} 行）里没有 {forbidden!r}")
    check("capture" not in added and "graph" not in added,
          "取槽段不引用 capture、不提图 ⇒ 本改动全在主机侧，图面为零")
    check("stage.slots" in added and "top_pages" in added,
          "取槽段只读主机侧 Python 结构（stage.slots / top_pages / report）")
    bake = src[end:]
    check("enumerate(stage.slots)" in bake and "selected[slot_j]" in bake,
          "烘焙循环仍只按 enumerate(stage.slots) 展开，selected 只喂它")
    check(bake.count("self.group_ids") + bake.count("_layers_per_group") >= 1,
          "逐槽 x 逐组的展开方式没被本改动动过")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项不合格")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("PASS: time 档逐字等于旧实现；score 档按分数取满槽再排页号序；"
          "两档槽数/覆盖/拷贝条数逐项相等；默认关；非法值 raise；取槽零同步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
