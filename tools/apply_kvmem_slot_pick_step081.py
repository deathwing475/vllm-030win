# -*- coding: utf-8 -*-
"""步骤 081：``_stage_in`` 的取槽规则加一个门控 ``VLLM_KVMEM_SLOT_PICK``。

背景（两次实测复现，不是推演）
------------------------------
视窗臂把超长 prompt 改写到固定槽位压缩视窗上：视窗 = 头部 sink + ``N`` 个检索槽
（``VLLM_KVMEM_VIEWPORT_PAGES``，臂上 55）+ 尾部 recent，检索槽每槽烘焙一个"被打
高分页"的 KV。打分模块 ``index._summarize`` 交出的 ``top_pages`` **本身已是分数降
序**（``order = np.argsort(-scores)`` 取前 ``VLLM_KVMEM_TOPN``），而 ``worker.py``
的取槽行是

    selected = sorted(top_pages)[: len(stage.slots)]

``sorted()`` 把分数序**整个丢掉**，于是真实规则是"分数前 64 名里页号最小的 55
个"——被挤掉的恰好是**页号最大的高分页**。079/080 的离线口径（``tools/kvmem_slot_
budget.py``）实测：针页（token 125,370，页长 1456 ⇒ 页 86）在 ``dot@32`` 口径下分
数序第 8-11 名，但时间序排第 **53/64** ⇒ 55 槽真实余量只剩 **2**；把 ``TOPN`` 从
64 抬到 96 更是把它直接挤出槽（079、080 各复现一次）。行 969-970 的注释"the
highest-scoring pages win the earliest slots"与代码不符：它**从来没发生过**。

本补丁做什么
------------
门控 ``VLLM_KVMEM_SLOT_PICK``（读取函数 ``config.slot_pick()``）：

* ``time``（**默认**）＝ 现行为逐字不动，生产与 079/080 各 boot 的可比性不破；
* ``score``＝ 先按分数序取满槽（``top_pages[:len(slots)]``），再把中选页**按页号升
  序**排版送去烘焙 ⇒ 设计 §5.3 的不变式（模型看到的是时间序窗口）两种取值下都成
  立，变的只有"哪些页拿到槽"。

取槽行之后加**一行自证 INFO**（前缀 ``vllm-030win patch (step 081):``）：pick 模
式、``len(top_pages)``、槽数、选中的页清单、以及针页在不在 selected 里（针的 token
由只读日志旋钮 ``VLLM_KVMEM_SLOT_PICK_NEEDLE`` 给出，不设则打 ``n/a``，选择逻辑一
个字都不受）。``_stage_in`` 是 ``wait_for_save`` 里的主机侧纯 Python，日志只读
dict / list，**没有任何新增 device→host 读取**（``.item()``/``.cpu()``/
``torch.equal`` 一个都不加；单测里有源码级断言钉住这条）。

纪律
----
* 槽数 / 每槽 ``(group, block_id)`` 覆盖 / 发出的拷贝条数**与选哪些页无关**：
  ``selected`` 只喂给 ``enumerate(stage.slots)``，逐槽仍按 ``self._layers_per_group``
  展开每个组（``VLLM_KV_GROUP_SIZE=8`` 下 16 个 full_attention 层切成组 6/组 7，
  少一组就是"半覆盖伪装成全成功"，073 的老坑）。该不变量由
  ``tools/kvmem_slot_pick_test.py`` **驱动真实 ``_stage_in``**（CPU、假 self、假
  remat）在两档门控下逐项核对，不是纸面推演。
* 默认关 ⇒ 生产不动（生产压根不设 ``VLLM_KVMEM_*``，连接器不在场）。
* 必守 14：``apply|revert|status`` 三态 + 每条锚点 ``assert count==1`` + 写前
  ``compile()`` + revert 后 ``assert MARK not in text``，**apply/revert 往返实测**。
* 不碰 ``capture.py``、不碰图：本改动全在连接器主机侧，CUDA graph 兼容面为零。

用法
----
    python tools/apply_kvmem_slot_pick_step081.py status --target both
    python tools/apply_kvmem_slot_pick_step081.py apply  --target both
    python tools/apply_kvmem_slot_pick_step081.py revert --target both
    python tools/sync_venv.py                 # 2734 全一致
    python tools/kvmem_slot_pick_test.py      # 跑臂前的必需闸门
"""
import argparse
import os
import sys

REPO_ROOT = r"G:\qwen3.8model\vllm-030win-git\vllm"
VENV_ROOT = r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm"
FILES = {
    "config": r"v1\kvmem_workspace\config.py",
    "worker": r"v1\kvmem_workspace\worker.py",
}
MARK = "step 081 slot-pick fix"


def T(*lines: str) -> str:
    return "\n".join(lines)


# (file_key, name, old, new) — LF 书写，读写按目标文件自身行尾转换。
EDITS = [
    # ------------------------------------------------------------------ config
    (
        "config",
        "config_slot_pick",
        T(
            "def timing_enabled() -> bool:",
        ),
        T(
            'SLOT_PICKS = ("time", "score")',
            "",
            "",
            "def slot_pick() -> str:",
            '    """vllm-030win step 081 slot-pick fix: which pages win the',
            "    retrieval slots.",
            "",
            "    ``worker._stage_in`` filled the slots with ``sorted(top_pages)",
            "    [:n]``, and ``sorted`` throws away the score order that",
            "    ``index._summarize`` hands back (``top_pages`` arrives already",
            "    sorted by score, descending). The rule that actually ran was",
            "    therefore *the lowest-numbered pages among the top",
            "    ``VLLM_KVMEM_TOPN``*, which drops precisely the LATE high-scoring",
            "    pages -- measured twice on the viewport arm: the needle at token",
            "    125,370 (page 86 at block 1456) sits at score rank 8-11 but at",
            "    time rank 53/64, so 55 slots left a slack of 2, and raising TOPN",
            "    64 -> 96 pushed it out of the slots altogether.",
            "",
            "    ``time`` is that code path, byte for byte, so production and the",
            "    079/080 boots stay comparable. ``score`` takes the slots by score",
            "    and then lays the winners out in page order, so the design's",
            "    invariant -- the model sees a time-ordered window (design",
            "    section 5.3) -- holds under both values, as does the per-slot",
            "    group expansion: only *which* pages win changes.",
            "",
            "    Read once per stage-in (not per layer), so no need for the",
            "    hot-path caching ``capture`` uses. Revert with",
            "    tools/apply_kvmem_slot_pick_step081.py revert.",
            '    """',
            '    raw = os.environ.get("VLLM_KVMEM_SLOT_PICK", "").strip()',
            "    if not raw:",
            '        raw = "time"',
            "    if raw not in SLOT_PICKS:",
            "        raise ValueError(",
            "            f\"VLLM_KVMEM_SLOT_PICK must be one of {SLOT_PICKS}, \"",
            "            f\"got {raw!r}\"",
            "        )",
            "    return raw",
            "",
            "",
            "def slot_pick_needle_token() -> int | None:",
            '    """Needle token the step 081 pick line reports membership for.',
            "",
            "    LOG ONLY -- nothing here can change which pages are picked, and",
            "    unset (the default) only costs the ``needle=n/a`` field. The",
            "    probe prints the needle's absolute token index (``needle at token",
            "    125370``); dividing it by the score report's ``block_size`` gives",
            "    the page, which is what ``tools/kvmem_slot_budget.py`` calls",
            "    ``needle_page``.",
            '    """',
            '    raw = os.environ.get("VLLM_KVMEM_SLOT_PICK_NEEDLE", "").strip()',
            "    if not raw:",
            "        return None",
            "    try:",
            "        value = int(raw)",
            "    except ValueError:",
            "        raise ValueError(",
            "            f\"VLLM_KVMEM_SLOT_PICK_NEEDLE must be a token index, got \"",
            "            f\"{raw!r}\"",
            "        ) from None",
            "    if value < 0:",
            "        raise ValueError(",
            "            f\"VLLM_KVMEM_SLOT_PICK_NEEDLE must not be negative, got \"",
            "            f\"{value}\"",
            "        )",
            "    return value",
            "",
            "",
            "def timing_enabled() -> bool:",
        ),
    ),
    # ------------------------------------------------------------------ worker
    (
        "worker",
        "worker_stage_in_pick",
        T(
            "        # Slots fill in time order (design \u00a75.3: the model sees a time-ordered",
            "        # window); the highest-scoring pages win the earliest slots.",
            "        selected = sorted(top_pages)[: len(stage.slots)]",
        ),
        T(
            "        # vllm-030win step 081 slot-pick fix, gated by",
            "        # VLLM_KVMEM_SLOT_PICK. ``time`` (the default) is exactly the",
            "        # line this replaces:",
            "        #",
            "        #     # Slots fill in time order (design \u00a75.3: the model sees a",
            "        #     # time-ordered window); the highest-scoring pages win the",
            "        #     # earliest slots.",
            "        #     selected = sorted(top_pages)[: len(stage.slots)]",
            "        #",
            "        # whose second clause never held: ``sorted`` discards the score",
            "        # order ``index._summarize`` returns ``top_pages`` in, so the",
            "        # rule that ran was \"the lowest-numbered pages of the top",
            "        # VLLM_KVMEM_TOPN\", which drops precisely the LATE",
            "        # high-scoring pages. Needle at token 125,370 (page 86 at",
            "        # block 1456): score rank 8-11, time rank 53/64 -- 2 slots of",
            "        # slack, and TOPN 64 -> 96 pushed it out (079 and 080 each",
            "        # reproduced it). ``score`` fills the slots by score and then",
            "        # re-sorts the winners into page order, so the model still",
            "        # sees a time-ordered window (design \u00a75.3) and only *which",
            "        # pages win* changes.",
            "        #",
            "        # What does NOT change with the pick, and why the bake stays",
            "        # whole: ``selected`` feeds nothing but",
            "        # ``enumerate(stage.slots)`` below, so the number of slots",
            "        # filled, each slot's (group, block_id) coverage and the",
            "        # resulting layer-page copy count are functions of the slot",
            "        # plan, not of the pages -- ``VLLM_KV_GROUP_SIZE=8`` splits the",
            "        # 16 full-attention layers over group 6 and group 7, and every",
            "        # slot still bakes both (073: a half-covered slot reads as",
            "        # half-rebuilt KV). tools/kvmem_slot_pick_test.py drives this",
            "        # very method under both gate values and checks that",
            "        # equivalence item by item.",
            "        pick = config.slot_pick()",
            "        selected = (",
            "            sorted(top_pages)",
            "            if pick == \"time\"",
            "            else sorted(top_pages[: len(stage.slots)])",
            "        )[: len(stage.slots)]",
            "        # One-shot self-proof of which rule ran, in the step's own",
            "        # prefix so a boot log can be read without the arm script. It",
            "        # reads host-side structures only: no tensor is pulled back to",
            "        # the host here, and the bake below stays exactly as it was",
            "        # (step 080's discipline still holds -- those reads live in",
            "        # drain(), never in a per-layer path).",
            "        _needle_token = config.slot_pick_needle_token()",
            "        _needle_block = report.get(\"block_size\") or stage.page_size",
            "        _needle_page = (",
            "            None",
            "            if _needle_token is None or not _needle_block",
            "            else _needle_token // _needle_block",
            "        )",
            "        _picked = set(selected)",
            "        logger.info(",
            "            \"vllm-030win patch (step 081): slot pick=%s req=%s \"",
            "            \"top_pages=%d slots=%d selected=%d needle=%s \"",
            "            \"in_selected=%s dropped=%s selected_pages=%s\",",
            "            pick,",
            "            stage.request_id,",
            "            len(top_pages),",
            "            len(stage.slots),",
            "            len(selected),",
            "            (",
            "                f\"page {_needle_page} (token {_needle_token})\"",
            "                if _needle_page is not None",
            "                else \"n/a (VLLM_KVMEM_SLOT_PICK_NEEDLE=<token>)\"",
            "            ),",
            "            (",
            "                _needle_page in _picked",
            "                if _needle_page is not None",
            "                else \"-\"",
            "            ),",
            "            [p for p in top_pages if p not in _picked],",
            "            selected,",
            "        )",
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
    ap.add_argument("--target", choices=["venv", "repo", "both"], default="both")
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
