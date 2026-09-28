# -*- coding: utf-8 -*-
"""补丁：允许显式指定 KV cache 分组的 layers-per-group（步骤 036）。

背景（步骤 035 实测）
--------------------
`vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`
从**最小的注意力类型桶**推导 group_size：

    min_num_layers = min(len(bucket) for bucket in layer_buckets)
    group_size = min_num_layers
    max_num_layers = max(...)
    if max_num_layers < min_num_layers * 1.5:
        group_size = max_num_layers

本模型的桶层数 = mamba 48 / full 16 / 草稿 sliding-window 5，所以
group_size = 5。kvdump 实测（pool=3.4e9，max_model_len=110,000）得到
15 组（mamba 8x5 + 2x4、full 4x4、sw 1x5）、
bytes_per_block = 16,312,320 = 5 x 3,262,464、num_blocks = 208、
199.00 blocks/request，容量 208/199 x 110,000 = 114,974（与引擎自报一致）。

为什么 5 是坏值
--------------
`unify_kv_cache_spec_page_size` 已把所有 spec 的 page size 统一到同一个
per-layer 值（3,262,464 B = 2832 token x 1152 B/token，nvfp4），所以：

  * bytes_per_block = (组内最大层数) x 3,262,464 —— 随 group_size 线性增长；
  * 一个请求在各组上独立取块，每组的块数 = cdiv(spec.max_memory_usage_bytes,
    page_size_bytes)，**与组内层数无关**（full 组恒为 cdiv(max_model_len,
    2832)，mamba 组恒为份数 2+N+checkpoints，sw 组恒为窗口块数）。
    所以 blocks_per_request 只取决于**各桶被切成几个组**。

于是容量 = [pool / (G x 3,262,464)] / [39 x cdiv(16,G) + 4 x cdiv(48,G)
+ 3 x cdiv(5,G)] x max_model_len。G=5 时 16 和 48 都除不尽（组数 4 / 10），
而 G=4 或 G=8 时分子大 25%（G=8 时分子减半但组数也减半，恰好抵消）：

    G      bytes_per_block   num_blocks   blocks/req   capacity(110k)
    1         3,262,464        1042          831          137,930
    2         6,524,928         521          417          137,434
    4        13,049,856         260          210          136,190
    5(上游)  16,312,320         208          199          114,974
    8        26,099,712         130          105          136,190
   16        52,199,424          65           54          132,407

改 G 不改变池大小、不改变每层 page size、不改变溢出量，只改
bytes_per_block 与 num_blocks；所有组仍然同层数（G）或同 page size，
假设 1「每 block 物理内存相同」依旧成立。

补丁形态
--------
env 开关 `VLLM_KV_GROUP_SIZE=N`：设置则强制 group_size = N，不设置则
上游行为**逐字节不变**（原注释与分支保持原样），因此默认零影响、可随时回退。

回退
----
  python apply_kv_group_size_patch.py revert
"""
import re
import sys

BASE = r"G:\qwen3.8model\vllm-029base-git\vllm\v1\core\kv_cache_utils.py"
VENV = (r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm\v1\core"
        r"\kv_cache_utils.py")

MARK = b"vllm-030win patch (step 036)"

ANCHOR = b"    grouped_layers = []\r\n    for layers in layer_buckets:\r\n"

_BLOCK = """    # vllm-030win patch (step 036): optional explicit layers-per-group
    # override. The heuristic above derives the group size from the SMALLEST
    # attention-type bucket, so a small speculative-drafter bucket (DFlash2:
    # 5 sliding-window layers) can pin it to a value that divides the target's
    # buckets badly. bytes_per_block is the widest group's page and therefore
    # scales with the layers per group, while the blocks a request claims
    # depend only on how many groups each bucket splits into, so a bad divisor
    # both enlarges the block and multiplies the group count. Setting
    # VLLM_KV_GROUP_SIZE=N forces the group size so capacity can be measured
    # per candidate; unset keeps upstream behaviour byte-for-byte.
    _forced_group_size = os.environ.get("VLLM_KV_GROUP_SIZE", "").strip()
    if _forced_group_size:
        try:
            _forced_group_size_n = int(_forced_group_size)
        except ValueError:
            raise ValueError(
                "VLLM_KV_GROUP_SIZE must be a positive integer, got %r"
                % (_forced_group_size,)
            ) from None
        if _forced_group_size_n < 1:
            raise ValueError(
                "VLLM_KV_GROUP_SIZE must be >= 1, got %d"
                % (_forced_group_size_n,)
            )
        logger.info(
            "vllm-030win patch (step 036): layers-per-group %d -> %d "
            "(VLLM_KV_GROUP_SIZE)",
            group_size,
            _forced_group_size_n,
        )
        group_size = _forced_group_size_n
"""

NEW = _BLOCK.replace("\n", "\r\n").encode() + ANCHOR

# 剥离已打补丁的块（连同前导空白一起吃掉，容忍历史上出现过的中间形态）。
_STRIP = re.compile(
    rb"[ \t]*(?:# )?vllm-030win patch \(step 036\).*?(?=    grouped_layers = \[\])",
    re.S,
)


def _one(path, mode):
    data = open(path, "rb").read()
    if data.count(b"\r\n") != data.count(b"\n"):
        raise AssertionError("%s: mixed line endings" % path)
    if mode == "apply":
        if MARK in data:
            assert data.count(NEW) == 1, "already-patched block not found"
            print("already patched: %s" % path)
            return
        assert data.count(ANCHOR) == 1, "anchor count=%d" % data.count(ANCHOR)
        data = data.replace(ANCHOR, NEW)
        open(path, "wb").write(data)
        chk = open(path, "rb").read()
        assert MARK in chk and chk.count(ANCHOR) == 1
        print("patch applied: %s" % path)
    else:
        if MARK not in data:
            assert data.count(ANCHOR) == 1
            print("already clean: %s" % path)
            return
        stripped, n = _STRIP.subn(b"", data)
        assert n == 1, "strip count=%d" % n
        open(path, "wb").write(stripped)
        chk = open(path, "rb").read()
        assert MARK not in chk and chk.count(ANCHOR) == 1
        print("patch reverted: %s" % path)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    assert mode in ("apply", "revert"), "usage: apply|revert"
    for p in (BASE, VENV):
        _one(p, mode)


if __name__ == "__main__":
    main()
