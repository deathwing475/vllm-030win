# -*- coding: utf-8 -*-
"""venv 补丁（步骤 066）：MambaManager 的外部命中只分配 1 个状态块（apply/revert 幂等）。

背景
----
KVMem 步骤 066 要把工作区页装配回后续请求的前缀（连接器 matched 前跳）。
前跳后调度器走 `allocate_slots(num_external_computed_tokens=B)`，基类
`SingleTypeKVCacheManager.allocate_external_computed_blocks` 会给**每个组**
分配 `cdiv(B, block_size)` 个真实块。对 mamba 组（align 模式，每块 = 一份
48 层状态里的 8 层，13.4 MiB）这是又贵又没人读的：引擎在递推锚定上只读
`preprocess_mamba` 的 `(num_computed_tokens - 1) // block_size` 号位 ——
即边界位 E-1 —— 中间位全不被读。

修法
----
给 `MambaManager` 覆写 `allocate_external_computed_blocks`：装配段 =
`[null × (E-1), 1 个真实块]`，真实块落在边界位，由 KVMem load 用 host 快照
填充。形状与 `find_longest_cache_hit` 给本地命中返回的
`[null × i, cached]` 完全一致，null 块不进 hash、不占池。
不改 `add_local_computed_blocks`（local 命中为 0 时它本就无操作；KVMem
装配在 local > 0 时主动短路为不装配）。

安全影响面
----------
只影响"外部连接器 matched > 0 且该组是 MambaSpec"的路径。生产配置没有
KVMemConnector（offloading 顶槽位被 KVMemConnector 替换只发生在
VLLM_KVMEM_WORKSPACE=1 的臂上），生产路径不经过本覆写。

回退
----
  python tools\\apply_kvmem_mamba_ext_step066.py revert
"""
import sys

P = r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm\v1\core\single_type_kv_cache_manager.py"

# MambaManager.get_num_common_prefix_blocks is the only occurrence of this
# docstring (CrossAttentionManager's reads "Cross-attention blocks contain
# request-specific encoder states"), so the anchor is unique.
ANCHOR = (
    b"    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:\r\n"
    b'        """\r\n'
    b"        cascade attention is not supported by mamba\r\n"
    b'        """\r\n'
    b"        return 0"
)

METHOD = (
    b"\r\n"
    b"    def allocate_external_computed_blocks(\r\n"
    b"        self,\r\n"
    b"        request_id: str,\r\n"
    b"        num_local_computed_tokens: int,\r\n"
    b"        num_external_computed_tokens: int,\r\n"
    b"    ) -> None:\r\n"
    b"        # vllm-030win patch (step 066): a mamba external load needs ONE state\r\n"
    b"        # block, at the boundary position. The base class allocates\r\n"
    b"        # cdiv(total_computed, block_size) real state slots for the loaded\r\n"
    b"        # prefix; for a recurrent cache that is both wasteful and unread --\r\n"
    b"        # only the boundary block (position E-1) is ever anchored on\r\n"
    b"        # (``preprocess_mamba`` uses (num_computed_tokens - 1) // block_size),\r\n"
    b"        # and at one ~13.4 MiB slot per block per group the base behaviour\r\n"
    b"        # burns the whole pool before the assembly step runs. This mirrors the\r\n"
    b"        # shape ``find_longest_cache_hit`` returns for a local mamba hit:\r\n"
    b"        # null placeholders up to the boundary, then the one real block whose\r\n"
    b"        # slot the KVMem load fills with the state snapshot.\r\n"
    b"        assert isinstance(self.kv_cache_spec, MambaSpec)\r\n"
    b"        num_total = num_local_computed_tokens + num_external_computed_tokens\r\n"
    b"        num_skipped_tokens = self.get_num_skipped_tokens(num_total)\r\n"
    b"        if num_skipped_tokens > 0:\r\n"
    b"            num_external_computed_tokens = min(\r\n"
    b"                num_total - num_skipped_tokens, num_external_computed_tokens\r\n"
    b"            )\r\n"
    b"        if num_external_computed_tokens <= 0:\r\n"
    b"            return\r\n"
    b"        req_blocks = self.req_to_blocks[request_id]\r\n"
    b"        num_boundary_blocks = max(\r\n"
    b"            0, cdiv(num_total, self.block_size) - len(req_blocks)\r\n"
    b"        )\r\n"
    b"        if num_boundary_blocks == 0:\r\n"
    b"            return\r\n"
    b"        req_blocks.extend([self._null_block] * (num_boundary_blocks - 1))\r\n"
    b"        allocated = self.block_pool.get_new_blocks(1)\r\n"
    b"        req_blocks.extend(allocated)\r\n"
    b"        if self._record_new_block_ids:\r\n"
    b"            self.new_block_ids.extend(b.block_id for b in allocated)\r\n"
)

FEATURE = b"vllm-030win patch (step 066): a mamba external load needs ONE state"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    assert mode in ("apply", "revert"), "usage: apply|revert"
    with open(P, "rb") as handle:
        data = handle.read()
    if mode == "apply":
        if FEATURE in data:
            print("already applied, nothing to do")
            return
        assert data.count(ANCHOR) == 1, (
            f"anchor count = {data.count(ANCHOR)}, expected 1; file drifted?"
        )
        replacement = METHOD + ANCHOR
        data = data.replace(ANCHOR, replacement)
    else:
        if FEATURE not in data:
            print("already reverted, nothing to do")
            return
        block = METHOD + ANCHOR
        assert data.count(block) == 1, "patched block not found verbatim"
        data = data.replace(block, ANCHOR)
    with open(P, "wb") as handle:
        handle.write(data)
    print(mode + "ed:", P)


if __name__ == "__main__":
    main()
