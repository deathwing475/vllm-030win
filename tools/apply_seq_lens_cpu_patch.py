# -*- coding: utf-8 -*-
"""venv 补丁：消除 DFlash2 草稿 non-causal 层每步一次的白做 D2H 同步（步骤 032）。

背景（实测数据）
----------------
`vllm/v1/attention/backends/flashinfer.py` 的 `build()` 里：

    needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm

`all_uses_trtllm` 里含 `causal`，而 DFlash2 草稿有 non-causal 层
（`dflash_has_any_non_causal` ⇒ `_group_causal[gid]=False` ⇒ 本 batch
`common_attn_metadata.causal=False`）⇒ `all_uses_trtllm=False` ⇒ 每步都去读
deprecated 的 `CommonAttentionMetadata.seq_lens_cpu`，其实现是
`self.seq_lens.to("cpu")` —— 一次 **D2H 同步**。

桩实测（`VLLM_DBG_SLCPU=1`，生产 launcher，500-token decode）：
  [SLCPU] 17.4ms p50（稳定 16.97-17.48）/ 步；
  causal=False all_trtllm=False ded_xqa=True trtllm_dec=True npre=0 ndec=1
  pre_trtllm=False dec_trtllm=True NEEDS_PAGED=False
即该批次 decode 走 dedicated XQA/TRTLLM API，`needs_paged_kv_indices=False`，
`seq_lens_np/num_blocks_np` 根本没人用 —— 同步结果被丢弃。

修法
----
条件改成真正消费 `seq_lens_np/num_blocks_np` 的路径（native paged prefill /
native paged decode / cascade / DCP），并把 `needs_native_paged_*` 提前定义、
后面复用。安全性：`seq_lens_cpu/seq_lens_np/num_blocks_np` 在该函数内的全部
使用点（DCP 分支、cascade 分支、`needs_paged_kv_indices` 分支、native decode
分支的 assert）都被新条件覆盖；旧条件比新条件多出的情形恰好是
`needs_paged_kv_indices=False`（值不被使用）。

验收（2026-09-27，交替 A/B 三对 + 正确性门）
  A/B（boot_e2e med_p50，同时段交替 SA/SB）：
    补丁 20.81 / 20.87 / 19.06  vs  无补丁 23.51 / 22.43 / 22.46
    ⇒ 均值 −2.55ms（−11.2%），吞吐 104.3 → 117.8 tok/s（+13.0%）
  acc：0.69/0.72/0.69 vs 0.69/0.70/0.69（无系统差异）
  needle 8k 3/3、32k 3/3 命中；32k 多轮 TTFT 21.8→3.15s
  GPU util p50=97%（补丁后 GPU 饱和）

回退
----
  python apply_seq_lens_cpu_patch.py revert
"""
import sys

P = (r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm\v1\attention"
     r"\backends\flashinfer.py")

OLD1 = (b"        needs_seq_lens_cpu = self.use_dcp or use_cascade"
        b" or not all_uses_trtllm\n")

NEW1 = (
    b"        # vllm-030win patch (step 032): gate on the paths that actually\n"
    b"        # consume seq_lens_np/num_blocks_np (native paged KV indices,\n"
    b"        # cascade, DCP) instead of on `all_uses_trtllm`, which is also\n"
    b"        # False for non-causal batches. A non-causal batch that decodes\n"
    b"        # through the dedicated XQA/TRTLLM API never builds paged KV\n"
    b"        # indices, so the old condition forced a per-step D2H sync\n"
    b"        # (~16 ms, measured on the DFlash2 draft's non-causal layers)\n"
    b"        # whose result was discarded.\n"
    b"        needs_native_paged_prefill = (num_prefills > 0\n"
    b"                                      and not prefill_use_trtllm)\n"
    b"        needs_native_paged_decode = (\n"
    b"            num_decodes > 0 and not decode_with_flashinfer_trtllm_api\n"
    b"        )\n"
    b"        needs_seq_lens_cpu = (\n"
    b"            self.use_dcp\n"
    b"            or use_cascade\n"
    b"            or needs_native_paged_prefill\n"
    b"            or needs_native_paged_decode\n"
    b"        )\n"
)

OLD2 = (
    b"        needs_native_paged_prefill = num_prefills > 0 and not prefill_use_trtllm\n"
    b"        needs_native_paged_decode = (\n"
    b"            num_decodes > 0 and not decode_with_flashinfer_trtllm_api\n"
    b"        )\n"
    b"        needs_paged_kv_indices = (\n"
)

NEW2 = b"        needs_paged_kv_indices = (\n"

MARK = b"vllm-030win patch (step 032)"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    data = open(P, "rb").read()
    if mode == "apply":
        if MARK in data:
            assert data.count(NEW2) == 1 and data.count(NEW1) == 1
            print("already patched")
            return
        assert data.count(OLD2) == 1, "anchor2 count=%d" % data.count(OLD2)
        assert data.count(OLD1) == 1, "anchor1 count=%d" % data.count(OLD1)
        data = data.replace(OLD2, NEW2).replace(OLD1, NEW1)
        open(P, "wb").write(data)
        chk = open(P, "rb").read()
        assert MARK in chk and chk.count(b"needs_seq_lens_cpu = (\n") == 1
        print("patch applied")
    else:
        if MARK not in data:
            assert b"needs_seq_lens_cpu = self.use_dcp or use_cascade" in data
            print("already clean")
            return
        assert data.count(NEW2) == 1, "new2 count=%d" % data.count(NEW2)
        assert data.count(NEW1) == 1, "new1 count=%d" % data.count(NEW1)
        data = data.replace(NEW1, OLD1).replace(NEW2, OLD2)
        open(P, "wb").write(data)
        chk = open(P, "rb").read()
        assert MARK not in chk
        assert b"needs_seq_lens_cpu = self.use_dcp or use_cascade" in chk
        print("patch reverted")


if __name__ == "__main__":
    main()
