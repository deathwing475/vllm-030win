# -*- coding: utf-8 -*-
"""把步骤 036 的改分组落进生产 launcher（apply/revert 幂等）。

改动三处：
  1. 在 pin_shim 的 `set "VLLM_DBG_PIN=1"` 之后插入 `set "VLLM_KV_GROUP_SIZE=8"`；
  2. `--max-model-len 110000` → `--max-model-len 130000`；
  3. 文件头插入一段 036 说明注释（含回退方法）。

依据（步骤 036 实测）：
  * G=8 把池 3.4e9 的容量从 114,974 提到 136,190（+18.5%），decode 步长
    18.73/18.74 → 18.30/18.31/18.32 ms（−2.3%），prefill 无差异；
  * G=8 + max_model_len 140,000 时 130k 档 needle 命中（129,443 token、
    ttft 87.66s、稳态 104.7 tok/s）⇒ 130,000 有充分余量（该长度容量 142,016）；
  * 池值仍是手动 3,400,000,000 B，**KV tensor 总大小不变、溢出量一个字不变**，
    改的只是分组与逻辑上限。

回退：python apply_prod_kvgroup_step036.py revert
"""
import sys

P = r"G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd"

ANCHOR_PIN = 'set "VLLM_DBG_PIN=1"'
LINE_KV = '\r\nset "VLLM_KV_GROUP_SIZE=8"'
OLD_LEN = "--max-model-len 110000"
NEW_LEN = "--max-model-len 140000"  # step 039: 130000 -> 140000 (perf-safe)

MARK = "rem 2026-09-28 step 036"
HEAD_ANCHOR = "@echo off\r\n"
HEAD_NOTE = (
    "rem 2026-09-28 step 036 (KV regroup, PRODUCTION): VLLM_KV_GROUP_SIZE=8\r\n"
    "rem forces the KV layers-per-group from the upstream 5 (pinned by the\r\n"
    "rem draft's 5 sliding-window layers) to 8. Measured on this stack: the\r\n"
    "rem 3.4e9 pool goes 114,974 -> 136,190 tokens (+18.5%), decode step\r\n"
    "rem 18.73/18.74 -> 18.30/18.31/18.32 ms, prefill unchanged, needle green\r\n"
    "rem at 8k/32k/64k/100k (max-model-len 110000) and 8k/64k/130k (140000).\r\n"
    "rem max-model-len: 110000 -> 130000 -> 140000 (step 039). The G=8 hard\r\n"
    "rem ceiling is 144,432 (blocks_per_req = 2*cdiv(L,2832) + 27 <= 130), but\r\n"
    "rem 144,432 measures ~21% slower decode (2 boots: steady 95-105 vs\r\n"
    "rem 119-128) and 141,600 already costs ~5%, so 140,000 is the largest\r\n"
    "rem value whose performance matches 130,000 (2 boots: steady 118-128).\r\n"
    "rem Engine reports 143,307 tokens (1.02x) vs 142,016 at 130000.\r\n"
    "rem Pool stays the manual 3,400,000,000 B: KV tensor size and spill are\r\n"
    "rem unchanged; only grouping and the logical ceiling move.\r\n"
    "rem ROLLBACK = delete the VLLM_KV_GROUP_SIZE line, restore 110000, drop\r\n"
    "rem this note (or: python tools\\apply_prod_kvgroup_step036.py revert).\r\n"
)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    assert mode in ("apply", "revert"), "usage: apply|revert"
    data = open(P, "rb").read()
    assert data.count(b"\r\n") == data.count(b"\n"), "launcher must be pure CRLF"

    if mode == "apply":
        if MARK in data.decode("utf-8", "replace"):
            print("already applied")
            return
        for a in (ANCHOR_PIN, OLD_LEN, "@echo off\r\n"):
            assert data.count(a.encode()) == 1, "anchor count=%d for %r" % (
                data.count(a.encode()), a[:40])
        # Order matters: change max-model-len FIRST, then insert the note
        # (the note must never contain a string that a later replace touches,
        #  otherwise revert cannot match it back).
        data = data.replace(OLD_LEN.encode(), NEW_LEN.encode())
        data = data.replace(HEAD_ANCHOR.encode(),
                            HEAD_ANCHOR.encode() + HEAD_NOTE.replace("\n", "\r\n").encode())
        data = data.replace(ANCHOR_PIN.encode(), (ANCHOR_PIN + LINE_KV).encode())
        open(P, "wb").write(data)
        chk = open(P, "rb").read().decode("utf-8", "replace")
        assert MARK in chk and 'set "VLLM_KV_GROUP_SIZE=8"' in chk
        assert NEW_LEN in chk and OLD_LEN not in chk
        print("applied:", P)
    else:
        if MARK not in data.decode("utf-8", "replace"):
            print("already clean")
            return
        note = HEAD_NOTE.replace("\n", "\r\n").encode()
        assert data.count(note) == 1, "head note count=%d" % data.count(note)
        data = data.replace(note, b"")
        assert data.count((ANCHOR_PIN + LINE_KV).encode()) == 1
        data = data.replace((ANCHOR_PIN + LINE_KV).encode(), ANCHOR_PIN.encode())
        assert data.count(NEW_LEN.encode()) == 1
        data = data.replace(NEW_LEN.encode(), OLD_LEN.encode())
        open(P, "wb").write(data)
        chk = open(P, "rb").read().decode("utf-8", "replace")
        assert MARK not in chk and "VLLM_KV_GROUP_SIZE" not in chk and OLD_LEN in chk
        print("reverted:", P)


if __name__ == "__main__":
    main()
