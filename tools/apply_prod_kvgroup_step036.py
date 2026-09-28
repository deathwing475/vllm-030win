# -*- coding: utf-8 -*-
"""把 KV 改分组 + 上下文提长落进生产 launcher（apply/revert 幂等）。

改动三处：
  1. 在 pin_shim 的 `set "VLLM_DBG_PIN=1"` 之后插入 `set "VLLM_KV_GROUP_SIZE=8"`；
  2. `--max-model-len 110000` → `--max-model-len 144432`
     （036 落 130000、039 落 140000、040 落 144432）；
  3. 文件头插入一段 036 说明注释（含回退方法）。

依据：
  * **036（改分组）**：G=8 把池 3.4e9 的容量从 114,974 提到 136,190（+18.5%），
    decode 步长 18.73/18.74 → 18.30/18.31/18.32 ms（−2.3%），prefill 无差异；
  * **039（提长到 140,000）**：容量 143,307；当轮把 144,432 读成「decode −21%」
    而停在 140,000；
  * **040（提长到 144,432）**：交替 boot 复测证明 039 看到的掉速是**随机慢 boot**
    （同一 L 两次 boot 118-121 vs 96-99、慢的那次专用显存反而少 284 MiB、
    pin 放置 15 次 boot 逐字节相同），144,432 才是 G=8 的硬上限
    （`blocks_per_req ≤ num_blocks − 1`，引擎永久留一个 null block）；
  * **041（双判据复核）**：用户两条门（decode ≥85；用到 90% max-model-len 时 ≥70）
    下，144,432 是满足两者的最大值；
  * 池值仍是手动 3,400,000,000 B，**KV tensor 总大小不变、溢出量一个字不变**，
    改的只是分组与逻辑上限。

回退：python apply_prod_kvgroup_step036.py revert
"""
import sys

P = r"G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd"

ANCHOR_PIN = 'set "VLLM_DBG_PIN=1"'
LINE_KV = '\r\nset "VLLM_KV_GROUP_SIZE=8"'
OLD_LEN = "--max-model-len 110000"
NEW_LEN = "--max-model-len 144432"  # step 040: 140000 -> 144432 (G=8 ceiling)

MARK = "rem 2026-09-28 step 036"
HEAD_ANCHOR = "@echo off\r\n"
# NOTE: line endings here are plain "\n"; apply() converts them to CRLF exactly
# once. The older form used "\r\n" here *and* a global replace, which produced
# double-CR ("\r\r\n") lines in the launcher.
HEAD_NOTE = (
    "rem 2026-09-28 step 036 (KV regroup, PRODUCTION): VLLM_KV_GROUP_SIZE=8\n"
    "rem forces the KV layers-per-group from the upstream 5 (pinned by the\n"
    "rem draft's 5 sliding-window layers) to 8. Measured on this stack: the\n"
    "rem 3.4e9 pool goes 114,974 -> 136,190 tokens (+18.5%), decode step\n"
    "rem 18.73/18.74 -> 18.30/18.31/18.32 ms, prefill unchanged; needle was green\n"
    "rem at 8k/32k/64k/100k (max-model-len 110000, step-036 era) and 8k/64k/130k.\n"
    "rem max-model-len: 110000 -> 130000 -> 140000 (step 039) -> 144432 (step 040).\n"
    "rem The G=8 hard ceiling is 144,432: the engine permanently holds back one\n"
    "rem null block before the admission check, so the real constraint is\n"
    "rem blocks_per_req = 2*cdiv(L,2832) + 27 <= num_blocks - 1 = 129, i.e.\n"
    "rem cdiv(L,2832) <= 51. Step 039 read the 21% decode drop at 144,432 as an\n"
    "rem L effect and stopped at 140,000; step 040 re-measured with alternating\n"
    "rem boots and showed that drop is a random slow boot, not an L effect (same\n"
    "rem L=144,432: steady 118-121 at fb 15,538 MiB vs 96-99 at 15,254 MiB;\n"
    "rem pin placement byte-identical across 15 boots). Multi-depth needle at\n"
    "rem 144,432 matches 140,000 depth for depth (8k 120.2/120.4, 64k 121.0/116.4,\n"
    "rem 115k 113.0/108.0, 135k 102.3/103.2). Engine reports 145,551 tokens.\n"
    "rem Step 041 re-checked against the user's two gates (decode >=85; >=70 when\n"
    "rem using 90% of max-model-len) and confirmed 144,432 is the largest value\n"
    "rem that satisfies both: G=1@147,264 breaks 85 on a slow boot, N=1@154,880\n"
    "rem breaks 70 at its 86.7% depth, N=0@164,256 breaks both.\n"
    "rem CAVEAT: at 144,432 a full-length request claims all 129 usable blocks,\n"
    "rem so a max-length request leaves no headroom for prefix-cache growth.\n"
    "rem Pool stays the manual 3,400,000,000 B: KV tensor size and spill are\n"
    "rem unchanged; only grouping and the logical ceiling move.\n"
    "rem ROLLBACK = delete the VLLM_KV_GROUP_SIZE line, restore 110000, drop\n"
    "rem this note (or: python tools\\apply_prod_kvgroup_step036.py revert).\n"
)


def _note_bytes():
    return HEAD_NOTE.replace("\n", "\r\n").encode()


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
                            HEAD_ANCHOR.encode() + _note_bytes())
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
        note = _note_bytes()
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
