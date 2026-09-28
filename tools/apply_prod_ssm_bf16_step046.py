# -*- coding: utf-8 -*-
"""step-046 生产切换（mamba ssm bf16 + max-model-len 163,072）的 revert 工具。

launcher 的 046 改动（2026-09-28，经 Edit 工具落地点）：
  1. 文件头插入 046 记注块（首行 `rem 2026-09-28 step 046 (mamba ssm bf16,
     PRODUCTION): adds`，末行 `rem BEFORE tools\\apply_prod_kvgroup_step036.py
     revert.`，共 23 行）；
  2. `  --mamba-cache-mode align ^` 之后插入
     `  --mamba-ssm-cache-dtype bfloat16 ^`；
  3. `--max-model-len 144432` -> `--max-model-len 163072`。

用法：
  python apply_prod_ssm_bf16_step046.py           # 验证 applied 状态，不写文件
  python apply_prod_ssm_bf16_step046.py revert    # 回滚三处（幂等）

链式回退次序：先本脚本 revert，再 tools\\apply_prod_kvgroup_step036.py revert
（036 的 revert 断言 launcher 里 max-model-len == 144432）。

依据 = 步骤 045：A/B 6 boot 同带（bf16 120.6-125.5 vs fp32 118.6-127.2）、
needle 18/18、PPL 同轮 max|delta| 1.25e-3（漂移带内）、20 条长稳健康、
容量 145,551 -> 163,719（+12.5%）；首编深塌 4/4 ⇒ 生产切换必须双 boot。
"""
import sys

P = r"G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd"

NOTE_FIRST = "rem 2026-09-28 step 046 (mamba ssm bf16, PRODUCTION): adds"
NOTE_LAST_PREFIX = "rem BEFORE tools\\apply_prod_kvgroup_step036.py revert."
SSM_LINE = "\r\n  --mamba-ssm-cache-dtype bfloat16 ^"
OLD_LEN = "--max-model-len 144432"
NEW_LEN = "--max-model-len 163072"


def _lines(data: bytes):
    return data.splitlines(keepends=True)


def _note_bounds(lines):
    starts = [i for i, l in enumerate(lines)
              if l.decode("utf-8", "replace").rstrip("\r\n") == NOTE_FIRST]
    ends = [i for i, l in enumerate(lines)
            if l.decode("utf-8", "replace").rstrip("\r\n").startswith(NOTE_LAST_PREFIX)]
    assert len(starts) == 1, "note first-line count=%d" % len(starts)
    assert len(ends) == 1, "note last-line count=%d" % len(ends)
    assert starts[0] < ends[0], "note order broken"
    return starts[0], ends[0]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    assert mode in ("verify", "revert"), "usage: verify|revert"
    data = open(P, "rb").read()
    assert data.count(b"\r\n") == data.count(b"\n"), "launcher must be pure CRLF"
    lines = _lines(data)
    s, e = _note_bounds(lines)
    n_note = e - s + 1

    if mode == "verify":
        assert data.count(SSM_LINE.encode()) == 1, "ssm line missing"
        assert data.count(NEW_LEN.encode()) == 1, "new max-model-len missing"
        print("applied state OK: note %d lines (l.%d-%d), ssm line, %s"
              % (n_note, s + 1, e + 1, NEW_LEN))
        print("remember: production switchover needs the DOUBLE-BOOT flow "
              "(boot #1 rebuilds AOT cache and reads ~17 tok/s; kill it, "
              "boot #2 is production). See the launcher head note.")
        return

    # revert
    del lines[s:e + 1]
    data2 = "".join(l.decode("utf-8", "replace") for l in lines).encode("utf-8")
    assert data2.count(SSM_LINE.encode()) == 1, "ssm line count!=1"
    data2 = data2.replace(SSM_LINE.encode(), b"")
    assert data2.count(NEW_LEN.encode()) == 1, "new len count!=1"
    data2 = data2.replace(NEW_LEN.encode(), OLD_LEN.encode())
    open(P, "wb").write(data2)

    chk = open(P, "rb").read()
    assert chk.count(b"step 046") == 0, "note residue"
    assert chk.count(b"mamba-ssm-cache-dtype") == 0, "ssm residue"
    assert chk.count(OLD_LEN.encode()) == 1, "old len missing"
    print("reverted:", P)


if __name__ == "__main__":
    main()
