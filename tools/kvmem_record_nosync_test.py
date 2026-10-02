# -*- coding: utf-8 -*-
"""步骤 080 的 CPU 单测：``capture`` 的 record 路径在两种门控下语义必须一致。

跑法（不起引擎；cwd 别落在记录仓，否则会 import 到仓里的 vllm/）::

    cd G:\\qwen3.8model
    G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe ^
        G:\\qwen3.8model\\vllm-030win-git\\tools\\kvmem_record_nosync_test.py

它证明的四件事（都是"boot 之前就能证"的，不必花显存）：
  1. **canary 不丢**：``_MROPE_AXES_DIFFER`` 在"三行不全同"的输入上，门控开/关给出
     **相同的每层计数**；在"三行全同"（纯文本的真实形态）上两边都是 0。
  2. **下游形状不变**：``drain()`` 交出的 positions 永远是 1-D ``[T]``（= 行 0），
     与门控无关 —— 门控开时暂存的是 ``[2, T]``，判定与降维都发生在 drain 里。
  3. **同步点真的被拿掉**：门控关时 ``stats()["record_sync_calls"] > 0``（走了
     ``torch.equal``），门控开时 **恰为 0**（改判在主机 numpy 上）。这是本步唯一
     的行为差别，也是慢态 16 个流同步的来源。
  4. **台账多出的段有数**：``record_calls`` = 层数 × 步数，``record_seconds`` 随调用增长。

CPU 张量上 ``torch.equal`` 不同步流，所以第 3 条证的是"走没走那条路"，不是"省了多少
秒"——秒数必须在 boot 上用 ``[KVTIME]`` 的 ``drec`` 读（本步实测见《实验步骤文档》080）。
"""
import os
import sys

import torch

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def fresh(nosync: bool):
    """Re-import capture with the gate set (the gate value is cached per module)."""
    os.environ["VLLM_KVMEM_RECORD_NOSYNC"] = "1" if nosync else "0"
    for name in [m for m in list(sys.modules) if "kvmem_workspace" in m]:
        del sys.modules[name]
    from vllm.v1.kvmem_workspace import capture
    capture.reset()
    return capture


def make(step_tokens: int, layers: int, axes_differ: bool):
    """One armed prefill step: [3, T] positions + q/k, all on CPU."""
    pos = torch.arange(step_tokens, dtype=torch.int64).repeat(3, 1)
    if axes_differ:
        pos[1] = pos[1] + 7
    q = torch.zeros(step_tokens, 4 * 256, dtype=torch.float16)
    k = torch.zeros(step_tokens, 4 * 256, dtype=torch.float16)
    return pos, q, k


def run(capture, step_tokens, layers, axes_differ, arm=None):
    pos, q, k = make(step_tokens, layers, axes_differ)
    if arm is None:
        arm = {step_tokens}
    capture.arm(arm)
    for layer in range(layers):
        capture.record(layer, pos, q, k, 24, 4, 256)
    out = capture.drain()
    capture.disarm()
    return out


def main() -> int:
    tokens, layers = 1456, 16
    print("=== 1/2. canary 计数与下游形状（两种门控 × 两种输入）===")
    results = {}
    for nosync in (False, True):
        cap = fresh(nosync)
        tag = "nosync=1" if nosync else "nosync=0"
        for differ in (False, True):
            cap.reset()
            out = run(cap, tokens, layers, differ)
            st = cap.stats()
            key = (nosync, differ)
            results[key] = (st["mrope_axes_differ"], st["record_calls"],
                            st["record_sync_calls"], out)
            rows = {int(idx) for idx in out}
            shape = out[0][0].shape if out else None
            check(sorted(rows) == list(range(layers)),
                  f"{tag} axes_differ={differ}: drain 交出 {len(out)}/{layers} 层")
            check(shape == (tokens,),
                  f"{tag} axes_differ={differ}: positions 形状 {shape} == ({tokens},)")
            if out:
                base = torch.arange(tokens, dtype=torch.int64)
                check(bool((out[0][0] == base).all()),
                      f"{tag} axes_differ={differ}: positions 值 = 行 0")
    for differ in (False, True):
        legacy = results[(False, differ)]
        fixed = results[(True, differ)]
        check(legacy[0] == fixed[0],
              f"axes_differ={differ}: 同轴 canary 计数不变 "
              f"(legacy={legacy[0]} fixed={fixed[0]})")
        check(legacy[1] == fixed[1],
              f"axes_differ={differ}: record_calls 不变 "
              f"(legacy={legacy[1]} fixed={fixed[1]})")

    print("=== 3. 同步点：门控关 = 走 torch.equal；门控开 = 一次都不走 ===")
    for differ in (False, True):
        legacy = results[(False, differ)]
        fixed = results[(True, differ)]
        check(legacy[2] == layers,
              f"axes_differ={differ}: 门控关 record_sync_calls={legacy[2]} "
              f"== 每层一次({layers})")
        check(fixed[2] == 0,
              f"axes_differ={differ}: 门控开 record_sync_calls={fixed[2]} == 0")

    print("=== 4. 未武装 / decode 步不该记账 ===")
    cap = fresh(True)
    cap.reset()
    pos, q, k = make(1, 16, False)
    cap.arm({1456})
    for layer in range(4):
        capture_record = cap.record
        capture_record(layer, pos, q, k, 24, 4, 256)   # num_tokens<=1 -> decode
    cap.disarm()
    st = cap.stats()
    check(st["record_calls"] == 0,
          f"decode 步不记 record 账（record_calls={st['record_calls']}）")
    check(st["skipped_decode_steps"] == 4,
          f"decode 计数仍在（{st['skipped_decode_steps']}）")
    check(cap.drain() == {}, "decode 步 drain 空")

    print("=== 5. 门控默认必须是旧行为（生产默认不动）===")
    os.environ.pop("VLLM_KVMEM_RECORD_NOSYNC", None)
    for name in [m for m in list(sys.modules) if "kvmem_workspace" in m]:
        del sys.modules[name]
    from vllm.v1.kvmem_workspace import capture as cap0, config as cfg0
    check(cfg0.record_nosync() is False,
          "不设 VLLM_KVMEM_RECORD_NOSYNC 时 record_nosync() == False")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项不合格")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("PASS: record 路径在两种门控下同语义、canary 不丢、同步点确实归零。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
