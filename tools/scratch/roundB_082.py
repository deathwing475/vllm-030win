# -*- coding: utf-8 -*-
"""082-B runner：score 档多深度正确性网格（阶段 1 验收，不是召回率优化）。

每深度一支净 boot（探针前零污染：--sweep 0 --no-main --prof 0 --pdh 0），
每格 2 次完整探针（各自独立 ingest + needle，TRAJ=2 恰好容纳，b6 模式），
一律 --ignore-eos --max-tokens 96（必守 25(3)：单次 32-token 预算会把命中
读成 miss）。判据只看分支指纹 recent_tokens=16384。
"""
from __future__ import annotations

import subprocess

PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
DRV = r"G:\qwen3.8model\vllm-030win-git\tools\step082_boot.py"
CWD = r"G:\qwen3.8model"


def run(args: list[str]) -> int:
    print(":: RUN", " ".join(args), flush=True)
    p = subprocess.run([PY, DRV] + args, capture_output=True, text=True,
                       cwd=CWD)
    print((p.stdout or "")[-4000:], flush=True)
    if p.returncode != 0:
        print(":: RC %s stderr: %s" % (p.returncode, (p.stderr or "")[-1000:]),
              flush=True)
    return p.returncode


# (boot, tag, condition, depth)
CELLS = [
    ("b9", "kvmem_k16i", "B040", 0.40),
    ("b10", "kvmem_k16j", "B055", 0.55),
    ("b11", "kvmem_k16k", "B070", 0.70),
    ("b12", "kvmem_k16l", "B085", 0.85),
]

for boot, tag, cond, depth in CELLS:
    rc = run(["next", "--boot", boot, "--tag", tag, "--condition", cond,
              "--pick", "score", "--sweep", "0", "--no-main",
              "--prof", "0", "--pdh", "0"])
    if rc != 0:
        print(":: %s (%s) next rc=%s -- skip probes, continue"
              % (boot, cond, rc), flush=True)
        continue
    for suffix in ("a", "b"):
        run(["vp", "--boot", boot, "--depth", str(depth),
             "--nonce", "%sd%02d%s" % (boot, int(depth * 100), suffix),
             "--ignore-eos", "--max-tokens", "96"])
print(":: ROUND-B DONE", flush=True)
