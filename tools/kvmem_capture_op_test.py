# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win step 067: offline test for the KVMem capture being opaque to torch.compile.

The arm used to run with ``--enforce-eager`` because an AOT
``torch.compile(fullgraph=True)`` of the model traced into ``capture.record``
and rejected it — first as a ``logging`` call, then as ``aten.equal.default``
(the data-dependent M-RoPE check). Wrapping the capture in a
``torch.library.custom_op`` makes dynamo treat it as a leaf, which is what let
the arm move to the production graph mode.

This test pins the three properties that trick relies on, so a future change to
``capture.py`` cannot silently break the graph compatibility:

1. the custom op is registered under the expected name;
2. calling it still fills the stash (eager path, and the single-token guard);
3. a ``fullgraph=True`` compile accepts it, and the op *implementation* is
   invoked on every execution of the compiled graph — not only at trace time.
   That last one is the whole reason the stash stays fed.

No GPU is required and no engine is started.

Usage:
    python tools\\kvmem_capture_op_test.py
"""

import sys

import torch

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    from vllm.v1.kvmem_workspace import capture

    # 1. the op is registered
    check(
        "custom op registered as torch.ops.vllm_kvmem.record",
        hasattr(torch.ops.vllm_kvmem, "record"),
    )

    positions = torch.arange(8, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
    q = torch.randn(8, 4, dtype=torch.float16)
    k = torch.randn(8, 2, dtype=torch.float16)

    # 2a. eager call fills the stash with the pre-RoPE q/k of one layer
    capture.record(3, positions, q, k, 4, 2, 8)
    step = capture.drain()
    check("eager call stashes one layer", sorted(step) == [3], f"keys={sorted(step)}")
    if step:
        pos, q_tail, k_rows = step[3]
        check(
            "stashed shapes",
            pos.shape == (8,) and k_rows.shape == (8, 2) and q_tail.shape[1] == 4,
            f"positions={pos.shape} q_tail={q_tail.shape} k={k_rows.shape}",
        )

    # 2b. a single-token (decode) call is guarded out and must not touch the stash
    capture.record(3, positions[:, :1], q[:1], k[:1], 4, 2, 8)
    check("single-token call is guarded out", capture.drain() == {})

    # 3. dynamo accepts it under fullgraph, and the body runs on every execution.
    #    backend="eager" executes the fx graph node by node, which is what makes
    #    "was the implementation called?" observable without an inductor build.
    calls: list[int] = []

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            kk = x * 2.0
            capture.record(7, positions, q, kk, 4, 2, 8)
            calls.append(1)
            return kk

    try:
        compiled = torch.compile(Model(), backend="eager", fullgraph=True, dynamic=False)
        out = compiled(torch.randn(8, 2, dtype=torch.float16))
        check("fullgraph=True compile accepts the capture", True, f"out={tuple(out.shape)}")
        check("implementation ran on the first execution", len(calls) == 1, f"calls={len(calls)}")
        compiled(torch.randn(8, 2, dtype=torch.float16))
        check(
            "implementation ran again on the second execution",
            len(calls) == 2,
            f"calls={len(calls)}",
        )
        check("compiled execution still feeds the stash", sorted(capture.drain()) == [7])
    except Exception as exc:  # noqa: BLE001
        check("fullgraph=True compile accepts the capture", False, f"{type(exc).__name__}: {exc}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("OK: KVMem capture is opaque to torch.compile and still feeds the stash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
