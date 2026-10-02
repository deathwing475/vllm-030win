# -*- coding: utf-8 -*-
"""步骤 079 计时台账的 CPU 单测（不起引擎、不碰 GPU）。

为什么要这个
------------
``[KVTIME]`` 台账是从 ``wait_for_save`` 里打出来的。格式化字符串一旦出错，抛的就是
引擎主循环里的异常 —— 那会在一支慢态 boot 跑到一半时打死 EngineCore，整个 boot 白赔。
所以在 CPU 上按假 ``self`` 把 ``_kvtime_add`` / ``_kvtime_bump`` / ``_kvtime_step``
跑一遍，确认：

1. 门控关时一个字都不产生，也不建累加器（生产默认路径）；
2. ``loaded`` 自证行每 boot 恰好一条且带几何信息（"改了文件 ≠ 在跑"的证据）；
3. 每 ``timing_every`` 次 ``wait_for_save`` 出恰好一条台账；
4. 首条台账 ``dt=0.000``（窗口未建立时不许拿 ``wall=0`` 相减出假读数）；
5. 十一个段字段齐全、可按 float 解析，且 ``fold == ingest - drain`` 自洽。

用法
----
    G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe tools\\kvmem_timing_unit_test.py
"""
import os
import re
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["VLLM_KVMEM_TIMING"] = "1"
os.environ["VLLM_KVMEM_TIMING_EVERY"] = "3"

from vllm.v1.kvmem_workspace import worker as worker_mod  # noqa: E402

W = worker_mod.KVMemWorkspaceWorker
CAPTURED: list[str] = []

# One ledger segment is emitted as four adjacent tokens:
#   <key>=<absolute> d<key>=<window delta> n<count> dn<count delta>
# The backreference on \1 is what keeps ``drain``/``ddrain`` from being read as
# each other. The pattern lives in the offline analyzer (tools/kvmem_time_budget.py)
# so a format change there fails this test instead of drifting silently.
from kvmem_time_budget import SEG_RE  # noqa: E402


class _Log:
    """Swap the module logger for a recorder (the patched code logs via it)."""

    def __enter__(self):
        self._real = worker_mod.logger
        worker_mod.logger = self  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        worker_mod.logger = self._real

    def info(self, msg, *args):
        CAPTURED.append(str(msg % args) if args else str(msg))

    def warning(self, msg, *args):
        CAPTURED.append("WARN " + str(msg))

    def error(self, msg, *args):
        CAPTURED.append("ERROR " + str(msg))


def stub(enabled: bool, every: int = 3) -> types.SimpleNamespace:
    """A stand-in for the worker holding only what the ledger reads."""
    return types.SimpleNamespace(
        _KVTIME_SEGMENTS=W._KVTIME_SEGMENTS,
        _kvtime_enabled=enabled,
        _kvtime_every=every,
        _kvtime={},
        _kvtime_n={},
        _kvtime_prev={},
        _kvtime_prev_n={},
        _kvtime_wall=0.0,
        _score_seconds=4.5,
        bytes_stored=3 * 25.59 * 1048576,
        page_bytes={6: 1677312, 7: 1677312},
        _layers_per_group={6: [f"L{i}" for i in range(8)],
                           7: [f"L{i}" for i in range(8)]},
        _host={(6, "L0"): None, (7, "L0"): None},
    )


def drive(s, seconds: dict[str, float]) -> None:
    """Record each named segment as if it had taken ``seconds``."""
    for key, took in seconds.items():
        W._kvtime_add(s, key, time.monotonic() - took)
    W._kvtime_bump(s, "entry", 16)
    W._kvtime_bump(s, "busy")


def ledger_lines() -> list[str]:
    return [x for x in CAPTURED if "[KVTIME] wall=" in x]


def fields(line: str) -> dict[str, str]:
    out = {}
    for chunk in line.replace(" | ", " ").split():
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            out[key] = value
    return out


def check(cond: bool, label: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        raise AssertionError(label)


def main() -> int:
    print("=== 1. 门控关：不产生任何台账（生产默认路径）===")
    with _Log():
        CAPTURED.clear()
        s = stub(enabled=False)
        W._kvtime_add(s, "drain", time.monotonic() - 1.0)
        W._kvtime_bump(s, "busy")
        W._kvtime_step(s)
        check(CAPTURED == [], "gate off -> zero log lines")
        check(s._kvtime == {} and s._kvtime_n == {}, "gate off -> no accumulators")

    print("=== 2. 每 every 次 save 出恰好一条；首条不许有假 dt ===")
    with _Log():
        CAPTURED.clear()
        s = stub(enabled=True, every=3)
        for i in range(1, 10):
            drive(s, {"drain": 0.5 * i, "ingest": 0.6 * i, "save": 0.7 * i})
            W._kvtime_step(s)
            time.sleep(0.005)  # keep dt above the 3-decimal print resolution
        ledger = ledger_lines()
        loaded = [x for x in CAPTURED if "patch loaded" in x]
        check(len(loaded) == 1, f"loaded line exactly once (got {len(loaded)})")
        check(len(ledger) == 3, f"9 steps / every=3 -> 3 lines (got {len(ledger)})")
        check("page_bytes=" in loaded[0] and "store_groups=" in loaded[0],
              "loaded line carries the geometry")
        first = fields(ledger[0])
        check(first["dt"] == "0.000", f"first window dt=0 (got {first['dt']})")
        check(first["acc"] == "0.000", f"first window acc guarded (got {first['acc']})")
        check(first["dsteps"] == "3", f"first window dsteps (got {first['dsteps']})")
        second = fields(ledger[1])
        check(float(second["dt"]) > 0.0, f"second window dt>0 (got {second['dt']})")
        check(float(second["drain"]) > float(first["drain"]), "accumulators grow")
        check(fields(ledger[2])["dsteps"] == "3", "window step count rolls over")

    print("=== 3. 段字段齐全、可解析、fold 自洽、计数器在位 ===")
    with _Log():
        CAPTURED.clear()
        s = stub(enabled=True, every=2)
        drive(s, {"drain": 44.0, "ingest": 46.0, "storing": 1.5,
                  "selftest": 0.4, "copy": 0.9, "score": 0.0, "bake": 0.0,
                  "snap": 0.0, "load": 0.0, "save": 47.0})
        first_dt = time.time()                 # when the first line hit the log
        W._kvtime_step(s)                      # first line: dt is 0 by design
        s._kvtime_wall = first_dt - 10.0       # pretend a 10 s page-step window
        drive(s, {"drain": 1.0, "ingest": 1.0, "save": 1.0})
        W._kvtime_step(s)
        f = fields(ledger_lines()[-1])
        for key in W._KVTIME_SEGMENTS:
            check(key in f and ("d" + key) in f, f"field {key} / d{key} present")
            float(f[key])
            float(f["d" + key])
        check(
            abs(float(f["fold"]) - (float(f["ingest"]) - float(f["drain"]))) < 5e-4,
            f"fold == ingest - drain ({f['fold']})",
        )
        check(abs(float(f["dt"]) - 10.0) < 0.5, f"dt = injected window ({f['dt']})")
        check(
            abs(float(f["acc"]) - float(f["dsave"]) / float(f["dt"])) < 5e-3,
            f"acc == dsave/dt ({f['acc']} vs "
            f"{float(f['dsave']) / float(f['dt']):.3f})",
        )
        check(float(f["score"]) >= 4.5, f"score folded from _score_seconds ({f['score']})")
        check(f["copy_calls"] == "1" and f["entries"] == "32" and f["busy"] == "2",
              f"counters (copy_calls={f['copy_calls']} entries={f['entries']} "
              f"busy={f['busy']})")
        check(len(ledger_lines()[-1]) < 700, f"line length {len(ledger_lines()[-1])} < 700")

    print("=== 4. 离线分析器口径：段名与增量名必须能无歧义配对 ===")
    print("    （注意 ``drain`` 与它的增量 ``ddrain``：按前缀切会串味，")
    print("      所以台账用 <key>=<acc> d<key>=<delta> n<n> dn<n> 的四元组，")
    print("      分析器必须用带反向引用的正则取段。）")
    with _Log():
        CAPTURED.clear()
        s = stub(enabled=True, every=1)
        # ingest must be >= drain: in the engine drain() is the first call
        # inside _ingest(), so a negative fold would mean the ledger is wrong.
        drive(s, {"drain": 2.0, "ingest": 2.5, "save": 3.0})
        W._kvtime_step(s)
        line = ledger_lines()[-1]
        found = {
            m.group(1): float(m.group(3))
            for m in SEG_RE.finditer(line)
        }
        check(set(found) == set(W._KVTIME_SEGMENTS),
              f"SEG_RE recovers all {len(W._KVTIME_SEGMENTS)} segments "
              f"(got {sorted(found)})")
        check(all(v >= 0.0 for v in found.values()),
              "every d<segment> parses as a non-negative float")
        check(found["fold"] == 0.5, f"fold = ingest - drain = 0.5 ({found['fold']})")
        check(found["drain"] == 2.0 and found["save"] == 3.0,
              f"no drain/ddrain mix-up: {found['drain']} / {found['save']}")

    print(f"\nPASS: [KVTIME] 台账门控正确、自证在位、{len(W._KVTIME_SEGMENTS)} 段齐全可解析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
