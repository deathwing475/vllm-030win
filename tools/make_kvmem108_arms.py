# -*- coding: utf-8 -*-
"""Generate the step-108 KVMem pool x context arms from the accepted arms.

The record repo keeps .cmd launchers pure CRLF (iron rule 4). This generator
reads an accepted arm, asserts it is pure CRLF, replaces only the header and
the injectable default block, and writes the variant back with newline="".
Every substitution asserts its expected hit count, so a future edit to the
source arm fails loudly instead of silently producing a stale variant.

Why the joint (pool, W) frontier: batch A fixed W=131,072 / L=200,704 while
lowering the pool, but the engine's startup capacity check needs at least one
request at the configured sliding window. 3.4e9 with W=131,072 measured 1.17x,
so pools below ~3.0e9 need a smaller W, not just a smaller L.

Three arms are generated:
  gsq main/recall  serve_gsq_kvmem108_frontier.cmd  (VIEWPORT 0/1, LOAD 1/0)
  orca main/recall serve_orcasaq2_029_kvmem108.cmd (VIEWPORT 0/1, LOAD 1/0)
  gsq pure serve   serve_gsq_serve108.cmd          (no KVMem; control at equal pool)

Page correction (step 108 read-code): page = 16*cdiv(mamba_raw, 16*1152); it
depends on spec N and model geometry, NOT on SW_WINDOW. GSQ +spec N=2 -> page
1456 and mbt >= 1458 at every W; the old "W -> page" note in load1456 was a
confound with the no-spec arm (W=163,072/no-spec -> 1424).

Usage:
  python tools/make_kvmem108_arms.py [--check]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GSQ_SRC = REPO / "tools" / "serve_gsq_kvmem_load1456.cmd"
ORCA_SRC = REPO / "tools" / "serve_orcasaq2_029_kvmem100.cmd"
SERVE_SRC = REPO / "tools" / "serve_gsq_prod029_n2.cmd"
GSQ_OUT = REPO / "tools" / "serve_gsq_kvmem108_frontier.cmd"
ORCA_OUT = REPO / "tools" / "serve_orcasaq2_029_kvmem108.cmd"
SERVE_OUT = REPO / "tools" / "serve_gsq_serve108.cmd"

GSQ_HEADER = """@echo off
rem ===========================================================================
rem STEP 108 ARM (KVMem line, pool x context frontier). Derived from
rem tools/serve_gsq_kvmem_load1456.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem Shapes: main leg = S108_VIEWPORT=0, S108_LOAD=1 (assembly + speed);
rem recall leg = S108_VIEWPORT=1, S108_LOAD=0 (step 074/078 viewport
rem protocol; the assembly path cannot answer an out-of-window needle --
rem step 066 header says so explicitly).
rem
rem Why: batch A's original fixed W=131,072 / L=200,704 is not bootable below
rem ~3.0e9. The startup check needs at least one request at the configured W,
rem and 3.4e9/W=131,072 measured 1.17x (234,000 tokens), so the pool and W
rem must walk the joint frontier.
rem
rem Page note (step 108 read-code correction): page = 16*cdiv(mamba_raw,
rem 16*1152); it depends on spec N and model geometry, NOT on SW_WINDOW.
rem +spec N=2 => page 1456, mbt >= 1458 at every W. The old W->page note in
rem load1456 was a confound with the no-spec arm (W=163,072/no-spec -> 1424).
rem
rem Injectables: S108_TAG, S108_KVMB, S108_W, S108_ML, S108_MBT, S108_WSMB,
rem S108_LOAD, S108_VIEWPORT, S108_PAGES, S108_PICK (default score = step
rem 081 delivery), S108_SPEC, S108_CAP, S108_PROF.
rem Fixed: RAWK/AUTH/WS/SWEEP/BAKE_VERIFY/SELFTEST, AUTHORITY_TRAJ=2,
rem TOPN=64, VIEWPORT_RECENT=16384, RECENT=32768.
rem ===========================================================================
"""

GSQ_DEFAULTS = """if "%S108_TAG%"=="" set "S108_TAG=kvmem_s108"
if "%S108_KVMB%"=="" set "S108_KVMB=3400000000"
if "%S108_W%"=="" set "S108_W=131072"
if "%S108_ML%"=="" set "S108_ML=200704"
if "%S108_MBT%"=="" set "S108_MBT=1458"
if "%S108_WSMB%"=="" set "S108_WSMB=5120"
if "%S108_LOAD%"=="" set "S108_LOAD=1"
if "%S108_VIEWPORT%"=="" set "S108_VIEWPORT=0"
if "%S108_PAGES%"=="" set "S108_PAGES=55"
if "%S108_PICK%"=="" set "S108_PICK=score"
if "%S108_SPEC%"=="" set "S108_SPEC=1"
if "%S108_CAP%"=="" set "S108_CAP=3"
if "%S108_PROF%"=="" set "S108_PROF=0"
"""

ORCA_HEADER = """@echo off
rem ===========================================================================
rem STEP 108 ORCA ARM (KVMem pool x context frontier). Derived from
rem tools/serve_orcasaq2_029_kvmem100.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem Shapes: main leg = S108_VIEWPORT=0, S108_LOAD=1 (assembly + speed);
rem recall leg = S108_VIEWPORT=1, S108_LOAD=0 (viewport protocol).
rem Eager + no spec (the O1 baseline). The speed verdict is eager same-caliber
rem relative non-degradation only; it must not be compared with the Orca
rem production FULL-graph + DFlash2 78-87 tok/s band.
rem
rem Why: L=16,384 / 49,152 < W=65,536 never triggers sliding-window eviction
rem (skipped <= 0 returns early), so the pool scan must use L > W (131,072)
rem or shrink W below L. 3.1e9/W=65,536 measured 2.11x; the low-pool floor is
rem ~1.6e9 at this W, so the matrix uses 3.1e9 / 2.0e9 / 1.8e9.
rem ===========================================================================
"""

ORCA_DEFAULTS = """if "%S108_TAG%"=="" set "S108_TAG=kvmem_orca_s108"
if "%S108_KVMB%"=="" set "S108_KVMB=3100000000"
if "%S108_W%"=="" set "S108_W=65536"
if "%S108_ML%"=="" set "S108_ML=131072"
if "%S108_MBT%"=="" set "S108_MBT=1456"
if "%S108_WSMB%"=="" set "S108_WSMB=5120"
if "%S108_LOAD%"=="" set "S108_LOAD=1"
if "%S108_VIEWPORT%"=="" set "S108_VIEWPORT=0"
if "%S108_PAGES%"=="" set "S108_PAGES=40"
if "%S108_PICK%"=="" set "S108_PICK=score"
if "%S108_EVERY%"=="" set "S108_EVERY=12"
if "%S108_KEEP%"=="" set "S108_KEEP=10"
"""

SERVE_HEADER = """@echo off
rem ===========================================================================
rem STEP 108 GSQ PURE-SERVE CONTROL ARM. Derived from
rem tools/serve_gsq_prod029_n2.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem No VLLM_KVMEM_* environment: this is the production stack with the pool,
rem max-model-len and mbt exposed. It answers "what does KVMem buy at the same
rem pool" -- KVMem can accept a longer prompt (exact attention inside W plus
rem retrieval outside it), while pure serve can only accept a shorter prompt
rem with full exact attention.
rem
rem At L=163,072 the 3.4e9 pool is exactly one request (163,719 tokens,
rem 1.004x); every lower pool needs a lower L, so the control is a staircase,
rem not a fixed-L scan. L changes trigger a first-compile boot (step 093:
rem 3.29 tok/s), so each new point is compile boot + sample boot.
rem ===========================================================================
if "%S108_KVMB%"=="" set "S108_KVMB=3400000000"
if "%S108_ML%"=="" set "S108_ML=163072"
if "%S108_MBT%"=="" set "S108_MBT=1024"
"""


def _read_crlf(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    assert "\n" not in text.replace("\r\n", ""), f"{path} is not pure CRLF"
    return text


def _replace(text: str, pattern: str, repl: str, count: int = 1) -> str:
    hits = list(re.finditer(pattern, text))
    assert len(hits) == count, (
        f"pattern {pattern!r} matched {len(hits)} times, expected {count}"
    )
    for m in reversed(hits):
        text = text[: m.start()] + repl + text[m.end() :]
    return text


def _replace_header(text: str, first_marker: str, header: str) -> str:
    start = text.index(first_marker)
    return header + text[start:]


def _replace_defaults(text: str, first_marker: str, next_marker: str, block: str) -> str:
    start = text.index(first_marker)
    end = text.index(next_marker)
    return text[:start] + block + text[end:]


def _to_crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def build_gsq() -> str:
    text = _read_crlf(GSQ_SRC)
    text = _replace_header(text, 'if "%S079_TAG%"==""', GSQ_HEADER)
    text = _replace_defaults(text, 'if "%S079_TAG%"==""', 'set "SPEC_ARGS="', GSQ_DEFAULTS)
    text = _replace(text, r"%S079_SPEC%", "%S108_SPEC%", count=1)
    text = _replace(text, r"%S081_PROF%", "%S108_PROF%", count=1)
    text = _replace(text, r"%S079_TAG%", "%S108_TAG%", count=6)
    text = _replace(text, r"%S082_KVMB%", "%S108_KVMB%", count=1)
    text = _replace(text, r"%S082_WSMB%", "%S108_WSMB%", count=1)
    text = _replace(text, r"%S082_MBT%", "%S108_MBT%", count=1)
    text = _replace(text, r"%S082_CAP%", "%S108_CAP%", count=1)
    text = _replace(text, r'--max-model-len 200704', '--max-model-len %S108_ML%', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_SW_WINDOW=131072"',
                    'set "VLLM_KVMEM_SW_WINDOW=%S108_W%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_WORKSPACE=%S079_WS%"',
                    'set "VLLM_KVMEM_WORKSPACE=1"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_RAWK=%S079_RAWK%"',
                    'set "VLLM_KVMEM_RAWK=1"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_AUTHORITY=%S079_AUTH%"',
                    'set "VLLM_KVMEM_AUTHORITY=1"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_LOAD=%S082_LOAD%"',
                    'set "VLLM_KVMEM_LOAD=%S108_LOAD%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_VIEWPORT=0"',
                    'set "VLLM_KVMEM_VIEWPORT=%S108_VIEWPORT%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_VIEWPORT_PAGES=%S079_PAGES%"',
                    'set "VLLM_KVMEM_VIEWPORT_PAGES=%S108_PAGES%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_VIEWPORT_RECENT=%S079_VRECENT%"',
                    'set "VLLM_KVMEM_VIEWPORT_RECENT=16384"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_RECENT=%S079_RECENT%"',
                    'set "VLLM_KVMEM_RECENT=32768"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_TOPN=%S079_TOPN%"',
                    'set "VLLM_KVMEM_TOPN=64"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_TIMING=%S079_TIMING%"',
                    'set "VLLM_KVMEM_TIMING=1"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_TIMING_EVERY=%S079_EVERY%"',
                    'set "VLLM_KVMEM_TIMING_EVERY=25"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_SLOT_PICK=%S081_PICK%"',
                    'set "VLLM_KVMEM_SLOT_PICK=%S108_PICK%"', count=1)
    text = _replace(text, r'set "VLLM_DBG_PIN=1"',
                    'set "VLLM_DBG_PIN=1"\r\nset "VLLM_SERVER_DEV_MODE=1"', count=1)
    return _to_crlf(text)


def build_orca() -> str:
    text = _read_crlf(ORCA_SRC)
    text = _replace_header(text, 'if "%S100_TAG%"==""', ORCA_HEADER)
    text = _replace_defaults(text, 'if "%S100_TAG%"==""', 'set "HOME=C:\\fi"', ORCA_DEFAULTS)
    text = _replace(text, r"%S100_TAG%", "%S108_TAG%", count=1)
    text = _replace(text, r"%S100_LOAD%", "%S108_LOAD%", count=1)
    text = _replace(text, r"%S100_ML%", "%S108_ML%", count=1)
    text = _replace(text, r"%S100_KVMB%", "%S108_KVMB%", count=1)
    text = _replace(text, r"%S100_MBT%", "%S108_MBT%", count=1)
    text = _replace(text, r"%S100_WSMB%", "%S108_WSMB%", count=1)
    text = _replace(text, r"%S100_EVERY%", "%S108_EVERY%", count=1)
    text = _replace(text, r"%S100_KEEP%", "%S108_KEEP%", count=1)
    text = _replace(text, r"%S100_PAGES%", "%S108_PAGES%", count=1)
    text = _replace(text, r"%S100_VRECENT%", "16384", count=1)
    text = _replace(text, r"%S100_RECENT%", "16384", count=1)
    text = _replace(text, r"%S100_SLOTPOLICY%", "%S108_SLOTPOLICY%", count=2)
    text = _replace(text, r"%S100_INTERLEAVE%", "%S108_INTERLEAVE%", count=2)
    text = _replace(text, r'set "VLLM_KVMEM_SW_WINDOW=65536"',
                    'set "VLLM_KVMEM_SW_WINDOW=%S108_W%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_VIEWPORT=0"',
                    'set "VLLM_KVMEM_VIEWPORT=%S108_VIEWPORT%"', count=1)
    text = _replace(text, r'set "VLLM_KVMEM_SLOT_PICK=time"',
                    'set "VLLM_KVMEM_SLOT_PICK=%S108_PICK%"', count=1)
    return _to_crlf(text)


def build_gsq_serve() -> str:
    text = _read_crlf(SERVE_SRC)
    # The production launcher has no injectable block: replace its long header
    # but keep the environment setup (the first "if not exist" line onward).
    first_env = text.index("if not exist")
    text = SERVE_HEADER + text[first_env:]
    text = _replace(text, r"--kv-cache-memory-bytes 3400000000",
                    "--kv-cache-memory-bytes %S108_KVMB%", count=1)
    text = _replace(text, r"--max-model-len 163072",
                    "--max-model-len %S108_ML%", count=1)
    text = _replace(text, r"--max-num-batched-tokens 1024",
                    "--max-num-batched-tokens %S108_MBT%", count=1)
    return _to_crlf(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the generated files are pure CRLF and clean of old knobs")
    a = ap.parse_args()

    out = {
        "GSQ": (GSQ_OUT, build_gsq()),
        "ORCA": (ORCA_OUT, build_orca()),
        "SERVE": (SERVE_OUT, build_gsq_serve()),
    }
    for name, (_, text) in out.items():
        assert "\n" not in text.replace("\r\n", ""), f"{name} output is not pure CRLF"
        leftovers = re.findall(r"%S(?:079|081|082|100)_[A-Z]+%", text)
        assert not leftovers, f"{name} has leftover knobs: {leftovers}"

    if a.check:
        print("check ok: all arms are pure CRLF with no leftover S079/S081/S082/S100 knobs")
        return 0

    for name, (path, text) in out.items():
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"wrote {path.relative_to(REPO)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
