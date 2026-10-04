"""Step 096 regression anchor: the KVMem config derivation chain must
reproduce the historical GSQ defaults bit-for-bit.

Two runs, both from OUTSIDE the record repo (iron rule 2: never start
python inside it), against the venv's vllm (the copy the engine actually
imports, kept content-identical by tools/sync_venv.py):

  1. fallback mode (no VLLM_PROFILE_CARD): every knob returns the
     pre-096 constant -- proves the chain is inert without a card.
  2. card mode (VLLM_PROFILE_CARD=profile/<GSQ card>.json): every knob
     returns the same value via the card's kvmem_workspace derivation --
     proves the formulas re-derive the step 060/066/072 filings exactly
     (GSQ: 262144 / 3072 / 20 / 16384 / 32768 / 2 / 55 / 256 / 512 / 128).

Usage:
  python tools/profile_card/verify_kvmem_defaults.py [--card PATH]

Exit 0 = all anchors match; nonzero with a diff listing otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys

# (knob, expected value) -- the pre-096 constants, i.e. the frozen GSQ
# defaults the 083 exit ran on. viewport_retrieval_pages is probed both
# with the engine page size the 083 arm resolves (1,424) and without one
# (the no-card fallback path).
GSQ_PAGE_TOKENS = 1424
ANCHORS = [
    ("workspace_host_bytes", None, 3072 << 20),
    ("trajectory_prefix_tokens", None, 512),
    ("max_workspace_tokens", None, 262144),
    ("authority_tokens", None, 262144),
    ("snapshot_keep", None, 20),
    ("snapshot_trajectories", None, 2),
    ("authority_trajectories", None, 2),
    ("index_subblock", None, 128),
    ("query_span", None, 256),
    ("recent_tokens", None, 32768),
    ("viewport_recent_tokens", None, 16384),
    ("viewport_retrieval_pages", GSQ_PAGE_TOKENS, 55),
    ("viewport_retrieval_pages", None, 55),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=None,
                    help="profile card path; omit for fallback mode")
    args = ap.parse_args()

    if args.card:
        os.environ["VLLM_PROFILE_CARD"] = args.card
    else:
        os.environ.pop("VLLM_PROFILE_CARD", None)

    from vllm.v1.kvmem_workspace import config

    mode = f"card={args.card}" if args.card else "fallback (no card)"
    print(f"[verify_kvmem_defaults] mode: {mode}")

    failures = []
    for name, page_tokens, expected in ANCHORS:
        fn = getattr(config, name)
        got = fn(page_tokens) if page_tokens is not None else fn()
        status = "OK" if got == expected else "MISMATCH"
        line = f"  {status}: {name}({page_tokens}) = {got} (expect {expected})"
        print(line)
        if got != expected:
            failures.append(line)

    if failures:
        print(f"[verify_kvmem_defaults] FAIL: {len(failures)} mismatch(es)")
        return 1
    print("[verify_kvmem_defaults] PASS: all anchors bit-for-bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
