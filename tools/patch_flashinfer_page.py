# -*- coding: utf-8 -*-
"""Apply / revert / verify the flashinfer page.py scalar-validation patch.

The patch caches validated scalar-tensor pointers so the .cpu() sync happens
once per unique tensor instead of once per call (step 025/026: the per-layer
per-step D2H sync serialized the decode pipeline). It lives in the venv's
site-packages, so sync_venv.py cannot see it and a reinstall drops it. This
tool is the in-repo revert path required by iron rule 14.

State machine:
  original  the stock block is present once, the pointer declaration is absent
  patched   the cached block is present once, the declaration is present once
Anything else is "unknown" and the tool refuses to write.

Usage (run from anywhere outside the record repo):
  python tools/patch_flashinfer_page.py verify
  python tools/patch_flashinfer_page.py revert
  python tools/patch_flashinfer_page.py apply

Each write is atomic (temp file + os.replace) and leaves a one-generation
backup next to page.py as page.py.bak_step108. The current venv state is
patched; a revert -> apply round-trip is the acceptance test.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PAGE = Path(r"G:\qwen3.8model\vllm-win029\Lib\site-packages\flashinfer\page.py")

DECL = (
    b"# vllm-030win patch: validated scalar-tensor pointers (see below)\n"
    b"_validated_scalar_ptrs: set = set()\n\n\n"
)

ORIGINAL = b'''    # Validate CUDA scalar tensors with one device-to-host transfer instead of
    # a separate implicit sync for each tensor.
    if cuda_tensors_to_validate:
        host_values = (
            torch.stack([value.reshape(()) for _, value in cuda_tensors_to_validate])
            .detach()
            .cpu()
            .tolist()
        )
        for (name, _), scalar in zip(
            cuda_tensors_to_validate, host_values, strict=True
        ):
            scalar = float(scalar)
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite global decode scale"
                )
'''

PATCHED = b'''    # Validate CUDA scalar tensors with one device-to-host transfer instead of
    # a separate implicit sync for each tensor.
    # vllm-030win patch: cache validated pointers so the .cpu() sync happens
    # once per unique tensor instead of once per call (the per-layer per-step
    # D2H sync serialized the decode pipeline, see step 025/026).
    if cuda_tensors_to_validate:
        to_check = [
            (name, value)
            for name, value in cuda_tensors_to_validate
            if value.data_ptr() not in _validated_scalar_ptrs
        ]
        if to_check:
            host_values = (
                torch.stack([value.reshape(()) for _, value in to_check])
                .detach()
                .cpu()
                .tolist()
            )
            for (name, _), scalar in zip(to_check, host_values, strict=True):
                scalar = float(scalar)
                if not math.isfinite(scalar) or scalar <= 0.0:
                    raise ValueError(
                        f"{name} must be a positive finite global decode scale"
                    )
            for _, value in to_check:
                _validated_scalar_ptrs.add(value.data_ptr())
'''

MARKER = b"def _as_float32_scalar_tensors"


def read_page(path: Path) -> bytes:
    if not path.exists():
        raise SystemExit(f"page.py not found: {path}")
    return path.read_bytes()


def state(data: bytes) -> tuple[str, dict]:
    counts = {
        "original": data.count(ORIGINAL),
        "patched": data.count(PATCHED),
        "decl": data.count(DECL),
        "marker": data.count(MARKER),
    }
    if counts["original"] == 1 and counts["patched"] == 0 and counts["decl"] == 0:
        return "original", counts
    if counts["patched"] == 1 and counts["original"] == 0 and counts["decl"] == 1:
        return "patched", counts
    return "unknown", counts


def write_atomic(path: Path, data: bytes) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak_step108"))
    tmp = path.with_suffix(path.suffix + ".step108.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def apply(path: Path) -> int:
    data = read_page(path)
    st, counts = state(data)
    if st != "original":
        print(f"refuse apply: state={st} counts={counts}")
        return 2
    if counts["marker"] != 1:
        print(f"refuse apply: marker count={counts['marker']}")
        return 2
    out = data.replace(ORIGINAL, PATCHED).replace(MARKER, DECL + MARKER)
    write_atomic(path, out)
    st2, counts2 = state(read_page(path))
    print(f"applied: state={st2} counts={counts2}")
    return 0 if st2 == "patched" else 1


def revert(path: Path) -> int:
    data = read_page(path)
    st, counts = state(data)
    if st != "patched":
        print(f"refuse revert: state={st} counts={counts}")
        return 2
    out = data.replace(PATCHED, ORIGINAL).replace(DECL, b"")
    write_atomic(path, out)
    st2, counts2 = state(read_page(path))
    print(f"reverted: state={st2} counts={counts2}")
    return 0 if st2 == "original" else 1


def verify(path: Path) -> int:
    st, counts = state(read_page(path))
    print(f"state={st} counts={counts} path={path}")
    return 0 if st in ("original", "patched") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("apply", "revert", "verify"))
    ap.add_argument("--page", default=str(PAGE))
    a = ap.parse_args()
    path = Path(a.page)
    if a.action == "apply":
        return apply(path)
    if a.action == "revert":
        return revert(path)
    return verify(path)


if __name__ == "__main__":
    sys.exit(main())
