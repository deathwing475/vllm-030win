# sync_venv.py — copy repo sources into the installed venv, preserving each
# side's own line-ending convention, then verify content parity.
#
# Why this exists: the record repo tree is LF for some files and the installed
# venv (vllm-win029\Lib\site-packages\vllm) is CRLF for those same files, so a
# plain byte-for-byte copy or comparison is wrong on both counts. The iron rule
# is content parity per file with the destination's EOL style preserved.
#
# Usage:
#   python tools\sync_venv.py                 # verify only, report mismatches
#   python tools\sync_venv.py --write <rel>   # copy these repo-relative paths
#   python tools\sync_venv.py --write-all     # copy every mismatching file
#
# Paths are relative to the repo's vllm/ root, e.g.
#   model_executor\models\qwen3_next.py

import argparse
import hashlib
import os
import sys

REPO = r"G:\qwen3.8model\vllm-030win-git\vllm"
VENV = r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_norm(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def eol_of(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    bare = data.count(b"\n") - crlf
    if crlf and bare:
        raise ValueError("mixed line endings")
    return "crlf" if crlf else "lf"


def rel_py_files() -> list[str]:
    out = []
    for root, _dirs, files in os.walk(REPO):
        for name in files:
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(root, name), REPO))
    return sorted(out)


def compare(rel: str) -> str:
    """Return 'same' | 'diff' | 'missing'."""
    src = os.path.join(REPO, rel)
    dst = os.path.join(VENV, rel)
    if not os.path.exists(dst):
        return "missing"
    return "same" if _digest(read_norm(src)) == _digest(read_norm(dst)) else "diff"


def sync_one(rel: str) -> str:
    src = os.path.join(REPO, rel)
    dst = os.path.join(VENV, rel)
    with open(src, "rb") as fh:
        repo_bytes = fh.read()
    # Destination decides the EOL style; if the destination does not exist yet,
    # fall back to the repo's own style.
    if os.path.exists(dst):
        with open(dst, "rb") as fh:
            style = eol_of(fh.read())
    else:
        style = eol_of(repo_bytes)
    content = repo_bytes.replace(b"\r\n", b"\n")
    if style == "crlf":
        content = content.replace(b"\n", b"\r\n")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(content)
    return style


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs="*", default=None)
    ap.add_argument("--write-all", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.write_all:
        targets = [r for r in rel_py_files() if compare(r) != "same"]
    elif args.write:
        targets = [t.replace("/", os.sep) for t in args.write]
    else:
        targets = []

    if targets:
        for rel in targets:
            style = sync_one(rel)
            print(f"synced {rel} ({style})")

    mismatches = []
    for rel in rel_py_files():
        state = compare(rel)
        if state != "same":
            mismatches.append((state, rel))

    if mismatches:
        print(f"MISMATCH: {len(mismatches)} file(s) differ between repo and venv")
        for state, rel in mismatches:
            print(f"  {state:8s} {rel}")
        return 1
    if not args.quiet:
        print(f"OK: all {len(rel_py_files())} repo .py files match the venv "
              f"(content-identical, per-file EOL preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
