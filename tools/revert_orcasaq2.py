#!/usr/bin/env python3
"""Revert the OrcaSAQ2 validation install from vllm-win029."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

DEFAULT_VENV = Path(r"G:\qwen3.8model\vllm-win029")
DEFAULT_MODEL = Path(r"G:\qwen3.8model\qwen3.8exl3")
SAFE_CWD = Path(r"G:\qwen3.8model")


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=str(SAFE_CWD), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keep-exllamav3", action="store_true")
    parser.add_argument("--keep-orcasaq2", action="store_true")
    args = parser.parse_args()
    python = args.venv.resolve() / "Scripts" / "python.exe"
    if not args.keep_orcasaq2:
        run([str(python), "-m", "pip", "uninstall", "--yes", "orcasaq2-kernel"])
    if not args.keep_exllamav3:
        run([str(python), "-m", "pip", "uninstall", "--yes", "exllamav3"])
    alias = args.model.resolve() / "model.safetensors.index.json"
    original = args.model.resolve() / "model.safetensors.index (1).json"
    if alias.exists() and original.exists() and alias.read_bytes() == original.read_bytes():
        alias.unlink()
        print(f"removed generated index alias: {alias}")
    run([str(python), "-c", "import vllm; print('vllm import OK', vllm.__version__)"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
