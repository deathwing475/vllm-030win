#!/usr/bin/env python3
"""Revert the external vllm-exl3 install and its Qwen4Exp vLLM patches.

The Qwen4Exp patch tools create these backups in the installed vLLM package:
``model.py.orig``, ``model.py.orig2``, ``ple_layer.py.orig`` and ``mtp.py.orig``.
This tool restores them before uninstalling the external plugin package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_VENV = Path(r"G:\qwen3.8model\vllm-win029")
SAFE_CWD = Path(r"G:\qwen3.8model")


def _python_for(venv: Path) -> Path:
    candidate = venv / "Scripts" / "python.exe"
    if not candidate.is_file():
        raise SystemExit(f"target venv Python not found: {candidate}")
    return candidate


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=str(SAFE_CWD), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(result.returncode)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)


def _restore(dst: Path, backup: Path) -> bool:
    if not backup.is_file():
        return False
    dst.write_bytes(backup.read_bytes())
    backup.unlink()
    print(f"restored {dst} from {backup}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument(
        "--keep-plugin",
        action="store_true",
        help="restore Qwen4Exp files but leave the vllm-exl3 package installed",
    )
    args = parser.parse_args()

    python = _python_for(args.venv.resolve())
    site = args.venv.resolve() / "Lib" / "site-packages" / "vllm"
    targets = {
        site / "models" / "qwen4_exp" / "nvidia" / "model.py": (".orig2", ".orig"),
        site / "models" / "qwen4_exp" / "nvidia" / "ple_layer.py": (".orig",),
        site / "models" / "qwen4_exp" / "nvidia" / "mtp.py": (".orig",),
    }
    restored = 0
    for dst, suffixes in targets.items():
        for suffix in suffixes:
            backup = Path(str(dst) + suffix)
            if backup.exists():
                restored += int(_restore(dst, backup))
                break

    if not args.keep_plugin:
        _run([str(python), "-m", "pip", "uninstall", "--yes", "vllm-exl3"])

    _run(
        [
            str(python),
            "-c",
            "import vllm; print('vllm import OK', getattr(vllm, '__version__', 'unknown'))",
        ]
    )
    print(f"restored_files={restored}; plugin_removed={not args.keep_plugin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
