#!/usr/bin/env python3
"""Install the external vllm-exl3 plugin into the 0.29 Windows venv.

This tool deliberately keeps vllm-exl3 outside this repository's vllm/ tree.
The default is the plugin's supported Python-only install because the Windows
runtime may not have ExLlamaV3 headers/extension sources. Pass --cuda only when
those native build prerequisites have been independently verified.

Examples:
  python tools/install_vllm_exl3.py
  python tools/install_vllm_exl3.py --plugin-root G:\\vllm-exl3-0.5.0
  python tools/install_vllm_exl3.py --cuda --skip-qwen4-patches
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_PLUGIN_ROOT = Path(r"G:\vllm-exl3-0.5.0")
DEFAULT_VENV = Path(r"G:\qwen3.8model\vllm-win029")
DEFAULT_STATUS = DEFAULT_VENV / "vllm_exl3_install_state.json"
SAFE_CWD = Path(r"G:\qwen3.8model")
PATCH_TOOLS = (
    "patch_vllm_qwen4_ple.py",
    "patch_vllm_vision_split.py",
    "patch_vllm_mtp_lmhead.py",
)


def _python_for(venv: Path) -> Path:
    candidate = venv / "Scripts" / "python.exe"
    if not candidate.is_file():
        raise SystemExit(f"target venv Python not found: {candidate}")
    return candidate


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd or SAFE_CWD),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(result.returncode)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.stdout


def _plugin_revision(root: Path) -> dict[str, str | None]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"kind": "source-snapshot", "commit": None}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"kind": "git-checkout", "commit": commit}


def _verify_runtime(python: Path) -> dict[str, Any]:
    code = r'''
import importlib.metadata as md
import json
import torch
import vllm
import vllm_exl3

vllm_exl3.register()
vllm_exl3.register()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
print(json.dumps({
    "python": __import__("sys").executable,
    "vllm": getattr(vllm, "__version__", "unknown"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "plugin": vllm_exl3.__file__,
    "entry_points": [str(x) for x in md.entry_points(group="vllm.general_plugins") if x.name == "vllm_exl3"],
    "exl3_registered": "exl3" in QUANTIZATION_METHODS,
    "config_class": get_quantization_config("exl3").__name__,
}, sort_keys=True))
'''
    output = _run([str(python), "-c", code])
    return json.loads(output.strip().splitlines()[-1])


def _patches(python: Path, root: Path, site: Path) -> list[str]:
    patch_dir = root / "tools" / "patch_vllm_qwen4_exp"
    applied: list[str] = []
    for name in PATCH_TOOLS:
        tool = patch_dir / name
        if not tool.is_file():
            raise SystemExit(f"patch tool not found: {tool}")
        _run([str(python), str(tool), str(site)])
        applied.append(name)
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, default=DEFAULT_PLUGIN_ROOT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="attempt the native CUDA extension; prerequisites must already exist",
    )
    parser.add_argument(
        "--skip-qwen4-patches",
        action="store_true",
        help="install only; do not modify the target vLLM Qwen4Exp files",
    )
    args = parser.parse_args()

    root = args.plugin_root.resolve()
    venv = args.venv.resolve()
    pyproject = root / "pyproject.toml"
    setup = root / "setup.py"
    if not pyproject.is_file() or not setup.is_file():
        raise SystemExit(f"not a vllm-exl3 source root: {root}")
    python = _python_for(venv)
    site = venv / "Lib" / "site-packages" / "vllm"
    if not site.is_dir():
        raise SystemExit(f"target vLLM package not found: {site}")

    env = os.environ.copy()
    if not args.cuda:
        env["VLLM_EXL3_NO_CUDA"] = "1"
    else:
        env.pop("VLLM_EXL3_NO_CUDA", None)

    _run(
        [str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "."],
        cwd=root,
        env=env,
    )

    patched: list[str] = []
    if not args.skip_qwen4_patches:
        patched = _patches(python, root, site)

    runtime = _verify_runtime(python)
    native: dict[str, Any] = {}
    native_code = (
        "import json\n"
        "result = {}\n"
        "for name in ('exllamav3', 'exllamav3_ext', 'vllm_exl3_c'):\n"
        "    try:\n"
        "        mod = __import__(name)\n"
        "        result[name] = {'available': True, 'file': getattr(mod, '__file__', None), 'abi': getattr(mod, 'P2B_MOE_ABI_VERSION', None)}\n"
        "    except Exception as exc:\n"
        "        result[name] = {'available': False, 'error': type(exc).__name__ + ': ' + str(exc)}\n"
        "print(json.dumps(result, sort_keys=True))"
    )
    native = json.loads(_run([str(python), "-c", native_code]).strip().splitlines()[-1])

    status = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plugin_root": str(root),
        "plugin_revision": _plugin_revision(root),
        "venv": str(venv),
        "native_requested": bool(args.cuda),
        "qwen4_patches": patched,
        "runtime": runtime,
        "native_modules": native,
        "production_launcher_changed": False,
    }
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"status written: {args.status_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
