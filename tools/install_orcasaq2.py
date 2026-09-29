#!/usr/bin/env python3
"""Install the OrcaSAQ2 EXL3 stack into the Windows vLLM 0.29 venv.

This keeps the model and third-party source trees outside this repository. It
replaces the generic vllm-exl3 registration, installs the matching ExLlamaV3
CUDA extension, installs the OrcaSAQ2 vLLM plugin, and creates the standard
checkpoint index name when the source directory contains a Windows duplicate
such as ``model.safetensors.index (1).json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_VENV = Path(r"G:\qwen3.8model\vllm-win029")
DEFAULT_EXLLAMA = Path(r"G:\exllamav3-master")
DEFAULT_ORCA = Path(r"G:\orcasaq2-kernel")
DEFAULT_MODEL = Path(r"G:\qwen3.8model\qwen3.8exl3")
DEFAULT_STATE = DEFAULT_VENV / "orcasaq2_install_state.json"
DEFAULT_VS_BAT = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
SAFE_CWD = Path(r"G:\qwen3.8model")


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
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


def build_exllamav3(python: Path, source: Path, vs_bat: Path) -> None:
    if not vs_bat.is_file():
        raise SystemExit(f"Visual Studio vcvars64.bat not found: {vs_bat}")
    cuda_home = r"C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
    command = (
        f'call "{vs_bat}" && '
        'set "DISTUTILS_USE_SDK=1" && '
        'set "MAX_JOBS=4" && '
        f'set "CUDA_HOME={cuda_home}" && set "CUDA_PATH={cuda_home}" && '
        f'cd /d "{source}" && "{python}" -m pip install '
        '--no-build-isolation --no-deps .'
    )
    run([os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", command], cwd=SAFE_CWD)


def verify(python: Path, model: Path) -> dict[str, object]:
    code = r'''
import importlib.metadata as md
import json
from pathlib import Path
import torch
import vllm
import orcasaq2
from exllamav3.ext import exllamav3_ext

orcasaq2.register()
from vllm.model_executor.layers.quantization import get_quantization_config
qcfg = json.loads((Path(r"MODEL").resolve() / "quantization_config.json").read_text())
parsed = get_quantization_config("exl3").from_config(qcfg)
print(json.dumps({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "orcasaq2": md.version("orcasaq2-kernel"),
    "exllamav3": md.version("exllamav3"),
    "extension": str(exllamav3_ext.__file__),
    "symbols": [name for name in ("reconstruct", "exl3_gemm", "exl3_moe") if hasattr(exllamav3_ext, name)],
    "quant_config": repr(parsed),
    "index_present": (Path(r"MODEL").resolve() / "model.safetensors.index.json").is_file(),
}, sort_keys=True))
'''.replace("MODEL", str(model).replace("\\", "\\\\"))
    return json.loads(run([str(python), "-c", code]).strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--exllama-root", type=Path, default=DEFAULT_EXLLAMA)
    parser.add_argument("--orca-root", type=Path, default=DEFAULT_ORCA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vs-bat", type=Path, default=DEFAULT_VS_BAT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    venv = args.venv.resolve()
    python = venv / "Scripts" / "python.exe"
    exllama = args.exllama_root.resolve()
    orca = args.orca_root.resolve()
    model = args.model.resolve()
    if not python.is_file():
        raise SystemExit(f"venv Python not found: {python}")
    for root, name in ((exllama, "ExLlamaV3"), (orca, "OrcaSAQ2 plugin"), (model, "checkpoint")):
        if not root.is_dir():
            raise SystemExit(f"{name} not found: {root}")

    run([str(python), "-m", "pip", "uninstall", "--yes", "vllm-exl3"])
    if not args.skip_build:
        build_exllamav3(python, exllama, args.vs_bat)
    run([str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", str(orca)])

    source_index = model / "model.safetensors.index (1).json"
    standard_index = model / "model.safetensors.index.json"
    created_index = False
    if not standard_index.exists() and source_index.exists():
        shutil.copyfile(source_index, standard_index)
        created_index = True
        print(f"created checkpoint index alias: {standard_index}")

    result = verify(python, model)
    state = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "venv": str(venv),
        "exllamav3_root": str(exllama),
        "orcasaq2_root": str(orca),
        "model": str(model),
        "created_index_alias": created_index,
        "production_launcher_changed": False,
        "verification": result,
    }
    args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
