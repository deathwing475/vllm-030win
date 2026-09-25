import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


PATCH_NAME = "qwen3_5_quantized_embedding"


def find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")

    if spec is None:
        raise RuntimeError(
            f"Could not find installed vLLM.\nPython executable: {sys.executable}"
        )

    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()

    if spec.origin:
        return Path(spec.origin).resolve().parent

    raise RuntimeError("Could not determine the installed vLLM path.")


def find_qwen35_file(vllm_root: Path) -> Path:
    path = vllm_root / "model_executor" / "models" / "qwen3_5.py"

    if not path.is_file():
        raise RuntimeError(f"Could not find Qwen3.5 implementation at:\n{path}")

    return path


def already_patched(text: str) -> bool:
    pattern = re.compile(
        r"self\.embed_tokens\s*=\s*VocabParallelEmbedding\(\s*"
        r"self\.vocab_size\s*,\s*"
        r"config\.hidden_size\s*,\s*"
        r"quant_config\s*=\s*self\.quant_config\s*,\s*"
        r'prefix\s*=\s*f["\']\{prefix\}\.embed_tokens["\']\s*,?\s*\)',
        re.DOTALL,
    )
    return pattern.search(text) is not None


def patch_source(text: str) -> str:
    class_pos = text.find("class Qwen3_5Model")

    if class_pos == -1:
        raise RuntimeError("Could not find 'class Qwen3_5Model'.")

    search_end = min(len(text), class_pos + 20_000)
    section = text[class_pos:search_end]

    pattern = re.compile(
        r"(?P<indent>^[ \t]+)self\.embed_tokens\s*=\s*VocabParallelEmbedding\(\s*\n"
        r"(?P=indent)[ \t]+self\.vocab_size\s*,\s*\n"
        r"(?P=indent)[ \t]+config\.hidden_size\s*,\s*\n"
        r"(?P=indent)\)",
        re.MULTILINE,
    )

    match = pattern.search(section)

    if match is None:
        raise RuntimeError(
            "Could not find the expected unpatched embedding block. "
            "The installed vLLM source may have changed."
        )

    indent = match.group("indent")
    inner = indent + "    "

    replacement = (
        f"{indent}self.embed_tokens = VocabParallelEmbedding(\n"
        f"{inner}self.vocab_size,\n"
        f"{inner}config.hidden_size,\n"
        f"{inner}quant_config=self.quant_config,\n"
        f'{inner}prefix=f"{{prefix}}.embed_tokens",\n'
        f"{indent})"
    )

    start = class_pos + match.start()
    end = class_pos + match.end()

    return text[:start] + replacement + text[end:]


def atomic_write(path: Path, content: str):
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        shutil.copymode(path, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main():
    print("=" * 70)
    print("vLLM Qwen3.5 quantized embedding patch")
    print("=" * 70)
    print(f"Python: {sys.executable}")

    vllm_root = find_vllm_root()
    target = find_qwen35_file(vllm_root)

    print(f"vLLM:   {vllm_root}")
    print(f"Target: {target}")

    text = target.read_text(encoding="utf-8")

    if already_patched(text):
        print("[OK] Patch is already installed.")
        return 0

    try:
        patched = patch_source(text)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not already_patched(patched):
        print("[ERROR] Generated patch failed verification.", file=sys.stderr)
        return 1

    try:
        compile(patched, str(target), "exec")
    except SyntaxError as exc:
        print(f"[ERROR] Patched source has invalid syntax: {exc}", file=sys.stderr)
        return 1

    backup = target.with_name(target.name + f".{PATCH_NAME}.bak")

    if not backup.exists():
        try:
            shutil.copy2(target, backup)
            print(f"Backup: {backup}")
        except PermissionError:
            print(f"[ERROR] Permission denied creating backup: {backup}", file=sys.stderr)
            return 1
    else:
        print(f"Backup already exists: {backup}")

    try:
        atomic_write(target, patched)
    except PermissionError:
        print(f"[ERROR] Permission denied patching: {target}", file=sys.stderr)
        return 1

    final_text = target.read_text(encoding="utf-8")

    if not already_patched(final_text):
        print("[ERROR] Patch verification failed after writing.", file=sys.stderr)
        return 1

    print("[OK] Successfully patched Qwen3.5 quantized embeddings.")
    print("Restart all vLLM processes before loading the model.")
    print(f"Restore with: cp {backup} {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
