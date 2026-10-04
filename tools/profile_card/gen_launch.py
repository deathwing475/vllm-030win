"""Launcher generator for the 092 generic serve path.

Human-intent parameters ONLY (design doc §3.3): model card name / L / spec
family / tier (+ optional port/pool overrides). Every other value comes from
the profile card (derived) or the platform card (platform). A `human-copied`
value used by the generator is printed as an explicit WARNING - copied values
are debt, never silently consumed.

Usage:
  python tools/profile_card/gen_launch.py --model <card-name-or-path>
      [--l TOKENS] [--tier speed|capacity|auto] [--family dflash2|none]
      [--spec N] [--pool BYTES] [--port P] [--out FILE.cmd]

If the model has no card yet, one is built on the spot (build_card.py) - a
new model therefore requires zero hand-written numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROFILE_DIR = os.path.join(REPO, "profile")
PLATFORM_CARD = os.path.join(HERE, "platform_card.json")
VENV_PYTHON = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
DEFAULT_PORT = 8001


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_card(model: str, draft: str | None, num_spec: int) -> tuple[str, dict]:
    """Return (card_path, card). model = card name or checkpoint path.
    The card is a function of (model, draft, spec family) - build it with the
    same intent parameters the launcher will run, or geometry silently
    describes the wrong tier (no-draft conv states are 2 columns shorter)."""
    name = os.path.basename(os.path.normpath(model))
    card_path = os.path.join(PROFILE_DIR, f"{name}.json")
    if os.path.isfile(card_path):
        return card_path, _load_json(card_path)
    if os.path.isdir(model):
        print(f"[gen_launch] no card for {name}; building one (zero-boot)...")
        cmd = [sys.executable, os.path.join(HERE, "build_card.py"), model,
               "--name", name, "--out-dir", PROFILE_DIR,
               "--num-spec", str(num_spec)]
        if draft:
            cmd += ["--draft", draft]
        subprocess.run(cmd, check=True)
        return card_path, _load_json(card_path)
    raise FileNotFoundError(f"no card {card_path} and {model} is not a directory")


def derive_pool(card: dict, tier: str, L: int, pool_override: int | None,
                warnings: list[str]) -> int | None:
    if pool_override:
        return pool_override
    if tier == "auto":
        warnings.append("tier=auto: engine-sized pool is a KNOWN speed trap "
                        "on this card under graphs+DFlash2 (step 090)")
        return None
    nd = card["need_decomposition"]
    a, b = nd["a_bytes"], nd["b_bytes_per_token"]
    if tier == "speed":
        # single-request need plus 20% headroom for block rounding and the
        # draft window; calibrated against step 090 (8e8 @ L=16,384 vs
        # need 0.63 GiB -> healthy band)
        return int((a + b * L) * 1.2)
    if tier == "capacity":
        warnings.append("tier=capacity pool 3.4e9 is human-copied from "
                        "production (the residency-cliff ceiling of this "
                        "card); not derived")
        return 3_400_000_000
    raise ValueError(tier)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--l", type=int, default=None, dest="max_len")
    ap.add_argument("--tier", default="speed",
                    choices=["speed", "capacity", "auto"])
    ap.add_argument("--family", default="dflash2", choices=["dflash2", "none"])
    ap.add_argument("--draft", default=r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c",
                    help="draft checkpoint; the only draft family on this "
                         "stack today (debt: default is a project fact, not "
                         "a derivation)")
    ap.add_argument("--spec", type=int, default=None)
    ap.add_argument("--pool", type=int, default=None)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", required=True, help="generated .cmd path")
    args = ap.parse_args()

    card_path, card = ensure_card(
        args.model, args.draft if args.family != "none" else None,
        args.spec or 2)
    platform = _load_json(PLATFORM_CARD)
    warnings: list[str] = []

    geo_block = card["capacity"]
    nd = card["need_decomposition"]
    spec_family = card.get("spec_family") or {}
    quant = card["source_flags"]["quantization_method"]

    L = args.max_len
    # L default: the card's own single-request ceiling keeps the human free
    # of numbers; auto tier falls back to a conservative 16k (its pool is
    # engine-sized, so the card ceiling does not apply).
    if L is None:
        L = nd.get("single_request_l_ceiling") or 16384
        if args.tier == "auto":
            L = 16384
    num_spec = args.spec or (spec_family.get("draft_slots") or 2)

    pool = derive_pool(card, args.tier, L, args.pool, warnings)
    # mbt: two resolved-safe shapes exist. mbt >= block+draft_slots (1458)
    # forces the aligned path but costs the draft SWA group one admission
    # block (-0.4% capacity); any mbt < block-draft_slots keeps the chunked
    # path with full capacity - production and steps 089/090 both run 1024
    # with zero ERROR across both models, so that is the default (debt: the
    # exact "best" chunked mbt is a prefill-throughput question, not safety).
    mbt = 1024
    warnings.append("mbt=1024 is human-copied (production + 089/090, both "
                    f"models, zero ERROR); resolved-safe aligned value = "
                    f"{nd['mbt_min']} costs ~0.4% capacity")

    # guard: never launch with an L the pool cannot hold
    if pool is not None:
        a, b = nd["a_bytes"], nd["b_bytes_per_token"]
        ceiling = nd.get("single_request_l_ceiling")
        # recompute the ceiling for THIS pool, not the card's build pool
        bpb = card["groups"]["bytes_per_block"]
        n_blocks = pool // bpb
        n_attn_groups = sum(1 for r in card["groups"]["rows"]
                            if (r["example"] or "").endswith("self_attn")
                            and not r["example"].startswith("draft."))
        this_ceiling = ((n_blocks - 1 - a // bpb) // n_attn_groups) * card["page_geometry"]["block_size_tokens"]
        if L > this_ceiling:
            warnings.append(f"L={L} exceeds single-request ceiling {this_ceiling} "
                            f"for pool {pool} (tier={args.tier}); engine will "
                            f"reject or throttle - lower L or raise tier")
        del ceiling

    # ---- assemble env from the platform card
    env_lines: list[tuple[str, str]] = []
    for k, v in platform["env"].items():
        if not k.startswith("_"):
            env_lines.append((k, v))
    env_lines.append(("PATH", platform["venv"]["path"] + r"\Scripts;%PATH%"))

    pythonpath = []
    if quant == "exl3":
        py = platform["engine_contract"]["orcasaq2_sitecustomize"]["path"]
        pythonpath.append(py)
        env_lines.append(("ORCA_EXL3_ALLOW_EMPTY_SHARED", "1"))
    pythonpath.append(platform["engine_contract"]["pin_shim"]["path"])
    env_lines.append(("PYTHONPATH", ";".join(pythonpath)))
    pin = platform["engine_contract"]["pin_env"]
    env_lines.append(("VLLM_DBG_TRACE", pin["VLLM_DBG_TRACE"]))
    env_lines.append(("VLLM_DBG_MIN", pin["VLLM_DBG_MIN"]))
    env_lines.append(("VLLM_DBG_PIN", pin["VLLM_DBG_PIN"]))
    warnings.append("pin doors VLLM_DBG_TRACE/MIN/PIN are human-copied from "
                    "GSQ production (step 091 diff); un-proven on this arm - "
                    "092 single-variable experiment pending")

    env_lines.append(("VLLM_KV_GROUP_SIZE", "8"))
    warnings.append("VLLM_KV_GROUP_SIZE=8 is human-copied (step 036 measured "
                    "+18.5% on THIS model family); sensitivity not mapped")

    # ---- argv
    model_path = card["model"]["path"]
    draft = (card.get("draft") or {}).get("path") if args.family != "none" else None
    argv = [
        f'"{VENV_PYTHON}"', "-m", "vllm.entrypoints.cli.main", "serve",
        f'"{model_path}"',
        "--served-model-name", os.path.basename(os.path.normpath(model_path)),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto",
        "--kv-cache-dtype", "nvfp4",
        "--gpu-memory-utilization", "0.922",
        "--max-model-len", str(L),
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(mbt),
        "--enable-prefix-caching",
        "--mamba-cache-mode", "align",
        "--mamba-ssm-cache-dtype", "bfloat16",
        "--kv-offloading-backend", "native",
        "--kv-offloading-size", "8",
        "--cudagraph-capture-sizes", "3",
    ]
    warnings.append("gpu-memory-utilization 0.922 / kv-offloading 8 GiB / "
                    "mamba keys / nvfp4 / FULL-graph tier are platform facts "
                    "(platform_card.json; production daily values)")
    if card["source_flags"]["language_model_only_required"]:
        argv.append("--language-model-only")
    if pool is not None:
        argv += ["--kv-cache-memory-bytes", str(pool)]
    if draft:
        argv += ["--speculative-config.method", "dflash",
                 "--speculative-config.model", f'"{draft}"',
                 "--speculative-config.num_speculative_tokens", str(num_spec)]

    # ---- write the .cmd (CRLF, iron rule 4)
    lines = ["@echo off", "rem GENERATED by tools/profile_card/gen_launch.py - "
             "do not hand-edit; regenerate instead",
             f"rem card = {card_path}", f"rem tier={args.tier} L={L} "
             f"pool={pool} mbt={mbt} spec={num_spec}", ""]
    for k, v in env_lines:
        lines.append(f'set "{k}={v}"')
    # self-contained: the generated cmd must boot alone (autoprobe launches it
    # directly), so it carries the vcvars64 + LIB contract like production.
    lines.append(
        'call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022'
        '\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1')
    lines.append('set "LIB=C:\\PROGRA~1\\NVIDIA~2\\CUDA\\v13.3\\lib\\x64;%LIB%"')
    lines += ["",
              'del /q "G:\\qwen3.8model\\_tmp_prod029\\vllm_offload_*.mmap" 2>nul',
              'del /q "G:\\qwen3.8model\\_tmp_orcasaq2\\vllm_offload_*.mmap" 2>nul',
              "cd /d G:\\qwen3.8model", ""]
    lines.append(" ".join(argv))
    lines.append("exit /b %errorlevel%")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\r\n".join(lines) + "\r\n")

    print(f"[gen_launch] wrote {args.out}")
    print(f"[gen_launch] card={card_path} tier={args.tier} L={L} pool={pool} "
          f"mbt={mbt} spec={num_spec} family={args.family}")
    if warnings:
        print("[gen_launch] == GUARD WARNINGS (human-copied / risky values) ==")
        for w in warnings:
            print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
