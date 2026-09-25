# Qwen3.8-27B-3Bit-GSQ native Windows vLLM

Environment: `G:\qwen3.8model\vllm-win`

Pinned runtime:

- CPython 3.13.15
- vLLM 0.27.1 (third-party native Windows wheel)
- PyTorch 2.13.0+cu130
- Triton 3.7.1
- CUDA driver: NVIDIA RTX 5070 Ti detected successfully
- Official model patch: `patch_vllm_qwen35_embedding.py`

The upstream vLLM PyPI package is Linux-only. This environment uses the
native-Windows build from `aivrar/vllm-windows-build`, release
`v0.27.1-win-cu130`, because the model card requires vLLM 0.27.1.

## Start text-only serving

Put the model in the Hugging Face cache or pass a local model directory:

```bat
G:\qwen3.8model\vllm-win\serve_qwen38_gsq.cmd
G:\qwen3.8model\vllm-win\serve_qwen38_gsq.cmd G:\qwen3.8model\hub\models--ISTA-DASLab--Qwen3.8-27B-3Bit-GSQ\snapshots\<revision>
```

The script uses `HF_ENDPOINT=https://hf-mirror.com`, the existing cache root
`G:\qwen3.8model\hub`, port `8000`, and `--language-model-only`.
The initial `--max-model-len 8192` is deliberate for a 16 GB RTX 5070 Ti;
increase it only after checking VRAM usage.

## Reapply the model patch

If vLLM is reinstalled or the environment is recreated, run:

```bat
G:\qwen3.8model\vllm-win\Scripts\python.exe G:\qwen3.8model\vllm-win\patch_vllm_qwen35_embedding.py
```

The patch creates a backup beside `qwen3_5.py`:

`qwen3_5.py.qwen3_5_quantized_embedding.bak`

## Important limits

- This is an unofficial native-Windows vLLM build, not upstream Windows support.
- Keep Python, torch, Triton, and vLLM pinned as installed. Do not run a broad
  upgrade against this environment.
- The published checkpoint has no MTP/speculative-decoding components.
- The model card's `--language-model-only` mode is recommended for text-only
  use and avoids loading the uncalibrated vision components.
- This Windows build should be treated as single-GPU. Linux-only vLLM
  features and some distributed backends are unavailable.
