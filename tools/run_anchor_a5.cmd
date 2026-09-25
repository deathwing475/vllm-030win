rem N=2 PRODUCTION (2026-09-25 user decision): pool 3.6e9 -> 121,776 tokens, PIECEWISE, offloading 8G
rem Source: stage2_pool_n2.cmd (git: nvfp4-win-experiment). Rollback: run_dflash2_default.cmd (N=3, auto).
@echo off
rem 200k 配方对齐版（2026-09-24）：用户 stable 配方 + DFlash2 草稿（gptq3c）。
rem 与 run_qwen_textonly_offload_200448_stable.cmd 的差异（全部有意）：
rem   + --spec-method/--spec-model/--spec-tokens 2（草稿）
rem   - --kv-cache-dtype-skip-layers sliding_window（草稿 5 层滑窗 KV bf16→nvfp4，
rem     单价 4,096→1,152 B/层·token；影响面只在草稿——目标模型无滑窗层）
rem   * --cudagraph-capture-sizes 1 2 → 4（用户纠正：带草稿 batch=请求数×(1+spec-tokens)，
rem     单并发 1+3=4；<150k 上下文无并发必要）
rem   * --kv-cache-memory-bytes 4500000000 → 3600000000 起步（带草稿 free≈2.9 GiB，
rem     200,448 需求≈3.72 GiB；逐步试 3.85e9/4.0e9，看 Auto-fit 到 200,448 且 free 不穿 200 MiB）
rem 注：本文件用 capture-sizes 图（同用户配方）；PIECEWISE 高吞吐变体见 run_dflash2_default.cmd，
rem     两种图模式未做共存测试，勿叠加。
if not exist C:\fi mkdir C:\fi
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_ENDPOINT=https://hf-mirror.com"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "PYTHONPATH=G:\qwen3.8model\nvfp4-win-experiment\vllm-overlay"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "VLLM_HAS_FLASHINFER_CUBIN=1"
set "FLASHINFER_WORKSPACE_BASE=C:\fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "HF_HUB_OFFLINE=1"
set "VLLM_LOGGING_LEVEL=INFO"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "PATH=G:\qwen3.8model\vllm-win\Scripts;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
"G:\qwen3.8model\vllm-win\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ" ^
  --served-model-name qwen3.8-27b-gsq ^
  --host 127.0.0.1 --port 8080 ^
  --language-model-only ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes 3400000000 ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len 110000 ^
  --max-num-seqs 1 --max-num-batched-tokens 1024 ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}" ^
  --cudagraph-capture-sizes 3 ^
  --spec-method dflash ^
  --spec-model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --spec-tokens 2 ^
  --dtype auto
exit /b %errorlevel%
