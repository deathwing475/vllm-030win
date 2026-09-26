@echo off
rem ===========================================================================
rem Line A of the three-line merged regression (stage-2 acceptance, 2026-09-26).
rem No-spec throughput baseline on the 0.29 base, CONFIG-IDENTICAL to the A5b
rem anchor launcher (run_anchor_nospec.cmd body, 0.27 overlay stack):
rem   nvfp4 KV + manual pool 3,400,000,000 + max-model-len 110000
rem   PIECEWISE + --cudagraph-capture-sizes 1 + mamba-cache-mode align
rem   seqs 1 / batched-tokens 1024 / language-model-only / no prefix-cache
rem Anchor to beat (median of warmup1+3): 4k 55.7 / 8k 55.6 / 32k 54.2 / 100k 50.3
rem Model name qwen3.8-27b-gsq + port 8080 kept for anchor_longctx.py compat.
rem Env contract per run_anchor_ppl_base029_batch5_nvfp4_r4.cmd (batch-5
rem validated nvfp4-on-0.29 contract: VLLM_KV_CACHE_LAYOUT=HND, flashinfer
rem 0.6.18 workspace C:/fw forward slashes, tvm_ffi LDFLAGS).
rem Run me from G:\qwen3.8model (never from the record repo -- cwd captures
rem the 0.27.1 tree lying there and explodes vllm._C_stable_libtorch).
rem ===========================================================================
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist G:\qwen3.8model\_tmp_line_a mkdir G:\qwen3.8model\_tmp_line_a
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CXX=C:\Program Files\LLVM\bin\clang++"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "VLLM_LOGGING_LEVEL=INFO"
set "TMP=G:\qwen3.8model\_tmp_line_a"
set "TEMP=G:\qwen3.8model\_tmp_line_a"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
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
  --no-enable-prefix-caching ^
  --cudagraph-capture-sizes 1 ^
  --dtype auto
exit /b %errorlevel%
