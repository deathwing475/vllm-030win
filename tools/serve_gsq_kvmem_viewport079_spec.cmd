@echo off
rem ===========================================================================
rem STEP 079 ARM -- the step 078 arm with the timing ledger switched on.
rem
rem Byte-identical geometry to tools/serve_gsq_kvmem_viewport078_spec.cmd: page
rem 1456, --max-num-batched-tokens 1458 (= page + 2 DFlash draft slots, iron
rem rule 16-10), SW_WINDOW 131072, L 200704, pool 3.4e9, FULL_AND_PIECEWISE via
rem --cudagraph-capture-sizes 3, nvfp4 KV, mamba align + bf16 ssm,
rem VLLM_KV_GROUP_SIZE=8, bake read-back on, roundtrip selftest on.
rem
rem What is new here, and why:
rem   * VLLM_KVMEM_TIMING / _EVERY -> the [KVTIME] segment ledger from the
rem     step 079 patch (tools/apply_kvmem_timing_step079.py). 078 established
rem     that the same launcher swings between 1.42 and 11.0 s per page step
rem     across boots, with prefill and decode anti-correlated, and ruled out
rem     seven external explanations on out-of-band metrics -- but it had no
rem     timing evidence at all, because workspace_stats() has no callers and
rem     store_seconds mixes the store and load paths and swallows the
rem     selftest's torch.cuda.synchronize().
rem   * SELFTEST stays 1 on purpose. Turning it off would change the arm's
rem     shape relative to the four 078 boots, so its sync cost is measured
rem     instead (dselft) and subtracted offline. Speed readings from a boot
rem     with SELFTEST=0 do NOT belong in the same table as 078's.
rem   * VIEWPORT_PAGES / VIEWPORT_RECENT / TOPN / RETRIEVAL_RECENT are knobs
rem     so a budget peel needs no new file (step 080 runs the boot-side
rem     sweep with them).
rem
rem Every boot MUST get a fresh S079_TAG (iron rule 16-8: kvmem_retrieval_%03d
rem restarts at 001 per boot, so reusing a directory silently overwrites
rem evidence). Default is kvmem_k13a, which no earlier step ever used.
rem ===========================================================================
if "%S079_TAG%"=="" set "S079_TAG=kvmem_k13a"
if "%S079_SPEC%"=="" set "S079_SPEC=1"
if "%S079_RAWK%"=="" set "S079_RAWK=1"
if "%S079_AUTH%"=="" set "S079_AUTH=1"
if "%S079_WS%"=="" set "S079_WS=1"
if "%S079_TIMING%"=="" set "S079_TIMING=1"
if "%S079_EVERY%"=="" set "S079_EVERY=25"
if "%S079_PAGES%"=="" set "S079_PAGES=55"
if "%S079_VRECENT%"=="" set "S079_VRECENT=16384"
if "%S079_RECENT%"=="" set "S079_RECENT=32768"
if "%S079_TOPN%"=="" set "S079_TOPN=64"
set "SPEC_ARGS="
if "%S079_SPEC%"=="1" set "SPEC_ARGS=--speculative-config.method dflash --speculative-config.model G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c --speculative-config.num_speculative_tokens 2"
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist G:\qwen3.8model\_tmp_prod029 mkdir G:\qwen3.8model\_tmp_prod029
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
set "TMP=G:\qwen3.8model\_tmp_prod029"
set "TEMP=G:\qwen3.8model\_tmp_prod029"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_KV_GROUP_SIZE=8"
set "VLLM_KVMEM_SW_WINDOW=131072"
set "VLLM_KVMEM_WORKSPACE=%S079_WS%"
set "VLLM_KVMEM_WORKSPACE_MB=5120"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=%S079_RAWK%"
set "VLLM_KVMEM_AUTHORITY=%S079_AUTH%"
set "VLLM_KVMEM_AUTHORITY_TRAJ=2"
set "VLLM_KVMEM_SWEEP=1"
set "VLLM_KVMEM_VIEWPORT=1"
set "VLLM_KVMEM_VIEWPORT_PAGES=%S079_PAGES%"
set "VLLM_KVMEM_VIEWPORT_RECENT=%S079_VRECENT%"
set "VLLM_KVMEM_RECENT=%S079_RECENT%"
set "VLLM_KVMEM_TOPN=%S079_TOPN%"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_TIMING=%S079_TIMING%"
set "VLLM_KVMEM_TIMING_EVERY=%S079_EVERY%"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S079_TAG%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
del /q "G:\qwen3.8model\_tmp_prod029\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ" ^
  --served-model-name qwen3.8-27b-gsq ^
  --host 127.0.0.1 --port 8080 ^
  --language-model-only ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes 3400000000 ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len 200704 ^
  --max-num-seqs 1 --max-num-batched-tokens 1458 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 3 ^
  %SPEC_ARGS% ^
  --dtype auto
exit /b %errorlevel%
