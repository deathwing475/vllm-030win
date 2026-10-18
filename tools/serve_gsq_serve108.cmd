@echo off
rem ===========================================================================
rem STEP 108 GSQ PURE-SERVE CONTROL ARM. Derived from
rem tools/serve_gsq_prod029_n2.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem No VLLM_KVMEM_* environment: this is the production stack with the pool,
rem max-model-len and mbt exposed. It answers "what does KVMem buy at the same
rem pool" -- KVMem can accept a longer prompt (exact attention inside W plus
rem retrieval outside it), while pure serve can only accept a shorter prompt
rem with full exact attention.
rem
rem At L=163,072 the 3.4e9 pool is exactly one request (163,719 tokens,
rem 1.004x); every lower pool needs a lower L, so the control is a staircase,
rem not a fixed-L scan. L changes trigger a first-compile boot (step 093:
rem 3.29 tok/s), so each new point is compile boot + sample boot.
rem ===========================================================================
if "%S108_KVMB%"=="" set "S108_KVMB=3400000000"
if "%S108_ML%"=="" set "S108_ML=163072"
if "%S108_MBT%"=="" set "S108_MBT=1024"
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
rem Fast-tier pin (step 030, 2026-09-27): inject
rem tools\pin_shim\sitecustomize.py via PYTHONPATH. MIN mode hooks only
rem load_dflash_model and pins the draft weights (gptq3c, 31 Humming
rem layers) into dedicated VRAM after their free=0 pressure-window
rem placement. Verified 21/21 boots fast-tier 20.3-21.6ms (clean boots
rem draw 20% fast / 80% mid 21-27ms) with acc 0.66-0.75 + needle green.
rem ROLLBACK = remove the 4 set lines below.
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_KV_GROUP_SIZE=8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
rem Sweep stale offload mmaps (crash/held-ref exits leave them; names are
rem per-engine unique so they never collide -- just reclaim the space).
del /q "G:\qwen3.8model\_tmp_prod029\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ" ^
  --served-model-name qwen3.8-27b-gsq ^
  --host 127.0.0.1 --port 8080 ^
  --language-model-only ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes %S108_KVMB% ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len %S108_ML% ^
  --max-num-seqs 1 --max-num-batched-tokens %S108_MBT% ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 3 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2 ^
  --dtype auto
exit /b %errorlevel%
