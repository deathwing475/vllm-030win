@echo off
rem ===========================================================================
rem STEP 108 ORCA ARM (KVMem pool x context frontier). Derived from
rem tools/serve_orcasaq2_029_kvmem100.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem Shapes: main leg = S108_VIEWPORT=0, S108_LOAD=1 (assembly + speed);
rem recall leg = S108_VIEWPORT=1, S108_LOAD=0 (viewport protocol).
rem Eager + no spec (the O1 baseline). The speed verdict is eager same-caliber
rem relative non-degradation only; it must not be compared with the Orca
rem production FULL-graph + DFlash2 78-87 tok/s band.
rem
rem Why: L=16,384 / 49,152 < W=65,536 never triggers sliding-window eviction
rem (skipped <= 0 returns early), so the pool scan must use L > W (131,072)
rem or shrink W below L. 3.1e9/W=65,536 measured 2.11x; the low-pool floor is
rem ~1.6e9 at this W, so the matrix uses 3.1e9 / 2.0e9 / 1.8e9.
rem ===========================================================================
if "%S108_TAG%"=="" set "S108_TAG=kvmem_orca_s108"
if "%S108_KVMB%"=="" set "S108_KVMB=3100000000"
if "%S108_W%"=="" set "S108_W=65536"
if "%S108_ML%"=="" set "S108_ML=131072"
if "%S108_MBT%"=="" set "S108_MBT=1456"
if "%S108_WSMB%"=="" set "S108_WSMB=5120"
if "%S108_LOAD%"=="" set "S108_LOAD=1"
if "%S108_VIEWPORT%"=="" set "S108_VIEWPORT=0"
if "%S108_PAGES%"=="" set "S108_PAGES=40"
if "%S108_PICK%"=="" set "S108_PICK=score"
if "%S108_EVERY%"=="" set "S108_EVERY=12"
if "%S108_KEEP%"=="" set "S108_KEEP=10"
set "HOME=C:\fi"
set "USERPROFILE=C:\fi"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "OMP_NUM_THREADS=8"
set "VLLM_LOGGING_LEVEL=INFO"
set "VLLM_SERVER_DEV_MODE=1"
set "TMP=G:\qwen3.8model\_tmp_orcasaq2"
set "TEMP=G:\qwen3.8model\_tmp_orcasaq2"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\orcasaq2_sitecustomize;G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_KV_GROUP_SIZE=8"
set "VLLM_KVMEM_SW_WINDOW=%S108_W%"
set "VLLM_KVMEM_WORKSPACE=1"
set "VLLM_KVMEM_WORKSPACE_MB=%S108_WSMB%"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=1"
set "VLLM_KVMEM_AUTHORITY=1"
set "VLLM_KVMEM_AUTHORITY_TRAJ=2"
set "VLLM_KVMEM_SWEEP=1"
set "VLLM_KVMEM_LOAD=%S108_LOAD%"
set "VLLM_KVMEM_VIEWPORT=%S108_VIEWPORT%"
set "VLLM_KVMEM_VIEWPORT_PAGES=%S108_PAGES%"
set "VLLM_KVMEM_VIEWPORT_RECENT=16384"
set "VLLM_KVMEM_RECENT=16384"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_SNAPSHOT_EVERY_PAGES=%S108_EVERY%"
set "VLLM_KVMEM_SNAPSHOT_KEEP=%S108_KEEP%"
if not "%S108_SLOTPOLICY%"=="" set "VLLM_KVMEM_SLOT_POLICY=%S108_SLOTPOLICY%"
if not "%S108_INTERLEAVE%"=="" set "VLLM_KVMEM_ASM_INTERLEAVE=%S108_INTERLEAVE%"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_TIMING=1"
set "VLLM_KVMEM_TIMING_EVERY=25"
set "VLLM_KVMEM_SLOT_PICK=%S108_PICK%"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S108_TAG%"
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist "G:\qwen3.8model\_tmp_orcasaq2" mkdir "G:\qwen3.8model\_tmp_orcasaq2"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
del /q "G:\qwen3.8model\_tmp_orcasaq2\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\qwen3.8exl3" ^
  --served-model-name orcasaq2 ^
  --host 127.0.0.1 --port 8001 ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes %S108_KVMB% ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len %S108_ML% ^
  --max-num-seqs 1 --max-num-batched-tokens %S108_MBT% ^
  --enable-prefix-caching ^
  --enforce-eager ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --dtype auto
exit /b %errorlevel%
