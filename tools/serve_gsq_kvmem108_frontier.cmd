@echo off
rem ===========================================================================
rem STEP 108 ARM (KVMem line, pool x context frontier). Derived from
rem tools/serve_gsq_kvmem_load1456.cmd by tools/make_kvmem108_arms.py.
rem Do not edit by hand; edit the generator and re-run it.
rem
rem Shapes: main leg = S108_VIEWPORT=0, S108_LOAD=1 (assembly + speed);
rem recall leg = S108_VIEWPORT=1, S108_LOAD=0 (step 074/078 viewport
rem protocol; the assembly path cannot answer an out-of-window needle --
rem step 066 header says so explicitly).
rem
rem Why: batch A's original fixed W=131,072 / L=200,704 is not bootable below
rem ~3.0e9. The startup check needs at least one request at the configured W,
rem and 3.4e9/W=131,072 measured 1.17x (234,000 tokens), so the pool and W
rem must walk the joint frontier.
rem
rem Page note (step 108 read-code correction): page = 16*cdiv(mamba_raw,
rem 16*1152); it depends on spec N and model geometry, NOT on SW_WINDOW.
rem +spec N=2 => page 1456, mbt >= 1458 at every W. The old W->page note in
rem load1456 was a confound with the no-spec arm (W=163,072/no-spec -> 1424).
rem
rem Injectables: S108_TAG, S108_KVMB, S108_W, S108_ML, S108_MBT, S108_WSMB,
rem S108_LOAD, S108_VIEWPORT, S108_PAGES, S108_PICK (default score = step
rem 081 delivery), S108_SPEC, S108_CAP, S108_PROF.
rem Fixed: RAWK/AUTH/WS/SWEEP/BAKE_VERIFY/SELFTEST, AUTHORITY_TRAJ=2,
rem TOPN=64, VIEWPORT_RECENT=16384, RECENT=32768.
rem ===========================================================================
if "%S108_TAG%"=="" set "S108_TAG=kvmem_s108"
if "%S108_KVMB%"=="" set "S108_KVMB=3400000000"
if "%S108_W%"=="" set "S108_W=131072"
if "%S108_ML%"=="" set "S108_ML=200704"
if "%S108_MBT%"=="" set "S108_MBT=1458"
if "%S108_WSMB%"=="" set "S108_WSMB=5120"
if "%S108_LOAD%"=="" set "S108_LOAD=1"
if "%S108_VIEWPORT%"=="" set "S108_VIEWPORT=0"
if "%S108_PAGES%"=="" set "S108_PAGES=55"
if "%S108_PICK%"=="" set "S108_PICK=score"
if "%S108_SPEC%"=="" set "S108_SPEC=1"
if "%S108_CAP%"=="" set "S108_CAP=3"
if "%S108_PROF%"=="" set "S108_PROF=0"
set "SPEC_ARGS="
if "%S108_SPEC%"=="1" set "SPEC_ARGS=--speculative-config.method dflash --speculative-config.model G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c --speculative-config.num_speculative_tokens 2"
set "PROF_ARGS="
if "%S108_PROF%"=="1" (
  if not exist "G:\qwen3.8model\prod029_logs\%S108_TAG%" mkdir "G:\qwen3.8model\prod029_logs\%S108_TAG%"
  if not exist "G:\qwen3.8model\prod029_logs\%S108_TAG%\prof" mkdir "G:\qwen3.8model\prod029_logs\%S108_TAG%\prof"
  set "PROF_ARGS=--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=G:\qwen3.8model\prod029_logs\%S108_TAG%\prof --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true"
)
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
set "VLLM_SERVER_DEV_MODE=1"
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
set "VLLM_KVMEM_RECENT=32768"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_TIMING=1"
set "VLLM_KVMEM_TIMING_EVERY=25"
set "VLLM_KVMEM_SLOT_PICK=%S108_PICK%"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S108_TAG%"
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
  --cudagraph-capture-sizes %S108_CAP% ^
  %SPEC_ARGS% ^
  %PROF_ARGS% ^
  --dtype auto
exit /b %errorlevel%
