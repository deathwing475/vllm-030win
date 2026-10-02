@echo off
rem ===========================================================================
rem STEP 082 ARM -- the step 081 arm verbatim, plus TWO new knobs that 082-A
rem needs to move GPU memory residency without touching the request path:
rem
rem   * S082_KVMB (default 3400000000 = the 079/080/081 pool value verbatim):
rem     --kv-cache-memory-bytes. Lowering it frees dedicated VRAM while shapes
rem     stay byte-identical (page 1456 depends on W/GROUP, not on the pool), so
rem     the anchor GEMM (same name, same grid, same block, same M) is still the
rem     081 anchor. This is the pure-residency A/B condition.
rem   * S082_WSMB (default 5120 = verbatim): VLLM_KVMEM_WORKSPACE_MB.
rem
rem Everything else is byte-identical to tools/serve_gsq_kvmem_viewport081_spec.cmd:
rem S079_SPEC / S079_RAWK / S079_WS / S079_PAGES / S079_TOPN stay env-tunable so
rem 082-A can also run spec-off and rawk-off as mechanism-isolation conditions
rem (labeled as such in the step document), and S081_PROF / S081_PICK keep the
rem 081 semantics (profiler window, SLOT_PICK gate). The dump dir MUST be fresh
rem every boot (iron rule 16-viii).
rem ===========================================================================
if "%S079_TAG%"=="" set "S079_TAG=kvmem_k16a"
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
if "%S081_PROF%"=="" set "S081_PROF=0"
if "%S081_PICK%"=="" set "S081_PICK=time"
if "%S082_KVMB%"=="" set "S082_KVMB=3400000000"
if "%S082_WSMB%"=="" set "S082_WSMB=5120"
set "SPEC_ARGS="
if "%S079_SPEC%"=="1" set "SPEC_ARGS=--speculative-config.method dflash --speculative-config.model G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c --speculative-config.num_speculative_tokens 2"
set "PROF_ARGS="
if "%S081_PROF%"=="1" (
  if not exist "G:\qwen3.8model\prod029_logs\%S079_TAG%" mkdir "G:\qwen3.8model\prod029_logs\%S079_TAG%"
  if not exist "G:\qwen3.8model\prod029_logs\%S079_TAG%\prof" mkdir "G:\qwen3.8model\prod029_logs\%S079_TAG%\prof"
  set "PROF_ARGS=--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=G:\qwen3.8model\prod029_logs\%S079_TAG%\prof --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true"
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
set "VLLM_KV_GROUP_SIZE=8"
set "VLLM_KVMEM_SW_WINDOW=131072"
set "VLLM_KVMEM_WORKSPACE=%S079_WS%"
set "VLLM_KVMEM_WORKSPACE_MB=%S082_WSMB%"
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
set "VLLM_KVMEM_SLOT_PICK=%S081_PICK%"
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
  --kv-cache-memory-bytes %S082_KVMB% ^
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
  %PROF_ARGS% ^
  --dtype auto
exit /b %errorlevel%
