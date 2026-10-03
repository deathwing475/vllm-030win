@echo off
rem ===========================================================================
rem STEP E1 ARM (KVMem line, stage-1 exit item 5): the step 082 arm verbatim with
rem the compressed window DISABLED and the step 066 prefix-assembly path enabled,
rem so tools/kvmem_assembly_probe.py runs at PAGE SIZE 1456 for the first time.
rem
rem Why: the 066/067 assembly evidence (matches -> issued -> prefix landed)
rem was collected at page 1424 (SW_WINDOW=163072). Since 075 the exit-list arm
rem runs SW_WINDOW=131072, which the engine serves with block_size 1456, and the
rem assembly chain (incl. the mamba page-boundary snapshot, whose guard 077
rem re-derived as "mbt >= page + num_spec") has NEVER been exercised at 1456.
rem This variant isolates the assembly line: VIEWPORT=0 (no prompt rewrite --
rem the 082 arm hardcodes VIEWPORT=1, which would rewrite the 200K probe prompt
rem and change the TTFT structure the compare subcommand judges), LOAD injected
rem via S082_LOAD so one file serves as both the LOAD=1 and the LOAD=0 leg.
rem
rem Injectables (same semantics as the 082 arm): S079_TAG (dump dir, MUST be
rem fresh per boot), S079_SPEC (set 0 to drop the draft if assembly x spec ever
rem misbehaves -- this combination is exercised for the first time here),
rem S079_RAWK / S079_AUTH / S079_WS / S079_TIMING / S079_EVERY / S079_TOPN /
rem S079_PAGES / S079_VRECENT / S079_RECENT / S081_PROF / S081_PICK /
rem S082_KVMB / S082_WSMB, plus S082_LOAD (1 = assembly on, 0 = full-prefill
rem control; the assembly probe's compare needs both).
rem
rem Judgement (step 066/067 protocol): engine log "KVMem assembly: request ...
rem matches" -> "issued (... page(s) + ... mamba state block(s) ...)" ->
rem "prefix landed", greedy serve output byte-identical between LOAD=1 and
rem LOAD=0, TTFT ratio inside the structural bound, needle HIT (depth 0.50 sits
rem inside the 131,072 window), decode stays on the CUDA graph, zero ERROR.
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md sections 12.12/12.13.
rem Probe: tools/kvmem_assembly_probe.py (ingest -> flush -> serve + tail).
rem ===========================================================================
if "%S079_TAG%"=="" set "S079_TAG=kvmem_e1c"
if "%S082_LOAD%"=="" set "S082_LOAD=1"
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
rem S082_MBT: batched-tokens cap. 1458 = the 082 arm verbatim (page 1456 +
rem num_spec 2, iron rule 16-10). When running S079_SPEC=0 (the step 066
rem no-spec protocol this regression falls back to), pass 1456 so the step tail
rem lands exactly on a page boundary again (the mamba snapshot precondition).
if "%S082_MBT%"=="" set "S082_MBT=1458"
rem S082_CAP: cudagraph capture sizes (3 = the 082 arm verbatim). Boot on this
rem arm sits on the residency cliff (082-A: ~0.4 GiB); if boot OOMs with
rem S082_KVMB lowered, 1 (the 066/067 assembly-arm value) is the next knob.
if "%S082_CAP%"=="" set "S082_CAP=3"
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
set "VLLM_KVMEM_LOAD=%S082_LOAD%"
set "VLLM_KVMEM_VIEWPORT=0"
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
  --max-num-seqs 1 --max-num-batched-tokens %S082_MBT% ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes %S082_CAP% ^
  %SPEC_ARGS% ^
  %PROF_ARGS% ^
  --dtype auto
exit /b %errorlevel%
