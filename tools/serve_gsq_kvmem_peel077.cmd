@echo off
rem ===========================================================================
rem STEP 077 DIAGNOSTIC ARM (KVMem ARM, NOT production): faithful reproduction of
rem the 076 peel E config (step076_peelE_mbt1456) plus the scheduler trace patch
rem (tools/apply_sched_trace_step077.py, armed by VLLM_SCHED_TRACE=1).
rem
rem Config = production recipe + VLLM_KVMEM_SW_WINDOW=131072, --max-model-len
rem 163072, --max-num-batched-tokens %SCHED077_MBT% (default 1456), DFlash2 N=2.
rem No KVMem connector / rawK / viewport (VLLM_KVMEM_WORKSPACE is simply unset),
rem so the only connector is the native 8 GiB OffloadingConnector -- exactly what
rem peel E ran. 076 measured: mbt=1456 hangs (0 forward steps), mbt=1024 runs
rem 120.79/124.90 tok/s.
rem
rem Prediction under test: DFlash gives max_num_new_slots_for_drafting = 2, so the
rem per-request budget is min(mbt, mbt-2) = 1454 < the 1456-token page. The mamba
rem "align" split then floors the chunk to aligned_end = 0 and scheduler.py breaks
rem out of the waiting loop without any log line, forever. With mbt=1458 the
rem budget is exactly 1456 = one page, so the chunk lands on the page boundary
rem again (the 066 mamba snapshot requirement still holds) and requests run.
rem mbt=1457 is predicted to stay hung, which is what separates this mechanism
rem from "mbt must be a multiple of 128".
rem
rem Override the pinned budget from the caller: set SCHED077_MBT=1458.
rem ===========================================================================
if "%SCHED077_MBT%"=="" set "SCHED077_MBT=1456"
if "%SCHED077_TAG%"=="" set "SCHED077_TAG=kvmem_k10a_peelE_mbt1456"
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
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%SCHED077_TAG%"
set "VLLM_SCHED_TRACE=1"
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
  --max-model-len 163072 ^
  --max-num-seqs 1 --max-num-batched-tokens %SCHED077_MBT% ^
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
