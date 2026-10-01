@echo off
rem ===========================================================================
rem STEP 075 DISCRIMINATOR ARM (KVMem ARM, NOT production): byte-identical to
rem serve_gsq_kvmem_viewport075_spec.cmd EXCEPT that the compressed window is off
rem (VLLM_KVMEM_VIEWPORT=0, no bake verify), dump dir kvmem_k9f.
rem
rem Why: with speculation on, the viewport arm booted and captured its CUDA
rem graphs cleanly (the step 075 capture allow-list fixed that), but NO request
rem was ever scheduled: an 8,153-token request and a 198,184-token one both
rem spun in the scheduler -- hundreds of thousands of connector calls per minute
rem at computed=0, zero forward steps, 0% GPU utilisation, and the request only
rem left when the client gave up. That fault is either "spec x the viewport path"
rem or "spec x the KVMem connector at large", and the two need different fixes.
rem Turning only the viewport off separates them:
rem   request runs  => the fault is in the viewport route (rewrite / deferral
rem                    interacting with the drafter), and this boot also yields
rem                    the 8k decode number WITH speculation on a KVMem arm;
rem   request hangs => the fault is in the connector + drafter combination
rem                    itself (K1 store path, block accounting, or the
rem                    WAITING_FOR_REMOTE_KVS interaction), and the viewport is
rem                    innocent.
rem ===========================================================================
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
set "VLLM_KVMEM_WORKSPACE=1"
set "VLLM_KVMEM_WORKSPACE_MB=5120"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=1"
set "VLLM_KVMEM_AUTHORITY=1"
set "VLLM_KVMEM_AUTHORITY_TRAJ=2"
set "VLLM_KVMEM_SWEEP=1"
set "VLLM_KVMEM_VIEWPORT=0"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k9f"
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
  --max-num-seqs 1 --max-num-batched-tokens 1456 ^
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
