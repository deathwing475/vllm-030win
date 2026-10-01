@echo off
rem ===========================================================================
rem STEP 073 EXPERIMENT ENTRY (KVMem ARM, NOT production): the same recipe as
rem serve_gsq_kvmem_viewport072.cmd plus VLLM_KVMEM_DEBUG=1, which makes the
rem connector report what the scheduler actually did to a rewritten request:
rem adopted computed tokens, block-hash count, forward steps, and the status /
rem sampled token ids at finish. Observation only, no mechanism changed (the
rem flag defaults off, so an unset environment is byte-equivalent to step 072).
rem
rem Why: the step 072 N=55 window request (96,128 tokens) prefilled, scored and
rem baked, then the FIRST sampled token ended the request with zero chunks. The
rem "coherent positive control" of that round is now known (from the retrieval
rem dumps' recent_tokens field: 16384 = window, 32768 = native) to have come
rem from a NON-window request, so every window run so far ended empty. This boot
rem decides between:
rem   A the model really samples EOS   B the request ends before any sample
rem   C a token is sampled but stop/stream handling drops it
rem Controls: tools/kvmem_window_control.py sends the window's own token
rem sequence as an ORDINARY prompt (no rewrite, no scoring, no bake) and the
rem tail alone, with logprobs + return_token_ids; the step 072 probe re-run with
rem --serve-ignore-eos forces generation past EOS (text => A, empty => B/C).
rem
rem Dump dir kvmem_k8a (kvmem_k7a holds step 072 evidence and is not written).
rem Everything else (pool 3.4e9, nvfp4, ssm bf16, G=8, pin shim, graph mode,
rem prefix caching, --max-num-batched-tokens 1424) matches the step 072 arm.
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (section 12.14 for step 072).
rem Probe: tools/kvmem_viewport_probe.py (ingest -> flush -> serve).
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
set "VLLM_KVMEM_SW_WINDOW=163072"
set "VLLM_KVMEM_WORKSPACE=1"
set "VLLM_KVMEM_WORKSPACE_MB=5120"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=1"
set "VLLM_KVMEM_AUTHORITY=1"
set "VLLM_KVMEM_AUTHORITY_TRAJ=2"
set "VLLM_KVMEM_SWEEP=1"
set "VLLM_KVMEM_VIEWPORT=1"
set "VLLM_KVMEM_VIEWPORT_PAGES=55"
set "VLLM_KVMEM_VIEWPORT_RECENT=16384"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k8c"
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
  --max-model-len 262144 ^
  --max-num-seqs 1 --max-num-batched-tokens 1424 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 1 ^
  --dtype auto
exit /b %errorlevel%
