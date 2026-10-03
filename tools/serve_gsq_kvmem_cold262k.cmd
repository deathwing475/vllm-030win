@echo off
rem ===========================================================================
rem STEP E1 ARM (KVMem line, stage-1 exit item 3, cold 262K transcript): the
rem step 074 arm verbatim with ONLY the dump dir made injectable.
rem
rem Why this shape: 262,144 is the KVMem design ceiling, and ONLY an arm with
rem --max-model-len 262144 can admit a 262K prompt at all (the 075+ exit-list
rem arm tops out at L=200,704 because spec x SW_WINDOW=131072 exhausted the
rem pool, step 075: "KVMem's 262,144 ceiling is unreachable on this arm"). The
rem 074 recipe is the largest-L KVMem arm on record: L=262,144, SW_WINDOW
rem 163072 (page 1424, mbt 1424), NO speculative decoding, VIEWPORT=1,
rem BAKE_VERIFY=1, CUDA graph capture 1. The stage-1a cold-prefill criterion
rem (step 057: cold 256K in 185 s) is re-exercised here on the FULL current
rem code state (K1 copy-before-free + K2 guard + K3 scoring/bake + 072 rolling
rem ingest + 074 group-complete bake + 067 graph compat).
rem
rem The probe sends ~262,000 tokens: ingest (sliding window evicts ~69 pages
rem into the workspace) -> flush (different nonce, invalidates native prefix
rem hashes) -> serve (same prompt rewritten onto the ~96K window, top pages
rem baked into the 55 retrieval slots). Judgement for this exit item: all three
rem requests finish ok, zero ERROR/Traceback, no OOM, boot guard passes
rem ("viewport needs ... pool has ..."). Needle recall at 262K is RECORDED, not
rem gated: 200K-262K is the design's known-risk band (only-recorded, not
rem acceptance; design doc section 7.2).
rem
rem Injectables: S074_TAG (dump dir; MUST be fresh per boot -- the worker's
rem kvmem_retrieval_%03d counter restarts every boot, so reusing a directory
rem silently overwrites the earlier boot's evidence; the committed 074 value
rem kvmem_k9b is 074 evidence and must NOT be reused).
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (sections 12.15/12.16; the
rem 262K known-risk band is section 7.2). Probe: tools/kvmem_viewport_probe.py.
rem ===========================================================================
if "%S074_TAG%"=="" set "S074_TAG=kvmem_e1e"
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
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S074_TAG%"
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
