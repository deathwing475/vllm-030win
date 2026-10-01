@echo off
rem ===========================================================================
rem STEP 072 VARIANT: the fixed-slot compressed window (design section 5.1,
rem re-bake route). Same recipe as serve_gsq_kvmem_ws163k_graph.cmd except:
rem
rem  1. VLLM_KVMEM_VIEWPORT=1 -> a request longer than the window gets its
rem     prompt REWRITTEN onto the compressed window: prompt[:S+N] + prompt[L-R:]
rem     (S = one 1424-token sink page, N = 55 retrieval pages, R = 16384).
rem     The window prefill runs in engine-native coordinates; afterwards the
rem     scored pages are baked into the retrieval slots from the raw-K
rem     authority (single rebuild per entry, fp32 cos/sin, bf16 K restore).
rem     VLLM_KVMEM_LOAD is deliberately OFF: the step 066 in-place assembly and
rem     the window are two separate routes, and this boot isolates the window.
rem  2. VLLM_KVMEM_SWEEP=1 -> the pages a finished request still holds (the
rem     ~114 in-window pages of a 200K ingest, which K1's eviction path never
rem     sees) are stored when the request finishes, with the same
rem     copy-before-free discipline (the scheduler frees the blocks only after
rem     the copies land). Without it the retrieval slots would have no V /
rem     non-rotary bytes for the mid-section.
rem  3. VLLM_KVMEM_WORKSPACE_MB=5120 -> 197 host slots (26.25 MiB per slot = a
rem     full 1424-token page across the 16 attention layers), enough for a
rem     200K trajectory (~140 pages) plus slack.
rem  4. VLLM_KVMEM_AUTHORITY_TRAJ=2 -> ingest + flush are two trajectories and
rem     both capture raw-K rows (2 x 2.0 GiB host).
rem  5. VLLM_KVMEM_TOPN=64 -> the scoring report must be able to name all 55
rem     retrieval slots' worth of pages.
rem  6. VLLM_KVMEM_DUMP -> kvmem_k7a (k5a is step 066/067 evidence).
rem Everything else (pool 3.4e9, nvfp4, ssm bf16, G=8, pin shim, graph mode,
rem prefix caching, --max-num-batched-tokens 1424) matches the step 067 arm.
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (section 12.14 for this step).
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
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k7a"
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
