@echo off
rem ===========================================================================
rem KVMem ARM (NOT production) - host KV workspace, stage 1 K1/K2.
rem
rem Status: step 060. The bounded-prefill half (step 057) plus copy-before-free:
rem the pages the sliding window drops are copied into a pinned host workspace
rem keyed by (trajectory, page_index) and held out of the block pool until the
rem copy completes. Retrieval / rematerialisation (K3) is NOT implemented yet,
rem so nothing reads the workspace back into the window.
rem
rem Design authority: docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (§12 = the
rem implemented mechanism). Acceptance tool: tools\kvmem_ws_probe.py.
rem
rem Delta vs tools/serve_gsq_prod029_n2.cmd:
rem   1. VLLM_KVMEM_SW_WINDOW=163072 -> qwen3_next.py gives per_layer_sliding_window
rem      to the full_attention layers only. The engine cannot set
rem      CacheConfig.sliding_window here: arg_utils only fills it when ALL
rem      layer_types are sliding_attention, and this model interleaves 16
rem      full_attention with 48 linear_attention.
rem   2. --max-model-len 262144 (= max_position_embeddings) instead of 163072, so
rem      a prompt longer than the pool is not rejected by _validate_prompt_len.
rem   3. Speculative decoding DROPPED: stage 1 keeps one variable, and the drafter
rem      brings its own sliding_attention layers. A decode step therefore carries
rem      1 token, so the capture list is [1] instead of production's [3].
rem   4. VLLM_KVMEM_WORKSPACE=1 -> the sliding-window managers hand their evicted
rem      pages to KVMemConnector, which takes over the offloading slot in
rem      vllm/config/vllm.py. NOTE: --kv-offloading-size 8 is still what enters
rem      that config path, but KVMemConnector ignores cpu_bytes_to_use and
rem      allocates its own pinned region sized by VLLM_KVMEM_WORKSPACE_MB, so the
rem      8 GiB prefix-cache region is NOT allocated in this arm.
rem   5. VLLM_KVMEM_WORKSPACE_MB=3072 -> 122 host slots (25.03 MiB per slot = one
rem      1424-token page x 16 layers), shared by both attention groups.
rem   6. VLLM_KVMEM_SELFTEST=1 -> the worker copies each stored page back and
rem      compares it byte for byte (both directions) and logs the result.
rem Everything else (pool 3.4e9, nvfp4, mamba align, ssm bf16, G=8, pin shim,
rem prefix caching) is byte-for-byte the production recipe.
rem
rem Expected boot numbers (step 060): attn block size 1424, mamba page padding
rem 0.38%, GPU KV cache size 275,997 tokens / 1.05x @262,144, viewport needs 234
rem of the pool's 259 blocks. Capacity is NOT production's 163,719.
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
set "VLLM_KVMEM_WORKSPACE_MB=3072"
set "VLLM_KVMEM_SELFTEST=1"
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
  --max-num-seqs 1 --max-num-batched-tokens 1024 ^
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
