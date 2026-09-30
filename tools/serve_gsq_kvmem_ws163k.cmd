@echo off
rem ===========================================================================
rem KVMem ARM (NOT production) - host KV workspace, stage 1 K1/K2/K3.
rem
rem Status: step 061. The bounded-prefill half (step 057) plus copy-before-free
rem (step 060: the pages the sliding window drops are copied into a pinned host
rem workspace keyed by (trajectory, token offset) and held out of the block pool
rem until the copy completes) plus the retrieval half's index and scoring
rem (step 061: pre-RoPE q/k capture on the 16 full_attention layers, a
rem 128-token sub-block Mean-K index on the host, and a softmax-over-pages
rem ranking dumped to VLLM_KVMEM_DUMP).
rem
rem NOT implemented yet: rematerialisation. Nothing the index selects is put
rem back into the window, so the answer to a needle outside the window is still
rem not generated - the dump only says whether retrieval WOULD have found it.
rem
rem Design authority: docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (§12 = the
rem implemented mechanism). Acceptance tools: tools\kvmem_ws_probe.py (K1/K2),
rem tools\kvmem_k3_probe.py (K3 retrieval quality).
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
rem   7. VLLM_KVMEM_RAWK=1 (step 061) -> capture the pre-RoPE q/k of the 16
rem      full_attention layers and build the Mean-K retrieval index. This also
rem      forces those 16 layers onto the eager norm+RoPE path (the production
rem      fused kernel exposes no pre-RoPE K), so the arm's numerics are its own
rem      and PPL must NOT be compared across arms. UNSET THIS to reproduce the
rem      step 060 arm, which was measured on the fused kernel.
rem   8. VLLM_KVMEM_DUMP=<dir> (step 061) -> where the retrieval ranking is
rem      written (one kvmem_retrieval_NNN.json per scored prompt).
rem   9. --enforce-eager replaces --cudagraph-capture-sizes 1. Stage 1 is
rem      specified as "text + no speculative + NO GRAPH MODE" (design §6), and
rem      the capture has to run in the forward: an AOT fullgraph compile rejects
rem      it outright (torch._dynamo.exc.Unsupported on the stash's side effects,
rem      first observed as "logging.Logger method not supported"). Consequence:
rem      this arm's prefill/decode speeds are NOT comparable with step 057/060,
rem      which were measured with compilation on.
rem  10. VLLM_KVMEM_AUTHORITY=1 (step 064) -> keep the pre-RoPE rotary prefix
rem      (64 of 256 head dims, fp16) as the rematerialisation authority, and
rem      VLLM_KVMEM_AUTHORITY_TRAJ=1 caps it at one trajectory (2.0 GiB host at
rem      a full 262,144-token workspace). With VLLM_KVMEM_SELFTEST=1 this also
rem      runs the rematerialisation round trip: each stored page's rotary prefix
rem      is rebuilt from the authority at the page's ORIGINAL positions and
rem      compared with the engine's own nvfp4 bytes. Nothing is put back into
rem      the window, so attention behaviour is unchanged; the report lands in
rem      <VLLM_KVMEM_DUMP>\kvmem_remat_selftest.json.
rem  11. VLLM_KVMEM_DUMP -> kvmem_k3e (step 064 run; k3c/k3d are step 062's
rem      evidence and must not be overwritten).
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
set "VLLM_KVMEM_RAWK=1"
set "VLLM_KVMEM_AUTHORITY=1"
set "VLLM_KVMEM_AUTHORITY_TRAJ=1"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k3e"
rem K3 measurement knobs. The index stores at the finest granularity and the
rem coarser ones are summed from it, so one 200K ingest reports every variant:
rem granularity 32/64/128 (design risk R1: the paper uses 32-token blocks, the
rem design's initial value is 128) x mode dot/cosine (the step 061 control run
rem showed the raw dot product's ranking is carried by per-page magnitude).
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
rem Step 062 diagnostic: also write the vectors behind the ranking to a sidecar
rem .npz beside each report (page mean-K, the per-head-group query, per-sub-block
rem and per-page logits). One 200K ingest costs ~4.5 minutes, so the arrays are
rem dumped once and re-scored offline rather than one engine run per hypothesis.
set "VLLM_KVMEM_DUMP_KBAR=1"
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
  --enforce-eager ^
  --dtype auto
exit /b %errorlevel%
