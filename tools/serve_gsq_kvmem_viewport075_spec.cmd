@echo off
rem ===========================================================================
rem STEP 075 EXPERIMENT ENTRY (KVMem ARM, NOT production): the step 074 arm with
rem the PRODUCTION speculation recipe bolted on -- DFlash2 gptq3c, N=2, and
rem --cudagraph-capture-sizes 3 (1 verified + 2 drafted tokens, same as
rem tools/serve_gsq_prod029_n2.cmd). Everything else is byte-identical to the
rem 074 viewport arm (viewport + sweep + raw-K + authority, pool 3.4e9, nvfp4,
rem ssm bf16, G=8, pin shim, prefix caching, --max-num-batched-tokens 1424).
rem
rem Why: the user green-lit speculation for the KVMem arm (2026-10-01) so the
rem arm stops being stuck at 69 tok/s (no spec) while production sits at 122.58
rem (spec DFlash2 N=2). This boot is the OBSERVATION run, no code changed:
rem   1. does the engine still boot with the KVMem connector + a drafter? the
rem      drafter brings its own sliding-window kv group (5 layers) --
rem      groups.workspace_group_ids filters on VLLM_KVMEM_SW_WINDOW so it must
rem      NOT be mistaken for a stored group;
rem   2. the K2 admission guard (viewport needs N blocks / pool has 259) with
rem      the draft groups added to the per-request block demand;
rem   3. capture.py only recognises decode as "num_tokens <= 1". A verify step
rem      is 3 tokens, so draft/verify rows may now be stashed, drained and fed
rem      to the index/authority -- watch 'KVMem capture drain' lines and the
rem      single-token judge counter.
rem Dump dir: kvmem_k9d.
rem
rem OBSERVED FIRST BOOT (kvmem_k9d_spec_boot1, 2026-10-01 21:44) = engine refused
rem to start: "To serve at least one request with the model's max seq len
rem (262144), 3.24 GiB KV cache is needed, which is larger than the available KV
rem cache memory (3.15 GiB) ... estimated maximum model length is 160160". The
rem pool value stays at 3.4e9 (hard ceiling, steps 034/035/040/046), so the only
rem lever left is the per-request demand: --max-model-len drops 262144 -> 200704,
rem which still admits the 198,184-token benchmark prompt plus a 2,000-token
rem answer, and the drafter's groups now fit. The KVMem workspace ceiling
rem (VLLM_KVMEM_WORKSPACE_TOKENS, 262,144) is untouched -- this is the engine's
rem context limit, not the host store.
rem
rem OBSERVED SECOND BOOT (kvmem_k9d_spec_boot3, W=131072, 21:52) = the KV capacity
rem check passed but CUDA graph capture aborted: "torch.AcceleratorError: CUDA
rem error: operation failed due to a previous error during capture
rem (cudaErrorStreamCaptureInvalidated)" inside qwen3_next.forward. Root cause =
rem a verify step is 3 tokens, so capture.py's "num_tokens <= 1 means decode"
rem rule recorded it, and the body's clone + M-RoPE torch.equal cannot run inside
rem a capture. Fix = the step 075 allow-list (capture.arm from the connector's
rem prefill spans before the forward, capture.disarm after); with W=131072 the
rem engine's page size becomes 1456 (attn block_size [1456, 1456]), so
rem --max-num-batched-tokens pins 1456 to keep every prefill step page-aligned.
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
set "VLLM_KVMEM_VIEWPORT=1"
set "VLLM_KVMEM_VIEWPORT_PAGES=55"
set "VLLM_KVMEM_VIEWPORT_RECENT=16384"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k9d"
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
