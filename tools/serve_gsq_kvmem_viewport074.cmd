@echo off
rem ===========================================================================
rem STEP 074 EXPERIMENT ENTRY (KVMem ARM, NOT production): the step 073 recipe
rem with the retrieval-slot bake fixed to cover EVERY stored kv cache group, plus
rem a read-back check that proves the baked bytes landed in the physical block
rem the decode step reads.
rem
rem Why: step 073 proved the compressed window runs end to end (68 prefill steps
rem -> 55 pages baked -> 340 chunks of reasoning) but an out-of-window needle is
rem still unreadable, while the same prompt prefilled natively answers it. The
rem bake log of that run said "688.4 MiB written" for 55 slots, which is exactly
rem 55 x 1,640,448 B x 8 layers: under VLLM_KV_GROUP_SIZE=8 the 16 full-attention
rem layers sit in TWO kv cache groups (6 and 7), and _emit_stage_request only
rem read tables[group_ids[0]], so group 7's 8 layers kept the placeholder prefill
rem in their retrieval slots. Half the attention never saw the needle page.
rem
rem What this boot has to show:
rem   stage-in plan: N slot(s) x 2 group(s) = 2N block(s) ... (covered [6, 7])
rem   baked N slot(s) x group(s) [6, 7] = N*16 layer-page copies ... ~2x MiB
rem   read-back V checked, 0 mismatch
rem Judgement (unchanged from 073): OUT-OF-WINDOW needle (depth 0.65, page 88)
rem answered from the retrieval slots, coherent output, no assembly-chain
rem regression, double boot.
rem
rem VLLM_KVMEM_BAKE_VERIFY=1 reads back the first layer-page of every slot of
rem every group (110 blocking D2H copies of 1.57 MiB); set it to 0 to drop the
rem check. VLLM_KVMEM_DEBUG stays on for the window observation lines.
rem
rem Dump dir: step 074 used kvmem_k9a (boot1) then kvmem_k9b (boot2); the
rem committed value below is the LAST one used, so a re-run must set a NEW dir
rem (the worker's kvmem_retrieval_%03d counter restarts every boot, so reusing a
rem directory silently overwrites the earlier boot's evidence).
rem Everything else (pool 3.4e9, nvfp4, ssm bf16, G=8, pin shim, graph mode,
rem prefix caching, --max-num-batched-tokens 1424) matches the step 072/073 arm.
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md (section 12.15 for steps 073-074).
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
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\kvmem_k9b"
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
