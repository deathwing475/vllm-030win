@echo off
rem ===========================================================================
rem STEP 078 ARM -- one launcher, one variable flipped per boot.
rem
rem Byte-identical to tools/serve_gsq_kvmem_viewport077_spec.cmd (page 1456,
rem mbt 1458 = page + 2 DFlash draft slots, SW_WINDOW 131072, L 200704, pool
rem 3.4e9, FULL graph, bake read-back) EXCEPT:
rem   * VLLM_SCHED_TRACE removed -- the step 077 diagnostic patch is reverted,
rem     so the env would only be noise.
rem   * knobs read from the environment so a peel needs no new file:
rem       S078_TAG   dump dir name   (ONE FRESH DIR PER BOOT, iron rule 16-8)
rem       S078_SPEC  1 = DFlash2 N=2 armed, 0 = no speculation at all
rem       S078_RAWK  1/0 raw-K capture (VLLM_KVMEM_RAWK)
rem       S078_AUTH  1/0 raw-K authority region (VLLM_KVMEM_AUTHORITY)
rem       S078_WS    1/0 the workspace connector itself (VLLM_KVMEM_WORKSPACE)
rem     --max-num-batched-tokens stays 1458 in BOTH spec settings on purpose:
rem     the control differs from the slow arm by speculation and nothing else,
rem     and with draft_slots = 0 the align split still floors to one 1456 page
rem     (the capture drain line is the witness).
rem
rem What this arm is for: step 077 measured ~11 s per 1456-token page step on
rem the spec'd workspace arm against ~1.4 s on the step 074 arm, and step 074's
rem arm differs in more than one variable (SW_WINDOW 163072 vs 131072, L, mbt,
rem capture sizes). So 078(b) first reproduces the 11 s on a fresh boot, then
rem peels spec / rawK / authority / connector at fixed geometry, with a single
rem 60K ingest as the cadence probe (tools/kvmem_ingest_cadence.py).
rem 078(a) then runs the full three-request out-of-window needle probe here.
rem ===========================================================================
if "%S078_TAG%"=="" set "S078_TAG=kvmem_k12a"
if "%S078_SPEC%"=="" set "S078_SPEC=1"
if "%S078_RAWK%"=="" set "S078_RAWK=1"
if "%S078_AUTH%"=="" set "S078_AUTH=1"
if "%S078_WS%"=="" set "S078_WS=1"
set "SPEC_ARGS="
if "%S078_SPEC%"=="1" set "SPEC_ARGS=--speculative-config.method dflash --speculative-config.model G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c --speculative-config.num_speculative_tokens 2"
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
set "VLLM_KVMEM_WORKSPACE=%S078_WS%"
set "VLLM_KVMEM_WORKSPACE_MB=5120"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=%S078_RAWK%"
set "VLLM_KVMEM_AUTHORITY=%S078_AUTH%"
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
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S078_TAG%"
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
  --max-num-seqs 1 --max-num-batched-tokens 1458 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 3 ^
  %SPEC_ARGS% ^
  --dtype auto
exit /b %errorlevel%
