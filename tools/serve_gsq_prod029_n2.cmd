@echo off
rem 2026-09-28 step 036 (KV regroup, PRODUCTION): VLLM_KV_GROUP_SIZE=8
rem forces the KV layers-per-group from the upstream 5 (pinned by the
rem draft's 5 sliding-window layers) to 8. Measured on this stack: the
rem 3.4e9 pool goes 114,974 -> 136,190 tokens (+18.5%), decode step
rem 18.73/18.74 -> 18.30/18.31/18.32 ms, prefill unchanged, needle green
rem at 8k/32k/64k/100k (max-model-len 110000) and 8k/64k/130k (140000).
rem max-model-len: 110000 -> 130000 -> 140000 (step 039). The G=8 hard
rem ceiling is 144,432 (blocks_per_req = 2*cdiv(L,2832) + 27 <= 130), but
rem 144,432 measures ~21% slower decode (2 boots: steady 95-105 vs
rem 119-128) and 141,600 already costs ~5%, so 140,000 is the largest
rem value whose performance matches 130,000 (2 boots: steady 118-128).
rem Engine reports 143,307 tokens (1.02x) vs 142,016 at 130000.
rem Pool stays the manual 3,400,000,000 B: KV tensor size and spill are
rem unchanged; only grouping and the logical ceiling move.
rem ROLLBACK = delete the VLLM_KV_GROUP_SIZE line, restore 110000, drop
rem this note (or: python tools\apply_prod_kvgroup_step036.py revert).
rem ===========================================================================
rem PRODUCTION launcher, vLLM 0.29 stack (stage-4 switchover, 2026-09-26).
rem Deployment twin of the regression-verified tools/serve_gsq_base029_lineB_spec.cmd
rem (line-B spec e2e PASSED: A2 73.00 / A5 68.59-24.39 / A4 acc 0.72-0.80),
rem plus the production-only KV feature kept from the 0.27 recipe
rem run_dflash2_n2.cmd (N=2 PRODUCTION, 2026-09-25 user decision):
rem   + --enable-prefix-caching
rem KV offloading is ON (user priority: multi-turn conversations rely on the
rem offload tier to keep earlier turns in CPU RAM). It crashed the stage-4
rem drill (2026-09-26) and was root-caused the same day to a cuMemcpyBatchAsync
rem driver defect on non-default streams; fixed in base d7cdb91 by routing
rem swap_blocks_batch to per-copy cuMemcpyAsync on Windows. Verified: 8k x2
rem needle green with repeat-request TTFT 4.7s -> 1.5s (prefix hit).
rem KV pool stays MANUAL (iron rule): 3,400,000,000 B (121,058 tokens with draft,
rem engine self-report on the 0.29 stack).
rem
rem Delta vs the frozen 0.27 recipe (all intentional):
rem   1. venv G:\qwen3.8model\vllm-win029 (0.29 base + batch-1..5 + regression
rem      fixes 6fc2108/e9d534d); PYTHONPATH overlay GONE.
rem   2. spec via 0.29 native --speculative-config.* keys (argv-probe verified;
rem      replaces 0.27 custom --spec-method/--spec-model/--spec-tokens).
rem   3. flashinfer 0.6.18 runner contract: FLASHINFER_WORKSPACE_BASE=C:/fw
rem      (FORWARD slashes -- 0.6.18 eats "\f" as form-feed) + tvm_ffi LDFLAGS
rem      pointing at vllm-win029. VLLM_HAS_FLASHINFER_CUBIN dropped (0.27-era
rem      flag, not part of the validated 0.29 contract).
rem   4. TMP/TEMP -> G:\qwen3.8model\_tmp_prod029 (offload mmap + JIT build
rem      temps stay off C:).
rem
rem ROLLBACK: the frozen 0.27 recipe run_dflash2_n2.cmd + venv
rem G:\qwen3.8model\vllm-win (read-only) remain untouched -- see
rem docs/切换与回退预案.md. Never run this from the record-repo directory
rem (cwd would capture the 0.27.1 tree lying there).
rem
rem 2026-09-27 graph-mode swap (step 027): the --compilation-config
rem {"cudagraph_mode":"PIECEWISE"} line is REMOVED, so the engine default
rem FULL_AND_PIECEWISE applies (prefill PIECEWISE / decode FULL whole-graph
rem replay). Explicit PIECEWISE disables FULL graphs and eagerly dispatches
rem 90+ splitting ops per decode step; the FULL decode graph is verified
rem safe with DFlash2 spec (three-segment capture passes, step 026) and was
rem measured 25.86-26.45ms vs 27.0ms over 3 boots (step 026, -4%).
rem --cudagraph-capture-sizes 3 kept. Long-stability soak + correctness
rem gates (step 027) precede this file becoming the live production config.
rem Rollback of this swap = re-add the compilation-config line above.
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
rem Fast-tier pin (step 030, 2026-09-27): inject
rem tools\pin_shim\sitecustomize.py via PYTHONPATH. MIN mode hooks only
rem load_dflash_model and pins the draft weights (gptq3c, 31 Humming
rem layers) into dedicated VRAM after their free=0 pressure-window
rem placement. Verified 21/21 boots fast-tier 20.3-21.6ms (clean boots
rem draw 20% fast / 80% mid 21-27ms) with acc 0.66-0.75 + needle green.
rem ROLLBACK = remove the 4 set lines below.
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_KV_GROUP_SIZE=8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
rem Sweep stale offload mmaps (crash/held-ref exits leave them; names are
rem per-engine unique so they never collide -- just reclaim the space).
del /q "G:\qwen3.8model\_tmp_prod029\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ" ^
  --served-model-name qwen3.8-27b-gsq ^
  --host 127.0.0.1 --port 8080 ^
  --language-model-only ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes 3400000000 ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len 140000 ^
  --max-num-seqs 1 --max-num-batched-tokens 1024 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 3 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2 ^
  --dtype auto
exit /b %errorlevel%
