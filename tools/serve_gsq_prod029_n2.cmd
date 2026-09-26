@echo off
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
  --max-model-len 110000 ^
  --max-num-seqs 1 --max-num-batched-tokens 1024 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --compilation-config {\"cudagraph_mode\":\"PIECEWISE\"} ^
  --cudagraph-capture-sizes 3 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2 ^
  --dtype auto
exit /b %errorlevel%
