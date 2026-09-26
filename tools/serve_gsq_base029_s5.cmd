@echo off
rem ===========================================================================
rem serve_gsq_base029.cmd - S4/S5 smoke on the PURE 0.29 base.
rem Qwen3.8-27B-3Bit-GSQ (compressed_tensors WNA16 int3 g128 + embed int4 g64),
rem port 8000. Tool-call + reasoning parsers per production serve_qwen38_gsq.cmd.
rem User directive (2026-09-26): test KV = fp8 (bf16 caps max-model-len 52288,
rem not enough context).
rem
rem --- KV pool (manual, iron rule) ------------------------------------------
rem fp8 KV price, engine-measured = 42,272 B/token at 32k boot (first guess
rem 32,768 B/token from pure full-attn accounting was 29% low -- GDN layer
rem states ride in the same pool; engine floor for one 32k request was
rem 1.29 GiB). Pool = 32,768 x 42,272 x 1.2 = 1,662,441,472 B.
rem
rem --- spec decode NOT in this config ---------------------------------------
rem XQA spec decode is not wired on the bare 0.29 base (see
rem serve_minicpm5_2b_dspark_base029.cmd header). DFlash2 lands with batch-4.
rem ===========================================================================
set "ROOT=G:\qwen3.8model"
set "MODEL=%ROOT%\Qwen3.8-27B-3Bit-GSQ"

if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "HF_HOME=%ROOT%\hub"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "FLASHINFER_WORKSPACE_BASE=C:\fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "VLLM_LOGGING_LEVEL=INFO"
set "TMP=G:\qwen3.8model\_tmp_s4"
set "TEMP=G:\qwen3.8model\_tmp_s4"
set "PATH=%ROOT%\vllm-win029\Scripts;%PATH%"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

rem S5 PIECEWISE retest (batch-3b): apply_humming_linear is now an opaque
rem custom op (vllm::apply_humming_linear), so dynamo no longer traces the
rem humming python pre-processing. Judgment: engine boots + functional
rem request passes; throughput is recorded, no bar set.
rem zmq orphan guard on 29550
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"%ROOT%\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "%MODEL%" ^
  --served-model-name gsq-27b ^
  --host 127.0.0.1 --port 8000 ^
  --kv-cache-memory-bytes 1662441472 ^
  --dtype auto ^
  --kv-cache-dtype fp8 ^
  --max-model-len 32768 ^
  --max-num-batched-tokens 8192 ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --compilation-config {\"cudagraph_mode\":\"PIECEWISE\"} ^
  --language-model-only
exit /b %errorlevel%
