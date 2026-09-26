@echo off
rem ===========================================================================
rem serve_minicpm5_2b_dspark_base029.cmd - S3 smoke on the PURE 0.29 base.
rem MiniCPM5-2B, port 8081 (API key sk-minicpm5). Co-resident with bge-m3
rem (conda env, port 6380, ~1.9GB fp16).
rem Derived from vllm-win\serve_minicpm5_2b_dspark.cmd (0.27.1 production);
rem changes: python = vllm-win029 (cp312), env contract per
rem run_anchor_ppl_base029.cmd (vcvars64 + HOME=C:\fi + CUDA 8.3 short path),
rem FLASHINFER_EXTRA_LDFLAGS repointed to vllm-win029 tvm_ffi.
rem NOTE --gpu-memory-utilization dropped: with --kv-cache-memory-bytes set,
rem 0.29 skips VRAM profiling and gpu_mem_util has no effect.
rem KV pool 946,234,982 B = 1.2x fp8 floor of 32768 x 24064 B/token (as prod).
rem
rem --- S3b PENDING batch-4: dspark spec disabled here ------------------------
rem The production config carries
rem   --speculative-config {"model":"...MiniCPM5-2B/dspark","method":"dspark",
rem    "num_speculative_tokens":7}
rem but on the pure 0.29 base this hard-fails at cudagraph capture:
rem   ValueError: xqa backend does not support cum_seq_lens_q
rem Root cause: SM120 statically selects the XQA decode kernel
rem (_get_flashinfer_trtllm_api_decode_kernel, unconditional on capability
rem family 120, KV-dtype independent), and XQA speculative decode is "not
rem wired in vLLM yet" (0.29 guard in flashinfer.py raises
rem NotImplementedError; the varlen path hits the flashinfer-side ValueError
rem first). Any multi-token spec decode (dspark/DFlash2/MTP) is structurally
rem unavailable on the bare 0.29 base -- same class of gap as nvfp4-KV
rem (batch-5). Re-enable the speculative-config line after batch-4 spec
rem wiring + batch-5 flashinfer.py hunks land. S3a co-residency + A6 memory
rem ledger run without spec; dspark re-smoke is batch-4 acceptance scope.
rem ===========================================================================
set "ROOT=G:\qwen3.8model"
set "MODEL=%ROOT%\MiniCPM5-2B"
set "DRAFT=%ROOT%\MiniCPM5-2B\dspark"

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
set "TMP=G:\qwen3.8model\_tmp_s3"
set "TEMP=G:\qwen3.8model\_tmp_s3"
set "PATH=%ROOT%\vllm-win029\Scripts;%PATH%"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

rem zmq orphan guard on 29550
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"%ROOT%\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "%MODEL%" ^
  --served-model-name minicpm5-2b ^
  --host 0.0.0.0 --port 8081 ^
  --api-key sk-minicpm5 ^
  --kv-cache-memory-bytes 946234982 ^
  --dtype bfloat16 ^
  --kv-cache-dtype fp8 ^
  --max-model-len 32768 ^
  --max-num-seqs 20 ^
  --max-num-batched-tokens 10240 ^
  --cudagraph-capture-sizes 1 2 4 8 16 32 ^
  --override-generation-config "{\"max_new_tokens\": 1024}"
exit /b %errorlevel%
