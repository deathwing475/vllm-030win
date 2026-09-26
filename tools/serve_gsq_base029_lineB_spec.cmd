@echo off
rem ===========================================================================
rem Line B of the three-line merged regression (stage-2 acceptance, 2026-09-26).
rem Spec end-to-end on the 0.29 base, CONFIG-IDENTICAL to the A5 anchor launcher
rem (run_anchor_a5.cmd body, 0.27 overlay stack) plus DFlash2 draft N=2:
rem   nvfp4 KV + manual pool 3,400,000,000 + max-model-len 110000
rem   PIECEWISE + --cudagraph-capture-sizes 3 + mamba-cache-mode align
rem   seqs 1 / batched-tokens 1024 / language-model-only / no prefix-cache
rem   draft = dflash2\gptq3c (auto-round 2-bit, same checkpoint as anchor)
rem Draft sliding-window KV rides nvfp4 like the anchor (skip-layers flag was
rem REMOVED in the anchor recipe -- everything quantized; 0.29 default same).
rem Adaptive verification stays OFF (overlay hard-coded False; 0.29 default
rem False == anchor semantics).
rem Anchors to beat: acc 0.68-0.80 (stage0_ttft real prompts, /metrics ratio);
rem throughput 8k 57.5 / 32k 34.1 / 100k 18.9 (anchor_longctx haystack);
rem doc_4k cell4 median 60.65 (runner v2).
rem speculative-config keys passed individually (argv-probe verified 2026-09-26:
rem dot-form tokens land intact incl. backslash paths).
rem ===========================================================================
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist G:\qwen3.8model\_tmp_line_b mkdir G:\qwen3.8model\_tmp_line_b
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
set "TMP=G:\qwen3.8model\_tmp_line_b"
set "TEMP=G:\qwen3.8model\_tmp_line_b"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
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
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --no-enable-prefix-caching ^
  --compilation-config {\"cudagraph_mode\":\"PIECEWISE\"} ^
  --cudagraph-capture-sizes 3 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2 ^
  --dtype auto
exit /b %errorlevel%
