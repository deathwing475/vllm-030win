@echo off
setlocal
rem step 089 (Orca O3 correction): the 088 DFlash2 arm rebuilt with the capacity levers that
rem GSQ production actually uses on THIS SAME 16 GB card with THIS SAME draft (gptq3c, N=2) at
rem max-model-len 163,072. Step 088 concluded "DFlash2 does not fit on 16 GB", but that was a
rem verdict about 088's OWN arm configuration, not about the card: 088 ran without any of the
rem five levers below, so the per-request floor it measured (198 KiB/token) is an artifact.
rem
rem The levers, taken from tools/serve_gsq_prod029_n2.cmd (each one is a measured step, not a
rem guess):
rem   VLLM_KV_GROUP_SIZE=8                 step 036  (+18.5% capacity; 16 full-attn layers and
rem                                                  48 GDN layers collapse into 2+6 groups
rem                                                  instead of one group per layer)
rem   --mamba-ssm-cache-dtype bfloat16     steps 045/046 (mamba page 3,248,128 -> 1,675,264 B,
rem                                                  attn block 2832 -> 1456, +12.5%)
rem   --mamba-cache-mode align             as production
rem   --enable-prefix-caching              as production
rem   --kv-offloading-backend native 8 GiB step R8-era production: KV spills to host mmap files
rem                                                  (the "offload 8G" of production; this is why
rem                                                  production can hold 163,719 tokens)
rem   --gpu-memory-utilization 0.922       production's own value. NOTE: 085's "0.95 is
rem                                                  unreachable" was measured against startup
rem                                                  free 14.68/15.89 GiB; 0.922 is what
rem                                                  production runs every day, so 088's 0.88 was
rem                                                  a self-imposed handicap, not a card limit.
rem
rem Kept identical to 088 so the comparison stays one-variable: same draft, nvfp4 KV, eager,
rem max-num-seqs 1, ORCA_EXL3_ALLOW_EMPTY_SHARED=1, pin_shim on PYTHONPATH.
rem
rem Knobs (all optional): S089_L / S089_MBT / S089_UTIL / S089_SPEC / S089_OFFLOAD, and
rem S089_POOL to pin --kv-cache-memory-bytes (empty by default: let the engine size the pool,
rem then read its own number from the log -- iron rule 28).
set "VENV=G:\qwen3.8model\vllm-win029"
set "MODEL=G:\qwen3.8model\qwen3.8exl3"
set "DRAFT=G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c"
set "HOME=C:\fi"
set "USERPROFILE=C:\fi"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "TMP=G:\qwen3.8model\_tmp_orcasaq2"
set "TEMP=G:\qwen3.8model\_tmp_orcasaq2"
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\orcasaq2_sitecustomize;G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "OMP_NUM_THREADS=8"
set "VLLM_KV_GROUP_SIZE=8"
set "ORCA_EXL3_ALLOW_EMPTY_SHARED=1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
if "%S089_L%"=="" set "S089_L=16384"
if "%S089_MBT%"=="" set "S089_MBT=1024"
if "%S089_UTIL%"=="" set "S089_UTIL=0.922"
if "%S089_SPEC%"=="" set "S089_SPEC=2"
if "%S089_OFFLOAD%"=="" set "S089_OFFLOAD=8"
rem S089_NOEAGER=1 drops --enforce-eager, i.e. the engine default graph mode. Production
rem (serve_gsq_prod029_n2.cmd) carries NO --compilation-config line at all since step 027,
rem which means it runs on the engine default FULL_AND_PIECEWISE -- so "graph tier here"
rem means: no enforce-eager plus --cudagraph-capture-sizes 3, exactly like production.
rem Iron rule 17: an L/dtype/graph change invalidates the AOT cache, so the FIRST such
rem boot is the compile boot (首编深塌) and may not be read as a speed number.
set "EAGERFLAG=--enforce-eager"
if "%S089_NOEAGER%"=="1" set "EAGERFLAG="
if "%S089_CGS%"=="" set "S089_CGS=3"
rem S089_GRAPHPROF=0 -> VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0: the memory profiler
rem then stops reserving the CUDA-graph capture peak out of the KV pool. Measured on this
rem arm (s1, L=16,384): graphs-on with the default reservation left only 0.32 GiB of KV
rem against 0.65 GiB needed, i.e. the graph reservation costs ~1.3 GiB versus the eager
rem boot's 1.64 GiB. Setting it to 0 is a bet that the real capture peak stays under the
rem pool, so it must be judged by whether the boot and the probes survive, not by the
rem banner. Production runs the default (1).
if "%S089_GRAPHPROF%"=="" set "S089_GRAPHPROF=1"
set "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=%S089_GRAPHPROF%"
set "POOLARG="
if not "%S089_POOL%"=="" set "POOLARG=--kv-cache-memory-bytes %S089_POOL%"
rem Reclaim stale offload mmaps the way production does (names are per-engine unique).
del /q "G:\qwen3.8model\_tmp_orcasaq2\vllm_offload_*.mmap" 2>nul
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len %S089_L% --gpu-memory-utilization %S089_UTIL% --max-num-batched-tokens %S089_MBT% --max-num-seqs 1 %EAGERFLAG% --dtype auto --kv-cache-dtype nvfp4 --cudagraph-capture-sizes %S089_CGS% --mamba-cache-mode align --mamba-ssm-cache-dtype bfloat16 --enable-prefix-caching --kv-offloading-backend native --kv-offloading-size %S089_OFFLOAD% %POOLARG% --speculative-config.method dflash --speculative-config.model "%DRAFT%" --speculative-config.num_speculative_tokens %S089_SPEC%
exit /b %errorlevel%
