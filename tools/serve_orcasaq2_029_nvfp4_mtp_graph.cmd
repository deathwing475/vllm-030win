@echo off
setlocal
rem step 093-A variant of tools/serve_orcasaq2_029_nvfp4_mtp.cmd (086): identical arm
rem (nvfp4 KV, L=16,384, util 0.88, mbt 2848, max-num-seqs 1, MTP spec steps = S086_SPEC,
rem default 1) minus --enforce-eager, i.e. the graph tier that step 090 proved healthy for
rem Orca DFlash2 (FULL_AND_PIECEWISE + capture sizes 3). Purpose: the same-gauge comparison
rem of 093-A must not hand the MTP tier a 4x eager handicap (iron rule 8) that DFlash2 does
rem not pay. The MTP draft path inside CUDA graphs was never measured before, so the first
rem boot is a smoke: judge by boot + probes, not by assumption.
rem Graph-profiling reservation: 086 measured only ~0.90 GiB free KV at util 0.88, and the
rem default VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 reserves ~1.3 GiB for the capture
rem peak (090 s1), which cannot fit -- so this variant defaults it to 0 like the 090 speed
rem recipe. Set V093_GRAPHPROF=1 only to reproduce the refused boot.
set "VENV=G:\qwen3.8model\vllm-win029"
set "MODEL=G:\qwen3.8model\qwen3.8exl3"
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
set "ORCA_EXL3_ALLOW_EMPTY_SHARED=1"
if "%V093_CGS%"=="" set "V093_CGS=3"
if "%V093_GRAPHPROF%"=="" set "V093_GRAPHPROF=0"
set "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=%V093_GRAPHPROF%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
if "%S086_SPEC%"=="" set "S086_SPEC=1"
rem V093_UTIL: 0.88 (the 086 eager tier's value) is NOT enough once graphs capture -- the
rem capture peak eats ~0.64 GiB of the 1.01 GiB the eager boot had, leaving 0.37 GiB against
rem 0.77 GiB needed for one 16,384-token request (mg1, refused). 0.92 is the measured-tier
rem utilisation the 090 DFlash2 speed recipe also uses; never a production value.
if "%V093_UTIL%"=="" set "V093_UTIL=0.92"
rem V093_POOL: pin --kv-cache-memory-bytes. mg1b (auto pool at util 0.92) let the engine
rem size the pool to 39,321 tokens (~1.9 GiB) and hit the residency slow mode (093 evidence:
rem util.gpu 100% / util.mem 0% / 74 W p50, TTFT 127 s, 43 tok/s) -- iron rule 33, the pool
rem byte count is the speed switch on this card. need(16,384) = 0.77 GiB on this tier
rem (engine refusal line), so 1.0e9 = need*1.2 leaves the same ~83% occupancy the healthy
rem DFlash2 tier runs at. Leave empty to reproduce the slow auto-pool mode.
set "POOLARG="
if not "%V093_POOL%"=="" set "POOLARG=--kv-cache-memory-bytes %V093_POOL%"
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization %V093_UTIL% --max-num-batched-tokens 2848 --max-num-seqs 1 --dtype auto --kv-cache-dtype nvfp4 --cudagraph-capture-sizes %V093_CGS% %POOLARG% --speculative-config.method mtp --speculative-config.num_speculative_tokens %S086_SPEC%
exit /b %errorlevel%
