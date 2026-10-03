@echo off
setlocal
rem step 088 (Orca O3: DFlash2) = the 086 nvfp4 arm with the draft swapped from the
rem checkpoint-native MTP head to DFlash2. Single variable vs b1 (auto KV) = the KV dtype,
rem and single variable vs 086's p1/p2/p3 = the draft family.
rem
rem Why this arm exists: b1 (auto KV + DFlash2 N=2, 12000/0.88) was refused by the engine
rem itself -- "To serve at least one request with the model's max seq len (12000), 1.66 GiB
rem KV cache is needed, which is larger than the available KV cache memory (0.82 GiB) ...
rem estimated maximum model length is 1600". Weights came up 11.46 -> 12.23 GiB for the
rem 0.545 GiB draft and the 5 draft layers' own KV is on top of the target's, so on this
rem 16 GB card auto KV cannot carry DFlash2 at any useful context (iron rule 28: read the
rem engine's own estimate, do not reason from page x blocks).
rem
rem nvfp4 is the precision Orca is actually served in (084/O1, 086), and it doubles the KV
rem pool, so it is the only tier with a chance here.
rem
rem mbt 2848 = "just over the page" per iron rule 11 (nvfp4 page 2784, 2832/2816 once a
rem draft is attached; 086 measured that going to 3072 eats the KV budget bare).
rem S088_SPEC=2 is this arm's point (DFlash2 N=2); note 086 judged nvfp4 x multi-step MTP
rem NEGATIVE because prompts >= ~10K never produced a first token, so a long-request probe
rem is mandatory here -- a chat-only pass would hide that whole fault class.
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
set "ORCA_EXL3_ALLOW_EMPTY_SHARED=1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
if "%S088_SPEC%"=="" set "S088_SPEC=2"
rem S088_L: b2 measured that at 16,384 / 0.88 the nvfp4+DFlash2 tier only has 0.27 GiB of KV
rem left against 1.03 GiB needed, so the engine refuses to boot. Lowering L (with the needle
rem probe shrunk to match) is what lets the draft actually run and be measured; the capacity
rem verdict stays separate from the acceptance-rate verdict.
if "%S088_L%"=="" set "S088_L=16384"
rem S088_UTIL: b4 showed the shortfall is a per-request FLOOR, not a linear term (4096 tokens
rem still needs 0.77 GiB against 0.27 GiB), so the only way to see the draft run at all is to
rem take more of the card. 085 measured startup free at 14.68/15.89 GiB, so ~0.92 is the
rem ceiling and 0.95 is unreachable. This is a measurement tier only: it deliberately spends
rem the step-049 headroom cliff and must never be copied into a production launcher.
if "%S088_UTIL%"=="" set "S088_UTIL=0.88"
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len %S088_L% --gpu-memory-utilization %S088_UTIL% --max-num-batched-tokens 2848 --max-num-seqs 1 --enforce-eager --dtype auto --kv-cache-dtype nvfp4 --speculative-config.method dflash --speculative-config.model "%DRAFT%" --speculative-config.num_speculative_tokens %S088_SPEC%
exit /b %errorlevel%
