@echo off
setlocal
rem step 088 (Orca O3: DFlash2 on the EXL3 Orca target) = the 085 auto-KV arm with the
rem draft swapped from the checkpoint-native MTP head to DFlash2.
rem
rem Draft = G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c (AutoRound GPTQ, 0.545 GiB).
rem Route chosen by the user on 2026-10-03 (step 087): reuse the GSQ draft, the plan's
rem "do not reuse gptq3c" ban is lifted FOR O3/088 ONLY. Known caveat: gptq3c's GPTQ
rem Hessians were captured on the GSQ stack, so its error compensation is extrapolated
rem to Orca's hidden-state distribution. The gate is therefore the measured acceptance
rem rate against GSQ's own band (62.82%), not "it booted".
rem
rem Why 12000 / 0.88 / mbt 2048: with a draft attached, one 16,384-token request needs
rem more KV than the 0.88 tier leaves on this 16 GB card (085 measured the engine's own
rem "estimated maximum model length" at 12,000 for auto KV + speculation), and
rem --gpu-memory-utilization 0.95 is unreachable (startup free 14.68/15.89 GiB).
rem mbt 2048 clears the auto-KV attention page (784 without a draft, 800 with one) plus
rem the 2 draft slots that iron rule 11/28 requires (mbt >= page + num_spec), and it is
rem the value 085's auto arm used, so 085-vs-088 stays a one-variable comparison.
rem
rem ORCA_EXL3_ALLOW_EMPTY_SHARED=1: step 087's meta probe showed the draft's own
rem embed_tokens/lm_head get claimed by the EXL3 plugin (orcasaq2 patches
rem VocabParallelEmbedding.__init__ globally) while the DFlash2 checkpoint ships neither,
rem i.e. the same empty-shared-module shape that killed MTP's boot in 085.
rem
rem DFlash2 only runs on the V2 model runner (config/vllm.py lists "dflash2 drafts" as
rem V1-unsupported, and on V1 the same checkpoint would silently degrade to DFlash1), so
rem every boot here must show "Using V2 Model Runner" in its log.
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
rem S088_SPEC = draft steps per decode step (the DFlash2 block is 1 + this value).
if "%S088_SPEC%"=="" set "S088_SPEC=2"
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len 12000 --gpu-memory-utilization 0.88 --max-num-batched-tokens 2048 --max-num-seqs 1 --enforce-eager --dtype auto --speculative-config.method dflash --speculative-config.model "%DRAFT%" --speculative-config.num_speculative_tokens %S088_SPEC%
exit /b %errorlevel%
