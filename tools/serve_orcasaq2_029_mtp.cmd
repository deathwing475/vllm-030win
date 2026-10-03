@echo off
setlocal
rem step 085 (Orca O2) = tools/serve_orcasaq2_029.cmd verbatim + MTP speculation.
rem Spec keys are passed individually (no JSON quoting), same house style as the GSQ
rem spec launchers. method=mtp -> draft arch comes from the checkpoint config
rem (model_type qwen3_5 + mtp_num_hidden_layers=1 -> Qwen3_5MTP); the draft model path
rem defaults to the target model path, so no --speculative-config.model is needed.
rem Gpu-memory-utilization stays 0.88 (059/O1), but max-model-len drops 16384 -> 12000: with
rem the MTP head a single 16,384-token request needs 1.71 GiB of KV while 0.88 only leaves
rem 1.44 GiB, and the engine itself estimated the maximum model length at 12,000 (b1,
rem 2026-10-03). 0.95 is not reachable on this card at all -- free memory at startup is
rem 14.68/15.89 GiB, below the 15.1 GiB that 0.95 asks for. The control arm uses the same
rem 0.88 / 12000 so speculation stays the only variable.
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
rem s085: the MTP draft shares embed_tokens/lm_head with the target, so those two draft
rem modules are empty during loading; orcasaq2 raises on empty unless this is set.
set "ORCA_EXL3_ALLOW_EMPTY_SHARED=1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len 12000 --gpu-memory-utilization 0.88 --max-num-seqs 1 --enforce-eager --dtype auto --speculative-config.method mtp --speculative-config.num_speculative_tokens 2
exit /b %errorlevel%
