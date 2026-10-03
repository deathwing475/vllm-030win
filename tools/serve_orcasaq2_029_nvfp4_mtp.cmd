@echo off
setlocal
rem step 086 (Orca O2 x NVFP4) = tools/serve_orcasaq2_029_nvfp4.cmd (084/O1) + MTP speculation.
rem max-num-batched-tokens = 2848, i.e. the minimum that clears the nvfp4+spec attention page
rem (measured 2832 tokens/page) plus the 2 draft slots: iron rule 11 needs
rem mbt >= page + num_spec, because mamba 'align' otherwise trims the first block to 0 tokens
rem and the scheduler spins in silence. It is NOT higher on purpose: every spare token of
rem batch budget is activation peak taken out of the KV pool (at 3072 one 16,384-token request
rem needed 0.93 GiB against 0.90 GiB available and the engine refused to boot). The control
rem arm uses the identical 2848 so speculation stays the only variable.
rem method=mtp: the draft arch comes from the checkpoint config (model_type qwen3_5 +
rem mtp_num_hidden_layers=1 -> Qwen3_5MTP); the draft path defaults to the target model.
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
rem S086_SPEC = draft steps per decode step. DEFAULT 1, because 086 measured that 2 steps on
rem the nvfp4 KV path never return a long prompt (>=~10K tokens: 90 s+ with the GPU pinned at
rem 100%, while the same arm at 1 step answers 10,704 tokens in 6.5 s and 14,241 in 5.6 s, and
rem the no-speculation control passes both). Set S086_SPEC=2 only to reproduce that stall.
rem One step is also what a single MTP layer is natively sized for (085: pos-2 acceptance 0.49-0.52).
if "%S086_SPEC%"=="" set "S086_SPEC=1"
cd /d G:\qwen3.8model
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve "%MODEL%" --served-model-name orcasaq2 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-batched-tokens 2848 --max-num-seqs 1 --enforce-eager --dtype auto --kv-cache-dtype nvfp4 --speculative-config.method mtp --speculative-config.num_speculative_tokens %S086_SPEC%
exit /b %errorlevel%
