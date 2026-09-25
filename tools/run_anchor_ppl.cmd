@echo off
rem Anchor PPL runner — must run inside VS env (cl) + production env contract.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "PYTHONPATH=G:\qwen3.8model\nvfp4-win-experiment\vllm-overlay"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "VLLM_HAS_FLASHINFER_CUBIN=1"
set "FLASHINFER_WORKSPACE_BASE=C:\fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "TMP=G:\qwen3.8model\_tmp_anchors"
set "TEMP=G:\qwen3.8model\_tmp_anchors"
set "ANCHOR_STACK=overlay-0.27.1-win"
"G:\qwen3.8model\vllm-win\Scripts\python.exe" "G:\qwen3.8model\vllm-030win-git\tools\anchor_ppl.py" --out "G:\qwen3.8model\vllm-030win-锚点数据包\ppl_anchor_overlay0271.json"
