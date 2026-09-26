@echo off
rem Anchor PPL runner on the PURE 0.29 base (no overlay) -- S1 smoke + PPL (bf16 KV). Batch-5 rerun: nvfp4 KV on SM120 (FA2-reader path landed; A1 口径, gpu_mem 0.922 per anchor protocol).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "TMP=G:\qwen3.8model\_tmp_anchors"
set "TEMP=G:\qwen3.8model\_tmp_anchors"
set "CXX=C:\Program Files\LLVM\bin\clang++"
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
set "ANCHOR_STACK=base-0.29-whl"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
"G:\qwen3.8model\vllm-win029\Scripts\python.exe" "G:\qwen3.8model\vllm-030win-git\tools\anchor_ppl.py" --kv-dtype nvfp4 --max-model-len 57344 --out "G:\qwen3.8model\vllm-030win-锚点数据包\ppl_anchor_base029_lineBfix_r5.json"
