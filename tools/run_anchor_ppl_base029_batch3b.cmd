@echo off
rem Anchor PPL runner on the PURE 0.29 base (no overlay) -- S1 smoke + PPL (bf16 KV). Batch-3b rerun (custom-op boundary check, eager equivalence; new filename per anchor-overwrite rule).
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
"G:\qwen3.8model\vllm-win029\Scripts\python.exe" "G:\qwen3.8model\vllm-030win-git\tools\anchor_ppl.py" --kv-dtype auto --max-model-len 52288 --out "G:\qwen3.8model\vllm-030win-锚点数据包\ppl_anchor_base029_batch3b.json"
