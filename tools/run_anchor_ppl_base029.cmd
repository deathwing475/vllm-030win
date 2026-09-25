@echo off
rem Anchor PPL runner on the PURE 0.29 base (no overlay) — S1/S2 smoke + PPL.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "TMP=G:\qwen3.8model\_tmp_anchors"
set "TEMP=G:\qwen3.8model\_tmp_anchors"
set "ANCHOR_STACK=base-0.29-whl"
"G:\qwen3.8model\vllm-win029\Scripts\python.exe" "G:\qwen3.8model\vllm-030win-git\tools\anchor_ppl.py" --out "G:\qwen3.8model\vllm-030win-锚点数据包\ppl_anchor_base029.json"
