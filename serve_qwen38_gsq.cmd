@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "PATH=%ROOT%Scripts;%PATH%"

"%ROOT%Scripts\vllm.exe" serve "%MODEL%" ^
  --host 127.0.0.1 ^
  --port 8000 ^
  --dtype auto ^
  --gpu-memory-utilization 0.88 ^
  --max-model-len 8192 ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --language-model-only

endlocal
