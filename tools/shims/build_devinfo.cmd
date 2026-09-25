@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
"C:\Program Files\LLVM\bin\clang++" -O3 -std=c++17 -shared "G:\qwen3.8model\vllm-win029\Lib\site-packages\humming\csrc\device_info.cpp" -I"C:\Users\deathwing475\AppData\Local\Programs\Python\Python312\Include" -I"C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\include" -I"C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\include\cccl" -L"C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64" -L"C:\Users\deathwing475\AppData\Local\Programs\Python\Python312\libs" -lcuda -o "C:\fi\.humming\cache\device_info\3a3f1e33f47a4caa\_device_info.abi3.so"
echo BUILD_RC=%ERRORLEVEL%
