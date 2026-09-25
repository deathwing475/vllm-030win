@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\Users\deathwing475\AppData\Local\Programs\Python\Python312\libs;%LIB%"
"G:\qwen3.8model\vllm-win029\Scripts\ninja.exe" -C "C:\fi\.humming\cache\launcher\torch211_stable_1f8991df268e25c8\7073e78b0c2cae92" -d explain
echo NINJA_RC=%ERRORLEVEL%
