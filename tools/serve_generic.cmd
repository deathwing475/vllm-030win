@echo off
rem =============================================================================
rem serve_generic.cmd (step 092) - the ONE generic launcher.
rem
rem Human-intent parameters ONLY (design doc: vllm-030win-调研-新模型全自动适配.md §3):
rem   SGEN_MODEL   card name under profile\ OR a checkpoint directory
rem                (no card yet -> one is built on the spot, zero hand numbers)
rem   SGEN_L       max-model-len          (default: card's single-request ceiling)
rem   SGEN_TIER    speed | capacity | auto (default: speed = hand-pinned pool)
rem   SGEN_FAMILY  dflash2 | none          (default: dflash2)
rem   SGEN_SPEC    num_speculative_tokens  (default: card's draft_slots)
rem   SGEN_POOL    explicit --kv-cache-memory-bytes override
rem   SGEN_PORT    serve port              (default 8001)
rem   SGEN_OUT     where the generated cmd goes (default _tmp_line_b\sgen_launch.cmd)
rem
rem Everything else is derived from profile\<model>.json + platform_card.json.
rem The generator prints GUARD WARNINGS for every human-copied value it uses.
rem
rem Usage:
rem   set "SGEN_MODEL=qwen3.8exl3" && set "SGEN_L=16384" && tools\serve_generic.cmd
rem =============================================================================
setlocal
if "%SGEN_MODEL%"=="" (
  echo [serve_generic] SGEN_MODEL is required ^(card name or checkpoint path^)
  exit /b 1
)
set "PY=G:\qwen3.8model\vllm-win029\Scripts\python.exe"
set "GEN=G:\qwen3.8model\vllm-030win-git\tools\profile_card\gen_launch.py"
if "%SGEN_OUT%"=="" set "SGEN_OUT=G:\qwen3.8model\_tmp_line_b\sgen_launch.cmd"

set "GENARGS=--model %SGEN_MODEL% --out "%SGEN_OUT%""
if not "%SGEN_L%"==""     set "GENARGS=%GENARGS% --l %SGEN_L%"
if not "%SGEN_TIER%"==""  set "GENARGS=%GENARGS% --tier %SGEN_TIER%"
if not "%SGEN_FAMILY%"=="" set "GENARGS=%GENARGS% --family %SGEN_FAMILY%"
if not "%SGEN_SPEC%"==""  set "GENARGS=%GENARGS% --spec %SGEN_SPEC%"
if not "%SGEN_POOL%"==""  set "GENARGS=%GENARGS% --pool %SGEN_POOL%"
if not "%SGEN_PORT%"==""  set "GENARGS=%GENARGS% --port %SGEN_PORT%"

"%PY%" "%GEN%" %GENARGS%
if errorlevel 1 (
  echo [serve_generic] generation failed
  exit /b 1
)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "%SGEN_OUT%"
exit /b %errorlevel%
