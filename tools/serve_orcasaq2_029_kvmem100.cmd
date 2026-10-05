@echo off
rem ===========================================================================
rem STEP 100 ARM (KVMem generalization final acceptance): the Orca O1 nvfp4 arm
rem with the full step-083 KVMem env suite on top, byte-for-byte the load1456
rem semantics on the qwen3.8exl3 checkpoint.
rem
rem Why: the generalization line (094-098) claims "a new sliding-window hybrid
rem model needs zero hand-written numbers". qwen3.8exl3 reuses
rem Qwen3NextAttention (qwen3_5.py), so the hook registry, the codec registry
rem and the derived-config chain all apply without model edits. This arm is
rem the acceptance leg: boot + mechanism self-proof + needle (design doc
rem docs/vllm-030win-调研-KVMem泛用化.md section 3, step 100).
rem
rem Capacity ledger (rule 32/34, derived not copied blindly): same text_config
rem as GSQ => same engine page geometry (attention block 1456 under the five
rem production keys, measured 089 c1/c2) and same mamba state bytes (rule 27
rem snapshot region). Pool 3.4e9 + L=200,704 = the load1456 exit-arm values
rem the 083 protocol was accepted with; Orca weights are 0.38 GiB smaller.
rem No spec (086: nvfp4 + spec>=2 stalls at >=10K = standing NO-GO), no CUDA
rem graphs (O1 baseline is enforce-eager; graph compatibility is O4 scope).
rem
rem Injectables: S100_TAG (dump dir, MUST be fresh per boot, iron rule 16-viii),
rem S100_LOAD (1 = assembly line on, 0 = full-prefill control; the assembly
rem probe's compare needs both), S100_ML / S100_KVMB / S100_MBT / S100_WSMB /
rem S100_EVERY / S100_KEEP (snapshot knobs, 12/10 = the 083 mitigation).
rem
rem Judgement: engine log "KVMem assembly: request ... matches" -> "issued"
rem -> "prefix landed"; remat self-test gates (pages=8, delta/step <= 0.25);
rem needle HIT at depth 0.55; snapshot-region banner recorded (rule 27);
rem recency-only boot (S100_PAGES=1) expected MISS. Zero ERROR.
rem
rem KVMem ARM (NOT production). Design authority:
rem docs/vllm-030win-调研-KVMem虚拟化KV工作区.md sections 12.12/12.13.
rem Probe: tools/kvmem_assembly_probe.py (ingest -> flush -> serve + tail).
rem ===========================================================================
if "%S100_TAG%"=="" set "S100_TAG=kvmem_k1a"
if "%S100_LOAD%"=="" set "S100_LOAD=1"
if "%S100_ML%"=="" set "S100_ML=131072"
if "%S100_KVMB%"=="" set "S100_KVMB=3100000000"
if "%S100_MBT%"=="" set "S100_MBT=1456"
if "%S100_WSMB%"=="" set "S100_WSMB=5120"
if "%S100_EVERY%"=="" set "S100_EVERY=12"
if "%S100_KEEP%"=="" set "S100_KEEP=10"
if "%S100_PAGES%"=="" set "S100_PAGES=40"
if "%S100_VRECENT%"=="" set "S100_VRECENT=16384"
if "%S100_RECENT%"=="" set "S100_RECENT=16384"
set "HOME=C:\fi"
set "USERPROFILE=C:\fi"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "OMP_NUM_THREADS=8"
set "VLLM_LOGGING_LEVEL=INFO"
set "VLLM_SERVER_DEV_MODE=1"
set "TMP=G:\qwen3.8model\_tmp_orcasaq2"
set "TEMP=G:\qwen3.8model\_tmp_orcasaq2"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\orcasaq2_sitecustomize;G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_KV_GROUP_SIZE=8"
set "VLLM_KVMEM_SW_WINDOW=65536"
set "VLLM_KVMEM_WORKSPACE=1"
set "VLLM_KVMEM_WORKSPACE_MB=%S100_WSMB%"
set "VLLM_KVMEM_SELFTEST=1"
set "VLLM_KVMEM_RAWK=1"
set "VLLM_KVMEM_AUTHORITY=1"
set "VLLM_KVMEM_AUTHORITY_TRAJ=2"
set "VLLM_KVMEM_SWEEP=1"
set "VLLM_KVMEM_LOAD=%S100_LOAD%"
set "VLLM_KVMEM_VIEWPORT=0"
set "VLLM_KVMEM_VIEWPORT_PAGES=%S100_PAGES%"
set "VLLM_KVMEM_VIEWPORT_RECENT=%S100_VRECENT%"
set "VLLM_KVMEM_RECENT=%S100_RECENT%"
set "VLLM_KVMEM_TOPN=64"
set "VLLM_KVMEM_SNAPSHOT_EVERY_PAGES=%S100_EVERY%"
set "VLLM_KVMEM_SNAPSHOT_KEEP=%S100_KEEP%"
set "VLLM_KVMEM_DEBUG=1"
set "VLLM_KVMEM_BAKE_VERIFY=1"
set "VLLM_KVMEM_INDEX_SUBBLOCK=32"
set "VLLM_KVMEM_SCORE_GRANULARITIES=32,64,128"
set "VLLM_KVMEM_SCORE_MODES=dot,cosine"
set "VLLM_KVMEM_TIMING=1"
set "VLLM_KVMEM_TIMING_EVERY=25"
set "VLLM_KVMEM_SLOT_PICK=time"
set "VLLM_KVMEM_DUMP=G:\qwen3.8model\prod029_logs\%S100_TAG%"
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist "G:\qwen3.8model\_tmp_orcasaq2" mkdir "G:\qwen3.8model\_tmp_orcasaq2"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
del /q "G:\qwen3.8model\_tmp_orcasaq2\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\qwen3.8exl3" ^
  --served-model-name orcasaq2 ^
  --host 127.0.0.1 --port 8001 ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes %S100_KVMB% ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len %S100_ML% ^
  --max-num-seqs 1 --max-num-batched-tokens %S100_MBT% ^
  --enable-prefix-caching ^
  --enforce-eager ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --dtype auto
exit /b %errorlevel%
