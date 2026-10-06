@echo off
setlocal
rem ===========================================================================
rem OrcaSPEC-2 PRODUCTION-FORM launcher (step 105, 2026-10-06): the 090 DFlash2
rem speed recipe FROZEN as Orca's speculation production form. User ruling closed
rem decision item (a) of the O3 wrap-up ("freeze the 090 recipe as Orca's
rem speculation production form"); step 093 had already cleared every debt that
rem ruling listed as outstanding (same-gauge comparison, pool-speed knee,
rem soak/multi-turn). NOTHING in the serve line below is new: it is byte-for-byte
rem what tools/serve_orcasaq2_029_nvfp4_dflash2_prodcap.cmd expands to under the
rem env that 090/092/093 measured healthy:
rem   S089_L=16384 S089_MBT=1024 S089_UTIL=0.922 S089_SPEC=2 S089_OFFLOAD=8
rem   S089_NOEAGER=1 S089_CGS=3 S089_GRAPHPROF=0 S089_POOL=800000000
rem Evidence chain for every line (all on THIS 16 GB card, THIS draft gptq3c N=2):
rem   --kv-cache-dtype nvfp4 ................ 084 (O1 GO; LBHNC default-layout fix)
rem   draft dflash2\gptq3c N=2 .............. 088 alignment GO (prose acc 0.5448 /
rem                                            countdown 0.9932), user-approved reuse
rem   VLLM_KV_GROUP_SIZE=8 .................. 036/089 (capacity lever)
rem   --mamba-ssm-cache-dtype bfloat16 ...... 045/046/089 (block 2832 -> 1456)
rem   --mamba-cache-mode align .............. 089 (as production)
rem   --enable-prefix-caching ............... 089 (as production; 093 soak: 8k 2nd-shot
rem                                            TTFT 2.21 -> 1.35 s)
rem   --kv-offloading-backend native 8 ...... 089 (KV spills to host mmap, production form)
rem   --gpu-memory-utilization 0.922 ........ 089 (production's own value)
rem   FULL graph (no enforce-eager) + capture 3 ... 090 (graph tier healthy; eager is the
rem                                            4x handicap, iron rule 8)
rem   VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 ... 090 s1 (default reservation ~1.3 GiB
rem                                            starves the pool; real capture peak fits)
rem   --kv-cache-memory-bytes 800000000 ..... 090/093: THE speed switch (iron rule 33).
rem                                            Auto pool self-sizes to 43,480 tokens and
rem                                            drops into the residency slow mode
rem                                            (4.83/5.49 tok/s); 8e8 (18,589 tokens)
rem                                            sits in the 78-87 health band.
rem Speed evidence, 8k anchor median (>=3 boots, iron rule 7): 090 s5/s6/s7 =
rem 86.63/86.01/79.91; 092 generic-launcher = 85.66/83.43/74.38; 093 d1/d2/d3 =
rem 82.3/78.16/89.83 -- health band 74.38-89.83 across 8 boots, needle every boot,
rem zero ERROR lines. Same-gauge comparison (093-A): DFlash2 ~82 > MTP graph ~70.5
rem > MTP eager 37.6 -- graph-vs-graph DFlash2 leads 1.14x; 086's "2.2x" was a
rem graph-vs-eager mixed gauge. Long stability (093-C): soak 2 rounds x 19 all
rem healthy, 16k pressure tier (pool at 85%) clean, multi-turn 23/24.
rem
rem NOT part of this freeze (recorded debts, do not silently add):
rem   * Pool up-tier 1.4e9 (32,768 tokens): single-boot 87.2 at L=16,384 (093-B1
rem     p1) -- the fastest point of the sweep, but one boot short of the >=3 gate
rem     and only 0.1e9 above the cliff onset (1.5e9 still retains ~80%, 1.6e9
rem     collapses to 19%). Promoting it requires >=3 boots + soak re-run.
rem   * Pool-speed cliff for the MTP-graph tier (only 1.0e9 measured), 093 debt.
rem   * VLLM_KV_GROUP_SIZE=8 sensitivity (card still marks it human-copied).
rem   * tool-call segment: GSQ production carries --enable-auto-tool-choice +
rem     qwen3_coder/qwen3 parsers; NO Orca arm has ever run them, and this
rem     freeze must not add unverified variables. Debt of soak_orca's tool-call
rem     segment.
rem   * Multi-concurrency: --max-num-seqs 1, like production. 090 measured the
rem     8e8 pool cost as concurrency 2.65x -> 1.13x; do NOT copy this pool value
rem     into a multi-concurrent deployment without re-measuring.
rem
rem This file does NOT replace GSQ production (tools/serve_gsq_prod029_n2.cmd,
rem port 8080): delivery form is coexisting variant launchers, production default
rem untouched. Orca production form lives here, port 8001. Never run from the
rem record-repo directory (cwd would capture the 0.27.1 tree lying there).
rem ===========================================================================
if not exist C:\fi mkdir C:\fi
if not exist G:\qwen3.8model\_tmp_orcasaq2 mkdir G:\qwen3.8model\_tmp_orcasaq2
set "VENV=G:\qwen3.8model\vllm-win029"
set "MODEL=G:\qwen3.8model\qwen3.8exl3"
set "DRAFT=G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c"
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
set "VLLM_KV_GROUP_SIZE=8"
set "ORCA_EXL3_ALLOW_EMPTY_SHARED=1"
rem 090 s1: the default graph-profiling reservation (~1.3 GiB) starves the KV
rem pool on this card; the real capture peak is measured to fit (s2-s7 boots).
set "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
rem Reclaim stale offload mmaps the way production does (names are per-engine unique).
del /q "G:\qwen3.8model\_tmp_orcasaq2\vllm_offload_*.mmap" 2>nul
cd /d G:\qwen3.8model
"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\qwen3.8exl3" ^
  --served-model-name orcasaq2 ^
  --host 127.0.0.1 --port 8001 ^
  --max-model-len 16384 ^
  --gpu-memory-utilization 0.922 ^
  --max-num-batched-tokens 1024 ^
  --max-num-seqs 1 ^
  --dtype auto ^
  --kv-cache-dtype nvfp4 ^
  --cudagraph-capture-sizes 3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --enable-prefix-caching ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --kv-cache-memory-bytes 800000000 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2
exit /b %errorlevel%
