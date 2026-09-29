@echo off
rem 2026-09-28 step 046 (mamba ssm bf16, PRODUCTION): adds
rem --mamba-ssm-cache-dtype bfloat16 and raises max-model-len 144432 ->
rem 163072. The GDN ssm state was fp32 (the model config.json declares
rem mamba_ssm_dtype=float32; step 045 corrected the old "already bf16"
rem note - only conv was bf16). bfloat16 is in the fused-kernel whitelist
rem FUSED_GDN_STATE_DTYPES (fp16 is NOT and would fall off the fused path).
rem Physics: mamba pool 594.75 -> 321.65 MiB (-273.1), which re-derives the
rem hybrid alignment: attn block 2832 -> 1456, unified page 1,677,312 B,
rem bytes_per_block(G=8) 26,099,712 -> 13,418,496, num_blocks 130 -> 253.
rem New G=8 ceiling: 2*cdiv(L,1456) + 28 <= 252 => L = 163,072; the engine
rem reports 163,719 tokens (+12.5% vs 145,551). Verified in step 045: A/B 6
rem boots same band (bf16 120.6-125.5 vs fp32 118.6-127.2), needle 18/18,
rem same-round PPL max|delta| 1.25e-3 (inside the boot-to-boot drift band),
rem 20-request soak healthy. WARNING (first-compile deep slump, 4/4): after
rem an L/dtype change the FIRST boot re-runs AOT compile (~145s longer) and
rem steady reads ~17 tok/s; the second boot (cache hit) recovers to 116-125.
rem Production switchover must double-boot: let boot #1 finish (it builds
rem the compile cache), kill it, boot #2 is the production instance;
rem tools/prod_watchdog.ps1 automates exactly this and its 8k probe
rem threshold 105 stays valid (8k band 116-125).
rem ROLLBACK = python tools\apply_prod_ssm_bf16_step046.py revert (deletes
rem this note + the ssm line, restores 144432). Chain order: revert 046
rem BEFORE tools\apply_prod_kvgroup_step036.py revert.
rem 2026-09-28 step 036 (KV regroup, PRODUCTION): VLLM_KV_GROUP_SIZE=8
rem forces the KV layers-per-group from the upstream 5 (pinned by the
rem draft's 5 sliding-window layers) to 8. Measured on this stack: the
rem 3.4e9 pool goes 114,974 -> 136,190 tokens (+18.5%), decode step
rem 18.73/18.74 -> 18.30/18.31/18.32 ms, prefill unchanged; needle was green
rem at 8k/32k/64k/100k (max-model-len 110000, step-036 era) and 8k/64k/130k.
rem max-model-len: 110000 -> 130000 -> 140000 (step 039) -> 144432 (step 040).
rem The G=8 hard ceiling is 144,432: the engine permanently holds back one
rem null block before the admission check, so the real constraint is
rem blocks_per_req = 2*cdiv(L,2832) + 27 <= num_blocks - 1 = 129, i.e.
rem cdiv(L,2832) <= 51. Step 039 read the 21% decode drop at 144,432 as an
rem L effect and stopped at 140,000; step 040 re-measured with alternating
rem boots and showed that drop is a random slow boot, not an L effect (same
rem L=144,432: steady 118-121 at fb 15,538 MiB vs 96-99 at 15,254 MiB;
rem pin placement byte-identical across 15 boots). Multi-depth needle at
rem 144,432 matches 140,000 depth for depth (8k 120.2/120.4, 64k 121.0/116.4,
rem 115k 113.0/108.0, 135k 102.3/103.2). Engine reports 145,551 tokens.
rem Step 041 re-checked against the user's two gates (decode >=85; >=70 when
rem using 90% of max-model-len) and confirmed 144,432 is the largest value
rem that satisfies both: G=1@147,264 breaks 85 on a slow boot, N=1@154,880
rem breaks 70 at its 86.7% depth, N=0@164,256 breaks both.
rem CAVEAT: at 144,432 a full-length request claims all 129 usable blocks,
rem so a max-length request leaves no headroom for prefix-cache growth.
rem Pool stays the manual 3,400,000,000 B: KV tensor size and spill are
rem unchanged; only grouping and the logical ceiling move.
rem ROLLBACK = delete the VLLM_KV_GROUP_SIZE line, restore 110000, drop
rem this note (or: python tools\apply_prod_kvgroup_step036.py revert).
rem ===========================================================================
rem PRODUCTION launcher, vLLM 0.29 stack (stage-4 switchover, 2026-09-26).
rem Deployment twin of the regression-verified tools/serve_gsq_base029_lineB_spec.cmd
rem (line-B spec e2e PASSED: A2 73.00 / A5 68.59-24.39 / A4 acc 0.72-0.80),
rem plus the production-only KV feature kept from the 0.27 recipe
rem run_dflash2_n2.cmd (N=2 PRODUCTION, 2026-09-25 user decision):
rem   + --enable-prefix-caching
rem KV offloading is ON (user priority: multi-turn conversations rely on the
rem offload tier to keep earlier turns in CPU RAM). It crashed the stage-4
rem drill (2026-09-26) and was root-caused the same day to a cuMemcpyBatchAsync
rem driver defect on non-default streams; fixed in base d7cdb91 by routing
rem swap_blocks_batch to per-copy cuMemcpyAsync on Windows. Verified: 8k x2
rem needle green with repeat-request TTFT 4.7s -> 1.5s (prefix hit).
rem KV pool stays MANUAL (iron rule): 3,400,000,000 B -> 145,551 tokens with
rem draft at G=8 / max-model-len 144,432 (the 121,058 figure was the pre-036
rem mamba-none accounting).
rem
rem Delta vs the frozen 0.27 recipe (all intentional):
rem   1. venv G:\qwen3.8model\vllm-win029 (0.29 base + batch-1..5 + regression
rem      fixes 6fc2108/e9d534d); PYTHONPATH overlay GONE.
rem   2. spec via 0.29 native --speculative-config.* keys (argv-probe verified;
rem      replaces 0.27 custom --spec-method/--spec-model/--spec-tokens).
rem   3. flashinfer 0.6.18 runner contract: FLASHINFER_WORKSPACE_BASE=C:/fw
rem      (FORWARD slashes -- 0.6.18 eats "\f" as form-feed) + tvm_ffi LDFLAGS
rem      pointing at vllm-win029. VLLM_HAS_FLASHINFER_CUBIN dropped (0.27-era
rem      flag, not part of the validated 0.29 contract).
rem   4. TMP/TEMP -> G:\qwen3.8model\_tmp_prod029 (offload mmap + JIT build
rem      temps stay off C:).
rem
rem ROLLBACK: the frozen 0.27 recipe run_dflash2_n2.cmd + venv
rem G:\qwen3.8model\vllm-win (read-only) remain untouched -- see
rem docs/切换与回退预案.md. Never run this from the record-repo directory
rem (cwd would capture the 0.27.1 tree lying there).
rem
rem 2026-09-27 graph-mode swap (step 027): the --compilation-config
rem {"cudagraph_mode":"PIECEWISE"} line is REMOVED, so the engine default
rem FULL_AND_PIECEWISE applies (prefill PIECEWISE / decode FULL whole-graph
rem replay). Explicit PIECEWISE disables FULL graphs and eagerly dispatches
rem 90+ splitting ops per decode step; the FULL decode graph is verified
rem safe with DFlash2 spec (three-segment capture passes, step 026) and was
rem measured 25.86-26.45ms vs 27.0ms over 3 boots (step 026, -4%).
rem --cudagraph-capture-sizes 3 kept. Long-stability soak + correctness
rem gates (step 027) precede this file becoming the live production config.
rem Rollback of this swap = re-add the compilation-config line above.
rem 2026-09-29 step 054 (vision CPU speed; user directive: image tok speed is
rem the goal, RAM does not matter): set OMP_NUM_THREADS=8. vLLM's worker pins
rem torch intra-op threads to 1 after warmup (gpu_worker.py
rem set_torch_threads_for_runtime; rationale = OMP spin-wait stealing cycles
rem from the GPU serving loop), which throttled the CPU vision tower to
rem ~170 GFLOP/s effective. An EXTERNAL OMP_NUM_THREADS is respected by the
rem engine (it logs a warning and keeps the setting). Measured on 9800X3D
rem (AVX-512_BF16): GEMM 604 -> 2907 GFLOP/s (4.8x), SDPA 272 -> 44 ms
rem (6.2x); the 1-thread cost model (375 ms/layer x 27 layers = 10.1 s)
rem matches the observed ~13 s per 512-token image. int8 measured NO-GO for
rem speed on this CPU (bf16 F.linear 5.9 ms vs _weight_int8pack_mm 690 ms /
rem quantized W8A8 11547 ms; bf16 is already the fastest dense GEMM dtype on
rem Zen4 -- int8 saves RAM only). Remaining bottleneck after this fix = CPU
rem SDPA (~44 ms/layer; flash-CPU kernel already the torch default, no faster
rem backend exists; a custom attention kernel is the next lever).
rem 2026-09-29 step 053 (VISION CPU arm, EXPERIMENT - not production default):
rem user directive: keep the vision tower in system RAM and run it on CPU,
rem and bound per-image tokens (min/max) so image turns are not slow.
rem Delta vs the PRODUCTION bat (serve_gsq_prod029_n2.cmd, everything else
rem byte-identical):
rem   1. --language-model-only REMOVED (vision enabled).
rem   2. VLLM_DBG_VIS_CPU=1 -> pin_shim moves model.visual (460.7M params,
rem      0.86 GiB BF16, unquantized) to CPU after load_model + empty_cache,
rem      forces the native pos-embed interpolation (triton kernel cannot run
rem      on CPU tensors), forces ApplyRotaryEmb.forward_native, pins
rem      MMEncoderAttention to TORCH_SDPA, and routes encoder outputs back to
rem      GPU before embed_input_ids (bool-mask merge requires same device).
rem      Tower device is the patch_embed weight device (property), so
rem      .to("cpu") flips every device-dependent path.
rem   3. --mm-encoder-attn-backend TORCH_SDPA: official knob; SDPA path is
rem      pure torch (varlen split + F.scaled_dot_product_attention) and runs
rem      on CPU. FLASH_ATTN/FLASHINFER/TRITON ViT backends are CUDA-bound.
rem   4. --mm-processor-kwargs: per-image token bounds. tokens = px/1024
rem      (patch16 x merge2). Default was 64..16384 tokens/image; now 64..512
rem      (max_pixels 524288 = a 2048x2048 photo lands at 484 tokens; measured
rem      with the HF processor, transformers 5.17, per-call kwargs work and
rem      vLLM's ProcessingInfo + encoder budget read the same keys).
rem   Vision tower cost: ~0.86 GiB VRAM NOT used (headroom cliff hedge);
rem   image encode runs on CPU (~5-10 s per 512-token image class, bounded by
rem   the cap). Text-only behavior is unchanged.
rem ROLLBACK = use the production bat; nothing else changed.
rem ===========================================================================
if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
if not exist G:\qwen3.8model\_tmp_prod029 mkdir G:\qwen3.8model\_tmp_prod029
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "HF_HOME=G:\qwen3.8model\hub"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CXX=C:\Program Files\LLVM\bin\clang++"
set "VLLM_KV_CACHE_LAYOUT=HND"
set "FLASHINFER_WORKSPACE_BASE=C:/fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "VLLM_LOGGING_LEVEL=INFO"
set "TMP=G:\qwen3.8model\_tmp_prod029"
set "TEMP=G:\qwen3.8model\_tmp_prod029"
set "PATH=G:\qwen3.8model\vllm-win029\Scripts;%PATH%"
rem Fast-tier pin (step 030, 2026-09-27): inject
rem tools\pin_shim\sitecustomize.py via PYTHONPATH. MIN mode hooks only
rem load_dflash_model and pins the draft weights (gptq3c, 31 Humming
rem layers) into dedicated VRAM after their free=0 pressure-window
rem placement. Verified 21/21 boots fast-tier 20.3-21.6ms (clean boots
rem draw 20% fast / 80% mid 21-27ms) with acc 0.66-0.75 + needle green.
rem ROLLBACK = remove the 4 set lines below.
set "PYTHONPATH=G:\qwen3.8model\vllm-030win-git\tools\pin_shim"
set "VLLM_DBG_TRACE=1"
set "VLLM_DBG_MIN=1"
set "VLLM_DBG_PIN=1"
set "VLLM_DBG_VIS_CPU=1"
set "OMP_NUM_THREADS=8"
set "VLLM_KV_GROUP_SIZE=8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "LIB=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64;%LIB%"
rem Sweep stale offload mmaps (crash/held-ref exits leave them; names are
rem per-engine unique so they never collide -- just reclaim the space).
del /q "G:\qwen3.8model\_tmp_prod029\vllm_offload_*.mmap" 2>nul
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 29550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

"G:\qwen3.8model\vllm-win029\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ" ^
  --served-model-name qwen3.8-27b-gsq ^
  --host 127.0.0.1 --port 8080 ^
  --mm-encoder-attn-backend TORCH_SDPA ^
  --mm-processor-kwargs "{\"min_pixels\":65536,\"max_pixels\":524288}" ^
  --kv-cache-dtype nvfp4 ^
  --kv-cache-memory-bytes 3400000000 ^
  --gpu-memory-utilization 0.922 ^
  --max-model-len 163072 ^
  --max-num-seqs 1 --max-num-batched-tokens 1024 ^
  --enable-prefix-caching ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --reasoning-parser qwen3 ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype bfloat16 ^
  --kv-offloading-backend native ^
  --kv-offloading-size 8 ^
  --cudagraph-capture-sizes 3 ^
  --speculative-config.method dflash ^
  --speculative-config.model "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\dflash2\gptq3c" ^
  --speculative-config.num_speculative_tokens 2 ^
  --dtype auto
exit /b %errorlevel%
