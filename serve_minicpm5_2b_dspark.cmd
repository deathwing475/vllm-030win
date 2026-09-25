@echo off
rem ===========================================================================
rem serve_minicpm5_2b_dspark.cmd - MiniCPM5-2B + DSpark speculative decoding
rem Target: RTX 5070 Ti 16GB / native Windows / vLLM 0.27.1 / Python 3.13
rem Requirement: 20 concurrent, GPU memory utilization <= 0.922
rem
rem --- CONTEXT: 32768, not the model's 65536 --------------------------------
rem The KV pool floor scales linearly with max-model-len
rem   (max_model_len * 24064 B/token * 1.2), so halving the context halves the
rem pool: 1.76 GiB -> 0.88 GiB. That 0.88 GiB is what buys headroom for the
rem OCR / embedding models on this card. 32768 still covers any realistic
rem RAG prompt (system prompt + ~8 retrieved 256-token chunks is ~4k tokens).
rem NOTE this is a context ceiling on prompt+output combined, so it is NOT a
rem substitute for the max_new_tokens guard below: at 32768 a degenerate reply
rem could still reach ~15k tokens (~60k bytes), well past Elasticsearch's
rem 32766-byte keyword limit.
rem
rem --- KV BUDGET: 1.2x the configured context -------------------------------
rem max-model-len is 32768 and the KV pool is pinned to 1.2x its fp8 floor:
rem   32768 * 24064 B/token * 1.2 = 946,234,982 B (~0.88 GiB).
rem This leaves room for one full-context request plus ~20% KV headroom. It does
rem NOT support 20 simultaneous 32768-token requests; it is sized for the real
rem RAG/chunk workload (short prompts), not that pathological maximum.
rem The pool must still be >= max_model_len * 24064 B or vLLM refuses to boot.
rem A pool near exactly 1x can silently stall a request needing the last block;
rem 1.2x avoids that boundary while honoring the requested compression.
rem
rem --- CO-RESIDENCY: bge-m3 runs on this same card --------------------------
rem RAGFlow embeds chunks and calls this model in the same pipeline, so both
rem remain live. bge-m3 uses fp16 (see E:\docker_data\bge-m3\run_bge_m3_shared.ps1).
rem The smaller KV pool reserves substantially more dedicated memory for BGE-M3
rem and its batch activations. If shared GPU memory grows with load, lower the
rem KV pin again or run BGE-M3 on CPU.
rem
rem --- SPILL SAFETY ----------------------------------------------------------
rem Pinning avoids the auto-sizing failure mode where prefill spikes demote pages
rem to WDDM shared system memory and those pages are not reclaimed. Always judge
rem per-process shared memory and its growth, not adapter-wide shared usage.
rem
rem --- WHY fp8 KV, not nvfp4 -----------------------------------------------
rem nvfp4 KV on this build is a local SM120 overlay validated on the 27B
rem (head_size 256 -> 144B rows / 128B data carve). MiniCPM is head_size 128;
rem its 72B carve is not validated, so fp8 is the safe choice.
rem
rem --- SPILL BASELINE (measured, do not misread the shared-memory number) ----
rem A minimal process that only inits CUDA and allocates 64 MB already shows
rem 76 MiB of shared GPU memory on this box: that is the WDDM/CUDA baseline,
rem not spill. This config sits at 162 MiB (78 MiB above baseline) and does not
rem grow under load. Spill shows up as GROWTH with load (the 600 MiB failure
rem above). Judge on growth, not on the absolute number.
rem
rem --- WHY fp8 KV, not nvfp4 -------------------------------------------------
rem nvfp4 KV on this build is a local SM120 overlay patch validated on the 27B
rem (head_size 256 -> 144B rows / 128B data carve). This model is head_size
rem 128 -> 72B rows / 64B carve, an unvalidated layout, and a bad carve
rem silently corrupts KV instead of erroring. The 2B does not need the extra
rem compression. So: no overlay, no PYTHONPATH, stock path.
rem
rem --- OTHER CHOICES ---------------------------------------------------------
rem  * --dtype bfloat16 : weights are 4.69 GiB; memory is not the constraint
rem    here, quality is. Do not quantize.
rem  * --max-num-batched-tokens 10240 : DSpark consumes 6 batch slots per
rem    concurrent request (parallel_drafting -> num_spec_tokens-1 = 6; no +1
rem    because uses_draft_model() is False for dspark). vLLM sets
rem    max_num_scheduled_tokens = mbt - 6*max_num_seqs = 10240-192 = 10048,
rem    clearing its "<8192 suboptimal" warning. Do NOT reuse the 27B's mbt=1024.
rem  * --cudagraph-capture-sizes 1..32 : the 27B's "1 2" was sized for
rem    max-num-seqs 2. NOTE vLLM warns that FULL cudagraph is unsupported for
rem    spec-decode with the FlashInfer backend (UNIFORM_SINGLE_TOKEN_DECODE
rem    only) and falls back to PIECEWISE, so the draft step is not fully
rem    graphed. Switching --attention-backend to TRITON_ATTN may allow FULL;
rem    not yet tested.
rem  * Prefix caching left ON (default): plain llama model, no GDN align-state
rem    checkpoints, and a shared RAG system prompt is exactly its best case.
rem
rem --- DEFAULT OUTPUT CAP (do not remove) -----------------------------------
rem RAGFlow's OpenAI-compatible chat path deletes max_tokens from gen_conf
rem (rag/llm/chat_model.py), so its keyword/question extraction requests arrive
rem with no output limit and vLLM falls back to max_model_len = 65536. A 2B model
rem given RAGFlow's "Output: " continuation prompt occasionally degenerates into
rem an English self-repetition loop; one such chunk generated ~29k tokens, held
rem the KV pool at 96% while 197 requests queued, and then broke the whole
rem document when Elasticsearch rejected the >32766-byte question_kwd keyword
rem field. This default bounds every request that omits max_tokens; explicit
rem per-request values still win. 1024 is ~10x what keyword/question extraction
rem needs and still leaves room for a real RAG answer.
rem
rem The key MUST be "max_new_tokens", not "max_tokens": vLLM's
rem ModelConfig.get_diff_sampling_param only honors repetition_penalty,
rem temperature, top_k, top_p, min_p and max_new_tokens, renaming the last to
rem max_tokens. Passing "max_tokens" is silently dropped and changes nothing.
rem ===========================================================================
set "ROOT=G:\qwen3.8model"
set "MODEL=%ROOT%\MiniCPM5-2B"
set "DRAFT=%ROOT%\MiniCPM5-2B\dspark"

if not exist C:\fi mkdir C:\fi
if not exist C:\fw mkdir C:\fw
set "USERPROFILE=C:\fi"
set "HOME=C:\fi"
set "HF_HOME=%ROOT%\hub"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_OFFLINE=1"
set "CUDA_HOME=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
set "CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3"
rem Short JIT cache base: FlashInfer op dir names are long and Windows nvcc has
rem a 260-char GetLongPathName buffer (see research/native-win/01-build-log.md).
set "FLASHINFER_WORKSPACE_BASE=C:\fw"
set "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864"
rem MSVC links eagerly; tvm_ffi symbols must be resolvable. The whole path must
rem be FORWARD slashes: shlex.split eats the backslash in "-LG:\qwen..." and
rem yields the relative "-LG:qwen..." -> LNK1181 cannot open tvm_ffi.lib.
set "FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win/Lib/site-packages/tvm_ffi/lib -ltvm_ffi"
set "VLLM_LOGGING_LEVEL=INFO"
set "PATH=%ROOT%\vllm-win\Scripts;%PATH%"

rem MSVC environment for FlashInfer's runtime JIT builds
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

"%ROOT%\vllm-win\Scripts\python.exe" -m vllm.entrypoints.cli.main serve ^
  "%MODEL%" ^
  --served-model-name minicpm5-2b ^
  --host 0.0.0.0 --port 8081 ^
  --api-key sk-minicpm5 ^
  --gpu-memory-utilization 0.922 ^
  --kv-cache-memory-bytes 946234982 ^
  --dtype bfloat16 ^
  --kv-cache-dtype fp8 ^
  --max-model-len 32768 ^
  --max-num-seqs 20 ^
  --max-num-batched-tokens 10240 ^
  --cudagraph-capture-sizes 1 2 4 8 16 32 ^
  --speculative-config "{\"model\":\"G:/qwen3.8model/MiniCPM5-2B/dspark\",\"method\":\"dspark\",\"num_speculative_tokens\":7}" ^
  --override-generation-config "{\"max_new_tokens\": 1024}"
exit /b %errorlevel%
