# anchor_ppl.py — PPL 逐位锚采集器（迁移验收的等价性判据）
#
# 口径（迁移计划 v2 §1）：
#   - 固定 prompt 集 = _spec_eval_prompts 标准族（code/doc/reason × 4k/16k/48k，8 条）
#   - PPL 在**裸 message content**上计算（不经 chat template），隔离权重/量化路径数值，
#     跨栈稳定；模板渲染漂移由 prompt sha256 + tokenizer sha256 记录捕获
#   - mean_nll = -sum(logprob)/n_tokens（prompt_logprobs=0 取实际 token 的 logprob）
#   - eager 模式 + 默认 KV dtype，记录进 meta；新栈复跑必须同口径后才可比
#
# 用法（现役栈 = vllm-win venv + overlay）：
#   G:\qwen3.8model\vllm-win\Scripts\python.exe tools\anchor_ppl.py ^
#     --out G:\qwen3.8model\vllm-030win-锚点数据包\ppl_anchor_overlay0271.json
# 新栈复跑同命令换 venv，输出名换新栈标识；比对用 tools\compare_ppl.py。

import argparse, hashlib, json, os, sys, time

PROMPT_DIR = r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\_spec_eval_prompts"
NAMES = ["code_4k", "doc_4k", "reason_4k", "code_16k",
         "doc_16k", "reason_16k", "code_48k", "doc_48k"]


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=57344)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    t0 = time.time()
    from vllm import LLM, SamplingParams
    import vllm, torch

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True,
              disable_log_stats=True)
    tok = llm.get_tokenizer()

    meta = {
        "stack": os.environ.get("ANCHOR_STACK", "unspecified"),
        "vllm_version": getattr(vllm, "__version__", "?"),
        "torch_version": torch.__version__,
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_mem": args.gpu_mem,
        "enforce_eager": True,
        "kv_dtype": "default",
        "tokenizer_sha256": sha256(tok.backend_tokenizer.to_str()
                                   if hasattr(tok, "backend_tokenizer") else str(type(tok))),
        "prompt_scheme": "raw message content (no chat template), _spec_eval_prompts",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
    results = {}
    for name in NAMES:
        path = os.path.join(PROMPT_DIR, name + ".json")
        msgs = json.load(open(path, encoding="utf-8"))["messages"]
        text = "\n".join(m["content"] for m in msgs if m.get("content"))
        out = llm.generate(text, sp)[0]
        plp, ptids = out.prompt_logprobs, out.prompt_token_ids
        sum_lp, n = 0.0, 0
        for tid, d in zip(ptids, plp):
            if d is None:
                continue
            sum_lp += d[tid].logprob
            n += 1
        mean_nll = -sum_lp / n
        results[name] = {
            "prompt_sha256": sha256(text),
            "n_tokens": n,
            "sum_logprob": round(sum_lp, 6),
            "mean_nll": round(mean_nll, 6),
            "ppl": round(2.718281828459045 ** mean_nll, 6),
        }
        print(f"[{name}] n={n} mean_nll={results[name]['mean_nll']}", flush=True)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    doc = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
