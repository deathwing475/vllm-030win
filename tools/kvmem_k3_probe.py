# kvmem_k3_probe.py — retrieval-quality probe for the KVMem line (step 061).
#
# The step 060 workspace could store and reload pages but never asked whether a
# page is worth reloading. This probe answers the design's risk R1 directly:
# at the 1424-token page granularity the block size forces on us, does the
# Mean-K index actually rank the page that holds the answer near the top?
#
# It is deliberately a *measurement* tool, not an acceptance gate: the engine
# dumps a policy-free artifact (per-page logits + the eligible set + the top-N
# it would use) and the probe compares that against the needle's real page,
# which only the client knows.
#
#   needle   one prompt with a needle at a given depth; report the needle
#            page's rank among the eligible pages, and whether the engine's
#            top-N contains it. A needle shallower than the eviction boundary
#            (prompt_tokens - sliding window) is one the window no longer
#            holds, so retrieval is the only way back to it.
#   control  the same prompt with no needle: the noise floor. A ranking tool
#            that puts a random page on top is not usable no matter what the
#            needle run says.
#
# Usage:
#   python tools\kvmem_k3_probe.py needle  --tokens 200000 --needle-depth 0.10 --out k3_needle.json
#   python tools\kvmem_k3_probe.py needle  --tokens 200000 --needle-depth 0.50 --out k3_mid.json
#   python tools\kvmem_k3_probe.py control --tokens 200000 --out k3_control.json
#
# Requires the K3 arm: VLLM_KVMEM_WORKSPACE=1 + VLLM_KVMEM_SW_WINDOW=163072 +
# VLLM_KVMEM_RAWK=1 + VLLM_KVMEM_DUMP=<dir>. The reports are read back from
# that dump directory.

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvmem_ws_probe import NEEDLE, NEEDLE_CODE, QUESTION, post  # noqa: E402

MODEL_DIR = r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ"
UNIT_JSONS = [
    os.path.join(MODEL_DIR, "_spec_eval_prompts", name)
    for name in ("doc_4k.json", "code_4k.json", "reason_4k.json")
]
UNIT_TOKENS = 4070
SLIDING_WINDOW = 163072

_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _TOKENIZER


def build(target_tokens: int, depth: float, nonce: str, with_needle: bool,
          question_style: str = "summary", question_repeat: int = 1):
    """Haystack of spec-eval units; returns (prompt, needle_char_offset).

    ``question_style`` decides what the query span actually contains. The
    engine scores the trailing ``VLLM_KVMEM_QUERY_SPAN`` (256) tokens, so with
    the default summary question most of the query is document text and a
    generic "write a detailed summary" instruction - the question itself is
    ~8% of it. The focused style repeats a short question so the query span is
    dominated by the question, which separates "the query was too diluted"
    from "the index has no signal".
    """
    units = [
        json.load(open(path, encoding="utf-8"))["messages"][0]["content"]
        for path in UNIT_JSONS
    ]
    reps = max(1, round(target_tokens / UNIT_TOKENS))
    needle_at = max(1, int(reps * depth))
    parts = [f"[SESSION {nonce}]\n"]
    needle_char = None
    for i in range(reps):
        parts.append(units[i % len(units)])
        if with_needle and i == needle_at - 1:
            needle_char = len("".join(parts))
            parts.append(NEEDLE)
    if question_style == "focused":
        parts.append(
            "\n\nQuestion: What is the secret access code? "
            "Answer with the number only.\nAnswer:" * max(1, question_repeat)
        )
    else:
        parts.append(QUESTION)
    return "".join(parts), needle_char


def needle_offset(prompt: str, needle_char: int) -> int:
    return len(tokenizer().encode(prompt[:needle_char], add_special_tokens=False))


def latest_report(dump_dir: str, since: float, num_tokens: int | None) -> dict | None:
    """Newest dump written after this request started.

    Matching on mtime rather than on the token count keeps the probe honest if
    the tokenizer the probe uses differs from the server's by a token or two.
    """
    best = None
    for path in glob.glob(os.path.join(dump_dir, "kvmem_retrieval_*.json")):
        stamp = os.path.getmtime(path)
        if stamp < since - 5:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if num_tokens is not None and abs(
            int(report.get("num_tokens", -1)) - num_tokens
        ) > 8:
            continue
        if best is None or stamp > best[0]:
            best = (stamp, path, report)
    return None if best is None else {**best[2], "path": best[1]}


def rank_of(report: dict, page: int) -> dict:
    """Rank the needle's page inside every (mode, granularity) variant.

    The engine reports all variants from one prefill; the probe is the only
    side that knows where the needle actually is, so this is where the
    retrieval question gets answered.
    """
    out = {"needle_page": page, "eligible": len(report["eligible"])}
    for name, variant in report.get("variants", {}).items():
        logits = variant["page_logits"]
        eligible = report["eligible"]
        scored = [
            (p, logits[p])
            for p in eligible
            if p < len(logits) and logits[p] is not None
        ]
        scored.sort(key=lambda item: -item[1])
        order = [p for p, _ in scored]
        out[name] = {
            "rank_by_logit": (order.index(page) + 1) if page in order else None,
            "in_top_pages": page in variant["top_pages"],
            "needle_logit": logits[page] if page < len(logits) else None,
            "best_logit": scored[0][1] if scored else None,
            "worst_logit": scored[-1][1] if scored else None,
            "needle_score": (
                variant["page_scores"][page]
                if page < len(variant["page_scores"])
                else None
            ),
            "kbar_norm": (
                variant["page_kbar_norm"][page]
                if page < len(variant["page_kbar_norm"])
                else None
            ),
            "top_pages": variant["top_pages"],
        }
    return out


def run(args, with_needle: bool) -> dict:
    prompt, needle_char = build(
        args.tokens,
        args.needle_depth,
        args.nonce,
        with_needle,
        args.question,
        args.question_repeat,
    )
    tokenized = len(tokenizer().encode(prompt, add_special_tokens=False))
    started = time.time()
    response = post(args.base, prompt, args.max_tokens)
    out = {
        "with_needle": with_needle,
        "needle_depth": args.needle_depth,
        "question_style": args.question,
        "question_repeat": args.question_repeat,
        "nonce": args.nonce,
        "tokenized_prompt_tokens": tokenized,
        "response": response,
    }
    if not response.get("ok"):
        return out

    prompt_tokens = response.get("prompt_tokens") or tokenized
    out["prompt_tokens"] = prompt_tokens
    out["prompt_tokens_delta_vs_tokenizer"] = (
        None if response.get("prompt_tokens") is None
        else response["prompt_tokens"] - tokenized
    )
    report = latest_report(args.dump_dir, started, response.get("prompt_tokens"))
    if report is None:
        out["error"] = (
            f"no kvmem_retrieval_*.json newer than the request in {args.dump_dir}; "
            "is VLLM_KVMEM_RAWK=1 and VLLM_KVMEM_DUMP set?"
        )
        return out

    block_size = report["block_size"]
    out["eviction_boundary_tokens"] = max(0, prompt_tokens - SLIDING_WINDOW)
    out["evicted_pages"] = max(0, prompt_tokens - SLIDING_WINDOW) // block_size
    out["report_summary"] = {
        key: report[key]
        for key in (
            "trajectory",
            "num_tokens",
            "block_size",
            "subblock",
            "score_modes",
            "granularities",
            "num_pages",
            "num_layers_scored",
            "query_span",
            "sink_tokens",
            "recent_tokens",
            "topn",
            "stored_subblocks",
        )
        if key in report
    }
    if with_needle:
        offset = needle_offset(prompt, needle_char)
        subblock = report.get("subblock") or 1
        # The engine attributes a sub-block to the page holding its first token,
        # and a page (1424) is not a whole number of sub-blocks (128 -> 11.125),
        # so a sub-block can straddle a boundary. Report both attributions and
        # let a disagreement be visible rather than silently judging the wrong
        # page.
        page = ((offset // subblock) * subblock) // block_size
        page_by_token = offset // block_size
        out["needle_token_offset"] = offset
        out["needle_offset_in_page"] = offset % block_size
        out["needle_page"] = page
        out["needle_page_by_token_offset"] = page_by_token
        out["needle_page_attributions_agree"] = page == page_by_token
        out["needle_is_evicted"] = offset < out["eviction_boundary_tokens"]
        out["rank"] = rank_of(report, page)
        if not out["needle_page_attributions_agree"]:
            out["rank_by_token_offset"] = rank_of(report, page_by_token)
    else:
        primary = next(iter(report.get("variants", {}).values()), None)
        if primary is None:
            out["error"] = "report has no variants"
            return out
        logits = sorted(
            (v for v in primary["page_logits"] if v is not None), reverse=True
        )
        out["noise_floor"] = {
            "top_logits": [round(v, 4) for v in logits[:8]],
            "median_logit": round(logits[len(logits) // 2], 4) if logits else None,
            "top_pages": primary["top_pages"],
            "top_scores": primary["top_scores"],
        }
    out["report_path"] = report["path"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["needle", "control"])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--needle-depth", type=float, default=0.10)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--nonce", default=None)
    ap.add_argument(
        "--question",
        choices=["summary", "focused"],
        default="summary",
        help="summary = the long summary request; focused = a short question "
        "repeated so the query span is mostly the question itself",
    )
    ap.add_argument("--question-repeat", type=int, default=8)
    ap.add_argument(
        "--dump-dir",
        default=os.environ.get("VLLM_KVMEM_DUMP", ""),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.nonce:
        import uuid

        args.nonce = uuid.uuid4().hex
    if not args.dump_dir:
        ap.error("--dump-dir (or VLLM_KVMEM_DUMP) is required")

    result = run(args, with_needle=args.mode == "needle")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2500], flush=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
