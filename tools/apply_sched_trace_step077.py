# -*- coding: utf-8 -*-
"""步骤 077 诊断补丁：给调度器的准入分支加一次性计数日志（`VLLM_SCHED_TRACE=1` 门控）。

背景
----
076 把"页长 `--max-num-batched-tokens` × 投机 ⇒ 任何请求都不被调度"归因到唯一变量，
机制未定位。本补丁在 `v1/core/sched/scheduler.py` 的每条静默退出路径上打点，跑 peel E
配置即可读出"请求到底卡在哪一个 break"。

用法
----
    python tools/apply_sched_trace_step077.py apply            # 默认只打 venv
    python tools/apply_sched_trace_step077.py apply --target both
    python tools/apply_sched_trace_step077.py status
    python tools/apply_sched_trace_step077.py revert           # 必守第 14 条

门控
----
`VLLM_SCHED_TRACE` 不设时，`_sched_trace()` 首行即 return，唯一的行为差别是
`_mamba_block_aligned_split` 里多了两个局部变量赋值。补丁不改图/AOT 缓存键
（调度器是 host 代码，不进编译图）。
"""
import argparse
import os
import sys

REPO = r"G:\qwen3.8model\vllm-030win-git\vllm\v1\core\sched\scheduler.py"
VENV = (r"G:\qwen3.8model\vllm-win029\Lib\site-packages\vllm"
        r"\v1\core\sched\scheduler.py")
MARK = "vllm-030win step 077 diagnostic"

# (name, old, new) — 全部用 LF 书写，读写时按目标文件自身行尾转换。
EDITS = [
    (
        "import_os",
        "import itertools\nimport time\n",
        "import itertools\nimport os\nimport time\n",
    ),
    (
        "trace_helper",
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n"
        "\n"
        f"# {MARK}: count which admission branch a request dies in.\n"
        "# Gated by VLLM_SCHED_TRACE=1; with the gate off every call below\n"
        "# returns on the first statement. Revert with\n"
        "# tools/apply_sched_trace_step077.py revert\n"
        "_SCHED_TRACE = os.environ.get(\"VLLM_SCHED_TRACE\", \"0\") == \"1\"\n"
        "_SCHED_TRACE_COUNTS: dict[str, int] = {}\n"
        "\n"
        "\n"
        "def _sched_trace(reason: str, **ctx: Any) -> None:\n"
        "    if not _SCHED_TRACE:\n"
        "        return\n"
        "    count = _SCHED_TRACE_COUNTS.get(reason, 0) + 1\n"
        "    _SCHED_TRACE_COUNTS[reason] = count\n"
        "    if count == 1 or count % 100 == 0:\n"
        "        detail = \" \".join(f\"{k}={v}\" for k, v in ctx.items())\n"
        "        logger.info(\n"
        "            \"[SCHEDTRACE] reason=%s count=%d %s counts=%s\",\n"
        "            reason,\n"
        "            count,\n"
        "            detail,\n"
        "            _SCHED_TRACE_COUNTS,\n"
        "        )\n",
    ),
    (
        "loaded_banner",
        "        # In-flight requests still prefilling (prefill chunks + in-progress\n"
        "        # async KV loads). Their remaining-block reservation gates async loads.\n"
        "        self._inflight_prefills: set[Request] = set()\n",
        "        # In-flight requests still prefilling (prefill chunks + in-progress\n"
        "        # async KV loads). Their remaining-block reservation gates async loads.\n"
        "        self._inflight_prefills: set[Request] = set()\n"
        "\n"
        "        if _SCHED_TRACE:\n"
        "            spec = self.vllm_config.speculative_config\n"
        "            logger.info(\n"
        "                \"[SCHEDTRACE] patch loaded: mbt=%d "
        "max_num_scheduled_tokens=%s draft_slots=%s block_size=%d \"\n"
        "                \"num_spec_tokens=%d use_eagle=%s align_split=%s \"\n"
        "                \"long_prefill_threshold=%d max_model_len=%d "
        "num_lookahead=%d prefill_ckpt=%s\",\n"
        "                self.scheduler_config.max_num_batched_tokens,\n"
        "                self.max_num_scheduled_tokens,\n"
        "                spec.max_num_new_slots_for_drafting if spec is not None\n"
        "                else 0,\n"
        "                self.block_size,\n"
        "                self.num_spec_tokens,\n"
        "                self.use_eagle,\n"
        "                self.need_mamba_block_aligned_split,\n"
        "                self.scheduler_config.long_prefill_token_threshold,\n"
        "                self.max_model_len,\n"
        "                self.num_lookahead_tokens,\n"
        "                self.mamba_has_prefill_checkpoint_blocks,\n"
        "            )\n",
    ),
    (
        "align_split_raw_end",
        "        end = start + num_new_tokens\n"
        "        use_internal_checkpoint = (\n",
        "        end = start + num_new_tokens\n"
        "        raw_end = end\n"
        "        max_prefill_tokens = self.max_num_scheduled_tokens\n"
        "        use_internal_checkpoint = (\n",
    ),
    (
        "align_split_return",
        "        # Stop at the earliest mandatory position strictly inside the chunk.\n"
        "        end = min((s for s in stops if start < s < end), default=end)\n"
        "        return max(end - start, 0)\n",
        "        # Stop at the earliest mandatory position strictly inside the chunk.\n"
        "        end = min((s for s in stops if start < s < end), default=end)\n"
        "        result = max(end - start, 0)\n"
        "        if _SCHED_TRACE and result == 0:\n"
        "            logger.info(\n"
        "                \"[SCHEDTRACE] align_clip_zero: step=%d request_id=%s \"\n"
        "                \"start=%d raw_end=%d aligned_end=%d clipped_end=%d \"\n"
        "                \"block_size=%d max_prefill_tokens=%d prefill_end=%d \"\n"
        "                \"use_internal_ckpt=%s last_cache=%s stops=%s\",\n"
        "                self.current_step,\n"
        "                request.request_id,\n"
        "                start,\n"
        "                raw_end,\n"
        "                raw_end // block_size * block_size,\n"
        "                end,\n"
        "                block_size,\n"
        "                max_prefill_tokens,\n"
        "                prefill_end,\n"
        "                use_internal_checkpoint,\n"
        "                last_cache_position,\n"
        "                stops,\n"
        "            )\n"
        "        return result\n",
    ),
    (
        "run_budget_break",
        "            request = self.running[req_index]\n"
        "            if input_budget <= draft_slots:\n"
        "                break\n",
        "            request = self.running[req_index]\n"
        "            if input_budget <= draft_slots:\n"
        "                _sched_trace(\n"
        "                    \"run_budget_break\",\n"
        "                    step=self.current_step,\n"
        "                    input_budget=input_budget,\n"
        "                    draft_slots=draft_slots,\n"
        "                    token_budget=token_budget,\n"
        "                )\n"
        "                break\n",
    ),
    (
        "wait_budget_break",
        "                if input_budget <= draft_slots:\n"
        "                    break\n",
        "                if input_budget <= draft_slots:\n"
        "                    _sched_trace(\n"
        "                        \"wait_budget_break\",\n"
        "                        step=self.current_step,\n"
        "                        input_budget=input_budget,\n"
        "                        draft_slots=draft_slots,\n"
        "                        token_budget=token_budget,\n"
        "                    )\n"
        "                    break\n",
    ),
    (
        "pad_break",
        "                            if padded_num_tokens > request_token_budget:\n"
        "                                # Prefer to not schedule than schedule "
        "un-padded.\n"
        "                                break\n",
        "                            if padded_num_tokens > request_token_budget:\n"
        "                                # Prefer to not schedule than schedule "
        "un-padded.\n"
        "                                _sched_trace(\n"
        "                                    \"pad_break\",\n"
        "                                    step=self.current_step,\n"
        "                                    padded_num_tokens=padded_num_tokens,\n"
        "                                    request_token_budget="
        "request_token_budget,\n"
        "                                )\n"
        "                                break\n",
    ),
    (
        "align_zero_break",
        "                        if num_new_tokens == 0:\n"
        "                            break\n"
        "                        if (\n"
        "                            pad_spec_decode\n",
        "                        if num_new_tokens == 0:\n"
        "                            _sched_trace(\n"
        "                                \"mamba_align_zero\",\n"
        "                                step=self.current_step,\n"
        "                                request_id=request_id,\n"
        "                                num_tokens=request.num_tokens,\n"
        "                                computed=num_computed_tokens,\n"
        "                                request_token_budget=request_token_budget,\n"
        "                                token_budget=token_budget,\n"
        "                                input_budget=input_budget,\n"
        "                                draft_slots=draft_slots,\n"
        "                            )\n"
        "                            break\n"
        "                        if (\n"
        "                            pad_spec_decode\n",
    ),
    (
        "lookahead_zero_break",
        "                    if num_new_tokens == 0:\n"
        "                        # The request cannot be scheduled.\n"
        "                        break\n",
        "                    if num_new_tokens == 0:\n"
        "                        # The request cannot be scheduled.\n"
        "                        _sched_trace(\n"
        "                            \"lookahead_zero\",\n"
        "                            step=self.current_step,\n"
        "                            request_id=request_id,\n"
        "                            num_prefill_lookahead="
        "self.num_prefill_lookahead,\n"
        "                        )\n"
        "                        break\n",
    ),
    (
        "alloc_none_break",
        "                if new_blocks is None:\n"
        "                    # The request cannot be scheduled.\n",
        "                if new_blocks is None:\n"
        "                    # The request cannot be scheduled.\n"
        "                    _sched_trace(\n"
        "                        \"alloc_none\",\n"
        "                        step=self.current_step,\n"
        "                        request_id=request_id,\n"
        "                        num_new_tokens=num_new_tokens,\n"
        "                        computed=num_computed_tokens,\n"
        "                        num_running=len(self.running),\n"
        "                        kv_usage=self.get_kv_cache_usage(),\n"
        "                    )\n",
    ),
    (
        "empty_step_heartbeat",
        "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
        "            self._update_after_schedule(scheduler_output)\n"
        "        return scheduler_output\n",
        "        if _SCHED_TRACE and not num_scheduled_tokens and self.has_requests():\n"
        "            running, waiting = self.get_request_counts()\n"
        "            _sched_trace(\n"
        "                \"empty_step\",\n"
        "                step=self.current_step,\n"
        "                running=running,\n"
        "                waiting=waiting,\n"
        "                token_budget=token_budget,\n"
        "                input_budget=input_budget,\n"
        "                draft_slots=draft_slots,\n"
        "            )\n"
        "\n"
        "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
        "            self._update_after_schedule(scheduler_output)\n"
        "        return scheduler_output\n",
    ),
]


def _read(path: str) -> tuple[str, str]:
    with open(path, "rb") as fh:
        data = fh.read()
    eol = "crlf" if data.count(b"\r\n") and not data.count(b"\n") - data.count(
        b"\r\n"
    ) else "lf"
    return data.replace(b"\r\n", b"\n").decode("utf-8"), eol


def _write(path: str, text: str, eol: str) -> None:
    data = text.encode("utf-8")
    if eol == "crlf":
        data = data.replace(b"\n", b"\r\n")
    with open(path, "wb") as fh:
        fh.write(data)


def _paths(target: str) -> list[str]:
    if target == "venv":
        return [VENV]
    if target == "repo":
        return [REPO]
    return [VENV, REPO]


def _state(path: str) -> str:
    text, _eol = _read(path)
    return "patched" if MARK in text else "clean"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["apply", "revert", "status"])
    ap.add_argument("--target", choices=["venv", "repo", "both"], default="venv")
    args = ap.parse_args()

    if args.mode == "status":
        for p in _paths(args.target):
            print(f"{_state(p):8s} {p}")
        return 0

    for path in _paths(args.target):
        text, eol = _read(path)
        if args.mode == "apply":
            if MARK in text:
                print(f"already patched: {path}")
                continue
            for name, old, new in EDITS:
                n = text.count(old)
                assert n == 1, f"anchor {name} count={n} in {path}"
            for name, old, new in EDITS:
                text = text.replace(old, new, 1)
            compile(text, path, "exec")
            _write(path, text, eol)
            print(f"patch applied ({eol}): {path}")
        else:
            if MARK not in text:
                print(f"already clean: {path}")
                continue
            for name, old, new in reversed(EDITS):
                n = text.count(new)
                assert n == 1, f"new block {name} count={n} in {path}"
            for name, old, new in reversed(EDITS):
                text = text.replace(new, old, 1)
            assert MARK not in text
            compile(text, path, "exec")
            _write(path, text, eol)
            print(f"patch reverted ({eol}): {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
