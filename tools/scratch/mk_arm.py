# -*- coding: utf-8 -*-
"""Derive a production-config launcher variant for one context-lever arm.

Only the requested knobs change; everything else (pin shim, manual KV pool,
graph mode, offload, spec method) stays byte-identical to the production
launcher, so any measured delta is attributable to the knob.

Anchors are regexes over the numeric value, so the script keeps working when
the production launcher's own values move (140000 -> 144432 already happened
once and silently broke the literal-anchor version).

Usage:
  python mk_arm.py <out.cmd> [--g N] [--len L] [--n N] [--mbt N]
"""
import argparse
import re

SRC = r"G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--g", type=int)
ap.add_argument("--len", dest="length", type=int)
ap.add_argument("--n", type=int)
ap.add_argument("--mbt", type=int)
a = ap.parse_args()

txt = open(SRC, encoding="utf-8", newline="").read()
assert "\n" not in txt.replace("\r\n", ""), "source launcher must be pure CRLF"


def sub_re(pattern, repl):
    global txt
    hits = list(re.finditer(pattern, txt))
    assert len(hits) == 1, "pattern %r matched %d times" % (pattern, len(hits))
    m = hits[0]
    txt = txt[:m.start()] + repl + txt[m.end():]


if a.g is not None:
    sub_re(r'set "VLLM_KV_GROUP_SIZE=\d+"',
           'set "VLLM_KV_GROUP_SIZE=%d"' % a.g)
if a.length is not None:
    sub_re(r"--max-model-len \d+", "--max-model-len %d" % a.length)
if a.n is not None:
    sub_re(r"--speculative-config\.num_speculative_tokens \d+",
           "--speculative-config.num_speculative_tokens %d" % a.n)
if a.mbt is not None:
    sub_re(r"--max-num-batched-tokens \d+",
           "--max-num-batched-tokens %d" % a.mbt)

assert "\n" not in txt.replace("\r\n", ""), "output must stay pure CRLF"
open(a.out, "w", encoding="utf-8", newline="").write(txt)
print("OK ->", a.out)
