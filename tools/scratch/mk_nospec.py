# -*- coding: utf-8 -*-
"""Derive a no-speculation launcher variant (N=0).

Dropping the drafter removes two per-request block consumers:
  * the 5 drafter sliding-window layers (sw bucket) disappear entirely;
  * MambaSpec copies drop from 2+N to 2.
So the ceiling rises sharply -- at G=1: 16k + 48*2 <= 1041 -> k=59 -> 167,088.
The cost is the DFlash2 speedup, which is what this arm measures.

Two other edits are mandatory for a drafterless run:
  * --cudagraph-capture-sizes must equal (1+N)*max_num_seqs = 1 (step 035
    measured decode 10.31 -> 17.57 tok/s when this is wrong);
  * the three --speculative-config.* lines are removed.

Usage: python mk_nospec.py <out.cmd> [--g N] [--len L]
"""
import argparse
import re

SRC = r"G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--g", type=int, default=8)
ap.add_argument("--len", dest="length", type=int, required=True)
a = ap.parse_args()

data = open(SRC, "rb").read()
assert data.count(b"\r\n") == data.count(b"\n"), "source must be pure CRLF"

# 1) drop the three speculative-config lines (each ends with " ^\r\n")
pat = re.compile(rb"  --speculative-config[^\r\n]*\r\n")
data, n = pat.subn(b"", data)
assert n == 3, "speculative-config lines removed=%d" % n

# 2) capture sizes back to 1 (= concurrent requests, no draft slots)
old_cap = b"  --cudagraph-capture-sizes 3 "
assert data.count(old_cap) == 1, "capture-sizes anchor"
data = data.replace(old_cap, b"  --cudagraph-capture-sizes 1 ")

# 3) knobs
old_g = b'set "VLLM_KV_GROUP_SIZE=8"'
assert data.count(old_g) == 1
data = data.replace(old_g, b'set "VLLM_KV_GROUP_SIZE=%d"' % a.g)

pat_len = re.compile(rb"--max-model-len \d+")
assert len(pat_len.findall(data)) == 1
data = pat_len.sub(b"--max-model-len %d" % a.length, data)

assert not re.search(rb"(?m)^  --speculative-config", data), \
    "spec config still present"
assert data.count(b"\r\n") == data.count(b"\n"), "output must stay CRLF"
open(a.out, "wb").write(data)
print("OK -> %s  G=%d len=%d (no spec)" % (a.out, a.g, a.length))
