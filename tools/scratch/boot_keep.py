# -*- coding: utf-8 -*-
"""boot_keep.py — start a launcher detached (survives this process), no wait."""
import subprocess
import sys

LAUNCHER = sys.argv[1]
OUT = sys.argv[2]
ERR = sys.argv[3]

fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
subprocess.Popen(
    ["cmd.exe", "/d", "/c", LAUNCHER],
    cwd=r"G:\qwen3.8model",
    stdout=fo, stderr=fe,
    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
)
print("spawned:", LAUNCHER)
