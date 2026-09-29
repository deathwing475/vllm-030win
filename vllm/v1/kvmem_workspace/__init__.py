# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem host KV workspace.

The workspace is the host-side store the KVMem design needs
(``docs/vllm-030win-调研-KVMem虚拟化KV工作区.md``): pages that scroll out of
the bounded attention window are copied here instead of being dropped, keyed by
``(trajectory, page_index)`` rather than by prefix-cache block hash. Stage 1a
(step 057) made a prompt longer than the pool *prefillable*; this package is the
other half — spilling what the window drops, with the pages held out of the
block pool until their copy has completed.
"""
