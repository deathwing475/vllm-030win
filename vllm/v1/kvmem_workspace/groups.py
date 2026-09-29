# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): which KV cache groups the workspace stores.

Both sides of the connector (scheduler and worker) must agree on this set
byte-for-byte, because the host slot layout is derived from it. Only the
KVMem sliding-window attention groups qualify: the 48 GDN (Mamba) layers are
excluded, exactly as the design requires ("DeltaNet layers must be kept out of
all block/tiering logic").
"""

import os

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig, SlidingWindowSpec

logger = init_logger(__name__)

_logged = False


def _arm_window() -> int | None:
    raw = os.environ.get("VLLM_KVMEM_SW_WINDOW", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"VLLM_KVMEM_SW_WINDOW must be an integer, got {raw!r}") from None


def workspace_group_ids(kv_cache_config: KVCacheConfig) -> list[int]:
    """Group ids the KVMem workspace stores, in kv-cache-group order.

    When ``VLLM_KVMEM_SW_WINDOW`` is set the window is the selector, so a
    speculative drafter's own (much smaller) sliding window cannot be mistaken
    for the target's KVMem window. Without it every sliding-window group
    qualifies.
    """
    window = _arm_window()
    ids: list[int] = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, SlidingWindowSpec):
            continue
        if window is not None and spec.sliding_window != window:
            continue
        ids.append(group_id)

    global _logged
    if not _logged:
        _logged = True
        if ids:
            logger.info(
                "vllm-030win patch (step 060): KVMem workspace stores kv cache "
                "group(s) %s (sliding_window=%s, block_size=%s)",
                ids,
                [kv_cache_config.kv_cache_groups[i].kv_cache_spec.sliding_window for i in ids],
                [kv_cache_config.kv_cache_groups[i].kv_cache_spec.block_size for i in ids],
            )
        else:
            logger.warning(
                "vllm-030win patch (step 060): KVMem workspace is armed but no "
                "sliding-window kv cache group matches VLLM_KVMEM_SW_WINDOW=%s; "
                "nothing will be spilled",
                window,
            )
    return ids
