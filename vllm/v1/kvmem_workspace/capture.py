# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 061): KVMem pre-RoPE q/k capture (K3 item 1).

The retrieval index needs the K *before* RoPE: the index lives in the content
frame, so scoring a stored page against the current query needs no inverse
rotation (design doc §5.2). ``ops.rotary_embedding`` rotates in place, so the
value has to be cloned out of the attention layer before the rotation runs.

Two deliberate restrictions keep the cost bounded:

* only the 16 ``full_attention`` layers record (the 48 GDN layers have no KV
  cache at all);
* only prefill steps record (``k.shape[0] > 1``). Decode steps carry a single
  token whose KV never reaches the workspace as a stored page, and they are the
  steps that run inside a CUDA graph, where a clone or a host copy would be
  illegal.

``record`` therefore only stashes references (plus a clone of the last
``query_span`` query rows); every host copy happens in :func:`drain`, which the
connector worker calls after the forward has finished. The stash is keyed by
layer index, so it is bounded by the layer count no matter how many forwards
run without a drain.

``record`` runs inside the model forward, so it must not do anything a
``torch.compile`` fullgraph rejects — no logging, no host copies, no
data-dependent Python control flow. The arm therefore runs with
``--enforce-eager``, which is what the design asks for at stage 1 anyway
("文本 + 无投机 + 无图模式", §6); a compiled forward would need the capture to
be an output of the graph instead.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# layer_idx -> (positions int64 [3|1, T], q_tail fp16 [<=span, H*D], k fp16 [T, Hkv*D])
_STASH: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_GEOMETRY: tuple[int, int, int] | None = None
_LOGGED = False
_SKIPPED_DECODE = 0
_MROPE_AXES_DIFFER = 0


def enabled() -> bool:
    from vllm import envs

    return envs.VLLM_KVMEM_RAWK


def _query_span() -> int:
    from vllm.v1.kvmem_workspace import config

    return config.query_span()


def record(
    layer_idx: int,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Stash the pre-RoPE q/k of one full-attention layer for this step.

    Called from ``Qwen3NextAttention._project_qkv_gate`` on the eager path,
    between ``k_norm`` and ``self.rotary_emb``. Must stay cheap and
    allocation-only: it runs inside the forward.
    """
    global _GEOMETRY, _SKIPPED_DECODE, _MROPE_AXES_DIFFER

    num_tokens = k.shape[0]
    if num_tokens <= 1:
        _SKIPPED_DECODE += 1
        return

    if _GEOMETRY is None:
        _GEOMETRY = (num_heads, num_kv_heads, head_dim)
    elif _GEOMETRY != (num_heads, num_kv_heads, head_dim):
        raise RuntimeError(
            "vllm-030win KVMem raw-K capture: attention geometry changed from "
            f"{_GEOMETRY} to {(num_heads, num_kv_heads, head_dim)}"
        )

    # M-RoPE hands the layer a [3, T] position tensor; text-only input repeats
    # the same row three times, which is the scalar position the index wants.
    # Vision would make the axes differ, and a single scalar cannot express
    # H/W — the design keeps vision out of stage 1 for exactly that reason.
    if positions.ndim == 2:
        if not torch.equal(positions[0], positions[1]):
            _MROPE_AXES_DIFFER += 1
        positions = positions[0]

    span = min(_query_span(), num_tokens)
    _STASH[layer_idx] = (
        positions.detach().clone(),
        q.detach()[-span:].clone(),
        k.detach().clone(),
    )


def drain() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Move the stashed step to the host and clear it.

    Returns ``layer_idx -> (positions int64 [T], q fp16 [<=span, H*D],
    k fp16 [T, Hkv*D])``. The host copies happen here rather than inside the
    forward so that no synchronising copy can land inside a compiled or
    graphed region.
    """
    global _STASH, _LOGGED
    if not _STASH:
        return {}
    stash, _STASH = _STASH, {}
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer_idx, (positions, q_tail, k) in stash.items():
        out[layer_idx] = (
            positions.cpu().numpy(),
            q_tail.cpu().to(torch.float16).numpy(),
            k.cpu().to(torch.float16).numpy(),
        )
    if not _LOGGED:
        _LOGGED = True
        num_heads, num_kv_heads, head_dim = _GEOMETRY or (0, 0, 0)
        logger.info(
            "vllm-030win patch (step 061): KVMem raw-K capture armed "
            "(num_heads=%d, num_kv_heads=%d, head_dim=%d, query_span=%d); "
            "first drain: %d layer(s), %d token(s), q tail %d row(s)",
            num_heads,
            num_kv_heads,
            head_dim,
            _query_span(),
            len(out),
            next(iter(out.values()))[2].shape[0] if out else 0,
            next(iter(out.values()))[1].shape[0] if out else 0,
        )
    return out


def stats() -> dict:
    return {
        "armed": enabled(),
        "layers_stashed": len(_STASH),
        "skipped_decode_steps": _SKIPPED_DECODE,
        "mrope_axes_differ": _MROPE_AXES_DIFFER,
        "geometry": _GEOMETRY,
    }


def reset() -> None:
    global _STASH, _GEOMETRY, _LOGGED, _SKIPPED_DECODE, _MROPE_AXES_DIFFER
    _STASH = {}
    _GEOMETRY = None
    _LOGGED = False
    _SKIPPED_DECODE = 0
    _MROPE_AXES_DIFFER = 0
