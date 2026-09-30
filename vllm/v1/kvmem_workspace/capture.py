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

``record`` runs inside the model forward. Up to step 066 that forced the arm
onto ``--enforce-eager``: an AOT ``torch.compile(fullgraph=True)`` of the model
traced straight into this module and rejected it (first as a ``logging`` call,
then as ``aten.equal.default`` — the data-dependent M-RoPE check). Step 067
makes the capture opaque to the compiler instead: :func:`record` is now a
``torch.library.custom_op`` whose body dynamo never sees, so the arm runs with
the production graph mode.

Two properties are what make that safe, and both were checked before the arm
was re-measured:

* A custom op is a single node in the compiled graph, but its *implementation*
  still runs on every execution of that graph. That is what keeps the stash fed
  — and it is also why the body must stay cheap and allocation-only.
* The arm keeps ``--cudagraph-capture-sizes 1``. ``CudagraphDispatcher.dispatch``
  returns ``CUDAGraphMode.NONE`` for any batch larger than the largest capture
  size, so a prefill step (up to ``max_num_batched_tokens``) is never recorded
  into a CUDA graph and the capture always runs on the host side of the stream.
  Decode (one token) *is* captured, but it returns from the ``num_tokens <= 1``
  guard without touching the device, so no device work of ours ends up in the
  graph; replays of that graph do not re-enter Python at all.

The one thing that must stay out of the body is a host synchronisation. The
device-to-host copies live in :func:`drain`, which the connector worker calls
after the forward has finished.
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


def _record_impl(
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


@torch.library.custom_op("vllm_kvmem::record", mutates_args="unknown")
def record(
    layer_idx: int,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """The compiler-visible face of :func:`_record_impl` (step 067).

    ``torch.compile`` treats a ``torch.library.custom_op`` as a leaf: it emits
    one call node and never traces the body, so the stash can keep its Python
    bookkeeping (the per-layer dict, the M-RoPE check, the counters) without
    dynamo rejecting any of it. ``mutates_args="unknown"`` says the op may
    write anything, which is what stops the compiler from reordering it or
    dropping it as dead code.
    """
    _record_impl(layer_idx, positions, q, k, num_heads, num_kv_heads, head_dim)


@record.register_fake
def _record_fake(
    layer_idx: int,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Nothing to infer: the op has no outputs."""
    return None


def drain() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Move the stashed step to the host and clear it.

    Returns ``layer_idx -> (positions int64 [T], q fp16 [<=span, H*D],
    k fp16 [T, Hkv*D])``. The host copies happen here rather than inside the
    forward so that no synchronising copy can land inside a compiled or
    graphed region.
    """
    global _STASH, _LOGGED, _SKIPPED_DECODE
    if not _STASH:
        return {}
    stash, _STASH = _STASH, {}
    # Step 067 health check. The op body is not re-entered when its node replays
    # from a CUDA graph, so the number of single-token calls since the previous
    # drain is a direct read on whether decode is actually running inside the
    # FULL graph: 0 means graphed, anything else means it fell back to eager.
    skipped, _SKIPPED_DECODE = _SKIPPED_DECODE, 0
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
    logger.info(
        "vllm-030win patch (step 067): KVMem capture drain: %d layer(s), "
        "%d token(s), %d single-token call(s) since the last drain "
        "(0 = decode replayed from a CUDA graph, >0 = decode ran the op body)",
        len(out),
        next(iter(out.values()))[2].shape[0] if out else 0,
        skipped,
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
