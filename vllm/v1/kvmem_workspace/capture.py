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

Step 075 (speculative decode on the arm) changed the second property: with
``num_spec_tokens=2`` a decode step verifies 3 tokens, so it is both larger than
one token *and* captured into a CUDA graph -- the ``num_tokens <= 1`` guard alone
let it into the body, whose clone and M-RoPE ``torch.equal`` then abort the
capture (``cudaErrorStreamCaptureInvalidated``, observed on the first 075 boot).
Recording is therefore armed explicitly: the worker calls :func:`arm` with the
token counts of the step's *prefill* spans when it binds the step's metadata
(before the forward) and :func:`disarm` afterwards, so a step the connector does
not expect prompt tokens from records nothing, whatever its size.
"""

from __future__ import annotations

import numpy as np
import time
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# layer_idx -> (positions int64 [3|1, T], q_tail fp16 [<=span, H*D], k fp16 [T, Hkv*D])
_STASH: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_GEOMETRY: tuple[int, int, int] | None = None
_LOGGED = False
_SKIPPED_DECODE = 0
_SKIPPED_UNARMED = 0
_MROPE_AXES_DIFFER = 0
# vllm-030win step 079 timing instrumentation: drain(), split.
# A dict so the loop below needs no ``global`` statements.
_DRAIN_ACC: dict[str, float] = {"sync": 0.0, "copies": 0.0}
# vllm-030win step 080 capture-record fix: the cost of the record()
# custom op, which runs INSIDE execute_model and therefore was never
# inside any window the wait_for_save-based [KVTIME] ledger measured.
#   seconds    = monotonic time spent in the armed body (per call)
#   calls      = armed calls (16 per page step on this model)
#   sync_calls = of those, how many paid the torch.equal stream sync
_REC_ACC: dict[str, float] = {"seconds": 0.0, "calls": 0.0,
                              "sync_calls": 0.0}
# Resolved once: _record_impl runs 16 times per page step.
_NOSYNC: bool | None = None


def _nosync() -> bool:
    """``VLLM_KVMEM_RECORD_NOSYNC``, cached (this is a hot path)."""
    global _NOSYNC
    if _NOSYNC is None:
        from vllm.v1.kvmem_workspace import config

        _NOSYNC = bool(config.record_nosync())
    return _NOSYNC
# Step 075: the token counts the connector expects to compute in the step about
# to run (its prefill spans). None = the connector has not armed this step, so
# nothing is recorded. An explicit allow-list is the only rule that survives
# speculative decode, where a verify step is 1 + num_spec_tokens tokens and runs
# inside a CUDA graph.
_ARMED: set[int] | None = None


def enabled() -> bool:
    from vllm import envs

    return envs.VLLM_KVMEM_RAWK


def arm(counts: set[int]) -> None:
    """Allow recording for a step of exactly these token counts (step 075).

    Called by the connector worker when it binds the step's metadata, i.e.
    before the forward. Passing an empty set means "this step computes no prompt
    tokens" -- a decode/verify step -- and nothing is recorded.
    """
    global _ARMED
    _ARMED = set(counts)


def disarm() -> None:
    """Forget the allow-list: an unbound step records nothing."""
    global _ARMED
    _ARMED = None


def armed() -> set[int] | None:
    return _ARMED


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
    global _GEOMETRY, _SKIPPED_DECODE, _SKIPPED_UNARMED, _MROPE_AXES_DIFFER

    num_tokens = k.shape[0]
    if num_tokens <= 1:
        _SKIPPED_DECODE += 1
        return
    # Step 075: with speculation a verify step is more than one token and IS
    # captured into a CUDA graph, so the token count alone cannot tell prefill
    # from decode. Only the connector knows, and it says so through ``arm``
    # before the forward; anything outside that allow-list records nothing.
    # (A draft step of the drafter's own model never reaches this module.)
    if _ARMED is None or num_tokens not in _ARMED:
        _SKIPPED_UNARMED += 1
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
    #
    # vllm-030win step 080 capture-record fix. The canary below was written
    # as ``torch.equal(positions[0], positions[1])``, and on a CUDA tensor
    # torch.equal BLOCKS until the stream drains. It runs once per
    # full-attention layer => 16 synchronisation points inside every
    # 1456-token forward, which py-spy measured at 94.6% of the wall clock
    # of a slow-state boot (94.62% of 7640 samples, leaf
    # capture.py:_record_impl). Iron rule 16 (vi) already says device->host
    # reads belong in drain(), not in the forward; this honours it: rows 0
    # and 1 are stashed and compared where the data is already headed to
    # the host, so the canary and its per-layer count survive with zero
    # syncs in the forward. Gate off (default) = the old code path.
    _t080 = time.monotonic()
    axes = None
    if positions.ndim == 2:
        if _nosync():
            axes = positions[:2]
        else:
            if not torch.equal(positions[0], positions[1]):
                _MROPE_AXES_DIFFER += 1
            _REC_ACC["sync_calls"] += 1.0
            positions = positions[0]

    span = min(_query_span(), num_tokens)
    _STASH[layer_idx] = (
        (axes if axes is not None else positions).detach().clone(),
        q.detach()[-span:].clone(),
        k.detach().clone(),
    )
    _REC_ACC["seconds"] += time.monotonic() - _t080
    _REC_ACC["calls"] += 1.0


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
    global _STASH, _LOGGED, _SKIPPED_DECODE, _SKIPPED_UNARMED
    global _MROPE_AXES_DIFFER
    # The counters flush even on a step that recorded nothing: with the step 075
    # allow-list a decode/verify step leaves an empty stash, and "0 calls
    # outside the armed sizes since the last drain" is the read that the graphed
    # step never re-entered the body (067's judge, generalised to speculation).
    skipped, _SKIPPED_DECODE = _SKIPPED_DECODE, 0
    unarmed, _SKIPPED_UNARMED = _SKIPPED_UNARMED, 0
    if not _STASH:
        if skipped or unarmed:
            logger.info(
                "vllm-030win patch (step 075): KVMem capture drain: nothing "
                "recorded this step (%d single-token call(s), %d call(s) "
                "outside armed sizes %s; 0/0 = decode ran on the CUDA graph "
                "without touching the capture body)",
                skipped,
                unarmed,
                sorted(_ARMED) if _ARMED else _ARMED,
            )
        return {}
    stash, _STASH = _STASH, {}
    # Step 067 health check. The op body is not re-entered when its node replays
    # from a CUDA graph, so the number of calls the allow-list rejected is a
    # direct read on whether decode is actually running inside the FULL graph:
    # 0 means graphed, anything else means it fell back to eager.
    _t079 = time.monotonic()
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    _split = False
    for layer_idx, (positions, q_tail, k) in stash.items():
        _pos = positions.cpu().numpy()
        if _pos.ndim == 2:
            # vllm-030win step 080 capture-record fix: the M-RoPE axis
            # canary moved here from the forward, where it cost a stream
            # sync per layer. Same read, same per-layer granularity, done
            # with numpy on data drain() copies to the host anyway.
            if bool((_pos[0] != _pos[1]).any()):
                _MROPE_AXES_DIFFER += 1
            _pos = _pos[0]
        if not _split:
            _DRAIN_ACC["sync"] += time.monotonic() - _t079
            _split = True
            _t079 = time.monotonic()
        out[layer_idx] = (
            _pos,
            q_tail.cpu().to(torch.float16).numpy(),
            k.cpu().to(torch.float16).numpy(),
        )
    _DRAIN_ACC["copies"] += time.monotonic() - _t079
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
        "%d token(s), %d single-token call(s) + %d call(s) outside armed sizes "
        "%s since the last drain (0 + 0 = decode replayed from a CUDA graph, "
        ">0 = decode ran the op body)",
        len(out),
        next(iter(out.values()))[2].shape[0] if out else 0,
        skipped,
        unarmed,
        sorted(_ARMED) if _ARMED else _ARMED,
    )
    return out


def stats() -> dict:
    return {
        "armed": enabled(),
        "layers_stashed": len(_STASH),
        "skipped_decode_steps": _SKIPPED_DECODE,
        "skipped_unarmed_steps": _SKIPPED_UNARMED,
        "armed_counts": sorted(_ARMED) if _ARMED else _ARMED,
        "mrope_axes_differ": _MROPE_AXES_DIFFER,
        "geometry": _GEOMETRY,
        "drain_sync_seconds": round(_DRAIN_ACC["sync"], 3),
        "drain_copy_seconds": round(_DRAIN_ACC["copies"], 3),
        # vllm-030win step 080 capture-record fix
        "record_seconds": round(_REC_ACC["seconds"], 3),
        "record_calls": int(_REC_ACC["calls"]),
        "record_sync_calls": int(_REC_ACC["sync_calls"]),
        "record_nosync": int(_nosync()),
    }


def reset() -> None:
    global _STASH, _GEOMETRY, _LOGGED, _SKIPPED_DECODE, _SKIPPED_UNARMED
    global _MROPE_AXES_DIFFER, _ARMED
    _STASH = {}
    _GEOMETRY = None
    _LOGGED = False
    _SKIPPED_DECODE = 0
    _SKIPPED_UNARMED = 0
    _MROPE_AXES_DIFFER = 0
    # vllm-030win step 080: the accumulators, not the gate -- the env
    # does not change within a boot, and re-reading it 16x per step was
    # the very cost being removed.
    _REC_ACC["seconds"] = 0.0
    _REC_ACC["calls"] = 0.0
    _REC_ACC["sync_calls"] = 0.0
    _ARMED = None
