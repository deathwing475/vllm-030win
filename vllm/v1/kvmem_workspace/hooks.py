# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 098): model-side KVMem hooks (design doc §1.2-C).

KVMem needs two decisions from the model and one description of where its
pre-RoPE K can be captured:

* ``rawk_layer(config, prefix)``      -- does this attention layer record raw K
  for the retrieval index? (step 061: only ``full_attention`` layers own a KV
  cache; requires ``VLLM_KVMEM_RAWK``.)
* ``per_layer_sliding_window(config, prefix)`` -- what per-layer window does
  this attention layer get? (step 057: ``VLLM_KVMEM_SW_WINDOW`` applies to the
  ``full_attention`` layers of an interleaved hybrid, which cannot be reached
  through the all-sliding ``CacheConfig.sliding_window``.)
* ``record_point``                    -- where in that model's forward the
  pre-RoPE K can be cloned out, so the next architecture implements the same
  capture instead of reverse-engineering it (step 094's census: the record
  call itself must stay inside the model file -- it sits between ``k_norm``
  and ``rotary_emb`` in the layer's own computation -- but everything
  *decidable* about it lives here).

Step 094's census found both decisions living as module-level functions inside
``qwen3_next.py``: correct for the one model they served, but invisible to the
next one. They move here behind a registry keyed by the attention class name,
because reuse in this tree is by class, not by architecture family: Orca
(``Qwen3_5ForCausalLM``) and Interns2-Mobius construct ``Qwen3NextAttention``
directly, so ``type(self).__name__`` resolves to the same entry for all of
them -- one registration, zero model-file edits, and the step-098 acceptance
self-proof (an Orca boot logs this module's arming line). A heterogeneous
model implements the two hooks and registers itself under its own attention
class name; it does not copy the Qwen3Next logic.

Both hooks are gated increments: with the envs unset the registry answers
``None``/``False`` and the model keeps upstream behaviour byte-for-byte, so a
model that never opted in sees nothing.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# Each arming line is logged once per process, exactly where the step-057/061
# functions logged them -- only the logger's name changes with the move.
_SW_LOGGED = False
_RAWK_LOGGED = False


class ModelKVMemHooks:
    """The per-model half of the KVMem adaptation.

    ``config`` is the model's HF text config and ``prefix`` the attention
    module's own prefix (the layer index is derived from it, so the two hooks
    share one signature).
    """

    #: What this entry serves, for logs and for :func:`describe_registry`.
    name: str = "abstract"
    #: Where this model exposes pre-RoPE K to ``capture.record``.
    record_point: str = ""

    def rawk_layer(self, config, prefix: str) -> bool:
        """Should this layer record raw K for the retrieval index?"""
        raise NotImplementedError

    def per_layer_sliding_window(self, config, prefix: str) -> int | None:
        """The per-layer sliding window for this layer, or None."""
        raise NotImplementedError


class Qwen3NextKVMemHooks(ModelKVMemHooks):
    """The step-057/061 decisions, moved here verbatim from qwen3_next.py.

    Only the ``full_attention`` layers own a KV cache; the GDN
    (``linear_attention``) layers carry recurrent state and are never touched,
    so an unarmed boot and a fully armed one both leave those layers alone.
    """

    name = "Qwen3Next hybrid (full_attention layers own the KV cache)"
    record_point = (
        "Qwen3NextAttention._project_qkv_gate, eager norm+RoPE path only "
        "(arming raw-K forces use_fused_qk_norm_rope_gate off -- the fused "
        "kernel exposes no intermediate): clone K after k_norm, before "
        "rotary_emb (ops.rotary_embedding rotates in place); gate qkv layout "
        "splits q_gate=[q|gate], k, v"
    )

    def rawk_layer(self, config, prefix: str) -> bool:
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.kvmem_workspace import capture

        if not capture.enabled():
            return False
        layer_idx = extract_layer_index(prefix)
        layer_types = getattr(config, "layer_types", None)
        if (
            layer_types is None
            or layer_idx >= len(layer_types)
            or layer_types[layer_idx] != "full_attention"
        ):
            return False
        global _RAWK_LOGGED
        if not _RAWK_LOGGED:
            _RAWK_LOGGED = True
            logger.info(
                "vllm-030win patch (step 061): raw-K capture on the "
                "full_attention layers (VLLM_KVMEM_RAWK); those layers use "
                "the eager norm+RoPE path because the fused kernel exposes "
                "no pre-RoPE K"
            )
        return True

    def per_layer_sliding_window(self, config, prefix: str) -> int | None:
        global _SW_LOGGED
        raw = os.environ.get("VLLM_KVMEM_SW_WINDOW", "").strip()
        if not raw:
            return None
        from vllm.model_executor.models.utils import extract_layer_index

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            return None
        if layer_types[extract_layer_index(prefix)] != "full_attention":
            return None
        try:
            window = int(raw)
        except ValueError:
            raise ValueError(
                "VLLM_KVMEM_SW_WINDOW must be a positive integer, got %r"
                % (raw,)
            ) from None
        if window < 1:
            raise ValueError(
                "VLLM_KVMEM_SW_WINDOW must be >= 1, got %d" % window
            )
        if not _SW_LOGGED:
            _SW_LOGGED = True
            logger.info(
                "vllm-030win patch (step 057): full_attention layers get a "
                "%d-token sliding window (VLLM_KVMEM_SW_WINDOW)",
                window,
            )
        return window


QWEN3NEXT_KVMEM_HOOKS = Qwen3NextKVMemHooks()

# attention class name -> hooks. Keyed by ``type(self).__name__`` at the
# query site, so every model that constructs Qwen3NextAttention (qwen3_next,
# qwen3_5/Orca, interns2_mobius, the qwen4_exp variants) hits this entry
# without any registration of its own.
MODEL_HOOKS: dict[str, ModelKVMemHooks] = {}


def register_model_hooks(
    attention_class_names: str | tuple[str, ...], hooks: ModelKVMemHooks
) -> None:
    """Register *hooks* under one or more attention class names.

    This is the whole adaptation for a heterogeneous model: implement the two
    hooks (and describe its record point), then register the class its
    attention layer is constructed as.
    """
    names = (
        (attention_class_names,)
        if isinstance(attention_class_names, str)
        else tuple(attention_class_names)
    )
    for name in names:
        MODEL_HOOKS[name] = hooks


def model_hooks_for(attention_class_name: str) -> ModelKVMemHooks | None:
    """The hooks registered for this attention class, or None.

    None means "no model opted in" -- the caller keeps upstream behaviour
    (no window, no capture), which is also what an unregistered architecture
    gets without any error: KVMem simply is not armed for it.
    """
    return MODEL_HOOKS.get(attention_class_name)


def describe_registry() -> str:
    """One-line account of the registry, for logs."""
    return "; ".join(
        f"{name}: {hooks.name}" for name, hooks in sorted(MODEL_HOOKS.items())
    )


register_model_hooks("Qwen3NextAttention", QWEN3NEXT_KVMEM_HOOKS)
