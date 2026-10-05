# SPDX-License-Identifier: Apache-2.0
"""Offline unit test for the KVMem model-hook registry (step 098).

Step 098 moved the two model-side KVMem decisions out of ``qwen3_next.py``
into ``vllm/v1/kvmem_workspace/hooks.py`` behind a registry keyed by the
attention class name (design doc §1.2-C). This test pins the moved logic to
the behaviour the frozen step-083 arm was accepted with, plus the registry
semantics that make the move worth doing:

* the Qwen3Next entry answers exactly as the old module-level functions did
  (env-off silence, layer_types gating, window validation, boundary indices);
* reuse is by class name, so every model that constructs
  ``Qwen3NextAttention`` (qwen3_5/Orca, interns2_mobius, qwen4_exp) resolves
  to the same entry without registering anything -- this is the offline half
  of the step-098 Orca self-proof (the boot half reads the arming log line);
* a heterogeneous model registers its own entry and that entry answers.

Usage (venv python, cwd outside the record repo):
    python kvmem_hooks_test.py
"""
import os
import sys

import torch  # noqa: F401  (imported to mirror the engine-side import order)

from vllm.v1.kvmem_workspace import hooks as hook_mod
from vllm.v1.kvmem_workspace.hooks import (
    Qwen3NextKVMemHooks,
    describe_registry,
    model_hooks_for,
    register_model_hooks,
)

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _raises(exc_type: type, call) -> bool:
    """True when *call* raises this error type -- a guard that stays silent fails."""
    try:
        call()
    except exc_type:
        return True
    except Exception as exc:  # noqa: BLE001 - the wrong type is a failure too
        print(f"    (raised {exc!r}, expected {exc_type.__name__})")
        return False
    return False


class _Cfg:
    """Stand-in for the HF text config: only layer_types is consulted."""

    def __init__(self, layer_types):
        self.layer_types = layer_types


# 64 layers, every 4th a full_attention layer -- the shape both GSQ and the
# Orca checkpoint (qwen3.8exl3) declare.
HYBRID = (
    ["linear_attention"] * 3 + ["full_attention"]
) * 16
FULL_PREFIX = "model.layers.3.self_attn"
GDN_PREFIX = "model.layers.0.self_attn"
OOB_PREFIX = "model.layers.64.self_attn"


def with_env(name: str, value: str | None, call):
    """Run *call* with one env var set, restoring it (and the log-once flags)."""
    old = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        hook_mod._SW_LOGGED = False
        hook_mod._RAWK_LOGGED = False
        return call()
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def main() -> int:
    hooks = model_hooks_for("Qwen3NextAttention")
    print("T1 registry lookup")
    check("the Qwen3NextAttention class name resolves to the built-in entry",
          isinstance(hooks, Qwen3NextKVMemHooks))
    check("the entry describes its record point (the next model's map)",
          bool(hooks.record_point) and "k_norm" in hooks.record_point
          and "rotary_emb" in hooks.record_point,
          hooks.record_point[:60])
    check("an unregistered attention class gets None (upstream behaviour)",
          model_hooks_for("SomeOtherAttention") is None)
    check("describe_registry names the one built-in entry",
          "Qwen3NextAttention" in describe_registry()
          and "full_attention" in describe_registry())

    print("T2 rawk_layer semantics (the moved step-061 logic)")
    check("env off answers False before layer_types is even consulted",
          with_env("VLLM_KVMEM_RAWK", None,
                   lambda: hooks.rawk_layer(_Cfg(HYBRID), FULL_PREFIX)) is False)
    check("env on + full_attention layer answers True",
          with_env("VLLM_KVMEM_RAWK", "1",
                   lambda: hooks.rawk_layer(_Cfg(HYBRID), FULL_PREFIX)) is True)
    check("env on + GDN layer answers False",
          with_env("VLLM_KVMEM_RAWK", "1",
                   lambda: hooks.rawk_layer(_Cfg(HYBRID), GDN_PREFIX)) is False)
    check("env on + missing layer_types answers False",
          with_env("VLLM_KVMEM_RAWK", "1",
                   lambda: hooks.rawk_layer(_Cfg(None), FULL_PREFIX)) is False)
    check("env on + layer index past the table answers False",
          with_env("VLLM_KVMEM_RAWK", "1",
                   lambda: hooks.rawk_layer(_Cfg(HYBRID), OOB_PREFIX)) is False)

    print("T3 per_layer_sliding_window semantics (the moved step-057 logic)")
    check("env unset answers None before layer_types is even consulted",
          with_env("VLLM_KVMEM_SW_WINDOW", None,
                   lambda: hooks.per_layer_sliding_window(
                       _Cfg(HYBRID), FULL_PREFIX)) is None)
    check("env set + full_attention layer returns the window",
          with_env("VLLM_KVMEM_SW_WINDOW", "16384",
                   lambda: hooks.per_layer_sliding_window(
                       _Cfg(HYBRID), FULL_PREFIX)) == 16384)
    check("env set + GDN layer answers None",
          with_env("VLLM_KVMEM_SW_WINDOW", "16384",
                   lambda: hooks.per_layer_sliding_window(
                       _Cfg(HYBRID), GDN_PREFIX)) is None)
    check("env set + missing layer_types answers None",
          with_env("VLLM_KVMEM_SW_WINDOW", "16384",
                   lambda: hooks.per_layer_sliding_window(
                       _Cfg(None), FULL_PREFIX)) is None)
    check("a non-integer window is refused",
          _raises(ValueError, lambda: with_env(
              "VLLM_KVMEM_SW_WINDOW", "big",
              lambda: hooks.per_layer_sliding_window(
                  _Cfg(HYBRID), FULL_PREFIX))))
    check("a zero window is refused",
          _raises(ValueError, lambda: with_env(
              "VLLM_KVMEM_SW_WINDOW", "0",
              lambda: hooks.per_layer_sliding_window(
                  _Cfg(HYBRID), FULL_PREFIX))))

    print("T4 reuse by class name (the offline half of the Orca self-proof)")
    check("Orca constructs Qwen3NextAttention directly (qwen3_5.py), so its "
          "layers carry that class name and hit this entry",
          model_hooks_for("Qwen3NextAttention") is hooks)
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    import vllm.model_executor.models.qwen3_5 as q35_mod
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    check("qwen3_5 binds the very class this entry is keyed on (no subclass)",
          q35_mod.Qwen3NextAttention is Qwen3NextAttention
          and Qwen3NextAttention.__name__ == "Qwen3NextAttention",
          f"decoder layer = {Qwen3_5DecoderLayer.__name__}")

    print("T5 heterogeneous registration (the extension path)")
    class _StubHooks(Qwen3NextKVMemHooks):
        name = "stub"

        def rawk_layer(self, config, prefix: str) -> bool:
            return True

        def per_layer_sliding_window(self, config, prefix: str) -> int | None:
            return 7

    stub = _StubHooks()
    register_model_hooks("StubAttention", stub)
    got = model_hooks_for("StubAttention")
    check("a registered heterogeneous entry is returned as-is",
          got is stub)
    check("its answers are its own (not the Qwen3Next logic)",
          got.rawk_layer(_Cfg(None), FULL_PREFIX) is True
          and got.per_layer_sliding_window(_Cfg(None), "") == 7)
    check("multi-name registration covers tuples",
          (register_model_hooks(("A1", "A2"), stub),
           model_hooks_for("A2") is stub)[-1])
    check("the registry description grew with the stub",
          "StubAttention" in describe_registry())
    # Leave the registry as it was found: the built-in map must not leak test
    # stubs into whichever engine process imports this module next.
    for name in ("StubAttention", "A1", "A2"):
        hook_mod.MODEL_HOOKS.pop(name, None)
    check("cleanup restored the built-in registry",
          model_hooks_for("StubAttention") is None
          and model_hooks_for("Qwen3NextAttention") is hooks)

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
