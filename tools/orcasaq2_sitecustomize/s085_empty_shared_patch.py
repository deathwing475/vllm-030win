"""Process-local patch (step 085 / Orca O2): let the MTP draft's shared children load empty.

Why this is needed: `Qwen3_5MTP` deliberately does not read `embed_tokens` / `lm_head` from the
checkpoint -- `vllm/v1/worker/gpu/spec_decode/eagle/utils.py::load_eagle_model` deletes those two
draft modules and rebinds the target's copies *after* weight loading. But
`base_loader.load_model` runs `process_weights_after_loading` over *every* module before that
rebinding happens, and orcasaq2's methods raise when a module received nothing:

    RuntimeError: embed_tokens: neither an int8 table nor a dense one loaded

So on an EXL3 checkpoint any MTP/EAGLE draft dies at boot for a state that is about to be
thrown away. This patch tolerates "no shards at all" only, leaving both modules in a
permanently-empty placeholder (if something ever does try to use them, `apply` /
`embedding` fail loudly rather than silently). Modules that received weights keep the
original code path byte-for-byte.

Gated by ORCA_EXL3_ALLOW_EMPTY_SHARED=1 and loaded only from the Orca launcher's
PYTHONPATH, so the GSQ production stack is untouched.
"""
import os

from vllm.logger import init_logger

logger = init_logger("vllm.orcasaq2.s085")

_FLAG = "ORCA_EXL3_ALLOW_EMPTY_SHARED"


def _empty(shards_by_suffix: list) -> bool:
    return not any(getattr(getattr(sh, "shards", None), "values", lambda: [])()
                   for sh in shards_by_suffix if sh is not None)


def apply():
    """Wrap the two EXL3 post-load hooks; no-op unless the env flag is set."""
    if os.environ.get(_FLAG) != "1":
        return False
    from orcasaq2.config import Exl3LinearMethod
    from orcasaq2.heads import Exl3EmbeddingMethod

    if getattr(Exl3LinearMethod, "_s085_patched", False):
        return True

    orig_linear = Exl3LinearMethod.process_weights_after_loading
    orig_embed = Exl3EmbeddingMethod.process_weights_after_loading
    tolerated: list[str] = []

    def linear(self, layer):
        if _empty([getattr(layer, s, None) for s in
                   ("trellis", "suh", "svh", "mcg", "mul1", "weight")]):
            for p in ("trellis", "suh", "svh", "mcg", "mul1", "weight"):
                if hasattr(layer, p):
                    delattr(layer, p)
            layer.exl3_shards = None
            layer.exl3_dense = None
            tolerated.append(self.prefix)
            logger.info("s085: %s received no weights at all; leaving it empty "
                        "(the MTP draft shares this module with the target)",
                        self.prefix)
            return
        return orig_linear(self, layer)

    def embed(self, layer):
        if _empty([getattr(layer, s, None) for s in ("qweight", "scales", "weight")]):
            for p in ("qweight", "scales", "weight"):
                if hasattr(layer, p):
                    delattr(layer, p)
            layer.exl3_embed = None
            layer.exl3_embed_q = None
            tolerated.append("embed_tokens")
            logger.info("s085: draft embed_tokens received no weights at all; "
                        "leaving it empty (shared from the target after loading)")
            return
        return orig_embed(self, layer)

    Exl3LinearMethod.process_weights_after_loading = linear
    Exl3EmbeddingMethod.process_weights_after_loading = embed
    Exl3LinearMethod._s085_patched = True
    Exl3LinearMethod._s085_toleration_log = tolerated
    logger.info("s085 armed: %s=1, empty draft-shared modules will be tolerated",
                _FLAG)
    return True
