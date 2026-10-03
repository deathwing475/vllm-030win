"""Optional process-local OrcaSAQ2 ExLlamaV3 embedding loader patch."""
import os

from orcasaq2.patches import int8_embedding
int8_embedding.apply()

# Step 085 (Orca O2): the MTP draft's embed_tokens / lm_head are shared from the target
# *after* weight loading, so during loading they are legitimately empty -- and orcasaq2
# raises on empty. Wrap the plugin's own register() so the fixup happens once vLLM is
# already imported (importing orcasaq2.config at interpreter startup would drag all of
# vLLM into sitecustomize). Default off: only the O2 launcher sets the flag.
_ORIG_REGISTER = None
if os.environ.get("ORCA_EXL3_ALLOW_EMPTY_SHARED") == "1":
    import orcasaq2

    _ORIG_REGISTER = orcasaq2.register

    def _register_with_s085():
        out = _ORIG_REGISTER()
        from s085_empty_shared_patch import apply as _apply_s085
        _apply_s085()
        return out

    orcasaq2.register = _register_with_s085
