"""CPU-only O3 pre-flight: does the DFlash2 drafter resolve against the OrcaSAQ2 EXL3
target, and where does the draft/target coupling actually get checked?

Why (主线计划 §4 阶段 O3 第 1-3 项): a boot arm costs a production stop + ~90 s + probe
time, and every question below is answerable without a GPU. Recorded into step 087:
  A. target-side facts Orca must satisfy for a DFlash2 draft (eagle3 / aux taps)
  B. draft-side resolution for the three checkpoints we actually have on disk
  C. which model runner the combination forces, and the scheduler numbers under 必守 11/28
  D. the draft module tree on the meta device: which quant method claims each linear, and
     whether embed/lm_head sharing leaves an empty module for a post-load hook to trip
     over (this is what killed MTP in 085)
  E. the one real draft/target compatibility guard, evaluated for Orca

Run with the venv python from a scratch cwd (never from the repo dir -- 必守 2):
  G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe <repo>\\tools\\step087_o3_meta_probe.py
"""

import collections
import json
from pathlib import Path

import torch

ORCA = str(Path(r"G:\qwen3.8model\qwen3.8exl3"))
GSQ = Path(r"G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ")
DRAFTS = {
    "gptq3c (auto-round GPTQ, production GSQ draft)": str(GSQ / "dflash2" / "gptq3c"),
    "ct4bit (compressed-tensors, dflash2 root)": str(GSQ / "dflash2"),
    "bf16 (unquantised original, dflash2/src)": str(GSQ / "dflash2" / "src"),
}

import orcasaq2  # noqa: E402

orcasaq2.register()

from vllm.config import (  # noqa: E402
    DeviceConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.cache import CacheConfig  # noqa: E402
from vllm.config.model import ModelConfig  # noqa: E402
from vllm.config.scheduler import SchedulerConfig  # noqa: E402
from vllm.config.speculative import SpeculativeConfig  # noqa: E402
from vllm.model_executor.models.interfaces import supports_eagle3  # noqa: E402
from vllm.model_executor.models.qwen3_dflash import (  # noqa: E402
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
)
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402
from vllm.transformers_utils.configs.eagle import EAGLEConfig  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (  # noqa: E402
    get_eagle3_aux_layers_from_config,
)


def rule(char="-"):
    print("\n" + char * 78)


def init_tp_stub():
    """tp=1 process group on gloo so module construction works without a GPU."""
    try:
        from vllm.distributed import ensure_model_parallel_initialized

        ensure_model_parallel_initialized(1, 1)
        return "ensure_model_parallel_initialized(1, 1)"
    except Exception as first:  # noqa: BLE001
        try:
            from vllm.distributed import (
                ensure_model_parallel_initialized,
                init_distributed_environment,
            )

            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29617",
                local_rank=0,
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1)
            return "init_distributed_environment(gloo) + ensure_model_parallel_initialized"
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: tp stub failed ({type(first).__name__} then "
                  f"{type(exc).__name__}: {exc}); model construction may not work")
            return "FAILED"


def show(name, value):
    print(f"  {name:46s} = {value}")


def spec_for(model, tokens=2):
    return SpeculativeConfig(
        target_model_config=TMC,
        target_parallel_config=ParallelConfig(),
        method="dflash",
        model=model,
        num_speculative_tokens=tokens,
    )


rule("=")
print("A. TARGET (OrcaSAQ2 EXL3) -- what a DFlash2 drafter needs from it")
rule("=")
TMC = ModelConfig(
    model=ORCA,
    tokenizer=ORCA,
    max_model_len=16384,
    enforce_eager=True,
    hf_overrides={},
    trust_remote_code=False,
)
TCFG = TMC.hf_config
TEXT = getattr(TCFG, "text_config", TCFG)
show("resolved vllm architectures", TMC.architectures)
show("model_type", TCFG.model_type)
show("quantization (runtime)", TMC.quantization)
show("num_hidden_layers", TEXT.num_hidden_layers)
show("hidden_size", TEXT.hidden_size)
show("vocab_size", TEXT.vocab_size)
show("full_attention_interval", getattr(TEXT, "full_attention_interval", None))
show("layer_types counts", collections.Counter(list(getattr(TEXT, "layer_types", []) or [])))
show("target sliding_window", getattr(TEXT, "sliding_window", None))
show("target dflash/eagle fields", [k for k in dir(TEXT) if "dflash" in k or "eagle" in k])

ARCH = TMC.architectures[0]
TCLS = ModelRegistry._try_load_model_cls(ARCH)
print(f"\n  target class        = {TCLS.__module__}.{TCLS.__qualname__}")
print(f"  supports_eagle3     = {supports_eagle3(TCLS)}")
print(f"  set_aux_... present = {hasattr(TCLS, 'set_aux_hidden_state_layers')}")

VC_PLAIN = VllmConfig(model_config=TMC, device_config=DeviceConfig("cpu"))
with set_current_vllm_config(VC_PLAIN):
    print(f"  tp stub                     = {init_tp_stub()}")
with torch.device("meta"), set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(VC_PLAIN):
    TGT = TCLS(vllm_config=VC_PLAIN)
print(f"  instance supports_eagle3 = {supports_eagle3(TGT)}")
try:
    print(f"  default aux layers (instance) = {TGT.get_eagle3_default_aux_hidden_state_layers()}")
except Exception as exc:  # noqa: BLE001
    print(f"  default aux layers (instance) = n/a ({type(exc).__name__})")
INNER = TGT.get_language_model() if hasattr(TGT, "get_language_model") else TGT
HOLDER = getattr(INNER, "model", INNER)
show("inner language model", type(INNER).__name__)
show("target .lm_head (share candidate A)", type(getattr(INNER, "lm_head", None)).__name__)
show("target .model.embed_tokens (what dflash/utils binds)", type(getattr(HOLDER, "embed_tokens", None)).__name__)
TGT.set_aux_hidden_state_layers((6, 20, 34, 48, 62))
show("aux layers stored on", f"{type(HOLDER).__name__} -> {getattr(HOLDER, 'aux_hidden_state_layers', 'MISSING')}")
for attr, obj in (("embed_tokens", getattr(HOLDER, "embed_tokens", None)),
                  ("lm_head", getattr(INNER, "lm_head", None))):
    print(f"  shared module {attr:14s}: type={type(obj).__name__} "
          f"quant_method={type(getattr(obj, 'quant_method', None)).__name__} "
          f"num_embeddings={getattr(obj, 'num_embeddings', 'n/a')}")

rule("=")
print("B. DRAFT resolution for every DFlash2 checkpoint on disk")
rule("=")
for name, path in DRAFTS.items():
    raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    sc = spec_for(path)
    dmc = sc.draft_model_config
    dc = dmc.hf_config
    dcfg = getattr(dc, "dflash_config", None) or {}
    print(f"\n  --- {name} ---")
    show("checkpoint architectures", raw.get("architectures"))
    show("architectures after EAGLEConfig rewrite", list(dmc.architectures or []))
    show("hf_config is EAGLEConfig wrapper", isinstance(dc, EAGLEConfig))
    show("draft model_type", getattr(dc, "model_type", None))
    darch = list(dmc.architectures or [""])[0]
    dcls = ModelRegistry._try_load_model_cls(darch)
    if dcls is None:
        show("registry class", f"NOT REGISTERED for {darch!r}")
    else:
        show("registry class", f"{dcls.__module__}.{dcls.__qualname__}")
    show("draft quantization (runtime)", dmc.quantization)
    show("draft num_hidden_layers / hidden / vocab", (getattr(dc, "num_hidden_layers", None), getattr(dc, "hidden_size", None), getattr(dc, "vocab_size", None)))
    show("dflash_config.target_layer_ids", dcfg.get("target_layer_ids"))
    show("-> aux layers vLLM will ask the target for", get_eagle3_aux_layers_from_config(sc))
    show("dflash_config.mask_token_id", dcfg.get("mask_token_id"))
    show("dflash_config conv taps / group", (dcfg.get("conv_kernel_size"), dcfg.get("conv_group_size")))
    show("dflash_config selector rank / top_k", (dcfg.get("selector_rank"), dcfg.get("selector_top_k")))
    show("is_causal -> non-causal backend required", dflash_has_any_non_causal(dc))
    show("draft layer_types", list(getattr(dc, "layer_types", []) or []))
    show("draft sliding_window", getattr(dc, "sliding_window", None))
    show("sc.kv_cache_dtype / num_spec_tokens", (sc.kv_cache_dtype, sc.num_speculative_tokens))

rule("=")
print("C. Runner + scheduler: Orca target + gptq3c draft, N=2")
rule("=")
SC = spec_for(DRAFTS["gptq3c (auto-round GPTQ, production GSQ draft)"])
for mbt in (1024, 2048, 2848):
    vc = VllmConfig(
        model_config=TMC,
        speculative_config=SC,
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(cache_dtype="nvfp4"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=mbt,
            max_model_len=16384,
            is_encoder_decoder=False,
        ),
    )
    print(f"\n  --- max_num_batched_tokens = {mbt} ---")
    show("use_v2_model_runner", vc.use_v2_model_runner)
    show("V2 unsupported", vc._get_v2_model_runner_unsupported_features())
    show("V1 unsupported", vc._get_v1_model_runner_unsupported_features())
    show("_is_dflash2_draft", vc._is_dflash2_draft())
    show("_dflash_needs_multi_kv_group", vc._dflash_needs_multi_kv_group())
    show("scheduler max_num_batched/scheduled", (vc.scheduler_config.max_num_batched_tokens, vc.scheduler_config.max_num_scheduled_tokens))
    show("spec.max_num_new_slots_for_drafting", SC.max_num_new_slots_for_drafting)
    show("cache dtype / mamba_cache_mode", (vc.cache_config.cache_dtype, vc.cache_config.mamba_cache_mode))

rule("=")
print("D. DRAFT module tree on the meta device (who claims each linear?)")
rule("=")
SEEN = []
from orcasaq2.config import Exl3Config  # noqa: E402

_ORIG_GQM = Exl3Config.get_quant_method


def _gqm(self, layer, prefix):
    out = _ORIG_GQM(self, layer, prefix)
    SEEN.append((prefix, type(out).__name__))
    return out


Exl3Config.get_quant_method = _gqm

VC_DRAFT = VllmConfig(
    model_config=TMC, speculative_config=SC, device_config=DeviceConfig("cpu")
)
with set_current_vllm_config(VC_DRAFT):
    try:
        from vllm.distributed.parallel_state import ensure_model_parallel_initialized

        ensure_model_parallel_initialized(1, 1)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN tp init: {type(exc).__name__}: {exc}")
    N0 = len(SEEN)
    with torch.device("meta"), set_default_torch_dtype(torch.bfloat16):
        from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

        DRF = DFlash2Qwen3ForCausalLM(vllm_config=VC_DRAFT)

print(f"  EXL3 get_quant_method calls made while building the draft = {len(SEEN) - N0}")
print(f"    (0 = the draft is quantised by its OWN scheme, so 085's empty-shared-module")
print(f"     crash cannot recur here; non-zero = it goes through the EXL3 plugin)")
for prefix, method in SEEN[N0:][:10]:
    print(f"    {prefix:54s} -> {method}")

print("\n  modules the draft expects to inherit from the target:")
show("draft .model.embed_tokens", type(getattr(DRF.model, "embed_tokens", None)).__name__)
show("draft .lm_head", type(getattr(DRF, "lm_head", None)).__name__)
show("draft has_own_embed_tokens", getattr(DRF, "has_own_embed_tokens", "attr absent"))
show("draft has_own_lm_head", getattr(DRF, "has_own_lm_head", "attr absent"))
print("\n  draft tensors that bind it to the target geometry:")
for name, p in DRF.named_parameters():
    if name in ("model.fc.qweight", "model.fc.weight", "model.fc.qzeros"):
        show(name, tuple(p.shape))
for name, p in DRF.named_parameters():
    if name.startswith("model.candidate_selector") and "codebook" in name:
        show(name, tuple(p.shape))
show("draft .model.hidden_norm / norm", (type(getattr(DRF.model, "hidden_norm", None)).__name__, type(getattr(DRF, "norm", None)).__name__))
show("draft attention_conv base_kernel", tuple(dict(DRF.named_parameters())["model.layers.0.attention_conv.base_kernel"].shape))
show("draft total params", f"{sum(p.numel() for _, p in DRF.named_parameters()):,}")

rule("=")
print("E. THE draft/target compatibility guard, evaluated against Orca")
rule("=")
VC_PAIR = VllmConfig(model_config=TMC, speculative_config=SC, device_config=DeviceConfig("cpu"))
with set_current_vllm_config(VC_PAIR):
    EXPECTED = _get_dflash_fc_input_size(VC_PAIR)
AUX = get_eagle3_aux_layers_from_config(VC_PAIR.speculative_config)
MASK = (SC.draft_model_config.hf_config.dflash_config or {}).get("mask_token_id")
show("aux layers requested from target", AUX)
show("len(aux) x target hidden_size", f"{len(AUX or ())} x {TEXT.hidden_size}")
show("fc input width the drafter demands", EXPECTED)
show("gptq3c fc logical width (checkpoint)", "25600  (qweight [2400, 5120] x 5 taps)")
show("mask_token_id < target vocab", f"{MASK} < {TEXT.vocab_size} -> {MASK < TEXT.vocab_size}")
show("max(aux) < target num_hidden_layers", f"{max(AUX)} < {TEXT.num_hidden_layers} -> {max(AUX) < TEXT.num_hidden_layers}")
print(f"\n  fc guard: {'PASS' if EXPECTED == 25600 else 'FAIL'} for Orca (hidden 5120, 64 layers)")
