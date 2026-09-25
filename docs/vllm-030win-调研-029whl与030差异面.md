# vLLM 0.30-on-Windows 调研：0.29 whl 溯源 与 0.30 差异面量化

> 调研日期：2026-09-25。方法：解包比对 + SHA-256 哈希（行尾归一化，规避 CRLF 干扰）+ 版本指纹法 + GitHub API/raw 探测官方 tag。全程只读，未安装/未构建/未运行 vLLM。临时目录 `G:\qwen3.8model\_tmp_whl29` 用后已删除。
> 素材：`vllm-0.29.0+cu132-cp312-cp312-win_amd64.whl`、`Lvllm-main.zip`、`vllm-8275b36b….zip`、`vllm-windows-build-master.zip`；对照：`G:\qwen3.8model\vllm-windows-0.27.1`、`G:\qwen3.8model\nvfp4-win-experiment\vllm-overlay\vllm`。为锚定官方水位，额外下载了官方 v0.29.0 / v0.30.0 源码 zip（codeload）做比对基准。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 0.29 whl 是谁构建的 | **SystemPanic/vllm-windows**（GitHub fork of vllm-project/vllm）构建的 Windows 原生 whl；= **官方 vllm v0.29.0 tag 源码（.py 98.9% 字节级一致）+ 407 个自有扩展文件 + 26 个 Windows 修改文件 + 12 个预编译 .pyd** |
| Lvllm-main.zip 是什么 | guqiong96 的 **LvLLM** fork（CPU+GPU NUMA 双并行 / lk_moe），源码代际 = **vLLM 0.27.1 时代**（与官方 v0.27.1 指纹逐项吻合） |
| snap8275 是什么 | **ch2lab/vllm** fork 在 commit `8275b36b`（2026-08-18，"feat: NVFP4 XQA decode + PP layer partition + MTP2 config"）的快照，代际 = **v0.28.0 发布前 8 天的 main dev 水位** |
| 哪个 zip ≈ 0.30 | **都不是**。真 0.30 是官方 v0.30.0 tag（2026-09-22 发布），两个 zip 分别是 0.27.1 代与 0.28/0.29 之交 |
| "humming 换 LayerConfig 是 0.30 特征" | **不成立**：官方 0.28/0.29/0.30/main 的 humming.py 全是 `humming_forward` 风格且均无 `LayerConfig`（该 API 切换发生在 0.27.1→0.28.0） |
| 0.29→0.30 移植面 | 官方 .py 层面 **815 改 + 179 增 + 12 删 ≈ 1006 文件、+113,518 / −26,006 行**；另有 12 个 Windows 补丁文件与 0.30 改动正面冲突 |
| 用户自研 vs whl | dflash/dflash2/humming/kv_cache_coordinator/gc_utils **上游全部已有**（用户是扩展版需 diff 合并）；`multi_turboquant_kv.py`、`reference_nvfp4.py`、`torch_wna16.py`、`inc/*` **上游没有，必须搬** |
| 最大摩擦点 | **Python ABI：whl 是 cp312，用户现役 venv 是 3.13（cp313）**，whl 无法安装；其次是 torch pin 降到 2.11.0+cu130、transformers 5.5→5.10 大跳 |

---

## B1. 0.29 whl 完整溯源

### B1.1 元数据层

- **WHEEL**：`Generator: setuptools (80.10.2)`、`Root-Is-Purelib: false`、`Tag: cp312-cp312-win_amd64`（单 ABI，只编了 CPython 3.12）。
- **dist-info 内容**：`METADATA / RECORD / WHEEL / entry_points.txt / licenses / top_level.txt` —— **没有 INSTALLER、没有 direct_url.json**（wheel 本身不带 INSTALLER 属正常；direct_url.json 是 pip 装直链时才写入的，此处无）→ 无法从 dist-info 直接看安装来源，溯源靠内容哈希 + METADATA 依赖 URL。
- **RECORD**：5119 行 = 5119 个文件，自洽。
- **METADATA 关键行**（版本/依赖本身就是溯源证据）：
  - `Name: vllm`、`Version: 0.29.0+cu132`；`Requires-Python: <3.15,>=3.10`，classifiers 列了 3.10–3.14（**误导**：实际只编了 cp312）。
  - `transformers>=5.10.4`、`huggingface_hub>=1.28.0` —— 与**官方 v0.29.0 的 requirements/common.txt 完全同水位**（0.30 是 5.10.4/1.31.0，0.28 是 5.5.3/1.27.0）。
  - 大量 `sys_platform` 分支暴露 Windows 移植层：win32 → `torch==2.11.0+cu130`、`torchaudio==2.11.0+cu130`、`torchvision==0.26.0+cu130`、`flashinfer-python@ https://github.com/SystemPanic/flashinfer-windows/releases/download/v0.6.11.post3/…whl`、`humming-kernels[cu13]@ git+https://github.com/SystemPanic/humming-windows.git@v0.1.15`、`triton-windows==3.6.0.post26`、`winloop`、`xformers==0.0.35`、`nvidia-cudnn-frontend<1.19.0,>=1.13.0`、`tilelang==0.1.10`、`apache-tvm-ffi>=0.1.13`；非 win → `torch==2.13.0`、`flashinfer-python==0.6.18`、`humming-kernels[cu13]==0.1.12`、`tilelang==0.1.12` 等。
  - **依赖里的 GitHub 账号 `SystemPanic`** 即构建者身份线索。

### B1.2 哈希比对（whl 全部 2719 个 .py vs 各树，行尾归一化）

| 对照树 | 共有路径 | 内容相同 | 内容不同 | 仅 whl 有 | 仅对照树有 | 说明 |
|---|---|---|---|---|---|---|
| **官方 v0.29.0**（codeload zip） | 2312 | **2286（98.9%）** | **26** | **407** | **0** | 官方每个 .py 都在 whl 里且几乎全部字节一致 |
| snap8275（ch2lab） | 2244 | 1381 | 863 | 475 | 4 | 4 个独有文件正是其 commit 新增 |
| Lvllm-main | 2159 | 1096 | 1063 | 560 | 1 | 完全对不上 |
| 本地 0.27.1 树 | 2160 | 1106 | 1054 | 559 | 0 | — |
| 用户 overlay | 2305 | 1202 | 1103 | 414 | 7 | — |

> 注：whl 内 .py 为 CRLF 行尾（比 LF 版每个文件大出若干字节），原始字节哈希偏低；表中为 `\r\n→\n` 归一化后口径。

**判定（对应选项）：(d) 某树 + Windows 补丁层，且"某树"= 官方 vllm v0.29.0 tag 源码。**
排除 (b)（与 Lvllm-main 仅 40.3% 相同）、排除 (c)（与 snap8275 仅 50.8% 相同且缺 dflash2 等 475 文件）、也排除"纯 (a)"（官方树之外还有 407 个自有文件 + 26 处 Windows 修改 + 预编译二进制）。

### B1.3 差异构成拆解

1. **26 个"官方同路径但内容不同"文件 = Windows 补丁层**：`collect_env.py、envs.py、distributed/parallel_state.py、distributed/utils.py、distributed/device_communicators/shm_broadcast.py、entrypoints/cli/{launch,openai,serve}.py、entrypoints/grpc_server.py、entrypoints/openai/api_server.py、entrypoints/launchers/api_server/entry.py、entrypoints/launchers/dp_supervisor.py、benchmarks/{throughput,sweep/server}.py、compilation/{compiler_interface,decorators}.py、model_executor/{warmup/kernel_warmup.py, layers/fused_moe/oracle/mxfp4.py, layers/quantization/mxfp4.py}、utils/system_utils.py、v1/{utils.py, engine/utils.py, executor/multiproc_executor.py, attention/backends/flashinfer.py, worker/gpu_worker.py, worker/gpu/sample/states.py}`。其中 22 个含 `win32/Windows` 字样。
2. **407 个"官方完全没有"的文件 = SystemPanic/vllm-windows 自有扩展层（记作 X）**，按目录：`third_party/triton_kernels`（51）、`vllm_flash_attn/cute`（51）、`third_party/fmha_sm100`（48）、`third_party/tml_fa4`（40，Flash-Attention-4 CuTe 实现）、`model_executor/layers`（41）、`model_executor/models`（37）、`entrypoints/serve/{cache,disagg,render,rlhf,rpc,sleep}`（21）、`entrypoints/openai/{engine,generate,generative_scoring}`（17）、`third_party/deep_gemm`（17）、`distributed/kv_transfer/.../p2p`（4，P2P NCCL KV 传输）、`entrypoints/sagemaker`（2）、`beam_search.py、_tilelang_ops.py、grpc/vllm_engine_pb2*.py、kernels/xpu_ops.py` 等。**官方 main 与 v0.30.1rc0 上抽查这些路径全部 404** → 不是未来上游代码，是 fork 自有。其中 `third_party/fmha_sm100/jit.py` 头部写着 `Copyright (c) 2026 MiniMax`、缓存目录 `~/.cache/minfer/` → 这套 kernel 栈来自 **MiniMax/Kimi-K3/DeepSeek-V4 生态**（KDA、NVFP4-KV、FA4 CuTe）。
3. **代码内溯源注释**（3 处）：`# VLLM_WINDOWS_MULTIPROCESS_CACHE_ISOLATION SystemPanic/vllm-windows/issues/85`（multiproc_executor.py / compiler_interface.py / decorators.py）。
4. **GitHub 谱系**：`SystemPanic/vllm-windows` 是 `vllm-project/vllm` 的 fork（API 确认 parent=source=vllm-project/vllm），同账号维护 `flashinfer-windows、humming-windows、DeepGEMM-windows、qutlass、nccl-windows、cutlass、fastsafetensors-windows、tvm-ffi、xgrammar` 等配套 Windows 化仓库 —— 与 METADATA 的依赖 URL 一一对应。

### B1.4 vllm-windows-build-master.zip 是什么

这是 **aivrar/vllm-windows-build**——另一条（更早的）Windows 构建线，**不是本 whl 的构建器**，但它是用户现役 0.27.1 底座的来源。清单（68 文件）要点：

- `vllm-windows-v2.patch … vllm-windows-v10.patch`：补丁链，`PATCHES.md` 声明 **v10 = 对上游 vLLM v0.27.1 的当前 Windows 增量**：167,335 字节 unified diff、约 72 文件、+1,900/−287；实测其文件清单含 `CMakeLists.txt、setup.py、csrc/**（含 csrc/libtorch_stable/ 大量 .cu）、rust/**、requirements/*` 及约 25 个 `vllm/**.py`（envs.py、parallel_state.py、cli/launch、cli/serve、grpc_server、api_server、dp_supervisor、kv_offload 系列、sample/states.py…）。
- 构建方式（PATCHES.md/README）：`git checkout v0.27.1 && git apply vllm-windows-v10.patch` → MSVC 2022（19.43）+ CUDA 13.0u2 + Ninja，`TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9;12.0`，Python 3.13.11（**cp313**）、PyTorch 2.13.0+cu130、triton-windows 3.7.1.post27 → 编译后用自带 `build_wheel.py` 把安装树打成 whl（另有 `assemble_wheel_cu128_v0.2x.py` 历史版）。
- 其它：`turboquant/`（Multi-TurboQuant 6 法 + 上游 TurboQuant 4 变体，另有 `patches/multi_turboquant_kv.py`——注意：**用户 overlay 里的 `multi_turboquant_kv.py` 与它同名，应源自这条线**）、`engine_dispatcher.py、vllm_launcher.py、verify_artifact.py、tests/`（含 windows kv offload/tiering 测试）、`docs/`（v0.27.1 build record、release notes、turboquant 说明）。
- 交叉验证：本地 `vllm-windows-0.27.1` 树与官方 v0.27.1 的 `envs.py/collect_env.py/parallel_state.py` 哈希均不同（略大）→ **该树 = 官方 0.27.1 + Windows 补丁（v10 水位）**，与 PATCHES.md 相符。
- 两条 Windows 线的改动面高度同构（都改 envs/collect_env/shm_broadcast/parallel_state/cli/* 等同一批文件），说明 Windows 移植的痛点文件是稳定的。

### B1.5 B1 结论

`vllm-0.29.0+cu132-cp312-cp312-win_amd64.whl` = **SystemPanic/vllm-windows 构建**：以**官方 vLLM v0.29.0 tag** 为 Python 源码底（98.9% 字节一致、官方文件一个不少），叠加 ① 407 个自有扩展文件（MiniMax 系 kernel 栈 + disagg/generate/sagemaker 服务栈 + P2P KV 等）、② 26 个 Windows 修改文件、③ 12 个预编译 .pyd、④ 1697 个 JIT 用 CUDA 头/源、⑤ pyproject 改写为 win32 依赖分支。**不是** Lvllm-main、**不是** snap8275。

---

## B2. 两个源码 zip 的版本代际判定

### B2.1 官方 tag 地标（GitHub raw 探测）

| 指纹 | v0.27.1（08-11 发布） | v0.28.0（08-26） | v0.29.0（09-09） | v0.30.0（09-22） |
|---|---|---|---|---|
| `qwen3_dflash.py` | 有（34,182B） | 有（36,029B） | 有（35,309B） | 有（35,663B） |
| `qwen3_dflash2.py` | **无** | 有（9,968B） | 有（9,968B） | 有（9,968B，0.28→0.30 一字未改） |
| `model_executor/hw_agnostic/` | **无** | 有 | 有 | 有 |
| `entrypoints/serve/middleware/`、`entrypoints/launchers/` | **无** | **无** | 有 | 有 |
| `utils/gc_utils.py` | 4,970B | 4,970B | 4,970B | 5,749B |
| `v1/core/kv_cache_coordinator.py` | 36,811B | 40,311B | 41,536B | 45,268B |
| `humming.py` | 32,915B（无 humming_forward） | 32,390B（humming_forward） | 32,390B | 32,390B |
| requirements | transformers>=5.5.3，无 hf_hub | 5.5.3 / hf_hub>=1.27.0 | **5.10.4 / 1.28.0** | 5.10.4 / **1.31.0** |

### B2.2 Lvllm-main.zip = LvLLM（guqiong96），代际 **0.27.1**

- README 自述："LvLLM is a special extension of vLLM… GPU parallel + NUMA parallel… lk_moe"，链接 `github.com/guqiong96/Lsglang、Lvllmds4`；`pyproject.toml` 中 `name = "lvllm"`；`config.yaml` 是 guqiong 本机 DeepSeek-V4-Flash-0731 启动配置（跑题外证据）。
- 代际指纹逐项 == 官方 v0.27.1：`qwen3_dflash.py` 34,182B（与官方 v0.27.1 同）、`humming.py` 32,915B（同）、`transformers>=5.5.3` 且无 `huggingface_hub` pin（同）、**没有** `qwen3_dflash2.py`（0.28 起才有）、**没有** `hw_agnostic/`（0.28 起）、**没有** `serve/middleware`（0.29 起）。
- 与本地 `vllm-windows-0.27.1` 树直接哈希：2159 共有、**2119 相同、仅 40 个 .py 不同**（差异集中在 fused_moe/lk_moe 相关 + 一批 Windows 文件）→ 同代无疑。
- **结论：0.27.1 代 fork，绝非 0.29/0.30。**

### B2.3 vllm-8275b36b….zip = ch2lab/vllm 快照，代际 **0.28.0 发布前的 main dev 水位（0.28/0.29 之交）**

- commit 实锤（GitHub API）：`8275b36b9e734d8437893707aa79d216dfb1711c`，标题 **"feat: NVFP4 XQA decode + PP layer partition + MTP2 config"**，作者 ch2lab（guo2017@guet.edu.cn），**2026-08-18**，+431/−32，父 `1fa8615f`，正文提到 `pp_spec_broadcast.py、quantized_draft_embedding.py、Qwen3.5 MTP、run2.sh VLLM_PP_LAYER_PARTITION=38,26`。
- 归属：`ch2lab/vllm` 是 vllm-project/vllm 的 fork（API 确认）。
- 文件级证据：zip 里 4 个"全网独有"文件正是该 commit 点名新增的 `v1/worker/pp_spec_broadcast.py、model_executor/layers/quantized_draft_embedding.py`，加 `models/kimi_k3/nvidia/ops/cute_dsl/gemm_rs.py、v1/attention/ops/dcp_utils.py`。
- 代际指纹：requirements == **v0.28.0 水位**（transformers>=5.5.3 + hf_hub>=1.27.0）；有 `hw_agnostic/`（0.28+）；有 `serve/middleware` 5 文件且与官方 v0.29.0 **字节一致**（说明该重构 8/18 前已进 main，不是 fork 自造）；**缺 `qwen3_dflash2.py`**（v0.28.0 tag 已有 → 该文件在 8/18 之后才落到 main/被合入 0.28 发布分支，或被 fork 移除换成了 MTP2 方案）。
- **结论：2026-08-18 的 main dev 快照 + ch2lab 研究改动（NVFP4 XQA / PP 广播 / MTP2），代际介于 v0.28.0 与 v0.29.0 之间，绝非 0.30。**

### B2.4 "LayerConfig 特征"辨伪与最终结论

- 传闻"0.30 特征 = LayerConfig 驱动的 humming_forward"**不成立**：官方 v0.28.0 / v0.29.0 / v0.30.0 / main（含 v0.30.1rc0）的 `humming.py` 全部为 `humming_forward` 风格、**全部没有 `LayerConfig`**（类名始终 HummingConfig/HummingLayerQuantizationConfig/HummingLinearMethod/HummingMoEMethod）。`humming_forward` 是 0.27.1→0.28.0 的切换。真正的代际指纹应看 requirements 水位与文件集（本报告即如此做）。
- **最终结论：两个 zip 都不是 0.30 源码。** 真 0.30 = 官方 v0.30.0 tag（2026-09-22 发布，hf_hub 下限升至 1.31.0）。为量化 B3-①，本调研下载了官方 v0.29.0 / v0.30.0 源码 zip 作为基准。

---

## B3. 三个差异面

### B3-① 0.29 whl → 0.30 的移植面（以官方 v0.30.0 树量化）

先把 whl 分解：whl = 官方 0.29.0（98.9% 一致）+ X 407 文件 + 26 个 win 文件。因此"把 0.30 新功能移植到 whl"≈ **官方 0.29→0.30 增量** + **26 个 win 文件的合并冲突** + **X 层对 0.30 新接口的适配**。

**官方 v0.29.0 → v0.30.0（仅 .py）**：新增 **179**、删除 **12**、修改 **815**、未动 1,485；**+113,518 行 / −26,006 行**。

按目录聚合（变动行合计 top，全表太长只列主干）：

| 目录 | 修改 | 新增 | 删除 | +行 | −行 | 变动合计 | 性质 |
|---|---|---|---|---|---|---|---|
| model_executor/layers | 131 | 10 | 4 | 9,779 | 4,940 | **14,719** | 量化/MoE/linear 大改 |
| models/deepseek_v4 | 25 | 13 | 0 | 10,128 | 3,786 | 13,914 | DeepSeek-V4 演进 |
| models/deepseek_v41 | 0 | 29 | 0 | 12,844 | 0 | 12,844 | **新模型** |
| v1/attention | 43 | 7 | 1 | 8,114 | 2,370 | 10,484 | attention 后端/ops |
| model_executor/models | 142 | 5 | 0 | 8,660 | 1,562 | 10,222 | 模型基架 |
| models/glm5next | 0 | 21 | 0 | 9,738 | 0 | 9,738 | **新模型** |
| distributed/ec_transfer | 9 | 21 | 0 | 6,428 | 76 | 6,504 | 弹性传输（mooncake/cpu control 等） |
| v1/worker | 61 | 3 | 1 | 4,276 | 1,708 | 5,984 | worker/模型执行 |
| distributed/kv_transfer | 33 | 4 | 0 | 4,832 | 1,007 | 5,839 | KV 传输 |
| models/qwen4_exp | 14 | 3 | 0 | 3,215 | 1,706 | 4,921 | 新模型 |
| model_executor/kernels | 35 | 5 | 0 | 3,644 | 944 | 4,588 | kernel 层 |
| v1/hisparse | 0 | 7 | 0 | 2,791 | 0 | 2,791 | **新子系统**（稀疏注意力） |
| model_executor/warmup | 7 | 7 | 3 | 2,142 | 553 | 2,695 | warmup 重构 |
| v1/kv_offload | 22 | 2 | 0 | 1,334 | 477 | 1,811 | KV offload |
| v1/watermarking | 0 | 10 | 0 | 1,067 | 0 | 1,067 | **新子系统**（水印） |
| config/engram.py 等新 config | — | 2 | 0 | ~204 | 0 | ~204 | engram/watermarking 配置 |

- **0.30 新增方向**：新模型（deepseek_v41、glm5next、qwen4_exp、kimi_k3/minimax_m3/hy_v4 扩展）、hisparse 稀疏注意力、watermarking、engram、ec_transfer/mooncake 分离式传输、kv_transfer 增强、量化层（layers 131 文件改动）。
- **whl vs 官方 v0.30.0 直接比**：共有 2,300、相同 1,471、不同 829、仅 whl 419（X 层 + 被删 12 文件）、仅 0.30 有 179。
- **Windows 补丁冲突面（重打 v10/26 文件时的热点）**：26 个 win 文件中 **12 个在 0.30 也被改**：`benchmarks/throughput.py、distributed/parallel_state.py、entrypoints/cli/openai.py、entrypoints/cli/serve.py、entrypoints/grpc_server.py、envs.py、model_executor/layers/fused_moe/oracle/mxfp4.py、model_executor/layers/quantization/mxfp4.py、model_executor/warmup/kernel_warmup.py、v1/attention/backends/flashinfer.py、v1/utils.py、v1/worker/gpu_worker.py`。
- **X 层风险**：0.30 大改 `model_executor/layers`（MoE/量化 API）、`v1/kv_cache_interface.py`（+262 行）、`config/*`；X 的 kernel 栈（fmha_sm100/triton_kernels/tml_fa4/qutlass 路径）与服务栈（disagg/generate）需要跟着适配，否则会咬到接口变更。

### B3-② 0.29 whl vs 用户自研 overlay：重叠度与逐域结论

先定底座：**overlay = 0.27.1 树（2,160 个 .py，2,088 个与 `vllm-windows-0.27.1` 字节一致）+ 152 个新增文件 + 72 个改动文件**。overlay 也携带 `third_party/fmha_sm100、third_party/triton_kernels、vllm_flash_attn/cute` 这套 MiniMax 系 kernel 栈（与 whl 的 X 层**同名同路径**——同源社区 kernel 栈，直接互相覆盖会冲突，见 B4）。

逐功能域（行数/哈希已逐树核对）：

| 功能域 | 0.29 whl 状态 | 用户 overlay 状态 | 结论 |
|---|---|---|---|
| **dflash 投机解码**（`v1/worker/gpu/spec_decode/dflash/` + `qwen3_dflash.py`） | 原生存在，== 官方 0.29（speculator 766 行；qwen3_dflash 878 行；0.30 微调至 767/886 行） | dflash/ 4 文件齐；speculator.py **919 行**（+153 自研）；qwen3_dflash.py **1,184 行**（+306 自研）；cudagraph/utils 为 0.27.1 原版 | **上游已有**。要搬的是用户增量（~460 行改动），并注意 0.30 也动了这两个文件 |
| **qwen3_dflash2 模型** | 原生存在（0.28 起；0.28/0.29/0.30 三版完全一致，290 行） | **449 行扩展版**（triton grouped-conv kernel、VocabParallelEmbedding 等），0.27.1 树里本来没有此文件（用户移植+扩展） | **上游已有基线**。用户扩展版（+159 行核心差异）需 diff 合并后搬 |
| **humming 量化** | 原生存在（humming.py 815 行、utils/humming 59 行、humming_utils 1,104 行，0.29==0.30 一字不差） | humming.py 841 行、humming_utils 1,060 行、utils/humming 61 行，另改 `model_executor/kernels/linear/{mixed_precision,mxfp4,mxfp8,nvfp4,scaled_mm}/humming.py` 5 个 kernel 适配文件 | **上游已有**。用户增量集中在 humming kernel 适配层；0.29→0.30 该域零变化，合并成本低 |
| **nvfp4 量化** | 有 compressed_tensors nvfp4/mxfp4 路径 + turboquant（config.py 五树一致，上游功能） | **自研独有**：`v1/attention/reference_nvfp4.py`（193 行）、`model_executor/kernels/linear/mixed_precision/torch_wna16.py`（139 行）、`quantization/inc/{calib.py, schemes/inc_embedding.py, schemes/inc_fp8.py}`+inc 配置解析；全树 120 文件涉 nvfp4 | **上游没有（必须搬）**：reference_nvfp4、torch_wna16、inc/* 三块；其余 nvfp4 相关为对已有路径的修改需 diff |
| **kv_cache_coordinator** | 原生存在（996 行 == 官方 0.29；**0.30 增至 1,072 行**） | 931 行（0.27.1 基 903 + 28 行自研） | **上游已有**。迁移时三方合并：0.29 基线 + 0.30 新增 76 行 + 用户 28 行 |
| **gc_utils** | 原生存在（151 行；0.27.1==0.29 一字不差；**0.30 增至 179 行**） | **180 行自研扩展** | **上游已有**。对比用户 180 行与 0.30 179 行的扩展方向（很可能部分同源），择优合并 |
| **KV 量化（multi_turboquant_kv）** | **没有** `v1/attention/ops/multi_turboquant_kv.py`（0.29/0.30 均无） | 295 行（疑似源自 aivrar Multi-TurboQuant 补丁，winbuild 仓有同名 patch） | **上游没有（必须搬）**，连带其 triton kernel（triton_turboquant_*） |
| **GSQ 3bit** | 全树无 "gsq" 字样 | 仅 1 处注释提及（`v1/spec_decode/llm_base_proposer.py: "GSQ with weight_packed/weight_scale/weight_shape"`）；3bit 相关散见 turboquant/config.py、qwen3_dflash.py、triton_turboquant_*（6 文件） | **上游没有独立 GSQ 模块**；其实现载体疑似在 turboquant/qwen3_dflash/llm_base_proposer 内，**需用户确认载体文件**（本调研未发现独立 gsq 实现文件） |

overlay 的 72 个改动文件集中在：humming kernel（5）、inc 量化（3+）、dflash/spec_decode（4）、kv_offload 系（5）、kv_cache_coordinator/kv_utils（2）、gc_utils、sample/gumbel、model_runner、block_table、logits_processor、qwen3_5/qwen3_5_mtp、以及 collect_env/envs/parallel_state/shm_broadcast/cli/* 等 Windows 基建的二次修改（与两条 Windows 线的痛点文件重合）。

### B3-③ 0.29 whl 的 Windows 使能面

**12 个 .pyd 清单与用途**（代码引用核对）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `vllm/_C.pyd` | 2.4 MB | 核心 C 扩展入口 |
| `vllm/_C_stable_libtorch.pyd` | **55 MB** | stable-libtorch ABI 的 CUDA 算子主体（`platforms/cuda.py` 导入） |
| `vllm/_moe_C.pyd` | 44 MB | fused MoE CUDA kernels |
| `vllm/_moe_C_stable_libtorch.pyd` | 46 MB | 同上，stable-libtorch ABI 版 |
| `vllm/_qutlass_C.pyd` + `_qutlass_C.cp312-win_amd64.pyd` | 2.3+3.0 MB | QuTLASS 量化 GEMM（NVFP4 路径 `linear_qutlass_nvfp4.py/fp_quant.py` 使用；源 SystemPanic/qutlass fork） |
| `vllm/_flashkda_C.pyd` | 3.4 MB | Kimi-K3 KDA attention prefill（`models/kimi_k3/nvidia/kda.py`，`--kda-prefill-backend flashkda`） |
| `vllm/fs_io_C.pyd` | 99 KB | KV offload 文件系统层 I/O（`v1/kv_offload/tiering/fs/io.py`，batch_lookup 等） |
| `vllm/spinloop.pyd` | 37 KB | 高效自旋轮询（`shm_broadcast.py`，`VLLM_USE_SPINLOOP_EXT`） |
| `vllm/cumem_allocator.pyd` | 260 KB | CUDA VMM/cuMem 分配器（`enable_cumem_allocator`） |
| `vllm/third_party/deep_gemm/_C.cp312-win_amd64.pyd` | 1.0 MB | DeepGEMM 预编译扩展 |
| `vllm/vllm_flash_attn/_vllm_fa2_C.pyd` | **379 MB** | FlashAttention2 fat binary（全 SM 架构打爆体积） |

**1,697 个 .h/.hpp/.cu/.cuh：是，给运行期 JIT 用**。两个来源：`vllm/third_party/deep_gemm/include`（827 个，CuTe DSL 头）与 `vllm/third_party/fmha_sm100/csrc`（870 个，MiniMax FMHA varlen kernel 源码）。`fmha_sm100/jit.py` 明示"per-variant lazy JIT compilation…cached to `~/.cache/minfer/fmha_sm100/`" → **运行时首次触发才编译并缓存，运行机需要 MSVC + CUDA toolkit**。

**flashinfer 不在 whl 里**（whl 顶层只有 `vllm/` 与 dist-info）：靠 pip 独立包——win32 装 `SystemPanic/flashinfer-windows` 的 v0.6.11.post3 直链 whl（METADATA 里是 URL 依赖），非 win 为 `flashinfer-python==0.6.18`。`humming-kernels` 同理走 `git+https://github.com/SystemPanic/humming-windows.git@v0.1.15`。

**torch 版本**：whl 名含 `+cu132`（构建工具链 CUDA 13.2 意象），但 METADATA 对 win32 **硬 pin `torch==2.11.0+cu130`**（torchvision 0.26.0+cu130、torchaudio 2.11.0+cu130），非 win 才是 torch==2.13.0。用户现役 **torch 2.13.0+cu130** → **冲突**：pip 解析会强制降级到 2.11.0+cu130。缓解线索：whl 带 `_C_stable_libtorch`（stable libtorch ABI），对 torch 小版本有一定容忍空间，可尝试 `--no-deps`/放宽 pin 验证，但不保证。cu132 与 cu130 的混搭属 CUDA minor 兼容范畴，一般可跑。

---

## B4. 迁移摩擦清单

| # | 摩擦点 | 说明与出处 |
|---|---|---|
| 1 | **Python ABI：cp312 vs 用户 venv 3.13（最大硬摩擦）** | whl tag `cp312-cp312-win_amd64`（WHEEL），pyd 命名带 cp312；用户现役 3.13 来自 aivrar 线（PATCHES.md：Python 3.13.11 cp313）。METADATA 的 `Requires-Python <3.15,>=3.10` + classifiers 3.10–3.14 **具误导性**，实际只能装进 3.12 环境。要么新建 3.12 venv，要么自建（成本极高） |
| 2 | **torch 降级** | METADATA win32 pin `torch==2.11.0+cu130`，用户现役 2.13.0+cu130；pip 会拉降级。stable-libtorch ABI 或可 `--no-deps` 硬扛，需验证（METADATA） |
| 3 | **transformers 5.5 → 5.10 大跳** | 0.29/0.30 需 `transformers>=5.10.4`（METADATA/官方 requirements）vs 0.27.1 线 `>=5.5.3`；用户改过的 `transformers_utils/configs/*`（arctic/cheers/fireredlid/flex_olmo/hunyuan_vl…）需按 5.10+ API 适配 |
| 4 | **huggingface_hub 抬升** | 0.29 要 `>=1.28.0`、0.30 要 `>=1.31.0`（官方 requirements），0.27.1 线无 pin，现役版本可能偏低 |
| 5 | **git/URL 依赖** | `flashinfer-python@ https://github.com/SystemPanic/flashinfer-windows/releases/...`、`humming-kernels@ git+https://github.com/SystemPanic/humming-windows.git@v0.1.15`（METADATA）——安装需 GitHub 网络 + 本机 git，离线/受限网络装不上 |
| 6 | **triton 版本倒挂** | whl pin `triton-windows==3.6.0.post26`，用户 0.27.1 线是 3.7.1.post27（PATCHES.md）；用户的 triton kernel（triton_turboquant_*、triton_kernels 栈）可能踩 3.6 API 差异 |
| 7 | **依赖树整体扰动** | xformers==0.0.35、numba==0.65.0、torchcodec>=0.14、PyNvVideoCodec==2.0.4、lm-format-enforcer==0.11.3、outlines_core==0.2.14、openai-harmony、mistral_common[image]、winloop（新事件循环库）等一串 pin 会重排现有 venv（METADATA） |
| 8 | **同名核栈互覆盖** | overlay 与 whl 的 X 层同含 `third_party/fmha_sm100、third_party/triton_kernels、vllm_flash_attn/cute`（同路径不同版本）；把 overlay 盖到 whl 上会互相覆盖/版本错乱，需要做文件级合并而不是整目录拷贝 |
| 9 | **运行期 JIT 依赖本机工具链** | fmha_sm100/deep_gemm 的 1,697 个 .cu/.h 首用时懒编译（`~/.cache/minfer/` 缓存），目标机必须有 MSVC + CUDA toolkit（fmha_sm100/jit.py） |
| 10 | **0.29→0.30 移植时的 win 补丁冲突** | 12 个 win 文件与 0.30 改动重叠（B3-① 列表）+ X 层 407 文件需适配 0.30 的 layers/kv_cache_interface/config 新接口 |
| 11 | **自编二进制需重编** | 用户若在 cp313 上自编过任何 .pyd（含 GSQ/量化 kernel），在 cp312 下全部失效需重编 |
| 12 | GSQ 3bit 载体不明 | 全部素材中未见独立 gsq 模块，仅注释级提及；迁移前需用户指认其实现文件（见 B3-②） |

---

## 附：关键数字备查

- whl：5,119 文件 = 2,719 .py + 12 .pyd + 1,697 .h/.hpp/.cu/.cuh + 其余（json/字体等）；RECORD 5,119 行自洽。
- 哈希口径：全部 SHA-256、`\r\n→\n` 归一化；whl 的 .py 为 CRLF。
- 官方 release 日期锚点：v0.27.1=2026-08-11、v0.28.0=2026-08-26、v0.29.0=2026-09-09、v0.30.0=2026-09-22。
- 0.29→0.30 总量：179 增 + 12 删 + 815 改 = 1,006 个 .py 变动，+113,518 / −26,006 行。
- 五树关键文件行数对照（whl==off29 处处相等，证明 whl 的 Python 层就是官方 0.29.0）：
  - `qwen3_dflash2.py`：whl/off29/off30 三者 **同为 290 行**；overlay 449 行。
  - `humming.py`：whl/off29/off30 **同为 815 行**；overlay 841 行、v0271 838 行。
  - `kv_cache_coordinator.py`：whl==off29 996 行；off30 1,072 行；overlay 931 行。
  - `gc_utils.py`：whl==off29==v0271 **151 行**；off30 179 行；overlay 180 行。
