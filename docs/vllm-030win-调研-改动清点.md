# vLLM 0.27.1 Windows 构建 — 自研改动清点报告

> 目的：评估迁移到 vLLM 0.29 / 0.30 的工作量，先精确清点「相对官方 0.27.1 我们到底改了多少东西」。
> 调研日期：2026-09-25。全程只读，未改动任何被调研文件。
> 三个对象：
> ① **基线锚点** `G:\qwen3.8model\vllm-win\dist-v0.27.1\vllm-0.27.1-cp313-cp313-win_amd64.whl` 内的 `vllm/`（下称 **whl**）
> ② **开发头** `G:\qwen3.8model\nvfp4-win-experiment\vllm-overlay\vllm`（下称 **overlay**，带 git）
> ③ **生产副本** `G:\qwen3.8model\vllm-win\Lib\site-packages\vllm`（下称 **site-packages**）
> 另有 ④ `G:\qwen3.8model\vllm-windows-0.27.1`（下称 **树**，Windows 构建源码树，无 git）与 ⑤ **官方 v0.27.1**（本次从 GitHub 下载原版做逐行比对）。

---

## 0. 方法与口径（可复核）

| 步骤 | 命令 / 方法 | 关键中间数字 |
|---|---|---|
| 解包 whl | `python zipfile` 解到 `G:\qwen3.8model\_tmp_dist271`（用完已删） | 3825 个 zip 条目；`vllm/` 下 **2304 个 .py** |
| 下载官方 0.27.1 | `curl -L https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v0.27.1` | 39,109,780 字节；`vllm/` 下 **2160 个 .py** |
| 逐文件比对 | 自写 python 脚本：读入后统一 `\r\n→\n`、`\r→\n`，SHA1 先筛同异，再用 `difflib.SequenceMatcher` 按行算**各自独有行数**（替换块两侧分别计入增/删） | 见下文各表 |
| git 口径 | `git diff --ignore-cr-at-eol df826b7`（本机 git 2.55 的 `git diff` 不认 `--strip-trailing-cr`，等价选项是 `--ignore-cr-at-eol`；实测**加与不加数字完全一致**，overlay 内文件本就是 LF，无 CRLF 虚高） | 27 文件 +1208/−195（vllm/） |
| mtime 判新旧 | `os.path.getmtime` 全量扫描 | 见 §2.4 |

> 说明：本报告的 `+ / −` 均为「A 相对 B 的独有行数」口径（不是 git 的 hunk 口径），因此与 git 的 1208/195 有约 10% 的计数差，属正常（替换块内偶有可对齐的行）。两套数字都在下面列出，便于交叉复核。
> 「未与上游逐行比对」的事项只有一处：whl 里 142 个 vendored 子模块 `.py`（见 §1.1）未与其各自上游仓库比对，仅按目录归属判定为上游代码。

---

## 1. A1：三方比对结果

### 1.1 先确认「基线」本身是什么（whl ↔ 官方 0.27.1 ↔ 树）

| 比对 | 文件集 | 改动 | 行数 |
|---|---|---|---|
| 树 `vllm/` ↔ 官方 0.27.1 | 2160 = 2160（0 增 0 删，完全同构） | 22 文件 | +295 / −83 |
| whl `vllm/` ↔ 官方 0.27.1 | whl 多 **144 个 .py** | 26 文件 | +603 / −59 |
| 树 ↔ whl（2160 同名文件） | — | **36 文件不同** | 树独有 338 / whl 独有 670 |

whl 那 144 个「多出来」的 .py 拆解（共 90,075 行）：

| 类别 | 文件数 | 行数 | 归属 |
|---|---|---|---|
| `vllm_flash_attn/cute|layers|ops` 53 + `third_party/fmha_sm100` 48 + `third_party/triton_kernels` 40 + `third_party/flashmla` 1 | 142 | 89,756 | **上游子模块代码**（官方 tarball 不含子模块；非自研，仅随 wheel 打包） |
| `vllm/v1/attention/ops/multi_turboquant_kv.py` | 1 | 295 | **自研**（Multi-TurboQuant KV 压缩集成，官方 404） |
| `vllm/_version.py` | 1 | 24 | 构建期生成 |

**两个重要结论**：

1. **树 ≠ whl**：两者的 Windows 修复是**两套不同实现**。例：`utils/system_utils.py` 树用 `os.kill(SIGKILL/SIGTERM)`、whl 用 `psutil` 的 `child.kill()`；`model_executor/model_loader/weight_utils.py` 的 Windows safetensors 读取器（绕开 mmap 提交内存，+175 行）**只在 whl 里有**，树里没有。所以「以谁为 0.27.1 基线」必须选 **whl**（生产实际运行的那份）。
2. **基线不是纯净官方 0.27.1**：whl 内已经含有 23 个 Windows 运行时修复文件 + 4 个 Multi-TurboQuant KV 功能文件（见 §4 总账）。若迁移目标是「官方 0.29/0.30 + 我们的全部能力」，这部分也要算进迁移面。

### 1.2 overlay 相对 whl 基线（核心数字）

| 项 | 文件数 | 行数 |
|---|---|---|
| 新增 .py | **8** | **+1,354** |
| 改动 .py | **37** | **+1,583 / −252** |
| 删除 .py | 0 | 0 |
| **合计** | **45** | **+2,937 / −252** |

新增 8 文件：`qwen3_dflash2.py`(449)、`spec_decode/dflash2/speculator.py`(233)、`spec_decode/dflash2/__init__.py`(6)、`v1/attention/reference_nvfp4.py`(193)、`inc/calib.py`(171)、`inc/schemes/inc_embedding.py`(90)、`inc/schemes/inc_fp8.py`(73)、`kernels/linear/mixed_precision/torch_wna16.py`(139)。

改动最大的 10 个文件（按增删合计排序）：`qwen3_dflash.py` +354/−25、`dflash/speculator.py` +232/−21、`v1/attention/backends/flashinfer.py` +137/−38、`humming_utils.py` +57/−26、`logits_processor.py` +82/−0、`inc_wna16_scheme.py` +65/−0、`inc/config_parser.py` +48/−17、`gumbel.py` +48/−0、`kv_cache_utils.py` +45/−5、`kernels/linear/scaled_mm/humming.py` +31/−23。其余 27 个改动文件每处增删均 ≤ 46 行（完整 37 文件明细可按 §0 方法复跑得到）。

**git 佐证分层**（overlay 仓库，基线提交 `df826b7 "baseline: vllm-overlay as of DFlash2 port session"`）：

- `df826b7` 快照相对 whl：**5 个新文件（共 1,020 行）+ 19 个改动文件 +754/−82** → 即「DFlash2 移植那一场」的成果在基线提交里已经落了一部分。
- `df826b7..HEAD` = **27 个提交**（任务描述写「约 29 个」，实际 27）：`git diff --ignore-cr-at-eol` → 27 个 vllm 文件 **+1,208/−195**；另有 `.serena/.gitignore`(+2) 与 `.serena/project.yml`(+169) 属 Serena 工具配置，非功能代码。
- 两者叠加（各自口径）≈ 我的逐文件口径 +2,937/−252，差异来自计数方法（§0）。

### 1.3 site-packages 相对 whl 基线

| 项 | 数字 |
|---|---|
| 改动文件 | **1 个**：`model_executor/models/qwen3_5.py`，**+2/−0** |
| 其余 2,303 个 .py | `\r` 归一后逐字节一致 |

那 +2 行是给 `Qwen3_5Model.__init__` 的 `VocabParallelEmbedding` 传 `quant_config=self.quant_config` 和 `prefix=f"{prefix}.embed_tokens"` —— 正是 `patch_vllm_qwen35_embedding.py`（175 行）打的补丁。**site-packages = whl 干净安装 + 1 个 embedding 补丁**。

### 1.4 overlay ↔ site-packages：谁新、谁是生产、要不要同步

| 比对 | 数字 |
|---|---|
| overlay 相对 site-packages | 新增 8 文件(1,354 行) + 改动 36 文件 **+1,581/−252**，合计 **+2,935/−252**（唯一互认的差异是 `qwen3_5.py` 那 +2 行，两边都有） |

- **生产在用 = site-packages**：`serve_qwen38_gsq.cmd` 调 `%ROOT%Scripts\vllm.exe`（`G:\qwen3.8model\vllm-win\Scripts\vllm.exe`）→ 该 venv 的 `Lib\site-packages\vllm`，serve `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ`，`--language-model-only`、`--max-model-len 8192`、port 8000。`serve_minicpm5_2b_dspark.cmd`（MiniCPM5-2B + DSpark 投机解码，32768 ctx、KV 池 1.2x、20 并发）走同一 venv。
- **开发头 = overlay**：mtime 证据 —— site-packages 全部 .py 停在 **2026-09-09 21:54~22:04**（whl 安装时刻，冻结）；overlay 的 .py 从 2026-09-09 21:54（从 whl 副本播种）一直改到 **2026-09-25 16:34**，最近改动是 `model_runner.py`(09-25 16:34，C1 计时)、`dflash/speculator.py`(16:07)、`envs.py`(14:50)、`gc_utils.py`+`cudagraph_utils.py`(11:53)。
- **结论**：overlay 明显更新，且内容是 site-packages 的**严格超集方向**（site-packages 只比 whl 多 2 行，而这 2 行 overlay 也有）；生产**完全不含** overlay 的 DFlash2 / INC 校准 / C1 tail-window / nvfp4 / GC 冻结等全部新工作。**迁移前必须先决定以 overlay 为唯一迁移源（推荐），并把 site-packages 视为「whl 基线 + 2 行补丁」的冻结生产副本**；两份目前不需要对齐内容，但要在迁移文档里写明生产链路 ≠ 开发头，避免拿 site-packages 去对 0.30 做 diff 而漏掉 2,900+ 行。

---

## 2. A2：改动按功能域分组（overlay 相对 whl 基线）

| # | 功能域 | 文件数 | +行 | −行 | 净增 | 一句话用途 |
|---|---|---|---|---|---|---|
| 1 | **GSQ/INC/humming 量化** | 16 | 813 | 129 | 684 | INC 量化配置/方案扩展（WNA16、fp8、embedding 码本量化）、GPTQ 校准 Hessian 采集（calib.py）、humming 2/3/5/6/7-bit 权重派发、torch WNA16 兜底内核、humming 迁到 0.30 layer-config API |
| 2 | **DFlash2 投机解码（新架构）** | 5 | 706 | 0 | 706 | DFlash2 draft 模型 + V2 speculator（候选选择器/码本）、架构注册（registry/config 判定 `_is_dflash2_draft`） |
| 3 | **DFlash(v1) + C1 tail-window + draft KV** | 6 | 647 | 51 | 596 | C1 尾窗：分块 prefill 期间推迟 draft 上下文 KV 写入、完成时一次性 precompute_and_store_context_kv；null-block 保护、关 FA aot_schedule、C1 计时日志、`VLLM_DFLASH_C1_TAIL_WINDOW` 开关 |
| 4 | **nvfp4 / flashinfer SM120** | 3 | 335 | 39 | 296 | SM120(消费级 Blackwell) 走 FA2 nvfp4-KV 路径、非因果 prefill 放行、XQA decode 放宽到 SM12x、nvfp4 参考实现 |
| 5 | **投机解码支撑** | 6 | 212 | 3 | 209 | draft 自带 quantization_config 时重建 quant_config（MTP 量化权重加载）、AOT 编译缓存键加 quant scheme 哈希、Gumbel-noised argmax 内核（draft/verifier 同噪声）、词表并行 top-k（flashinfer radix topk） |
| 6 | **KV cache 量化/异构池/块大小** | 4 | 130 | 18 | 112 | 异构池（跳量 draft 层 + 量化 target）块大小解析、`_dense_kv_rows` 读 inc GPTQ 布局、INT4 per-token-head 页尺寸折半、细粒度哈希命中与滑窗 manager 兼容、`prefix_cacheable` 声明 |
| 7 | **OpenAI 协议（长上下文）** | 2 | 39 | 2 | 37 | ≥128k 模型的输出预留只留 1 token，避免渲染期截断把本来放得下的 prompt 拒掉 |
| 8 | **GC / cudagraph** | 2 | 36 | 2 | 34 | 批量 CUDA graph capture 期间冻结并禁用 GC（防 Triton 内核被卸载导致图失效），`VLLM_ENABLE_CUDAGRAPH_GC=1` 可退出 |
| 9 | **hybrid block 实验开关** | 1 | 19 | 8 | 11 | `VLLM_EXPERIMENTAL_ALLOW_SMALL_HYBRID_BLOCK=1` 允许小 hybrid block（mamba/GDN 布局不保证，仅冒烟） |
| | **合计** | **45** | **2,937** | **252** | **2,685** | |

各域文件清单（路径相对 `vllm/`）：

1. **量化 16 文件**：新增 `inc/calib.py`、`inc/schemes/inc_embedding.py`、`inc/schemes/inc_fp8.py`、`kernels/linear/mixed_precision/torch_wna16.py`；改动 `inc/schemes/inc_wna16_scheme.py`(+65/−0)、`inc/config_parser.py`(+48/−17)、`inc/inc.py`(+44/−2)、`layers/quantization/utils/humming_utils.py`(+57/−26)、`layers/quantization/humming.py`(+22/−19)、`utils/humming.py`(+6)、`kernels/linear/scaled_mm/humming.py`(+31/−23)、`kernels/linear/mixed_precision/humming.py`(+15/−12)、`kernels/linear/mxfp4| mxfp8| nvfp4/humming.py`(各 +16/−10)、`kernels/linear/__init__.py`(+4)
2. **DFlash2 5 文件**：新增 `models/qwen3_dflash2.py`、`v1/worker/gpu/spec_decode/dflash2/{__init__,speculator}.py`；改动 `models/registry.py`(+1)、`config/vllm.py`(+17)
3. **DFlash v1 6 文件**：`models/qwen3_dflash.py`(+354/−25)、`v1/worker/gpu/spec_decode/dflash/speculator.py`(+232/−21)、`v1/worker/gpu/spec_decode/speculator.py`(+38/−5)、`v1/worker/gpu/spec_decode/__init__.py`(+6)、`envs.py`(+8)、`v1/worker/gpu/model_runner.py`(+9)
4. **nvfp4 3 文件**：`v1/attention/backends/flashinfer.py`(+137/−38)、新增 `v1/attention/reference_nvfp4.py`、`utils/flashinfer.py`(+5/−1)
5. **投机解码支撑 6 文件**：`v1/spec_decode/llm_base_proposer.py`(+40)、`compilation/caching.py`(+36)、`v1/worker/gpu/sample/gumbel.py`(+48)、`layers/logits_processor.py`(+82)、`models/qwen3_5_mtp.py`(+4/−3)、`models/qwen3_5.py`(+2)
6. **KV 4 文件**：`v1/core/kv_cache_utils.py`(+45/−5)、`v1/core/kv_cache_coordinator.py`(+28)、`v1/kv_cache_interface.py`(+25/−2)、`layers/attention/attention.py`(+32/−11)
7. **协议 2 文件**：`entrypoints/openai/chat_completion/protocol.py`(+22/−1)、`entrypoints/openai/completion/protocol.py`(+17/−1)
8. **GC 2 文件**：`utils/gc_utils.py`(+30/−1)、`v1/worker/gpu/cudagraph_utils.py`(+6/−1)
9. **hybrid 1 文件**：`platforms/interface.py`(+19/−8)

> 迁移线索：`gc_utils.py`、`spec_decode/speculator.py` 的 3 处改动提交信息明确写着「port from 0.30」（0.30 的 `gc_utils.py:96-121`、`speculator.py:189-200`、`speculator.py:567-622`），即这些在 0.30 已是上游形态，迁移时是**收敛**而非搬运。

---

## 3. A3：非 vllm 包的自研资产

### 3.1 `G:\qwen3.8model\vllm-win`（生产 venv 与运维资产）

| 资产 | 规模 | 用途 |
|---|---|---|
| `serve_qwen38_gsq.cmd` | 22 行 | 生产启动：`Scripts\vllm.exe` → site-packages；ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ，`--language-model-only`、`--max-model-len 8192`、`--tool-call-parser qwen3_coder`、HF 镜像与缓存根 `G:\qwen3.8model\hub` |
| `serve_minicpm5_2b_dspark.cmd` | 135 行 | MiniCPM5-2B + DSpark 投机解码，32768 ctx、KV 池按 1.2x 预算、20 并发、与 bge-m3 共卡（文件内含完整预算推导注释） |
| `stop_minicpm5.ps1` | 23 行 | 按命令行特征（而非端口）杀掉全部 vLLM 进程，避免 SO_REUSEPORT 残留占显存 |
| `patch_vllm_qwen35_embedding.py` | 175 行 | 给 `qwen3_5.py` 打量化 embedding 补丁（即 §1.3 的 +2 行），带 `.bak` 备份，可重放 |
| `README-WINDOWS.md` | — | 环境钉版说明（CPython 3.13.15 / torch 2.13.0+cu130 / Triton 3.7.1 / RTX 5070 Ti），并注明该 Windows 构建非官方支持 |
| `dist-v0.27.1\` | 2 个 whl | `vllm-0.27.1-cp313-cp313-win_amd64.whl`（自建）+ `multi_turboquant-0.1.0-py3-none-any.whl` |
| `flashinfer-inspect\` | 1 个 whl | `flashinfer_python-0.6.16.post3` 旧轮子留档（现装 0.6.18.post1） |
| `external\ffmpeg\src\ffmpeg-8.1.1.tar.xz` | 源码包 | ffmpeg 8.1.1 源码（配套 readme.txt） |
| venv 关键包 | — | torch 2.13.0+cu130、flashinfer 0.6.18.post1、`humming_kernels 0.0.0`、`multi_turboquant 0.1.0`（两个量化内核库都是独立 pip 包，vllm 侧只是集成方） |
| `research\`（**独立 git 仓库**，198 文件） | 见下 | 实验记录主库 |

`research\` 概览：`experiments\`（e0-baseline、e1-vision-enable、e5-nvfp4-kernel-landscape、e6-upstream-solutions、e7-prefill-cost-structure、e8-headroom-and-maxlen、e9-fp4-tile-repack、e10-prefill-chunk-size，每个含协议+结果）、`native-win\`（00-MASTER-PLAN、01-build-log、03-results、04-session-summary 与大量 bench 日志）、`src\`（bench_decode/bandwidth/maxlen/nvfp4_writer、nvfp4 探针、needle_bench、vision_smoke、GPU 显存监控 ps1 等 17 项）、`literature\`、`findings.md`、`research-log.md`、`research-state.yaml`、`to_human\night-report-2026-09-12.html`。git log 显示其研究主线是 nvfp4 KV 写路径、XQA 路由、prefill 成本结构、fp4 tile repack 等（E5–E10），与 overlay 的 nvfp4/SM120 改动同源。

### 3.2 `G:\qwen3.8model\vllm-windows-0.27.1`（构建树，**已与官方 v0.27.1 逐行比对**）

先说归属：该树的 `README.md` 是 **SystemPanic/vllm-windows**（README-WINDOWS.md 里称 `aivrar/vllm-windows-build`）项目的 README，而 `AGENTS.md`/`CLAUDE.md` 与官方逐字一致（0 差异）。即：**这棵树 = 第三方 Windows 移植项目的 v0.27.1 源码**，不是官方树。

Windows 构建侧改动（相对官方 v0.27.1）：

| 文件 | +/− | 内容 |
|---|---|---|
| `fix_cutlass_msvc.py`（新增，25 行） | — | 改 CUTLASS 头（`platform.h` 的 `__cplusplus` 判断加 `_MSC_VER`、`cuda_host_adapter.hpp` 的 `memsetDevice` 降级 `CUTLASS_HOST`）；由 `CMakeLists.txt:509` 与 `cmake/external_projects/vllm_flash_attn.cmake:85` 自动调用 |
| `fix_cuda_13_align.py`（新增，37 行） | — | CUDA 13 的 `CUtensorMap` 对齐从 128 降到 64（MSVC 下），按 README 需管理员手动跑一次 |
| `requirements/windows.txt`（新增，4 行） | — | Windows 专属依赖，setup.py 里 `requirements.extend(_read_requirements("windows.txt"))` |
| `setup.py` | +43/−10 | `IS_WINDOWS`/`IS_WSL` 判定、Windows/WSL 自动禁 FA3（编译器崩溃）、ccache 优先于 sccache、路径反斜杠→正斜杠喂 CMake、`nvtx3_dir`/`cublas.lib`、`nvcc.exe`、按 nvcc 实际版本取 CUDA 主次版本 |
| `CMakeLists.txt` | +63/−19 | MSVC `/Z7 /Zc:__cplusplus /Zc:preprocessor /DWIN32_LEAN_AND_MEAN`、CUTLASS WIN32 选项（cuBLAS 开关、警告与优化 flag）、调用 `fix_cutlass_msvc.py`、marlin 生成脚本改 `WORKING_DIRECTORY`、`_C_stable_libtorch` 链接 cuBLAS |
| `cmake/external_projects/`（5 文件） | +118/−31 | deepgemm(+12/−5)、flashkda(+6/−1)、qutlass(+65/−21)、triton_kernels(+7/−3)、vllm_flash_attn(+28/−1) 的 Windows 构建适配 |
| `csrc/`（33 文件） | +275/−192 | MSVC/CUDA13 兼容：`spinloop.cpp`(+23/−3)、`merge_attn_states.cu`(+44/−45)、`activation_kernels.cu`(+28/−21)、`selective_scan_fwd.cu`(+26/−14)、marlin/marlin_moe `generate_kernels.py`(各 +24/−18、+7/−11)、awq/nvfp4/gptq 内核等 |
| `README.md` | +103/−0 | Windows 构建/安装文档（SystemPanic 项目文档，非代码） |
| `vllm/`（22 个 .py） | +295/−83 | Windows 运行时使能：`utils/system_utils.py`(+63/−20)、`distributed/parallel_state.py`(+52/−17)、`v1/executor/multiproc_executor.py`(+18/−3)、`shm_broadcast.py`(+15/−4)、`entrypoints/cli/serve|launch|openai`、`api_server`、`grpc_server`、`dp_supervisor`、`compilation/compiler_interface.py`、`model_executor/warmup/kernel_warmup.py`、`v1/engine/utils.py`、`v1/utils.py`、`envs.py`(+5/−2) 等 |

**注意**：这 22 个 `vllm/*.py` 与 whl 里的 23 个 Windows 运行时修复是**两套不同实现**（§1.1），且树里的这 22 个**没有进入生产 whl**。迁移时应以 whl 那套为准，树里的这套可作为 0.30 Windows 移植的参考素材。

---

## 4. A4：总账

### 4.1 分层账（每层的参照系都写清楚）

| 层 | 参照系 | 文件数 | +行 | −行 | 性质 |
|---|---|---|---|---|---|
| L1a | whl 内 Windows 运行时修复（23 个改动文件，相对官方 0.27.1） | 23 | 544 | 59 | Windows 使能 |
| L1b | whl 内 Multi-TurboQuant KV（`multi_turboquant_kv.py` 295 行 + `triton_attn.py`+37 + `config/cache.py`+12 + `torch_utils.py`+10） | 4 | 355 | 0 | 功能改进 |
| L2a | 树内构建侧（setup.py、CMakeLists.txt、cmake 5、csrc 33、fix 脚本 2、requirements/windows.txt） | 43 | 565 | 252 | Windows 使能 |
| L2b | 树内文档 `README.md` | 1 | 103 | 0 | 文档 |
| L2c | 树内 `vllm/` Windows 运行时（另一套实现，未进生产） | 22 | 295 | 83 | Windows 使能 |
| L3 | **overlay 相对 whl（全部功能开发）** | **45** | **2,937** | **252** | 功能改进 |
| L4 | site-packages 的 `qwen3_5.py` +2 行 | — | （与 L3 同一改动，不重复计） | | 功能改进 |

### 4.2 汇总

| 类别 | 文件数 | +行 | −行 |
|---|---|---|---|
| **功能改进**（L1b + L3） | **49** | **+3,292** | **−252** |
| **Windows 构建/运行使能**（L1a + L2a + L2c，含文档 L2b 则 +103） | **88**（+1 文档 = 89） | **+1,404**（+103 = 1,507） | **−394** |
| **自研改动合计**（相对官方 0.27.1） | **137**（含文档 138） | **+4,696**（含文档 4,799） | **−646** |

其中「生产链路实际运行」的自研改动 = L1a + L1b + L3 前的 qwen3_5 补丁 = 27 文件 +899/−59；「开发头 overlay 新增」= L3 的 45 文件 +2,937/−252。

**不计入自研**：142 个 vendored 子模块 `.py`（89,756 行，上游 vllm-flash-attn / fmha_sm100 / triton_kernels / flashmla 代码）与 `_version.py`（24 行，构建生成）。

### 4.3 分组排行（按净增行）

1. GSQ/INC/humming 量化 16 文件 +813/−129
2. DFlash2 新架构 5 文件 +706/−0
3. DFlash(v1)/C1 tail-window 6 文件 +647/−51
4. nvfp4/flashinfer SM120 3 文件 +335/−39
5. 投机解码支撑 6 文件 +212/−3
6. KV cache 量化/异构池 4 文件 +130/−18
7. OpenAI 协议 2 文件 +39/−2
8. GC/cudagraph 2 文件 +36/−2
9. hybrid block 开关 1 文件 +19/−8

---

## 5. 迁移提示（基于清点事实）

- **迁移源应选 overlay**：45 文件 +2,937/−252 的功能开发全部在 overlay，生产 site-packages 只有 whl + 2 行。
- **overlay 里已有 3 处是从 0.30 反向移植的**（gc 冻结、FA aot_schedule 关闭、draft KV null-block 保护），到 0.30 是「去掉本地补丁、改用上游」。
- **量化域（16 文件）是最大单块**，且注释显示已按 0.30 的 layer-config API 重构过（`prepare_layer_config` / `transform_humming_tensors`），迁移面主要是 INC 配置解析与 calib。
- **Windows 使能共 88 文件 +1,404/−394**，其中构建侧（csrc 33 + cmake 5 + setup.py + CMakeLists + fix 脚本）随上游 0.29/0.30 的构建系统变化需要重做；whl 内那 23 个运行时修复（safetensors Windows 读取器 +175、block_table Triton 兜底 +73、cuda_mem_ops Windows 回退 +56、/dev/shm→tempdir 等）需要逐个在新版本里找对应位置重新落点。
- **树（vllm-windows-0.27.1）与 whl 不同源**，只能当参考；不要拿树去 diff 0.30 得出结论。

---

*生成方式：全部数字来自临时脚本逐文件比对（difflib，已去 \r），中间结果 JSON 与临时目录 `G:\qwen3.8model\_tmp_dist271` 在写完本报告后已删除。*
