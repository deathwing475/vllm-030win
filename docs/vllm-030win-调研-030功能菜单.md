# vLLM v0.30.0 功能菜单（Windows 移植选单）+ SystemPanic/vllm-windows 现状

- 调研日期：2026-09-25
- 上游 vLLM v0.30.0：2026-09-22 发布，762 commits / 315 位贡献者（官方 release notes 全文已核对）
- 移植底座：SystemPanic/vllm-windows v0.29.0+cu132 win whl（2026-09-18 发布）
- 本文用途：下一轮由用户从菜单中**点名**要移植的功能。每项给出「给你什么 + 面向哪类工作负载 + 兴趣点标注」。

**兴趣点标注**（贯穿全文）：
`[Q3.5/3.8]`=Qwen3.5/3.8 系（qwen3_5/qwen3_next 架构）｜`[WNA16]`=compressed_tensors WNA16 int3 g128｜`[DFlash]`=DFlash/DFlash2 投机解码｜`[nvfp4/SM120]`=nvfp4/Blackwell 消费卡｜`[KVoff]`=KV offload/prefix cache 长上下文（204k 配方）｜`[量化]`=量化层（GPTQ/humming/compressed_tensors）｜`[MoE新]`=MoE 新模型本地跑

**三类分组**：A=性能/正确性修复（无脑有益）｜B=新模型支持（想要才搬）｜C=新子系统（工程量大、按需）｜D=SM120 硬件专项｜E=破坏性变更与依赖水位

---

## 一、A 组：性能/正确性修复（无脑有益，建议默认全搬）

| 编号 | 功能（给你什么） | 工作负载 | 兴趣点 | 关键 PR |
|---|---|---|---|---|
| A1 | **引擎启动/图捕获大提速**：捕获期冻结 gc，CUDA graph 捕获 12s→2s、引擎初始化 28.9s→8.2s（H200 实测）；kernel 预热提前到捕获前，Triton JIT warmup 统一注册表迁移（DeepSeek V4/shared attention/sampling/DFlash/mHC 全部纳入），renderer warmup 与引擎初始化重叠，FlashInfer autotune 不再随源码变动重跑 | 所有（尤其反复重启调试） | 全部 | #54646 #55341 #50175 系列 #54557 #54794 |
| A2 | **RL/训练侧回归修复**：`--return-sampling-mask` 改 GPU 端压缩，修复约 2x RL step-time 回归；sleep 模式释放 NCCL 通信内存；DP 状态首个 wave 同步（pause/sleep 每 rank 不再烧 32 次 dummy forward） | RLHF/refit/sleep-mode | 全部 | #54901 #51485 #52957 |
| A3 | **投机解码正确性包**：EAGLE 重发 block 对齐 prompt 仍命中 prefix cache；offload 命中在 MTP/EAGLE 下生效；Mamba 状态在 EAGLE resume 位置缓存；**自适应验证（在线 acceptance 估计）对所有 draft 模型生效**；DFlash drafter 去掉 FlashAttention AOT schedule 依赖；`disable_eagle_block_drop` 开关；fastsafetensors PP 草稿死锁修复 | DFlash/MTP/EAGLE 推理 | `[DFlash]` `[Q3.5/3.8]` | #54713 #52771 #52228 #54374 #53388 #54416 |
| A4 | **KV offload/长上下文正确性包（15+ 修复）**：异步查找、SWA 可达性与覆盖、磁盘对齐、DiskBackend 缓冲竞态、offload key 排序、prefetch slot 所有权、加载边界、tiering 关闭、ARC 回退、超大 offer、末 token 槽位等 | 204k offload 配方 | `[KVoff]` | #54872 #55075 #55823 #54362 #55712 #56486 #55424 #51667 #52923 #54975 #52807 #52290 |
| A5 | **量化修复包**：compressed_tensors WNA16 MoE 未设 `group_size` 修复；奇数行 per-token-group 量化精度悬崖修复；NVFP4 padding/scale/SiLU+mul 三处修复；MoE fused sum int32 溢出修复 | 量化模型推理 | `[WNA16]` `[量化]` `[nvfp4/SM120]` | #53163 #56478 #53568 #52501 #55643 #50220 |
| A6 | **Qwen3.5/3-Next kernel 启动预热**：Qwen3.5/Qwen3-Next Triton kernel 与 GDN gated RMSNorm 启动即预热（消除首步卡顿/编译抖动）；Qwen3.8-Flash-Next 的 FP8 PLE weight scale、融合 PLE conv stride 修复 | Qwen3.5/3.8 部署 | `[Q3.5/3.8]` | #54797 #54251 #54722 #54882 #55375 |
| A7 | **Windows/WSL 直接受益**：无 pinned memory 平台（WSL）支持（#56908）；更多 H2D 拷贝 pin；graph 不可用时**抛错而非产出乱码**；非编译模型回退 `FULL_DECODE_ONLY` graph | Windows 部署稳定性 | 全部 | #56908 #54660 #54782 #55095 |
| A8 | **API/结构化输出修复**：非法结构化输出请求不再拖停引擎；XGrammar 支持 `patternProperties`/`propertyNames`；全历史 reasoning 扫描移除；多模态双 BOS、beam search、无 tokenizer 的 VL processor 等一批修复 | OpenAI 兼容服务 | 全部 | #51450 #42904 #55223 #55288 |
| A9 | **安全修复**：验证错误响应体限幅（堵住约 5300x 响应放大）；`cache_salt` 进 LMCache 前校验（单请求不再打挂引擎）；客户端稀疏 embedding 稠密化前限幅 | 公网服务 | 全部 | #54684 #51444 #54632 |

---

## 二、B 组：新模型支持（想要才搬）

| 编号 | 功能（给你什么） | 工作负载 | 兴趣点 | 关键 PR |
|---|---|---|---|---|
| B1 | **DeepSeek-V4.1-Flash**（`vllm/models/deepseek_v41` 新目录）：整 KV 存 MXFP8（FlashMLA V4.1 record）、Mega-mHC、异步 Engram 预取 + Engram DP 分片、DSpark 草稿、V4.1 严格工具参数 schema | 大 MoE 推理（主打 SM100 数据中心卡） | `[MoE新]` `[DFlash]` | #56214 #56228 #56208 #56893 #56962 #56512 |
| B2 | **GLM-5.3-Flash**（`vllm/models/glm5next` 新目录）：EPLB、FlashKDA chunked prefill（比 Triton 路径快 1.7–3.8x）、NoPE (256,0,256) 稀疏 prefill；`glm5next` NVIDIA 子树已打进 wheel | GLM5 系 MoE 本地跑 | `[MoE新]` | #53906 #55119 #55737 #55738 #55214 |
| B3 | **Qwen4Exp / Qwen3.8-Flash-Next**（`vllm/models/qwen4_exp`）：Qwen3Next 架构扩展（QSA 稀疏 indexer + HyperConnection + PLE ngram + Engram），含 MTP。0.30 新增：分离 prefill/decode QSA indexer kernel、融合 PLE、FP8 indexer cache、UVA PLE offload、Engram TP（`--engram-config`）、去掉 torch.compile（FP8 单卡 GB300 可容） | Qwen3.5/3.8 系模型本地跑 | `[Q3.5/3.8]` `[MoE新]` `[量化]` | #54513 #54517 #54890 #55309 #54371 #55272 |
| B4 | **K2-Horizon / Cohere Compass / Bailing V3 VL（带 MTP）/ Nanbeige4.2（Transformers 后端）**：四个新模型族，K2-Horizon 自带 reasoning/tool parser | 按需 | `[MoE新]` | #55063 #54774 #55921 #56071 |
| B5 | **DeepSeek-V4-Flash-Vision-Exp**：视觉版 V4，含 LoRA、ROCm 支持 | 多模态 MoE | `[MoE新]` | #54566 #55897 |
| B6 | **DeepSeek-V4 CPU 后端**：AVX512/AMX 稀疏 MLA、indexer、mHC、compressor kernel | CPU 推理 | 低相关 | #55355 |
| B7 | **Qwen3.5 系配套**：Qwen3.5/3.6 多模态 MTP（`n_predict` 从 text config 解析）、Qwen3 DSpark padded-vocab 草稿、A100 Qwen3.5-122B TP2 fused-MoE 调优、ColQwen3.5 pooler projector | Qwen3.5/3.8 投机解码与多模态 | `[Q3.5/3.8]` `[DFlash]` | #55369 #55133 #55511 #54847 |
| B8 | **MiniMax-M3**：0.30 主要是 ROCm 侧 indexer/top-k 优化与 fused allreduce+GemmaRMSNorm；CUDA 侧 `minimax_m3` 目录此前已有 | MiniMax-M3 本地跑 | `[MoE新]` | #54682 #52664 #55235 #56170 #54787 |

> **Qwen3.5/3.8 定位结论**：0.30 里 Qwen3.8-Flash-Next 的代码目录是 `vllm/models/qwen4_exp`（`Qwen4ExpTextConfig` 继承 `Qwen3NextConfig`，即 qwen3_5/qwen3_next 架构直系），release notes 中「Qwen3.8-Flash-Next」与「Qwen4-exp」指同一族。这是与你本地 Qwen3.8-27B 工作最直接相关的一块。

---

## 三、C 组：新子系统（工程量大，按需搬）

| 编号 | 功能（给你什么） | 工作负载 | 兴趣点 | 关键 PR |
|---|---|---|---|---|
| C1 | **Fast Start 权重缓存守护进程**（`--load-format ipc_cache`）：量化后 TP 分片权重常驻 GPU，引擎重启经 CUDA IPC 映射，不再从磁盘重载；覆盖 FP4 checkpoint 与多节点 TP | 反复重启/多实例同卡 | `[nvfp4/SM120]` | #54921 #55465 #55468 |
| C2 | **HiSparse**（`vllm/v1/hisparse` 新目录）：sparse-MLA decode 的 host 层 KV 溢出——GPU 压力下 KV page spill 到 pinned host memory，top-k miss 走每请求 GPU hot buffer；TP 共享 host cache、Prometheus 计数、`HiSparseConnector` 启用 | 长上下文 sparse-MLA（DeepSeek/GLM 系）；**Qwen3.5 非 sparse-MLA，用不上** | `[KVoff]`（有条件） | #53781 #56061 #56629 #57041 |
| C3 | **Watermarking**（`vllm/v1/watermarking` 新目录）：Gumbel-max 水印生成+检测（keyed PRF、每请求可退出、检测端点示例）；双 key 版**兼容投机解码**；Rust 前端转发每请求控制 | 内容溯源/合规 | `[DFlash]`（双 key 兼容 spec decode） | #54053 #56122 #56338 |
| C4 | **Engram 子系统**（`--engram-config`）：UVA PLE offload、Engram 张量并行、CPU offload 查找异步预取、Engram DP 分片 | Qwen3.8-Flash-Next 长上下文记忆 | `[Q3.5/3.8]` | #54371 #56512 |
| C5 | **EC transfer / 编码器缓存**（`distributed/ec_transfer`）：P2P NIXL + CPU EC connector、ECMooncakeConnector、EPD 分离、encoder-only 实例 GPU NVDEC | 多模态 P/D 分离集群 | 低（Windows 单卡无 NIXL/Mooncake） | #47941 #41567 #53675 |
| C6 | **量化新能力包**：`quantization_config.targets` 定向在线量化；部分预量化 checkpoint 在线续量化（任意量化方法）；W4A16 DSA + `nvfp4_fp8_ds_mla` KV cache；FlashInfer CuTeDSL NVFP4 W4A16（SM100/103 默认替代 Marlin）；NVFP4 torch linear backend；`linear_backend_per_quant` 每量化法后端覆盖；**AutoRound 2/3/5/6/7-bit**（CUDA） | 量化模型生产 | `[量化]` `[WNA16]` `[nvfp4/SM120]` | #51285 #51392 #51724 #53014 #53319 #51204 #52890 |
| C7 | **GPTQ `g_idx` 移除（breaking）**：GPTQ 动态激活排序废弃，`g_idx` 被忽略，Marlin/GPTQ/CPU/RDNA3 相关 kernel 删除。**若你的 GPTQ checkpoint 依赖 g_idx 乱序，搬 0.30 前必须先验证加载** | GPTQ 模型 | `[量化]` | #54809 |
| C8 | **KV offload 新能力**：KVCR 二级 tier 适配器、P2P tier 超时/迟到拒绝、OffloadingConnector retention interval、SimpleCPUOffload 细粒度 hybrid prefix 命中、DSA 混合 page size、prefix-cache bypass、无前向步存储 | 204k offload 配方升级 | `[KVoff]` | #53624 #53453 #51886 #54736 #54756 #54998 |
| C9 | **大规模 serving 一揽子**（PCP/DCP 上下文并行、Elastic EP、DeepEP v2、Mooncake Store、NIXL、PCIe IPC all-reduce） | 多机多卡集群 | 建议整体跳过 | #56157 #54985 #52781 #53129 #53576 |

---

## 四、D 组：SM120 / Blackwell 消费卡专项（`[nvfp4/SM120]`）

| 编号 | 功能（给你什么） | 说明 | PR |
|---|---|---|---|
| D1 | **B12X causal paged attention**（`--attention-backend B12X_ATTN`）：SM120/SM121 专属注意力后端 | 50 系消费卡直接可用的 attention 路径 | #52017 |
| D2 | **W4A4 NVFP4 在 SM120/121 成为默认**（优先于 weight-only kernel） | nvfp4 量化在 50 系上的默认路径变化 | #55170 |
| D3 | **DeepGEMM 钉到 vLLM fork 2.8.0**，带 SM120 与 SM90 paged-MQA 移植 | MoE/GEMM 底座升级 | #56876 |
| D4 | **FlashInfer GDN prefill（SM12x）+ fused GDN MTP decode（SM110）** | GDN 系模型（Qwen3-Next 系）在消费卡的 prefill/投机路径 | #55715 #53835 |
| D5 | **SM12x blockwise FP8 CTA raster swizzle**（GB10/DGX Spark） | FP8 blockwise 在 SM12x 的性能 | #55180 |
| D6 | **NVFP4 KV cache FMHA 加速 + fused add-RMSNorm+NVFP4** | nvfp4 路径整体提速 | #55031 #51925 |

---

## 五、E 组：破坏性变更与依赖水位（移植前必读）

### 破坏性变更
1. scale-out 端点（`/render` `/derender` `/inference/v1/generate`）改为 `--enable-scale-out` 显式开启，`VLLM_ENABLE_SCALE_OUT_ENDPOINTS` 环境变量删除（#54579 #55176）。
2. **GPTQ `g_idx` 移除**（见 C7）。
3. 0.29 弃用项已删：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`、`VLLM_MM_HASHER_ALGORITHM` 等环境变量改用 config 字段（#55353）。
4. Mamba `all` cache 模式弃用（回退 MRV1）；`python -m vllm.entrypoints.grpc_server` 弃用（#55041 #56746）。
5. **YaRN 对齐 Transformers**：厂商 YaRN 别名不再二次缩放 `max_model_len`，部分模型推导出的最大长度骤降（例：131072→32768）。**204k offload 配方若依赖 YaRN 推导长度，移植后必须重新核对 `max_model_len`**（#56446）。
6. DCP 需 attention 实现显式声明支持；MoRI-IO WRITE 模式加限制；音频 resampler 默认 PyAV→torchaudio。
7. 新默认值：SM100/103 上 NVFP4 W4A16（CuTeDSL）替代 Marlin；SM120/121 上 W4A4 NVFP4；SM100 BF16x3 router GEMM。

### 依赖水位对比（v0.29.0 → v0.30.0，源：requirements/cuda.txt）
| 依赖 | 0.29.0 | 0.30.0 | Windows 移植影响 |
|---|---|---|---|
| torch | 2.13.0 | 2.13.0（不变） | 好消息，无需跟 torch |
| flashinfer-python/cubin | 0.6.18 | **0.6.18.post1** | 小版本 +post1；但配套 flashinfer-windows 停在 0.6.11.post3（见任务 B） |
| nvidia-cutlass-dsl | 4.6.2 | **4.7.1** | CUTLASS 4.7.1，需重编相关 kernel |
| quack-kernels | 0.6.4 | **0.6.5** | 小升级 |
| humming-kernels | 0.1.12 | 0.1.12（不变） | 无影响 |
| compressed-tensors | — | 0.17.0 | WNA16 相关，注意版本 |
| transformers | ≥5.10.4 | 5.16.1 | Transformers 后端模型受影响 |
| tilelang / tvm-ffi | 0.1.12 / 0.1.11 | 不变 | 无影响 |

---

## 六、兴趣点 → 菜单项交叉索引（供点名）

| 你的兴趣点 | 建议优先看 |
|---|---|
| Qwen3.5/3.8 系（qwen3_5 架构） | **B3**（Qwen4Exp/Qwen3.8-Flash-Next）、B7、A6、C4（Engram）、D4（GDN on SM12x） |
| compressed_tensors WNA16（int3 g128） | A5（WNA16 MoE group_size 修复）、C6（AutoRound 2–7bit 含 3bit）、C7（GPTQ g_idx 移除风险） |
| DFlash/DFlash2 投机解码 | A3（自适应验证+一批修复）、B1/B7（DSpark 草稿）、C3（水印兼容 spec decode）、A1（DFlash kernel 进 warmup 注册表） |
| nvfp4 / Blackwell 消费卡（SM120） | **D1–D6 整组**、C6（NVFP4 新能力）、A5（NVFP4 修复）、C1（FP4 checkpoint 缓存） |
| KV offload / prefix cache 204k | A4（修复包）、C8（新能力）、C2（HiSparse，注意适用面）、E5（YaRN 长度变更核对） |
| 量化层（GPTQ/humming/compressed_tensors） | A5、C6、C7、依赖表（humming-kernels 不变） |
| MoE 新模型本地跑 | B1（DeepSeek-V4.1）、B2（GLM-5.3-Flash）、B3（Qwen3.8）、B4/B5/B8（K2/Compass/Bailing/Nanbeige4.2、V4-Vision、MiniMax-M3） |

---

## 七、任务 B：SystemPanic/vllm-windows 现状（观察哨事实）

### 7.1 最新 release / whl 水位
| 项目 | 事实 |
|---|---|
| 最新 release | **v0.29.0**，2026-09-18 发布（whl：`vllm-0.29.0+cu132-cp312-cp312-win_amd64.whl`，259 MB，424 次下载） |
| 0.30 迹象 | **无**。releases 无 0.30/0.30-rc；tags 无 0.30；issues 全文搜 "0.30" = 0 条；README 无 roadmap、无 0.30 计划 |
| 分支 | `main`（上游同步）+ `vllm-for-windows`（构建分支，README 指定用它）。`vllm-for-windows` 最新 commit 为 "vLLM 0.29.0 release"（2026-09-18 20:14，tag v0.29.0） |
| 活跃度 | `main` 最后 push 2026-09-18（同步上游 main 至 0.30 前夜，含 #57576 等 0.30 期 PR）；维护者 2026-09-22（v0.30.0 发布当天）仍在 issue #88 回复用户 → **人还在** |

### 7.2 发布节奏（上游 → win whl 时滞）
| 上游版本 | 上游发布 | win whl | 时滞 |
|---|---|---|---|
| 0.25.0 | 2026-07-11 | 2026-07-12 | +1 天 |
| 0.26.0 | 2026-07-27 | 2026-07-28 | +1 天 |
| 0.27.1 | 2026-08-11 | 2026-09-18 | +38 天（维护者搬家欧美，停摆 7 周） |
| 0.28.0 | 2026-08-26 | **跳过未发** | — |
| 0.29.0 | 2026-09-09 | 2026-09-18 | +9 天（0.27.1+0.29.0 同日双发） |
| 0.30.0 | 2026-09-22 | 未发（截至 09-25） | 观察中 |

- issue #84（"0.28 win whl + dflash2 sync"请求）中维护者自述 8 月底搬家，承诺补发；9-18 实际补发 0.27.1+0.29.0 并跳过 0.28。
- issue #88（Python 3.14 wheel）9-22 被回复"请自行从源码构建"——新需求排期靠后。

### 7.3 配套仓版本水位
| 配套仓 | 最新版本 | 日期 | 对 0.30 的水位 |
|---|---|---|---|
| flashinfer-windows | v0.6.11.post3 | 2026-07-01 | **落后**：上游 0.29/0.30 均要求 flashinfer-python==0.6.18(.post1)，此仓停在 0.6.11.post3 且 7 月后无 commit。凡依赖 FlashInfer 的 0.30 特性（NVFP4 CuTeDSL、XQA、GDN SM12x、KDA、PCIe IPC all-reduce）移植时都要评估此项 |
| humming-windows | v0.1.15 | 2026-09-18 | **够用**：与 vllm-windows 0.29.0 同日发布；上游 0.29/0.30 都钉 humming-kernels==0.1.12 不变 |
| nccl-windows | 无 release（源码自编译） | 最后 push 2026-05-20 | NCCL 移植久未动，多卡用它；单卡不受影响 |

### 7.4 一句话结论
**「等它出 0.30 win whl」是现实选项而非遥遥无期**——维护者活跃、历史上无搬家干扰时交付时滞仅 +1~+9 天，合理预期 0.30 win whl 在数天到两三周内出现；但有跳版本（0.28）与单人搬家停摆 7 周的先例，且 flashinfer-windows 停在 0.6.11.post3 是 0.30 新 kernel 特性的最大水位缺口，故建议：修复类（A 组）不等 whl 也可先行 cherry-pick 到 0.29 底座验证，新子系统（C 组）等 0.30 whl 落地再评估。

---

## 附录 1：本地调研目录 ↔ 0.30.0 功能对照（已核对源码树）
| 本地调研所述改动目录 | 对应本菜单 |
|---|---|
| model_executor/layers（量化/MoE） | A5、C6、C7 |
| models/deepseek_v4 | B1 的 V4 修复包（release notes "DeepSeek V4" 段） |
| models/deepseek_v41（新） | B1 |
| models/glm5next（新） | B2 |
| models/qwen4_exp | B3（Qwen4Exp / Qwen3.8-Flash-Next） |
| v1/attention | A3/A5 的 kernel 部分、D1/D4 |
| v1/worker | A1、A2、C1 |
| v1/hisparse（新） | C2 |
| v1/kv_offload、v1/simple_kv_offload | A4、C8 |
| v1/watermarking（新） | C3 |
| distributed/kv_transfer | C8、C9 |
| distributed/ec_transfer | C5 |
| warmup 重构 | A1 |
| engram config | C4（`--engram-config`） |
| model_executor/kernels | D 组、A5 |

（已核对 v0.30.0 源码树：`vllm/models/` 含 deepseek_v41、glm5next、qwen4_exp、minimax_m3、kimi_k3；`vllm/v1/` 含 hisparse、kv_offload、simple_kv_offload、watermarking。）

## 附录 2：来源链接
- 官方 release notes（v0.30.0，2026-09-22）：https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- 上游版本时间线：https://github.com/vllm-project/vllm/releases
- v0.30.0 依赖钉版：https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/requirements/cuda.txt （对照 v0.29.0：https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/requirements/cuda.txt ）
- v0.30.0 源码树：https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/models 、https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/v1
- Qwen4Exp 架构证据：https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/models/qwen4_exp/config.py （`Qwen4ExpTextConfig(Qwen3NextConfig)`）
- SystemPanic/vllm-windows（releases/branches/commits/issues）：https://github.com/SystemPanic/vllm-windows 、https://github.com/SystemPanic/vllm-windows/releases
- 维护者搬家/交付节奏证据（issue #84）：https://github.com/SystemPanic/vllm-windows/issues/84
- flashinfer-windows：https://github.com/SystemPanic/flashinfer-windows/releases
- humming-windows：https://github.com/SystemPanic/humming-windows
- nccl-windows：https://github.com/SystemPanic/nccl-windows
- 各功能 PR 号均可拼接 https://github.com/vllm-project/vllm/pull/<号> 查看
