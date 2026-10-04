# vllm-030win 调研 — 新模型全自动适配（profile card + 一次自证 boot + 一致性守卫）

> **来源**：2026-10-04 用户令「**这样子就完全不泛用，我需要进来一个收支持的模型，然后能真正意义上的全自动适配，而不是靠改代码来达到相对最优的设置**」。
> **本档状态 = 设计（091 出设计，未实现）**；实现与验收 = 头名 **092**。本档每条**已实测**的都给步骤号，**没实测**的一律写在 §7 未验证项里，不得当结论用。
> **判据总纲**：一个"收支持的模型"进来，**人不写任何一个数字**，launcher 只有 `MODEL / DRAFT / L / 投机家族` 四个人类意图级参数，其余全部派生或实测；凡是"抄自另一个模型"的值必须留在 profile card 里并标 `source=human-copied`，守卫看到它就不算通过。

---

## 1. 问题定义

本卡（RTX 5070 Ti 16 GB / 23.1 GiB 主机内存）上，同一份"能跑 163,072 上下文 + DFlash2 投机"的能力，GSQ 生产天天在跑（`tools/serve_gsq_prod029_n2.cmd`），而 Orca 线走了三步才够到：088 判"本卡装不下"（错，089 作废）、089 靠**逐条手抄** GSQ 的 flag/env 才起得来、090 靠**手写** `--kv-cache-memory-bytes 800000000` 才回到正常速度带。

⇒ **手抄不是交付物**：下一个模型（O4 的任何新 checkpoint、或换量化家族）同样不会自己抄。要泛用的不是"更多旋钮"，而是**每个旋钮的值由谁算出来**。

## 2. 现状盘点：Orca 臂相对 GSQ 生产 diff 出来的钥匙全集

来源 = 089/090 逐条搬运（步骤 089、090）+ 091 把两支 launcher 逐行 diff。分三类：**①可从 checkpoint/config 派生（profile card 可算）②必须一次 boot 实测（引擎自报才可信）③平台/驻留类（既不由 config 决定也不由模型决定，是这张卡的现场）**。

| # | 钥匙 | 值 | 类别 | 派生 / 实测路径 |
|---|---|---|---|---|
| 1 | `VLLM_KV_GROUP_SIZE` | `8` | ①可算 | 混合模型的层分布：`num_linear_layers=48 / num_full_attn=16`（config）⇒ 组数从"每层一组"塌成 2+6 组；实测收益 +18.5%（036）。**具体 8 是最优还是可行，未做敏感性**（§7） |
| 2 | `--mamba-ssm-cache-dtype` | `bfloat16` | ①可算（默认 fp32 → bf16 直接砍 `a` 的一半） | 045/046 实测：mamba 页 3,248,128 → 1,675,264 B、attn block 2832 → 1456、容量 +12.5% |
| 3 | `--mamba-cache-mode` | `align` | ①半可派生 | 由层类型混合与投机家族决定：`Qwen3_5MTP` 只拒 `all`（085 预检）；`none/align/all` 的取舍规则**未从源码完整推导**（§7） |
| 4 | `--enable-prefix-caching` | on | ②实测 | 对混合模型可能触发额外驻留，属"起了才知道"，须与拒起读数一起判 |
| 5 | `--kv-offloading-backend native` + `--kv-offloading-size` | `8` GiB | ③平台 | 上限 = **主机可用内存**（本机 23.1 GiB 总），不是模型属性；**注意这是 KV 卸载到主机 mmap，不是权重卸载**（089 澄清的含混点） |
| 6 | `--gpu-memory-utilization` | `0.922` | ③平台 | 上界 = `request_memory` 比的**启动空闲**（本机 14.68/15.89 GiB ⇒ 0.95 不可达，085）；0.922 是生产每日实跑值 |
| 7 | `--kv-cache-memory-bytes` | GSQ 生产 `3.4e9`；Orca 090 `8e8` | ②**必须实测** | 这是 090 的速度开关：同一 L=16,384，自动池 1.72 GiB → **4.83/5.49 tok/s**（mem-util 2% / 80 W = 082 的等待型慢模态）；手压 0.745 GiB → **86.63/86.01/79.91**（mem-util 55-64% / 269-283 W）。⇒ **自动 sizing 的"给得更多"在这张卡上是负收益**，泛用机制必须把它当**搜索变量**而不是当容量上限 |
| 8 | `--max-num-batched-tokens` | `1024`（生产/089-090 臂）；088 用 `2048` | ①②混合 | 下界 = 页几何：`mbt ≥ attention/mamba 页 + num_spec`（**必守 11**，086 实测 nvfp4 页长 2784 而 mbt 3072 会把预算吃光 ⇒ **必需但不能多抬**）；上界由拒起读数决定 |
| 9 | 图模式：无 `--compilation-config` + `--cudagraph-capture-sizes 3` | 生产 027 起如此 | ①②混合 | 由 model runner 支持度决定（DFlash2 ⇒ **只能 V2**，必守 29④）；代价由实测：`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 默认从 KV 池里扣走约 **1.3 GiB** 捕获峰值（090 s1 因"需 0.65 > 可用 0.32"拒起） |
| 10 | `--language-model-only` | on（GSQ 生产） | ①可派生 | 多模态包装存在时才需要；Orca 是 `Qwen3_5ForCausalLM` ⇒ 预期 no-op，**未实测**（§7） |
| 11 | `VLLM_KV_CACHE_LAYOUT=HND` | on（生产） | ③平台 | 与 attention backend / 量化 KV 布局耦合（084 的 nvfp4 head-major 布局定罪过一类正确性 bug）；**未与 090 的快慢分型做过单变量对照**（§7） |
| 12 | `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864`、`CXX=clang++`、`LIB`/`PATH` 追加、`HF_HOME`、`HF_HUB_OFFLINE=1`、`TMP/TEMP`、`HOME/USERPROFILE=C:\fi` | 环境契约 | ③平台 | 全属《交接文档.md》§2 的环境契约，模型无关 ⇒ 应进 **platform card**（一份、所有模型共用），不进 profile card |
| 13 | `VLLM_DBG_TRACE=1 VLLM_DBG_MIN=1 VLLM_DBG_PIN=1`（+ `pin_shim` 在 `PYTHONPATH`） | **GSQ 生产有，Orca 臂只挂了 shim 没开这三个门** | ③平台/驻留 | **091 diff 出来的未迁移项，且直指 090 的慢模态机制**：权重页 pin/驻留正是 081/082 定性的"kernel 常驻但等待"的成因侧。当前只是记账，**不得当作"这就是原因"**（§7，092 首选单变量实验） |
| 14 | `ORCA_EXL3_ALLOW_EMPTY_SHARED=1`（sitecustomize 补丁） | Orca 臂专有 | ①可派生（按量化插件与投机家族） | 085 定罪 = 插件对"draft 共享后即为空"的 `embed_tokens`/`lm_head` 抛错 ⇒ **任何 EXL3 checkpoint 的 MTP/EAGLE 都会 boot 死**；泛用形态 = 该量化插件 × 该投机家族的**组合事实**，不该由人记得（必守 28③） |

**另两条 091 实测到的"该自动做掉"的事**：
- **权重构成盘点**：Orca 11.43 GiB = `layers 9.156 + embed 1.185(int8) + lm_head 0.889(exl3 6-bit) + mtp 0.198` ⇒ **DFlash2 档不需要 `mtp.*`，而 GSQ 那份里它更大**（Orca `mtp.*` = 0.198 GiB / GSQ = **0.791 GiB**；GSQ 全账 = `layers 8.899 + visual 0.858 + mtp 0.791 + lm_head 0.629 + embed 0.629` = 11.806 GiB）。**它们是否真占显存 = 未验证**（091 查过 boot 日志：没有 unused-weight 类警告；EXL3 插件按模块清单取张量，很可能根本没读）⇒ 验证法见 §7 首条。泛用解法仍是 **按启用的投机家族跳过未用模块**，而不是把 `util` 往上顶。
- **每请求需求的正确分解**：`need(L) = a + b·L`，`a` = 每请求 GDN 递归状态（nvfp4+N=2 实测 ≈ **0.68 GiB**），`b` = 注意力页（**22.2 KiB/token**）。`a` 与 `b` 都是 config + cache dtype + group size 的函数 ⇒ **可算**；088 的"198 KiB/token"之所以误导，是因为它拿含 `a` 的总量除小 `L`（`L=4,096` 时 `a` 占 89%），于是把"砍 `a` 的两把钥匙"这个正解遮掉了。

## 3. 目标形态（三件交付物）

1. **profile card（`profile/<model>.json`，纯派生，零 boot）**：从 `config.json` + `quantization_config` + `*.safetensors` header 算出
   层类型分布（linear/full-attn 数与位置）、hidden/head 几何、**页几何**（attention 与 mamba 页字节、可推 `mbt` 下界）、`need(L) = a + b·L` 的 `a`/`b`（按 dtype 与 group size 的函数）、投机家族可用性（`supports_eagle3` / aux tap 基类 / V1-V2 runner 门 / `fc` 宽 = `len(target_layer_ids) × hidden`，即必守 29 的三个数）、**未被使用的模块清单**（如 `mtp.*` 在无 MTP 档下）⇒ 输出**候选集**，不输出单一"最优值"。
2. **platform card（一份，所有模型共用）**：`C:\fi`/`TMP`/`CUDA_HOME`/`FLASHINFER_*`/`CXX`/`LIB`/`PATH`/`HF_*` 这些环境契约 + `request_memory` 的启动空闲读数 + 主机可用内存 ⇒ 从各 launcher 里**抽出去**，别再每个模型手抄一遍。
3. **一次自证 boot（`autoprobe`）+ 落档**：profile card 生成候选 → 单 boot 扫描 → 从**引擎自己的日志**取值（`Available KV cache memory` / `GPU KV cache size` / `Maximum concurrency` / `Setting attention block size` / 拒起句的 `estimated maximum model length` / `vllm:spec_decode_*`）→ 用"快档判据"选池字节 → 写死成 profile card 的 `measured` 段 → 由**同一支泛用 launcher** 消费。

**泛用 launcher = 一支**：`tools/serve_generic.cmd`（暂名），人类参数只有 `MODEL / DRAFT / L / SPEC_FAMILY / TIER`；其余 flag 一律从 card 读。**禁止**出现"为了这个模型加一行"的情形——需要新 flag 就说明 card 缺字段，补 card。

## 4. 自证 boot 的扫描协议（草案）

- **拒起免费**：vLLM 在池不够时是 **boot 期 ValueError**，不烧 token 不计时间 ⇒ 候选池字节 / `L` / `mbt` 可以在同一支 boot 序列里做二分/阶梯，把引擎自己的读数当唯一裁判（这是 088/089 已经用过、零成本的路子）。
- **速度必须带外分型**：沿用 090 的 `step090_orca_speed.py` 采样线程（`utilization.gpu / utilization.memory / power.draw / clocks` @1 Hz）与 `anchor_longctx.py` 的 8k anchor（同生产 watchdog 口径）。**判"慢"之前必须先报池字节数**（必守 33）；**并发倍数不是健康证据**（090 的 2.65x 比 1.13x 慢 16 倍）。
- **首编深塌不入样本**：任何 L/dtype/图模式变化后的第一支是编译 boot（必守 17），只作自证不作读数。
- **验收必须含 ≳10K 长请求**（必守 28⑤：086 的 `nvfp4 × 多步 draft × 长 prompt` 停摆只有长请求能抓到）。

## 5. 一致性守卫（这是"泛用"的实质，不是搜索算法）

**双轨对照 = 同一支卡跑两条推导路径，读数字节数相等才算通过**：
- 轨 A：**从 config 算**页几何与 `a`/`b`（profile card）。
- 轨 B：**引擎自己 resolve**出来的 `Setting attention block size` + 拒起句 + `GPU KV cache size`。
- **守卫判据**：轨 A 的 `page 字节 × 组数` 与轨 B 的池几何换算一致（允许块对齐误差）；不一致 ⇒ **报错并拒绝生成 launcher**，不许"人肉挑一个看起来对的"。091 的教训正是这类：报数 `N = max_concurrency × max_model_len`（`kv_cache_utils.py:2040-2052`）被我当成字节容量除了一次，一个假异常（"2 倍差"）就写进了三处文档。
- **魔数记账**：card 里每个值带 `source ∈ {derived, measured, human-copied, platform}`；launcher 生成器遇到 `human-copied` **必须显式警告** ⇒ 抄来的东西永远是债务，不能伪装成结论。089/090 的配方今天全部记为 `human-copied`，等 092 把它们逐条转成 `derived/measured`。

## 6. 验收标准（092 的出口判据）

1. 对**已跑通的两个模型（GSQ、Orca）**用同一套泛用 launcher 出配置：GSQ 臂必须复现生产的 163,072 / 163,719 tokens / 8k anchor 快档，Orca 臂必须复现 090 的 86 tok/s 带 —— **两条都不许人写数字**（人只给 `MODEL/L/SPEC_FAMILY`）。
2. profile card 的 `a`/`b` 与引擎拒起读数在同档下拟合残差 < 5%（现有两点：`b4 0.77@4,096`、`b2 1.03@16,384`）。
3. **权重账对平**：先按 §7 首条定 `mtp.*` 是否占显存 —— 若占，跳过它应让 Orca 权重读数降 ≥0.15 GiB 且 needle 仍 3/3；若不占，则本条改为"把 088 那笔'引擎比草稿文件多算 ~1.09 GiB'的差额逐项归零"（候选 = 草稿侧副本 / CandidateSelector 码本 / 图与激活峰值）。
4. 新增一个"收支持的模型"时，人的改动 = 0 个数字（若做不到，本步判 INCOMPLETE 并列出仍缺的字段）。

## 7. 未验证项（不得当结论用）

- **`mtp.*` 到底占不占显存**（§2 的 0.198 / 0.791 GiB 只是**盘上**构成）：判据 = 对照 boot 日志 `Model loading took ... GiB memory` 与"删掉 / 跳过 `mtp.*` 后"的同一读数，并读模型类 `load_weights` 与 orcasaq2 插件的权重映射清单（091 只查到"日志里没有 unused-weight 警告"，**不足以定"载入"**）。若确认不占，则 088 权重账里"引擎比草稿文件多算 ~1.09 GiB"要另找去处（草稿侧 fp32 副本 / CandidateSelector 码本 / 图与激活峰值都还是候选）。
- **`VLLM_DBG_PIN/TRACE/MIN` 是否为 090 慢模态真因** —— 只是 diff 出来的未迁移项；须单变量对照（同 `L`、同池字节、只开关 pin）才配说因果。
- **`VLLM_KV_GROUP_SIZE=8` 的敏感性**（8 vs 4/6/16 的容量-速度曲线）与它对 `a`/`b` 的具体作用式。
- **`--language-model-only` 在纯 causal-LM checkpoint 上是否 no-op**。
- **`mbt` 上界与 `--max-num-seqs>1` 的交互**：090 配方是单序列档，**多并发不得照抄**（压池把并发 2.65x→1.13x）。
- **auto sizing 为何在带全部钥匙时仍把池撑到 1.72 GiB**（GSQ 是被显式喂 3.17 GiB 才 1.00x 并发）—— 这条是"泛用"里最关键的一个未知量：**引擎的自动 sizing 目标函数与"这张卡的驻留悬崖"方向相反**。
- **`mtp.*` 跳过载入**在 vllm 0.29 里有无现成开关（未找过）。
- 泛用 launcher 的实现语言/形态（`.cmd` + python 生成器 vs 纯 python）未定。

## 8. 风险登记

| 风险 | 缓解 |
|---|---|
| 搜索变成"每模型跑 20 支 boot" | 拒起读数免费、速度读数只在候选末端跑；单批预算 = 6 支以内 |
| card 与引擎漂移（版本升级后 `N` 的定义变了） | 守卫每次 boot 复核轨 A/轨 B，不一致就拒绝生成配置而不是静默用旧值 |
| 压池换速度砍掉并发 | 池字节按目标并发反推，`--max-num-seqs>1` 时禁用 090 配方（§7） |
| "全自动"被理解成"可以擅自停/换生产" | 生产默认不动（硬停止条款）；autoprobe 只在显式调用时跑，跑前按必守 30 记录生产状态 |
| 泛用化被理解成"允许改 `vllm/` 源码" | 交付物是 card + launcher；改引擎需另立步骤并配 revert 脚本（必守 14） |

## 9. 落地顺序建议（092 起）

1. 先做 §5 的**守卫**（换算与校验函数），因为它立刻能让 089/090 的手抄值变成可核对的账；
2. 再做 profile card 的 `a`/`b` 与权重构成盘点（纯只读，零显存，可当天验）；
3. 然后 pin 那条未迁移钥匙的单变量实验（§7 首项）；
4. 最后合并成一支 `serve_generic.cmd` 并跑 §6 的双模型复现验收。
