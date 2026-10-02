# 调研 — KVMem 虚拟化 KV 工作区（vLLM 0.29 栈实现方案）

> **路线边界（必须遵守）**：本文是 KVMem 的移植设计与阶段 1 验收档，不是无限优化 backlog。当前授权目标是完成 GSQ 栈的最小可用移植与正确性出口；阶段 1 出口后冻结 KVMem 线并回到项目原主线。检索算法改良、N/R/VIEWPORT 参数寻优、GPU 化烘焙、长稳产品化、缩池换余量和生产切换均不属于当前默认授权，除非用户另行明确批准。
> **状态：阶段 1 移植收口（当前头名 082）——057-081 的有界 prefill、工作区、检索/重物化、固定槽位、窗外 needle、全图兼容与 kernel 取证均已有结果；当前只做阶段 1 验收缺口，不开启长期 KVMem 优化。**
> 本文是这条新线的**唯一设计权威**：目标、决策记录、架构、接口落点、参数标定、阶段出口、验证台口径、风险登记册。
> 接手者先读本文，再读《交接提示词.md》（索引版：头名任务与红线一行式）；**纪律全文（必守 1-24）、臂与证据目录、水位与回滚见《vllm-030win-铁律与现场详版.md》——其它文档里「必守 N」引用的就是该文件同编号**。**§12.1-12.3 = 阶段 1a 有界 prefill（057）｜§12.5 = K1 copy-before-free（060）｜§12.6 = K2 准入守卫（060）｜§12.4 = 阶段 1a 剩余清单（已全部完成）。**
> 相关步骤号：**步骤 056（立项与设计调研）**、**057（阶段 1a 有界 prefill，GO）**、**060（K1 copy-before-free + K2 准入守卫，GO）**。

---

## 0. 一句话

把论文 KVMem（arXiv 2609.04852，*Virtualizing Million-Token Agent Workspaces on a Consumer GPU*）的"**KV 工作集虚拟化**"能力实现在我们现役 vLLM 0.29 栈上：**GPU 只保留一个有界的"视窗"，超出部分的历史 KV 分页到 host，每步用当前 query 从 host 索引里检索回相关块、重排进视窗**。目标是让 agent 会话**不再被客户端压缩打断、不再全量重 prefill**，并且顺带把"上下文长度"与"GPU 池大小"解耦（缩池可换显存余量，直接对冲 049 的余量悬崖）。

---

## 1. 立项背景与目标能力

### 1.1 用户指令链（2026-09-29）

1. 读论文 `https://arxiv.org/html/2609.04852v1`，在现役 vLLM 上实现该功能；第一轮只做调研，然后拷问对齐颗粒度。
2. 四轮拷问（见 §9 决策记录），全部按推荐定稿。
3. 追加参考：`github.com/kvmem/kvmem-llama.cpp`（"已有基础功能"的 llama.cpp 移植）——本方案据此**重开并改定了位置策略**。
4. 追加指令：查该仓库 release 日志与 issue（取证见 §4）；**先不开工，把各项计划全部更新、交接文档与交接提示词写详细**。

### 1.2 参考实现三件套（都是 Apache-2.0，都可读源码）

| 参考 | 性质 | 对本方案的价值 |
|---|---|---|
| `kvmem-qw3`（论文作者引擎） | C++/CUDA 自研引擎（QW3），KVMem 是它的一个子系统 | 算法与内核级细节的权威：Mean-K、softmax-over-pages 打分、raw-K 不变式、有界 delta re-RoPE、分层存储 |
| `kvmem-llama.cpp` | **往一个已有引擎里塞 KVMem** 的完整先例（llama.cpp 子模块 + 3 张补丁 + 207 KB adapter） | **与我们处境同构**：引擎侧改动面、接口落点、验证台设计、真实失败清单 |
| 论文 §3–§7 | 方法与实验 | 目标与判据（LongMemEval 32K 预算对 ~115K 问题仅比全上下文低 1.0 分） |

**⭐ 同构性证据（llama.cpp 参考的关键事实）**：

- 引擎侧补丁**仅 +182/−15 行、三张、未改 Flash Attention kernel**：`0001` 在 `llama_model::create_memory` 开工厂钩子（整个 KVMem = `llama_memory_i` 的一个实现，编译开关 `LLAMA_KVMEM` 默认 OFF，vanilla 构建逐位不变）；`0002` 在 `k_norm` 之后、`ggml_rope_ext` 之前插 `kvmem_capture_k/q`；`0003` 保留 KV 洞 + 关掉 mask 的 copy-and-patch 短路。⇒ **引擎侧风险远低于预期，工作量大头在 adapter/连接器**。
- 它**明确否掉了压缩重排路线**：v0.3.0 里程碑记「计划 recency 用窗口坐标 → 实际**保留原始单调 pos，FA + mask 走因果；禁止 pack `[0..W)`**」；架构文档记 "Restore is packed GPU-format memcpy at that orig pos — **no unrotated raw-K and no re-RoPE on the product path**"。它的做法是**在 KV cache 里留洞**、resurrect 的块坐在**原始位置**、洞由 **KQ mask 丢**。
- 它的 16 GiB 配方与实测（RTX 5060 Ti 16 GiB / 32 GiB RAM）：`budget 36,864` / `gen_reserve 16,384` / `ctx 262,144` / **Q8_0 KV**；任务二 = **32 轮工具累积、最终 prompt 261,546 token、总计 262,058/262,144、全部请求成功**；聚合 prefill 首遍 437 tok/s / **有效 242 tok/s**；工具轮 decode **31.74 tok/s**；MTP 接受率 64.70%；整卡峰值 **15,617 MiB**、峰值时可用 **434.90 MiB**、进程 RSS 峰值 **13,483 MiB**。
- 它的**验证台设计比我们原稿更锋利**，本方案直接采纳：**identity canary**（`budget = n_ctx` 时 greedy token 必须与 KVMem-off 完全一致）+ **紧预算 needle 对照**（`budget=256/block=32` 时 recency 必然打不出 needle、retrieval 打出）+ **速度只记录、不自动判失败**。

### 1.3 目标能力（定稿）

| 能力 | 定义 | 现状 |
|---|---|---|
| **A. 接受超长 transcript（主）** | 客户端照旧发全量，服务端把超过视窗的历史虚拟化到 host，GPU 只保留有界视窗；工作区上限 = **262,144 token** | 做不到（生产 `max-model-len 163,072`，超长直接被拒/需客户端压缩） |
| **B. 免客户端压缩、免全量重 prefill（副产品）** | 会话持续增长时只 prefill 差量；不因客户端裁历史而整段重算 | 现有 prefix cache + 8G offload 已覆盖"同前缀增长"的常规情形；**覆盖不了**"客户端压缩后 prompt 变短/变形" |
| **C. 上下文长度与 GPU 池解耦（结构收益）** | 视窗大小成为可调旋钮；缩小视窗 ⇒ 释放显存 ⇒ 直接换余量（049 的主判据） | 现在视窗 = 池容量，绑死 |

**明确不做**（非目标）：长上下文**召回**（在巨文档里找针）——论文作者本人回复"长上下文的召回确实不是这个框架最擅长的……建议 `-c` 小一些（比如 200k 内）"，其定位是 **agent 长程任务**。我们按同一口径：目标是 agent 会话不被压缩打断，不是"262K 巨文档问答"。

---

## 2. 我们的栈：对得上的地方

| 事实 | 位置 / 证据 | 意义 |
|---|---|---|
| 模型 64 层 = **16 层 full attention + 48 层 GDN 线性注意力**（`full_attention_interval=4`） | `config.json` `layer_types` | 与论文/参考的 1/4 比例**完全一致**；只有那 16 层有分页 KV，GDN 层走原路（参考："DeltaNet 层必须排除在所有块/分层逻辑之外"） |
| **`MRotaryEmbedding` + `partial_rotary_factor=0.25`** ⇒ head_dim 256 里**只有前 64 维被旋转** | `model_executor/layers/rotary_embedding/__init__.py:72`、`mrope.py:341-413` | 重 RoPE 只需处理 64 维；位置无关的 192 维永不参与 |
| `mrope_interleaved=True`、`mrope_section=[11,11,10]`，但 `--language-model-only` 下**三轴同位置** | `mrope.py`、`qwen3_vl.py:2752-2754` | 文本路径等价于标准 partial RoPE；**参考实现对多模态块强制走 raw-K 重建（标量 delta 表达不了 H/W 坐标），我们文本路径不需要那套**（vision 面另议） |
| **去 RoPE 用同一张 `cos_sin_cache` 即精确抵消** | 融合 kernel 收的正是 `self.rotary_emb.cos_sin_cache`（`qwen3_next.py:408`） | 不需要参考实现那种"同一段 `__sincosf` 序列"的讲究 |
| "在任意位置写 KV"已有现成先例 | `qwen3_dflash.py:874 precompute_and_store_context_kv`（对调用者指定的 position 集合算 K/V 再 `do_kv_cache_update` 散射进 paged cache） | 装配路径的技术可行性已被生产代码证明 |
| `slot_mapping` 是显式 int64 槽位索引 | `v1/worker/block_table.py:201/397`、`csrc/.../cache_kernels.cu:806`（"key.size(0) can be larger than slot_mapping.size(0)"，负值跳过） | 写 KV 到任意槽位是既有一等能力 |
| 分层搬运的 **Windows 移植层已存在** | 生产 `--kv-offloading-backend native --kv-offloading-size 8`（记录仓 commit `b315e77`+1） | host 层不必新写存储子系统 |
| 连接器接口齐备 | `KVConnectorBase_V1`（`distributed/kv_transfer/kv_connector/v1/base.py:171`）：调度侧 `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta`，worker 侧 `start_load_kv` / `save_kv_layer` / `wait_for_save` | 跨请求存活的 store + 调度/worker 两侧分工，正是 KVMem 的形状 |
| 每步块表可重写的钩子存在 | `AttentionMetadataBuilder.update_block_table`（`v1/attention/backend.py:675`，`supports_update_block_table` 门控，`gpu_model_runner.py:2570` 调用） | 视窗装配的落点 |

---

## 3. 硬约束（决定设计的四条）

### 3.1 块粒度是 1456 token，不是 32/128

混合模型的页对齐强制：`mamba page raw 1,675,264 B` 要 ≈ `attn page 1456×1152 = 1,677,312 B`，所以 `block_size = 1456`、统一页 1,677,312 B（`bytes_per_block(G=8) = 13,418,496`）。**把 block 改小到 128 会让容量掉约 11×**（页字节被 mamba 状态主导，容量 ∝ block_size）。⇒ **搬运与分配的最小粒度只能是 1456 token**；163,072 视窗 = 112 页。

**缓解**：检索**索引**用细粒度子块（128 token，1456/128 = 11 子块/页），**搬运**仍按整页。索引精度是检索质量唯一可调的杠杆（参考实现本身就有 `--kvmem-subblocks` + `sub-block-mean-k` 这条组合）。

### 3.2 GDN 循环状态不能回退

`MambaSpec.num_prefill_checkpoint_blocks = 0`（仅 Kimi-K3 的 KDA 会设）、align 模式每请求 4 个 state 块（1 running + 1 previous/CoW 边界 + 2 投机 scratch）、单条前进状态；`preprocess_mamba` 在 `num_computed_tokens=0` 时直接重头开始；`_preempt_request` 把 `num_computed_tokens=0` 并重排等待队列。⇒ **论文 QW3 的"先 prefill 当前 query → 检索 → 重装配视窗 → 再 prefill 同一 query"流程不能照搬**（会把 query token 在 48 层 GDN 里算两遍）。

**这是本方案选择"固定槽位布局"（§5.1）的直接原因**——见 §5.1 的关键不变式。

### 3.3 显存余量悬崖（049/055）

池建好之后**任何新分配**都会把热权重页挤进系统内存、8k 解码 121→85（−30%）且**不回弹**（049：外部 64 MiB 即触发；055：vis 臂 16 次 boot 里 10 次被挤 260-434 MiB）。参考实现自己的 capture staging 默认是 **448 MB pinned + 448 MB 显存**（issue #55），对我们就是致命量级。

**缓解**：所有新增显存**必须在池创建之前分配、并从池预算里扣出来**（我们本来就显式 `--kv-cache-memory-bytes 3.4e9`，可精确扣）。预算：**192 MiB**（索引 staging 2×64 MiB + 一页重物化暂存 2×25.6 MiB，取整）。

### 3.4 host RAM 与磁盘

主机 RAM **23.1 GiB 总量**；生产 offload 现占 8 GiB。**用户 2026-09-29 裁定：offload 可缩到 2G/1G，不用管 NVMe 方案。**

host 需求估算（nvfp4 KV = 18,432 B/token，16 层）：
- 工作区 262,144 token 的全部 KV = **4.83 GiB**；
- 溢写部分（视窗 163,072 之外的 99,072 token）= V + 位置无关 192 维 K ≈ 15.4 KiB/token ⇒ **1.5 GiB**；
- raw K 权威（只存旋转的 64 维，fp16）= 8,192 B/token × 262,144 = **2.1 GiB**；
- 检索索引（128-token 子块、16 层 × 4 KV head × 256 维 fp16）= **约 67 MiB**。

⇒ 合计约 **3.7 GiB**，落在"offload 缩到 1-2 GiB 后可用"的范围内，**不需要 NVMe**。（对照：参考实现同规模 RSS 峰值 **13.5 GiB**——因为它用 Q8_0 KV 且 32 KiB/token。）

---

## 4. 参考实现的已暴露缺陷清单（open，未修）与我们的规避

> 这是本文最有价值的一节。**不是"它有问题所以我们不做"，而是"它的每条失败都对应我们设计里的一个具体决策"**。

| # | 状态 | 症状（实测） | 根因 | **我们的设计如何规避** |
|---|---|---|---|---|
| **#13** | ✅ closed（rc3 修，作者 09-28 回 "repaired by rc3"） | 144 条消息、纯文本、83,475 token 请求：`mandatory_trim kept=1151 dropped=8 budget=1152` → `E kvmem_fill_slot_info: block 1450 has no GPU slot (pos 46426)` → `failed to prepare KVMem attention ubatches` → **HTTP 200、165 秒、0 内容字符** | `--kvmem-budget` 是硬界；replay/decode 路径没用上已经 stage-in 的数据（报告者原话："retrieval stage-in works, the replay/decode path just never uses it"）。提到 45,056 后同请求 136 s 正常 | 见 #43 |
| **#43** | ❌ **open（0 评论，未修）** | 同根因的**优雅降级版**：`query_replay_fits()` 把强制 replay 范围（sink + 后缀）与**策略预算** `budget_blocks()`（288 块）比，而非**物理池**（416 槽）。`required = 293 > 288` ⇒ replay 跳过 + `mandatory_trim` 恰好丢掉那 5 块（**紧贴 query 之前刚读到的工具输出**）。`first_ms=136,013`（83,909 token **全量重 prefill 136 秒**） | **"query replay"这个概念本身**：query 需要被重新编码，于是要判断 replay 范围装不装得下；而 replay 范围的下界由请求形状决定，压不下去 | **（C）布局里根本没有 replay**：query 在请求开始前位置就定死，只 prefill 一次，工作集在它周围换内容 ⇒ **"跳过 replay → 全量重 prefill"这条退化路径不存在** |
| **#4** | ❌ open（7 评论，作者承认定位） | **>210K 中段串台**：190K/200K 六针全对，210K 后开始**张冠李戴**（问敦煌答滇池的编号）；246K 时能写出"敦煌窟体传感项目的标定档案编号为 ZY-747860"这种半对半错句。有报告者测得 **100% 可复现、两次运行逐字相同、坍缩目标固定**；另一报告者发现**提问顺序是隐藏变量**（同文档 3% 起问 4/6、50% 起问 6/6，同顺序逐位可重复）。作者回复："长上下文的召回确实不是这个框架最擅长的……建议 `-c` 小一些（比如 200k 内）" | 报告者埋 fprintf 确认**检索与选块都正常**（针的块每轮都在 working set、得分 0.24-0.51），"但模型从窗口里读出来的内容就是错的/糊的"——归因**量化副本上反复 in-place delta re-RoPE 累积误差**（机制标注"仍在本地定位，尚未确认"） | **我们不做 delta re-RoPE**：每次入槽**一律从 raw K 单次重建**（§5.3）⇒ **零累积、零漂移**。另外按作者口径把 **200K-262K 标为"已知风险带"**（只记录不判死，§7.2） |
| **#56** | ❌ open（0 评论） | **全量重发会杀服**：池 81,920、`prompt 81,056 + gen` → `mtp_follow`/cache-commit 耗尽 → **CUDA OOM → ggml_abort → 整进程死**；systemd 重启成崩溃循环，还毁掉其他客户端在途请求。原文点名这就是 "an agentic loop with **full-history resend**" | 守卫只查 `llama_n_ctx()`（262,144），**从不查真实池大小**；预填充期的溢写路径在 MTP 跟随池上失效 | **①冷启动 prefill 期 stage-out 列为阶段 1 独立子里程碑（§6 阶段 1a）**；**②我们的客户端正是全量重发**，所以这条不是"别人的坑"而是**我们的必经路径**；③另加一条准入守卫：`prompt + max_tokens > 池` 时返回 400 而不是崩（vLLM 侧本来就是 400，但视窗装配后语义变了，须复核） |
| **#55** | ❌ open | capture staging 按**进程历史峰值**定尺，最大 **448 MB pinned + 448 MB 显存**，**永不随请求自适应** | 实现取进程级高水位 | 我们按 §3.3 扣 **192 MiB 且池前分配**，不做进程级高水位 |
| **#34** | ❌ open | **prefill 单 token 成本 ∝ KVMem 窗口大小**（窗口 53,248 → 2.98 µs/token；28,672 → 1.49，比值 2.00 vs 窗口比 1.86），深度超过窗口后饱和 | prefill 注意力要扫常驻集 | **①缩小视窗不伤检索质量**（同报告：needle 埋在工作集外 93K 仍召回）⇒ 缩池换余量是**双赢**；②视窗大小成为一等调参旋钮（§5.5） |
| **#22 / #82** | ❌ open | Windows 下 **20K→155K 输入解码平在 14-20 tok/s**（#22，RTX 4080 16GB）；**Windows+ROCm 多轮 agent 解码被钉在 12-13 tok/s**（#82），而单发长 prompt 44-56 tok/s | 未定罪；多轮 agent 是最差场景 | **参考实现的性能不可作我们的锚**（我们生产 8k 122.58 / 146k 107.73）。阶段 2 才谈性能，阶段 1 判据不含性能 |
| **#11 / #23 / #73** | ❌ open | Windows **自建**脆弱：MSVC 自建首请求 `0xC0000409` 崩溃（#11，RTX 5080）；build.ps1 默认选项 ~36% prefill/decode 退化 + 24K+ prompt CUDA 崩溃（#23）；rc3 长上下文"multimodal prefill failed or cancelled 然后卡循环预填充"（#73） | 工具链/平台 | **预期 Windows 专项坑**，阶段 1 预留排期；我们整条栈都是 Windows 自建 |

**版本/发布面**：仓库 6 个 release 全是预编译包（Windows/Linux/ROCm/Bonsai），**没有 changelog 形式的 release notes**；真正的能力演进在 `docs/milestones/v0.3.0.md` … `v0.17.0.md` 与 `docs/releases/v0.17.0.md` 里。issue/PR 共 85 条（截至 2026-09-29），其中 **19 条 open**。

---

## 5. 架构设计（定稿）

### 5.1 ⭐ 位置策略 =（C）固定槽位布局

**布局（槽位位置在请求开始前由确定性规则算出，此后本请求内不变）**：

```
[ 0 , S )          sink（最初 S 个 token，按页对齐）
[ S , S+N )        检索槽 N 个（放被选中的历史页，按时间序）
[ S+N , S+N+R )    recent（最近 R 个 token，必须覆盖当前工具输出）
[ B , B+q )        query（本步新增的 prompt 差量）
[ B+q , B+q+g )    生成预留 g
```

**关键不变式（这是整个方案的地基）**：

> **query 的位置 `B = S+N+R` 与"最终选了哪些块"无关** —— 因为检索槽数量 N 是固定的，选择只决定槽里的**内容**，不决定槽的**数量**。

由此一次性消掉三个问题：

1. **不需要重 prefill** ⇒ **GDN 每步只前进一次** ⇒ 不需要 GDN 状态快照/恢复（约束 §3.2 被绕开）。
2. **视窗是稠密连续的**（每个逻辑块都指向真实物理块）⇒ **不需要 mask** ⇒ 生产注意力路径（nvfp4 flashinfer + FULL_AND_PIECEWISE 图模式）**一行不动**。
3. **不存在 "query replay" 概念** ⇒ 参考实现的 #43 / #13 退化路径**结构性不存在**。

**执行流水（每步一次）**：

1. 用轨迹键定位 store，算 incoming prompt 与已存工作区的 **LCP**（§5.4），得出差量 ΔP。
2. 由确定性规则算布局（S/N/R/q/g），**query 位置 B 此刻确定**。
3. 把 LCP 部分从 host store 按槽位装配进 GPU（沿用 store 里已烘焙的位置；**位置未变的块零工作量**），再 prefill ΔP（写进 `[B, B+q)`）。
4. 在 prefill 图里捕获 **RoPE 前的 q 与 k**（§5.2）。
5. 用 query span 的 q 对 host 索引打分（§5.3），选出 top-N 页。
6. 把新入槽的页 stage-in（从 raw K 单次重烘焙到槽位位置）、被淘汰的页 stage-out，**recent 区按新内容重烘焙**（R 很小）。
7. decode。

**三种运行态**：

| 态 | 条件 | 行为 |
|---|---|---|
| **identity** | workspace ≤ `budget_max` | 整段全驻留、零检索、零重烘焙。**必须与 KVMem-off 逐 token 一致**（identity canary，§7.1） |
| **检索** | workspace > `budget_max` | 上述流水；sink + 检索槽 + recent 填满视窗 |
| **冷启动** | 本请求的**新增** token 数 > 池容量（无前缀缓存的首轮长 prompt） | **prefill 期 stage-out**：边 prefill 边把已完成的页溢写到 host（§6 阶段 1a） |

**与参考实现的关键差异（对照表）**：

| 维度 | kvmem-llama.cpp | 本方案（C） | 理由 |
|---|---|---|---|
| 位置 | 原始单调 pos + 洞 + KQ mask | **压缩到槽位 + 重 RoPE** | vLLM 的 mask 路径只支持 fp16/bf16 KV（`FlexAttentionBackend.supported_kv_cache_dtypes = ["auto","float16","bfloat16"]`），我们的生产 KV 是 nvfp4；改用 flex 要么视窗缩到 ~48K（远小于生产 163K，对 ≤163K 的 prompt 反而退化），要么给 nvfp4 新写 mask/稀疏 kernel（动全栈最脆弱处） |
| 重 RoPE | 不需要 | 需要（只动旋转的 64 维） | 上一条的代价；用同一张 `cos_sin_cache` 精确抵消，且**每次从 raw K 单次重建**（零漂移） |
| query 编码时机 | prefill 时基于旧工作集 | 同（不重 prefill） | 与参考共享该性质；参考的 needle 测试证明可接受 |
| 注意力路径 | 未改 FA kernel（靠 mask） | **完全不动** | 视窗稠密连续 |
| 生成上限 | `gen_reserve` 是单次生成的**硬上限**，超了硬失败 | 同（保留该语义，见 §5.5） | 参考的已知限制；我们另加准入守卫 |
| 存储 | 溢写 packed GPU 格式 KV（原样 memcpy 回原位） | 溢写 V + 位置无关 192 维 K（原样）+ 旋转 64 维 K（fp16 raw） | 我们必须重 RoPE，所以要 raw 权威；但只存 64 维 ⇒ 比参考省 |

**被否的两条路线（留档，勿重开）**：

- **(A) 论文/QW3 路线**：先 prefill query → 打分 → 压缩重排 → **重 prefill 同一 query**。需要 GDN 状态快照/恢复（48 层 × 148.68 MiB/份），且带来 #43 同族的"replay 装不下"退化路径。**否**。
- **(B) 参考路线（原始位置 + 洞 + mask）**：被 vLLM 的 KV dtype 支持面卡死（见上表第一行）。**否**（若未来给 nvfp4 写了 mask/稀疏路径可重开）。

### 5.2 捕获钩子（RoPE 前的 q 与 k）

**需要两样东西**：**RoPE 前的 K**（raw K，post-`k_norm`）与 **RoPE 前的 Q**（query span 专用）。**两边都在 RoPE 前捕获 ⇒ 打分完全不需要逆旋转**（索引与 query 同在"内容帧"）。

**我们的两个坑（与参考不同）**：

1. 参考的 `ggml_rope_ext` **不是 in-place**，它用 `ggml_set_output()` 把 RoPE 前的张量挂成图输出、白拿一份副本、零额外 GPU 拷贝。**我们的 `ops.rotary_embedding` 是 in-place 的**（`rotary_embedding/base.py:242-252` 注释明写 "ops.rotary_embedding() is an in-place operation"）⇒ **必须显式 clone**。
2. 生产走**融合 kernel** `fused_qk_rmsnorm_rope_gate`（`qwen3_next.py:403`），norm 与 RoPE 在同一个 kernel 里，**抓不到中间量**。

**阶段 1 决策**：KVMem 变体臂**强制走 eager 路径**（`qwen3_next.py:424-444`：`k = self.k_norm(...)` 一行之后、`self.rotary_emb(...)` 之前 clone 留档），并且**只对 16 层 full attention 留档**。
**代价认下来**：eager 与融合 kernel 数值不完全逐位一致 ⇒ **臂内自洽、不跨臂比 PPL**；性能代价不计入阶段 1 判据。
**阶段 2 优化**：写融合 kernel 的变体（额外落一份 side buffer），去掉 clone。

**成本控制的关键**：raw K **只为"新 prefill 的差量 token"捕获**，不是整个视窗（整个视窗会是 5.4 GiB）。mean-K 也是从差量批算（参考的 `block_kmean_content_batch_kernel` 同思路）。

### 5.3 检索流水线

| 环节 | 定稿 |
|---|---|
| 索引粒度 | **128-token 子块**（一页 11 子块，末页按实际） |
| 索引内容 | 每 (层, 子块, KV head) 一个 **Mean-K**（RoPE 前 K 的均值，fp16 存储，fp32 累加）；总量约 **67 MiB**，host 常驻 |
| 打分 | **softmax-over-pages**（不是原始点积——参考明确说原始 `q·k̄` 会偏向高范数的 sink/早期块）；`logit = q·k̄/√d`，对候选页做 softmax，再对 query token 求和、对 (层, head) 取均值 |
| 页分归约 | **页内子块分取 max**（对齐参考 `--kvmem-group-score-reduce max` 默认；用 sum 会让"一堆平庸子块"的页压过"一个精确命中"的页） |
| query 侧 | **只对当前 query span 打分**（`--kvmem-query-conditioned`），不是全序列 |
| 排除项 | sink 与 recent 从候选里 **mask 成 -inf** |
| 排序 | 选满 N 个后**按时间序**分配槽位（模型看到的是时间序） |
| **执行位置** | **必须在 GPU 上分块做**：参考用 CPU mean-K，32k 时打分要 **4.7 秒**（他们自己标为"最大剩余账单"）。我们的索引 67 MiB 分块过 GPU staging |
| **重烘焙** | **每次入槽一律从 raw K 单次重建**（读 fp16 raw 的 64 维 → 按槽位位置旋转一次 → 重新量化写 nvfp4 的对应 group；192 维与 V 原样拷贝）。**不做 delta re-RoPE** ⇒ **无累积漂移** ⇒ 参考 #4 的串台机制在我们这里不存在，**"8 次 remap / 累计位移阈值"那套参数直接作废** |

**存储形态（host 侧）**：

| 部分 | 格式 | 大小（262,144 工作区） |
|---|---|---|
| V | nvfp4 原样（位置无关） | 9,216 B/token |
| K 的非旋转 192 维 | nvfp4 原样（位置无关） | 6,144 B/token |
| K 的旋转 64 维 | **fp16 raw（重建权威）** | 8,192 B/token |
| Mean-K 索引 | fp16 | 约 67 MiB 总量 |

### 5.4 session 身份与请求协议（客户端零改动）

| 项 | 定稿 |
|---|---|
| 轨迹键 | `hash(root task message 身份) + hash(current query 身份)`（参考做法；map 上限 128 条，满了整体清空） |
| 工作区推进 | **最长公共前缀（LCP）**算差量（复用我们已有的 block hash 链思路） |
| 分歧判定 | prompt **缩水 > max(1024, 1%)** ⇒ 判为新/被压缩轨迹，**冷启动重 prefill** |
| 客户端 | **零改动**（ZCode 照旧发全量） |
| 明确的能力边界 | **客户端侧压缩（prompt 变短）时我们只能冷启动**——这一块能力来自"服务端接受更长 transcript"，不是"救回被压缩的历史"（参考实现同） |

### 5.5 接口落点

| 落点 | 用在哪 | 备注 |
|---|---|---|
| **`KVConnectorBase_V1` 子类**（主） | 跨请求存活的 store（轨迹 → 块元数据：`orig_pos` / `baked_pos` / tier / 槽位）、host 层搬运、准入守卫 | 唯一"跨请求存活 + 调度/worker 两侧 + 已有 Windows CPU 层"的落点。**但要扩展**：`get_num_new_matched_tokens` 只能返回"已算好的 token 数"（前缀语义），表达不了"这些具体块、按这个顺序、放在这些位置" |
| **`AttentionMetadataBuilder.update_block_table`**（补一刀） | 视窗装配：每步重写块表行 + slot_mapping | `supports_update_block_table` 门控（`v1/attention/backend.py:675`） |
| 捕获钩子 | `qwen3_next.py` 的 eager 路径 + clone（阶段 1） | 见 §5.2 |
| 存储后端 | 复用 `v1/kv_offload/` 的 CPU 层（固定槽位、pinned 内存、Windows 移植层已在） | **键从 block hash 改成 (轨迹, 页序号)** |
| 不作候选 | 调度器插件直接改 block table（会动核心调度器；worker 侧块表只有 append/整体 replace 两条路 `gpu_model_runner.py:1462`）；像 DFlash speculator 那样的独立模块 | 风险更高 |

**生成的硬上限（保留参考语义）**：`gen_reserve` 是**单次生成的硬上限**，超了必须优雅失败（返回错误而不是崩）。我们的 reasoning token 会吃预算，所以 `gen_reserve` 取值必须覆盖最长的思考 + 工具调用。

---

## 6. 阶段划分与出口

> 用户指令：**阶段二再冲性能**。阶段 1 判据 = 正确性 + 能跑通，性能**只记录不判死**（对齐参考 v0.3.0 的 "v1 成功标准是能跑 KVMem，速度只记录"）。

### 阶段 1a — 冷启动 prefill 期 stage-out（独立子里程碑，**必须先做**）

| 项 | 内容 |
|---|---|
| 为什么单独 | 参考 #56 就是这个路径上的**杀服级**失败；而我们的客户端每次会话**第一条请求都是冷启动**（无前缀缓存）⇒ 不解决它，阶段 1 的验证台跑不出真实负载 |
| 出口 | 冷 262,144-token prompt（无前缀缓存）能**边 prefill 边溢写**跑完、不崩、不 OOM；准入守卫在 `prompt + max_tokens > 池` 时返回 400 而非崩 |

### 阶段 1 — 正确性打通（文本 + 无投机 + 无图模式 + 池值不动 3.4e9）

| 出口项 | 判据 |
|---|---|
| identity canary | `budget = 不收紧` 时 greedy token 与 KVMem-off **逐 token 一致** |
| 紧预算 needle 对照 | 紧预算（96K 压力配置）下 **recency 必须打不出 needle、retrieval 必须打出** |
| 262K transcript 跑通 | 冷启动 + 多轮增长两条路径都不崩 |
| 163K 配置 needle 多深度 | 工作区 40%/55%/70%/85% 四深度全中，**不退化于现役** |
| 重物化往返单测 | 一块按位移 d 重物化再搬回原位，与原始**量化步内一致**（记录最大偏差） |
| 副判据 | 输出连贯；**视窗容量不缩水**（163,072 不变）；8k/146k 稳态解码**只记录不判死** |
| 中间量（调参抓手） | 检索命中率（选中的页里含 needle 的比例）、每步搬运字节数、每步检索耗时 |

### 阶段 2 — 性能

融合 kernel 捕获（替掉 eager + clone）→ GPU 打分调参 → 重 RoPE 批量化 → 视窗/预算标定。

### 阶段 3 — 叠生产特性 + 缩池治悬崖

DFlash2 投机解码（草稿 KV 与视窗的位置一致性）→ 图模式（FULL_AND_PIECEWISE；注意"元数据分支/状态算子入图会输出塌缩"的既有教训）→ **缩池治悬崖**（把 96K 压力配置正式化：池值降 ⇒ 释放显存 ⇒ 换余量，同时检索变**选择性**的）。

**交付形态**：全程**变体 bat 并存、生产默认不动**（项目惯例；参考实现也是 `LLAMA_KVMEM` 默认 OFF）。

---

## 7. 参数标定与验证台口径

### 7.1 预算切法

| 配置 | 视窗 | 生成预留 | sink | recent | 检索预算 | 留取率（vs 262,144 工作区） |
|---|---|---|---|---|---|---|
| **阶段 1 主配置** | 163,072 | 32,768 | 1,456（1 页） | 动态 [16K, 64K] | 余量（约 65-113K） | 62% |
| **96K 压力配置** | 98,304 | 32,768 | 1,456 | 32,768 | ~31K（约 21 页） | **21%** |
| identity 态 | = workspace | 32,768 | — | — | 0 | 100% |

**为什么主配置留取率 62% 是弱信号**：视窗 163,072 − 生成 32,768 = 130,304 上下文预算 = 89 页，而工作区 262,144 = 180 页 ⇒ 只需要丢掉 38% 的历史，任何随机策略都能留 62%。**所以"检索到底有没有用"必须靠 96K 压力配置来判**（21% 留取率，检索变刚需），主配置只用来证**不退化**。

**`recent` 的尺子（Round 4 定稿，直接来自参考的失败）**：**动态 = 最近一次工具输出的长度，clamp 到 [16K, 64K]**。参考 #13/#43 的根因就是 recent 不够——紧贴 query 的工具输出后缀（37,288 token）超过了 budget（36,864）⇒ replay 被跳过 + 刚读的工具输出被 trim 掉。**budget 下界 = sink + recent + ≥16K 检索（约 96K）**。

### 7.2 工作区上限（Round 4 定稿）

**能力上限 = 262,144**（模型 `max_position_embeddings`），但：

- **200K-262K 标为"已知风险带"**：只记录、不当验收（依据：参考 #4 的 >210K 串台 + 作者本人建议 `-c` ≤ 200K；**注意该机制在我们的设计里不存在**，所以不主动放弃能力，但也不把 200K+ 当已验收）；
- **阶段 1 的闸门放在 ≤200K**；
- 结构事实：**视窗位置是压缩后的 `0..B-1`，RoPE 永远在范围内 ⇒ 262,144 不是代码硬上限，而是 host 预算的配置量**（想放到 512K 只需加 host 预算，但 RoPE 外推是另一件事）。

### 7.3 验证台口径（阶段 1 出口闸门）

| 判据 | 口径 | 出处 |
|---|---|---|
| **identity canary（最强锚）** | `budget = 不收紧` ⇒ greedy token 与 KVMem-off **逐 token 一致** | 参考 v0.3.0 |
| **紧预算 needle 对照** | 紧预算下 recency 打不出 / retrieval 打出 | 参考 v0.3.0 |
| **主判据 = needle 多深度** | 工作区 40%/55%/70%/85% 四深度；复用 046 的 146k needle 口径 | 项目惯例（PPL 非位精确 ⇒ needle 主判） |
| **单测级 = 重物化往返** | 位移 d 后搬回原位，**量化步内一致**（nvfp4 往返要重新量化，不是逐位）+ 记录最大偏差 | 项目"PPL 非位精确"纪律的延伸 |
| **基线 = 客户端压缩** | 造 262,144 token transcript，基线把中段裁到工作区内（等价论文 Compact-only），KVMem 臂跑全量 | 论文 §6 |
| **冷启动路径** | **必须包含"无前缀缓存的冷 262K prompt"** | 参考 #56 |
| **串台检测** | 不只判命中/未命中，还要检测**答成另一根针的编号**（张冠李戴） | 参考 #4 |
| **提问顺序固定 + 重复** | 同文档换提问顺序结果会变（#4 实证）⇒ **跨顺序不可比**，必须固定顺序并重复 | 参考 #4 |
| 副判据 | 输出连贯；视窗容量不缩水；8k/146k 稳态解码**只记录不判死**（阶段 1 关投机关图模式） | 用户指令"阶段二再冲性能" |
| 中间量 | 检索命中率、每步搬运字节数、每步检索耗时 | 调参抓手 |

---

## 8. 风险登记册

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **块粒度 1456 太粗** ⇒ 检索质量（论文是 32-token 块） | 检索可能无效 | 128-token 子块索引；96K 压力配置量化；索引子块大小列为可调旋钮（64 备选） |
| R2 | **冷启动 prefill 期 stage-out 是参考的失败点**（#56） | 杀服级 | 独立子里程碑 1a，先做先验 |
| R3 | 视窗大 ⇒ **prefill 每 token 更贵**（#34） | 吞吐 | 视窗是一等旋钮；缩视窗不伤检索质量 ⇒ 缩池双赢 |
| R4 | **Windows 自建脆弱**（#11/#23/#73） | 排期 | 阶段 1 预留 Windows 专项时间 |
| R5 | **余量悬崖**（049/055） | 性能 | 192 MiB staging **池前分配**；缩池是机会不是风险 |
| R6 | **200K+ 检索质量未知**（#4） | 质量 | 标已知风险带；我们的设计无 delta 漂移 ⇒ 该机制不适用，但要实测确认 |
| R7 | 客户端 prompt 缩水（压缩）只能冷启动 | 能力边界 | 明确写进文档；靠"服务端接受更长 transcript"覆盖主场景 |
| R8 | 阶段 1 eager 路径 + clone 的性能代价 | 性能 | 不计入阶段 1 判据；阶段 2 换融合 kernel |
| R9 | **参考实现性能不可作锚**（12-32 tok/s vs 我们 107-122） | 误判 | 阶段 2 才谈性能，且以我们自己的现役为锚 |
| R10 | 48 层 GDN 的循环状态在"视窗内容变化"时是否仍正确 | 正确性 | 设计上 GDN 不受影响（它不看 attention KV）；但**必须用 identity canary + needle 实测确认** |

---

## 9. 决策记录（四轮拷问）

### Round 1 — 框架

| # | 决策 | 定稿 |
|---|---|---|
| Q1 | 目标能力 | **(a) 接受超长 transcript 为主**，(b) TTFT 作副产品；判据用自家 needle 多深度 + 性能三件套，不引外部 benchmark |
| Q2 | 与论文的保真颗粒度 | **核心骨架**：raw-K 源 + Mean-K 索引 + softmax-over-pages 检索 + GPU/host 两层 + 重 RoPE + 步级刷新。**砍掉** NVMe、prefix-cache 复用、mid-decode 刷新、semantic expansion、GC、quota policy |
| Q3 | 位置策略 | 压缩视窗 + 重 RoPE（**Round 3 修订为 (C) 固定槽位布局**） |
| Q4 | 块/索引粒度 | 索引 **128-token 子块**；搬运按 **1456-token 页** |
| Q5 | host/磁盘预算 | 取代现有 offload 层；NVMe 列第二阶段 → **用户修正：offload 可缩到 2G/1G，不用管 NVMe** |
| Q6 | 交付形态 | **分阶段、变体并存，生产默认不动** |

### Round 1 用户修正

- offload 8 GiB 不是约束，**可缩到 2G/1G**；**不用管 NVMe 方案**。
- **模型本质上只支持 262144 上下文，字节账无意义**。

### Round 2 — 架构

| # | 决策 | 定稿 |
|---|---|---|
| Q1 | 工作区上限与 GPU 预算 | 阶段 1 **池值不动 3.4e9**；扣 **192 MiB** staging 且**池前分配**；缩池治悬崖记阶段 3 |
| Q2 | session 身份与请求协议 | **内容哈希轨迹键 + LCP 差量，客户端零改动**；prompt 缩水 > max(1024,1%) ⇒ 冷启动 |
| Q3 | GDN 状态 / 视窗装配时机 | 一步滞后 → **被 Round 3 的 (C) 取代**（query 位置请求前定死 ⇒ 既不需要重 prefill 也不需要滞后） |
| Q4 | 接口落点 | **`KVConnectorBase_V1` 为主 + `AttentionMetadataBuilder.update_block_table` 补一刀** |
| Q5 | 捕获钩子 | **eager 路径 + 显式 clone**（因我们的 rotary 是 in-place + 生产走融合 kernel） |

### Round 3 — 据 llama.cpp 参考重出

| # | 决策 | 定稿 |
|---|---|---|
| Q1 | 位置策略（重开） | **(C) 固定槽位布局** |
| Q2 | 工作集预算 | 阶段 1 主配置 163,072 + 紧预算压力配置（**Round 4 修订为 96K**） |
| Q3 | 检索与重 RoPE 参数 | q/k 均在 **RoPE 前**捕获；子块 128；页分取 **max**；**softmax-over-pages**；只对 query span 打分；**GPU 分块打分**；staging 192 MiB 池前分配；**重物化一律从 raw K 单次重建（无 delta、零漂移）** |
| Q4 | 验证台 | **identity canary + 紧预算 needle 对照** + needle 多深度 + 重物化往返单测 + 副判据 + 中间量；基线 = 客户端压缩 |
| Q5 | 阶段划分 | 阶段 1 文本/无投机/无图/池值不动；阶段 2 性能；阶段 3 叠投机+图+缩池 |

### Round 4 — 据 issue 取证

| # | 决策 | 定稿 |
|---|---|---|
| Q1 | 工作区上限 | **262,144 为能力上限**；200K-262K 标已知风险带（只记录）；阶段 1 闸门 ≤200K |
| Q2 | `recent` 尺子与 budget 下界 | **动态 = 最近一次工具输出长度，clamp [16K, 64K]**；budget 下界 = sink + recent + ≥16K 检索（约 96K）；紧预算压力配置改 **96K** |
| Q3 | 冷启动 prefill 期 stage-out | **阶段 1，独立子里程碑（1a）** |

---

## 10. 未决与挂账

| 项 | 说明 |
|---|---|
| 缩池治悬崖的具体池值 | 阶段 3；每减 100 MiB 池值换 ~100 MiB 余量、容量少 ~5,050 tokens（049 数据） |
| `recent` 动态尺子的上界 | 需按真实工具输出长度分布标定（当前 clamp 上界 64K 是估值） |
| 索引子块大小 | 128 为初值；64 是备选（更细 ⇒ 更准但索引与打分成本翻倍） |
| **(B) 路线的复活条件** | 若将来给 nvfp4 写了 mask/稀疏注意力路径，可回到"原始位置 + 洞 + mask"（更简单、零重 RoPE） |
| 工作区突破 262,144 | 需要 RoPE 外推（YaRN 等）+ compaction 语义扩展；不在本次范围 |
| vision 面 | 本方案先做文本；vision 的 M-RoPE 三轴坐标需要 raw-K 重建路径（参考对多模态块强制走 raw-K，标量 delta 表达不了 H/W） |
| 多会话/多轨迹 | 参考有 `--kvmem-conversations N`；我们生产 seqs=1 串行，列为后续 |
| 上游回馈 | 本方案的通用部分（如"prefill 期 stage-out"）若验证成功，可考虑回馈参考实现/上游 |

---

## 12. 阶段 1a 实施机制（步骤 057 落地：引擎侧「超池长 prompt 有界 prefill」）

> 本节记录 §6 阶段 1a 的**实现机制**与实测结论。设计层不变（§5.1 固定槽位布局仍成立），本节只回答"超池长的 prompt 到底怎么 prefill 进来"。

### 12.1 问题与选型

阶段 1a 的要求是：冷 262,144-token prompt（无前缀缓存）能边 prefill 边溢写跑完、不崩不 OOM。**纯 vLLM 路径做不到**：请求自己的 KV 块在 prefill 期间不会被回收（`allocate_slots` 只为 `num_computed_tokens + num_new_tokens` 申请，且长 prompt 必须整段驻留才能算后续 token 的注意力），池 3.4e9 只能装 163,719 token ⇒ 262K prompt 永远申请不到块。`scheduler_reserve_full_isl=True` 会让它**卡在等待队列里不动**（不是崩，是挂——比崩更难发现）。

**选型：给 16 个 `full_attention` 层加一个逐层滑窗（`per_layer_sliding_window`）**，窗口 W。于是：

- 每个请求的注意力组需求从 `cdiv(max_model_len, block)` 降到 `cdiv(min(W, max_model_len), block) + 1`（`SlidingWindowSpec.max_admission_blocks_per_request`），**与 prompt 长度解耦**；
- `SlidingWindowManager.get_num_skipped_tokens` + `remove_skipped_blocks` 在每次 `allocate_slots` 前把滑出窗口的块**释放**，驻留量稳定在 ~W；
- 注意力侧由 flashinfer 的 `window_left` 生效（`window_left = W - 1`），**走的是原生 FA2 路径、不是被 §5.1 否掉的 flex mask 路径**，因此 nvfp4 可用（`use_fa2_nvfp4_kv` 在 SM120 上为真）。

**为什么必须逐层传参**：`CacheConfig.sliding_window` 只在模型的 `layer_types` **全部**是 `sliding_attention` 时才被填充（`engine/arg_utils.py:2061-2067`），本模型是 16 `full_attention` + 48 `linear_attention` 的交错混合 ⇒ 全局滑窗会被引擎主动拒绝，只能走 `per_layer_sliding_window`。

**实现**：`model_executor/models/qwen3_next.py` 新增 `_kvmem_per_layer_sliding_window()`，`VLLM_KVMEM_SW_WINDOW=N` 时对 `layer_types[i] == "full_attention"` 的层返回 N，其余层返回 None（**草稿模型自带 `sliding_attention` 层，不受影响**）。**未设环境变量时逐字节等价于上游**（返回 None，行为与改前一致；实测生产 boot 布局数字与改前逐项相同）。

### 12.2 实测（步骤 057，臂 = `_tmp_line_b/serve_kvmem_sw32k.cmd`：W=32768 / `--max-model-len 262144` / 无投机 / `--cudagraph-capture-sizes 1`）

| 项 | 生产（046 配置） | KVMem 臂（W=32,768） | KVMem 臂（W=163,072，阶段 1 主配置） |
|---|---|---|---|
| attn block size | 1456 | 1424 | 1424 |
| mamba 页 padding | 0.12% | 0.38% | 0.38% |
| `GPU KV cache size` | 163,719 tokens | 1,060,864 tokens | 275,997 tokens |
| 单请求并发（对 `max_model_len`） | 1.00× @163,072 | **4.05× @262,144** | **1.05× @262,144** |
| 冷 210K prompt prefill | 不可行（>池） | **152.2 s**（~1,360 tok/s） | **249.1 s** |
| 冷 256K prompt prefill | 不可行 | **185.2 s**（~1,383 tok/s） | **320.2 s**（~800 tok/s） |
| needle 窗内（95% 深） | — | **命中** | **命中** |
| needle 窗外（10% 深） | — | **未命中（预期）** | **未命中（预期）** |
| 8k needle | 122.58 | 69.4（无投机，capture size 1） | — |
| 余量悬崖判据（共享 − 8,298） | 0-68 MiB | 0 MiB | 0 MiB |
| offload 溢写量（200K 请求） | — | **314.7 MB** | — |

**结论**：①超池长 prompt 的**有界 prefill 机制打通**（阶段 1a 的"不崩不 OOM"半边达成）；②**窗外 needle 必失**是这套机制的**正控**——它证明窗口真的在生效，而不是"其实做了全注意力所以碰巧能跑"；③窗口注意力让 prefill **更快**（W=32,768 时 ~1,370 vs 生产 867 tok/s），与参考 #34「prefill 单 token 成本 ∝ 视窗」同向；**但窗口越宽越慢**（W=163,072 时 256K 冷 prefill 320 s ≈ 800 tok/s，已略低于生产 867）⇒ **窗口大小是"检索质量 / prefill 速度"之间的一等旋钮**，也说明 §7.1 的主配置窗口（163,072）是**上限**而不是最优值；④**`W=163,072` 在池 3.4e9 上可行但只剩 1.05× 并发余量**——池值不能动（035 硬上限）⇒ 若要给视窗留更多余量，只能缩窗口（而这恰好是 §5.5 说的"缩池换余量"的同一枚硬币）。

### 12.3 与设计的关系（重要边界，勿误读）

- **滑窗 ≠ 工作区存储**。滑窗把滑出窗口的历史 KV **丢掉**，只保留 ~W；offload 层只把"滑窗可达的那几块"（~300 MB）写进 host mmap，**不是** §5.3 要求的完整工作区（V + 非旋转 K + raw K 权威）。⇒ **阶段 1a 的"溢写"只做到一半**，KVMem 的 host 工作区仍需要**自己的 copy-before-free 钩子**（落点 = `SingleTypeKVCacheManager._remove_blocks_in_range`，`block_pool.free_blocks(freed)` 之前；键从 block hash 改成 `(轨迹, 页序号)`；复用 `v1/kv_offload/` 的 CPU 层与 `OffloadingConnector` 的 pending-job 记账）。
- **W 必须 ≥ 装配后的视窗**。装配后的视窗位置是压缩后的 `0..B-1`，若 `B ≤ W` 则滑窗对 serve 请求完全不生效（sink 不会被淘汰）。阶段 1 主配置视窗 163,072 ⇒ **KVMem 臂的 W 取 163,072**（**已实测：`275,997 tokens / 1.05x @262,144`，冷 256K 320.2 s 跑通**）；本次另用 32,768 证明"有界"这一性质本身。**注意窗口越宽 prefill 越慢**（256K 冷 prefill：32,768 → 185 s，163,072 → 320 s）⇒ 窗口大小是"检索质量 / prefill 速度 / 池余量"的三向旋钮，阶段 2 标定时应把它当一等参数。
- **滑窗只用于 ingest 相**。serve 相的位置是压缩后的槽位坐标、视窗稠密连续、**无 mask**（§5.1 不变式），与滑窗互不干扰。
- **不改 §5.1 的三条收益**：仍然不重 prefill、仍然无 GDN 快照、仍然无 mask（滑窗只出现在 ingest 相，而 ingest 相本来就要重算/丢弃）。
- **raw K 捕获（§5.2）与滑窗正交**：仍需在 RoPE 前 clone 留档，且只对差量 token 捕获。

### 12.4 下一步（阶段 1a 剩余工作）

1. ~~**W=163,072 复验**（阶段 1 主配置的窗口；判据 = 冷 262K prompt 仍能跑完 + 池不爆）。~~ **✅ 已完成（057 收尾补跑）**：`275,997 tokens / 1.05x @262,144`、冷 256K 320.2 s 跑通、窗外 needle 仍未命中。**只剩 5% 并发余量**是本配置的紧处（池值不能动）。
2. ~~**1a-2：copy-before-free 钩子**（把滑出窗口的块写进 KVMem 工作区，而非丢弃）——这是"溢写"的另一半。~~ **✅ 已完成（060）**：见 §12.5。
3. ~~**准入守卫复核**：`prompt + max_tokens > 工作区上限(262,144)` 时返回 400。~~ **✅ 已完成（060）**：见 §12.6，**并修正了本条的前提**——上游会把 `max_tokens` 夹到 `上限 − prompt`，所以真正的形态是"夹紧 + 守卫兜底"。

---

## 12.5 阶段 1 K1 实施机制（步骤 060：copy-before-free 落地）

### 12.5.1 为什么需要自己的钩子（而不是复用 offload 连接器）

057 的实测已经把话说清楚了：滑窗确实把历史 KV **丢掉**了，`OffloadingConnector` 只存"滑窗可达的那几块"（200K 请求期间 314.7 MB），**不是**完整工作区。原因在触发口径——那个连接器的 store 触发是"**算完一个 chunk** 就按 **block hash** 存"，而滑出窗口的块在它来得及建 job 之前就已经从请求的块表里被抹掉、还回池子了。

⇒ 需要的是"**淘汰即溢出**"：在块**回池之前**把它交出去。落点只有一个：`SingleTypeKVCacheManager._remove_blocks_in_range`（`v1/core/single_type_kv_cache_manager.py`，滑窗组由基类实现，`SlidingWindowManager` 没有覆写 `remove_skipped_blocks`）。

### 12.5.2 五处改动

| 层 | 落点 | 做什么 |
|---|---|---|
| 核心管理器 | `single_type_kv_cache_manager.py` | `_remove_blocks_in_range` 里**把块置 null 之后、`free_blocks` 之前**调 `_retain_for_workspace(req, idx, block)`；返回 True 则**不释放**（ref_cnt 不递减 ⇒ 池子拿不到它），记入 `_pending_workspace_evictions`。门控 `VLLM_KVMEM_WORKSPACE`（转调 `envs`），且只对 `SlidingWindowSpec` 组生效；未设时恒 False |
| 协调器 | `kv_cache_manager.py` | `take_workspace_evictions()` → `({req_id: [(group_id, block_id, page_index)]}, retained_blocks)` |
| 调度器 | `sched/scheduler.py` + `sched/output.py` | `KVConnectorBlockState` 加 `workspace_evictions`；drain 后立刻 `connector.register_workspace_retained_blocks(retained)`，**connector 不接受就当场释放**（退化成改前行为，不泄漏） |
| 连接器基类 | `kv_transfer/kv_connector/v1/base.py` | 新增 `register_workspace_retained_blocks(blocks) -> bool`（默认 False 拒绝） |
| 新子系统 | `v1/kvmem_workspace/` + `kvmem_connector.py` | `config` / `groups` / `metadata` / `manager`（调度侧）/ `worker`（worker 侧）/ connector；已注册进 `KVConnectorFactory` |

**connector 选型**：不改造 `OffloadingConnector`（触发口径与键都不对，且与生产前缀缓存 offload 共用会互相污染），新写 `KVMemConnector`，**并借 `envs.VLLM_KVMEM_WORKSPACE` 顶掉 `config/vllm.py:_post_init_kv_transfer_config` 里的 offloading 槽位**——于是 KVMem 臂与生产 launcher 的差异只有环境变量，`--kv-offloading-backend native --kv-offloading-size 8` 仍保留（进入该配置路径用），但 `KVMemConnector` 忽略 `cpu_bytes_to_use`，**8 GiB 前缀缓存区不再分配**（设计 §3.4 的"工作区取代 offload 层"）。

### 12.5.3 工作区身份：`(轨迹, token 偏移)`

- **轨迹键** = 前 `VLLM_KVMEM_TRAJ_PREFIX`（默认 512）个 prompt token 的 blake2b-16。客户端照旧全量重发 ⇒ 前导 token 就是"根任务消息"：会话增长时不变、换会话即变，**客户端零改动**。实测两轮同 nonce（210K → 230K）落在**同一条** `a6de2ec47e98`，冷启动那轮是另一条 `b3d7ff46c37`。
- **页键** = `(轨迹, page_index × block_size)`，即 **token 偏移**而非页序号。这一条是本步修掉的一个真缺陷：首版键里带 `group_id`，而本模型 16 层 full_attention 因 `VLLM_KV_GROUP_SIZE=8` 被分成 **2 组**，于是同一个逻辑页被分到 2 个 slot，**3 GiB 只装得下 61 个逻辑页**（冷 210K 实测 dropped=48）。改成 token 偏移后同 token 段的两个组**共用一个 slot**，容量翻倍（冷 258,854 prompt 只吃 94/122 slots、dropped=0）；块大小不同的组会自然落到不同 slot，不会误共享。
- **容量口径（重要）**：工作区**只需装"窗口外的部分"**——262,144 − 163,072 = 99,072 token ≈ 70 逻辑页 ≈ **1.83 GiB**，所以 3 GiB 预算有余量。而"整个 262K 工作区整页存"要 **4.4 GiB**（设计 §3.4 的 3.7 GiB 依赖 K3 的**拆分存储**：只存 V + 非旋转 192 维 K，旋转 64 维从 raw K 重建）。
- **丢页不静默**：工作区满时逐页 `WARNING` + `pages_dropped` 计数，并在每个请求结束时打一行 summary（含**独立期望值** `(num_tokens − W) // block_size`，使"淘汰量 = 窗口外页数"可一行核对）。实测：198,209-token prompt → 50 entries / 25 logical slots，日志 `window expects [25, 25]` ✓；4,144-token prompt → 0 entries、`expects [0, 0]` ✓。

### 12.5.4 搬运与完成回传

- **host 区**：每个（组 × 层）一个 `(num_slots, page_bytes)` int8 张量，`pin_memory=True`（失败则退化成 pageable 并告警——正确性不变，只是慢）。实测每层页 **1,640,448 B**（= 1424×1152，**未 padding**），122 slots，**2.98 GiB pinned 分配成功**。
- **搬运原语**：`ops.swap_blocks_batch`（本平台走 `_WIN_BATCH_MEMCPY_BROKEN` 的 `cuMemcpyAsync` 循环）。在 **`wait_for_save`（forward 之后、同一条流）** 上发出，因此严格排在写 KV 的 kernel 之后；`page_index` 由 `_remove_blocks_in_range` 给出的是**已提交**的页（`processed_computed_tokens` 口径），所以不存在"拷到正在写的页"。
- **完成回传**：每个 job 记一个 `torch.cuda.Event`，`get_finished` 里 `query()` 为真才把 job id 放进 `KVMemWorkerMetadata` 交回调度器；调度器在 `update_connector_output` 里**释放该 job 扣住的块**（每块恰好一次）。**这就是"拷贝完成前禁止回池"的实现**——不是"等 N 步"的启发式。
- **往返自检**（`VLLM_KVMEM_SELFTEST=1`）：对前 8 页 × 16 层做双向比对——①从 GPU 页**重新**拷一份到 host，必须等于 store 写下的内容；②store 的内容经 host→GPU→host 必须逐字节不变。三次 boot 全 **`byte-identical, 0 mismatch`**。这是 K1 出口"历史页可被重新加载"的机制性证明；**把页按新位置重烘焙进视窗是 K3**。

### 12.5.5 实测（步骤 060）

| 项 | 数据 |
|---|---|
| connector 选择 | `Creating v1 connector with name: KVMemConnector`；`KVMem workspace stores kv cache group(s) [6, 7] (sliding_window=[163072,163072], block_size=[1424,1424])` |
| 布局（未变） | attn block 1424 / mamba pad 0.38% / `275,997 tokens / 1.05x @262,144`（与 057 逐字相同） |
| 工作区 | 122 host slots / 3.00 GiB / 25.03 MiB per slot；pinned 分配成功（每组 1.49 GiB） |
| 冷 210K | TTFT 249.2 s（057 = 249.1 s）、needle 命中、**66 entries 淘汰 → 66 stored → 0 dropped**、33 个 job 各"released 2 block(s)" |
| 冷 258,854（≈上限） | 每请求 69 逻辑页、**94/122 slots、0 dropped**（`ignore_eos` 输出跑满 3,290 token） |
| 冷 198,209 | 50 entries / 25 slots，`window expects [25, 25]` ✓ |
| 往返自检 | 8 页 × 16 层 **byte-identical，0 mismatch**（三次 boot） |
| 跨请求轨迹 | 同 nonce 两轮（210K → 230K）**同一条轨迹**；冷启动另一条 |
| 泄漏/静默丢 | 0 次 "workspace is full"（修缺陷后）、0 次 "was not registered as retained"、0 次 "had no eviction entry" |

### 12.5.6 本步的边界（勿误读）

- **只做到"存得下、取得回、不泄漏"**：重物化到槽位、Mean-K 索引、softmax-over-pages 检索、128-token 子块**全未开工**（K3）。
- **工作区只增不减**：满则计数丢页，**无 LRU/GC/跨会话淘汰策略**（阶段 1 只判正确性）。
- **多轨迹未测**：`--max-num-seqs 1` 串行；参考实现有 `--kvmem-conversations N`。
- **性能不判**（用户指令"阶段二再冲性能"）。本臂无投机，数字不可与生产比。
- **余量判据要换口径**：`tools/prod_headroom_check.ps1` 的 `共享 − 8,298` 在本臂上**不可直接用**——本臂没有 8 GiB 前缀缓存 mmap，共享基线是 4,198 MiB，会算出 −4,100 的假读数。

---

## 12.6 阶段 1 K2 实施机制（步骤 060：准入守卫）

| 上限 | 实现 | 实测 |
|---|---|---|
| 工作区上限 262,144 | `--max-model-len` 的 `_validate_prompt_len`（既有） | `prompt > 262,144` → **HTTP 400**（"maximum context length is 262144 tokens ... total of at least 262145"） |
| `prompt + max_tokens` | **上游夹紧**：`renderers/params.py` 的 `TokenizeParams` 把 `max_total_tokens − max_length` 当输出上限，`_tokens_len_check` 再按 `max_length` 卡输入 | `prompt 258,854 + 请求 20,000（ignore_eos）` → **completion 3,290，total = 262,144 恰好等于 `max_model_len`** ⇒ 工作区**不可能被越过** |
| 同上（兜底） | `input_processor.py` 新增守卫：`VLLM_KVMEM_WORKSPACE` 开启且 `prompt + max_tokens > VLLM_KVMEM_WORKSPACE_TOKENS` → `VLLMValidationError`（400） | HTTP 路径被夹紧后不触发；**对绕过夹紧的路径（直接 `SamplingParams`）生效** |
| 视窗装不下 | `KVMemWorkspaceScheduler.bind_gpu_block_pool` 按 `SlidingWindowSpec.max_admission_blocks_per_request` 求和对比 `len(block_pool.blocks)`，不足则 **boot 期 RuntimeError**（明确 admission error，**不让请求在等待队列里等死**） | `viewport needs 234 block(s) ... pool has 259`（234 = 2×117，117 = `cdiv(163072−1+1024,1424)+1`）⇒ 通过；**234/259 只剩 10% 余量**是紧处 |
| 控制组 | — | 4,144-token prompt → 200 且 needle 命中（守卫不误杀合法请求） |

**修正计划的一处前提**：计划写"`prompt + max_tokens > 上限` 返回 400"，实测**上游先夹紧**（而不是报错），所以真正的形态是"**夹紧 + 守卫兜底**"；结构目标（工作区不被越过）达成，且不会像参考实现 #56 那样杀服。

---

## 12.7 阶段 1 K3 前半实施机制（步骤 061：raw-K 捕获 + Mean-K 索引 + 页级检索打分）

> 本节记录 §5.2（捕获钩子）与 §5.3（检索流水线）中**读取侧**前三项的落地机制与实测。**重物化（§5.3 的"重烘焙"与 §5.1 的槽位装配）未做**——本步只回答一个问题：**按 1424-token 页粒度、128-token 子块索引，检索到底能不能把含答案的那一页挑出来？**（风险登记册 R1。）

### 12.7.1 为什么先做"打分"而不是先做"重物化"

R1 说得很直白：论文用 32-token 块，我们的块粒度被引擎钉在 1456/1424 token，**检索可能根本无效**。如果检索无效，重物化（位置解耦 + 重 RoPE + 块表改写，本线最重的一块工程）就是白做。所以顺序是**先用一次 prefill 量出检索质量，再决定要不要建重物化**。

因此本步的产物是一个**策略无关的测量工件**：引擎每个 prompt prefill 完成时，把「每一页的页级 logit（+ 每个 (模式, 粒度) 变体）」写进 `VLLM_KVMEM_DUMP`，由客户端探针（`tools/kvmem_k3_probe.py`，它才知道针在哪一页）判排名。

### 12.7.2 落点（七处）

| 层 | 落点 | 做什么 |
|---|---|---|
| 模型 | `model_executor/models/qwen3_next.py` | `_kvmem_rawk_layer()` 门控；命中时**关掉融合 kernel**（`use_fused_qk_norm_rope_gate=False`）；`_project_qkv_gate` 的 eager 分支里 `k_norm` 之后、`self.rotary_emb` 之前调 `capture.record(...)` |
| 捕获 | `v1/kvmem_workspace/capture.py`（新） | 按层暂存（`positions.clone()` + `q[-span:].clone()` + `k.clone()`），`drain()` 在 forward 之后做 host 拷贝 |
| 调度侧 | `v1/kvmem_workspace/manager.py` | 每步下发 `KVMemStepSpan(trajectory, start, num_tokens)`；prompt 完成的那一步下发 `KVMemScoreRequest`（判据 = `start < prompt_len ≤ start + num_tokens`，`prompt_len` 在 `update_state_after_alloc` 时固定，**不能用会随 decode 增长的 `request.num_tokens`**） |
| 元数据 | `v1/kvmem_workspace/metadata.py` | `KVMemStepSpan` / `KVMemScoreRequest` 加入 `KVMemConnectorMetadata` |
| 索引 | `v1/kvmem_workspace/index.py`（新） | 轨迹键 → 每层 fp32 子块和 + 计数；打分时归约到各粒度、各模式 |
| worker | `v1/kvmem_workspace/worker.py` | `wait_for_save` 里 `capture.drain()` → 按位置区间归属轨迹 → `index.add`；`KVMemScoreRequest` → `index.score` → 写 JSON |
| 配置 | `v1/kvmem_workspace/config.py` + `envs.py` | `VLLM_KVMEM_RAWK`、`VLLM_KVMEM_INDEX_SUBBLOCK`、`VLLM_KVMEM_SCORE_GRANULARITIES`、`VLLM_KVMEM_SCORE_MODES`、`VLLM_KVMEM_QUERY_SPAN`、`VLLM_KVMEM_RECENT`、`VLLM_KVMEM_TOPN` |

**位置区间是必须的**：捕获只拿到绝对 `positions`，不知道属于哪条轨迹；`build_connector_meta` 在 `_update_after_schedule` **之前**调用（`scheduler.py:1379` vs `:1398`），所以那时 `request.num_computed_tokens` 仍是本步起始位置，能可靠地组成 `[start, start+n)`。

### 12.7.3 捕获的四条限制

1. **只对 16 层 `full_attention`**：48 层 GDN 没有 KV cache 可索引。
2. **只对 `k.shape[0] > 1` 的步**：decode 步的单 token 不会成为工作区页，且单 token 步正是走图/编译的那一步。
3. **必须 `clone`**：`ops.rotary_embedding` 是 in-place（`rotary_embedding/base.py:242-252` 注释明写），不 clone 拿到的就是旋转后的值。
4. **必须关掉融合 kernel**：生产走 `fused_qk_rmsnorm_rope_gate`，norm 与 RoPE 在同一 kernel 里，抓不到中间量。代价 = **臂内数值自成一套，禁止跨臂比 PPL**（设计 §5.2 已认下）。

**且 `record()` 不能出现在任何编译/图区域内**——首 boot 实测 `torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases`（AOT fullgraph 把 `capture.record` 连同一行 `logger.info` 一起吃了）。修法有二：把日志搬出 `record()`（搬到 `drain()`），以及**本臂改用 `--enforce-eager`**——这正是设计 §6 对阶段 1 的要求（"文本 + 无投机 + **无图模式**"）。**副作用：本臂的 prefill/decode 速度不可与 057/060 比较**（那两个臂是带编译跑的）。

### 12.7.4 索引与打分

- **存储粒度 = 最细粒度**（`VLLM_KVMEM_INDEX_SUBBLOCK`）。粗粒度由细粒度**求和**得到（和相加、计数相加 ⇒ 均值精确），所以**一次 prefill 可以报告多个粒度**，不必一轮一个点。单测已证：32-索引导出的 `dot@128` 与 128-索引直接算的 `dot@128` 最大差 **6.99e-10**。
- 内存：子块 32 时 8192 子块 × 16 层 × 1024 维 × 4 B ≈ **537 MB/轨迹**（host，只增不减）。
- **打分**：`logit = q·k̄/√d`（`dot`）或先各自归一化（`cosine`）→ 对 query span 取均值 → 页内子块取 **max** → 对候选页 softmax、对 query token 求和。`dot`/`cosine` × 32/64/128 共 6 个变体同时输出。
- **候选集**：排除 sink（首页）与 recent 尾部（`VLLM_KVMEM_RECENT`，默认 32768 = 23 页）。

### 12.7.5 实测（步骤 061）

**臂** = `tools/serve_gsq_kvmem_ws163k.cmd`（池 3.4e9 / `--max-model-len 262144` / `W=163072` / **`--enforce-eager`** / 无投机 / 无 offload 8 GiB 区），**探针** = `tools/kvmem_k3_probe.py`，**离线复核** = `tools/kvmem_k3_analyze.py`，**单测** = `tools/kvmem_index_test.py`。prompt = 198,3xx token（≈140 页），query = **聚焦问题重复 8 次**（使最后 256 token 几乎全是问题本身）；`recent=32768` ⇒ **候选 116 页**；针 = 15 token。

| 项 | 数据 |
|---|---|
| 捕获 | `KVMem raw-K capture armed (num_heads=24, num_kv_heads=4, head_dim=256, query_span=256)`；首次 drain = **16 层 × 1024 token + q tail 256 行**（每步 ≤ `max_num_batched_tokens`） |
| 索引覆盖 | `stored_subblocks` 1549 ≈ 198231/128 ✓；`num_pages` 140 = ⌈198231/1424⌉ ✓ |
| 打分耗时 | 6 个变体（dot/cosine × 32/64/128）单次 ≈ 秒级（含在 prefill 收尾里） |
| 冷 200K prefill | **TTFT ≈ 266 s**（≈745 tok/s；带编译的 057/060 是 ~249 s ⇒ eager 代价约 +7%，**不可直接比较**） |

**检索质量（同页有针 vs 无针，页 0..68 在两次请求里 token 完全相同 ⇒ 可直接对减）**：

| 变体 | d16（针在 **page 19**，**窗口外**，偏移 28,366） | d50（针在 **page 68**，窗内，偏移 97,064） |
|---|---|---|
| `dot@32` | 无针 rank 8 / logit 19.66 → **有针 rank 2 / 25.57（Δ+5.91）** | 无针 rank 28 / 17.64 → **有针 rank 1 / 26.83（Δ+9.19）** |
| `cosine@32` | rank 8 → **rank 2**（Δ+0.26） | rank 24 → **rank 1**（Δ+0.39） |
| `dot@64` | rank 9 → **rank 2**（Δ+6.76） | rank 40 → rank 2（Δ+7.85） |
| `dot@128` | rank 10 → **rank 7**（Δ+3.00） | rank 28 → rank 2（Δ+7.64） |
| `cosine@128` | rank 8 → rank 5（Δ+0.15） | rank 29 → rank 2（Δ+0.36） |

**读数**：
1. **检索确实有信号，而且不小**：针只占一页 1424 token 里的 15 个，却把该页的页级 logit 抬高 **+5.9（窗口外）/ +9.2（窗内）**，排名从 8→2 与 28→1。**6 个变体全部把针页放进 top-16**。
2. **粒度是一等旋钮，设计初值 128 偏粗**：d16 上 `dot@32` rank 2 vs `dot@128` rank 7，趋势 **32 > 64 > 128**，与 R1 的预测（论文 32-token 块 vs 我们的页）一致 ⇒ **后续以 32 为准**（代价：索引 537 MB/轨迹 host 常驻）。
3. **`cosine` 不解决偏置也不提升排名**：它在 d16/d50 给出与 `dot` 相同的名次（2/1），只是把 logit 压到 [0,1]。⇒ **残余偏置不是"页均值范数"效应**。
4. **残余"早期页偏置"是真实的、且成因未明**：无针控制组的 logit 前列是 **页 2 / 5 / 1 / 11 / 8 / 3（22.7–26.2）**，把 top 槽位占掉了；而页均值范数几乎恒定（19.85–23.73，中位 20.83，Pearson(norm, logit)=0.474，范数只变 ±7% 而 logit 变 3×）⇒ **不是范数**。**这是本步留下的头号挂账**（它直接吃掉检索槽位）。

**两个查出来并修掉的真缺陷（都属测量层，都会给出错误结论）**：
1. **页内归约走错页边界**。`block_size // granularity` 分组假设粒度整除页长，而 **1424 = 16×89，32/64/128 都不整除**：128 子块时每页偏 **16 token**，到第 19 页引擎的"页 19"已偏离真实页 **304 token**。同一批数据在修前读出来是"rank 74/116 阴性"，修后是"rank 2/116 阳性"——**修前那版结论已作废**。修法 = 按 `token 偏移 // block_size` 归属（`_page_reduce`，`np.maximum.reduceat`），并承认**子块可跨页**（1424/128 = 11.125 ⇒ 每 8 页有一个跨界子块，其尾部内容不计入下一页）。单测已钉死：标记 token 放在页 1/19/70 三处 × 三粒度，top 页必须等于 `offset // block_size`（9/9 通过）。
2. **AOT fullgraph 编译拒绝捕获**。首 boot 直接 `torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases`（`capture.record` 连同一行 `logger.info` 被吃进编译区）。⇒ 日志搬出 `record()`（搬到 `drain()`）+ **本臂改 `--enforce-eager`**（设计 §6 对阶段 1 本来就要求"无图模式"）。

**单测（`tools/kvmem_index_test.py`，合成数据，先于引擎跑）**：①子块均值 = 喂入行的算术均值（max diff 6.1e-5 = fp16 量化）；②相邻子块计数正确、未触及子块恒零；③sink 页 0 与 recent 尾页**不在候选集**；④打分两次调用逐位一致；⑤**粗粒度 = 细粒度精确求和**（32-索引导出的 `dot@128` vs 128-索引直接算的 `dot@128`，max diff **6.99e-10**）；⑥`cosine` 模式同样把针页排第一；⑦页边界对齐 9/9。

### 12.7.6 本步的边界（勿误读）

- **只做到"存得下、取得回、不泄漏、能排序"**：**没有任何页被放回视窗**，所以窗外 needle 仍然**答不出来**（`needle_hit=false` 是预期）。dump 说的是"检索**会不会**找到"。
- 本臂**无投机、无图模式**（阶段 1 判据不含性能）。
- 工作区与索引**只增不减**（无 LRU/GC）；多轨迹未测（`--max-num-seqs 1`）。
- **臂内自洽**：eager 路径与融合 kernel 数值不逐位一致，禁止跨臂比 PPL（设计 §5.2）。
- **`dot@32` 是当前最好变体，但"最好"仅指把针页送进 top-2**；残余早期页偏置仍占着 top 槽位（§12.7.5 读数 4），**检索槽位可用性未解决**。

### 12.7.7 本步留下的挂账（按优先级）

1. **残余早期页偏置的成因**（读数 4）：无针时页 1/2/5/8/11 就有 22.7–26.2 的 logit，把检索槽位占掉。已排除"页均值范数"（范数恒定 ±7%、Pearson 0.474）。候选解释：文档是 3 个单元循环 16 次 ⇒ 早期页更"模板化"；或 `k_norm` 的**学习权重**在某些维度上系统性偏置。**已由步骤 062 结案（§12.8）**：成因 = query 的共同成分（模板）与 k̄ 的位置成分（PC2，与 offset 相关 +0.797）对齐；候选解释**两条都被排除**（不是维度偏置，也不是"早期页更模板化"）；**"去偏置"判 NO-GO**（背景与信号共享子空间，见 §12.8.4）⇒ 不再是阻塞项。
2. **重物化（§5.1 槽位装配 + §5.3 重烘焙）**：位置解耦 + 重 RoPE + 块表改写。已知的唯一缝 = `gpu_model_runner.py` 的 `positions`（2204-2207 重建、2213 派生 `slot_mapping`、无任何现成钩子），且 `update_block_table` 只在 flash_attn/mamba 后端实现、**flashinfer（nvfp4 路径）没有**。这是本线最重的一块工程。
3. **查询 span 的自动界定**：本步用"尾部 256 token"当查询；真实 agent 轮的 delta 会更聚焦。设计 §5.4 的 LCP 差量才是正解。
4. **索引只增不减**：无跨会话淘汰；`num_subblocks` 按 262,144 上限预分配（32 子块时 537 MB/轨迹）。

---

## 12.8 阶段 1 K3 后半第一步（步骤 062：残余早期页偏置的成因、危害与"不可修"的判定）

> 本节回答 §12.7.7 的第 1 条挂账（本线头号）。结论是**成因查清、危害量化、"去偏置"路线 NO-GO**——它不阻塞重物化，但把阶段 1 出口的判据从"先修偏置"改成"多深度 needle 实测端到端"。

### 12.8.1 测量通道（`VLLM_KVMEM_DUMP_KBAR`）

排名的"为什么"必须能从一次 200K ingest（~4.5 分钟）里反复分析，所以 `index.py` 在打分时把向量写进与报告同名的 sidecar `.npz`：**页级 mean-K**（`_layer_page_mean`：按真实页边界把子块和/计数分开累加再相除，不是"均值的均值"）、**子块级 mean-K + 计数**（引擎的页分是"页内子块取 max"，针只占一页 1424 token 里的 15 个 ⇒ **页均值把信号稀释 100 倍，只有子块级能重放引擎的归约**）、**query**（`q_group_mean`，打分实际用的 head-group 均值）、**子块级与页级 logit**。默认关闭；`tools/kvmem_index_test.py` 用合成数据真值做外部锚（页均值 9.4e-7、q 0、子块均值 6.3e-6、离线重放 3.4e-8）。

`tools/kvmem_k3_replay.py` 离线**逐位重放**引擎的 max 归约（每 query token 单独打分 → 页内子块 max → 层求和 → span 均值；**顺序必须与引擎一致，max 与 span 均值不可交换**），与引擎报告差 **3.3e-4**（fp16 量化级）。**这是"离线结论 = 引擎结论"的前提**，也是本步所有修法评估的基准。

### 12.8.2 成因（三条独立证据）

1. **与 query 无关**：同一文档、两个语义无关的问题（"访问码是多少" vs "哈希表怎么解决冲突"），页级 logit 的 **Pearson = +0.986**、mean|Δ| = 0.70、top 页完全相同。原因 = **两个 query 向量本身的 cos = +0.964**（夹角仅 15°）：query span 是"问题模板"的 8 次重复，**模板的共同成分压过了语义差异**。⇒ §12.7.5 的"跨三次运行稳定"**不是证据**（三次用的是同一个问题）。
2. **方向 = k̄ 的位置成分**：k̄ 的**第 2 主成分占 22.2% 方差、与页偏移相关 +0.797**（正端是后部页 132/138/137，负端是前部页 7/1/2/0/6/8）；**query 与该方向对齐 cos = −0.259，是随机基线的 49 倍**（随机 |cos| ~ 0.0053）。
3. **排除其他候选**：**16 层各自单独打分都给出同一批早期页**（不是某层偶然）；**top-8 维只携带 15.1% 的 logit 离散度**（排除"少数巨大维度 / `k_norm` 权重偏置"）；**页均值归约与 max 归约的 top-8 完全重叠**（排除归约方式）；范数恒定 ±4%（061 已排除）。

### 12.8.3 危害（这才是要修的理由）

针的实测 lift = **+6.48**（页 19：control 20.02 → needle 26.51，rank 8 → 1）。把同样大小的针**逐页放到每个候选页上**：**中位 rank 18、最差 71、59/116（51%）落在 top-16 之外**；前部三分之一中位 rank 1、后部三分之一中位 ~50 ⇒ **检索只在文档前 20% 有效**。

配套：**针只在 max 归约下可见**（页均值归约下针页 rank 24/116，max 归约下 rank 1）⇒ §5.3 的"页内子块取 max"是**必需**而非可选。

### 12.8.4 为什么"去偏置"判 NO-GO

位置偏置只占 logit 方差的 **15%**（corr = −0.387，max 归约）。更根本的量级是 **97% 的页 logit 方差与 query 无关**（两个不同 query 的 r = 0.986）。eligible 页 logit：中位 11.69、**std 5.38**、范围 3.47–26.27。

七类修法（一律用**该修法自身的真实 lift** 评估——用 baseline 的 lift 会造出假改善）：

| 修法 | corr | std | lift | 真实针 | 合成针中位 | >top-16 | SNR |
|---|---|---|---|---|---|---|---|
| baseline | −0.387 | 5.38 | +6.48 | 1 | 18 | 59/116 | 1.20 |
| q 投影掉 β（去位置） | −0.102 | 4.50 | +5.59 | 1 | 31 | 62/116 | 1.24 |
| 子块线性去趋势 | −0.077 | 4.97 | +6.46 | 1 | 34 | 64/116 | 1.30 |
| 子块高通 w=2 | +0.047 | 2.13 | +2.77 | 1 | 30 | 75/116 | 1.30 |
| 子块高通 w=8 | −0.011 | 3.40 | +4.32 | 1 | 28 | 72/116 | 1.27 |
| 双边去 top-2 PC | −0.093 | 4.03 | +5.52 | 1 | 22 | 61/116 | 1.37 |
| 双边去 top-8 PC（in-sample） | −0.159 | 1.56 | +4.74 | 1 | **1** | **2/116** | **3.04** |
| query 中心化 | −0.242 | 1.27 | **+0.68** | 5 | 49 | 77/116 | 0.54 |

三条否定：
- **去位置类**（q 投影 β / 子块去趋势 / 高通）能把 corr 降到 ~0 且**保住真实针 rank 1**，但 **SNR 不变**（1.20 → 1.24~1.30）——位置只占 15% 方差，去掉它换不来检索。
- **去主成分** in-sample 极好（`drop PC8`：std→1.56、SNR 3.04、合成针超 top-16 仅 2/116），**但留一验证反向崩**（用后 70 页估 basis、在前 70 页评估时真实针 lift → 0.48、rank 10）⇒ **140 页 × 16384 维估不出稳定的主成分**，漂亮数字是过拟合。
- **query 中心化**把背景压到 std 1.27，但**真实针的 lift 同时从 +6.48 掉到 +0.68** ⇒ **背景与信号同向**。

⇒ **"早期页偏置"不是实现缺陷，是"1424-token 页粒度下检索信噪比"的体现（风险 R1 的定量形态）**：背景与信号共享子空间，线性变换无法稳健分离。

### 12.8.5 对路线的影响

- **不阻塞重物化**：真实针在引擎的 max 归约下 rank 1，检索在针落于文档前部时有效。
- **阶段 1 出口判据调整**：不再要求"先修偏置"，改为**多深度 needle 实测端到端**（40/55/70/85% 深度），把"后部被淹没"作为**已知量级**记录而不是阻塞项。
- **真正的杠杆**（按价值排序）：①**查询 span 的自动界定**（§5.4 的 LCP 差量 + §12.7.7-3）——本步证明 query span 的"模板共同成分"是背景与信号共同的载体，更聚焦的 query 会同时改变两者；②更细的检索单元（受块粒度硬约束，只能在打分侧做，而 max 归约已经在用子块级信号）；③接受现状并在重物化后用多深度 needle 量化端到端损失。
- **caveat（勿当成通用常数）**：本步文档是**合成高重复文档**（3 个 4k 单元循环 49 次），其页级背景方差可能大于自然对话历史；"51% 被淹没"是这份文档上的量级。

---

## 12.9 阶段 1 K3 后半开工（步骤 063：重物化原语 + 往返单测）

> 本节记录 §5.3 的"重烘焙"与 §5.1 的槽位装配里**算法核心那一半**的落地机制与实测：给定一张页的 **pre-RoPE 旋转前缀**与**目标位置**，把该页的旋转前缀重建出来。**引擎接线（位置解耦 + 块表改写 + 存储格式）未做**——本步只回答一个问题：**"存 raw、重建页"这条路径在字节级成不成立？**

### 12.9.1 落点（两个新文件，零调用者）

| 文件 | 内容 |
|---|---|
| `v1/kvmem_workspace/remat.py`（新） | `PageGeometry`（页字节几何）/ `rotated_byte_offsets` / `bake_rotated_k` / `quantize_rotated` + `dequantize_rotated` / `write_rotated` + `read_rotated` / `rematerialize_page` |
| `tools/kvmem_remat_test.py`（新） | 离线单测（CPU），18 条断言，锚全部取自引擎自己的代码 |

**为什么不需要门控**：新模块**没有任何调用者**，引擎行为逐字节不变。本步是纯增量。

### 12.9.2 页的字节几何（从三处源码对账得到）

NVFP4 页 = `[K_data | K_scale | V_data | V_scale]`，HND 物理序 `(blocks, 2, heads, block, full_dim)`：

| 量 | 值 | 来源 |
|---|---|---|
| `full_dim = head_size//2 + head_size//16` | **144** | `utils/torch_utils.py:nvfp4_kv_cache_full_dim` |
| 每页 `2 × heads × block × full_dim` | **1,640,448 B** | 与工作区每槽字节数逐字吻合（§12.5.4 实测） |
| K 侧 = `heads×block×data_dim` + `heads×block×scale_dim` | **729,088 + 91,136 = 820,224 B** | `reference_nvfp4.py:side_carve_views` |
| 旋转前缀（每 token 每头） | **32 B 打包 + 4 B 尺度** | `rotary_dim//2` / `rotary_dim//16` |
| 旋转前缀（整页） | **205,056 B = 12.5%** | 1424 × 4 × 36 |

**每侧内部不是逐行交错的 144 字节**：是 `[heads][block][data_dim]` 紧接 `[heads][block][scale_dim]`（写错会静默毁掉所有 NVFP4 读，见 §5.2 的 `!` 风暴同族）。

### 12.9.3 ⭐ 决定方案可行性的结构前提（设计档此前没写）

**`rotary_dim = head_size × partial_rotary_factor = 256 × 0.25 = 64 = 4 × 16`，恰好落在 NVFP4 的尺度组边界上**（尺度每组 16 个 head 维）。于是：

- 旋转前缀覆盖**恰好 4 个完整尺度组**（data 字节 0..31、scale 字节 0..3 每头）；
- **不存在"一个尺度组跨旋转/非旋转两侧"的撕裂** ⇒ 可以**只重写旋转前缀**，其余 192 维与整个 V **一个字节都不动**；
- 量化侧的等价性也由此成立：**组的归约域不相交 ⇒ 只在旋转组上算 amax 与算整行再取那 4 组，结果逐位相同**。

**这是硬前提**：`rotary_dim % 16 != 0` 时"只重写旋转前缀"不成立（`PageGeometry.__post_init__` 已把它写成断言）。对本模型成立（64 = 4×16）。

### 12.9.4 RoPE 约定的两个反直觉点（按模型实际路径查，不按"标准 RoPE"假设）

模型实际用 **`MRotaryEmbedding`**（`get_rope` 同参命中同一 `_ROPE_DICT`）：`mrope_interleaved=True`、`mrope_section=[11,11,10]`、`is_neox_style=True`，cache 形状 **`[1,048,576, 64]`**（`max_position × 4`）。

1. **cache 宽度 = `rotary_dim`，不是 `2 × rotary_dim`**：cos 在前 32 列、sin 在后 32 列（`chunk(2, -1)`）。
2. **位置是 2D `[3, T]` 且三行相同**（`uses_mrope=True`；`Qwen3_5.get_mrope_input_positions` 返回 `arange(...).unsqueeze(0).expand(3,-1)`）⇒ 引擎走 `triton_mrope`（交错分支）；但**三行相同使 `apply_interleaved_rope` 成为恒等** ⇒ **文本输入下 2D 路径与 1D 路径逐位等价**（实测 max|Δ| = 0）。**这使重烘焙可以按标量位置实现**，不必复刻 M-RoPE 的通道置换（但代码仍保留 2D 分支，且不等行时用真置换——单测已钉住）。

**实现选择：不复刻公式**。旋转直接调引擎的 `ApplyRotaryEmb.forward_static`（`MRotaryEmbedding.forward_native` 用的同一个函数）与 `apply_interleaved_rope`；量化复用 `reference_nvfp4._e2m1_codes`。**约定不会漂移**，因为用的就是引擎自己的函数。

### 12.9.5 实测（步骤 063，CPU，18/18）

| 组 | 断言 | 结果 |
|---|---|---|
| T0 几何 | 页 1,640,448 B；旋转前缀落在尺度组边界 | 2/2 |
| T1 旋转 | vs **逐行移植的 `_triton_mrope_forward`**：文本 2D ≡ 1D、2D ≡ 内核、**不等行 2D ≡ 内核** | 4/4，**max\|Δ\| = 0.000e+00（逐位）** |
| T1 反平凡 | 不等行结果与文本结果差 3.457e-01 | ✓（置换确实在动） |
| T2 字节 | vs 引擎 `write_reference_nvfp4_cache`：**205,056 字节 0 mismatch**；其余 1,435,392 字节未动 | 2/2 |
| T3 偏移 | 经引擎 `side_carve_views` 独立复推：data `(1424,4,32)` / scale `(1424,4,4)` 全等 | 3/3 |
| T4 无漂移 | 位移 `d=4096` 后搬回原位**逐位一致**；8 次位移后搬回仍**逐位一致** | 3/3（位移确实改了 147,975 字节） |
| T5 量化误差 | `max\|dequant − post_rope\| = 2.734e-02`，最大 E2M1 步长 `7.813e-02`，**比值 0.35** | ✓（一个量化步之内） |
| T6 只动前缀 | 对**引擎写的整页**重烘焙：1,435,392 个非旋转字节**一个未变** | 2/2 |
| T7 整页重建 | **以 pre-RoPE 旋转前缀为唯一输入重建整页 == 引擎整页，逐位一致** | 1/1 |

**读数**：①设计出口要的是"量化步内一致"，实测给的是**逐位一致**（因为重建是纯函数且与引擎共用同一套原语）；②**"禁 delta re-RoPE ⇒ 零漂移"从设计陈述变成可执行断言**——8 次位移后仍逐位一致，参考实现 #4 的 >210K 串台机制在我们这里**结构性不存在**这一点现在有单测背书；③**只重写 12.5% 的字节**（205,056 / 1,640,448）⇒ 搬运与带宽账远小于整页重写。

### 12.9.6 本步的边界（勿误读）

- **引擎接线完全未做**：位置解耦（`gpu_model_runner.py` 的 `positions`）、块表改写、以及**存储格式**（工作区当前存"整页原样"，需改成"V + 非旋转 192 维原样 / 旋转 64 维 fp16 权威"）都还没动 ⇒ **窗外 needle 仍答不出**。
- **单测跑在 CPU**：对过的是 triton 内核的**公式**（逐行移植），**不是它的实际执行**（fp 舍入差异未对）。生产占 15,891/16,303 MiB，起 CUDA context 会撞余量悬崖（§3.3），故不打扰；**停服后可 `--device cuda` 补跑**。
- **"工作区页 == 块原始字节"是结构推断**（`1,640,448 = 2×4×1424×144` 且 `swap_blocks_batch` 搬整块），未在引擎里取证。
- **未重跑**栈内 `research/src/nvfp4_writer_swap_probe.py` 那条"writer 输出 == 真实 GPU 页字节"的既有证据（引用旧结论）。
- 本步**不判性能**（阶段 1 判据不含性能）。

### 12.9.7 下一步（接线顺序建议）

1. **先做存储格式 + raw-K 权威区**（`V + 非旋转 192 维原样 / 旋转 64 维 fp16`；容量账见 §3.4），并在工作区自检里加一项"**重烘焙往返**"——**这一步不改变注意力行为**，可以先把臂跑起来验证；
2. **再动 `gpu_model_runner.py` 的 `positions` 与块表**（位置解耦 + 槽位装配）。这一步一开始就必须跑臂 + 双 boot（必守第 17 条：改被模型引用的源文件会改 AOT 缓存键）。

---

## 12.10 阶段 1 K3 后半接线第一半（步骤 064：raw-K 权威区 + 实时往返自检 → **页布局假设被证伪**）

> 本节记录 §12.9.7-1 的落地与实测。**结论先行**：权威区机制 GO（2.0 GiB、16 层、零未映射），但**实时重物化往返失败**——根因是 §12.9 的页几何假设**对真实缓存不成立**，本节给出实证拟合出的真实布局与修复公式。**本节是 `PageGeometry` 的权威依据**。

### 12.10.1 落点与门控

| 文件 | 内容 |
|---|---|
| `v1/kvmem_workspace/config.py` | `VLLM_KVMEM_AUTHORITY`（默认关）/ `VLLM_KVMEM_AUTHORITY_TOKENS`（默认 262,144）/ `VLLM_KVMEM_AUTHORITY_TRAJ`（默认 2） |
| `v1/kvmem_workspace/worker.py` | 层号↔层名映射、权威区分配/写入/读回、`_run_remat_selftest`（实时往返）、报告与数组转储 |
| `tools/serve_gsq_kvmem_ws163k.cmd` | 3 行环境变量 + dump 目录 `kvmem_k3e` |

**存储格式决策（与设计 §5.3 的"槽内拆分"不同，勿误读）**：工作区页**保持整页原样**（K1 的整页 memcpy 与往返自检语义不变），旋转前缀的 pre-RoPE 权威放**独立区域**。理由：①槽内拆分会把每逻辑页从 25.03 MiB 涨到 33.02 MiB ⇒ 122 槽 = 4,028 MiB **超出** 3,072 MiB 预算；独立权威区只要 **2,147,483,648 B = 2.0 GiB**（512 B/token/层 × 16 层 × 262,144，与 §3.4 的 2.1 GiB 估计吻合）；②拆分要在 store 路径逐头重排（host ~26 MB/逻辑页），阶段 1 判正确性不判性能，不值得先付。**副作用**：槽内旋转前缀的 nvfp4 字节成为死重（12.5%），重物化时总是被覆盖，无正确性问题。

### 12.10.2 机制

- **写入**：`_ingest` 在喂检索索引的**同一批** capture 行上追加切片（`remat.rotated_prefix_from_packed_k`），按**绝对 token 偏移**写入 `region[abs_pos]`（允许前缀缓存空洞）；键 = (轨迹, 层名)，层号经 `extract_layer_index` 桥接到工作区的层名空间。
- **读回 + 自检**：对每张已存页取 `page_index × block_size` 起的权威行，用**引擎自己的** `get_rope(...)`（命中同一 `_ROPE_DICT` ⇒ 与模型同一实例；日志核实 `MRotaryEmbedding / cache (1048576,64) fp32 / rotary_dim 64 / NeoX / interleaved`）在**原始位置**重建旋转前缀，与页内字节比对 + 去量化偏差。
- **实测（机制侧）**：16 层全映射（3,7,…,63）、`rows_written = 2,659,328`、`unmapped = 0`、`missing = 0`；K1 整页往返自检不受影响（`8 pages × 16 layers byte-identical`）。

### 12.10.3 ⭐ 实时往返失败 → 真实页布局（实证拟合，本节的权威结论）

**现象**：8 页 × 16 层全部重建，**8/8 页字节不同**、`max_byte_diff = 255`、去量化差 **44.6% 非有限**（NaN/inf = 把 data 字节当 scale 读的特征）；差异**全部落在 K 侧**（V 侧 0 字节不同）；差异模式 = **token 0..15 的旋转数据完全一致、token 16 起整段不同**。

**真实布局（数据区 15/15 探针精确命中 + 尺度区 58.84% vs 旧假设 0.93%）**：

```
页（1,640,448 B）= 89 个 18,432 B 的 chunk（89 = 1424 / 16；16 = 内核块大小 kbs）
chunk(t//16) = [ K 侧 9,216 B | V 侧 9,216 B ]
每侧 = [heads × kbs × data_dim (8,192 B)] 紧接 [heads × kbs × scale_dim (1,024 B)]

data 偏移  = (t//kbs)*chunk_bytes + side*side_bytes + h*(kbs*data_dim) + (t%kbs)*data_dim + j
scale 偏移 = (t//kbs)*chunk_bytes + side*side_bytes + heads*kbs*data_dim
             + h*(kbs*scale_dim) + (t%kbs)*scale_dim + k
```

即 **K/V 按 16-token 内核块交错**，**不是** §12.9 假设的"整块 K 侧 + 整块 V 侧、每侧内部 `[heads][1424]`"。`side_carve_views` 的公式对 **kbs-token 内核块**成立、对 **1424-token 管理块**不成立——它硬编码了 `base + heads*block*data_dim` 的尺度基址与 `[heads][block]` 内序。

**修法（已定，未实现）**：`ref = group_kernel_blocks(kv_caches[layer], num_blocks)` 返回 `cache.unflatten(0, (num_blocks, -1))`，故 **`kbs = spec.block_size // ref.shape[1]`**（本例 1424 // 89 = 16）。把 `PageGeometry` 换成上面的公式，并**同步改离线单测的页构造**（单测现在自造视图 = 自证陷阱，与《格式搬运必须有外锚》同构）。

### 12.10.4 本步的边界（勿误读）

- **重物化在真实页上仍未通过**：`PageGeometry` 尚未按真实布局实现，离线单测尚未在真实布局上重建。
- **没有任何页被放回视窗** ⇒ 注意力行为逐字节不变；全部新行为在 `VLLM_KVMEM_AUTHORITY`（默认关）门控后。
- 位置解耦（`gpu_model_runner.py` 的 `positions`）+ 块表改写**仍未开工**。
- 权威区只增不减（无跨轨迹淘汰，超出 `AUTHORITY_TRAJ` 计数丢弃并告警）。
- **教训（进必守）**：**页字节布局这类断言的锚必须是引擎的真实张量**（`ref.shape`/stride 或内核块公式 + 真实页比对），**不能是"参考 writer 写进自己构造的视图"**；自检必须带**可核对量**（`rows_written` / 非零行数），否则映射类缺陷会静默。

---

## 12.11 阶段 1 K3 后半接线第二半（步骤 065：`PageGeometry` 内核块重写 + 实时往返 GO）

> 本节记录 §12.9.7-1 的收口：按 §12.10.3 的真实布局重写 `PageGeometry`、离线单测在真实布局上重建、实时往返重跑。**结论先行：实时重物化往返 GO（量化步内一致）**——8 页 × 16 层 delta ≤ 0.127 步长、nan = 0；位级残余 ~0.5% = triton kernel 与 host 复算的 fp32 舍入序列差，对注意力等价。

### 12.11.1 落点

| 文件 | 改动 |
|---|---|
| `v1/kvmem_workspace/remat.py` | `PageGeometry` 加 `kernel_block_size`（必填），偏移公式换内核块形式（§12.10.3） |
| `v1/kvmem_workspace/worker.py` | `register_kv_caches` 从真实 cache 张量推 kbs；自检精度契约（bf16 还原）；报告加 `max_e2m1_step` / `delta_over_step`；dump 转 fp32 落盘 |
| `tools/kvmem_remat_test.py` | T0/T2/T3 重写：物理 buffer = 真实布局 flat 字节，writer 经**内核块粒度幻象视图**写入（25/25 全过） |

### 12.11.2 机制

- **kbs 的来源**：`ratio = cache.shape[0] // num_blocks`（未拆分时 1），`kbs = spec.block_size // ratio`。worker 注册日志打 `89 chunks x 18432 B, kernel block 16` 供现场核对。
- **离线单测去自证**：物理 buffer 是 flat 字节（真实布局），`write_reference_nvfp4_cache` 经 `as_strided` 的内核块粒度 NHD 幻象视图写入（stride(0)=18,432、V 侧 offset=9,216），`slot_mapping` 用真实引擎语义（绝对 token 号，writer 内部 `slot // 16` 落 kernel block）。T3 的 carve 复推从同一视图出发。**manager 块粒度的视图会重建被证伪的布局——这正是 064 的自证陷阱**。
- **⭐精度契约（实时往返从 2% 码差到 0.5% 的关键）**：权威区以 fp16 存 pre-RoPE K（对 bf16 值**无损**）；引擎 triton kernel 的输入精度 = **bf16 K × fp32 cos/sin** ⇒ 重建时必须 `raw.to(bfloat16)` 且 cos/sin 保 fp32，再走 `bake_rotated_k`。fp16 行直喂 + cos/sin 截 fp16 = 双重舍入 ⇒ amax 抖动 ⇒ ~2% 码差且 delta 达 2.0（超步）。
- **残余差异的定性（判定依据）**：写页的 triton kernel 与重建的 host 复算在 fp32 舍入序列上不同 ⇒ ~0.5% 码在**邻档**翻动（932 个幅度差 1 档 + 13 个近零符号翻转），dequant 误差 **≤ 0.56 步**——对注意力等价，满足设计出口"逐位**或**量化步内一致"。

### 12.11.3 实测（步骤 065）

| 项 | 结果 |
|---|---|
| 离线单测（CPU，真实布局） | **25/25**：旋转字节 205,056 **0 mismatch**、carve 复推全等、整页重建逐位一致、writer 只碰页 0 |
| 实时往返 r2（8 页 × 16 层） | 字节差率 **0.41-0.56%**、`delta_over_step` **0.0068-0.127**、**nan = 0**、`rows_written = 2,659,328`、`unmapped/missing = 0` |
| 离线逐元素口径 | 码差 0.5% 全邻档；dequant 误差/步长 **max 0.5625、mean 5e-4、0/364,544 超步** |
| 检索侧 | r2 `dot@64` rank 2（与 061 量级一致，未回归） |
| 证据 | `prod029_logs\kvmem_k3f\`（r1 失败态已改名 `_r1_fp16asis` 保留）+ `kvmem_k3f_boot{,2,3}_20260930.{out,err}` + `kvmem_k3f_r{1,2}_needle_d10.json` |

### 12.11.4 本步的边界（勿误读）

- **实时往返通过 ≠ 页能被放回视窗**：往返自检只测"原位重建"；位置解耦（`gpu_model_runner.py` 的 positions）+ 块表改写未开工 ⇒ **窗外 needle 仍答不出**。
- 真实"压缩到固定槽位 0..B-1"的目标位置装配未测（063 T4 只测离线位移）。
- 本步不判性能（阶段 1 判据不含性能）。
- 诊断 dump 的数组转 fp32（numpy 无 bf16）——第一版直接 `.numpy()` 会让 EngineCore 崩在 `np.savez`。
- **精度契约已进必守**：重建必须还原 bf16 + fp32 cos/sin，违反则码差超步。

---

## 12.12 阶段 1 K3 后半接线第三半（步骤 066：连接器级前缀装配 —— matched/异步装载/前跳 + mamba 边界快照）

> 本节记录把工作区里已有的页**放回后续请求的视窗**的第一半（也是唯一不需要动模型执行器的一半）：按**原始位置**装回，让调度器把 `num_computed_tokens` 前跳、只 prefill 差量。**结论先行**：机制全链打通（匹配 → 异步装载 → 前跳 → 差量 prefill → mamba 状态恢复），同一 200K prompt 的 serve 请求 **TTFT 256.63 s → 229.76/230.13 s（−10.4%）**、needle 命中、输出连贯。**但窗外 needle 仍答不出**——原位装配装回的页在最终注意力窗口之外（见 §12.12.5）。

### 12.12.1 ⭐ 为什么不需要动模型执行器（推翻 063/064 的接线判断）

063 §12.9.7 与 064 判定都写"下一步 = 位置解耦（`gpu_model_runner.py` 的 `positions`）+ 块表改写，且 flashinfer 没有 `update_block_table` ⇒ 确实要动模型执行器"。**这个判断对"压缩到固定槽位"成立，对"原始位置装配"不成立**：

- `positions` 与 `slot_mapping` 本来就是从 `request.num_computed_tokens` 派生的（`gpu_model_runner.py:2204-2217`：`positions = num_computed_tokens + query_pos`，`slot_mapping = block_table.compute_slot_mapping(...)`）。**`num_computed_tokens` 一旦被前跳，二者自动只覆盖差量**，块表也不用改写——请求块表的 0..E-1 行本来就是本步新分配的空块，装载把页写进去即可。
- 前跳的唯一入口 = `KVConnectorBase_V1.get_num_new_matched_tokens` 返回 matched ⇒ 调度器 `num_computed_tokens = local + ext`（`scheduler.py:908-910`）；返回 `load_async=True` 则请求停在 `WAITING_FOR_REMOTE_KVS`、本步不 forward，等 worker 通过 `finished_recving` 放行（`:1127-1157`、`:2910-2937`）。
- `update_block_table` 与 KV 连接器**无关**（它是混合模型多组共用 attention 元数据的优化，`gpu_model_runner.py:2507-2582`）；flashinfer 缺它只是"多组时不能复用元数据"，**不影响装载路径**。

⇒ **零改动 `gpu_model_runner.py` / `block_table.py` / 任何 attention backend**。改动全在连接器（调度侧 + worker 侧）+ 一处 mamba 分配覆写。

### 12.12.2 ⭐ mamba：装配路径上真正的硬问题

注意力页可以按 token 偏移原样搬回，**mamba（GDN）不行**：它没有可回装的"历史 KV"，只有一份**递推状态**，而递推状态只在**块边界**上被引擎锚定。装配请求前跳到 E 后，`preprocess_mamba` 会算 `prev_state_idx = (num_computed_tokens - 1) // block_size = E/block_size - 1`（`mamba_utils.py:1483`）并从那里续算 ⇒ **块表第 `E/block_size - 1` 行的槽里必须已经是"算完 E 个 token 的精确状态"**，否则整个续算从错的状态出发（比 KV 糊更严重：是**静默的错误生成**）。

| # | 子问题 | 本步答案 |
|---|---|---|
| 1 | 状态从哪来 | **在 ingest 的 prefill 过程中逐边界抓**。滑窗把工作区的连续页前缀压在活序列后约 W token（114.5 页）处，等页被淘汰时它的状态槽早已被 CoW 复用（align 模式只有 2 个状态块）⇒ **不能事后取，只能在块边界那一刻取**。引擎自己的不变式背书："slot p holds the state after exactly (p + 1) * block_size tokens. State is written at chunk ends, so chunk ends must be block aligned"（`scheduler._mamba_block_aligned_split`）⇒ 抓取条件 = **步尾恰好落在页边界**（`end % block_size == 0`）。 |
| 2 | 抓多少 / 存多久 | 每个边界要活到"页前缀追上它"（W/block_size = 114.5 步）。抓取间隔 8 页 ⇒ 需 14.3 份 ⇒ 环形保留 **20 份**（每份 80.4 MiB = 48 层 ×（conv 102,400 B + bf16 ssm 1,572,864 B））。**环形区按轨迹分槽**（`SNAPSHOT_TRAJ=2`）——全局 FIFO 会让第二条轨迹的 prefill 把第一条的快照挤空（本步实测，见 §12.12.4 的"两个真缺陷"）。 |
| 3 | 装配边界取哪 | `min(连续页前缀终点, 最新的已捕获边界)`。抓取是稀疏的（每 8 页）⇒ 装配边界最多比页前缀低 7 页；这是 `SNAPSHOT_EVERY_PAGES` 这个旋钮的全部含义。 |

**必须同时动的引擎点**：基类 `allocate_external_computed_blocks` 会给**每个组**分配 `cdiv(E, block_size)` 个真实块；对 mamba 组（每块 13.4 MiB × 6 组）这会在装配时**一次性吃掉约 2 GiB 池**，而中间位**根本没人读**。覆写 `MambaManager.allocate_external_computed_blocks` = `[null × (E-1), 1 个真实块]`（形状与 `find_longest_cache_hit` 给本地命中返回的 `[null × i, cached]` 完全一致；null 不进 hash、不占池）。

### 12.12.3 落点

| 文件 | 改动 |
|---|---|
| `v1/kvmem_workspace/config.py` | `load_enabled()`（`VLLM_KVMEM_LOAD`，默认关）/ `snapshot_keep()`（20）/ `snapshot_every_pages()`（8）/ `snapshot_trajectories()`（2） |
| `v1/kvmem_workspace/metadata.py` | `KVMemPageLoad` / `KVMemLoadJob` / `KVMemSnapshotRequest`；`KVMemConnectorMetadata.load_jobs`、`.snapshot_requests`；`KVMemWorkerMetadata.completed_snapshots` / `.removed_snapshots` / `.finished_load_reqs` |
| `v1/kvmem_workspace/manager.py` | `_assembly_match`（连续页前缀 ∩ 已捕获快照边界 ∩ **页 token 哈希一致**）/ `get_num_new_matched_tokens`（返回 `(boundary, True)`）/ `_emit_load_jobs` / `_emit_snapshots` / 页哈希记录 / 完成回报处理 |
| `v1/kvmem_workspace/worker.py` | mamba 组注册（状态槽视图 + 按轨迹分槽的快照区）/ `start_load_kv`（发装配拷贝）/ `_run_loads` / `_take_snapshots` / `get_finished` 回填 `finished_recving` |
| `kv_connector/v1/kvmem_connector.py` | `start_load_kv` 转调 worker（**关键**：装配请求的步调度 0 token，零前向路径**只**调 `start_load_kv`，不调 `wait_for_save`） |
| `v1/core/single_type_kv_cache_manager.py` | `MambaManager.allocate_external_computed_blocks` 覆写（+ `tools/apply_kvmem_mamba_ext_step066.py` apply/revert） |
| `tools/kvmem_assembly_probe.py`（新） | 三请求探针：ingest → **flush**（换 nonce 的 200K 请求，用来打掉原生前缀缓存）→ serve；`compare` 判 TTFT 比 + 输出一致 |

**为什么探针需要 flush 请求**：引擎**原生的前缀缓存**在块还没被复用时就能救回同一 transcript 的约 97%（本步第一次 boot 实测：serve prefill 塌到 ~8 s，连接器什么都没做）。flush 用一条不同 nonce 的 200K 请求把池块覆写掉，留下的才是"只有 KVMem 工作区还持有这段前缀"的残余场景。

### 12.12.4 实测（步骤 066）

臂 = `tools/serve_gsq_kvmem_ws163k.cmd`（池 3.4e9 / `--max-model-len 262144` / W=163072 / **`--enforce-eager`** / 无投机 / 无 offload 8 GiB 区 / `VLLM_KVMEM_LOAD=1`）；prompt = 198,205 token（serve 加 787 token 尾巴 = 198,992）。

| 项 | 数据 |
|---|---|
| boot | `snapshot region 2 x 20 row(s) (2.93 GiB host)`、`GPU KV cache size: 273,771 tokens`、`viewport needs 236 / pool has 259` |
| 装配匹配 | `matches at 34176 tokens (24 stored page(s)); async load` —— 与结构预期 `⌊(198992−163072)/1424⌋×1424 = 34176` **逐字吻合** |
| 装载作业 | `issued (48 page(s) + 6 mamba state block(s), boundary 34176)` → `prefix landed; cumulative requested=1 completed=1`，**零 error** |
| **TTFT** | 装配 **229.762 / 230.133 s**（两次）vs 全量 **256.627 s** ⇒ **−10.4%**（比值 0.896/0.897，两次复现 ±0.4 s） |
| 跳过的 token 占比 | 34176/198992 = **17.2%**（上界）；实测省 10.4% —— **亚线性**，因为 prefill 单 token 成本随视窗增长，而被跳过的是**最便宜的最早期 token**（与 §12.2 的"窗口越宽 prefill 越慢"同向） |
| needle | 两臂**均命中**（`77349`，depth 0.50，在窗内） |
| 输出 | 同 prompt 跨臂共同前缀 285 字符；**但同臂两次不同 prompt 的共同前缀只有 99 字符** ⇒ 分歧由 prompt 差异主导，装配未引入额外分歧 |
| 代价（host） | 工作区 3.00 GiB + 权威区 2.0 GiB + **快照区 2.93 GiB** |

**两个真缺陷（本步实测暴露并修掉）**：
1. **快照环全局 FIFO ⇒ 多轨迹互相挤空**。第一次实测（boot3）：flush 请求（第二条轨迹）的快照把 ingest 轨迹的快照全挤出去，serve 请求匹配不到任何边界 ⇒ 回落到全量 prefill（TTFT 257.1 s）。修法 = **环形区按轨迹分槽**（`SNAPSHOT_TRAJ=2`，每轨迹各 20 行）。
2. **装载必须发在 `start_load_kv` 而不是 `wait_for_save`**。装配请求在 `WAITING_FOR_REMOTE_KVS` 时其步调度 0 token，worker 走**零前向路径**（`kv_connector_no_forward`）——它**只调 `start_load_kv`**（`kv_connector_model_runner_mixin.py:86-95` 的 `wait_for_save=False`）。首版把拷贝放在 `wait_for_save` ⇒ 永不发出 ⇒ 请求**永久挂在等待队列**（本步 boot4 实测）。

**一次判据口径修正**：`--max-num-batched-tokens` 必须**钉在页长 1424**。①快照只在"步尾恰为页边界"时精确（引擎不变式），9968（7 页）也可行；②但滑窗组的 admission 项 `cdiv(W−1+max_in_flight, 页)+1` 在 9968 下从 117 涨到 123 块/组 ⇒ 启动检查报"需要 3.32 GiB > 可用 3.15 GiB"（**实测 boot 失败**）；1424 时 `cdiv` 与 1024 同为 117 ⇒ **需求零增长**。

### 12.12.5 本步的边界（勿误读）

- **窗外 needle 仍答不出**：原位装配把页装回**原始位置**，而问题在序列末尾 ⇒ 最终注意力窗口（`[L−W, L]`）**看不到**装回的页。本步证明的是**投递机制**（匹配/装载/前跳/状态恢复），不是召回能力。要窗外召回必须做**固定槽位重烘焙**（设计 §5.1 的压缩视窗，需要重 RoPE）——那是下一半。
- **"逐 token 一致"在本臂上不可测**：引擎自身在同配置下**不是位精确**（同族既有结论"PPL 非位精确"）。**这不是 seed 能解决的**——`SamplingParams.seed` 只在 `temperature ≥ eps` 时生效（`sampling_params.py:757-761`，temperature=0 直接 `GREEDY`，seed 不被读取），分歧在 **logits 层**。vLLM 的对应开关是 **`VLLM_BATCH_INVARIANT=1`**（`envs.py:620-623`，需 SM ≥ 9.0，本机 SM120 满足）⇒ 要位精确闸门时开它，代价是禁 split-K 等 ⇒ **变慢**，只作测量档。而且装配路径与全量路径的**计算路径本来就不同**（chunk 起点、`seq_len`、KV 物理块），即使 kernel 全确定，归约序也可能不同 ⇒ 逐位一致不是"加开关就能拿到"的性质。
- 本步不判性能（阶段 1 判据不含性能）；臂**仍 `--enforce-eager`** ⇒ 速度不可与生产比（**下一步：全图捕获兼容**，见 §12.13）。
- 装配边界受**连续页前缀**限制 ⇒ 可省比例 = `(L−W)/L`（本例 17.2%），且**亚线性**。工作区只装"窗口外的部分"是设计既定的（§12.5.3）。
- 快照只增不减（按轨迹环形，超出整环丢弃并告警）；多轨迹串行未压测（`--max-num-seqs 1`）。
- 页哈希一致检查用**页 token 哈希**（`(轨迹, 页偏移) → blake2b8`），它补上了轨迹键只钉前 512 token 的缺口。

---

## 12.13 阶段 1 全图捕获兼容（步骤 067：`record` 变不透明算子 → 臂进生产同构图模式，decode 4.0×）

**用户指令（2026-09-30）**：「先把 full cuda graph 兼容修复了，速度起码翻一倍。下一个优化就是这个了，**然后之后的优化全部都要全图捕获的兼容**」⇒ 阶段 1 的"无图模式"约束作废；**"能不能进 FULL 图"升级为一等设计判据**。

### 12.13.1 唯一拦路虎与修法（一个文件，零改动调用点）

061 起臂用 `--enforce-eager` 的唯一原因是 `capture.record` 跑在模型 forward 里，而 AOT `torch.compile(fullgraph=True)` 会追进它的函数体。本步 boot 一次拿到**唯一**剩余报错（`logger.info` 061 已搬走，所以它是第一个）：

```
torch._dynamo.exc.Unsupported: Data dependent operator
  Operator `aten.equal.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  from user code: capture.py:97  if not torch.equal(positions[0], positions[1]):
```

dynamo 的 Hint 直接给出方向：**"wrap the operator into a PyTorch-understood custom operator"**。修法 = 把 `record` 包成 `torch.library.custom_op("vllm_kvmem::record", mutates_args="unknown")`：

- **dynamo 把它当叶子**，不再追踪函数体 ⇒ dict 记账、M-RoPE 检查、计数器、`clone()` 全部保留原样，**一行逻辑没改**（原 `record` 改名为 `_record_impl`，新 `record` 只转调它）。
- `mutates_args="unknown"` 声明"可能写任何东西" ⇒ 编译器不会重排、不会当死代码删掉。
- `register_fake` 返回 `None`（算子无输出）。
- **`qwen3_next.py` 的调用点一字未改**；`capture.py` 不被模型引用 ⇒ **不改 AOT 缓存键**（062 的同类结论在本步再次被实测确认：第二次 boot 直读缓存、25 s 完成、无退化）。

### 12.13.2 为什么这样是安全的（两条必须先核清的性质）

1. **不透明算子仍是图里的一个节点，但它的实现每次执行都会真正被调用**（离线 `probe_custom_op_067.py` 实测：编译后连续两次执行，实现调用计数 1 → 2）。这正是 stash 能持续被喂饱的原因，也是"函数体必须保持廉价、只做分配"这条纪律的由来。
2. **prefill 根本不进 CUDA graph**：`CudagraphDispatcher.dispatch` 对 `num_tokens > max_cudagraph_capture_size` 一律返回 `CUDAGraphMode.NONE`（`v1/cudagraph_dispatcher.py:270-276`），而臂的 `--cudagraph-capture-sizes 1` 使上限 = 1 ⇒ prefill（≤ `max_num_batched_tokens` = 1424）走**编译产物逐算子执行**，捕获始终在流的 host 侧。decode（1 token）**进** FULL 图，但它在 `num_tokens <= 1` 守卫处直接返回、**不产生任何 device 工作**，图重放也不重新进入 Python。⇒ 唯一必须留在函数体外的只有 **host 同步**，而 device→host 拷贝本来就在 `drain()` 里（forward 之后由连接器 worker 调用）。

### 12.13.3 新增判据：`drain` 的单 token 调用计数

`drain()` 现在报告"自上次 drain 以来的单 token 调用次数"。**图重放不会重新进入算子实现**，所以这个计数是"decode 是否真的在 FULL 图里"的直接读数：**0 = 走图**，>0 = 回落 eager。实测：首次（编译/预热期）为 `80`，此后**全部为 0**。

### 12.13.4 实测（步骤 067）

**臂** = `tools/serve_gsq_kvmem_ws163k_graph.cmd`（原臂只把 `--enforce-eager` 换成 `--cudagraph-capture-sizes 1`）；**对照** = `tools/serve_gsq_kvmem_ws163k_eager067.cmd`（原臂，仅 dump 目录改名 `kvmem_k5b` 防覆盖 066 证据）。**两者其余字节相同，唯一变量 = 图模式。**

| 项 | eager（原臂） | 带图（本步） | 倍数 |
|---|---|---|---|
| 8k decode（3 次中位） | 17.28 tok/s | **69.16 tok/s** | **4.00×** |
| 200K 装配 serve TTFT | 229.76 / 230.13 s（066） | **223.94 s** | 0.973× |
| needle 命中 / 错误 | 命中 / 零 | 命中 / **零** | — |
| 编译 + 图 | `CompilationMode.NONE` + `CUDAGraphMode.NONE` | `VLLM_COMPILE` + `FULL_AND_PIECEWISE`（PIECEWISE 1/1、FULL 1/1） | — |

**⭐eager 臂日志坐实机理**：`Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none` ⇒ **`--enforce-eager` 是"编译 + 图"双重禁用**，臂此前跑的是**完全未编译的朴素路径**。decode 每步只 1 token、kernel 计算量小、launch 开销占绝对主导，所以收益集中在 decode（4×）；prefill 每块 1424 token、计算密集，收益只有 2.7%（与 §12.2 的"eager 代价约 +7%"同向）。

### 12.13.5 ⭐推翻既有记录：臂的 eager 基线是 17.3，不是 69

此前 memory 与文档记的"臂 8k 约 69 vs 生产 122"把 **69 当成了 eager 基线**——**实测 eager 只有 17.28**，69 实为**带编译/带图**态的数字。生产的 122.58 是**编译 + 图 + DFlash2 N=2 投机**（接受率 62.82% ⇒ 每步约 2.26 token）；臂每步 14.45 ms 已**快于**生产每步 18.4 ms ⇒ **69 vs 122 的差额来自投机，不是图模式**。

### 12.13.6 本步的边界（勿误读）

- **臂仍无投机**（阶段 1 设计排除该变量）⇒ **69.16 与生产 122.58 不可直接比较**。要到生产的 90-120 需给臂加投机，**属用户拍板项**（用户 2026-09-30 已指出"生产下配置草稿模型解码能到 90-120"）。
- 本步只验"**能进图 + 不回归**"，**未做** decode 侧进一步优化。
- **09-25 的"FULL 图挂死 = flashinfer BatchPrefill nvfp4 reader"（WONTFIX_WITH_ROOT_CAUSE）本步未复现**，生产 `FULL_AND_PIECEWISE` 亦长期正常 ⇒ 该结论应视为**已过时**。
- 未跑：多轨迹串行、长稳、`VLLM_KVMEM_LOAD=0` 对照、位精确闸门（`VLLM_BATCH_INVARIANT=1`）。
- **下一半 = 固定槽位重烘焙**（设计 §5.1 的压缩视窗，需要重 RoPE）——那才是**窗外 needle 能答出来**的一半。

---

## 12.14 阶段 1 K3 后半接线第四半（步骤 072：固定槽位压缩视窗 —— 重烘焙路线 + prefill 期滚动入库）

> 本节记录 §5.1 原案（重烘焙路线）的接线与实测。**结论先行**：机制落地——单坐标视窗改写、prefill 期滚动入库、打分后 stage-in 烘焙三条链全部接通，离线单测 16/16（槽位烘焙与引擎 writer **逐位一致** 205,056 字节 0 mismatch、非旋转 1,435,392 字节原样、重烘焙零漂移）；**N=1 视窗端到端 GO**（改写 198,204→19,232、TTFT 16.966 s、输出连贯）。**但 N=55 大视窗（96,128 token）输出为空（首个采样即 EOS），卡点未解**（见 12.14.6）。**与 068-071 被回滚的双坐标路线的根本区别：本路线没有第二坐标系**——视窗请求的 prefill token 序列**就是**压缩视窗本身，positions/slot_mapping/块表全部停留在引擎原生视窗坐标，没有帧翻译、没有调度前跳、没有 mamba 快照恢复、没有取数侧平移。

### 12.14.1 ⭐ 机制：视窗 prefill = 改写 prompt（单坐标）

布局（§5.1，`B = S+N+R` 与选择无关的不变式不变）：`S = 1 页 = 1424`、`N = 55 页 = 78,320`、`R = 16,384` ⇒ `B = 96,128`；加生成预留 32,768 共 128,896 ≤ 滑窗 163,072。**改写规则**：

```
prefill 序列 = prompt[:S+N] + prompt[L-R:]
                ├─ sink [0,S)：原位 token 原位相位 ⇒ KV 天然正确
                ├─ 占位段 [S,S+N)：原位连续段（prompt[S:S+N]）⇒ KV 天然正确
                └─ recent [S+N,B)：原 prompt 尾部 R token，按视窗位置 prefill ⇒ KV 正确
positions = 0..B-1（引擎原生派生，零改动）
```

三个性质全部由这个形状免费得到：①**GDN 递推的是真实历史 token**（头部+尾部），无快照/恢复需求（硬约束 §3.2 被 §5.1 承诺的方式绕开）；②**视窗稠密连续**，生产注意力路径零改动、无 mask；③**只有检索槽需要烘焙**（prefill 后覆写），sink/占位/recent 全部是 prefill 自己算的正确 KV——比 §5.1 执行流水的"装配 + 重烘焙 recent"还省（recent 的重烘焙只在多轮增量场景才需要，本步未做）。

**改写落点（一处）**：`get_num_new_matched_tokens`（scheduler.py:857，`num_computed_tokens==0` 时必被调）。改写 `request.prompt_token_ids`（in-place）+ `_all_token_ids` + `num_prompt_tokens`，然后返回 `(None, False)` ⇒ 调度器把请求放回等待队列（scheduler.py:862-868）⇒ **下一步循环用改写后 prompt 重跑前缀匹配**，连接器第二次调用返回 `(0, False)` 正常调度。

**激活守卫（全部不满足则放弃视窗、原生跑）**：`VLLM_KVMEM_VIEWPORT=1` + RAWK+AUTHORITY 开、`L > B`（中段存在）、该轨迹 store 有页、本地前缀命中 `≤ S`（一页）。最后一条的双重理由：hit > S 说明第一轮的池块还活着、**原生前缀缓存已在救**（视窗无事可做）；且 hash 链断点之后改写序列的 recent 段**结构性不会再命中**（累积哈希），所以 hit ≤ S 时命中块全部在 sink 段内、检索槽行全是新分配块、零 CoW 风险。

**零污染**：改写后 `num_tokens = B + 生成 < W` ⇒ 视窗请求**无滑窗淘汰** ⇒ 无 K1 store、无快照抓取、页哈希不写入；`KVMemStepSpan.viewport=True` 让 `_ingest` 只提取 q（打分用）而**不进索引/权威区**（视窗的 K 行不是轨迹在那些位置的行——sink/占位段 token 相同但 recent 段相位已变）。

### 12.14.2 ⭐ prefill 期滚动入库（K1 的必要补全；两版 sweep 都死锁后改此）

**K1 只存滑窗淘汰页**——200K ingest 结束时 store 只有窗外约 25 页，而检索槽烘焙需要的 V/非旋转 192 维字节恰恰在**滑窗内活着的中段页**里（请求结束即释放）。补全的第一版是"请求结束时 sweep 剩余页"（`request_finished` 返回 True 持有块、req_id 经 `get_finished()` 的 finished-sending 回来才释放），**实测两种形态都死锁**：

1. 存全部剩余页：200K ingest 结束时 232 页（116 物理块/组）被持有 ⇒ 池只剩 27 块 ⇒ 后续请求（需 141 块）**无法准入** ⇒ **没有 forward 步 ⇒ 拷贝永不发生 ⇒ 块永不释放**（引擎日志 `Waiting: 1 reqs` 后静默）。
2. 只存打分选中的页（84 页）仍死锁：持有量小了，但"要靠下一个 forward 步来驱动拷贝"这个依赖没变。

**定案 = prefill 期逐页滚动入库**（`_emit_incremental_stores`）：某页的最后一个 token 在本步算完（`end // block_size` 越过它）即当步发 store job——页 KV 刚写好、请求还持有该块、**零块持有**，拷贝在同一步的 forward 之后完成。死锁在结构上不可能发生（拷贝不需要额外步）。块的来源 = **每步权威的 `kv_connector_block_state.block_ids`**（scheduler 侧每步填的块表快照；int 行、null = 0），**不能用 admission 时刻的副本**（会漏掉 chunked prefill 后续分配的块）。页 token 哈希也在这里记录（保持 066 的"装配前缀必须是同一批 token"性质）。视窗请求被排除在外（它的 recent 段相位已变，其页不是轨迹在那些位置的页）。

### 12.14.3 stage-in（打分后烘焙进检索槽）

prefill 完成步：score 触发照旧（061 的 `start+num ≥ prompt_len`），但 `KVMemScoreRequest.num_tokens` 传**原轨迹长度 L**（store 页键是原轨迹页号；排除项 = 原 sink + 原 recent 尾，eligible = 中段 ✓），q = 改写序列尾部 `query_span` token 的 pre-RoPE q（capture 现成）。随后 `_stage_in`：

1. top 页按**页号升序（时间序）**分配检索槽（§5.3"选满 N 个后按时间序分配槽位"）；
2. 每页每层：工作区页字节 → 重建 buffer（V+非旋转原样）→ 权威区 raw K 行 `rematerialize_page` 到**槽位位置**（065 精度契约：bf16 还原 + fp32 cos/sin，完整 cache 引用传入）→ `dst.copy_` 进请求块（**块表取自每步权威的 `kv_connector_block_state.block_ids`**，manager 侧随 `KVMemStageInRequest` 下发，含页号→host slot 映射）；
3. 同步在 `wait_for_save` 内完成（引擎单线程串行）⇒ 第一个 decode 步在流序上必然看到烘焙后的块，无竞争。

本步**不做**的：多轮 LCP/ΔP（query 段 q>0 的形态）、recent 滚动重烘焙、GPU 化烘焙、烘焙耗时优化（host 每页 16 层旋转+量化，阶段 1 只记录）。

### 12.14.4 落点

| 文件 | 内容 |
|---|---|
| `kvmem_workspace/config.py` | `VLLM_KVMEM_VIEWPORT` / `_VIEWPORT_PAGES`(55) / `_VIEWPORT_RECENT`(16384) / `VLLM_KVMEM_SWEEP`（滚动入库开关） |
| `kvmem_workspace/metadata.py` | `KVMemStepSpan.viewport` / `KVMemStageInRequest`（blocks/slot_start/page_size/pages） |
| `kvmem_workspace/manager.py` | `_viewport_layout` / `_viewport_rewrite`（改写+推迟一步）/ `update_state_after_alloc` 视窗分支 / `_emit_spans`（视窗标志 + stage 组装）/ `_emit_stage_request`（块表取自每步快照）/ `_emit_incremental_stores`（滚动入库） |
| `kvmem_workspace/worker.py` | `_ingest` 视窗 span 只提 q（不进索引/权威）/ `_stage_in`（烘焙 + 拷进槽块） |
| `tools/kvmem_viewport_test.py`（新） | 离线单测（见 12.14.5） |
| `tools/kvmem_viewport_probe.py`（新） | ingest → flush → serve 三请求探针（窗外 needle 判据；`--serve-ignore-eos` 可选） |
| `tools/serve_gsq_kvmem_viewport072.cmd`（新） | 臂变体（dump `kvmem_k7a`；LOAD=0、VIEWPORT=1、SWEEP=1、WORKSPACE_MB=5120、AUTHORITY_TRAJ=2、TOPN=64） |
| `tools/serve_gsq_kvmem_viewport072_n1.cmd` / `..._top1.cmd`（新） | 对照变体（N=1 小视窗；TOPN=1 单页烘焙） |

**engine 文件零改动**（`scheduler.py`/`gpu_model_runner.py`/注意力路径全部未动；本步只改 `kvmem_workspace/` 四文件 ⇒ 不触发 AOT 重编译，062 结论适用）。

### 12.14.5 离线单测（16/16，CPU）

外锚与 065 同源：字节锚 = 引擎自己的 `write_reference_nvfp4_cache`（经内核块粒度幻象视图），数值锚 = 逐行移植的 `_triton_mrope_forward`。

| 组 | 断言 | 结果 |
|---|---|---|
| T0 布局 | S=1 页 / N 整页 / 槽行页对齐 / B+生成 ≤ 滑窗 / 200K 有中段 / 改写序列长度与三段内容 | 9/9 |
| T1 槽位烘焙 | 槽位位置烘焙的旋转前缀 vs **引擎 writer 在同一槽位位置写的页**：205,056 字节 **0 mismatch** | 1/1 |
| T2 只动前缀 | 非旋转 1,435,392 字节 == 工作区页原字节；旋转前缀确实因位置而变 | 2/2 |
| T3 时间序 | 相邻两槽两页：旋转字节互异、各自非旋转字节保持 | 2/2 |
| T4 零漂移 | 同一 raw 重烘焙两次**逐位一致** | 1/1 |
| T5 数值 | dequant vs triton 移植：max\|Δ\| = 2.9e-02 ≪ 最大 E2M1 步长，0 超步 | 1/1 |

### 12.14.6 实测（步骤 072）

臂 = `tools/serve_gsq_kvmem_viewport072.cmd`；探针 = `tools/kvmem_viewport_probe.py`（ingest 200K → **flush** 200K 异 nonce → serve 同 prompt）。prompt = 198,205 token，needle 深度 0.65（token 125,367）。**证据**：`prod029_logs\kvmem_k7a_boot1..7.*`、`prod029_logs\kvmem_k7a\`（dump）、`kvmem_k7a_n1\`、`kvmem_k7a_top1\`。

| 项 | 结果 |
|---|---|
| boot | `workspace scheduler ready: groups=[6, 7], 204 host slots (5.00 GiB)`；`viewport needs 236 block(s) ... pool has 259`；`GPU KV cache size: 273,771 tokens` |
| 改写 | `rewritten onto the compressed window: 198205 -> 96128 token(s) (sink 1424 + 55 retrieval page(s) + recent 16384; stored page(s) 139); deferring one scheduling pass` |
| 滚动入库 | 每步 `job N stored 2 entry(ies) ... slots k/204` 递增、`dropped=0`；**两轨迹 × 200K 会把 204 槽用满**（`logical slots=204/204`） |
| 打分 | `retrieval (req=...) 198205 tokens / 140 pages (block 1424, sub-block 32), 16 layer(s) scored, eligible 128` |
| stage 计划 | `stage-in plan: 55 slot block(s) from row 1, page table carries 139 page(s) of this trajectory` |
| 烘焙 | `baked 55 page(s) into the retrieval slots in 3.42 s (55 offered, 0 without a stored page, 0 layer-row(s) without authority rows); 688.4 MiB written` |
| **serve（N=55）** | **`finish_reason="stop"`、`n_chunks=0`、`text=""`**（即首个采样 = EOS）、总耗时 89.6 s；视窗 prefill 本身正常跑完 68 步 |
| **serve（N=1 对照，`_n1` 变体）** | 改写 198,204 → 19,232；1 槽烘焙 0.08 s；**TTFT 16.966 s**；**输出连贯**（`<think>\nThe user is asking me to:\n1. Find a "secret access code ..."`） |
| **serve（TOPN=1 对照，`_top1` 变体）** | 视窗仍 96k、只烘焙 1 页 ⇒ **同样空输出**（stop、86.2 s） |
| **serve（阳性对照，深度 0.30，needle 在视窗头部）** | 视窗路径日志齐全（检索 top-64 里**针页 39 排第 3**、`baked 55 page(s)`）；**输出连贯**（32 chunk、`finish_reason="length"`）；needle 未命中（`max_tokens=32` 截在推理中途）；**但 TTFT 245.9 s ≈ 全量 prefill 量级（0.65 轮 89.6 s）、`Prefix cache hit rate` 升到 45.2% ⇒ 内部路径与 0.65 轮不同，对照未完全定性** |
| 崩溃（已修，过程留档） | boot3 `AttributeError: 'tuple' object has no attribute 'get'`（`KVCacheBlocks.blocks` 是按组 tuple）；boot4 `AttributeError: 'int' object has no attribute 'block_id'`（每步块表是 int 行） |

**⭐卡点（未解）**：N=55 视窗**首个采样即 EOS**；`TOPN=1` 对照已**排除"烘焙数量"**（只烘焙 1 页同样空）⇒ 落在**视窗结构/长度本身**（19,232 正常、96,128 异常）。**阳性对照（深度 0.30，needle 落在视窗头部）结果矛盾、未定性**：视窗路径日志齐全（改写 198,205→96,128、检索 top-64 里**针页 39 排第 3**、`baked 55 page(s)`），**输出连贯**（32 chunk、`finish_reason="length"`，模型在"逐段查找文档"），needle 未命中（`max_tokens=32` 截在推理中途，属预期）；**但 TTFT 245.9 s（≈全量 prefill 量级，0.65 轮只有 89.6 s）且 `Prefix cache hit rate` 由 0.0% 升到 45.2%** ⇒ 这一轮的内部路径与 0.65 轮不同（疑与缓存/重 prefill 交互），**对照本身需重跑**（先打掉缓存再测）。待查线索：`--serve-ignore-eos` 看原始输出、`num_computed_tokens`、前缀缓存命中、**`block_hashes` 在改写后未重算**（仍为原 198,205 token 的 140 个，而 prompt 只 68 块——陈旧哈希理论上可能造成越界命中）。

**两个真缺陷（已修）**：①**完库 sweep 死锁**（两版都死，见 12.14.2）⇒ 改 prefill 期滚动入库；②**`_allocate_slot` 键语义用错**（期望页号、传了 token 偏移）⇒ 中段页在页表里键错位、检索按绝对页号查不到（只有 K1 淘汰的早期页键正确）⇒ boot5 出现"55 槽仅 23 个有页"；修后 55/55 全中。

### 12.14.7 本步的边界（勿误读）

- 单请求单轮形态：视窗 = 头部 + 尾部拼接，**多轮 ΔP（query 段）未实现**——§5.1 执行流水的步骤 1（LCP 差量）留待下一步；因此"prefill 时基于旧工作集"的检索槽内容本步是**占位段**（prompt 原位连续段），不是上一轮的检索结果。
- 检索槽**只覆盖 55 页**（78K token），工作区其余部分不召回；needle 若落在未被选中的页上仍答不出——检索质量本身是 061 的判据，本步判据是"选中的页真能被读到"。
- 烘焙在 host CPU 同步做，TTFT 尾部加烘焙耗时（55 页 × 16 层，实测值见 12.14.6）；阶段 1 不判性能。
- **⭐N=55 大视窗空输出未定性**：视窗改写/入库/烘焙三链都跑通（日志齐全、烘焙 55/55 成功），但 96,128-token 视窗的 serve 首个采样即 EOS；`TOPN=1` 对照已排除烘焙数量，根因未定。**在定性之前，不要把"视窗路径可用"当结论**。
- 滚动入库的块来源必须用**每步的 `kv_connector_block_state.block_ids`**（admission 时刻的副本会漏掉 chunked prefill 后续分配的块）；`_allocate_slot` 的入参是**页号**不是 token 偏移。
- **不能靠"持有块等后续步驱动拷贝"**：请求结束时持有块会把池挤死、没有 forward 步就没有拷贝 ⇒ 死锁（本步实测两版）。任何"请求结束后还要搬东西"的设计都必须让拷贝发生在**同一步的 forward 之后**。
- **identity canary 口径**：视窗只在 `L > B` 且 store 有页时激活 ⇒ canary 用 `L ≤ B` 的请求（或 VIEWPORT=0 的 boot）对照。
- 烘焙在 host CPU 同步做，TTFT 尾部加烘焙耗时（55 页 × 16 层 3.42 s）；阶段 1 不判性能。
- 检索槽**只覆盖 55 页**（78K token）；needle 落在未被选中的页上仍答不出（检索质量本身是 061 的判据，本步判据是"选中的页真能被读到"）。
- **`block_hashes` 陈旧**（已知未修）：改写后 `request.block_hashes` 仍是原 prompt 的哈希（长 140 vs 现 68 块）——本次运行未观察到越界命中（日志 `Prefix cache hit rate: 0.0%`），但属正确性隐患，下一步一并核。
- **⭐⭐本节两处结论已被步骤 073 推翻/更正（见 §12.15）**：①"N=55 空输出怀疑落在视窗结构/长度本身" = **错**，实测为**模型首 token 采出 EOS**（同 boot 只换 nonce 即复现或消失；同一视窗 token 序列当**普通 prompt** 发也照样在 im_end 与换行间近并列；`ignore_eos` 后立刻连贯）⇒ 机制侧 B/C 已排除。②"阳性对照（深度 0.30）给出连贯输出但路径未定性" = **该请求根本没走视窗**（`kvmem_retrieval_*.json` 的 `recent_tokens=32768` 是原生分支指纹，16384 才是视窗）。③"针页排在 top-64、55 页全烘焙"仍成立，且由此把真卡点重定为**窗外 needle 读不出**（同 prompt 原生全量能答出 77349）。

---

## 12.15 步骤 073：072 卡点定性 —— **「N=55 输出为空」不是机制故障，是模型首 token 采出 EOS**；卡点重定为**窗外 needle 读不出**（原生全量能答出 77349，视窗请求答不出）

> 接手先读本节。它推翻 12.14.6/12.14.7 里两处结论（"根因未定，怀疑落在视窗结构/长度"与"阳性对照给出连贯输出"），并把头名从"修 EOS"改成"修检索槽读不到"。

### 12.15.1 归因口径：**A/B/C 三分 + 一条分支指纹**

- **A/B/C**：A = 模型真采 EOS；B = 请求在采样前被引擎结束；C = 采样到但被 stop/流式丢弃。判 A 的最便宜手段是 **同一请求加 `ignore_eos`**（A ⇒ 立刻有输出；B/C ⇒ 仍 0 chunk）。
- **⭐分支指纹 `recent_tokens`**：`kvmem_retrieval_*.json` 里 `recent_tokens` = **16,384 ⇒ 该请求走了视窗**（`_emit_spans` 传 `plan["recent"]`），= **32,768 ⇒ 原生请求**（传 `self.recent_tokens`）。**判"某请求是否走视窗"只能用这个（或 `rewritten onto the compressed window` 的 req id），不能用 TTFT 或"输出是否连贯"反推**——072 就是这么误判的。

### 12.15.2 实测（5 boot，`tools/serve_gsq_kvmem_viewport073.cmd` = 072 臂 + `VLLM_KVMEM_DEBUG=1`）

| 实验 | 结果 |
|---|---|
| **原生 window**（同一视窗 token 序列当普通 prompt 发，96,128） | **正常生成**，首 token 换行 -0.63、im_end -1.01 ⇒ 长度/形状本身不坏 |
| **原生 tail**（只发尾部 16,384，summary 问题） | **首 token = im_end**（-0.37，P=0.69），只出 1 token ⇒ EOS 可在完全无机制参与时发生 |
| **原生 tail 8,192** | 正常生成（think -0.72 / im_end -1.22）⇒ 尾部截断形状敏感，非长度阈值 |
| 视窗 serve，nonce `vp073a`（198,207） | 改写→**68 步 prefill**→打分→**baked 55（3.2 s、688.4 MiB、0 缺页）**→**decode 32 chunk 连贯** |
| 视窗 serve，**与 072 逐字同参数** nonce `vp`（198,203） | **0 chunk、finish=stop（EOS）复现** ⇒ 同臂同 boot 只换 nonce 就能复现/消失 |
| 视窗 serve，nonce `vpi` + **`ignore_eos`** | **94 chunk 连贯**（准确列出文档内容）⇒ **B/C 排除，判 A** |
| 视窗 serve，**focused 窄问题**（只要数字），max_tokens 512 | **340 chunk、推理完整**、逐条点名文档内容后答"没有 secret access code" ⇒ **窗外 needle MISS** |
| **原生 full**（同一 prompt 198,181 走原生，focused） | **答出 77349 ⇒ needle HIT** ⇒ 基准与判据有效 |

**debug 读数要点**：`rewrite computed=0` → `adopted external=0 computed=0` ⇒ **flush 真打掉池块时，072 担心的陈旧 `block_hashes`（全程 139 哈希 vs 68 块）没有造成误命中**（隐患仍在，未修）；`forward1..68` 每步 num=1424；`finish outputs=32/341 status=FINISHED_LENGTH_CAPPED first_ids=[271, 248068, 198]`。

### 12.15.3 ⭐剩余卡点与首要嫌疑（未测，074 的靶子）

> **已由 §12.16（步骤 074）结案**：嫌疑 1 成立并修好（改逐槽 × 逐组发射/烘焙），嫌疑 2 由读回校验排除；窗外 needle 双 boot 命中 77349。下面三条保留为当时的推理现场。

针在 token 125,367 = **页 88**，`top_pages` 里排 **3-9**，`selected = sorted(top_pages)[:55]` **含 88** ⇒ **针页确实被选中并烘焙进了某个检索槽**，但模型读不出。同一内容走原生全量能答 ⇒ 差别只在"烘焙进槽"与"原样在窗内"。

- **嫌疑 1（读码所得，未测）＝组覆盖不全**：`_emit_stage_request` 只取 `tables[group_ids[0]]`（**组 6**）的块表行作为 55 个槽块，而 16 个 full_attention 层被 G=8 分在**组 6 与组 7 两个组**里、同一请求在两组用的是**不同物理块** ⇒ 若如此，**组 7 的 8 层从未被烘焙**，检索区一半层仍是占位段 KV。`baked ... 16 layer(s)` 那行是**层迭代计数**，不证明两组物理块都被覆盖。
- **嫌疑 2＝写了但落点不对**：缺"烘焙后读回"校验（目标块 dequant 回来与重建字节比对 + 按组打印 `(group, block_id)` 覆盖集合）。离线单测只证"槽位位置的字节 == 引擎 writer 在该位置写的页"，**没证"引擎真的把那块当作该请求在该位置的 KV 来读"**。
- **074 做法**：先加读回与逐组覆盖观测（一次 boot 即可判定嫌疑 1），再按结论最小修复；判据不变 = **窗外 needle 命中 + 输出连贯 + 装配链不回归 + 双 boot**。

### 12.15.4 本步的边界（勿误读）

- 本步**没有**改任何机制：改动 = 只读观测（`VLLM_KVMEM_DEBUG` 默认关）+ 工具。072 的代码路径一字未动。
- "EOS 不是机制故障" **不等于**"视窗已经能用"：窗外 needle 仍答不出，**阶段 1 正确性出口仍未达成**。
- EOS 属**采样边界**（im_end 与 换行/空格 差 0.3-0.5 nat），换 nonce/换问题就翻转 ⇒ **不要用"某次有没有输出"判机制好坏**，要用分支指纹 + debug 的 outputs/status。
- **证据管理纪律**：worker 的 `kvmem_retrieval_%03d` **每 boot 从 001 重号**，同一 dump 目录会被后一 boot 覆盖 ⇒ **每个 boot 一个新 dump 目录**（本轮 `kvmem_k8a/b/c`，boot1-3 已存 `kvmem_k8a_boot3keep/`）。
- **工具可用性**：committed 的 `tools/kvmem_viewport_probe.py` 曾引用未定义的 `args.serve_ignore_eos` ⇒ **serve 步骤必崩**（072 实测用的不是这份文件）；本步补齐并加 `--stages serve`（只重发视窗请求，省一次 8 分钟 ingest+flush）与 `--question-style`。

## 12.16 步骤 074：窗外 needle **判据 GO** —— 根因是"检索槽只烘焙了一个 KV 组"，修法 = 逐槽 × 逐组发射/烘焙 + 读回校验

> **本步把 §12.15.3 的两条嫌疑一次结案**：嫌疑 1（组覆盖不全）**成立并修好**；嫌疑 2（写了但没落地）由读回校验**排除**。设计 §5.1 的固定槽位压缩视窗自此在真机上第一次同时满足"免全量重 prefill"与"窗外内容读得出"。

### 12.16.1 为什么会坏：G=8 把 16 个注意力层切成两个 KV 组，而 stage-in 只认第一个

`VLLM_KV_GROUP_SIZE=8` 下 16 个 full_attention 层分成 **kv cache group 6 与 group 7**（boot 日志：`KVMem workspace stores kv cache group(s) [6, 7] (sliding_window=[163072, 163072], block_size=[1424, 1424])`）。同一个逻辑页号在两组落在**不同物理块**上，因此：

- **入库/装配都是逐组展开的**（`_emit_incremental_stores` 里 `for gid in self.group_ids`、066 的 load 同样），host 侧每组每层都有自己的槽 ⇒ 数据一直是全的；
- **只有 stage-in 是单组的**：`_emit_stage_request` 取 `tables[group_ids[0]]`，`KVMemStageInRequest.blocks` 是"每槽一条 `(group, block)`"，worker `_stage_in` 按 `self._layers_per_group[group_id]` 迭代 ⇒ **只写组 6 的 8 层**，组 7 的 8 层在检索槽位置保留的是**占位段原生 prefill 的 KV**。

对模型而言针页变成"半重建"：16 层里 8 层读到针、8 层读到占位内容，注意力在检索区拿到互相矛盾的 K/V ⇒ 窗外 needle 答不出。**旧日志看不出这件事**，因为 `baked 55 page(s)` 数的是槽，`16 layer(s) scored` 是打分侧层数；只有把 MiB 摊开才暴露：`55 × 1,640,448 B × 8 层 = 688.4 MiB`（注册日志明写每组 `8 layers, page 1640448 B`），两组应是 1376.8 MiB。

**一般化教训（进必守）**：任何"逐层/逐槽"的 KV 写入，**必须按 KV 组展开并打印覆盖集合**；`VLLM_KV_GROUP_SIZE>1` 时"层数"与"组数"是两个维度，只数其中一个必然漏。

### 12.16.2 修法（仍只落在 `kvmem_workspace/`，engine 零改动）

- **发射**：`KVMemStageInRequest.blocks` → `slots: list[list[(group_id, gpu_block_id)]]`（逐槽、槽内逐组）。任一组在该逻辑行没有真块 ⇒ **整体截断**，绝不发半覆盖槽；日志改为 `55 slot(s) x 2 group(s) = 110 block(s) ... stored group(s) [6, 7] (covered [6, 7])`。
- **烘焙**：外层按槽、内层按 `(group, block)`、再按 `_layers_per_group[group]` 每层重建（065 精度契约不变：raw K → bf16、cos/sin 保 fp32、单次重建、绝不 delta re-RoPE）；写入字节**实测累加**而不是推算。
- **读回校验**：新旋钮 `VLLM_KVMEM_BAKE_VERIFY=n`（默认 0），每槽每组读回前 n 层目标物理块与重建字节 `torch.equal`。它证的是离线单证不了的那半件事：**引擎真的把字节放进了 decode 会读的那个物理块**。
- **守卫**：各存储组 `block_size` 不一致 ⇒ `_decline_window("mixed-page-size")`（页表与烘焙都以单一页长为键，混长会静默半烘焙）。
- **顺带清账**：`_viewport_rewrite` 原地改 token 序列后清空并重算 `request.block_hashes`（§12.15 记的陈旧 `hashes=139` vs 68 块 ⇒ 本轮 `hashes=67`）。
- **离线覆盖单测**：`tools/kvmem_stage_coverage_test.py`（CPU，12/12）用**按组索引**的假块表快照钉住发射契约：两组齐块 ⇒ 每槽 2 条且两组物理块互不相交、行序一致；组 7 缺行 ⇒ 槽数截断且无半覆盖槽；混长 ⇒ 不发射；worker 侧结构为逐槽嵌套且旧 `blocks` 字段不存在。

### 12.16.3 实测（`tools/serve_gsq_kvmem_viewport074.cmd` = 073 臂 + `BAKE_VERIFY=1`；dump `kvmem_k9a`/`kvmem_k9b`）

- **覆盖与读回（两次 boot 逐字一致）**：`stage-in plan: 55 slot(s) x 2 group(s) = 110 block(s) ... (covered [6, 7])` → `baked 55 slot(s) x group(s) [6, 7] = 880 layer-page copie(s) ... 1376.7 MiB written, read-back 110 checked 0 mismatch`（6.59-6.64 s；`0 without a stored page`、`0 layer-row(s) without authority rows`）。880 = 55 × 16 层，**MiB 正好 073 的 2×**。
- **窗外 needle 双 boot 命中**：针 token **125,370**（depth 0.65，`in-window=False`）= 页 88，视窗请求打分排 top-64 第 9。boot1（`vp074a`，`--serve-ignore-eos`、512 token）首 token 即答案 `first_ids=[271, 22, 22]` → 文本以 `\n\n77349` 开头，TTFT 92.477 s；**boot2（`vp074b`，自然采样、64 token）`text="\n\n77349"`、`finish_reason="stop"`、6 chunk、TTFT 92.072 s** ⇒ 命中 + 连贯，不需要越 EOS。
- **分支指纹自证（§12.15.1 的口径）**：同一 boot 三份 `kvmem_retrieval_*.json` 的 `recent_tokens` = 32768（ingest）/ 32768（flush）/ **16384（serve）** ⇒ 命中确实来自视窗分支。改写读数 `hashes=67`、`adopted external=0 computed=0 rows=1`、`forward68 num=720 end=96128`（68 步 prefill）。
- **收益**：同一 prompt 原生全量 198,184 token 需 TTFT 224-250 s；压缩视窗 96,128 token **92 s**，窗外针仍答得对。
- **装配链无回归**（067 带图臂 + `kvmem_assembly_probe.py`，dump 改指 `kvmem_k9c`）：`matches at 34176 tokens (24 stored page(s))` → `issued (48 page(s) + 6 mamba state block(s), boundary 34176)` → `prefix landed; requested=1 completed=1`，TTFT **224.478 s**（067 223.94 / 066 229.76-230.13 ⇒ 带内），needle 命中、**零 ERROR**、`capture drain: 0 single-token call(s)` ⇒ **decode 仍在 FULL 图上**（必守 21 一等判据未破）。
- **AOT 缓存未破**：两类 boot 都 `Directly load the compiled graph(s) ... 1.488-1.503 s`、`init engine 11.5 s` ⇒ 只改 `kvmem_workspace/` 不改缓存键（062/072 结论第三次复现）。

### 12.16.4 本步的边界（勿误读）

- **"needle 能读出" ≠ "召回稳”**：本步只测了**一个深度（0.65）× 两个 nonce**。针页能否**总**进 `sorted(top)[:55]` 属检索质量问题（§12.8 早期页偏置已判 NO-GO，不可用"关掉缓存"解锁），本步没有改变打分算法，也没有统计命中率。
- **读回不是全覆盖**：`BAKE_VERIFY=1` 只读回每槽每组 1 层（110/880 次拷贝）。全量读回是 1.4 GB 的阻塞 D2H，只应在专门的取证 boot 上开大。
- **不代表性能结论**：阶段 1 不含性能判定；视窗 TTFT 92 s 是**同一请求少 prefill 一半以上 token**的直接结果，不是优化出来的吞吐。臂仍**无投机**，与生产 122.58 tok/s 不可直接比较（用户拍板项）。
- **仍未做**：N/R 预算扫描、recent 段滚动重烘焙、多轮 ΔP、GPU 化烘焙、多轨迹与长稳、`VIEWPORT=0` 的同臂对照。
- **证据目录**：`kvmem_k9a`（074 boot1）、`kvmem_k9b`（074 boot2）、`kvmem_k9c`（装配回归）。装配臂的默认 dump 已从 `kvmem_k5a` 改指 `kvmem_k9c` —— **k5a 是 066/067 的证据，复跑不得覆盖**。

## 12.17 步骤 075：把投机（DFlash2 N=2）接进 KVMem 臂 —— **判据 INCOMPLETE**，但定死两件事

> 用户 2026-10-01 在 074 收尾时拍板"先给 KVMem 臂加投机"。今晚的结论是**这条路还没打通**（8k decode 一次都没测到），但拿到了两条可长期复用的硬事实。**本节所有速度判据均未达成，勿当已验。**

### 12.17.1 ⭐容量互斥（实测，非推演）：投机与 `W=163,072` 视窗在池值 3.4e9 下装不下

- `tools/serve_gsq_kvmem_viewport075_spec.cmd` = 074 视窗臂 + 生产投机三行 + `--cudagraph-capture-sizes 3`，其余逐字节相同。第一次 boot 即被引擎拒绝：`3.24 GiB KV cache is needed ... larger than the available KV cache memory (3.15 GiB) ... estimated maximum model length is 160160`。
- **降 `--max-model-len`（262,144 → 200,704）需求量一字不变** ⇒ 每请求需求由**滑窗**（`VLLM_KVMEM_SW_WINDOW`）钉住，不由 L 钉住；引擎那句"估算最大长度 160,160"是线性外推的读数，不是可用杠杆。
- **降滑窗到 131,072** 才通过容量检查（`Maximum concurrency for 200,704: 1.17x`、K2 `needs 188 / pool has 253`）。代价有两条：①**页长从 1424 变 1456**（引擎给的 `block_size=[1456,1456]`）⇒ `--max-num-batched-tokens` 必须同步钉 1456，而 066 装配线的"步尾必须落在页边界才抓 mamba 快照"口径要按 1456 重验；②KVMem 设计的 **262,144 上下文上限在本臂不可达**（L 只能 ≤ 200,704）。
- 池值 3.4e9 是 必守 6 的硬上限（上探打死 prefill），所以**"投机 + 大视窗"要同时成立就得先解决这 0.09 GiB 的缺口**，而缺口只能从窗口/页长/生成预留里出。

### 12.17.2 ⭐"token 数"在投机下不再能区分 prefill 与 decode（capture allow-list）

- 第二次失败在图捕获：`torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture (cudaErrorStreamCaptureInvalidated)`，栈 = `qwen3_next.forward → compilation/cuda_graph.py → torch.cuda.graph/capture_end`。
- 机制：`capture._record_impl` 认 decode 的唯一依据是 `num_tokens <= 1`（061 立的规则、067 靠它保证"图里没有我们的 device 工作"）。**N=2 投机把 decode 变成 verify 步 = 3 token** ⇒ 它被当成 prefill 录制，于是在**图捕获期间**执行 `clone` 与 M-RoPE 的 `torch.equal`（host 同步）⇒ 捕获作废。
- 修法 = **让连接器说了算**：`KVMemStepSpan.prefill` 新标志（`_emit_spans` 用请求**当前** prompt 长度判定；视窗请求用改写后的窗口长度，因为它 `_req_prompt_len` 仍是原长）；worker `bind_connector_metadata`（forward 之前）`capture.arm({本步各 prefill span 的 num_tokens})`，`clear_connector_metadata` `capture.disarm()`；`_record_impl` 只录 allow-list 内的步，**未授权 = 不录**（连 boot 的 warmup/profiling 步也不录，比 067 的"靠大小猜"更严）。
- 判据升级：`drain` 现在报「单 token 调用数 **+** 落在 allow-list 之外的调用数」，**两个都 0 ⇒ decode 真在 CUDA 图上**（067 判据的推广）。
- **副作用检查（必做）**：无投机 074 臂回归（`kvmem_k9g`）140 步全录、每步 `0 + 0 outside armed sizes [1424]`、打分里**针页 88 仍排第 4**、`evicted=48 stored=326 dropped=0`、视窗 `stage-in plan: 55 slot(s) x 2 group(s) = 110 block(s)` → `baked 880 layer-page copies` → **serve `text="\n\n77349"`、TTFT 92.764 s、needle_hit=True** ⇒ §12.16 的成果未被本步改动破坏。

### 12.17.3 ⭐未定罪的活锁：投机 + KVMem 连接器 ⇒ 调度器一步都不推进

- 现象：boot4（修复后可正常捕获并 ready）上 198,184 ingest 从 21:59:09 到 22:10:29 刷 **43.5 万行** `decline:no-store`、`computed=0`、**0 前向步、0 次 capture drain、GPU util 0%**；boot5 上 8,153 token 小请求同样（`decline:no-mid`、0 前向）。连接器一侧只被反复询问，返回的是正常的 `(0, False)`。
- **判别实验**（`tools/serve_gsq_kvmem_ws075_spec.cmd` = 仅 `VLLM_KVMEM_VIEWPORT=0`，dump `kvmem_k9f`）：请求 22:20:45 进、22:26:00 客户端超时出，`evicted=0 stored=0`、0 drain、0% GPU、**无 decline 噪声** ⇒ **与压缩视窗无关，是"投机 × KVMem 连接器"这个组合让请求无法被调度**。
- 与"每请求块分配失败 ⇒ 留在 waiting 队列反复重问"同向，但引擎自报 `Maximum concurrency 1.17x` 与之矛盾 ⇒ **今晚不作定罪**（判据未达成的部分一律写 INCOMPLETE）。
- **076 的定罪手段（按代价排序）**：①给 `has_pending_stores` / `WAITING_FOR_REMOTE_KVS` / 每请求块需求各加**一次性计数日志**后跑一次 boot，直接看调度器卡在哪一步；②对照臂 = **生产配方（无 `VLLM_KVMEM_*`）+ 同投机**跑 8k，先排除 drafter 自身；③按 `WORKSPACE=1 / RAWK=0 / AUTHORITY=0 / SWEEP=0 / VIEWPORT=0` 逐层剥，看剥到哪一层请求开始动。

### 12.17.4 本步的边界（勿误读）

- **本节没有任何速度结论**。8k decode 在带投机的 KVMem 臂上**一次都没测到**；69.16（无投机、W=163,072）仍是唯一有效的臂内 decode 读数。
- capture allow-list 是**机制级修复**（GO），但它只证明"能捕获、能 ready、无投机路径零回归"，**不**等于"投机可用"。
- 页长 1424 ↔ 1456 的切换会换 **AOT 缓存键**（boot4 `compilation 64.26 s` 首编、boot5 命中缓存 0.024 s）⇒ 任何窗口/页长变更都要按 必守 17 双 boot 或走 watchdog。
- 生产默认**未动**，且生产服务自 073 起一直未起（恢复 = 用户发话）。

## 12.18 步骤 076：KVMem × 投机活锁**归因**（单变量剥离）—— 卡死条件是 `mbt = 页长档 × 投机`，不是 KVMem 的代码

> §12.17.3 留的"未定罪活锁"在本步被剥到只剩一个变量。**归因成立，机制仍未抓行**，所以本节只给"什么组合会死、什么组合能活"，不给根因结论。

### 12.18.1 剥履表（每轮只动一个变量，判据 = 8k 探针能否出数）

| 配置 | 与上一行的差量 | 结果 |
|---|---|---|
| 生产 launcher 原样（L=163,072 / **mbt=1024** / 无 `VLLM_KVMEM_*` / DFlash2 N=2） | 控制组 | **125.41 / 121.68 / 123.93 tok/s，needle 命中** |
| 075 投机臂 + `VLLM_KVMEM_WORKSPACE=0`（连接器不存在；窗口/rawK 仍开；L=200,704 / mbt=1456） | 去掉连接器 | **卡死**（0 前向、0 drain、0% GPU） |
| 上一行再关掉全部 KVMem env（L=200,704 / mbt=1456 / spec） | 去掉滑窗 | **boot 即被拒 `3.81 GiB > 3.15 GiB`** ⇒ 第三次证明每请求需求由滑窗决定 |
| 只把 `VLLM_KVMEM_SW_WINDOW=131072` 加回（无连接器、无 rawK） | 只留滑窗 | **卡死** |
| 上一行 `L→163,072` + `mbt→1024` | 回到生产长度档 | **跑通 120.79 / 124.90，needle 命中** |
| 上一行只把 `mbt→1456` | **唯一变量 = mbt** | **卡死** |

**结论**：`mbt = 页长档（1456；1424 推定同理）+ 投机 ⇒ 任何请求都不被调度`；`mbt = 1024 + 投机 + KVMem 滑窗 ⇒ 正常`。这条死法**与 KVMem 连接器、rawK 捕获、authority、压缩视窗全部无关**（剥到只剩滑窗时仍然死，滑窗在生产档 mbt 下又活）。

### 12.18.2 为什么这对 KVMem 线是硬的

视窗与装配两条线都**要求 `mbt = 页长`**：①066/必守 22① —— mamba 边界快照只在"步尾恰落在页边界"时精确，`mbt` 必须是页的整数倍且与页长一致（9968 那种"7 页"值还会把滑窗 admission 从 117 块抬到 123 块）；②072 的逐页滚动入库按"某页最后一个 token 在本步算完"发 store job，同样依赖页步进。而"页长 mbt × 投机"正是本步的死条件 ⇒ **在机制解开之前，投机和压缩视窗/装配不能同臂共存**。

### 12.18.3 正面可用的组合

**"KVMem 有界 prefill（`VLLM_KVMEM_SW_WINDOW`）+ 投机（N=2）+ `mbt=1024`"是通的**，且 decode 120.79 / 124.90 ≈ 生产 123.93。它的含义：只要"超长 prompt 不炸池 + 生产级解码速度"，不需要检索槽与装配，就有一个现成的可交付变体候选（**注意：本步没为它跑正确性判据**——多深度 needle、窗外召回、双 boot 都欠着，不能作为已验结论）。

### 12.18.4 未定位的机制与 077 的手段

卡死时引擎零 error、零拒绝日志，`Maximum concurrency` 自报 1.00x-1.17x（按其账本一个请求装得下）⇒ 不是简单"池不够"。候选（按可测性）：①`mbt` 非 128 倍数（1456 = 128×11.375、1424 = 128×11.125，而 1024 = 128×8）撞上 spec 的 per-step 预算或编译档校验；②mamba `align` 的 chunk 整除要求遇上 `1 + num_spec = 3` 的 verify 预算；③`SlidingWindowManager.get_num_blocks_to_allocate` 在 `num_tokens + num_spec + 窗口` 组合下返回超限；④`long_prefill_token_threshold` / `max_in_flight` 在 mbt=1456 档把每请求需求抬过池。**077 = 在这四处加一次性计数日志后跑 peel E**（引擎文件改动必须配 revert 脚本，且留一条日志证明补丁在跑），而不是继续读码推演；判死则把 KVMem 的速度目标改成 12.18.3 的组合，压缩视窗的加速另寻路径。

---

## 12.19 步骤 077：投机 × KVMem 活锁的**机制**（抓到那一行）+ 解锁后暴露的 draft 组缺陷

> §12.18 只给了"什么组合会死"；本节给根因、修法口径、以及修好准入后立刻撞出的第二个 bug。**逐 boot 数据在《实验步骤文档.md》步骤 077**，本节只留机制与边界。

### 12.19.1 ⭐ 机制：`draft_slots` 从预算里扣走，但 align 裁块的保护条件不看它

四行算术（全部有日志实证，配置 = `mbt = 1456`、页长 1456、DFlash2 N=2）：

| 步 | 代码 | 值 |
|---|---|---|
| ① | `scheduler.py:954` `request_token_budget = min(token_budget, input_budget − draft_slots)` | `min(1456, 1456 − 2) = ` **1454** |
| ② | `speculative.py:1806` DFlash ⇒ `max_num_new_slots_for_drafting = K` | **2** |
| ③ | `scheduler.py:431-438` 裁块保护 `aligned_end > start or block_size <= max_prefill_tokens`，其中 `max_prefill_tokens = max_num_scheduled_tokens`（**未扣 draft_slots**，日志 `set to 1456`） | `1456 <= 1456` ⇒ **成立** ⇒ `end = 1454 // 1456 × 1456 = ` **0** |
| ④ | `scheduler.py:1000` `if num_new_tokens == 0: break` | 静默 break，跳出整个 waiting 循环 ⇒ **每步 0 token** |

⇒ **死条件不是"mbt = 页长"，而是"分块 prefill 的 prompt 长过 `mbt − draft_slots`"**：boot4（mbt=1457，预算 1455）里 62-token 短请求正常完成、8,153-token 请求永不准入。076 的候选①（mbt 非 128 倍数）被 boot2（1458 活）+ boot4（1457 死）否掉——1458 同样非 128 倍数；候选③④（SWA 块分配 / `long_prefill_threshold`·`max_in_flight`）由计数器直接排除（`alloc_none` / `lookahead_zero` / `pad_break` / `wait_budget_break` 各 0 次，只有 `mamba_align_zero` 涨了 **162,151** 次）。

**这是上游缺陷**：`_mamba_block_aligned_split` 用"块能否塞进一个 chunk"来决定"允许塞不进去就子块推进"，但它比的是 `max_num_scheduled_tokens`，而准入用的是扣掉 draft slots 之后的预算 ⇒ 任何 `mbt ∈ [页长, 页长 + num_spec)` 的配置都会把首块裁成 0，且**没有任何日志**。本战役按用户 2026-10-02 拍板**只在臂参数上绕**，引擎侧补丁未做。

### 12.19.2 修法口径：**`mbt ≥ 页长 + num_spec`**（不是"钉页长"）

`mbt = 1458`（页 1456 + K 2）⇒ 预算 = 1456 = **正好一页** ⇒ 裁块不触发（`aligned_end == end`），步尾仍精确落页边界：**实测 `capture drain: 16 layer(s), 1456 token(s), 0 single-token call(s) + 0 call(s) outside armed sizes [1456]`** ⇒ §12.12/必守 22① 的 mamba 边界快照口径与 072 的逐页滚动入库**都不破**。无投机的臂 `draft_slots = 0`，"mbt = 页长"仍是对的（074 臂 1424 未动）。

| 配置 | 准入 | 读数 |
|---|---|---|
| peel E（mbt=1456 + 投机） | **死** | 0 前向、162,151 次裁零 |
| 1457 + 投机 | **死**（预测成立） | 同上；同 boot 短请求可跑 |
| **1458 + 投机（peel 形）** | **活** | boot2 首次 18.03/16.95（该档冷 boot 退化态）→ boot3 **108.73 / 103.45**、needle 3/3 |
| **1458 + 投机 + 视窗臂** | **活** | boot6 8k **103.67 / 108.10**（075 一次都没测到）；但 ingest **~11 s/页步**（074 = ~1.7 s） |

### 12.19.3 第二个 bug：投机的 draft 组被 §12.5 的 workspace 门误认领

`single_type_kv_cache_manager.py:140` 的 060 判据是「`isinstance(滑窗管理器)`」⇒ 加上投机后**草稿模型自己的滑窗组（组 8，5 层）**也 arm 了 `_retain_for_workspace`；`kv_cache_manager.take_workspace_evictions()` 原样按 `mgr.kv_cache_group_id` 发出组 8，而连接器的存储组只有 `[6, 7]` ⇒ `manager.py:324 self.block_size[group_id]` **`KeyError: 8` 打死 EngineCore**（boot5，ingest 推进到 job 2 时）。修法（`kvmem_workspace/manager.py::build_connector_meta`）：**eviction 循环先按 `self.group_ids` 过滤**，非存储组的块**立刻 `block_pool.free_blocks([block])` 归还池**（060 故意没减它们的引用计数，不还就是**块泄漏**），并新增 `pages_nonstore` 计数进累计日志。boot6 实测 `stored=474 dropped=0 nonstore=134`、零 ERROR。

**潜在第二处（本步未修，只登记）**：`_retained_by_block_id` 只按 `block_id` 索引、不带组号 ⇒ 不同组的同号块可能假命中"我保留的块"。boot5 崩在分配而非假命中，但多组并存（投机 / G=8 分组）时这条是真实风险，修 needle 判据之前应先把它改成 `(group_id, block_id)`。

### 12.19.4 本步的边界（勿误读）

1. **"准入通了" ≠ "判据通了"**：074 的**窗外 needle 判据在带投机的臂上未测**（`rewritten onto the compressed window` / `stage-in plan` / `baked` 各 0 次；唯一检索报告 `recent_tokens=32768` = 原生分支）。两条成因已记死：①探针的 `timeout 1500` 在 11 s/页下砍在 serve 之前（三请求需 ~50 分钟）；②`--stages serve` 复跑时**换了 nonce ⇒ 换了轨迹**，工作区里没有它的页，引擎直接原生 prefill。用 `--stages serve` 必须**沿用同一 boot 同一 `--nonce`**。
2. **decode 读数只有单 boot**：103.67 / 108.10 是同一 boot 内 2 次重复，落在 boot 间连续谱（83-129）里，**不得**据此宣称"投机把臂拉到了生产级"；与 074 的 69.16 基线比也要注意两者页长不同（1424 vs 1456）。
3. **ingest 6× 慢未定位**：这是新的头号性能卡点——KVMem 的卖点是"免全量重 prefill"，若入库本身慢 6×，收益会被吃回去。候选：draft 组每步的块分配与 `num_lookahead_tokens=3` 的额外预留、capture allow-list 在 1456 档的图/编译形态、`nonstore` 块的归还节奏与 K1 复制串行。
4. **`mbt ≥ 页长 + num_spec` 只是绕开**：引擎里那条保护条件仍用错量（见 12.19.1）。任何"把 mbt 往页长上靠"的新臂（含装配臂换页长）都要按这条重算。

### 12.19.5 未验证项（078 的靶子）

①窗外 needle 判据在投机臂上跑通（同 nonce + flush + 指纹确认走视窗）；②ingest 6× 慢的归因与修复；③`_retained_by_block_id` 改组号键；④命中率统计（多深度 × 多 nonce）、`VIEWPORT_PAGES`/`_RECENT` 预算扫描、recent 滚动重烘焙、多轮 ΔP、GPU 化烘焙——**全部仍未做**。

## 12.20 步骤 078：窗外 needle 判据在**带投机的臂**上 GO；但 077 的"ingest 慢 6×"作为**配置属性**的前提被实测推翻

**零代码改动**：本步没动 `vllm/` 任何文件（引擎、连接器、`kvmem_workspace/` 全未改），只新增两个工具——变体臂 `tools/serve_gsq_kvmem_viewport078_spec.cmd`（`S078_TAG`/`S078_SPEC`/`S078_RAWK`/`S078_AUTH`/`S078_WS` 全部可用环境变量注入，一次 peel 不必新建文件）与单请求节拍探针 `tools/kvmem_ingest_cadence.py`（客户端 `TTFT / ceil(tokens/页长)` + 引擎侧 `capture drain` 相邻差的中位/p10/p90 两个独立读数）。

### 12.20.1 ⭐判据达成（b2 = `kvmem_k12c`，nonce `vp078a`，200,000 tokens / depth 0.65）

074 的四条判据（窗外 needle 命中 / 输出连贯 / 装配链不回归 / 双 boot）逐条对账：

| 判据 | b2 实测 |
|---|---|
| 窗外 needle | 针 token **125,370**（页 88，`in_window=False`）；serve 输出 `"\n\n77349"`、`needle_hit=true` |
| 连贯 | `finish_reason=stop`、6 chunk、无 `ignore_eos`（与 074 boot2 同形态） |
| 走了视窗（必守 16⑦） | 三份检索报告指纹 = ingest `32768` / flush `32768` / **serve `16384`**；且 serve 有 `rewritten onto the compressed window: 198184 -> 97920` |
| 逐组覆盖（必守 16⑨） | `stage-in plan: 55 slot(s) x 2 group(s) = 110 block(s) ... (covered [6, 7])` → `baked 55 slot(s) x group(s) [6, 7] = 880 layer-page copie(s) in 7.02 s; 1407.7 MiB written, read-back 110 checked 0 mismatch` |
| 全图兼容（必守 21） | 340 条 drain 全部 `0 single-token call(s) + 0 call(s) outside armed sizes`，serve 的 decode 是 `num=3` 的 verify 步 ⇒ 投机解码仍在图上 |
| 装配链 | 本步零代码改动 ⇒ 无回归对象；未重跑装配探针（见 12.20.5） |
| 双 boot | **b2（快 prefill 态）+ b3（慢 prefill 态）两支都命中**，见 12.20.2 表与 §12.20.2 末段 |

算术自证：1407.7 MiB / 074 的 1376.7 MiB = **1.0225** = 页长 1456/1424 ⇒ 覆盖量与 074 完全同构，只是页更长。收益账：同一 prompt **原生 ingest 258.994 s** vs **视窗 serve TTFT 105.066 s**（40.6%），页 88 在 serve 打分里排 **top-64 第 40**（074 是第 9），仍在 `sorted(top)[:55]` 的预算内。

**b3 = 慢 prefill 态的第二支 boot，判据同样成立**：`needle_hit=true`、`text="\n\n77349"`、`finish=stop`、5 chunk、`stage-in plan 55 x 2 = 110 (covered [6,7])`、`baked 880 copies in 6.94 s / 1407.7 MiB / read-back 110 checked 0 mismatch`、页 88 排**第 39**、指纹 `recent_tokens=16384`、ERROR/Traceback 各 0、347 条 drain 全 `0 + 0`。两态对照给了这条线最重要的一句结论：**慢态吃掉的是绝对时间，不是 KVMem 的相对收益**——省下的 prefill 比例在快/慢两态分别是 **40.6%**（105.066 / 258.994）与 **48.6%**（608.063 / 1250.341）。

### 12.20.2 ⭐"ingest 慢 6×"不是配置属性——四 boot 对照（同一 launcher、同一 AOT 缓存）

| boot | 页步（1456 tok） | 8k decode | 采法 |
|---|---|---|---|
| 077 boot6（`k11b`） | **11.0 s** 中位（p10 10 / p90 11，自第 11 步起恒定，201 条 drain） | 103.67 / 108.10 | 077 据此写下"慢 6×" |
| 078 b1（`k12b`） | **1.42 s**（客户端 44.058 s / 31 步；drain 中位 1 s） | 未测 | cadence 探针 |
| 078 b2（`k12c`） | **1.89 s**（ingest 258.994 s / 137 步；drain 中位 2 s、p90 2 s） | **62.03 / 62.82 / 64.00** | 三请求全流程 |
| 078 b3（`k12d`） | **9.13 s**（ingest 1250.341 s / 137 步；drain 中位 9.0 / p10 8 / p90 10；冷 8k TTFT 47.844 s / 5.6 步同向） | **104.44 / 108.85 / 111.67** | 锚点在 ingest 之前 + 三请求全流程 |

⇒ 077 的"投机 × 连接器 ⇒ ingest 慢 6×"**作为配置结论作废**：完全同一份 launcher（只差 dump 目录与 077 那条已 revert 的 `VLLM_SCHED_TRACE` env）在 boot 之间跳 1.4 ↔ 11 s，跨度 7.8×。b2 的 1.89 s/步与 074 的 ~1.7 s/步（页长 1424）同量级 ⇒ **074 与 077 之间的"6×"是 boot 态差，不是投机差**。

同时暴露一条新的、更硬的现象：**同一臂的 prefill 与 decode 在 boot 之间呈反相关**（慢 prefill ↔ 快 decode：boot6、b3；快 prefill ↔ 慢 decode：b1、b2），n=4 各两例。decode 差不是投机失效：b2 `accepted/draft_tokens = 429/718 = 0.597`（每步多产 1.195 token）、boot6 = 415/736 = 0.564（1.126），几乎相同，但 256-token 段的每 token 墙钟差 1.7× ⇒ 差在 GPU 侧执行速率。

### 12.20.3 慢态的指纹（实测，未定罪）

在 b3 的慢态 ingest 期间外置观测（`nvidia-smi`，5 s 一次）：

- **`utilization.gpu` 100%、`clocks.sm` 3060-3067 MHz（=boost）、`memory.used` 15,834 MiB、`temperature` 55 °C**，但 **`utilization.memory` 只有 1%、功耗 82-105 W**（power limit 350 W）。
- **PCIe 链路 `gen.current=5 / width.current=16`，与 `max` 相同** ⇒ "链路降速"当场否掉。
- 六个 boot 的宿主页日志逐字相同：`200 host slots (2.50 GiB pinned)` ×2 组，且**没有任何** `pinned host allocation ... falling back to pageable` 警告 ⇒ `worker._alloc_host()` 的 pageable 回退这条解释**否掉**。
- 启动横幅无可观测差：`Initial free memory 14.68 GiB` / `GPU KV cache size 234,000 tokens` / `num_gpu_blocks=253` / `reserved 3.17 GiB` / `compilation 4.39-5.08 s` 两态一致；pin_shim 日志 `PIN_MOVED p=pass1 moved=44 params=113 skipped=4 mb=545` 在 boot6 与 b2 逐字相同 ⇒ 快慢态钉（030/033/044 那条 WDDM 掷骰子线）**不是**这次的开关。
⇒ 慢态画像 = **"kernel 常驻但不烧算力也不烧显存带宽"的等待型**，不是算力型也不是带宽型；与"显存被挤"（049 判据）也不同向。每步入库量算术：一个 slot = 25.59 MiB（每组 8 层 × page 1,677,312 B × 2 组），`wait_for_save` 把 16 条 `(src_ptr, dst_ptr, 1,677,312 B)` 交给 `ops.swap_blocks_batch` 一次发完 ⇒ 若 8 s 全花在入库上，等效带宽 3.3 GB/s（远低于 Gen5 x16 与该路径的量级）⇒ 时间**大概率不在**拷贝本身，但**没有计时证据**，本步不定罪。

现成的读数拿不到：`worker.stats()` 里有 `store_seconds`/`bytes_stored` 累加器，但 `KVMemConnector.workspace_stats()` **全仓无调用者**（既没接路由也没落文件）⇒ 079 的第一手工具 = 给它加一条门控日志（或直接在 `wait_for_save` 处按 50 步打一条），并配 revert 脚本。

### 12.20.4 12.19.5 里第③条（`_retained_by_block_id` 改组号键）**前提不成立，不动代码**

077 的担心是"组 6/组 7/组 8 的块号可能同号 ⇒ 假命中"。核账结果相反：
- `kv_cache_coordinator.py:99` **全引擎只有一个 `BlockPool`**；`kv_cache_utils.py:166` 明写 `block_id` = "ranging from 0 to num_gpu_blocks - 1"，`KVCacheBlock` 是**终身对象、id 固定**，不随组重新编号。
- 发出侧 `kv_cache_manager.py:895` 用的是 `block.block_id`（同一个全局 id），登记侧 `manager.py:321` 用 `block.block_id` 存**同一个对象** ⇒ 两侧同一命名空间，`block_id` 单独作键已经无歧义。
- 反而**改成 `(group_id, block_id)` 会立刻坏**：`register_workspace_retained_blocks(blocks)` 只收到块对象、拿不到组号 ⇒ 登记键与查回键不同 ⇒ 每次查回都 miss，块既不被 copy 也不 `free_blocks` ⇒ **真块泄漏**（077 修的正是泄漏）。
⇒ 结论：登记未修的判断作废；`KeyError: 8` 那类跨组误认领已经由 077 的 `self.group_ids` 过滤修掉，键本身不需要动。

### 12.20.5 本步的边界（勿误读）

1. **"判据 GO"只覆盖 depth 0.65 × nonce `vp078a`/`vp078b` 两点**，与 074 的"一个深度 × 两个 nonce"同规格；页 88 的排名在投机臂上从第 9 掉到第 40 ⇒ **离 55 槽预算只剩 15 名余量**，命中率统计（多深度 × 多 nonce）欠得比 074 更紧了。
2. **cadence 的 1.42 s（b1）与 1.89 s（b2）不可当"臂的 ingest 速度"**——同一配置能出 1.4 也能出 11，任何 ingest/TTFT 数字**必须先报页步中位再报结论**，且要 ≥3 boot。
3. **"慢态成因"未定位**：本步只否掉了 pageable 回退、PCIe 降速、显存被挤、pin 快慢态、配置差（投机）五条候选，**没有**正面定罪。
4. **b3 的判据落在慢态** ⇒ 它证明的是"**正确性与 boot 态无关**"（慢 boot 也能读出窗外针），**不**构成性能证据；两支的绝对时间差 5.8×（serve 105 s vs 608 s）。
5. **证据目录卫生（发现一处历史违例）**：`prod029_logs/kvmem_k9b/` 的 `kvmem_retrieval_00{1,2,3}.json` 与 `kvmem_remat_*` 时间戳是 **10-01 22:28-22:39（077 的 `k9g` 回归跑）**，而 `vp074b.json` 是 10-01 19:25 ⇒ 074 boot2 的检索证据已被 077 复跑覆盖（必守 16⑧）。**读 074 的打分排名请以《实验步骤文档》步骤 074 的原文为准**，不要以 `kvmem_k9b` 目录当前文件为准。

### 12.20.6 未验证项（079 的靶子）

①给 `workspace_stats()`/`wait_for_save` 出门控计时日志，在**慢态 boot** 上读 `store_seconds` 与步时之比（先判"时间在哪"再谈修法）；②若入库不是主因，同法判 `capture.drain` 与 `_ingest`（raw-K 折叠）两段；③prefill↔decode 反相关需要 n≥6 才谈"规律"，每次 boot 必须两个都测；④命中率统计（多深度 × 多 nonce）、`VIEWPORT_PAGES`/`_RECENT` 预算扫描、recent 滚动重烘焙、多轮 ΔP、GPU 化烘焙、装配探针在新页长（1456）上的回归——**仍全部未做**（①②是本线交付形态的前置：KVMem 的卖点是"免全量重 prefill"，若慢态随机出现，收益账不能算）。

### 12.21 ⭐计时证据：一个整页步的钱花在哪（步骤 079，2026-10-02，门控补丁 `VLLM_KVMEM_TIMING`）

**机制（本节新建立的事实，按实测而非推演）**

1. **一个整页步 = 四块**：①`outside`（连接器根本没被调到的部分：这一页的模型前向发射 / 调度 / 图重放 / runner 里可能存在的更早的隐式设备同步）②`sync`（`capture.drain()` 里**第一次** `.cpu()` 返回之前 = 等本步已异步排上的算子跑完）③`copies`（drain 里其余 47 次阻塞 D2H 的净搬运）④`host_py`（连接器 Python：`_ingest` 的位置掩码与索引折叠 + 组装 16 条 copy 条目）。另有三项在本臂上实测为零：**`copy_issue`**（`ops.swap_blocks_batch` 的发射）、**`selftest`**、**`bake/snap/load`**（视窗装配链在这几支请求里没走）。
2. **`drain` 必须拆成 `sync` + `copies` 才有意义**。第一次 `.cpu()` 会等整条流，所以不拆的话"模型自己的计算时间"会被记成"拷贝慢"——这正是 078 那句"若 8 s 全在入库则等效 3.3 GB/s"差一点踩进去的坑。拆点在 `capture.py` 的 drain 循环里（`_DRAIN_ACC["sync"] / ["copies"]`，经 `capture.stats()` 暴露，被台账折成 `sync`/`copies` 两段）。
3. **快态的四 boot 归因高度一致**（b2/b4/b5/b6，整页步窗口中位）：`dt/步` 1.273-1.433 s（与客户端页步 1.307-1.464 s 差 ≤3%，两条独立口径互洽），`host_py` **0.141-0.142 s/步**、`copies` **0.028-0.031 s/步**、`sync` 0.037-0.169 s/步、`copy_issue` **0.000 s**、`outside` **1.060-1.117 s/步（76-84%）**。残差 `|save − accounted| = 0.000`。
4. **⇒ 结论级事实**：**KVMem 工作区连接器的自身开销 ≈ 0.17 s/页步（整页步的 ~12%）**；入库的合批**发射是免费的**，净 D2H 只有 0.03 s/步。**"修拷贝形态（是否真合批 / 上 GPU 化烘焙）"作为 ingest 提速路线判死**——它没有可回收的时间。078 的 3.3 GB/s 推测被否。
5. **连接器不参与 decode（现在有计时证据）**：`wait_for_save` 首行 `if metadata is None: return`，纯 decode 窗口里 `save = 0`，台账顺手给出**毫秒分辨的 decode 步时 0.024-0.025 s/步**。⇒ "KVMem 拖慢 decode"这类直觉不成立，decode 的快慢要在连接器之外找。
6. **自测的同步只在窗口含引擎起步时显形**：b1 首窗 `dselftest = 1.454 s/25 步`（0.058 s/步），b2-b6 的整页步窗口 `≈ 0.000`——因为 `drain()` 的 `.cpu()` 已经把流同步过，随后自测里的 `torch.cuda.synchronize()` 立即返回。⇒ **`SELFTEST=1` 不是 ingest 的固定税**（与 078 的担心相反），但**它确实会计入 `store_seconds`**（同一计时窗口内），所以裸读 `store_seconds` 依然不可信。

**边界（勿误读）**

- **这是快态的归因，不是慢态的**。慢态今天 0/5 boot 抽到 ⇒ **慢态那 8-11 s/页乘在哪一块仍未定罪**（§12.20.3 的"等待型"画像依旧只是画像）。
- `outside` 只能说"不在连接器里"。**不能**据此说"是 GPU 在算"——GPU 算的那部分已单列为 `sync`。它的真实归属（host 发射 vs runner 的更早同步）需要 scheduler 侧一次性诊断，本步**故意没打**（快态的 `outside` 与模型自身 prefill 吞吐 987-1105 tok/s 同量级，先解释一个已经能解释的数是浪费 boot；靶子是慢态样本）。
- **`S079_WS=0` 不是干净对照**：那支（b3）页步 1.588 s（不比开连接器快，与"连接器只占 ~12%"同向），但 8k decode 塌到 **12.26 tok/s**（比 078 记过最差带 62-64 还差 5×）。我给的"KV 组布局变了"解释**未被证实**（`Add 3 padding layers` 两 boot 都有、引擎配置横幅逐字无差）⇒ 机制未定位，且**结论是方法论的**：臂上的 env 旋钮彼此耦合，"关掉一个连接器"不等于只差一个变量。
- 台账是**门控**的（`VLLM_KVMEM_TIMING` 默认 0）⇒ 生产路径的行为差别只有几个 `time.monotonic()` 局部赋值；6 boot 全部 `Directly load … took 1.519-1.545 s` ⇒ **062 的"改 `kvmem_workspace/` 不换 AOT 键"扩到 `capture.py`**；`capture drain` 每 boot 51/52 条全 `0 + 0` ⇒ 全图一等判据未破。
- **b1 用 `EVERY=25` 是失败设计**：42 页步只出 `dt=0` 首窗 ⇒ 不可归因（分析器正确拒判）。校准值：**`EVERY=10`** 对 60k 探针（得 3-4 个整页步窗口）。b1 还早于字段改名（`copies=` → `copy_calls=`），分析器按现行格式**拒绝解析**它，这是有意为之。
- 分组纪律（分析器的三条现场口径）：整页步 / 短请求步（8k 探针那种收尾块，`dt/步 ≈ 0.31 s`，`score_pure` 占 36-38% = 0.120 s/步）/ 纯 decode **必须分开取中位**；`dt` 含两次探针之间的空闲 ⇒ 用客户端 `s_per_page_step` 剔 `dt/步 > 2×` 的窗口；`|save − accounted|` 不清零不许定罪。

**未验证项（080 的靶子）**

①慢态 boot 的同款四分表（**0/5 未抽到**，这是 080 第一优先）；②`outside` 的归属（已授权的 scheduler 侧一次性诊断，等第一次真慢态再打）；③`host_py` 0.141 s/步 能否降（唯一还在连接器里的可回收项）；④"一次 boot 同时 ~1.4 s/页 且 ≥100 tok/s"今天 **0/6** ⇒ 按判据是 NO-GO，不是未测；⑤078 的两条 decode 带（62-64 / 104-111）未复现，今天落在 71.94-94.47 的连续带 ⇒ n 还不够谈规律；⑥预算扫描的判据已换口径（见 §12.22 的槽位几何），`TOPN↑` 会把针页挤出槽。

### 12.22 ⭐检索槽预算的真实口径（步骤 079 的离线核账，`tools/kvmem_slot_budget.py`）

**先更正一条被当成事实引用了两步的数**：078/提示词里的"针页名次从 074 的第 9 掉到第 39/40，距 55 槽还剩 ~15 名"**不成立**，成因是页号没跟着页长改：

- 针的绝对 token 位由探针自己写进日志（`step078_vp078a.log` / `vp078b.log`：`needle at token 125370`）。074 那批臂页长 1424 ⇒ `125370//1424 = 88`（与 074 原文"深度 0.65 = 页 88"一致）；075 把 `SW_WINDOW` 降到 131,072 后页长变 **1456** ⇒ `125370//1456 = 86`。⇒ **真针页是 86**，078 在 `top_pages` 里查"页 88"查到的是另一页（1456 页号 88 覆盖 token 128,128-129,583）。
- 实测 `kvmem_k12c/kvmem_retrieval_003.json`（b2 快态 serve）与 `kvmem_k12d/…_006.json`（b3 慢态 serve，两份 `recent_tokens=16384` ⇒ 都是视窗分支）：**页 86 的 `rank_by_logit` 在两 boot 都是第 9**，与 074 记录的"第 9"**同值** ⇒ **针页名次没有退化**。页 88 才是第 40 / 第 39。

**三个"名次"是三个量，以后必须标明字段**：

| 口径 | 出处 | 值（b2 / b3） | 用途 |
|---|---|---|---|
| `rank_by_logit`（跨层**均值** logit 降序，限定 eligible） | `index.py:517-519`；仓内 `kvmem_k3_probe.rank_of` 用的就是它 | 9 / 9 | 历史文档里的"第 9 / 第 4"是这一口径 |
| 引擎实际选页序（`top_pages` 里的位置） | `index.py:501-521`，`page_scores` = 逐层 softmax 后求和 | 11 / 11 | 决定"进不进 top-N" |
| **时间序槽位（`sorted(top_pages)` 里的位置）** | `worker.py:953 selected = sorted(top_pages)[:len(stage.slots)]` | **53 / 53** | **真正决定"进不进 55 槽"的判据** |

⇒ **真实余量 = 55 − 53 = 2 名**，比交接档写的 ~15 名**更紧**（方向相反：不是"名次掉了"，是"槽位几乎满了"）。

**扫描的可算性边界（离线能做什么、不能做什么）**

- `TOPN`、`VIEWPORT_PAGES`：**可精确离线重算**（名次序不随截断变，改的只是取前多少 / 填多少槽）。
- `VIEWPORT_RECENT`：**换的是 eligible 集合本身**（`index.py:524-536 end = num_pages − recent//block_size`），而 `page_scores` 是"对 eligible 逐层 softmax 再求和"的结果 ⇒ 换 eligible 就换了归一化，**不能**从 dump 的分数精确重算，只能用 logit 口径近似（分析器对此类格点标 `approx`）。
- 现成证据只有 **1 深度 × 2 nonce**；`VLLM_KVMEM_DUMP_KBAR` 未开 ⇒ dump 里没有 kbar 向量，**换归约 / 中心化不可离线复算**。⇒ 这套输出是**预算几何**，不是命中率统计。

**两条结构性发现（080 的先验）**

1. **`TOPN` 不是单调旋钮**：把 TOPN 从 64 抬到 96，针页会被**挤出** 55 槽（实测 `topn=96, pages=55` ⇒ 不进槽，`pages=80` 才回槽）。机理：槽是按**时间序**填的，多收进来的低页号把针页的时间序位置往后推。⇒ "加大 TOPN 提高命中"的直觉在这个布局下是**反的**。
2. **boot 态不改检索名次**：b2（快 prefill）与 b3（慢 prefill）在同一针页上给出**逐字相同**的 `eligible=125`、`num_pages=137`、`rank_by_logit=9`、时间序槽 53 ⇒ 快慢态影响的是**时间**，不是**选页**。
3. 不可比声明：074 的原始 dump 已被 077 的 `k9g` 回归跑覆盖（违必守 16⑧，只剩 `vp074b.json`），而页长不可改小（必守 22①）⇒ **"第 9 → 第 11（引擎序）是不是页长造成的"没有数据可判，只能标 unverified**。

工具：`tools/kvmem_slot_budget.py`（纯离线）。**自检**：用 dump 里的 `page_scores` 重排前 64 名，与引擎自己写出的 `top_pages` 逐字相同 ⇒ 六变体 × 两 dump 全 `replay_ok=True`，重实现可信；回归锚 = 必须复现"针页 86 / logit 序第 9 / 引擎序第 11 / 时间序第 53 / 余量 2"。

### 12.23 ⭐慢态计时与 `record` 同步定罪（步骤 080，2026-10-02，py-spy + 带外指纹 + 门控修法）

**结论级事实（按实测，不按推演）**

1. **079 的 `outside` 是个筐，不是归因**。`capture.record` 是 `torch.library.custom_op`，在 **`execute_model` 内部**每个 full-attention 层被调一次；079 的台账只从 `wait_for_save` 里取窗口，所以 `record` 从来没被计量，全部落进 `outside`，并被 079 读成"连接器之外 = 模型前向/调度/图侧"。**必守 24① 的反例就是这一条**（未计量的段读起来像无罪）。本步起台账有第 14 段 `rec=`，分析器把 `record` 从 `outside` 剥出来，并新增判定档 `(d)`。
2. **慢态的栈顶只有一行**。b4（门控关，9.78-9.80 s/页步，同 boot 4 次复测）录到 7,640 个 GIL-holding 样本，**94.62% 栈顶 = `vllm/v1/kvmem_workspace/capture.py:166`**，即 `torch.equal(positions[0], positions[1])`；inclusive 链 `execute_model → qwen3_next.forward → piecewise/cuda_graph → record → _record_impl` = 94.63%。这行**每个 full-attention 层跑一次 ⇒ 每页步 16 个 CUDA 流同步**，违反必守 16⑥（device→host 一律留在 `drain()`）。
3. **修法把同步归零，但等待换了地方**。`VLLM_KVMEM_RECORD_NOSYNC=1`（默认 0 = 旧行为逐字不动）把 `positions[:2]` 一起暂存，同轴判定挪到 `drain()` 里用 numpy 做（那份数据本来就要 `.cpu()`）。实测：`cap(... rec_sync=0)`、CPU 单测 20 项（canary 计数不变、`drain()` 交出的一仍是 1-D 行 0）；b6/b9 的栈顶搬到 `drain` 的第一次 `.cpu()`（91.65%），`execute_model/forward` 只剩 2.48%。**页步：慢态 9.784 → 8.384（−14%），快态 1.269 vs 门控关 1.315-1.452（不变）** ⇒ **修复的是不变量，不是速度**。
4. **慢态的 GPU 没在算**（这是本步真正的判据）：快态 `power.draw` 中位 **196.7 W**、`utilization.memory` **13%**、**60 °C**；慢态 **84-87 W / 1% / 47 °C**；SM 3060 MHz、显存 14801 MHz、PCIe gen5 x16 = max **两态相同**；`pstate` 都是 P1、`clock_reasons` 都以 GpuIdle(0x1) 为主。"util≥90% 且 <130 W 且 mem-util<10%" 占比：快态 **0.000**，慢态 **0.64-0.84**。
5. **四条候选被直接否掉**：①**显存被挤/权重页落系统内存**——引擎专用/共享记账在快慢两态**逐字相同**（15,780.5-15,792.5 / 8,654-8,666 MiB，40 s 轮询 8 次无变化），且 8.66 GB 共享是本臂"锁页工作区 5.00 GiB + raw-K 权威区"的正常足迹（**不可套生产 `共享 − 8,298` 公式**，必守 19）；②**主机内存压力/换页**——**快态的空闲 RAM 反而更低**（0.71 GB vs 慢态 5.33/1.47 GB），慢态 pagefile 与磁盘读平直（~0 MB/s）；③**降频/热墙**——两态时钟相同、慢态更凉；④**链路**——gen5 x16 = max。
6. **同一支慢态 boot 的 decode 更快**：b4 的 8k 锚点 = 110.99/115.94/116.05（独立采样窗口），快态 b8 = 96.15；台账在 decode 窗口 `save = 0`。⇒ 反相关成立且有计时证据；单图提交（decode）不受影响，慢的是"多 kernel 提交的整页前向"。
7. **机制未定罪的准确说法**：能说的是"整页前向的提交-完成握手被拖住 ×7-8，而 GPU 自身强度极低"；不能说的是"被谁拖住"。本步取证手段的边界：`gpu_apps` 只有引擎自己的 8 个 PID（无他占客户端读数）、py-spy 不带 `--native` 就看不见驱动调用、`--idle` 采样（b9，39,893 样本）里 `schedule`/zmq 帧 ≈0 ⇒ **scheduler 侧一次性诊断（079 已授权）判定为不必打**，因为它不能改变任何决策。

**边界（勿误读）**

- **"KVMem 无责"与"KVMem 有罪"都不成立**：连接器的自有开销（`host_py` 0.141-0.144、净拷贝 0.030-0.034）在两态**逐字不变**；它贡献的是**同步形态**（把等待显形化的位置），不是等待的量。079 那句"连接器只占 ~12%"因此需要限定：**它只统计了 `wait_for_save` 窗口**。
- **态会在 boot 内漂**：b11 起服 1.44 s/页，同 boot 同 prompt 只重发 serve 的 TTFT 108.0 → 180.3 s。所以"按 boot 取中位"是**下限**口径，跨 boot 比较时必须记录测量时刻。
- **抽样率**：本步 11 boot = 6 慢 / 5 快，且沿时间成簇（13:30-13:37 全快 → 13:38-15:12 几乎全慢 → 15:13 又快）。与 078（4 boot 2 慢）、079（6 boot 0 慢）合起来 ⇒ **"再抽几次一定抽到"仍不成立**，任何"快慢态发生率"的数字都要带时段。
- **`VLLM_KVMEM_RECORD_NOSYNC` 默认 0** ⇒ 生产与所有历史 boot 行为不变；本步所有正确性判据（needle/指纹/全图）都是在门控开的臂上重取的，**没有**沿用 078/079 的结论。

**未验证项（081 的靶子）**

①把 `record` 整条路径摘掉后的页步地板（`S079_RAWK=0` 是现成旋钮，属 078 peel 血统，**注意它同时改变工作**，只能当"地板参考"不能当干净对照）；②驱动级取证（`--native` 需符号，或换 ETW/xperf 类工具）；③多深度命中率网格（本步只 0.65 × 2 次采样，判据口径已改：单次 32-token serve 会把命中读成 miss）；④`record` 段在 079 六支旧 boot 上的回溯（旧日志无该段，只能标"未知"）；⑤recent 滚动重烘焙、多轮 ΔP、装配探针在 1456 上的回归、阶段 1 出口五项。

**工具（本步新增，全留存）**：`tools/kvmem_boot_fingerprint.py`（带外 1 Hz，`run|summarize`）、`tools/step080_boot.py`（`next|classify|measure|vp|stop|verify|show`，状态落 `prod029_logs/step080_boots.json`）、`tools/apply_kvmem_record_sync_step080.py`（三态 + `--target`，可逆性已实测）、`tools/kvmem_record_nosync_test.py`（CPU 20 项，跑臂前闸门）。证据：`step080_b{1..11}_arm.{out,err}.log`、`step080_b4_pyspy_slow.raw`（7,640 样本）、`step080_b6_pyspyS.raw`、`step080_b9_pyspyI.raw`（39,893 样本，`--idle --threads`）、`step080_b{4,6,8}_fingerprintS.{csv,json}`、`step080_b{4,6}_headroomS.json`、`step080_b{4,6,8,9}_timebudget*.json`、`step080_b11_vp_d65_a.log` + `kvmem_k14k/vp080a_d065{,_ignoreeos}.json`、`step080_slotbudget_{078anchor,b11}.json`；dump 目录 `kvmem_k14a…k14k`（每 boot 一个新名）。

### 12.24 ⭐kernel 级取证：慢态归口到量化 GEMM 的每调用双峰；连接器第三次无责；槽位选取门控（步骤 081，2026-10-02）

**结论级事实（全部实测，工具与取证路径在步骤 081）**

1. **080 的"提交-完成握手被拖住"作废**。torch profiler（本机实测可用，CUPTI 出带时间戳的 `cat=="kernel"` 事件）给出的 device busy 是**区间并集**占比：b1（请求侧 3.4 s/页步）**86.8%**、b4（9.3-10.4 s/页步）**94.3%**，空隙只 13.2% / **5.7%**，单个 ingest 块内 95-98%。⇒ 慢态的时间**在 kernel 执行里**，不在 host 提交与完成通知之间。`capture.drain()` 第一次 `.cpu()` 的等待、`cudaMemcpyAsync` 平均阻塞 6.9 ms，都是被 kernel 顶回来的下游症状。⇒ **不需要**再升级取证手段去"看驱动帧"（`py-spy --native` / ETW / xperf 这条路线对本问题不再有诱惑力）。
2. **慢的对象被指名**：两个窗口的第一主导 kernel 都是 **`humming` WNA16 量化 GEMM**（`MmaOpClass`，逐层 gate/up `Shape<0,34816,5120>`、down `Shape<0,5120,17408>` 等），b1 里 gate/up 一族 Σ62.7 s = **device busy 的 61%**。
3. **⭐同一 `(kernel 名, grid, block)` 的调用存在 15-28× 双峰，且"态"= 慢模态调用的占比**。b1 的 gate/up 变体（grid `[70,1,1]`、block `[384,1,1]`，2048 次 = 32 前向 × 64 层）：快簇 1328 次中位 **5.15 ms**、慢簇 720 次中位 **77.67 ms = 15.1×**；**每次前向稳定 19-23 个慢调用，且慢的是固定的层位次**（跨 32 次前向同一位次 76-81 ms、其余 5.1-5.2 ms）。一次前向的账 = 23×78 ms + 41×5.15 ms ≈ **2.0 s**。b1→b4 同名 kernel 中位时长：down-proj **2.586 → 73.62 ms（28.5×）**、`<0,5120,25600>` 25×、`_causal_conv1d_fwd_kernel` 13.1×、`layer_norm_fwd_kernel` 11.1×、fused rms_norm 3.8×、`chunk_gated_delta_rule` 1.42×，而 `BatchPrefillWithPagedKVCache` **反而 0.57×**。
4. **连接器仍然无责（第三次独立复现）**：台账里连接器自有开销 `host_py + gpu_copy` 在 3.4 s 档与 10.6 秒档**逐字不变 = 0.165-0.172 s/页步**（079 与 080 各自量过同一事实），device 侧拷贝 31.4 GiB / 7,767 条的**总时长只有 0.47 s**（平均 61 μs）⇒ 079 判死的"修拷贝形态"继续成立，且现在有了 kernel 级旁证。`record` 绝对值 0.42-0.47 s 在两档也基本不变（NOSYNC=1）。
5. **窗口读数的地位=描述性仪表，不是判态器**（本步否证了一条刚提出的推论）：flashinfer autotune 窗（`[Autotuner] starts/ends` 之差）确实是一次 `mbt=1458` 的**无连接器/无投机/无长注意力**整页前向，080 的 11 支上与请求侧标签 10/11 同判；但 081 的 **b3 = 窗口 1.112 s（快簇）而请求 8.686 s/页步（慢簇）**，b1/b2 同样是"窗口快、请求 mid"。⇒ **判态必须按请求测**（必守 25 配套 2 不变），窗口只用于描述"那次纯前向快不快"。
6. **15 分钟内的第三个档位**：079 快档 1.3、080 慢档 9.8 之外，081 的 b1/b2/b5 稳定在 **3.4-3.7 s/页步**（单 boot 内三条独立读数一致：sweep 中位、主节拍、台账 `dt/步` 互差 ≤1%）⇒ 080 已有的"态会在 boot 内漂"要再软化为**"慢模态占比是时变量，boot 只是它的采样单位"**。
7. **槽位选取门控交付**（`VLLM_KVMEM_SLOT_PICK=time|score`，默认 `time` = §5.3/§12.16 的旧行为逐字不动）：旧行 `selected = sorted(top_pages)[:len(stage.slots)]` 的真实规则是"**分数前 TOPN 名里页号最小的 PAGES 个**"，丢掉的高分页恰是深位针页（079/080 两次复现"TOPN 64→96 反把针页挤出槽"）；`score` 档 = 先按分数取满槽、再把中选者按页号升序排版 ⇒ **模型看到的仍是时间序窗口**（§5.1 的"query 位置与选择无关"不变式不破），只换"谁进槽"。CPU 单测 63 项证两档 **槽数 55 / `(group, block_id)` 覆盖 110 项 / 880 = 55×2×8 层页拷贝逐项相等**（必守 16⑨ 的"半覆盖伪装全成功"没有引入），针页时间序第 53/64（余量 2）→ 分数序取满后第 48/55（**余量 7**）；`TOPN` 抬到 96 时 `time` 档丢针、`score` 档免疫。在线自证行：`vllm-030win patch (step 081): slot pick=score req=… top_pages=64 slots=55 selected=55 needle=page 72 (token 104858) in_selected=True dropped=[…]`。

**边界（勿误读）**

* **"名集合不同"不等于"tactic 非确定"**：`--compare` 确实报出"只在 b4 出现 tile-112 变体、只在 b1 出现 tile-16 + `cublasLt::splitKreduce` + cutlass bf16 路径"，但两个 profile 窗的**工作负载混合不同**（页步 / verify 步比例、M 分布），humming 本就按 M 分 tile 档 ⇒ 这条**只能记为不可判**。本步的决定性证据是第 3 条的**同 launch 配置双峰**，它不依赖名集合差。
* **profiler 可用但有代价边界**：本步量到的侵入度 ≤2%（窗前/中/后三发同形节拍，判据见步骤 081 第 4 条），但它的前提是 `with_stack=false` + `ignore_frontend=true`；打开栈回溯就不是这个数字。`profiler_out_<rank>.txt` 是固定名，**同 boot 第二个窗必须先改名**（`step081_boot.py prof` 已内置改名）。
* **humming 的 config 选择入口**是 `get_heuristics_config(...)`（`humming/tune/__init__.py:134`，未显式给 `tuning_config` 时走它），且 `humming/ops/gemm.py:27` 接受显式 `tuning_config` ⇒ **082 若要"钉住 GEMM 变体"是有口的**（仓内 `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:191` 已是先例）。但**本步没有任何证据说是选择错了**，第 3 条说的是"同一选择下每次调用差 15×"，两者不要混。
* **`SLOT_PICK=score` 的端到端 needle 判据已达成（b6 = GO）**：b6 是探针前零污染的净 boot，连续两次 serve 均 HIT；`recent_tokens=16384`、880 copies、read-back 0 mismatch。**不得**把多深度/多 nonce 网格统计当作已完成。
* **b5 的 MISS 是本步协议自伤，不是修法的失败**：探针前先跑了 5 发节拍请求，`VLLM_KVMEM_AUTHORITY_TRAJ=2` 只保两条轨迹的 raw-K 权威区 ⇒ 针轨迹的权威行被挤掉，serve 时 `baked 0 slot(s) … 880 layer-row(s) without authority row`。**教训一般化：视窗臂的探针之前不许插任何会入库的节拍请求**（一入库就占掉权威区的两条轨迹名额）。
* **两条仪器缺陷**：`step080_boot.py` 的 `kvtime_loaded` 恒 False（横幅在 health 之后 ~24 s 才打，字段在 health 时点就读日志）⇒ 080 的"门控自证"入表硬门实际没被执行过，靠人工兜住；080 的 triton 缓存取证目录错（臂用 `C:\fi\.triton`，不是 shell 的 `~/.triton`），结论方向不变。

**082 阶段 1 必验项（按顺序）**

① **同 launch 配置双峰的定罪**：候选排序 = 权重页驻留（16 GB 卡上 11.46 GiB 权重 + 3.17 GiB KV 池 + 0.53 GiB 草稿 + 5 GiB 锁页工作区 + raw-K 权威区 = 必然超订，WDDM 只能把东西推来推去）> 数据相关分支 > 电源/时钟（后者已被 080 的带外指纹否）。判据 = 同一 `(shape, M)` 在"驻留被人为改变"的两种条件下（例如把 KV 池 / 工作区调小一档，或去掉一个竞争者）的时长分布是否合并成单峰。② `score` 档的 in-situ needle（净 boot、探针前零污染、≥2 次采样）。③ 多深度网格（前置卡点已由 ⑦ 解开，余量 2 → 7）。④ **"他占 GPU 客户端"仍未否证**：`nvidia-smi --query-compute-apps` 看不见 WDDM 图形/拷贝客户端；本步想用 PDH `GPU Engine(pid…,engtype…)` 补，venv 有 pywin32 但 `EnumObjectItems` arity 未探明、`typeperf` 通配符报 `No valid counters`。⑤ `record` 段在 079 六支旧 boot 上的回溯（旧日志无该段）。⑥ recent 滚动重烘焙、多轮 ΔP、装配探针在 1456 上的回归、阶段 1 出口五项。**其中只有阶段 1 既定验收协议明确要求的项目属于当前必做；其余 KVMem 优化与产品化工作阶段 1 后暂停，须用户另行授权。阶段 1 出口达成后冻结 KVMem 线并回到原项目主线。**

**工具与证据（本步新增，全留存）**：`tools/kvmem_boot_window.py`（离线读 autotune 窗，含"必须配对读 out/err 两个流"的处理）、`tools/kvmem_trace_budget.py`（trace → busy 区间并集 / 空隙 / host-API 分桶 / 提交延迟 / 切块 / top kernel，`--compare` 出名集合差与同名时长比）、`tools/serve_gsq_kvmem_viewport081_spec.cmd`（079 臂逐字 + `S081_PROF`/`S081_PICK`，非注释行 diff 只 4 处）、`tools/step081_boot.py`（新判态协议 + `Sweeper` + `prof` 窗 + 侵入度夹逼）、`tools/apply_kvmem_slot_pick_step081.py` + `tools/kvmem_slot_pick_test.py`（63 项，apply/revert 往返实测）。取证：`step081_b{1..6}_arm.{out,err}.log`、`step081_b{1,2,3,4}_timebudget.json`、`step081_b{1,4}_tracebudget.json`、`kvmem_k15{a..f}/prof/b{1,4}_w1_rank0.*.pt.trace.json.gz`（b1 877,340 事件 / b4 593,426 事件）、`step081_b{1,4}_engine_fp.csv`、`step081_b{1..6}_window.json`、`kvmem_k15{e,f}/vp081_vp081{a,b}_d80*.json`、`prod029_logs/step081_boots.json`。**离线可复算的原始数字**：`_tmp_line_b/s081_gate_profiler.py`（前提闸门）、`s081_trace_deepdive.py`、`s081_gemmpat.py`、`s081_onecfg.py`（双峰与位次稳定性）。dump 目录用到 `kvmem_k15a…k15f`。

---

### 12.25 ⭐步骤 082：移植验收收口——双峰定罪「驻留因果支持」；PDH 取证打通；score 网格 4/4 HIT 但欠 2 格（2026-10-02 深夜，用户令断电收兵）

**路线性质重申**：本步是阶段 1 移植验收（082-A 排除/记录 ingest 性能风险、082-B 阶段 1 正确性网格、082-C 补充归因取证），不构成任何 KVMem 优化立项。

**082-A（双峰定罪）——三值结论 = `驻留因果支持`**：

- 仪器：`tools/kvmem_bimodal082.py`（锚定 081 同一 `(humming Shape<0,34816,5120>, grid [70,1,1], block [384,1,1])`；081 b1 已知读数逐字复现校验通过：快簇 5.149 ms×1328 / 慢簇 77.671 ms×720 / 比值 15.09 / 每前向 modal 23 min19 max23）；`tools/serve_gsq_kvmem_viewport082_spec.cmd`（= 081 臂逐字 + `S082_KVMB`/`S082_WSMB`，非注释 diff 4 处）；`tools/step082_boot.py`（= 081 协议逐字 + PDH 线程）。
- 条件（每条件降级预算 2 boot，正序 + 反序两轮交叉防时段漂移）：**A0** 基线（081 臂逐字默认）；**A1** `S082_KVMB=3000000000`（KV 池 3.4e9→3.0e9，**纯驻留 A/B**：页长 1456 不变、请求路径不变、AOT 键不变——b2/b7 `compiling_n=0`）；**A2** `S079_SPEC=0`（去草稿 0.53 GiB，机制隔离）；**A3** `S079_RAWK=0`（摘 capture/D2H 路径，机制隔离）。
- 结果（每 trace 锚配置 n=2432 = 38 前向 × 64 调用，`fallback=False` 全 8 支）：**A0 两支全双峰**（b1 mid 态：快簇 5.14 ms×1762 / 慢簇 99.93 ms×670 = 19.4×，慢占计数 27.6% / 时间 88.6%，每前向 modal 20、稳定度 0.816；b8 全慢态 9.90 s/页步：83.74 ms×850、34.9%/89.3%、modal 23、慢位次 6/12/14/16/18/20 全 1.0）；**六个缓解 boot 全部零慢调用**（单峰合并，max 5.2-14.8 ms；页步 1.16-1.46 s fast）。慢簇中位沿基线族连续（081 b1 77.67 → 081 b4 82.81 → 082 b8 83.74 → 082 b1 99.93 ms），快簇 5.14-5.60 ms 全条件不动。
- 归因链：交叉顺序排除时段漂移（b5-b7 于 22:15-22:40 全 fast，b8 于 22:45 落全慢态）；**A1 单独生效 ⇒ 约 0.4 GiB 余量就是悬崖宽度**；A2/A3 证实"任何降低 GPU 侧/链路压力的改动都消除慢模态"（机制隔离，不参与纯驻留归因）。**边界（勿误读）**：本结论说"慢模态由驻留压力因果驱动"，**不**说 humming"选错了 kernel"（081 的口径不变：同一选择下每次调用差 15×）；也不给出"生产应该缩池"的建议（缩池换余量 = 待授权项）。

**082-C（PDH 取证）——GO**：

- 工具 `tools/pdh_gpu_engine.py`。**081 四探针真败因更正**：`win32pdh.EnumObjectItems` 的 machine 参数必须传 `None`，081 传的空串被当成远程机器名报 buffer-size 错——不是本地化（本机对象名就是英文 `GPU Engine`）、不是 arity。**第二条坑**：`Utilization Percentage` 是速率计数器，同一查询句柄须 Collect ≥2 次才有真值 ⇒ 每 tick 重建查询恒 0，必须持久句柄（首版踩过，`eng` 恒空）。
- 实战读数：所有采样窗内**除引擎外无 GPU 客户端**（b8 慢态窗：引擎 3D 均值 88.9%/max 100%，其余 pid 均值 ≤0.01%/max 0.0%）⇒ 必守 26④ 的"他占客户端"候选由无效否证升级为**有效否证**。引擎 local 15.50-15.60 GiB + non_local 8.36-8.41 GiB、committed 总 24 GiB 对 16 GB 物理 = **结构性超订直接可见**；**边界**：non_local 含 8 GiB kv-offloading 主机池（WDDM 记为该进程 shared 面），判"权重页降级"要看动态增量；实测稳态窗 local 15.53 GiB 零瞬态下探 ⇒ **PDH 粒度看不到页故障级动态**，降页取证不能只靠 PDH 绝对值。

**082-B（score 网格）——INCOMPLETE（机制再证 GO、欠 2 格）**：

- 已完成：**b9 d0.40 双发双 HIT**（TTFT 98.6/242.1 s、指纹 `recent_tokens=16384`、finish=length、`baked 55×[6,7]=880 copies` 0 缺 authority、`read-back 110 checked 0 mismatch`）+ **b11 d0.70 双发双 HIT** ⇒ score 档累计 6/6（含 081 b6）。
- 欠账：**b10 d0.55 = 0/2**（慢态 boot 三段探针 ingest/flush/serve 各 ~198k token，~45 分钟超出驱动默认 2400s 超时被杀）与 **b12 d0.85 未跑**（断电收兵）。补测 = 净 boot（`--sweep 0 --no-main`）+ `--timeout 7200`，脚本 `_tmp_line_b/roundB2_082.py` 已备（b13/k16m 起）。**不得以 4 格外推召回率。**
- 仪器坑：`in_selected=-`（臂未设 `VLLM_KVMEM_SLOT_PICK_NEEDLE`，该诊断字段无效，不影响判定）；A3 臂 autotune 窗读数失真（window 1.76-1.85 "mid" 而请求侧全 fast）——必守 26①"窗不能当判态器"再添一证。

**阶段 1 出口清单状态**：identity canary、紧预算 needle、冷 262K transcript、多轨迹/串台、1456 页长装配回归 **未做**；082-A 有效结果 **已做**；082-B **欠 2 格**。出口达成后冻结 KVMem 线回原项目主线。

**证据**：`prod029_logs/step082_boots.json`（8 A-boot + B 台账）、`step082_b{1..8}_pdh.jsonl`、`step082_b{1..8}_arm.{out,err}.log`、`kvmem_k16{a..l}/`（dump + prof trace 8 支 + vp json 4 份）、`_tmp_line_b/round{1,2,B,B2}_082.py` + `roundB_082.log`、`bimodal_082_all.json`（8 trace 终表）。boot 号：b1-b8 = A 轮（k16a-k16h），b9-b12 = B 网格（k16i-k16l，b12 未完成）。生产默认未动；收兵时 kill 全部孤儿、独显回落 85 MiB；**断电未恢复生产，下次上电先跑 `tools/prod_watchdog.ps1`**。

---

## 11. 参考索引

| 资源 | 位置 |
|---|---|
| 论文 | `https://arxiv.org/html/2609.04852v1`（KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU） |
| QW3 引擎参考实现 | `github.com/kvmem/kvmem-qw3`（Apache-2.0） |
| llama.cpp 移植参考 | `github.com/kvmem/kvmem-llama.cpp`（Apache-2.0，默认分支 `master`，194 commits） |
| llama.cpp 参考的关键文档 | `docs/architecture.md`（GPU 池 = budget + gen_reserve、无 raw-K 无 re-RoPE 的产品路径）、`docs/milestones/v0.3.0.md`（identity canary + 紧预算 needle 对照 + 速度只记录）、`docs/recommended-config-performance.md`（16 GiB 配方与 262K 任务二实测）、`patches/0001-0003`（引擎侧三补丁） |
| 本地取证的 issue | #4（>210K 串台）、#13（closed，trim/abort）、**#43（open，replay 按 policy budget 判定）**、#34（prefill ∝ 窗口）、#55（448 MB staging）、#56（全量重发杀服）、#82/#22（Windows 多轮 agent 12-20 tok/s）、#11/#23/#73（Windows 自建脆弱） |
| 本仓相关既有结论 | 049 余量悬崖（`tools/prod_headroom_check.ps1`）、034/035/036/040/046 容量与池值、055 vis 臂余量悬崖、`v1/attention/backends/flex_attention.py`（mask 路径的 dtype 限制） |
