# 调研 — KVMem 虚拟化 KV 工作区（vLLM 0.29 栈实现方案）

> **状态：设计定稿；阶段 1a 已开工——有界 prefill（057）+ copy-before-free 与准入守卫（060）已 GO，检索/重物化（K3）未开工。**
> 本文是这条新线的**唯一设计权威**：目标、决策记录、架构、接口落点、参数标定、阶段出口、验证台口径、风险登记册。
> 接手者先读本文，再读《交接提示词.md》顶部收档块。**§12.1-12.3 = 阶段 1a 有界 prefill（057）｜§12.5 = K1 copy-before-free（060）｜§12.6 = K2 准入守卫（060）｜§12.4 = 阶段 1a 剩余清单（已全部完成）。**
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
