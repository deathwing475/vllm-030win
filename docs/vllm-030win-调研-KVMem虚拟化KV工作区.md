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

## 11. 参考索引

| 资源 | 位置 |
|---|---|
| 论文 | `https://arxiv.org/html/2609.04852v1`（KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU） |
| QW3 引擎参考实现 | `github.com/kvmem/kvmem-qw3`（Apache-2.0） |
| llama.cpp 移植参考 | `github.com/kvmem/kvmem-llama.cpp`（Apache-2.0，默认分支 `master`，194 commits） |
| llama.cpp 参考的关键文档 | `docs/architecture.md`（GPU 池 = budget + gen_reserve、无 raw-K 无 re-RoPE 的产品路径）、`docs/milestones/v0.3.0.md`（identity canary + 紧预算 needle 对照 + 速度只记录）、`docs/recommended-config-performance.md`（16 GiB 配方与 262K 任务二实测）、`patches/0001-0003`（引擎侧三补丁） |
| 本地取证的 issue | #4（>210K 串台）、#13（closed，trim/abort）、**#43（open，replay 按 policy budget 判定）**、#34（prefill ∝ 窗口）、#55（448 MB staging）、#56（全量重发杀服）、#82/#22（Windows 多轮 agent 12-20 tok/s）、#11/#23/#73（Windows 自建脆弱） |
| 本仓相关既有结论 | 049 余量悬崖（`tools/prod_headroom_check.ps1`）、034/035/036/040/046 容量与池值、055 vis 臂余量悬崖、`v1/attention/backends/flex_attention.py`（mask 路径的 dtype 限制） |
