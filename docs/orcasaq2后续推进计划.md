# KVMem 完整移植与 OrcaSAQ2 兼容计划

> 建立日期：2026-09-29
>
> **用户重新裁定的优先级：先把现有 GSQ 栈的 KVMem 完整移植做完，再做 OrcaSAQ2 的 NVFP4/MTP/DFlash2/KVMem 兼容。**
>
> 步骤 059 已证明 OrcaSAQ2 能在 Windows vLLM 0.29 上加载和生成；这只是模型兼容 smoke，不是 KVMem 完成的替代品。

## 1. 当前事实

### 1.1 已完成

- 现役 GSQ vLLM 0.29 栈：生产配置稳定，KVMem 默认关闭。
- KVMem 步骤 057：`VLLM_KVMEM_SW_WINDOW=N` 已完成“有界 prefill”半边。
- 057 已证明：冷 210K/256K prompt 可跑，窗内 needle 命中，窗外 needle 必失。
- **KVMem 步骤 060：K1 copy-before-free + K2 准入守卫 = GO**（见 §3 阶段 K1/K2 的完成记录）。滑出窗口的历史页现在会写进 host 工作区（键 `(轨迹, token 偏移)`），往返逐字节一致，零丢页零泄漏；上限不可被越过，视窗装不下在 boot 期响亮失败。
- **KVMem 步骤 061：K3 前半（raw-K 捕获 + Mean-K 索引 + 页级检索打分）= 机制 GO、R1 部分消解**（见 §3 阶段 K3）。检索有强信号（针页 logit +5.9/+9.2、rank 8→2 / 28→1），**粒度取 32 而不是 128**；**残余"早期页偏置"是新的头号挂账**；**重物化未做**。
- OrcaSAQ2 步骤 059：ExLlamaV3 1.5.3 native CUDA + OrcaSAQ2 plugin 已跑通，端口 8001 的 chat smoke 通过。
- **Orca 步骤 085（O2 = checkpoint 自带 MTP）：GO**。auto KV + eager + 0.88，与同参无投机控制臂 3 轮交替（六 boot 全 health、零 ERROR）。接受率 0.6272–0.6457、平均接受长度 2.25–2.29、逐位置 0.77/0.52；解码中位 45.5/45.9/45.5 vs 21.3/21.5/21.6 tok/s = **2.12x**；正确性 on 9/9 + off 9/9（chat 三连 + needle 三深度 8,169 token 全 HIT）。**代价账**：KV 池 19,636 → 12,444 tokens（−36.6%）、并发 1.64x → 1.04x、权重 +0.21 GiB、needle TTFT 反升约 0.9 s，且**单请求 16K 上下文在 16 GB 卡上起不来**（引擎自估最大长度 12,000；厂商 16GB MTP 预设的 0.95 档实测不可达，启动空闲仅 14.68/15.89 GiB），故两臂定在 `--max-model-len 12000 --gpu-memory-utilization 0.88`。随步绕开一个**通用缺陷**：`Qwen3_5MTP` 的 `embed_tokens`/`lm_head` 由 target 共享（载入后才绑定），而 orcasaq2 的 post-load 钩子对"没收到任何权重"的模块直接抛 ⇒ 任何 EXL3 checkpoint 的 MTP/EAGLE 都会 boot 死；修 = 进程级门控补丁 `tools/orcasaq2_sitecustomize/s085_empty_shared_patch.py`（默认关、GSQ 生产零暴露）。证据 `prod029_logs/orca_o2/`、台账 `prod029_logs/step085_boots.json`。
- **Orca 步骤 084（O1 = NVFP4 KV smoke）：GO（修复后）**。boot 一次成、NVFP4 写路径生效、KV 池 43,690 tokens（auto 21,845 的 2.0x）、attention block 2784 tokens/页（mamba 页约束推导，不能套 GSQ 1456）；chat identity 3/3 + needle 三深度 3/3 HIT（12,206 token）。途中定罪并泛化修复一个**引擎级默认布局 bug**：nvfp4 KV 专用 kernel 假设 head-major 页布局，默认 resolve 的 LBNHC 下页读回乱序（静默乱码零 ERROR）——GSQ 生产靠 launcher 显式 HND 恰好踩对；修复 = engine core 在 nvfp4 且用户未钉布局时优先 LBHNC。同 boot 窗口落地 KVMem 泛化六点（用户授权"顺便修"，为 O3/O4 铺路）。证据 `prod029_logs/orca_o1/`。

### 1.2 未完成的核心工程

**KVMem 的阶段 1 正确性主体（K3）只做完了读取侧前三件。** 当前缺少：

- ~~raw-K 捕获（RoPE 前，只对差量 token）；~~ **✅ 061**
- ~~sub-block Mean-K 索引；~~ **✅ 061（粒度取 32）**
- ~~page 级 softmax-over-pages 检索；~~ **✅ 061（有强信号）**
- **残余"早期页偏置"的成因**（无针时页 2/5/1/11/8/3 就占住 top 槽位；已排除范数效应）——**新的头号挂账**；
- fixed-slot rematerialization（**从 raw K 单次重建**，禁 delta re-RoPE）；
- identity canary、紧预算 retrieval needle、40/55/70/85% 多深度、串台检测、重物化往返单测；
- 工作区淘汰策略（LRU/GC/跨会话容量）与多轨迹并发。

因此，步骤 057 + 060 + 061 只能写成“阶段 1a 有界 prefill GO + K1/K2 GO + K3 读取侧前三件 GO”，**不能写成 KVMem 已完成**（**重物化未做 ⇒ 窗外 needle 仍答不出**）。

## 2. 总路线裁定

### 路线 0：现有 GSQ 栈完整 KVMem（当前主线）

先在已验证的 GSQ 目标模型和生产 0.29 引擎上完成 KVMem 阶段 1 出口。期间：

- 生产默认保持关闭；
- 变体 launcher 独立存在；
- 池值不因 KVMem 试验擅自缩小；
- 不叠 DFlash2、vision、Orca 或性能优化变量；
- 阶段 1 只判正确性和能力，不判性能。

### 路线 1：OrcaSAQ2 兼容（KVMem 阶段 1 出口之后）

Orca 后续按以下顺序：

1. Orca NVFP4 KV smoke；
2. Orca 自带 Qwen3.5 MTP；
3. Orca 专用 DFlash2 target adapter + 匹配 draft checkpoint；
4. Orca/KVMem 有界 prefill 和完整 workspace 兼容。

Orca 的步骤 059 smoke 只作为后续兼容基线，不能倒置主线优先级。

## 3. KVMem 完整移植计划（先做）

### 阶段 K1：copy-before-free（1a-2）

**✅ 已完成（步骤 060，GO）。** 落点与实现：

```text
v1/core/single_type_kv_cache_manager.py   _remove_blocks_in_range + _retain_for_workspace
v1/core/kv_cache_manager.py               take_workspace_evictions
v1/core/sched/scheduler.py                register_workspace_retained_blocks + block_state 字段
v1/core/sched/output.py                   KVConnectorBlockState.workspace_evictions
.../kv_connector/v1/base.py                register_workspace_retained_blocks (默认拒绝)
v1/kvmem_workspace/{config,groups,metadata,manager,worker}.py + kvmem_connector.py
```

在 `block_pool.free_blocks(freed)` 之前：

1. ✅ 判断块是否属于 KVMem workspace 轨迹（滑窗组 + `VLLM_KVMEM_WORKSPACE` 门控）；
2. ✅ 用 `(trajectory, token 偏移)` 而不是 block hash 建立工作区键；
3. ✅ 将页面提交到 host store（pinned，`swap_blocks_batch`）；
4. ✅ 把 pending job 绑定到页面生命周期（`torch.cuda.Event` 完成回传）；
5. ✅ 拷贝完成前禁止页面回池（ref_cnt 不递减，完成时恰好释放一次）；
6. ✅ copy 失败/工作区满必须显式可见（worker 抛错；满则逐页 WARNING + `pages_dropped` 计数，绝不静默）。

出口：滑出窗口的历史页可被重新加载，且不会出现 use-after-free、重复回池或静默丢页。

**实测**：冷 210K → 66 entries 淘汰 / 66 stored / 0 dropped；冷 258,854 → 94/122 slots / 0 dropped；往返自检 8 页 × 16 层 byte-identical；跨请求同轨迹；每 job "released N block(s)" 恰好一次；0 次 full/stray/unregistered 告警。

### 阶段 K2：准入守卫

**✅ 已完成（步骤 060）。** 两个上限：

- workspace 上限 262,144 token：`prompt > 上限` → **HTTP 400**（既有 `_validate_prompt_len`）；
- `prompt + max_tokens`：**上游把 `max_tokens` 夹到 `上限 − prompt`**（实测 `258,854 + 3,290 = 262,144` 恰好等于 `max_model_len`）⇒ 工作区**不可能被越过**；另加 KVMem 守卫兜底（`input_processor.py`，对绕过夹紧的路径生效）；
- execution viewport 上限：`KVMemWorkspaceScheduler.bind_gpu_block_pool` 在 boot 期按 `max_admission_blocks_per_request` 求和校验，不足则**明确 RuntimeError 拒绝启动**（实测 `needs 234 / pool has 259`）——不再出现 `scheduler_reserve_full_isl=True` 把请求静默留在等待队列。

### 阶段 K3：阶段 1 正确性主体（**前半已完成（步骤 061）；后半未开工**）

按既定固定槽位布局：

```text
[sink S | retrieval N | recent R | query q | generation reserve g]
```

**已完成（步骤 061，读取侧前三件，机制 GO、R1 部分消解）**：

1. ✅ **raw-K 在 RoPE 前捕获**（`VLLM_KVMEM_RAWK` 门控；`qwen3_next.py` eager 路径 `k_norm` 之后、`rotary_emb` 之前 clone；只 16 层 `full_attention`、只对差量 token；**必须关掉融合 kernel** 才抓得到中间量 ⇒ 臂内自洽、禁跨臂比 PPL）；
2. ✅ **Mean-K 子块索引**（按最细粒度存储、粗粒度由求和精确导出 ⇒ 一次 prefill 出全部变体）。**粒度取值改为 32 而不是原定的 128**：实测 `dot@32` 在窗口外针上 rank 2，而 `dot@128` 只有 rank 7（趋势 32 > 64 > 128），**与设计 §5.3 的 128 初值不同**（代价 = 索引 537 MB/轨迹 host 常驻）；
3. ✅ **page 级 softmax-over-pages 检索**（页内子块取 max、只对 query span 打分、sink/recent mask 成 −inf）。**实测检索有强信号**：针只占一页 1424 token 里的 15 个，却把该页 logit 抬高 **+5.91（窗口外，rank 8→2）/ +9.19（窗内，rank 28→1）**，6 个变体全部进 top-16。**`cosine` 与 `dot` 同排名 ⇒ 归一化不是杠杆**。
   **新挂账（头号）**：残余"早期页偏置"——无针时页 2/5/1/11/8/3 就有 22.7–26.2 的 logit，占住检索槽位；**已排除"页均值范数"**（范数恒定 ±7% 而 logit 跨 3×）。**不查清它，第 4 项做完也拿不到有效检索槽位。**
   **两个测量层缺陷已修**：页归属走错边界（`1424 = 16×89`，32/64/128 都不整除 ⇒ 修前"rank 74 阴性"是错的）＋ AOT fullgraph 编译拒绝捕获（⇒ 本臂改 `--enforce-eager`，即设计 §6 的"阶段 1 无图模式"）。

**未开工（K3 后半）**：

> **✅ 步骤 063 已开工并完成其中第 5 与第 12 项的算法核心**：新增 `vllm/v1/kvmem_workspace/remat.py`（重物化原语）+ `tools/kvmem_remat_test.py`（**离线 18/18，逐位**）。**锚全部取自引擎自己的代码**（逐行移植的 `_triton_mrope_forward` / `write_reference_nvfp4_cache` / `side_carve_views`）：**位移 d=4096 后搬回原位逐位一致、连续 8 次位移后搬回仍逐位一致**、**以 pre-RoPE 旋转前缀为唯一输入重建整页 == 引擎整页逐位一致**。**新查出的结构前提 = `rotary_dim 64 = 4 × 16` 恰好落在 NVFP4 尺度组边界上** ⇒ 只重写旋转前缀（**全页 12.5%** = 205,056 / 1,640,448 B），其余 192 维与整个 V 一字不动。**该模块零调用者 ⇒ 无门控、引擎行为逐字节不变**。**剩下 = 接线**（建议顺序：①**先做**存储格式 + raw-K 权威区并在工作区自检里加"重烘焙往返"——**不改变注意力行为**；②**再做**位置解耦 + 块表改写）。细节见设计档 §12.9。

4. **固定槽位布局，query 位置不随选页变化**——位置解耦 + 重 RoPE + 块表改写；**已知唯一缝 = `gpu_model_runner.py` 的 `positions`**（每步从头重建、`slot_mapping` 由它派生 ⇒ 必须"先改位置、后算 slot"，且无任何现成 per-request 钩子），而 **`update_block_table` 只在 flash_attn/mamba 后端实现、flashinfer（nvfp4 路径）没有** ⇒ 确实要动模型执行器（本线最重工程）；
5. ✅ **（063）** 每次从 raw K 单次重建，禁止 delta re-RoPE —— **原语已落地并单测钉死（8 次位移后搬回逐位一致）**；
6. V 与非旋转 K 按整页搬运；
7. identity canary：预算不收紧时与 KVMem-off 逐 token 一致；
8. 紧预算 needle：recency-only 必须失败，retrieval 必须成功；
9. 40/55/70/85% 多深度 needle；
10. 冷 262K prompt；
11. 固定提问顺序重复，串台检测；
12. ✅ **（063）** rematerialization 往返单测，记录量化步内误差 —— **实测逐位一致（0 字节差），量化误差 2.734e-02 vs 最大 E2M1 步长 7.813e-02（比值 0.35）**。

### 阶段 K4：性能与生产特性（K1-K3 出口之后）

依次评估：

- GPU Mean-K 打分；
- 批量 RoPE/rematerialization；
- CUDA graph 捕获；
- DFlash2 投机；
- FULL_AND_PIECEWISE；
- 96K 压力配置；
- 缩池换余量。

任何一项都必须作为独立变体，不能把阶段 1 正确性结果和性能结果混写。

## 4. Orca 兼容计划（KVMem 阶段 1 出口之后）

### 阶段 O1：Orca NVFP4 KV

仅改变：

```text
--kv-cache-dtype nvfp4
```

保持 eager、TP1、单序列、16K、GPU utilization 0.88。重新记录 Orca 的 attention block、GDN page、统一 page 和 KV tokens；不能套用 GSQ 的 1456-token 页账。

验收：boot、FlashInfer FA2 NVFP4、chat、identity/needle。性能只记录，不作第一门。

### 阶段 O2：Orca 自带 MTP

**✅ 已完成（步骤 085，GO）。** 计划原文用 `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`；实测口径 = `method` 无论写 `qwen3_next_mtp` 还是 `qwen3_5_mtp`，都会被 `config/speculative.py` 判为 deprecated 并 alias 成 `mtp`，**draft 架构实际由 `hf_config_override` 从 target config 推出**（`model_type=qwen3_5` + `mtp_num_hidden_layers=1` → `qwen3_5_mtp` / `Qwen3_5MTP`），故两臂 launcher 直接写 `--speculative-config.method mtp --speculative-config.num_speculative_tokens 2`。

先在 auto/bf16 KV、eager 下验证；和 MTP-off 做 3 轮交替，记录接受率、正确性、KV 容量和显存。**本卡实测约束（原计划未预见）**：带投机时单请求 16,384 需 1.71 GiB KV 而 0.88 只剩 1.44 GiB，引擎自估最大长度 12,000 ⇒ 两臂必须同退 `--max-model-len 12000`；`--gpu-memory-utilization 0.95`（厂商 16GB MTP 预设）在本卡不可达（启动空闲仅 14.68/15.89 GiB）。结果与代价账见 §1.1 与步骤 085。

### 阶段 O3：Orca 专用 DFlash2

用户明确要求 DFlash2，因此正式纳入，但必须先过资产和接口门：

1. 盘点 `qwen3_dflash2.py` 对 target config、`dflash_config`、GDN state、full-attention、mask token、grouped-conv 的假设；
2. 找到或生成 Orca-compatible DFlash2 draft checkpoint；
3. 为 `Qwen3_5ForCausalLM` 写独立 target adapter；
4. 在 auto KV、eager、无 MTP 下做单步 hidden/logits 对齐；
5. 再做完整 speculative service、接受率、needle 和长稳。

**禁止直接复用 GSQ 的 `dflash2\\gptq3c`。**

### 阶段 O4：Orca + KVMem

只有 K1-K3 完成且 O1 通过后，才把 KVMem 接到 Orca：

- 先复用 `VLLM_KVMEM_SW_WINDOW` 做机制 smoke；
- Orca 单独重新核算 page/viewport；
- 再复用 K1-K3 的 workspace store/retrieval/rematerialization；
- 不把 GSQ 的 1456 页常量和 Orca 的 784 attention block 混用。

## 5. 共同纪律

- 所有实验使用独立 launcher/端口；生产 GSQ 8080 不作为实验入口。
- KVMem K1-K3 完成前，不把 Orca 后续能力列为当前主线。
- 所有 venv 变化必须有 revert；优先进程级 sitecustomize，不改生产 venv 源码。
- 每次换 KV/spec/KVMem 变量都新建变体；至少 3 boot 交替；首编不判性能。
- kill 后等待独显 `memory.used < 800 MiB` 再起下一臂。
- 任何阶段都不能把“能加载模型”写成“完整 KVMem 已完成”。

## 6. 当前结论

- KVMem 阶段 1a 有界 prefill：GO（057）；**K1 copy-before-free：GO（060）**；**K2 准入守卫：GO（060）**；完整 workspace 的**读取侧（检索/重物化）未开工**。
- 当前真正下一步：**KVMem K3 —— raw-K 捕获 → Mean-K 索引 → softmax-over-pages 检索 → 固定槽位重物化**。
- Orca native CUDA + auto KV + eager chat：GO，仅作为兼容基线。
- **Orca NVFP4：GO（084，修复后）**——默认布局 bug 已在 engine core 泛化修复（nvfp4 未钉布局 → LBHNC 优先），O1 launcher = `tools/serve_orcasaq2_029_nvfp4.cmd`；性能只记录（12.2K TTFT 7.15s）。
- **Orca MTP：GO（085）**——checkpoint 自带 1 层 MTP 头可用，接受率 0.627–0.646、平均接受长度 2.25–2.29、解码 **2.12x**（45.5 vs 21.5 tok/s），正确性 on/off 各 9/9。代价 = KV 池 −36.6%、并发 1.64x→1.04x、权重 +0.21 GiB，且 **16 GB 卡上带投机起不了 16K**（引擎自估上界 12,000），两臂同用 12,000/0.88。下一头名 = O3。
- Orca DFlash2：正式纳入，但需要 Orca adapter 和匹配 draft。
- Orca KVMem：顺延到 GSQ K3 和 Orca O1 之后。
