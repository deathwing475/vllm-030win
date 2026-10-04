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
- **Orca 步骤 090（速度档）＝ 正常速度达成：FULL 图 + 手压 KV 池 ⇒ Orca DFlash2 的 8k anchor 中位 79.91-86.63 tok/s（s5/s6/s7 三支，散布 ±4%），而自动池档只有 4.83/5.49（池 43,480 tokens，mem-util 2% / 80 W = 驻留悬崖）⇒ `--kv-cache-memory-bytes 8e8` 就是本卡速度开关（必守 33）**。DFlash2 与 FULL 图兼容达成、needle 每支 3/3、零 ERROR；配方 `S089_NOEAGER=1 S089_CGS=3 S089_GRAPHPROF=0 S089_POOL=800000000` @ L=16,384。
- **Orca 步骤 089（更正 088）：容量问题解决 ⇒ O3 收档条件齐备**。088 的"本卡容量 NO-GO"是**臂配置结论**（臂没带 GSQ 生产的容量杠杆），不是卡上限；照抄生产的 `VLLM_KV_GROUP_SIZE=8` + `--mamba-ssm-cache-dtype bfloat16`（**block 2832→1456**）+ `--mamba-cache-mode align` + `--enable-prefix-caching` + `--kv-offloading-backend native --kv-offloading-size 8`（KV 卸载到主机 mmap）+ util 0.922 之后，**同一支 `gptq3c` 草稿在 Orca 上：L=16,384 → 池 41,275 / 并发 2.52x；L=131,072 + 池 3.4e9 → 池 157,910 / 并发 1.20x**，两支零 ERROR、needle 三深度各 3/3 HIT、接受率 0.4935 / 0.4811（310 / 265 drafts）。088 摆出的"换 >16 GB 卡 / 减权重 / 改草稿"三条候选**撤销**。纪律 = **必守 32**。臂 = `tools/serve_orcasaq2_029_nvfp4_dflash2_prodcap.cmd`；台账 `prod029_logs/step089_boots.json`。欠账 = 速度档（≥3 boot + 分型）、长稳、与 086 MTP 档同参对比。
- **Orca 步骤 088（O3 第 4-5 项：DFlash2 接 Orca，路线①复用 `gptq3c`）＝ 第 4 项 GO、~~本卡容量 NO-GO~~（**已被 089 作废 = 臂配置结论**）、第 5 项 INCOMPLETE（其 needle / ≳10K 两条门已由 089 跑出 3/3 HIT）**。七支 boot：自证全在（aux taps (6,20,34,48,62) / `Using V2 Model Runner` / `DFlash2DraftModel` / 空共享补丁触发 4 次 ⇒ **087 预测命中，085 补丁够用**）；**对齐 GO** = prose 接受率 **0.5448**（524 drafts、接受长度 2.09、34.99 tok/s）、countdown **0.9932**（444 drafts、2.99、49.2-49.5 tok/s）⇒ **GSQ 标定外推到 Orca 不崩塌**；**容量 = 每请求地板** = auto/12,000 需 1.66 > 0.82（引擎自估最大长度 **1,600**）、nvfp4/16,384 需 1.03 > **0.27**、**降 N=1 不救**、降 L=4,096 仍要 0.77 ⇒ **198 KiB/token**、权重 11.46 → **12.55 GiB**（草稿文件只 0.545）；**唯一能起 = util 0.92 + L=4,096**（池 4,818 / 并发 1.18x / 零 ERROR）⇒ 牺牲长上下文，**不可交付**。第 5 项（needle / ≳10K / 长稳）在该档**不可执行** ⇒ INCOMPLETE。**O3 收口 = 用户三选一（a 收档回 086 的 MTP 1 步 / b 换 >16 GB 卡或减权重 / c 先定罪地板构成）**。新增**必守 31**（探针 `hit` 恒 False、`build_prompt` 无视 `--tokens`、读数一致≠样本充分）。
- **Orca 步骤 087（O3 前三项 = 盘点 + 资产门 + 接口门，零显存）：前三项完成，第 4-5 项 INCOMPLETE**。三条硬结论：①**"给 `Qwen3_5ForCausalLM` 写独立 target adapter"这条前提被实测推翻**——`supports_eagle3=True`、`set_aux_hidden_state_layers` 可用、tap 捕获本体在 `Qwen3NextModel`（GSQ 生产用的同一套），唯一兼容守卫 `fc` 宽 **25600 = 5 × 5120 对 Orca PASS**；②**真门 = 资产**：全盘只有 GSQ 的 `dflash2\` 一个草稿族、**Orca 目录零 draft**，而草稿本体是按 **base Qwen3.8-27B**（`z-lab/Qwen3.8-27B-DFlash2`，1.9B）训练的、GSQ 特异性只来自重量化标定那一层；bf16 源 3.58 GiB 在 16 GB 卡（Orca 权重 11.94 GiB）**装不下** ⇒ 必须量化草稿，三条路线待用户裁定；③**预测 boot 期会撞 085 那个"空共享模块"错**（建草稿时 EXL3 恰认领 `embed_tokens`/`lm_head` 两个零权重模块），缓解件 = 已存在的 `ORCA_EXL3_ALLOW_EMPTY_SHARED=1`。另：DFlash2 **只能走 V2 model runner**（V1 明确拒绝），而 085/086 的 Orca 臂已在 V2 上跑过 ⇒ 非新风险。工具 `tools/step087_o3_meta_probe.py` + 证据 `prod029_logs/step087_o3_meta_probe.txt`。
- **Orca 步骤 086（O2 在 nvfp4 KV 上重做，用户指定档）：GO（限 `num_speculative_tokens=1`）；`≥2` 判 NEGATIVE**。双臂同参 nvfp4 / 16,384 / 0.88 / eager：spec=1 = block 2816、KV 池 **20,753**、并发 **1.27x**、权重 11.94 GiB、接受率 **0.7938/0.7816/0.7774**、平均接受长度 1.78–1.79、解码中位 **39.18/37.93/38.13 vs 控制臂 21.57/21.38/21.38 = 1.78x**、正确性两臂各 **15/15**（needle 12,208 token 三深度全 HIT）、needle TTFT 仅 +0.12 s；控制臂 = block 2784、池 **35,498**、2.17x、11.72 GiB ⇒ **投机代价（nvfp4 档）= KV 池 −41.5%、并发 2.17x→1.27x、权重 +0.22 GiB**。**停摆定罪**：spec=2 臂 chat 正常且投机在跑（引擎自报 Mean acceptance length 2.14），但 ≳10K 的长 prompt 90 s+ 不出首 token（`Running: 0 / Waiting: 1 / KV usage 0.0%`、GPU 钉 100%/333 W；py-spy 三次给出不同帧 ⇒ 慢而非死锁）；同参无投机控制臂 10,704 ✓ 6.2 s、14,241 ✓ 3.95 s；同臂降成 1 步立刻 6.51 s / 5.58 s 通过 ⇒ **触发条件 = nvfp4 KV × 多步 draft × 长 prompt**，机制未定罪（挂账，再查须授权）。另：nvfp4 页长 2784（投机后 2832/2816）⇒ `mbt` 必须抬过页长（必守 11），但抬到 3072 会把 KV 预算吃光（16K 需 0.93 > 可用 0.90，引擎自估 14,160）⇒ **2848 = 刚好压过页长**是本卡 nvfp4+投机的可用组合。跨档答复：**要上下文选 nvfp4+1 步（16K 保住、1.78x）；要极致 decode 比值才用 auto+2 步（2.12x 但必须砍到 12K）**。交付 = `serve_orcasaq2_029_nvfp4_mtp.cmd`（默认 `S086_SPEC=1`）+ `serve_orcasaq2_029_nvfp4_nomtp.cmd` + `step085_orca_boot.py` 扩 `on4/off4`。
- **Orca 步骤 085（O2 = checkpoint 自带 MTP，auto/bf16 KV 档）：GO**。auto KV + eager + 0.88，与同参无投机控制臂 3 轮交替（六 boot 全 health、零 ERROR）。接受率 0.6272–0.6457、平均接受长度 2.25–2.29、逐位置 0.77/0.52；解码中位 45.5/45.9/45.5 vs 21.3/21.5/21.6 tok/s = **2.12x**；正确性 on 9/9 + off 9/9（chat 三连 + needle 三深度 8,169 token 全 HIT）。**代价账**：KV 池 19,636 → 12,444 tokens（−36.6%）、并发 1.64x → 1.04x、权重 +0.21 GiB、needle TTFT 反升约 0.9 s，且**单请求 16K 上下文在 16 GB 卡上起不来**（引擎自估最大长度 12,000；厂商 16GB MTP 预设的 0.95 档实测不可达，启动空闲仅 14.68/15.89 GiB），故两臂定在 `--max-model-len 12000 --gpu-memory-utilization 0.88`。随步绕开一个**通用缺陷**：`Qwen3_5MTP` 的 `embed_tokens`/`lm_head` 由 target 共享（载入后才绑定），而 orcasaq2 的 post-load 钩子对"没收到任何权重"的模块直接抛 ⇒ 任何 EXL3 checkpoint 的 MTP/EAGLE 都会 boot 死；修 = 进程级门控补丁 `tools/orcasaq2_sitecustomize/s085_empty_shared_patch.py`（默认关、GSQ 生产零暴露）。证据 `prod029_logs/orca_o2/`、台账 `prod029_logs/step085_boots.json`。
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

**补测（步骤 086，用户指定 nvfp4 KV）**：同一判据在 `--kv-cache-dtype nvfp4` 上重做 = **GO，但只限 `num_speculative_tokens=1`**。`≥2` 步在 nvfp4 上对 ≳10K 的长 prompt 停摆（chat 正常、投机在跑，但长请求 90 s+ 不出首 token；同参无投机臂 6.2 s 通过、同臂降成 1 步 6.51 s 通过）。nvfp4+spec1 的账 = 16K 上下文保住、接受率 0.777–0.794、解码 **1.78x**、KV 池 **−41.5%**、并发 1.27x、正确性 15/15。**投机型验收从此必须包含 ≳10K 的长请求，不能只跑 chat。**

### 阶段 O3：Orca 专用 DFlash2

用户明确要求 DFlash2，因此正式纳入，但必须先过资产和接口门：

1. 盘点 `qwen3_dflash2.py` 对 target config、`dflash_config`、GDN state、full-attention、mask token、grouped-conv 的假设；
2. 找到或生成 Orca-compatible DFlash2 draft checkpoint；
3. 为 `Qwen3_5ForCausalLM` 写独立 target adapter；
4. 在 auto KV、eager、无 MTP 下做单步 hidden/logits 对齐；
5. 再做完整 speculative service、接受率、needle 和长稳。

**禁止直接复用 GSQ 的 `dflash2\\gptq3c`。** —— **✅ 用户 2026-10-03 裁定解除：O3/088 走路线①，复用 `dflash2\gptq3c`（0.545 GiB）**。口径：**解除仅限 O3/088 的 Orca DFlash2 臂**，不代表其它场景可默认复用；已知代价 = 其 GPTQ Hessian 在 GSQ 栈上采 ⇒ 对 Orca 属外推，**判据 = 实测接受率**（GSQ 带 62.82%；参照量级 = 051 只改码本位宽就动了 −0.91pp），**明显掉出该带即回到路线②/③**。

**✅ 步骤 087 已完成第 1-3 项（零显存；工具 `tools/step087_o3_meta_probe.py`，证据 `prod029_logs/step087_o3_meta_probe.txt`）**：

- **第 1 项（盘点）**：12 条假设带 file:line 落在步骤文档。**三条是装饰品**——`dflash_config.block_size: 8` 在 dflash 路径不被读（`speculative.py` 只在 `method=="dspark"` 碰它；草稿块长实为 `1 + num_speculative_tokens`，`qwen3_dflash2.py:268`）、`num_target_layers: 64` 全树唯一消费者是 DSpark、`use_sliding_window`/`max_window_layers` 不读。**"对 GDN state / full-attention 的假设"这一条 = 空集**（草稿 config 无 mamba/linear_attention 字段，5 层纯 Qwen3 滑窗注意力；投机与 GDN 的关系只落在 target 侧页几何 = 必守 28②）。真被读的：`target_layer_ids`（+1 ⇒ (6,20,34,48,62)）、`is_causal=false`、5×`sliding_attention`+`sliding_window 2048`、`mask_token_id 248070`（走共享 target `embed_tokens`，两模型 `vocab.json` 逐字节相同）、`conv_kernel/group`（要求 `hidden % group == 0`）、`selector_rank/top_k`、以及唯一的兼容守卫 `fc` 宽度。
- **第 3 项（target adapter）前提被推翻 ⇒ 无需实现**：`Qwen3_5ForCausalLM` `supports_eagle3=True`、`set_aux_hidden_state_layers` 可用、tap 捕获本体在 `Qwen3NextModel`（`qwen3_next.py:789, 821-853`），GSQ 生产用的就是同一套。兼容守卫对 Orca 实算 **PASS**：`_get_dflash_fc_input_size = 25600 = 5 × 5120`、`248070 < 248320`、`max(aux)=62 < 64`。runner：`use_v2_model_runner=True` 且 **V1 明确拒绝 dflash2**（`_get_v1_model_runner_unsupported_features() = ['dflash2 drafts']`）⇒ 只能走 V2，而 085/086 的 Orca 臂横幅已是 `Using V2 Model Runner` ⇒ **EXL3 × V2 已被证可用**。`max_num_new_slots_for_drafting = 2` ⇒ 必守 11/28 的 `mbt ≥ 页长 + num_spec` 原样适用。
- **第 2 项 = 当前真门（✅ 用户 2026-10-03 已裁定 = 路线①）**：全盘只有 `Qwen3.8-27B-3Bit-GSQ\dflash2\` 一个草稿族，**`qwen3.8exl3` 里零 draft**。溯源 = GGUF 元数据 `general.source.url = huggingface.co/z-lab/Qwen3.8-27B-DFlash2`（1.9B）⇒ **草稿权重是按 base Qwen3.8-27B 训练的，不是按 GSQ 训练的**；GSQ 特异性只来自"重量化标定"这一层（gptq* 的 GPTQ Hessian 在 GSQ 栈上用零重叠标定集采，见《DFlash2-混合精度量化-交接文档v4.md:148-157》）。体量实测 = bf16 源 **3.58 GiB** / ct4bit **1.19 GiB** / gptq3c **0.545 GiB**；Orca 权重 11.94 GiB ⇒ **bf16 草稿在 16 GB 卡上装不下，任何 Orca DFlash2 臂只能用量化草稿**。三条候选：

| 路线 | 成本 | 与"禁止复用 gptq3c"的关系 |
|---|---|---|
| ①复用 `gptq3c`（0.545 GiB，GSQ 接受率 62.82% 已知） | 零准备，直接 boot | **✅ 用户 2026-10-03 裁定采用（仅限 O3/088）**；已知代价 = Hessian 属 GSQ 标定、对 Orca 分布是外推 ⇒ 判据 = 实测接受率，明显掉出 62.82% 带即回②/③ |
| ②从 bf16 源做 Orca 线 **RTN 无标定**重量化（`quant_dflash2.py --rtn --bits 3/4 --group 128`，产物 int4 ≈ 1.19 GiB / int3 ≈ 0.8 GiB） | CPU 峰值内存 ~8-10 GB（主机 23.1 GiB + 生产占 8 GiB offload mmap ⇒ 须停生产或夜间跑） | **符合**（新 artefact、独立目录） |
| ③**用 Orca 自采 Hessian 做 GPTQ** | 一次 Orca boot + capture + 重量化 + ≥3 boot 验收 | **最贴禁令本意**，成本最高 |

- **预测的 boot 期阻塞（未定罪，留给 088）**：建草稿时 `Exl3Config.get_quant_method` 恰被叫 2 次 = `model.embed_tokens → Exl3EmbeddingMethod`、`lm_head → Exl3LinearMethod`（orcasaq2 全局 patch 了 `VocabParallelEmbedding.__init__`），而草稿这两个模块零权重、靠共享 ⇒ 与 085 的 `embed_tokens: neither an int8 table nor a dense one loaded` **同形**；缓解件已在仓内（`ORCA_EXL3_ALLOW_EMPTY_SHARED=1`，086 投机臂默认开）。
- **第 4-5 项 = INCOMPLETE**（要 boot；不许拿"接口门通过"替代）。**O3 也是投机 ⇒ 仍按红线必守 28 定 `mbt` 与 `max-model-len`，验收必须含 ≳10K 长请求。**

**✅ 步骤 088 已执行第 4-5 项（2026-10-03 深夜，7 支 boot；路线① = 复用 `gptq3c`）＝ 第 4 项 GO、"可用投机"本卡容量 NO-GO、第 5 项 INCOMPLETE**：

- **机制与对齐 = GO**：七支 boot（含被判废的四支）自证一字不差 —— `Using Eagle3 auxiliary layers from config: 6, 20, 34, 48, 62` + `Using V2 Model Runner` + `Resolved architecture: DFlash2DraftModel` + `empty_shared_tolerated=4`（**087 的预测命中：DFlash2 也撞 EXL3"空共享模块"，085 补丁够用**）。接受率（同一支 b7，temp=0）：**countdown 0.9932（444 drafts、接受长度 2.9865、逐位置 1.000/0.9865、49.2-49.5 tok/s）**、**prose 0.5448（524 drafts、2.0897、0.672/0.418、34.99 tok/s）** ⇒ **GSQ 标定的 Hessian 外推到 Orca 不崩塌**，路线①的赌注成立。
- **容量 = 本卡 NO-GO，且是每请求地板项**：auto KV/12,000/0.88 需 1.66 GiB > 可用 0.82（**引擎自估最大长度 1,600**）；nvfp4/16,384/0.88 需 1.03 > 可用 **0.27**；**降到 `num_speculative_tokens=1` 可用仍是 0.28（降步数不救）**；降到 L=4,096 仍要 0.77 GiB ⇒ **198 KiB/token**（086 的 MTP+nvfp4 档 ~50 KiB/token）；权重 无投机 11.46 → MTP 11.67 → **DFlash2 12.55 GiB**（草稿文件只有 0.545 ⇒ 引擎侧 ~1.09 GiB）。**唯一能起的形态 = `util 0.92` + L=4,096**（KV 池 4,818、并发 1.18x、零 ERROR），代价是丢掉长上下文（正是 Orca 线的目标）且吃光 049 余量悬崖 ⇒ **不构成可交付**。
- **第 5 项（needle 三深度 / ≳10K 长请求 / 长稳）= INCOMPLETE 且本档无法执行**：`build_prompt` 无视 `--tokens`（实发 ≥4001）⇒ needle 三发全 `HTTP 400`；L=4,096 < 10K ⇒ 必守 28⑤ 那条门跑不了。正确性侧 chat 三发文本全对（`42` / `The quick brown fox.` / `77349`），但探针 `hit` 字段对三发统一问"含 needle 码吗"⇒ **`hit=False` 是口径不是回归**（三条口径已升**必守 31**）。
- **⚠️ 步骤 089 更正（2026-10-04）= 上面 088 的"本卡容量 NO-GO"作废**：088 那三支臂**没带 GSQ 生产的容量杠杆**，所以量到的"198 KiB/token 地板"是臂配置而非卡上限。新臂 `tools/serve_orcasaq2_029_nvfp4_dflash2_prodcap.cmd` 逐条照抄 `tools/serve_gsq_prod029_n2.cmd`（同卡、同 `gptq3c` 草稿、生产 L=163,072）：`VLLM_KV_GROUP_SIZE=8`〔036〕+ `--mamba-ssm-cache-dtype bfloat16`〔045/046 → **block 2832→1456**〕+ `--mamba-cache-mode align` + `--enable-prefix-caching` + `--kv-offloading-backend native --kv-offloading-size 8`（**KV 卸载到主机 mmap，不是权重卸载**）+ util **0.922**。读数：**c1 L=16,384 → 池 41,275 / 并发 2.52x / 零 ERROR / needle 3/3 HIT / 接受率 0.4935（310 drafts）**；**c2 L=131,072 + `--kv-cache-memory-bytes 3400000000` → 池 157,910 / 并发 1.20x / 零 ERROR / needle 3/3 HIT / 接受率 0.4811（265 drafts）**。⇒ **O3 第 4、5 项的容量与 needle 判据都达成，"换 >16 GB 卡 / 减权重 / 改草稿"三条候选撤销**；纪律入**必守 32**（容量结论前必须逐条 diff 同卡跑通的生产 launcher，并限定结论范围）。**欠账**：速度档（≥3 boot + `power.draw`/`utilization.memory` 分型；c1 34.81 vs c2 3.63 tok/s 属两支不同 boot 状态，不进结论）、长稳/多轮、与 086 MTP 档的同参对比。
- **✅ 步骤 090（2026-10-04）= 速度问题解决，Orca DFlash2 首次落到正常速度带**：按用户顺序"必须 DFlash2 → 先 FULL 图 → 再降 KV 池直到正常速度"。**FULL 图兼容 = 达成**（`FULL_AND_PIECEWISE` + `--cudagraph-capture-sizes 3`，s2-s7 零 ERROR、needle 每支 3/3）；**8k anchor 中位 = 手压池档 s5/s6/s7 = 86.63 / 86.01 / 79.91**（散布 ±4%），而**自动池档 s3/s4 只有 4.83 / 5.49**（池被撑到 43,480 tokens ≈1.77 GiB；带外分型 = util 100% 而 mem-util 2% / 80 W ⇒ 081/082 的等待型慢模态）。⇒ **`--kv-cache-memory-bytes 800000000`（池 18,589）就是本卡的速度开关**，代价是并发 2.65x→1.13x（`--max-num-seqs 1` 下无感，多并发不得照抄）；s1 判废 = 默认 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 预留 ~1.3 GiB 捕获峰值。纪律 = **必守 33**；配方 = `S089_NOEAGER=1 S089_CGS=3 S089_GRAPHPROF=0 S089_POOL=800000000` @ L=16,384 / mbt 1024 / util 0.922。工具 = `tools/step090_orca_speed.py` + `anchor_longctx.py --model`；台账 `step090_boots.json`。**仍欠**：长稳/多轮、与 086 MTP 档的**同口径**对照、L 往上顶的拐点。
- **O3 收口方式（089 后只剩两个裁定项，详版 §A-1）**：**a) 定档** —— 把 089 臂做成 Orca 的投机生产形态（需先补速度档 / eager vs FULL 图 / 长稳）；**b) 进 O4** —— Orca/KVMem（Orca 自己记账 page/viewport，**不要把 1456 当现成常量套**：089 测出 Orca 在这五把钥匙下也是 1456，那是两者 `text_config` 逐字节相同的结果，不是可免验的前提）。
- **交付** = `tools/serve_orcasaq2_029_dflash2.cmd` + `tools/serve_orcasaq2_029_nvfp4_dflash2.cmd`（`S088_SPEC`/`S088_L`/`S088_UTIL` 三旋钮，0.92 是**测量档**永不进生产）+ `step085_orca_boot.py` 扩 `d2`/`d24` 臂与三条自证 + `_tmp_line_b/o88_acc_sample.py`；证据 `prod029_logs/orca_o3/` + 台账 `prod029_logs/step088_boots.json`。**本步收尾按新的必守 30 执行：不恢复生产，只记录**（臂后生产处于 WARN / 降页 250 MiB / 主机 RAM 仅 2.69 GiB 可用）。

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
- **Orca MTP：GO（085）**——checkpoint 自带 1 层 MTP 头可用，接受率 0.627–0.646、平均接受长度 2.25–2.29、解码 **2.12x**（45.5 vs 21.5 tok/s），正确性 on/off 各 9/9。代价 = KV 池 −36.6%、并发 1.64x→1.04x、权重 +0.21 GiB，且 **16 GB 卡上带投机起不了 16K**（引擎自估上界 12,000），两臂同用 12,000/0.88。**nvfp4 KV 档 = GO（086，限 `num_speculative_tokens=1`）**：16K 上下文保住、解码 **1.78x**、接受率 0.777–0.794、正确性两臂各 15/15，代价 = KV 池 −41.5%、并发 2.17x→1.27x；**`≥2` 步在 nvfp4 上对 ≳10K 长 prompt 停摆 = 判 NEGATIVE**（机制未定罪）。下一头名 = O3。
- Orca DFlash2：**O3 前三项已完成（087）**——target adapter 无需新写（前提被推翻）、兼容守卫对 Orca PASS、DFlash2 只能走 V2 runner 而该组合已被 085/086 证实可用；**剩下的门 = 草稿资产**（三条路线：①复用 gptq3c / ②从 bf16 源做 Orca 线 RTN 无标定重量化 / ③用 Orca 自采 Hessian 做 GPTQ），**用户 2026-10-03 已裁定走 ①（禁令解除仅限 O3/088）⇒ 088 执行：机制 + 对齐 GO（prose 接受率 0.5448 / 524 drafts；countdown 0.9932），但 088 写的"本卡容量 NO-GO"已被 089 作废**。**089（2026-10-04）= 逐条照抄 GSQ 生产的容量杠杆（`VLLM_KV_GROUP_SIZE=8` + `--mamba-ssm-cache-dtype bfloat16` + `--mamba-cache-mode align` + `--enable-prefix-caching` + 8 GiB **KV** 卸载 + util 0.922）后，同一支草稿在 Orca 上跑到 L=16,384（池 41,275 / 并发 2.52x）与 L=131,072（池 157,910 / 并发 1.20x），零 ERROR、needle 三深度 6/6 HIT、接受率 0.4935 / 0.4811** **090 再把速度问题解决**：FULL 图 + 手压 KV 池 8e8 ⇒ **8k anchor 中位 79.91-86.63 tok/s**（自动池那两支只有 4.83/5.49 = 驻留悬崖，必守 33），needle 每支 3/3、零 ERROR。**088 摆出的三条候选（换 >16 GB 卡 / 减权重 / 改草稿）撤销，纪律 = 必守 32 + 33**。**O3 五项全部达成、可收档；剩余裁定项两个**：a) 把 **090 配方**定档为 Orca 的投机生产形态（还欠长稳/多轮与 086 MTP 档的同口径对照）；b) 进 O4 Orca/KVMem。
- Orca KVMem：顺延到 GSQ K3 和 Orca O1 之后。
