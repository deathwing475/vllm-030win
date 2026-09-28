# vllm-030win — Windows 原生 vLLM 引擎（Qwen3.8-27B + DFlash2 投机解码）

基于 **vLLM v0.27.1** 的 Windows 移植 + 自研增强，面向 **ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ**（compressed_tensors WNA16 int3）在消费级 Blackwell 卡（RTX 50 系，SM120）上的高性能推理。本仓库同时是「vLLM 0.30-on-Windows 迁移」战役的记录仓（见 `docs/`）。

English: A Windows-native vLLM v0.27.1 build with custom enhancements (DFlash2 speculative decoding, NVFP4 KV cache on SM120, WNA16 int3 quantization), targeting the Qwen3.8-27B-3Bit-GSQ model on consumer Blackwell GPUs. See `docs/` for the migration-to-0.30 campaign records. Upstream README: [README-UPSTREAM.md](README-UPSTREAM.md).

## 迁移战役状态（2026-09-28 更新）

> 把现役 0.27.1 自研栈迁到 SystemPanic vLLM 0.29 底座 + 甄选 0.30 功能。deadline 2026-10-30（新 27B 发布即止损），硬承诺 10-18 前完成生产等价+切换（**实际 9-26 提前 22 天收口**）。

| 阶段 | 状态 | 说明 |
|---|---|---|
| 阶段 0 锚点+底座 | ✅ 收官 | PPL 主锚/长上下文锚/无草稿基线；cp312 venv；底座 git 化 |
| 阶段 1 底座冒烟 | ✅ 收官 | S1 GSQ int3 裸加载 + S2 PPL 对锚 \|Δ\|≤0.0025；S3-S6 冒烟全过（A6/A8 锚补采）；两个结构缺口定性（投机解码挂批 4；图模式已由批 3b custom-op 恢复） |
| 阶段 2 自研迁移+验收 | ✅ **收官（2026-09-26，三线合并回归全过）** | 批1-5 全落地（`57188ef`…`78db8a6`）；**三线合并回归 ✅（步骤 018）**：线A 无 spec 四档超锚（56.12/55.97/54.32/50.38 vs 55.7/55.6/54.2/50.3）、线B spec 端到端三锚全过（A2 73.00 vs 60.65；A5 68.59/41.45/24.39 = +19~29%；A4 接受率带内偏上 0.72-0.80）、线C 图模式合并覆盖；PPL 复锚 8/8 带内；两回归修复（`6fc2108` INC 派发错位、`e9d534d` HummingLinearMethod 过 custom-op）——**生产等价判定成立，10-18 硬承诺条件达成** |
| 阶段 3 0.30 甄选 | ✅ **大部分收档（步骤 037-038）** | **A7/A1 移植 2 项**（`#54782` PIECEWISE 图不可用抛错的安全网 + `#55341` 运行期顺序缺陷修复），6 项 NO-GO；**关键发现：0.29 底座已含大量 0.30 改动**（`#54782` 移植后自检 17/17 = 100%、`#54646` 89%）；**D 组/C6 全 NO-GO**（CuTe DSL Linux-only 实测定罪）；剩余项（C1/A8/A9/A2/B7/C8）评估后价值低或不适用 |
| 阶段 4 回归+切换 | ✅ **生产切换完成（2026-09-26 步骤 020，0.29 栈服务运行中）** | 切换演练 ✅ + 四件验证/日志核对 ✅；回退=预案 §4 两步（0.27 冻结锚）；**⭐R8 KV offload 崩溃已修复回填（步骤 021，底座 `d7cdb91`：cuMemcpyBatchAsync 驱动缺陷→逐条 cuMemcpyAsync）**——多轮场景实证 32k 二轮 TTFT 21.0→3.1s；vision 面未切换（A7 锚挂账） |
| 阶段 5 收尾/优化 | ✅ **性能 + 容量两线均已到头（046 容量再 +12.5% 并落地生产）** | **性能**：图模式换装 −4.8%（027）+ gap 三胜（031/032/033，−1.3/−2.55/−2.16ms）+ 改分组 −2.3%（036）⇒ **步长 18.30ms**；034 四档 kernel 账确认 GPU 侧无余量（humming 带宽效率 92.4%、GPU 93%、与上下文长度无关）。**容量**：改分组 +18.5%（036）+ 提长 110k→144,432（039/040）+ **ssm bf16 → L=163,072（045 验收 / 046 生产落地）**⇒ **163,719 tokens** |
| T-perf 诊断战役 | ✅ **诊断收官 + 优化首胜（步骤 022/023）** | 诊断：溢出排除、GPU 贴权重读地板 14.9ms/步、其余=图外 CPU 派发；**boot 方差根因修复（底座 `3498ef1`：AOT 缓存尾换行非对称→每 boot 重编译→生成码漂移，修复后同产物 ±0.4%）**；优化首胜：**草稿 FULL_DECODE_ONLY 图化解耦 ⇒ decode 31.53→27.35ms（−13.3%）** + GDN 包装减脂五处（−1.3%）⇒ **累计 31.53→27.0ms（−14.4%）、吞吐 77→90.5 tok/s、正确性/接受率保持** |
| 快慢态定罪+五臂对照 | ✅ **步骤 025（2026-09-26 深夜）** | 8-boot 三臂采样（R=重编译 20.9×2 复现/A=AOT 27.x×4/W=CPU 热身无效）；机制定名=**每步 flashinfer NVFP4 KV 写校验 `.cpu()` D2H 同步点**（`page.py _as_float32_scalar_tensors`，DFlash2 propose 热路径）等 marlin lm_head 核（双峰挂账）；**五臂对照：新配置（删 PIECEWISE 编译、capture1）无 spec 18.0→14.2ms（−21%）、GPU 占用 76→85% 吃满** |
| FULL_AND_PIECEWISE 机制+spec 复用 | ✅ **步骤 026（2026-09-27 凌晨）** | 机制=显式 PIECEWISE 禁 FULL 图（decode 每步 90+ eager 派发）vs 默认 FULL_AND_PIECEWISE（decode FULL 整图）；**DFlash2 兼容已修复**（spec 三段捕获全过含草稿 FULL 图、无挂死，「spec 必须 PIECEWISE」历史约束失效）；3-boot 复测 **25.86-26.45ms（vs 27.0 = −4%）**；flashinfer 校验 `.cpu()` 补丁 A/B=阴性已回退；**已换装（步骤 027）** |
| E1 生产化换装 | ✅ **步骤 027（2026-09-27）** | 真实请求长稳 **20/20 零挂死**（1/20 死锁闸门过）+ 正确性门全过（needle 8k×2+32k×3+100k 全中、**多轮 TTFT 32k 21.13→3.05s 复验**、reasoning/tool-call、共享显存平线无溢出、三段捕获全过）；**生产已切 FULL_AND_PIECEWISE**（`serve_gsq_prod029_n2.cmd` 删 compilation-config 行）；**3-boot 水位 25.69/25.73ms = −4.8% vs 27.0** |
| 快慢态机制战役 | ✅ **步骤 028/029/030 收口** | 028：marlin 双峰=PIECEWISE 特有（全图下不存在）、载入路径定罪；029：**用户「随机」假说成立并精化**（现场编译 x8 全快确定性；AOT 载入随机多档）+ 记账差 528MB 专用显存 + 慢 GEMM 61GB/s=PCIe5 签名；**030 定案**：草稿 31 层权重在 `load_dflash_model` free=0 压力窗内被 WDDM 逐块掷骰子（Spearman −0.994、8 个离散档），落共享段层被 propose GEMM 走 PCIe=慢档 |
| 快慢态收口+钉快档交付 | ✅ **步骤 030（2026-09-27 晚）** | **钉快档交付：`tools/pin_shim`（sitecustomize 注入+草稿权重逐层搬回）21/21 boot 全快档 20.3-21.6ms**（干净 boot 仅 20% 快档）、acc 与锚同形、needle 全中、长稳真实语义 20/20；**生产换装后 20.57ms（vs 25.7=−20%）**，回滚=launcher 删 4 行。池 3.9e9 探针推翻「扩池钉慢端」（5/5 随机依旧）。挂账：注入偏置源归因、soak 阈值判定代码债、指针普查工具债 |
| gap 优化战役 | ✅ **三胜入账（步骤 031/032/033，目标 18.5ms 达成）** | **031** page 校验缓存补丁平反入账 −1.3ms（交替三对全胜）；**032** 草稿 non-causal 层白做 D2H 同步消除 **−2.55ms/吞吐 +13.0%**（`tools/apply_seq_lens_cpu_patch.py`；栈=`_build_draft_attn_metadata <- build_attn_metadata <- build <- seq_lens_cpu`，结果被丢弃=纯浪费）；**033** 钉快档搬移空转修复 **−2.16ms/吞吐 +11.4%/方差消失**（机制：`p.data=new` 后旧张量只回 torch 缓存池，**必须 `empty_cache()` 归还 WDDM 后重分配才落专用显存**；草稿 qkv_proj 223.2→18.2µs）。**033 更正**：轻量 profiler 下 GPU busy 20.03ms/93.9%、idle 仅 1.30ms（031 的「idle 8ms」是 profiler 假象）。**补丁后 GPU util p50 97% ⇒ CPU 侧优化到头** |
| 性能侧到头 + 容量账 + 池值上探 | ✅ **步骤 034（2026-09-27 深夜）** | ①**四档 kernel 账**：kernel/step **17.60/17.60/18.40/20.36 ms**、GPU 利用率 93%、**humming GEMM 带宽效率 32k/100k 均 92.4%**（与上下文无关）⇒ **性能侧无余量**。②**容量账**：单价 29,570–29,630 B/token 纯线性；分解 = target KV 18,432（62%）+ 草稿 8,029（27%）+ mamba 3,111（10.5%）。③**池值上探** 3.4e9→4.6e9 = 155,326（+35%）但代价未定罪。④**两条原假设被推翻**：GDN state 已是 bf16；草稿 KV 窗口化受 vLLM 设计约束（假设「所有 group 每 block 物理内存相同」） |
| 池值定罪 + 容量真公式 + 草稿 27% 真机制 | ✅ **步骤 035（2026-09-28 凌晨）** | ①**池值定罪：3.4e9 是硬上限**（同长度 101,166 token 对照：3.4e9 **92.25/92.33 s** vs 4.6e9 **382.78/379.95 s** ⇒ 池值本身让 prefill 慢 4.13×；3.8e9 一半概率崩）。②短上下文 8 臂无趋势（同池值两次 boot 差 20%）⇒ 034 的「8k 单调降」是方差。③**容量公式 6 点全中**。④**草稿 27% 真机制推翻重写**：草稿自己只占 **1.5%**，真因是它把 `group_size` 从 16 压到 **5**（**L=5 是取整最差点**）⇒ 改 L=4/8 得 136,190（+18.5%）。⑤**prefill 与 decode 对溢出敏感度不同**（无草稿 200k prefill 全正常 773–796 tok/s，decode 随溢出 10.31/17.57/52.78 tok/s）。⑥**capture-sizes 无草稿必须为 1**（用户指出，decode +70%） |
| **改分组落地（步骤 036，2026-09-28）** | ✅ **容量 +18.5%、decode 步长 −2.3%** | ①**补丁** `tools/apply_kv_group_size_patch.py`：env 开关 `VLLM_KV_GROUP_SIZE=N`，**不设置则逐字节走上游分支**（默认零影响、apply/revert 幂等、底座仓 `f0bff05` + venv 双路径）。②**机制（kvdump 实测校准）**：`unify_kv_cache_spec_page_size` 把三种 spec 的每层 page 统一到 **3,262,464 B** ⇒ `bytes_per_block = 组内最大层数 × 3,262,464`，而每组 `blocks/request` 与组内层数无关 ⇒ blocks_per_req 只取决于各桶切成几组。③**G 表实测 6/6 与公式逐点吻合**。④**选 G=8**：容量 +18.5%、**组数 15→9**、decode **18.73/18.74 → 18.30/18.31/18.32 ms**、prefill 无差异、acc 无系统差异；**池大小与溢出量一个字不变**。⑤**needle 门** @110k 四档 + @140k 三档全中。⑥**草稿 KV 窗口化定罪**：sw 组恒 **3 blocks/请求** ⇒ 草稿是窗口常数占用、不随上下文增长。⑦**PPL 判据被推翻**（同配置两次 boot 差 1e-3 量级）⇒ 正确性主判据改用 needle |
| **阶段 3 开篇 A7/A1（步骤 037，2026-09-28）** | ✅ A7/A1 收档：净产出 2 项移植、6 项 NO-GO | **关键发现：0.29 底座（SystemPanic win whl）已含大量 0.30 改动** —— 自检工具（PR diff 的 `+` 行在底座逐行查找）显示 #54782 移植后 **17/17 = 100%**、**#54646 89%**、#54794 37%。①**前置修复**：底座仓 ↔ venv 全树比对（2724 个 `.py`）**只有 1 个文件不同**（032 补丁 venv-only，重建 venv 会静默丢 −2.55ms/步），已回填（`8ef5f64`）。②**A7**：`#54782` **移植**（PIECEWISE 图不可用抛错，安全网）；`#56908`/`#54660`/`#55095` NO-GO。③**A1**：`#55341` **移植运行期**（**真实顺序缺陷**：`warmup_kernels` 在 `capture_model()` 之后会触发 workspace resize → `empty_cache()` → 可能释放 graph buffer）；`#54646`/`#54557`/`#54794` NO-GO。④**验收**：三段图捕获全过、无 `Workspace is locked` 断言、needle 4 档 × 3 = **12/12 命中**。⑤**副产品**：`max_num_scheduled_tokens = 1024` 警告（步骤 040 已结案：增大 mbt 反而吃容量） |
| **阶段 3 续：D 组/C6 评估（步骤 038，2026-09-28）** | ❌ **全部 NO-GO**（D4 实测定罪：CuTe DSL Linux-only） | ①**硬件确认**：RTX 5070 Ti / compute_cap 12.0 = **SM120**。②**D4（GDN prefill SM12x）—— 移植成功但 Windows 上不可用**：`#55715` 逐 hunk 移植（**自检 9/9 = 100%**），启动确实切换了后端路径，但**引擎启动失败** `No module named 'cutlass'`；**根因**：flashinfer 0.6.18 的 SM120 GDN prefill 是 **CuTe DSL 实现**，而 `nvidia-cutlass-dsl` 的包元数据明写 `Operating System :: POSIX :: Linux`。已回退。③**D 组其余**：D1 历史 NO-GO；D2 本栈是 WNA16 int3 不经过；D3 改的是 cmake/Dockerfile（构建期）；D4 另一半只加 SM110；D5 不用 FP8；D6 的 `fused_output_quant_supported` 已含 `is_device_capability_family(100)`（SM120 不满足）。④**C6** 绝大多数 NVFP4/CuTeDSL 专用；AutoRound 2-7bit 是**离线量化算法**，应独立立项。**根因一句话**：D 组/C6 的前提是 **nvfp4/FP8 权重 + CuTe DSL + Linux**，本栈是 **WNA16 int3 + Windows + 预编译 whl**。**教训**：平台专项类移植必须先查「依赖在目标 OS 上是否存在」 |
| **上下文提长（步骤 039，2026-09-28）** | ✅ 130,000 → **140,000**（+7.7%、性能零代价） | ①**硬上限**：G=8 时 `blocks_per_req(L) = 2*cdiv(L,2832) + 27` ⇒ **L ≤ 144,432**（引擎自报 145,551 tokens、needle 6 档含 140k 档全中）。②当轮把 144,432 的掉速读成「decode −21%」（2 boot）故停在 **140,000**（容量 143,307）、needle 4 档全中、revert/apply 往返逐字节一致。③**A5 115k 档 400 结案**（当时 max-model-len = 110,000，115k 被拒属正常）。④（**040 修正**：该「悬崖」实为随机慢 boot，144,432 已定稿）**教训：容量上限 ≠ 可用上限 —— 提长类判据必须含性能** |
| **上下文杠杆穷举（步骤 040，2026-09-28）** | ✅ **定稿 144,432**（+3.2% 上下文）；三条新杠杆全 NO-GO；039 的「144k 悬崖」定罪为随机慢 boot | ①**三条新杠杆全 NO-GO**：**G=1**（唯一有额外容量，147,264 = +5.2%）但 8k steady → **98.60（−16%）**；**G=2** 引擎上限**也是 144,432**（无优势）；**N=1** 上限 **152,064**（容量 154,421）但 steady → **85-95（−27%）**；**mbt** 2048/4096 使容量（142,187/141,085）与性能（steady 91/79）**同时变差**。②**039 悬崖定罪**：同一 L=144,432 两次 boot **118-121 vs 96-99（−19%）**，慢的那次专用显存反而**少 284 MiB**，`pin_shim` 放置日志 **15 次 boot 逐字节相同** ⇒ **随机慢 boot，不是 max_len 效应**。③**定稿 144,432**（145,551 tokens）：多档 needle（8k/64k/115k/135k）与 140,000 **逐档对齐**（ttft 逐位相同，steady 120.2/121.0/113.0/102.3 vs 120.4/116.4/108.0/103.2，needle 4/4×2）。④**容量公式定版**：真实约束是 **`blocks_per_req ≤ num_blocks − 1`**（引擎永久留一个 null block）；**sw 组块数依赖 mbt**（`cdiv(2047+2×mbt,2832)+1`，实测 3/4/5 三档全中）⇒ **mbt 与容量耦合**。⑤**风险记录**：144,432 时 blocks_per_req = 129 恰好用满，满长请求不留前缀缓存余量（8k-135k 实测不受影响） |
| **双判据上下文复核（步骤 041，2026-09-28）** | ✅ **144,432 是「8k ≥85 且 90% 档 ≥70」的最大值**（生产无需改动）；N=0 无投机新杠杆实测 NO-GO | 用户追加两条验收判据后，把 040 判 NO-GO 的配置**全部重审**并补测 N=0。**①判据②（90% `max_model_len` 档 ≥70）是最严格的约束** —— 把所有扩展路径挡在约 145k 以内。**②G=1@147,264**：快态 8k 98.60 达标，但**慢态 80.02-81.08 破 85**（慢态幅度约 −19%）⇒ 得选型原则 **快态 ≥106 才能保证慢态 ≥85**；现役快态 120 有余量故**对慢态免疫**。**③G=2@154,880（N=1）**：8k 85.42-86.59 勉强过，但 86.7% 档只有 **58.25** 破判据②。**④N=0 无投机**（新写 `mk_nospec.py`）：容量 **166,783（+14.6%）** 但 8k 只有 **68.73**（实测投机加速 **+75%**）⇒ **投机是必需的**。**⑤现役在慢态下 90% 档实测 85.69-86.60** ✓ 余量充足（快态 135k/93.5% 档为 102.29）。**⑥`mamba_cache_mode=none` 评估不可用**（其语义就是「prefix caching 关闭时」，与生产多轮对话需求冲突）。**下一步头名 = 慢态根因与规避** |
| **慢态识别+规避交付（步骤 042，2026-09-28）** | ✅ **`tools/prod_watchdog.ps1` 交付并实战验证**；显存指纹破维 + 性能连续谱定性 | 15 boot 实测（census 8 + watchdog 3 + tick 4）。**①显存指纹破维（040 的「低 284 MiB」指纹推翻）**：慢 boot 15,474 vs 快 boot 15,486 **只差 12 MiB** 且跨天漂移 ⇒ 专用显存是影子变量，判定只有性能探针一条路。**②性能是连续谱非二态**：8k 单条全域 **83.45-129.27 tok/s**（低带 83-99 / 中带 101-110 / 高带 112-129），watchdog 三连 boot 89.15→103.59→109.86 直观展示；040/041 的「快/慢二态」是小样本印象。**③引擎侧零可见**：pin 日志 8 boot 逐字节相同、阶段日志快慢逐行等价、`Initial free memory` 是 boot 起点快照（tick 曲线证明）无归因价值 ⇒ 分叉在 WDDM 物理放置层，与已钉死的草稿放置是**两个独立随机源**。**④交付**：watchdog（boot→health→8k needle 探针×3 中位 ≥105 判快→慢则杀重启上限 3 次）实战 2 重启后留下 median 109.86/needle 3/3 的服务，**实测把 89 级 boot 换成 110 级（+24%）**；配置变更后须重标定阈值。**⑤tick 1Hz 曲线工具固化**（`_tmp_line_b/dbg_shim`+`tick_arm.py`；fast 曲线已采：载入→稳态 3,701→free=0 压力窗→PIN 后 3,223→AOT/KV 3,047→满载 0），slow 对照未抽到挂账。**⑥坑**：脚本读写 CRLF 的 .cmd 双端必须 `newline=""`（第三踩）；WDDM 下 nvidia-smi 无 Dedicated/Shared 分列。**下一步头名 = 草稿权重再压缩** |
| **草稿权重再压缩收档（步骤 043，2026-09-28）** | ❌→✅ **gptq4c（clip-H GPTQ）NO-GO，0.545 GiB 体积地板确证**；「草稿权重和草稿 KV 的影响」全链路结案 | 收益三分账定性为低赔率小赌注：配方内唯一大杠杆 = qkv/fc 3→2bit（**−34 MiB → 0.511 GiB**）、吞吐 ≤0.5%、显存释放无池值收益（035：池值上限是 prefill 性能崩非 OOM）。**AutoRound 侦察**（038 C6 独立量化实验）：`auto-round 0.15.1` 可装（纯 Python，PYTHONPATH 隔离 + `auto_round_extension` 从旧 venv 复制）；**gptq3c 打包格式本来就是 `auto_round:auto_gptq`** ⇒ 写回/推理端零适配；真嫁接点 = clip 目标修正（原 `mse_clip` 是权重 MSE，改对角 H 加权输出误差）。**gptq4c**（qkv/fc/kproj 全 2bit + clip-H，0.511 GiB 与预测一致）B1 协议实测 **weighted 0.465 vs 复测锚 gptq3c 0.537（历史 0.525）——差门 6pp+，连无 clip 的 gptq2c（0.478）都不如**（kproj 4→2 ≈ −1pp + clip-H ≈ −0.3pp）⇒ NO-GO。RTN/GPTQ/H-clip 三代算法全试，**qkv/fc 2bit 三判不可用**；AutoRound 原生层级跑挂账（文献增益 1-3pp vs 缺口 6pp+，收益上限不变）。**附带发现：B1 同 checkpoint 复测漂移 +1.2pp ⇒ 协议臂间可比精度约 ±1pp**。残余影响（propose 带宽 ~0.65-1.3 ms/步 + 显存 1.5%）不可再压。**下一步头名 = tick 曲线快慢对照收口（slow 样本未至）** |
| **第二随机源钉住（步骤 044，2026-09-28）** | ✅ **tick 快慢对照收口：慢态机制定罪 + PIN 前置清理钉住（A/B 6 boot 全高带）**；上游调研直接命中官方同款问题 | tick3 抽到极端对照对（slow **72.85** / fast **129.68**）。**①机制定罪（pin 日志 mem 字段）**：两 boot 在草稿载入完成时逐字节相同（free=0 alloc=16462），分叉在 PIN pass1 后的 empty_cache——**fast free 回升 3349 / slow 依然 0（WDDM 提交预算被吃光）** ⇒ pass2 slow 在 free=0 下搬移、**草稿 545 MiB 全落共享段**（propose 走 PCIe → 深低带）。机制链：载入瞬态 ~5.1 GiB 滞留池 → free=0 压力窗 → WDDM 过承诺掷骰子；alloc 记账相同 ⇒ 引擎侧零可见。**②123 秒无日志空档**（dflash_head→dflash2_candidate_s 之间，fast 1s）推翻 042「阶段日志逐行等价」；「pin 日志逐字节相同」也修正（漏看 mem 字段）。**③上游调研**（用户指示，关键词式）：**#57440 直接命中**（离散 GPU 权重载入内存滞留池、官方修法 = gc+empty_cache）、#36973（Triton cubin 常驻 context）、#54383（冷启动 boot 差 5%）、#53436/#58849 备查。**④钉住实施**：pin_shim PIN 前置 `gc.collect()+empty_cache()`（归还载入瞬态，新日志 PIN_PRE_EMPTY_CACHE）+ pass2 前 free 门（PIN_LOW_FREE_WARN）。**A/B：6 boot 全高带 115.68-126.41（中位 121.7）、boot_s 61-75s 无异常；PIN 日志 4 boot 逐字节一致（PRE_EMPTY free=3313 → pass2 free=3185，零警告）**。watchdog 保留作保险。挂账：~110s 未知平台（只影响 boot 时长）；83-99 残谱若再现 = target 权重页 demote（下一战役 target PIN ~26s）。**现头名 = 用户新提案：mamba 0.47 GiB 显存回收方式调研** |
| **mamba 显存回收 + ssm bf16（步骤 045，2026-09-28）** | ✅ **调研 + 方案 A 实测全过：容量 145,551→163,719（+12.5%）；发现「首编深塌」新随机源；生产待切换** | **账本闭环**：mamba = 48 层 ×（conv bf16 102,400 B + ssm **fp32** 3,145,728 B）× align 4 份（2 硬编码 + N=2 + ckpt 0）= **594.75 MiB 物理**；**034「GDN state 已是 bf16」勘误**（conv bf16 / ssm fp32，后者来自模型 config.json 的 `mamba_ssm_dtype=float32` 显式声明）。**方案 A = `--mamba-ssm-cache-dtype bfloat16`**（官方参数、`FUSED_GDN_STATE_DTYPES=(float32,bfloat16)` 白名单内、零代码）**全门 PASS**：①对账 7/7 精确命中（block 2832→1456、num_blocks 130→253、pad 0.12%）；②A/B 6 boot 同带（bf16 123.4/120.6/125.5 vs fp32 118.6/120.5/127.2）、needle 18/18；③PPL 同轮 max\|Δ\|=1.25e-3 漂移带内（fp16 不在 fused 白名单不推）；④长稳 20 条健康（5 FAIL 全为已知假阳性）；⑤**L=163,072 引擎接受、容量 163,719、146k 档（判据②）103.6-104.8 ✓、8k 116.9/125.0 ✓**。**结构替代不可行**（48 层 GDN 是训练权重）；份数侧全锁死（公式硬编码 "2"）。**⭐「首编深塌」新随机源（4/4）**：L/dtype 变更后第一次 boot（重编译 ~145s）8k 仅 16.8-18.7 tok/s，同配置第二 boot（缓存）恢复 116-125；投机未失效（pos1 接受率 77%）、每步开销暴涨 ~103ms，嫌疑 autotune-miss（#53436）；统一 039/040 历史悬案。**044 残谱预测命中**（86.7 低带再现一次）。详账 = `docs/vllm-030win-调研-mamba显存账.md` |
| **生产切换落地（步骤 046，2026-09-28）** | ✅ **bf16 ssm + L=163,072 切进生产：watchdog 一次过（8k 中位 122.58），容量 163,719 上线** | launcher 两行（`--mamba-ssm-cache-dtype bfloat16` + `--max-model-len 163072`）+ 头部记注 + `tools/apply_prod_ssm_bf16_step046.py`（verify/revert 幂等，**链式回退先 046 再 036**——036 的 revert 断言 len==144432）+ 回退预案四处同步 + grep 全仓旧值 12 文件判定（`pytorch.py:147` 是 PyTorch PR 编号巧合不改）。**双 boot 流程未触发**：日志三行 `Directly load the compiled graph(s) ... from the cache` 坐实 **AOT 缓存被 045 实验臂预热** ⇒ **实操结论：双 boot 的本质是「先有同配置编译缓存」，有缓存则无需双 boot，无缓存仍必须（watchdog 判慢重启天然覆盖）**。日志九项全对上（**`GPU KV cache size: 163,719 tokens`** 与 045 预测逐字命中、attn block 1456、ssm bf16 覆盖模型 fp32 声明的 WARNING、G=8、xqa/uint8/sm120、NVFP4 writer、图捕获完成）；`/v1/models` max_model_len 163072；**146k 档 needle 命中 + steady 107.73**（门 70）。本轮零新坑（步骤文档插入踩一次「吞下一节标题」老坑，grep 复核当即修复）。挂账：池值联动 E 轨（bf16 后守恒上探 3.67e9，理论容量再 +7%）、首编深塌机制定罪、83-99 残谱（target 权重 PIN ~26s） |
| **1bit 可行性核实（步骤 047，2026-09-28）** | ❌ **NO-GO：1bit 全族在推理端不可行（四道门全否 + 实测）** | 动手前先核实推理端（**生成侧可行 ≠ 推理侧可行**）。**四道门**：①配置白名单 `INCConfig.SUPPORTED_BITS = {2,3,4,5,6,7,8}`（`inc/inc.py:42`）②路由 `CUDA_HUMMING_SUPPORTED_BITS = {2,3,5,6,7}`（`inc_wna16_scheme.py:18`），1 落入 fallback 只支持 4/8bit marlin → `NotImplementedError` ③**humming csrc 的 layout/TMA 按 4/8bit 写死**（`csrc/launcher/utils.h:104`、`tensor.h:191,402` 的 `num_bits<=4?4:8`、`humming/config/config.py:314` 的 `assert b_dtype.num_bits % 2 == 0`）④auto-round pack 只覆盖 2/3/4/8（1bit 需新打包 pack_factor=32）。**实测钉死**：`INCConfig(weight_bits=1)` → `ValueError`（bits=2 同轮通过作对照）。**生成侧**：`gptq_utils` 的 `qmax=2**(bits-1)` 在 bits=1 时码域 `clamp(-1,0)`、正半轴不可达 ⇒ 量化函数亦须新写。⇒ **调研档阶段 1（mlp 1bit −171 MiB）与阶段 2 的 o_proj/kernel_proj 1bit（−22 MiB）作废、激进档 190,736 作废**。**教训：调研档把 1bit 估为「RTN sign 即可、实现最小」，只算了量化生成侧，漏了推理端要跨「配置白名单/路由/CUDA 内核 layout/打包格式」四层** |
| **E 轨池值守恒回填（步骤 048，2026-09-28）** | ❌ **NO-GO：容量 +9.3% 达成（178,938 tokens）但长档冷 prefill −18.6%；归因 = `max-model-len` 而非池值** | 池值 3.4e9→3.686e9 + L 163,072→177,632。**✓ 通过项**：容量 **178,938 tokens（+9.3%）**、needle 8k/32k/146k/160k 全中、8k decode 与现役**同分布**（A/B 各 3 boot：A 89.26 / B 85.10，t≈0.73 不显著；单臂内 boot 波动 ±21% 远大于臂间差 4.7%）、90% 档 steady 72.87（门 70）、PIN 日志与 044 **逐字一致**（`PIN_PRE_EMPTY_CACHE free=3313`——池值分配在 PIN 之后，044 钉法不受影响）。**✗ 长档冷 prefill −18.6%**（160k 冷档、prompt 157,740、各 2 次）：A **909/919** vs E 轨 **727/761** tok/s；两臂 `/metrics` 逐项一致（`prefix_cache_hits` 均 144,144）无 cache 混淆。**归因（决定性）**：臂 D（池值同 3.686e9 + L 保持 163,072）= **907 tok/s ≈ 基线 914** ⇒ **池值零影响、代价来自 `max-model-len`**。**调研档「守恒回填在机制上避开 035 触发条件」被实测否定**。**机制挂账（扩展 039）**：`max_model_len` 是性能敏感量（039 见 8k decode −21%、本轮见长档 prefill −18.6%），与 num_blocks / 剩余块数 / 池值均无关 ⇒ **任何以抬高 max-model-len 换容量的路线都带此代价**。**生产保持 046 配置不动**。**剩余可行杠杆 = 码本 4→2bit（−30 MiB ⇒ L 180,544，+1.6%）**。坑：`anchor_longctx.py` warmup 用 `lengths[0]`（测冷 prefill 要把长档放后面）；分离式起服判 ready 必须读 `.err`；`Path.read_text()` 无 `newline` 参数 |

底座仓（0.29 侧一切代码改动）：`G:\qwen3.8model\vllm-029base-git`（baseline `9948275`，每批一 commit + venv 同步）。运行/验收铁律、环境契约、垫片清单见 `docs/交接文档.md` 与 `tools/shims/SHIMS.md`。

## 功能特性

- **Windows 构建使能**：MSVC 2022 + CUDA 13 全链路构建修复（CUTLASS/MSVC 适配、CUDA 13 对齐、进程/共享内存 Windows 化）
- **GSQ 3-bit 量化**：compressed_tensors WNA16（int3 g128 + embed/lm_head int4 g64，pack-quantized）加载与推理，含 torch WNA16 兜底内核
- **DFlash2 投机解码**：DFlash2 draft 模型 + V2 speculator（含 GPTQ 码本、triton grouped-conv），4k 接受率 ~39%、长上下文 ~56%，decode +35~60%（**实测加速 +75%：8k 档 120 vs 无投机 68.7 tok/s，步骤 041**）
- **NVFP4 KV cache（SM120）**：FA2 nvfp4-KV 路径、非因果 prefill、XQA decode 放宽到 SM12x
- **Multi-TurboQuant KV 压缩**：KV 带宽压缩（KV 容量 +33%、接受率 −0.7pp）
- **KV 异构池**：跳层 draft + 量化 target 的块大小解析、INT4 页尺寸、细粒度 prefix hash
- **KV 分组可调（`VLLM_KV_GROUP_SIZE`）**：把 layers-per-group 从上游启发式的 5 强制到 8，容量 +18.5%（步骤 036）
- **工程化**：GC 冻结 CUDA graph 捕获、长上下文协议输出预留、量化 embedding（码本）、MTP 量化权重加载、**快档钉（`tools/pin_shim`）**、**慢态 watchdog（`tools/prod_watchdog.ps1`，boot 间性能谱的启动时规避）**

## 环境要求

| 项 | 版本 |
|---|---|
| OS | Windows 10/11 x64 |
| GPU | NVIDIA SM120（RTX 50 系）验证；其他架构理论可用（FA2 fat binary 全架构） |
| MSVC | Visual Studio 2022（19.4x） |
| CUDA | 13.0+（nvcc 在 PATH） |
| Python | **0.27.1 栈 cp313；0.29 生产栈 cp312**（`G:\qwen3.8model\vllm-win029`） |
| PyTorch | **0.27.1 栈 2.13.0+cu130；0.29 生产栈 2.11.0+cu130** |
| Triton | triton-windows 3.7.1 |
| Ninja | 构建用 |

## 构建

```powershell
# 1. 准备 venv
python -m venv venv; .\venv\Scripts\activate
pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements\common.txt -r requirements/windows.txt
pip install triton-windows==3.7.1.post27 flashinfer-python==0.6.18.post1 humming-kernels

# 2. 构建（fix_cuda_13_align.py 需管理员跑一次；fix_cutlass_msvc.py 构建时自动调用）
python fix_cuda_13_align.py   # admin, one-time
pip install . -v              # or: python setup.py bdist_wheel

# 3. 模型（HF 镜像可选）
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=G:\qwen3.8model\hub
```

## 运行

```powershell
# ★现役生产（0.29 栈）：入口 shim -> 记录仓真源
G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ\run_dflash2_n2_base029.cmd
#   真源 = vllm-030win-git\tools\serve_gsq_prod029_n2.cmd
#   配置 = DFlash2 gptq3c N=2 + nvfp4 KV + 池 3.4e9 手动 + FULL_AND_PIECEWISE
#          + prefix-cache + offload 8G + VLLM_KV_GROUP_SIZE=8 + max-model-len 144432
#          + pin_shim 快档钉；容量 145,551 tokens

# 旧栈（0.27.1，冻结只读，回退用）
.\serve_qwen38_gsq.cmd                    # 默认 ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ, port 8000

# MiniCPM5-2B + DSpark 投机解码 + BGE-M3 共存（RAGFlow 后端）
.\serve_minicpm5_2b_dspark.cmd

# 停止（按命令行杀，端口 kill 会漏实例）
.\stop_minicpm5.ps1
```

推荐参数要点：`--language-model-only`（不加载视觉塔）；KV 池手动 `--kv-cache-memory-bytes`（勿用 auto，压缩省出的显存会被 auto 吃掉）；**CUDA graph 用默认 `FULL_AND_PIECEWISE`**（2026-09-27 步骤 026/027 起生产配置；历史上「FULL 必挂死」是 `FULL` 单模式的旧结论，已被 FULL_AND_PIECEWISE 推翻并长稳验证）。

## 目录结构

```
├── vllm/                  # 引擎（v0.27.1 + Windows 修复 + 自研 45 文件增强）
├── csrc/ rust/ cmake/     # 构建侧（MSVC/CUDA13 适配）
├── fix_cuda_13_align.py   # CUDA 13 对齐修复（管理员一次性）
├── fix_cutlass_msvc.py    # CUTLASS MSVC 适配（构建自动调用）
├── serve_*.cmd / *.ps1    # 生产启动/停止脚本
├── patch_vllm_qwen35_embedding.py  # 量化 embedding 补丁（可重放）
└── docs/                  # 迁移计划、调研报告、实验步骤文档、术语表
```

## 文档（docs/）

- `vllm-030win-迁移计划.md` — 0.30 迁移执行计划 v2（deadline 2026-10-30；§0.5 执行实况以此为准）
- `vllm-030win-调研-*.md` — 三份调研报告（改动清点 / 0.29 whl 溯源与 0.30 差异面 / 0.30 功能菜单）
- `实验步骤文档.md` — 逐步实验记录（协议/操作/结果/判定）
- `实验日志.md` — 项目日志（按日工作记录：做了什么/结论/踩坑/下一步）
- `进度文档.md` — 阶段总览 + 任务板 + 资产地图
- `交接文档.md` — 接手入口（现状/环境契约/未决项）
- `交接提示词.md` — 新会话开场提示词（自包含）
- `切换与回退预案.md` — 生产切换操作手册 + 回退步骤
- `锚点采集协议.md` — 验收对照系（锚点矩阵与复跑规则）
- `CONTEXT.md` — 项目术语表

## License

Apache-2.0（基于 vLLM v0.27.1，保留上游 LICENSE/NOTICE 与 [README-UPSTREAM.md](README-UPSTREAM.md)）。自研增强部分同 Apache-2.0。
