# vllm-030win — Windows 原生 vLLM 引擎（Qwen3.8-27B + DFlash2 投机解码）

基于 **vLLM v0.27.1** 的 Windows 移植 + 自研增强，面向 **ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ**（compressed_tensors WNA16 int3）在消费级 Blackwell 卡（RTX 50 系，SM120）上的高性能推理。本仓库同时是「vLLM 0.30-on-Windows 迁移」战役的记录仓（见 `docs/`）。

English: A Windows-native vLLM v0.27.1 build with custom enhancements (DFlash2 speculative decoding, NVFP4 KV cache on SM120, WNA16 int3 quantization), targeting the Qwen3.8-27B-3Bit-GSQ model on consumer Blackwell GPUs. See `docs/` for the migration-to-0.30 campaign records. Upstream README: [README-UPSTREAM.md](README-UPSTREAM.md).

## 迁移战役状态（2026-09-27 更新）

> 把现役 0.27.1 自研栈迁到 SystemPanic vLLM 0.29 底座 + 甄选 0.30 功能。deadline 2026-10-30（新 27B 发布即止损），硬承诺 10-18 前完成生产等价+切换。

| 阶段 | 状态 | 说明 |
|---|---|---|
| 阶段 0 锚点+底座 | ✅ 收官 | PPL 主锚/长上下文锚/无草稿基线；cp312 venv；底座 git 化 |
| 阶段 1 底座冒烟 | ✅ 收官 | S1 GSQ int3 裸加载 + S2 PPL 对锚 \|Δ\|≤0.0025；S3-S6 冒烟全过（A6/A8 锚补采）；两个结构缺口定性（投机解码挂批 4；图模式已由批 3b custom-op 恢复） |
| 阶段 2 自研迁移+验收 | ✅ **收官（2026-09-26，三线合并回归全过）** | 批1-5 全落地（`57188ef`…`78db8a6`）；**三线合并回归 ✅（步骤 018）**：线A 无 spec 四档超锚（56.12/55.97/54.32/50.38 vs 55.7/55.6/54.2/50.3）、线B spec 端到端三锚全过（A2 73.00 vs 60.65；A5 68.59/41.45/24.39 = +19~29%；A4 接受率带内偏上 0.72-0.80）、线C 图模式合并覆盖；PPL 复锚 8/8 带内；两回归修复（`6fc2108` INC 派发错位、`e9d534d` HummingLinearMethod 过 custom-op）——**生产等价判定成立，10-18 硬承诺条件达成** |
| 阶段 3 0.30 甄选 | ⚪ 切换后滚动 | A 组+B7+D 组+C1/C6/C8，砍尾 C8→C1/C6→D |
| 阶段 4 回归+切换 | ✅ **生产切换完成（2026-09-26 步骤 020，0.29 栈服务运行中）** | 切换演练 ✅ + 四件验证/日志三行 ✅（pool 114,974 容量无缩水、xqa+writer+spec capture 在）；回退=预案 §4 两步（0.27 冻结锚）；**⭐R8 KV offload 崩溃已修复回填（步骤 021，底座 `d7cdb91`：cuMemcpyBatchAsync 驱动缺陷→逐条 cuMemcpyAsync）**——多轮场景实证 32k 二轮 TTFT 21.0→3.1s；vision 面未切换（A7 锚挂账） |
| T-perf 诊断战役 | ✅ **诊断收官 + 优化首胜（2026-09-26 步骤 022/023）** | 诊断：溢出排除、GPU 贴权重读地板 14.9ms/步、其余=图外 CPU 派发；**boot 方差根因修复（底座 `3498ef1`：AOT 缓存尾换行非对称→每 boot 重编译→生成码漂移，修复后同产物 ±0.4%）**；优化首胜：**草稿 FULL_DECODE_ONLY 图化解耦（`dflash/speculator.py` 草稿图模式原被绑死在 target 模式）⇒ decode 31.53→27.35ms（−13.3%）**（实验 B GDN 入图=阴性已回退）+ GDN 包装减脂五处（−1.3%）⇒ **战役累计 31.53→27.0ms（−14.4%）、吞吐 77→90.5 tok/s、正确性/接受率保持**。挂账头名：**快慢态孤例**（同产物出现 20.85ms 快态未复现，运行时层机制，捕获=再降 ~24%） |
| 快慢态定罪+五臂对照 | ✅ **步骤 025（2026-09-26 深夜）** | 8-boot 三臂采样（R=重编译 20.9×2 复现/A=AOT 27.x×4/W=CPU 热身无效）；机制定名=**每步 flashinfer NVFP4 KV 写校验 `.cpu()` D2H 同步点**（`page.py _as_float32_scalar_tensors`，DFlash2 propose 热路径）等 marlin lm_head 核（双峰 0.75 vs 8-17ms 同核同配置，挂账）；**五臂对照：新配置（删 PIECEWISE 编译、capture1）无 spec 18.0→14.2ms（−21%）、GPU 占用 76→85% 吃满**，0.29/0.27 打平（14.19/14.43）；GPU 没吃满=spec 模式 CPU 派发空转（60-74%）；新 launcher 入库 `run_anchor_nospec.cmd`(改)+`serve_gsq_base029_nospec_nograph.cmd`(新) |
| FULL_AND_PIECEWISE 机制+spec 复用 | ✅ **步骤 026（2026-09-27 凌晨）** | 机制=显式 PIECEWISE 禁 FULL 图（decode 每步 90+ eager 派发）vs 默认 FULL_AND_PIECEWISE（decode FULL 整图）；**DFlash2 兼容已修复**（spec 三段捕获全过含草稿 FULL 图、无挂死，「spec 必须 PIECEWISE」历史约束失效）；3-boot 复测 **25.86-26.45ms 可复现（vs 现役 27.0 = −4%）**、首测 20.06ms 判快态孤例不入账；flashinfer 校验 `.cpu()` 补丁 A/B=阴性（E1 配置下 KV 写在图内无此同步点）已回退；~~生产化前需长稳~~ **已换装（步骤 027）** |
| E1 生产化换装 | ✅ **步骤 027（2026-09-27）** | 真实请求长稳 **20/20 零挂死**（1/20 死锁闸门过）+ 正确性门全过（needle 8k×2+32k×3+100k 全中、**多轮 TTFT 32k 21.13→3.05s 复验**、reasoning/tool-call、共享显存平线无溢出、池账 114,974 无缩水、三段捕获 PIECEWISE+FULL+dflash2 FULL 全过）；**生产已切 FULL_AND_PIECEWISE**（`serve_gsq_prod029_n2.cmd` 删 compilation-config 行）；**3-boot 水位 25.69/25.73ms = −4.8% vs 27.0**（post-soak 22.29 判快态孤例不入账，快慢态谱 22-26ms 挂账） |
| 快慢态机制战役 | 进行中（**步骤 028，2026-09-27**） | ①**marlin 双峰=PIECEWISE 特有，全图下不存在**（用户指名补测：TA2 621/TR2 603 次全 <1ms、慢峰 0）②**载入路径定罪**：8 boot 矩阵现场编译 19.4-20.4 全快 / AOT 载入 25.0-25.8 全慢，载入刚生成的全新产物仍慢（+29%）⇒锅在载入路径非产物 ③递归树打点对比（debug 启动器 pydbg）两路径结构同构，差异=submod 命名偏移+inductor 装载层（挂账）④载体=GPU 并集一致、差在 propose 链 D2H 同步 gap（5.27 vs 1.45ms/步）；**止血方案待拍板：启动跳过 AOT 载入=decode 25.7→20.35（−21%）、boot +2-3 分钟**；生产当前快态 20.37ms（重编译产物） |
| 快慢态=随机放置（步骤 029，2026-09-27） | 机制未定案，假说部分证实 | **用户「随机」假说成立并精化：现场编译 x8 全快（19.4-20.4）确定性；AOT 载入随机多档（19.5/21.4/22.8/23.4/25.x/36.6）**——步骤 028「载入=慢」修正为「载入=随机抽样」。force-all 阴性（参数全克隆进显存仍 23.39，且触发换页风暴卡死）；记账差 528MB 专用显存（≈草稿权重体量）+ 慢 GEMM 61GB/s=PCIe5 签名真实。指针普查引擎内死锁（工具债，8 bug 累计）。**「拿回 500MB 扩上下文」方案有基础但依赖可控放置，机制未明不落地**（收口三径：外置记账相关性 / 池 +500MB 试 / ncu 直读） |
| 快慢态收口+钉快档交付 | ✅ **步骤 030（2026-09-27 晚）** | **机制定案**：草稿 31 层（gptq3c）权重在 `load_dflash_model` free=0 压力窗内被 WDDM 逐块掷骰子（10-boot Spearman −0.994、三线记账对冲 ded−325/free+325/shr+326、8 个离散档=分块放置），落共享段层被 propose GEMM 走 PCIe（61GB/s）=慢档。**钉快档交付：`tools/pin_shim`（sitecustomize 注入+草稿权重逐层搬回）21/21 boot 全快档 20.3-21.6ms**（干净 boot 仅 20% 快档）、acc 与锚同形、needle 全中、长稳真实语义 20/20；**生产换装后 20.57ms（vs 25.7=−20%）**，回滚=launcher 删 4 行。池 3.9e9 探针推翻「扩池钉慢端」（5/5 随机依旧）——「扩池拿 500MB」可行但代价=decode 随机档（与钉快档组合未验）。挂账：注入偏置源归因、soak 阈值判定代码债、指针普查工具债 |
| gap 优化战役（进行中） | 步骤 031（2026-09-27 深夜） | 目标 decode 18.5ms（−10%）。步内剖定罪：`flashinfer page.py _as_float32_scalar_tensors` `.cpu()` 14ms 同步/步（025 路径，026「图内消失」判定错误）+ GDN build 7.4ms/步 CPU + GPU idle 4.35ms/步。**page 校验缓存补丁平反入账：交替三对全胜 −1.3ms（−5.6%），026 阴性=假阴性（随机频谱淹没）**；GDN 复用补丁阴性回退（净 0.21ms=build 被流水线掩盖，排除）。时段热态摆动 20.7-25.1ms 须净段复测。**下一靶=GPU idle 的 Python 胶水**（非 aten 层） |
| gap 优化第二胜（步骤 032，2026-09-27 深夜） | ✅ **−2.55ms / 吞吐 +13%** | 靶=031 的「GPU idle 的 Python 胶水」。新写断点采样法 idle 归因器（v1 重复计算作废）：idle **8.01ms/步**中**主线程独占 7.83ms**，栈指纹全在 **`prepare_attn <- build_attn_metadata <- build`**（attention metadata 构建）。全 trace 扫描证明 **`cudaMemcpyAsync` 每步恰好 1 次长同步**（217/212 步；测量态 13.3ms、生产态桩实测 **p50 17.38ms**），栈 = 草稿 `_build_draft_attn_metadata` 读 **`CommonAttentionMetadata.seq_lens_cpu`**（上游 deprecated property = `seq_lens.to("cpu")`）；桩实证 `causal=False`（DFlash2 草稿 non-causal 层）⇒ `all_uses_trtllm=False` ⇒ 触发同步，而 `ded_xqa/dec_trtllm=True`、**`NEEDS_PAGED=False`** ⇒ **结果被丢弃（纯浪费）**。修法=`needs_seq_lens_cpu` 改为真正消费 `seq_lens_np/num_blocks_np` 的路径（**`tools/apply_seq_lens_cpu_patch.py`**，apply/revert 幂等）。**交替 A/B 三对全胜**：20.81/20.87/19.06 vs 23.51/22.43/22.46 ⇒ **均值 −2.55ms（−11.2%）**、吞吐 104.3→117.8 tok/s（**+13.0%**）；acc 无系统差异；needle 8k 3/3、32k 3/3；32k 多轮 TTFT 21.8→3.15s；长稳过。**⭐补丁后 GPU util p50 97% ⇒ GPU 已饱和、CPU 侧优化空间到尽头**（SA3 19.06ms/124.6 tok/s 逼近 18.5ms 目标）——下一步须转 **GPU 侧**（权重读取带宽利用率 ~80%、草稿 31 层每步全跑） |
| gap 优化第三胜（步骤 033，2026-09-27 深夜） | ✅ **步长 −10.4% / 吞吐 +11.4% / 方差消失** | 接 032「转 GPU 侧」。①**GPU 账更正**：轻量 profiler（无 stack/shapes）下 **GPU busy 20.03ms/步（93.9%）、idle 仅 1.30ms** —— **031 的「idle 8ms」是带 stack profiler 自身开销的假象**。②kernel 账：humming GEMM 14.50ms（72.5%）、cutlass bf16 2.196ms、marlin lm_head 1.523ms。③**异常定罪**：草稿 `qkv_proj`（6144x5120）**223.2µs、带宽效率 6.1%**，而同为 11.8MB int32/3bit、**launch 配置完全相同**（grid70/block128/smem45056/reg102）的 target o_proj（5120x6144）只 18.5µs ⇒ 差异只在数据位置；每步 5 次 dur 序列 `[267,18,271,267,276]` = 草稿 5 层只 1 层在专用显存。④**机制（修正 030）**：`p.data=new` 后旧张量回 **torch 缓存池（块仍在共享段）**，下一遍 `empty_like` 复用同位置块 → **搬移在共享段内空转**；`empty_cache()` 把池还给 WDDM（**free 0→3349MB**）后重分配才落专用显存。⑤**效果**：qkv **223.2→18.2µs（eff 6.1%→75.3%，单峰）**、humming GEMM 总 **14.50→13.00 ms/step**。⑥**交替 A/B 三对**：PA 22.31/19.03/20.78（摆动 3.3ms）vs PB **18.55/18.54/18.55（三臂 p50 完全一致）** ⇒ **−2.16ms（−10.4%）**、吞吐 117.7→131.1 tok/s（**+11.4%**）、**快慢态方差消除**。⑦正确性门：needle 8k 3/3 + 32k 3/3、32k 多轮 TTFT 20.41→3.00s、**8k 稳态 120.6-122.8 tok/s（032 为 93-98，+25%）**；用户指定题目测试（短提示词长输出）答案 `\boxed{3}` **正确**、长输出稳态 ~129 tok/s。**18.55ms 已达 031 目标**；三胜累计 −1.3 −2.55 −2.16ms。固化：pin_shim 默认 `VLLM_DBG_PIN_PASSES=2` + `SKIP_SHARED=1` |
| 性能侧到头 + 容量账 + 池值上探（步骤 034，2026-09-27 深夜） | ✅ 测量收官（性能侧确认到头；容量 +35% 可期，性能代价待定罪） | ①**四档 kernel 账**（短/8k/32k/100k；`/start_profile` 精确包 decode 段 + trace 内注解切窗口）：kernel/step **17.60 / 17.60 / 18.40 / 20.36 ms**，GPU 利用率全程 **93%**，humming GEMM 带宽效率 **32k/100k 均 92.4%**（与上下文长度无关），长上下文增量全在 attention（0.42→1.30→3.37 ms/step）⇒ **性能侧无余量**（首次在长上下文验证，修正 031/033 只在短上下文验证的局限）。②**容量账精确化**（同池多臂）：池 = 单价 × token，**单价 29,570–29,630 B/token 纯线性**（三点一致）；分解 = target KV **18,432**（62%）+ 草稿 **8,029**（27%）+ mamba **3,111**（10.5%）；同池无草稿 157,826 token（单价 21,543）。③**池值上探（零代码改动）**：3.4e9 → **4.6e9 容量 114,974 → 155,326（+35%）**、needle 全中、进程存活；**代价未定罪**（8k steady 118.91→114.90→103.28 单调降，交替 A/B 只跑完第 1 臂即被叫停）。④**两条原假设被推翻**：(b) GDN state **已是 bf16**（`--mamba-ssm-cache-dtype=auto` 跟随 model_dtype），0.491GB 真来源是 align 模式份数 `2 + N(=2) + checkpoints`；(a) 草稿 KV 窗口化受 vLLM **设计约束**（`_get_kv_cache_groups_uniform_page_size` 假设 1「所有 group 每 block 物理内存相同」+ 假设 4「每 group 层数相同」，注释自陈 breaking it is non-trivial due to memory fragmentation）⇒ `num_blocks = available // bytes_per_block` 让所有 group 共享同一 num_blocks，**草稿 5 层（sliding_window=2048）按全长占池 27%** |
| 池值定罪 + 容量真公式 + 草稿 27% 真机制（步骤 035，2026-09-28 凌晨） | ✅ **池值定罪：3.4e9 是硬上限**（改分组是唯一免费容量来源，+18.5%） | ①**短上下文 8 臂时间对称交替**（3.4/3.8/4.2/4.6e9 ×2，2k）：med_steady 均值 101.5/95.0/111.4/99.3，**同一池值两次 boot 可差 20%** ⇒ 034 的「8k 单调降」是 boot 间方差。②**长上下文各自容量 90%**（冷 prefill，用户要求不能只测 32k）：3.4e9@101,166 **92.25/92.33 s**；3.8e9@113,295 **103.46 / 506.40 s**（**一半概率崩**）；4.6e9@101,166 **382.78/379.95 s**。③**同长度对照**（两组都 101,166 token）⇒ **池值本身让 prefill 慢 4.13 倍**，长度因素彻底排除。④**真实溢出**（per-process Shared 扣 offload mmap 8,183 MiB + 76 MiB 基线）：3.4e9 → 170/246、3.8e9 → 239/323、4.2e9 → 565/625、4.6e9 → 992/1,086 MiB；ded 饱和 ~15.7/16.3 GiB。⑤**容量公式 6 点全中**（114,974 / 128,241 / 142,060 / 155,326 / 165,608 / 223,576），`blocks_per_req = Σ over groups cdiv(max_memory_usage_bytes, page_size_bytes)`。⑥**草稿 27% 真机制推翻重写**（shim 实测 15 组）：草稿自己只占 **1.5%**（3/199 blocks），真因是它的 5 层把 `group_size` 从 16 压到 **5** ⇒ 目标 16 层 full 从 1 组变 4 组、mamba 48 层从 3 组变 10 组，而每组都要满 max_model_len 的 39 个 block；**L=5 恰好是取整最差点**；**改 L=4/8 → 136,190 token（+18.5%）且溢出量一个字不变**。⑦**prefill 与 decode 对溢出敏感度不同**：无草稿 200k 三档 prefill 全正常（773–796 tok/s）而 decode 随溢出单调劣化（溢出 205.6/141.6/87.6 MiB → **10.31/17.57/52.78 tok/s**），因 decode 每步要读全部 KV（3.98 GiB）走 PCIe。⑧**有草稿 200k 当前架构做不到**：现分组需 5.33e9、改 L=4 需 4.41e9，都超溢出阈值；**改分组能可靠拿到 ~145k**。⑨**capture-sizes 无草稿必须为 1**（= 并发数，用户指出），decode 10.31 → 17.57（+70%） |
| **改分组落地（步骤 036，2026-09-28）** | ✅ **容量 +18.5%（114,974→136,190）、decode 步长 −2.3%、max_model_len 110k→140k** | 接 035「改分组是唯一免费容量来源」把它落地。①**补丁** `tools/apply_kv_group_size_patch.py`：env 开关 `VLLM_KV_GROUP_SIZE=N` 强制 KV 分组 layers-per-group，**不设置则逐字节走上游分支**（默认零影响、apply/revert 幂等、底座仓 commit `f0bff05` + venv 双路径）。②**机制（kvdump 实测校准）**：`unify_kv_cache_spec_page_size` 把三种 spec 的每层 page 统一到 **3,262,464 B = 2832 token × 1152 B/token** ⇒ `bytes_per_block = 组内最大层数 × 3,262,464`（随 G 线性），而每组 `blocks/request` **与组内层数无关** ⇒ blocks_per_req 只取决于各桶切成几组。③**G 表实测 6/6 与公式逐点吻合**（3.4e9·110k）：G=1 → 137,930 / G=2 → 137,434 / G=4 → 136,190 / **G=5（上游）→ 114,974** / **G=8 → 136,190** / G=16 → 132,407。④**选 G=8**：容量 +18.5%、**组数 15 → 9（比上游还少）**、decode 步长 **18.73/18.74 → 18.30/18.31/18.32 ms（−2.3%）**、prefill 64k 冷档 **47.263/47.248 → 47.171/47.171 s（无差异）**、acc 无系统差异；`--max-model-len` 上限 **110,000 → ≈144,432**（实测 140,000 可用、容量 143,307），**池大小与溢出量一个字不变**。⑤**needle 门**：@110k（8k/32k/64k/100k）四档 + @140k（8k/64k/**130k**）三档**全中**（130k ttft 87.66s、稳态 104.7 tok/s）；**生产已落地**（`serve_gsq_prod029_n2.cmd` + `VLLM_KV_GROUP_SIZE=8` + `--max-model-len 130000`，apply/revert 幂等脚本；冒烟实测 **142,016 tokens / 1.09x** = 预测值、20 条长稳 **20/20 健康**，5 个 FAIL 全是 `n_tokens>500` 阈值假阳性）。⑥**草稿 KV 窗口化定罪**（回答用户「草稿占用随上下文增长」）：逐组块数实测草稿 sw 组**恒 3 blocks/请求**、target full 组 **39 blocks @110k → 50 blocks @140k** ⇒ **草稿是窗口常数占用、不随上下文增长**；034 的「27%」是分组效应的混合量。⑦**PPL 判据被推翻**：同配置两次 boot 的 mean_nll 就差 **1e-3 量级（最大 2.01e-03、无系统方向）**，G=8 与基线差异落在同一漂移带 ⇒ **「PPL 逐位判等」在本栈不成立**，正确性主判据改用 needle、PPL 改容差 + 多采样（根因待挖：nvfp4 KV / flashinfer autotune 每次 boot 重跑）。⑧坑：PowerShell `--lengths $s` 数组须 `-split` 展开、Git Bash `cmd /c` 须 `MSYS_NO_PATHCONV=1`、heredoc 折叠 `\\`、PowerShell 未定义变量在 `Continue` 模式下静默跳过 |

底座仓（0.29 侧一切代码改动）：`G:\qwen3.8model\vllm-029base-git`（baseline `9948275`，每批一 commit + venv 同步）。运行/验收铁律、环境契约、垫片清单见 `docs/交接文档.md` 与 `tools/shims/SHIMS.md`。

## 功能特性

- **Windows 构建使能**：MSVC 2022 + CUDA 13 全链路构建修复（CUTLASS/MSVC 适配、CUDA 13 对齐、进程/共享内存 Windows 化）
- **GSQ 3-bit 量化**：compressed_tensors WNA16（int3 g128 + embed/lm_head int4 g64，pack-quantized）加载与推理，含 torch WNA16 兜底内核
- **DFlash2 投机解码**：DFlash2 draft 模型 + V2 speculator（含 GPTQ 码本、triton grouped-conv），4k 接受率 ~39%、长上下文 ~56%，decode +35~60%
- **NVFP4 KV cache（SM120）**：FA2 nvfp4-KV 路径、非因果 prefill、XQA decode 放宽到 SM12x
- **Multi-TurboQuant KV 压缩**：KV 带宽压缩（KV 容量 +33%、接受率 −0.7pp）
- **KV 异构池**：跳层 draft + 量化 target 的块大小解析、INT4 页尺寸、细粒度 prefix hash
- **工程化**：GC 冻结 CUDA graph 捕获、长上下文协议输出预留、量化 embedding（码本）、MTP 量化权重加载

## 环境要求

| 项 | 版本 |
|---|---|
| OS | Windows 10/11 x64 |
| GPU | NVIDIA SM120（RTX 50 系）验证；其他架构理论可用（FA2 fat binary 全架构） |
| MSVC | Visual Studio 2022（19.4x） |
| CUDA | 13.0+（nvcc 在 PATH） |
| Python | 3.13（构建产物 cp313） |
| PyTorch | 2.13.0+cu130 |
| Triton | triton-windows 3.7.1 |
| Ninja | 构建用 |

## 构建

```powershell
# 1. 准备 venv
python -m venv venv; .\venv\Scripts\activate
pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements\common.txt -r requirements\windows.txt
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
# Qwen3.8-27B-3Bit-GSQ（生产配置）
.\serve_qwen38_gsq.cmd                    # 默认 ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ, port 8000

# MiniCPM5-2B + DSpark 投机解码 + BGE-M3 共存（RAGFlow 后端）
.\serve_minicpm5_2b_dspark.cmd

# 停止（按命令行杀，端口 kill 会漏实例）
.\stop_minicpm5.ps1
```

推荐参数要点：`--language-model-only`（不加载视觉塔）；KV 池手动 `--kv-cache-memory-bytes`（勿用 auto，压缩省出的显存会被 auto 吃掉）；CUDA graph 用 `PIECEWISE`（FULL 模式有已知挂死）。

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

- `vllm-030win-迁移计划.md` — 0.30 迁移执行计划 v2（deadline 2026-10-30）
- `vllm-030win-调研-*.md` — 三份调研报告（改动清点 / 0.29 whl 溯源与 0.30 差异面 / 0.30 功能菜单）
- `实验步骤文档.md` — 逐步实验记录（协议/操作/结果/判定）
- `实验日志.md` — 项目日志（按日工作记录：做了什么/结论/踩坑/下一步）
- `进度文档.md` — 阶段总览 + 任务板 + 资产地图
- `锚点采集协议.md` — 验收对照系（锚点矩阵与复跑规则）
- `CONTEXT.md` — 项目术语表

## License

Apache-2.0（基于 vLLM v0.27.1，保留上游 LICENSE/NOTICE 与 [README-UPSTREAM.md](README-UPSTREAM.md)）。自研增强部分同 Apache-2.0。
