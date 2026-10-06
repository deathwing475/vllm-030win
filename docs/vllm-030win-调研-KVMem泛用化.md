# vllm-030win-调研-KVMem泛用化 — 设计档（把 KVMem 从 GSQ 专用适配改成泛用模块）

> 立项：2026-10-04，用户令「下一步应该是把 kvmem 适配了，做成泛用的模块而不是 hack」。
> **授权边界（写在最前）**：本次授权 = 对已冻结验收的 KVMem 做**泛用化模块改造**（行为不变的重构 + 抽象接口），以及为"下一个滑窗混合模型（Orca / 未来收支持 checkpoint）"铺适配路径。**不属于**被硬停止条款冻结的检索改良 / N,R,VIEWPORT 寻优 / GPU 化烘焙 / 长稳产品化 / 缩池换余量 / 生产切换；那些项仍须单独授权。变体并存、生产默认不动、env 门控默认关的铁律不变。
> 配套文档：机制与实测权威 = 《vllm-030win-调研-KVMem虚拟化KV工作区.md》§12.x；派生工具 = `tools/profile_card/`（092，轨 A geometry 双锚点逐位）；本文 = 泛用化设计权威。

## §0 现状定性（2026-10-04 全仓盘点，子代理逐文件核查，全部断言带 file:line）

**结论：KVMem 的"引擎臂"（core / connector / scheduler 管道）几乎全是门控增量且 spec 驱动，已经是泛用模块；真正的 GSQ 专用性集中在四处。**「hack」的总量比直觉小得多——泛用化是四处收敛，不是推翻重来。

**已经是模型无关的（盘点确认，保持原样即可）**：
- 轨迹判定与淘汰管道：`single_type_kv_cache_manager.py:140-142`（isinstance `SlidingWindowSpec` 判定）、`:660-683`（`_retain_for_workspace` 未 armed 首行 return False）、`kv_cache_manager.py:877-898`（`take_workspace_evictions`，对组数/模型零假设）、`sched/output.py:210-225`、`sched/scheduler.py:1285-1297`、`kv_connector/v1/base.py:453-468`（其他连接器默认回退 free）。非混合模型 / 不同组数下 armed 也安全退化为"无 spill"。
- 簿记与索引：`groups.py` 全量（spec 驱动 + 投机草稿防护）、`metadata.py` 全量、`index.py` 全量（几何由 capture 灌入，无模型名/层数写死）。
- 拷贝机与装配簿记：`manager.py` 的 store/snapshot/load 全套、`worker.py` 的拷贝机（`kernel_block_size` 从引擎 cache tensor 现场推导 :246-259；rotary 实例用 `get_rope` 记忆化取回模型同一个 :378-433）。
- 启用与门禁：`factory.py:244-248`（标准注册）、`config/vllm.py:983-991`（接管 offload 位，env 门控）、`input_processor.py:377-405`（准入门）、`envs.py:2181-2189`。

**四处真正的专用性（泛用化工作面）**：
1. **模型内钩子**：`qwen3_next.py:278-304`（`_kvmem_rawk_layer`）与 `:307-351`（`_kvmem_per_layer_sliding_window`）。两者都是【门控增量】（env 未设即 None/False，上游行为逐字节不变），但物理上住在模型文件里，且 raw-K 捕获点假设了 `Qwen3NextAttention` 内部结构（fused `qk_rmsnorm_rope_gate` 无中间量 ⇒ armed 时强制 eager 分支 :477-479；捕获点插在 `k_norm` 后 `rotary_emb` 前 :540-554，因 `ops.rotary_embedding` in-place）。**关键事实：Orca（`Qwen3_5ForCausalLM`）直接复用 `Qwen3NextAttention`（qwen3_5.py:77-81、:152-160）⇒ 钩子对 Orca 零改动生效**；异构架构才需要接口化。
2. **config.py 的 GSQ 数值默认**：`config.py:14-20` 模块头自认"defaults tuned for the GSQ production model"。GSQ 数值 7 项 = `WORKSPACE_MB 3072`(:46)、`WORKSPACE_TOKENS 262144`(:64)、`RECENT 32768`(:290)、`VIEWPORT_PAGES 55`(:356)、`QUERY_SPAN 256`(:274)、`AUTHORITY_TRAJ 2`(:200)、`SNAPSHOT_KEEP 20`(:160)；弱 GSQ（结构性假设）= `TRAJ_PREFIX 512`(:57)、`INDEX_SUBBLOCK 128`(:211)、`VIEWPORT_RECENT 16384`(:344)。全部 env 可覆盖，但默认值是 human-copied 债（必守 34）。
3. **NVFP4 页编解码**：`remat.py:47-49`（`_SCALE_GROUP=16`、E2M1 幅值表）、`:83-88`（`data_dim=head_size//2`、`scale_dim=head_size//16`）、`:241`（直接 import 引擎 `_e2m1_codes`）；`worker.py:223-233`（**RAWK armed 时硬性要求 kv-cache-dtype 以 "nvfp4" 开头**的 fail-fast 门槛）。⇒ 当前 KVMem 只能在 nvfp4 KV 上跑，其他 dtype 需页编解码器抽象。**（⭐已收口 = 步骤 097，见 §1.2-A 执行块）**
4. **唯一未门控的行为性改动**：`single_type_kv_cache_manager.py:1612-1657` 的 `MambaManager.allocate_external_computed_blocks` override（029base 无此 override，基类 :295-330 整块分配；066 新增）——任何走 mamba 组外部 KV 加载的连接器（如 OffloadingConnector）都会被改道，**与 KVMem 门控无关**。这是全仓唯一一处"行为性 hack"。（⭐095 开工判读已推翻其"hack"定性 = 上游化形状修复，见 §1.2-D 判读更新块；095 已按上游化定性落地 GO。）

**次级专用性（按需处理）**：`manager.py` 的代表组假设（`group_ids[0]` 取 block_size，8 处：377/479/597/719/989/1085/1237/1284）与单页大小页表（:80-85，混合页大小只在 stage-in decline）；`capture.py:186-211` 的 M-RoPE [3,T] 假设（vision 轴差被显式排除在阶段 1 外）；`remat.py:332-345` 的 rotary prefix 连续前缀假设（GPT-J 交错式会错，有 `rotary_dim % 16 == 0` 校验兜底）；`manager.py:158-169`/`worker.py:164-172` 的 `MambaSpec` isinstance（recurrent-state provider 抽象，非 mamba recurrent 组才需要）。

## §1 泛用化设计

### 1.1 目标判据（对标 092 的"人写数字 = 0"）
**新的滑窗混合模型进来，适配 KVMem 的人写数字 = 0**，只需三件声明式输入：
1. 模型的 attention 实现了 pre-RoPE 捕获接口（Qwen3Next 家族由注册表自动覆盖；异构架构实现一次接口）；
2. profile card 派生 KV 几何（092 轨 A 已有能力，KVMem 默认值从 card 派生）；
3. 选页编解码器（nvfp4 → 现有 codec；bf16/fp8 → 新 codec，未实现则 fail-fast 报"unsupported kv dtype"）。
验收锚 = **083 五项出口回归 + 092 autoprobe 双轨守卫在 KVMem 臂上全绿**（行为不变），终验收 = Orca 上 KVMem smoke（boot 成 + 机制自证 + needle，见 §3 步骤 100）。

### 1.2 四个工作面的抽象设计
- **A. 页编解码器（page codec）**：`remat.py` 的 NVFP4 字节几何 + `worker.py:223-233` 的 dtype 门槛收敛为接口 `PageCodec`（`page_geometry(head_size) → data_dim/scale_dim/scale_group`、`decode_rotary_prefix(page_bytes) → pre-RoPE 前缀`、`encode_rotary_prefix(...)`、`supported(kv_cache_dtype)`）。nvfp4 实现 = 现有代码搬家；bf16/fp8 codec 占位（bf16 天然 trivial：整 head 连续、无 scale 组，rotary prefix 重建同理）。`worker.py` 门槛改查 codec 注册表。

  > **⭐已执行（2026-10-04，步骤 097 GO）**：落点 = 新模块 **`vllm/v1/kvmem_workspace/codec.py`**（`PageGeometry` + `PageCodec` + `Nvfp4PageCodec` + `UnimplementedPageCodec` + `PAGE_CODECS`/`select_codec`/`codec_for_dtype`/`describe_registry`）；`remat.py` 收敛为 dtype 无关编排层（`bake_rotated_k` + `rotated_prefix_from_packed_k` + `rematerialize_page(.., codec, ..)`）；`worker.py` 门槛查表、几何由 codec 造、自检走 `RotaryPrefix`、报告加 `page_codec`。**实现相对本节设计的四处偏离（记账，均为"设计未写全"而非"改道"）**：①`page_geometry` 收 **5 个关键字**（head_size/num_heads/block_size/rotary_dim/kernel_block_size）——只给 head_size 算不出页账；`data_dim/scale_dim/scale_group` 成为 `PageGeometry` 的**字段**（codec 填），其余页账（`full_dim/chunk_bytes/page_bytes/rot_*`）全部由此派生 ⇒ 偏移算式不再知道自己是 NVFP4。②`decode_rotary_prefix` 返回 **`RotaryPrefix(values, codes, scales, max_step)`** 而不是只给浮点前缀：worker 自检的判据是"逐字节 + ≤1 量化步长"，只回浮点会丢 064 的 `max_e2m1_step` 证据账（`max_step` 由 codec 算，worker 不再写 E2M1 知识）。③**占位 codec 连 `page_geometry` 也拒绝**（设计原文称"bf16 天然 trivial"）：未经引擎实测的布局不许进现役路径算出一个数字来（必守 32/34 精神），实装者落地时一并删掉这层拒绝。④注册表把 `supported()` 与 `implemented` 分开 ⇒ "在册但未实装"（bf16/fp8）与"完全不在册"（`auto`）在 fail-fast 消息里可区分，消息由 `describe_registry()` 现算，不再硬写 dtype 名。**回归**：离线 `kvmem_remat_test.py` **35/35**（原 T0-T9 的 25 项逐字节判据一字未改 + T10 注册表 10 项）、`kvmem_viewport_test` 16/16、slot_pick/index/record_nosync/timing/stage_coverage 全绿；boot 6 支 **083 五项全 GO**，nvfp4 页账注册行逐字复现（`1640448 B (89 chunks x 18432 B, kernel block 16)`），remat 自检误差尺与 096 **逐位相同**（0.17187499487772598 / 0.15039064100710628）；负向腿 g1n2 用新变体臂 `tools/serve_gsq_kvmem_codec097.cmd`（注入项 `S097_KVDTYPE/S097_ML/S097_SW`，默认与 load1456 逐字同形）实测 **bf16 + RAWK armed 被注册表当场 ValueError 拒绝**。**两条新记账口径**：`max_byte_diff 129 vs 128` 经 npz 定量否证为"引擎原始页逐 boot 差 3.4k-9.7k 个码"（旧代码三份历史读数之间同样如此）；**测 dtype 门槛的负向腿必须先按必守 34 把 `need(L)` 算到能起服**（g1n 死于引擎容量账，未触达 `register_kv_caches`）。
- **B. config 派生化**：§0-2 的 7+3 个默认值改为派生链——优先级 env 显式 > profile card 派生（`WORKSPACE_TOKENS`←池容量派生、`VIEWPORT_PAGES/RECENT`←池 tokens 与页几何派生、`SNAPSHOT_KEEP`←mamba 组状态字节派生、`WORKSPACE_MB`←主机预算策略值）> 现值 fallback + `human-copied` WARNING（必守 34 记账转 derived）。card 缺失时不 fail（保持现行为），但 WARNING 指名道姓。

  > **⭐已执行（2026-10-04，步骤 096 GO）**：落地 = `config.py` 十函数接三层链（lazy card 加载器 + `_card_row()` + `_derived_viewport_pages()`）+ `manager.py:120` 传引擎块长 + `build_card.py` 新增 `kvmem_workspace` 段 + 新工具 `tools/profile_card/verify_kvmem_defaults.py`。**定案账 → 派生公式**（对 GSQ 逐位复现）：WORKSPACE_TOKENS = `max_position_embeddings`；WORKSPACE_MB = `⌈host_ram_total_gib/8⌉ GiB`（platform_card）；SNAPSHOT_KEEP = `⌊WS_MB×2²⁰/(snapshot_traj × mamba 组状态字节)⌋`；VIEWPORT_RECENT = `2^(⌊log2(L−gen)⌋−2)`；RECENT = 2×R；AUTHORITY_TRAJ = snapshot_traj；**VIEWPORT_PAGES = `⌊(L−gen)/p⌋ − 1 − ⌈R/p⌉ − ⌊gen/p⌋`（p = 滑窗 spec 的引擎块长 1,424，**运行时**派生——card 侧无页长，故 `viewport_retrieval_pages(page_tokens)` 由 manager 传块长，card 行 value=null/source=derived-runtime）**；QUERY_SPAN/TRAJ_PREFIX/INDEX_SUBBLOCK = 无几何式，card 显式 `human-copied` 记账。静态对照三支 13/13（fallback/GSQ card/Orca card 降级）；boot 回归 083 五项 card 态与 fallback 态全 GO + autoprobe 双轨 GO。**附带排雷**：084 在 `worker.py` RAWK 门槛读错字段名（`kv_cache_dtype`→真名 `cache_dtype`）与 `capture.py` record 计数漏 global 声明，两处在 GSQ RAWK 路径必炸、因 083 后 KVMem 臂冻结潜伏，096 回归首轮即炸即修（详见步骤 096）。
- **C. 模型钩子接口**：`qwen3_next.py` 两个钩子提为 `v1/kvmem_workspace/hooks.py` 的注册表（键 = attention 类名或 config 架构名）：`per_layer_sliding_window(config, prefix)` 与 `rawk_layer(config, prefix)` + `record 点`描述（模块、k_norm 后、rotary_emb 前、gate qkv 布局）。Qwen3Next 条目 = 现逻辑搬家；模型文件里只剩一行注册表查询。**对 Orca 零改动自证**（Qwen3NextAttention 复用 ⇒ 同一注册条目命中）。异构模型 = 实现接口 + 注册，不复制函数。

  > **⭐已执行（2026-10-05，步骤 098 GO）**：落点 = 新模块 **`vllm/v1/kvmem_workspace/hooks.py`**（`ModelKVMemHooks` 基类 + `Qwen3NextKVMemHooks`（057/061 逻辑逐字搬家，签名统一 `(config, prefix)`，layer_idx 改由 prefix 内部解析；两条 arming INFO 文本逐字保留，logger 名变 `vllm.v1.kvmem_workspace.hooks`；`record_point` 描述 = eager 分支、k_norm 后 rotary_emb 前、in-place RoPE 先 clone、gate qkv 布局）+ `MODEL_HOOKS` + `register_model_hooks`/`model_hooks_for`/`describe_registry`）；`qwen3_next.py` 收敛为一行 `model_hooks_for(type(self).__name__)`（净 −69 行，`_project_qkv_gate` 零字节触及）。**键取"attention 类名"**（本节二选一）：本树复用按类不按家族——qwen3_5（Orca）/interns2_mobius/qwen4_exp 都直接构造 `Qwen3NextAttention`，类名键一次覆盖全部。record 调用点留在模型 forward（物理位置不可搬），其接法由 `record_point` 字符串描述。**回归**：离线 `kvmem_hooks_test.py` 22/22（未注册回退 / rawk·SW 各五语义 / 异构注册清理 / qwen3_5 无子类绑定自证）+ 既存套件全绿；boot 7 支 = 083 五项全 GO（canary 逐字节、remat 误差尺与 097 逐位相同 0.15039064100710628、页账逐字）+ **Orca 注册条目命中自证 GO**（h1 RAWK arming 行 + h1b SW arming 行且 boot 健康）。**记账**：h1（RAWK armed）boot 死于 eager 分支 × Orca 运行时 rotary 实例（`RotaryEmbeddingBase.forward_cuda` 收 [3,T] dummy）；roundE1 系 runner `health()` 硬编码 8080 对 Orca（8001）失效（h1b 手工 verify 核销）。**⭐2026-10-05 步骤 100 更正与收口**：h1 死因**终定罪 = Orca config 处理链对 rope_parameters 的 mrope 键 unrecognized 剥离**（铁证行 = boot 警告 `Unrecognized keys in 'rope_parameters' for 'rope_type'='default': {'mrope_interleaved', 'mrope_section'}`；Orca 路径 Qwen3_5DecoderLayer 用 hf_text_config ⇒ 剥键 ⇒ get_rope 返回普通 RotaryEmbedding + 引擎造 [3,T] positions）——**上文"差异在 EXL3 plugin 栈运行时处理"推断作废**（orcasaq2 包全扫无 rotary 代码，与 plugin 无关）。修复 = eager 捕获分支补 [3,T]→[T] 归一（qwen3_next.py +7 行，镜像 fused 分支 ：427-430 既有逻辑，置于 record 之前 = 全链 [T] 协议；影响面 = RAWK armed × [3,T] × 非 MRoPE 实例 = Orca KVMem 臂专属，GSQ 零触及经 K0 双 boot canary 逐字节证明）；runner health() 端口口径已修（KVMEM_RUNNER_BASE env）。**下一步 = 100 Orca KVMem smoke 终验收（099 按需）**。
- **D. MambaManager 补丁门控**：`single_type_kv_cache_manager.py:1612-1657` 补 `kvmem_workspace_enabled()`（或独立 `VLLM_KVMEM_MAMBA_EXTALLOC` 门）——KVMem 臂行为不变；非 KVMem 臂走回上游整块分配。**风险（最高优先）**：生产 GSQ 不带 KVMem env ⇒ 门控后生产公共路径行为改变（从"补丁行为"回到"上游行为"）——必须先跑生产 watchdog + 8k anchor 回归证明无感，否则改为"门控反向"（补丁仅在 KVMem connector 在场时生效，按 connector 存在性而非 env 判定）。

> **⭐2026-10-04（步骤 095 开工即判读，推翻本条的门控方向）= 定案"上游化定性"**：代码对照验证成立——`MambaManager.find_longest_cache_hit`（030win 树 :1446-1520）对**本地**命中的返回形状 = `[null_block] * i` + 1 个真实 cached 块，与 066 补丁 `allocate_external_computed_blocks` 的形状（null 占位到边界 + 1 真实块）**完全同形**；而基类（029base :295-333）给外部加载分配 `cdiv(total, block_size)` 个**真实**块——对 recurrent cache 这是语义错误（state slot 是每请求一个递归状态，中间块分配了无人读写）+ 池浪费（~13.4 MiB/slot/组）。基类注释里的 issue #33775 时序约束（外部分配须在所有组 local blocks 之后）补丁已保留。⇒ **该补丁不是 KVMem 专属 hack，是 mamba 外部加载的形状修复（与引擎自己的本地命中形状自洽），KVMem 装配只是消费者之一**；env 门控会把生产切回有缺陷的基类行为（prefix 命中一次分配 ~1.5 GiB state slots @163k），方向错误。**095 执行清单（改判后）**：①改写 `:1612-1657` 注释定性（"mamba external-load shape fix, mirrors find_longest_cache_hit local-hit shape; consumers: any external KV load on a mamba group, incl. the KVMem assembly fill"），代码零行为变化；②`python tools/sync_venv.py --write v1/core/single_type_kv_cache_manager.py` 同步 venv；③生产 watchdog 一至两支（注释改动可能触发 AOT 重编译 = 首编，判态丢首支读数）；④四段式收尾。若未来发现补丁行为对某场景错误，再回门控方案并按本条风险清单走。**⭐已执行（2026-10-04，步骤 095 GO）**：清单四项全落地；watchdog 两支全 FAST 一次过门（121.13 / 121.82 tok/s，boot 110s/87s，vram 15,613 MiB）；AOT 首编预案未触发（首支即干净快档，未落必守 17 的 3.2-3.3 慢态签名 ⇒ 该注释不在 AOT 缓存键内）。本工作面（MambaManager 补丁定性）就此收口，下一步 096 config 派生（§1.2-B）。

### 1.3 明确不做的（本轮边界）
- 检索质量改良、寻优、GPU 化索引/烘焙、长稳产品化、生产切换（冻结线不动）。
- 混合页大小页表、非 mamba recurrent provider、M-RoPE vision 轴差、GPT-J 交错 rotary——登记为接口预留，不实现（无真实模型需求前不动）。
- 非 nvfp4 codec 的完整实现与验证（bf16 codec 占位接口，等有 bf16 KV 的目标模型再做实测验收）。

## §2 回归锚（每步必跑，行为不变的证明）

1. **083 五项出口**（identity canary / 紧预算 needle / 冷 262K / 串台 / 1456 装配回归）——KVMem 臂逐项 GO 不回退。
2. **092 autoprobe 双轨守卫**（GSQ 163,719 / Orca 18,904 双轨逐位）——引擎几何零扰动。
3. **生产公共路径回归**（仅步骤 095 需要）：生产 launcher watchdog + 8k anchor 快档带 116-125。
4. **无门控自证**：env 全关 boot，日志零 kvmem 活动痕迹（门控增量的"零影响"自证）。

## §3 落地顺序（每步一 commit，四段式）

| 步骤 | 内容 | 改动 | 回归 |
|---|---|---|---|
| **094**（本步） | 盘点落档 + 本设计档 + 授权边界声明 | 零代码 | 零 boot |
| **095** ✅ GO（2026-10-04） | MambaManager 外部分配补丁**上游化定性**（§1.2-D 判读更新块改判，原"门控"方案作废） | `single_type_kv_cache_manager.py` 注释改写一处（代码零行为变化） | 生产 watchdog 两支全 FAST（121.13/121.82）✅ |
| **096** ✅ GO（2026-10-04） | config.py 默认值派生化（§1.2-B） | `config.py`（十函数三层链）+ `manager.py:120`（传块长）+ `build_card.py`（kvmem_workspace 段）+ `tools/profile_card/verify_kvmem_defaults.py` | 静态对照三支 13/13（派生值逐位）+ 083 五项 card 态与 fallback 态全 GO + autoprobe 双轨 GO ✅ |
| **097** ✅ GO（2026-10-04） | PageCodec 抽象（§1.2-A） | 新 `vllm/v1/kvmem_workspace/codec.py`（nvfp4 搬家 + 注册表 + bf16/fp8 占位）+ `remat.py` 收敛为 dtype 无关 + `worker.py` 门槛查表 + 新变体臂 `serve_gsq_kvmem_codec097.cmd` | 离线 35/35（原 25 项逐字不动）+ 083 五项全 GO（6 支 boot）+ 页账/误差尺逐字·逐位复现 + bf16+RAWK 负向腿实测被拒 ✅ |
| **098** ✅ GO（2026-10-05） | 模型钩子注册表（§1.2-C） | `hooks.py` 新增 + `qwen3_next.py` 收敛为一行查询（净 −69 行） | 离线 hooks 22/22 + 083 五项全 GO（7 支，canary 逐字节）+ Orca 注册条目命中自证（RAWK 行 + SW 行且 boot 健康）✅ |
| **099**（按需） | manager 代表组/混合页大小泛化 | `manager.py` | 同上 |
| **100** ✅ GO（2026-10-05） | 终验收：Orca KVMem smoke | eager 捕获分支补 [3,T]→[T] 归一（qwen3_next.py +7 行，镜像 fused 既有逻辑）+ 工具债三修（KVMEM_RUNNER_BASE / KVMEM_PROBE_MODEL env / assembly probe makedirs）+ 新臂 `serve_orcasaq2_029_kvmem100.cmd` | boot 6 支（Orca 4 + GSQ K0 双 boot）+ watchdog 双支全 FAST；remat 0.1396484538272484 四腿逐位 + 页往返 byte-identical + 检索指纹 recent_tokens=16384 + needle 3 HIT + 紧预算 MISS；**装配三行未触发 = INCOMPLETE 挂账 O4 首项** ✅ |
| **101** ✅ GO（2026-10-05，O4 首项） | O4 首项：定罪 Orca 装配短路并打通（步骤 100 的"快照无条目"归因被日志否定） | 探针 `--reset-prefix-cache`（默认关）+ kvmem100 臂 `VLLM_SERVER_DEV_MODE=1`（`/reset_prefix_cache` = dev 路由，不开必 404）+ `manager.py` `_assembly_match` asm-miss 门控诊断行 | 日志考古定罪（k1d serve 段 hit rate 32.9% / k101g 诊断行 `num_computed=128160` = 98.4% 命中，逐 boot 漂移 0%↔98.4%）+ k101h 装配三行完整出现（matches 51,264 = 36 页，与 GSQ 083 同结构边界）+ needle HIT + remat 逐位零回归；boot 5 支（重启前 k101a/k101c OOM = 旧会话设备侧压力，重启后 4 支同参数全过）；新纪律 = 必守 36 ✅ |
| **103** ✅ GO（2026-10-06，O4 ③ 第一问） | prefix cache 共存策略第一问：轨迹槽替换语义（§5，lru 门控） | `config.py` `slot_policy()` + `worker.py`（`_slot_touch`/`_step_active`/`_evictable_victim`/`_evict_ring`/两驱逐点）+ `manager.py` removed 空键删除 + 探针 `--leg-order` + 臂 `S100_SLOTPOLICY` + 新测试 `tools/kvmem_slot_policy_test.py`（13 项） | 主判据 lru ×3（k103e/f/g）驱逐序 a→b→a2 下 A2 全 landed（matches 51,264，TTFT 154-159s）+ 阴性对照 legacy（k103a）复现 miss（A2 224.09s ≈ 全量、驱逐者 = P）+ 驱逐者换位正确（lru = flush 轨迹 + authority 2 GiB freed）+ remat 0.15039064100710628 四支逐位 + needle 12/12 + 既存 8 套件全绿 + sync_venv 2736；k103b/c/d 三连崩 = 第一版 assert 前提错误（authority 先满驱逐场景），修复 = 快照认领已交接 base + 幽灵防御（T12/T13），教训升级必守 36⑥ ✅ |
| **104** ✅ GO（2026-10-06，O4 ③ 第二问） | prefix cache 共存策略第二问：守卫 (a) 交插（§5.7，`VLLM_KVMEM_ASM_INTERLEAVE` 门控） | `config.py` `asm_interleave()` + `manager.py`（守卫交插分支 / `_assembly_plan` 记账 / `get_num_new_matched_tokens` 增量 / pending 4 元组 / `_emit_load_jobs` 页循环偏移 / 两清理点）+ 探针 `--partial-warmup-tokens` + w/i 腿 + 臂 `S100_INTERLEAVE` + 新测试 `tools/kvmem_asm_interleave_test.py`（15 项）+ runner roundL104.py | 主判据交开 ×3（k104a/b/c）i 腿 landed 逐字 `interleaves at 51264 over a 15664-token native hit (25 stored page(s) from page 11)` + needle HIT + 阴性对照 k104d 复现守卫（asm-miss local-prefix-hit，i TTFT 209.7 ≈ 全量带 218.8）+ 机制零回归（段 1 reset 腿序 a/a2 landed ×4 + remat 0.15039064100710628 四支逐位 + needle 16/16 + boot 健康）+ 收益 i 交开 159.7-160.4 vs 交关 209.7 = −24%（记录）+ 既存 9 套件全绿 + sync_venv 2736 ✅ |

## §4 风险登记

1. ~~**095 触生产公共路径**（最高）~~ **已解除（2026-10-04）**：判读改判为"上游化定性"（零代码零门控，见 §1.2-D），生产公共路径行为零变化；watchdog 两支全 FAST 回归确认。遗留尾巴 = 若未来发现补丁行为对某场景错误，再回门控方案并按本条原缓解清单走。
2. **重构引入行为漂移**：KVMem 已冻结验收，任何"顺手改"都可能破 083 判据。缓解 = 每步只动一个工作面 + 回归锚四件套。
3. ~~**Orca × KVMem 组合未知**~~ **已验收（2026-10-05 步骤 100）**：EXL3 + nvfp4 KV + KVMem 同臂 boot 成、remat/检索/needle 全绿；pinned 账与 GSQ 同构（workspace 5.00 GiB + 快照区 1.47 GiB，页账逐字）。**现场教训两条**：①Orca EXL3 的 `apply` 每次 prefill 动态 reconstruct dense 权重（shard_gemm 170 MiB 级）= GSQ kernel 不存在的设备侧工作区 ⇒ 手压池值不可跨量化栈照抄（3.4e9 池 OOM，3.1e9 实证可 boot）；②~~装配三行未触发（`self._snapshots` 对 ingest 轨迹无条目）= O4 首项定罪对象~~ **⭐101 已定罪并打通（2026-10-05）**："快照无条目"归因不成立（快照每 12 页正常捕捉登记）；真断点 = native prefix cache 部分命中（逐 boot 漂移 0%↔98.4%）触发 `_assembly_match` 的整前缀守卫，根因 = Orca 池 276,187 tokens 装得下 ingest+flush 两请求、探针 flush"覆盖全部池块"前提失效（GSQ 083 能过纯因池 163,719 < 200K 请求）；修复 = 探针 `--reset-prefix-cache` + 臂 `VLLM_SERVER_DEV_MODE=1` + asm-miss 门控诊断行；k101h 装配三行完整出现（matches 51,264 与 GSQ 083 同结构边界）+ remat 逐位零回归；纪律 = 必守 36；读数 = 步骤 101 + 详版 ⭐101。
4. **authored defaults 的派生精度**：profile card 轨 A 已双锚点逐位，但 KVMem 参数（RECENT/VIEWPORT_PAGES）是策略值不是几何值，派生 = "几何约束下取合法值"，不承诺与 GSQ 调参值同优。缓解 = 派生值仅在 card 存在时生效，env 显式值永远最高优先。**（⭐2026-10-04 步骤 096 实测：对 GSQ 本模型，派生式恰好逐位复现全部定档值——16,384/32,768/55 都从 §7.1 的 L/gen/页长几何重算出来，说明这些"策略值"在本模型上其实就是几何约束下的唯一取整解；缓解条款仍对"其他模型调参值可能不同"有效。）**

## §5 O4③ prefix cache 共存策略 —— 轨迹槽替换语义（2026-10-06 立项，设计权威节）

> 授权 = 用户 2026-10-06 指令"推进 O4③ prefix cache 共存策略"（102 判定④的关键输入已到手）。本节先定**槽替换语义**（102 指名的首个设计问题）；共存策略的另一半（守卫 (a) 的"native 命中块与装配块交插"）**明确不在本轮**（见 §5.6 边界）。

### 5.1 机制链定案（k102e 日志逐行归因，2026-10-06）

102 e 支（腿序 ingest→flush→reset→A→B→A2，`AUTHORITY_TRAJ=2`）的授权槽挤出全程：

1. ingest P（轨迹 `f09f08686187`）→ authority 槽 1（`worker.py:473`，262,144 tokens × 512 B/层）、快照环槽 1，7 条边界快照（`manager.py:953` captured 行）；
2. flush 段请求（129,453 tokens，**独立内容 = 独立轨迹** `61d85ad75299`）→ authority 槽 2、快照环槽 2，7 条快照——flush 不是 P 重发，三轨迹并存是探针结构自带的；
3. A 腿（P+tail）00:37:07 装配三行完整（matches 51,264 → issued → landed），P 环与 authority 完好；
4. B 腿（P'+tail，轨迹 `6497e3a87ba2`）首调度 no-page-run（P' 无页，正常 defer）；P' prefill 每 12 页捕获快照，worker 需要第 3 条轨迹环 ⇒ **`worker.py:1545-1562` FIFO-by-start 整环驱逐 P**（"evicting the whole ring of trajectory f09f08686187 (7 boundary(ies))"，00:40:22 WARNING）——`next(iter(self._snapshot_bases))` 拿的是最早**开始**的环，不是最不最近使用的；
5. manager 收 `removed_snapshots` 逐条 `discard`（`manager.py:960-965`），**空键不删** ⇒ `_snapshots[P]` = 空集仍占键（e 支 A2 诊断行 `snapshot_traj=3` 的出处——该字段 = `len(self._snapshots)`，语义是"见过的轨迹数"非"活跃数"）；
6. A2 腿（P+tail）00:44:06 asm-miss `no-snapshot`（45 页命中、快照空）→ 143-169s 全量。**authority 区域此时仍持有 P**（`worker.py:468` 满则拒绝新轨迹不驱逐）——同一"授权槽"概念两侧行为分叉：mamba 环挤最老、authority 拒最新。

### 5.2 现状语义的三处缺陷（设计输入）

1. **两侧不一致**：mamba 快照环 FIFO 驱逐最老开始轨迹（`worker.py:1549`），authority 区域满则拒绝新轨迹（`worker.py:468-470`）⇒ 同一轨迹可能"环没了 authority 还在"（不可装配的孤儿态）或反之（remat 无 authority，自检必炸）。
2. **无使用感知**：驱逐选择 = 按环开始时间，不看装配命中（A 腿 00:37:07 刚用 P 装配过，00:40:22 照挤）——在"服务里反复复用同前缀"的真实流量下会系统性挤掉最有装配价值的轨迹。
3. **无生命周期绑定**：轨迹槽从不因请求结束释放（`_req_trajectory` 只跟 in-flight 请求，环与 authority 长存）。

### 5.3 候选方案与裁定

| 方案 | 语义 | e 支场景 | 裁定 |
|---|---|---|---|
| A 现状 legacy | 环 FIFO-by-start 整环驱逐 + authority 拒新 | A2 仍 miss | 默认保底（行为逐字节不变） |
| B **LRU 整轨迹驱逐 + 活跃保护** | 环与 authority **同生共死**整轨迹驱逐；候选 = 无本步活动者中 touch 最老；候选空则拒绝新快照 | 驱逐 P-flush（touch 00:36:48 < P 00:37:07 装配 touch）→ A2 landed | **⭐裁定采用** |
| C 拒绝新轨迹（环也拒） | 与 authority 侧对齐成"满即拒" | A2 landed（P 快照保住） | 并入 B 作候选空时的回退；单独采用 = 早期轨迹死占、新前缀永远进不了快照区，不可取 |
| D in-flight FIFO | 驱逐选择加"无 in-flight"过滤但不看 touch | 候选含 P（A 腿已完）→ 仍驱逐 P → miss | 不解决 102 场景，弃 |
| E 扩容 AUTHORITY_TRAJ | 4 条 = +2 authority 区（每条 262,144×8,192 B ≈ 2.0 GiB host）+ 2 环（每条 keep×80.4 MiB） | A2 landed | host pinned/常驻 +5 GiB 级，撞必守 27 家族，治标不治本（N+1 条轨迹总会来），弃 |

### 5.4 设计（方案 B 落地面）

- **开关**：`VLLM_KVMEM_SLOT_POLICY` ∈ {`legacy`（默认，现状逐字节）| `lru`}；`config.slot_policy()`。
- **驱逐单位 = 轨迹整体**：mamba 快照环（边界全部上报 removed，manager 清空键）+ authority 区域（全部层的 region tensor，host 引用删除即释放）+ touch 记录，三者一起摘除。**页表/页哈希不动**（页区有自己的逐槽容量语义，被挤轨迹的页继续可被 no-snapshot 之外路径使用，A2 的 45 页命中证明两者独立）。
- **touch 事件三点记账**（worker，`time.monotonic()`）：快照捕获（`_take_snapshots` 处理到该轨迹）、装配 load（job 执行处）、页存储提交（store 路径）——三点齐 = "在捕获/在被装配/在滚动 prefill"三种活跃形态都有 touch。
- **活跃保护**：worker step 入口收集本步活跃轨迹集合 = store 数据轨迹 ∪ snapshot_requests 轨迹 ∪ load job 轨迹；驱逐候选 = 持槽轨迹 − 活跃集 − 来者自身。**候选空 ⇒ 拒绝本次快照/authority 分配**（回退 C 行为，`snapshots_refused` 计数 + 门控 WARNING；该轨迹页照存、可走全量）。并发 prefill 条数 > AUTHORITY_TRAJ 的超订场景即落此回退，语义安全（正在跑的请求不缺装配，缺也轮不到）。
- **removed 上报链不变**：整环驱逐仍走 `_removed_snapshots` → manager `discard`；附带修正 = manager 侧删空键（诊断口径从"见过的轨迹数"改"活跃快照轨迹数"，`if not available` 对空集与缺键行为等价，零行为影响）。
- **诊断**：驱逐 WARNING 保留并附候选 touch 值；拒绝计数进 worker 报告。

### 5.5 判据（步骤 103 协议，⭐2026-10-06 全部达成）

1. **主判据**：Orca kvmem100 臂 + 探针五段 a→b→a2，`lru` 模式下 A2 landed（matches = 51,264 同结构边界）≥ 3 支 boot（必守 7）——**✅ k103e/f/g 三支全 landed**（TTFT 154.3-158.6s = A 腿带 +9s 内支内漂移）；
2. **阴性对照**：`legacy`（默认）下同腿序复现 A2 no-snapshot = 默认行为不变的直接证据——**✅ k103a**（A2 224.09s ≈ 全量带，asm-miss no-snapshot，驱逐者 = P = 102 根因逐字复现）；
3. **机制零回归**：每支 needle HIT（A/A2/B）、remat 误差尺逐位（0.15039064100710628）、页账横幅逐字——**✅ 4 支全绿，boot 40.2-47.2s 健康 err=0**；
4. **驱逐可观测**：lru 支整环驱逐 WARNING 且被驱逐者 = P-flush（非 P）——**✅ 三支统一 `policy=lru, authority 2147483648 B freed`**（环 + authority 同生共死 + host 真归还）；`snapshots_refused` 超订腿未实测 = 挂账（离线 T6 钉死语义）。

**实现期事故记账**：第一版实现的防御 assert `incoming not in _snapshot_bases` 被 k103b/c/d 三连崩当场拦截——capture 每 span 都发生而快照每 12 页才发生 ⇒ **authority 表先满**，authority 满分支驱逐把 base 交接给 incoming 后，其第一条快照建环时 assert 炸（设计时"进驱逐分支的 incoming 必无 base"只对快照侧成立）。修复 = `_take_snapshots` ring None 分支先认领已交接 base + `_evict_ring` 幽灵 base 防御（base 在环不在可干净驱逐）；离线 T12/T13 钉死两场景。教训升级**必守 36⑥**（门控新路径必须连同全部触发序离线走一遍，084 双死雷同型）。崩支日志保留 `prod029_logs/kvmem_k103b|c|d/`。

### 5.6 明确不做的（本轮边界）

- ~~**守卫 (a) 不动**~~ **（2026-10-06 更新：守卫 (a) = 第二问已获用户裁定推进，设计 = §5.7；本条对 103 轮有效）**：`_assembly_match` 的"装配必须拥有整个前缀"（`manager.py:1082`）在 103 轮保持原样。
- 页区（host slots）驱逐策略不动；`trajectory_key` 定义与 `TRAJ_PREFIX` 不动；检索/视窗语义不动。
- 本设计不触 GSQ 臂（kvmem100 臂专属验证；GSQ 侧 `AUTHORITY_TRAJ` 同样生效但 GSQ 探针流量天然两轨迹，行为面不变，回归由 083 判据链兜底）。

### 5.7 O4③ 第二问 —— 守卫 (a)：native 命中块与装配块交插（2026-10-06 立项，设计权威节）

> 授权 = 用户 2026-10-06 裁定"继续推进① O4③ 第二问 = 守卫 (a) 的 native 命中块与装配块交插"。101 定罪的根因（native prefix cache 部分命中逐 boot 漂移 0%↔98.4%，命中即触发守卫 (a) 短路装配）当时用探针 `--reset-prefix-cache` 绕开；本问把"命中态下装配仍工作"做成产品语义。

**5.7.1 引擎侧已有能力（读码定案，零改动）**：

1. **调度器原生支持"本地命中 + 外部补充"合并**（`sched/scheduler.py:838-910`）：首次调度做 `_get_local_prefix_cache_hit` → connector 收到**块对齐的**本地命中数 `block_aligned_local` → connector 返回增量 `ext_tokens` → `num_computed_tokens = local + ext`；`partial_tail` 分支（:870-890）协调"子块尾巴 vs 外部加载"（外部更深则砍尾巴让 load 覆盖，反之弃外部）。
2. **KVMem connector 未覆盖 `supports_divergent_local_hybrid_hits`**（基类默认 False，`kv_connector/v1/base.py:178`）⇒ connector 收到的 `num_computed_tokens` = `get_computed_blocks` 的**全组协调统一边界**（各组命中取公共前缀），`hit_diverged` 恒 False ⇒ 交插时各组块表在该边界处形状自洽，无需处理组间分叉。
3. **外部块分配天然交插布局**：attention 组基类 `allocate_external_computed_blocks`（`single_type_kv_cache_manager.py:320-358`）= `len(req_blocks)`（本地命中块）之后 append 新真实块到 `cdiv(total, bs)` ⇒ 位置 `[hit_pages, total_pages)`；mamba 组 066 补丁（`:1612-1651`）= append `null × (n-1)` + 1 真实 state 块 ⇒ 真实块恒在 `total_pages - 1`，与 `_emit_load_jobs` 的 `position = num_pages - 1`（按**全局边界**算）对齐。
4. **SW skip 死角不受影响**：装配边界 ≤ evict_edge ≤ sw（本臂探针场景）⇒ `get_num_skipped_tokens = 0`；通用场景（快照边界 > sw）现状同样存在，不在本问扩大范围（missing_slot 防御已在）。

**5.7.2 交插语义（`VLLM_KVMEM_ASM_INTERLEAVE`，默认 `0`）**：

- **关（默认）**：守卫 (a) 逐字节（`num_computed_tokens > 0` → asm-miss `local-prefix-hit` → 0）。
- **开**：`_assembly_match` 在 `num_computed_tokens > 0` 时不再短路，改为：
  1. `start_page = num_computed_tokens // block_size`（native 命中已块对齐）；
  2. 页 run 扫描**从 `start_page` 起**（`[0, start_page)` 段由 native 哈希链负责，KVMem 页表有没有那几页无关紧要）；
  3. 快照候选 = `num_computed_tokens < b ≤ 页 run 边界`（严格大于命中边界——命中段的 recurrent state 由 native 管；无候选 = native 已盖过全部快照边界，装配无增益 → miss）；
  4. 页哈希校验段 = `[start_page, boundary)`（只校验要装配的段）；
  5. 返回**全局 boundary**（约束不变：`< prompt_len`、页对齐、`≥ block_size`）。
- **返回值改增量**：`get_num_new_matched_tokens` 返回 `(boundary - num_computed_tokens, True)`——`num_computed_tokens == 0` 时增量 = 全局 boundary = 现状值逐字节不变。
- **plan 记账**：命中时 `_assembly_plan[req_id] = (start_page, boundary)`；`update_state_after_alloc` 合并进 `_pending_loads`（legacy 无 plan 键 = `(0, num_external_tokens)` = 现状元组）；`_emit_load_jobs` 页循环 `range(start_page, num_pages)`（`num_pages = boundary // bs` 按全局边界）、mamba position 不变、`job.num_tokens = boundary`（快照行寻址不变）。worker 零改动（pages 已带具体 block_id）。
- **残键防御**：调度器 `partial_tail` 分支可把 ext 清零（`ext ≤ tail`）⇒ `update_state_after_alloc` 不记 pending、不发 job；plan 键随请求 finish 清理（与 `_pending_loads` 同点）+ boot 清空。

**5.7.3 探针（确定性交插腿）**：交插需要一个"native 命中恰好落在 KVMem 快照边界之间"的场景。101 现场的自然命中逐 boot 漂移赌不得（必守 36②）⇒ 主动构造：`ab` 子命令加 `--partial-warmup-tokens X`（默认 0 = 不跑）与 `w`/`i` 腿：

- `w`（warm-up）腿：**先 reset**（清 native 哈希，`reset_external=False` 不触 KVMem）→ 发 ingest prompt 的**前 X tokens 页对齐截断**（实跑 `X = 17,088 = 12 页`，`decode∘encode` 后断言 token 数不变）→ native 缓存 `[0, X)` 哈希链；KVMem 侧该腿与 ingest 同轨迹（TRAJ_PREFIX=512）⇒ 快照 12 页边界同值覆盖写幂等、touch P 轨迹（lru 保护）、无页存储（X < sw 无驱逐）。
- `i`（interleave）腿：**不 reset** → 发 P+tail → native 命中 w 腿的链（实跑 = 15,664 = 页 11，链在页 11 断）→ 交插装配 `[15,664, 51,264)` = **25 页 ext**（快照候选 {24,36 页} 取最大 36 页边界）——matches = 51,264 与现状 reset 装配同结构边界，但**命中段 15,664 由 native 管、装配段 35,600 由 KVMem 补** = 交插生效的结构证据（`interleaves at` 行）。
- 腿序 `ingest → w → i`；机制零回归在**同一支 boot** 先跑 103 口径 reset 腿序 `a → a2 → b`（51,264 landed）再跑交插段。

**5.7.4 判据（步骤 104，⭐2026-10-06 全部达成）**：

1. **主判据**：交插开 ×3 支，i 腿 landed + needle HIT——**✅ k104a/b/c 三支逐字 `interleaves at 51264 tokens over a 15664-token native hit (25 stored page(s) from page 11)`**（native 命中 15,664 = w 腿哈希链止于页 11，装配从页 11 补 25 页到 51,264 快照边界）；
2. **阴性对照**：交插关（默认）同腿序，i 腿复现 asm-miss `local-prefix-hit` + TTFT ≈ 全量带——**✅ k104d**（`num_computed=15664` 守卫原文，i TTFT 209.67 ≈ b 218.8）；
3. **机制零回归**：每支 reset 腿序 a/a2 landed + remat 误差尺逐位 + needle 全 HIT + boot 健康——**✅ 四支全绿**（a/a2 matches 51,264 ×3 行/支、remat 0.15039064100710628 逐位、needle 16/16（w 腿无针 = 预期）、boot 41.7-54.3s err=0）；
4. **收益可观测**：i 腿 TTFT vs a 腿——**✅ i 交开 159.65-160.42s vs 交关 209.67s = 交插省 ~49s（−24%）**；i 交开 vs a（146.2s）的 ~14s 差在支内漂移带内（a2−a = 26.8s），两者 skip 总量相同，不判系统性差异（记录不作主结论）。

**实现期事实记账**：w 腿 17,088 tokens（12 页）round-trip 精确，但 i 腿实际命中 = 15,664（页 11）——w 链最后一页未进哈希表（页对齐链在页 11 断），交插语义不受影响（`start_page` 取命中页，run 从那里扫）；四支 i 腿命中逐支同值 = 确定性成立。

**5.7.5 明确不做的（本问边界）**：页表/host 淘汰语义不动；`partial_tail` 上游协调逻辑不动；SW skip > 0 的快照边界死角（现状已有）不扩大；GSQ 臂不触；交插与 viewport 重写的组合不构造实验（viewport 分支在交插之前 return，语义互斥）。
