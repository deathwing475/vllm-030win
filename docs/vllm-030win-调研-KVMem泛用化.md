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
3. **NVFP4 页编解码**：`remat.py:47-49`（`_SCALE_GROUP=16`、E2M1 幅值表）、`:83-88`（`data_dim=head_size//2`、`scale_dim=head_size//16`）、`:241`（直接 import 引擎 `_e2m1_codes`）；`worker.py:223-233`（**RAWK armed 时硬性要求 kv-cache-dtype 以 "nvfp4" 开头**的 fail-fast 门槛）。⇒ 当前 KVMem 只能在 nvfp4 KV 上跑，其他 dtype 需页编解码器抽象。
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
- **B. config 派生化**：§0-2 的 7+3 个默认值改为派生链——优先级 env 显式 > profile card 派生（`WORKSPACE_TOKENS`←池容量派生、`VIEWPORT_PAGES/RECENT`←池 tokens 与页几何派生、`SNAPSHOT_KEEP`←mamba 组状态字节派生、`WORKSPACE_MB`←主机预算策略值）> 现值 fallback + `human-copied` WARNING（必守 34 记账转 derived）。card 缺失时不 fail（保持现行为），但 WARNING 指名道姓。

  > **⭐已执行（2026-10-04，步骤 096 GO）**：落地 = `config.py` 十函数接三层链（lazy card 加载器 + `_card_row()` + `_derived_viewport_pages()`）+ `manager.py:120` 传引擎块长 + `build_card.py` 新增 `kvmem_workspace` 段 + 新工具 `tools/profile_card/verify_kvmem_defaults.py`。**定案账 → 派生公式**（对 GSQ 逐位复现）：WORKSPACE_TOKENS = `max_position_embeddings`；WORKSPACE_MB = `⌈host_ram_total_gib/8⌉ GiB`（platform_card）；SNAPSHOT_KEEP = `⌊WS_MB×2²⁰/(snapshot_traj × mamba 组状态字节)⌋`；VIEWPORT_RECENT = `2^(⌊log2(L−gen)⌋−2)`；RECENT = 2×R；AUTHORITY_TRAJ = snapshot_traj；**VIEWPORT_PAGES = `⌊(L−gen)/p⌋ − 1 − ⌈R/p⌉ − ⌊gen/p⌋`（p = 滑窗 spec 的引擎块长 1,424，**运行时**派生——card 侧无页长，故 `viewport_retrieval_pages(page_tokens)` 由 manager 传块长，card 行 value=null/source=derived-runtime）**；QUERY_SPAN/TRAJ_PREFIX/INDEX_SUBBLOCK = 无几何式，card 显式 `human-copied` 记账。静态对照三支 13/13（fallback/GSQ card/Orca card 降级）；boot 回归 083 五项 card 态与 fallback 态全 GO + autoprobe 双轨 GO。**附带排雷**：084 在 `worker.py` RAWK 门槛读错字段名（`kv_cache_dtype`→真名 `cache_dtype`）与 `capture.py` record 计数漏 global 声明，两处在 GSQ RAWK 路径必炸、因 083 后 KVMem 臂冻结潜伏，096 回归首轮即炸即修（详见步骤 096）。
- **C. 模型钩子接口**：`qwen3_next.py` 两个钩子提为 `v1/kvmem_workspace/hooks.py` 的注册表（键 = attention 类名或 config 架构名）：`per_layer_sliding_window(config, prefix)` 与 `rawk_layer(config, prefix)` + `record 点`描述（模块、k_norm 后、rotary_emb 前、gate qkv 布局）。Qwen3Next 条目 = 现逻辑搬家；模型文件里只剩一行注册表查询。**对 Orca 零改动自证**（Qwen3NextAttention 复用 ⇒ 同一注册条目命中）。异构模型 = 实现接口 + 注册，不复制函数。
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
| **097** | PageCodec 抽象（§1.2-A） | `remat.py`/`worker.py` 拆 nvfp4 codec + 注册表 + bf16 占位 | 083 五项（nvfp4 路径逐字节回归）+ remat 离线单测 18/18 |
| **098** | 模型钩子注册表（§1.2-C） | `hooks.py` 新增 + `qwen3_next.py` 收敛为一行查询 | 083 五项 + Orca 臂 boot 自证（同一注册条目命中） |
| **099**（按需） | manager 代表组/混合页大小泛化 | `manager.py` | 同上 |
| **100** | 终验收：Orca KVMem smoke | 零代码（launcher 变体） | boot 成 + 机制自证 + needle；结果决定 O4 是否展开 |

## §4 风险登记

1. ~~**095 触生产公共路径**（最高）~~ **已解除（2026-10-04）**：判读改判为"上游化定性"（零代码零门控，见 §1.2-D），生产公共路径行为零变化；watchdog 两支全 FAST 回归确认。遗留尾巴 = 若未来发现补丁行为对某场景错误，再回门控方案并按本条原缓解清单走。
2. **重构引入行为漂移**：KVMem 已冻结验收，任何"顺手改"都可能破 083 判据。缓解 = 每步只动一个工作面 + 回归锚四件套。
3. **Orca × KVMem 组合未知**：EXL3 权重 + nvfp4 KV + KVMem 工作区从未同臂跑过；池字节与 pinned 快照区在 Orca 的账未算（必守 27 的 WDDM 提交预算风险在 3.4e9 池下更紧）。缓解 = 100 步 smoke 按必守 27 查 pinned 横幅。
4. **authored defaults 的派生精度**：profile card 轨 A 已双锚点逐位，但 KVMem 参数（RECENT/VIEWPORT_PAGES）是策略值不是几何值，派生 = "几何约束下取合法值"，不承诺与 GSQ 调参值同优。缓解 = 派生值仅在 card 存在时生效，env 显式值永远最高优先。**（⭐2026-10-04 步骤 096 实测：对 GSQ 本模型，派生式恰好逐位复现全部定档值——16,384/32,768/55 都从 §7.1 的 L/gen/页长几何重算出来，说明这些"策略值"在本模型上其实就是几何约束下的唯一取整解；缓解条款仍对"其他模型调参值可能不同"有效。）**
