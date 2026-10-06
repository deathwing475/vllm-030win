# vllm-030win 执行计划 — KVMem 下的「GPU 池 × 上下文长度」扫描（GSQ + Orca，三判据：速度 / 智力 / 召回）

> **立项**：2026-10-06（步骤 107）。**指令原文**：「下一步是测试 kvmem 下，gsq 与 orca 的池的大小设置与上下文能拉到多少不影响正常使用（速度，智力方面，上下文召回）」。
> **授权登记（重要）**：这是对**已冻结的 KVMem 线的再入**。详版「硬停止条款」写明"超出口的 KVMem 工作（含缩池、重新进入）一律须单独授权"⇒ **本条指令即该授权，范围严格限定 = ①GPU 池字节扫描 ②上下文长度上限探测 ③三判据验收**；**不含**：检索算法改良、参数寻优、烘焙 GPU 化、产品化、**生产切换**（GSQ 8080 与 Orca 8001 默认配置一律不动，交付仍是变体臂 + env 门控）。
> **本文件只定协议与臂；执行 = 下一批 boot。** 步骤 107 本身零 boot、零显存。

---

## 1 三条判据的定义（每条都要给工具、门限、口径红线）

| 判据 | 主口径 | 工具 | 门限 / 红线 |
|---|---|---|---|
| **速度** | ①8k decode anchor 中位；②**90% L 档** decode（用户判据②，最严）；③长档**冷 prefill** TTFT 与 tok/s；④KVMem 页步中位（装配/入库开销） | `tools/anchor_longctx.py`（`--model` 可切 Orca）+ `tools/step090_orca_speed.py`（带外 1 Hz 分型）+ `tools/kvmem_time_budget.py`（页步台账）+ `tools/prod_watchdog.ps1`（GSQ 判快门 105） | **每档 ≥3 boot + 分型**（`utilization.memory`/`power.draw`/温度/`pcie.link.*` = 必守 7/25）；**断言"慢"之前必须先报池字节数**（必守 33）；**并发倍数不是健康证据**（090 的 2.65x 比 1.13x 慢 16 倍）；GSQ 快态须 ≥106 才保慢态 ≥85（041）；首编 boot 不入样本（必守 17，093 实测 26 倍塌落签名） |
| **智力（输出质量）** | **needle 多深度 = 主判**；PPL 只作粗筛；多轮连贯性 | `tools/kvmem_assembly_probe.py`（needle 三深度 + `--nonce`）+ `tools/orca_nvfp4_probe.py`（Orca chat/needle）+ `tools/anchor_ppl.py` + `tools/soak_prod.py` / `soak_orca.py` | PPL **非位精确**（同配置两 boot 差 1e-3 ⇒ 不得当判等，主判 = needle，必守 20）；**Orca 探针 `hit` 字段不可信**（chat 三发统一问"含 needle 码吗" ⇒ 报正确性必须附文本，必守 31①）；`build_prompt` **无视 `--tokens`** 实发更长 ⇒ 短档必 400，报数须附服务端 `usage.prompt_tokens`（31②）；思考型输出预算：16k 档 needle 曾假阴（32 token 还在 `<think>`，须 ≥64） |
| **上下文召回** | ①**窗外 needle 答出**（074/078 协议）；②装配 landed（`matches` / `issued` 页账）；③读回校验 0 mismatch；④命中率**必须报采样次数** | `tools/kvmem_viewport_probe.py` + `tools/kvmem_assembly_probe.py` + `tools/kvmem_stage_coverage_test.py`（逐槽 × 逐组）+ 臂 `VLLM_KVMEM_BAKE_VERIFY=1` | 命中率假阴性：单次 `--max-tokens 32` 会把命中读成 MISS ⇒ **同 boot 二次采样**（`--serve-ignore-eos`，必守 25③）；**视窗探针之前不许插任何会入库的节拍请求**（`AUTHORITY_TRAJ=2` 只有两条名额，081 b5 因此假 MISS）；装配没触发先读 **asm-miss 诊断行**（101）；**native prefix cache 必须确定性清除**：探针 `--reset-prefix-cache` + 臂 `VLLM_SERVER_DEV_MODE=1`（必守 36①②，命中量逐 boot 漂移 0%↔98.4% 赌不得） |

---

## 2 前置卡点（本会话读码 + 实测记录查出，**必须先解决才谈扫描**）

1. **Orca 的 KVMem 臂是 `--enforce-eager`**（`tools/serve_orcasaq2_029_kvmem100.cmd:116`；该臂注释自陈"O1 基线是 eager，图兼容属 O4 范围"）。⇒ **"速度不受影响"这条判据在 Orca 侧目前没有可比基准**：105 定档是 FULL 图 78-87 tok/s，而 KVMem 臂只能读 eager（MTP eager 参照 = 37.6）。**要么另开一条 Orca KVMem 图兼容腿（100 只修了 eager 捕获分支的 `[3,T]→[T]` positions 归一，fused/图分支未验），要么把 Orca 的速度判据限定为"同 eager 口径内的相对不劣化"**。**这是本实验最大的方法学缺口，需要用户裁定按哪种口径走。**
2. **GSQ 侧 `W=163,072 + 投机` 装不下**（075 实测：需 3.24 GiB > 池给的 3.15 GiB，且**降 `--max-model-len` 无效**——需求由滑窗 `SW_WINDOW` 钉住）⇒ 扫描起点只能落在 **`SW_WINDOW=131,072`**（现成臂 `serve_gsq_kvmem_load1456.cmd:91` 就是这个档，且 077/078 已在带上跑通 needle 判据）。
3. **改 `SW_WINDOW` 会换页长 ⇒ 换 AOT 缓存键**（075：W 131,072 时页长 1456；W 变 ⇒ 页长变 ⇒ mbt 必须同步）⇒ 每个 W 档都必须 **双 boot 或走 watchdog**，且 `mbt ≥ 页长 + num_spec`（077 定罪：投机从预算里扣 `draft_slots=2`，`mbt = 页长` 会把首块裁到 0 token ⇒ 调度器**静默 break、零日志**，单 boot 162,151 次）。
4. **缩 GPU 池 ≠ 缩 host pinned**：`VLLM_KVMEM_WORKSPACE_MB`（现 3072）与 raw-K 权威区（2.0 GiB）、快照区都是**锁页 host 分配，吃 WDDM 提交预算**（必守 27，083 的 `LOAD=1 boot OOM` 根因）⇒ 池压到 1.0e9 以下若撞 boot OOM，**先查 pinned 分配横幅与 driver 级报错（没有 "Tried to allocate" 就不是 vLLM 的容量账）**，旋钮 = `SNAPSHOT_EVERY_PAGES` / `SNAPSHOT_KEEP`（`keep × every ≥ 边界存活期`）。
5. **缩池很可能把 103 的 `authority 先满驱逐` 场景逼出来**（k103b/c/d 三连崩 = 同一场景，修复已并入 103）⇒ 扫描必须带 **`VLLM_KVMEM_SLOT_POLICY=legacy` 与 `lru` 各一腿**：若"只有开 lru 才缩得下去"，那本身就是结论（lru 从 O4 研究项变成缩池前提）。`VLLM_KVMEM_ASM_INTERLEAVE` 在扫描期**一律关**（隔离变量），交插只在末尾做一次对照腿。

---

## 3 变量矩阵（臂与旋钮都是现成的；每批 ≤6 支 boot）

**固定项（两栈一致）**：`WORKSPACE=1` + `RAWK=1` + `AUTHORITY=1` + `LOAD=1`（装配腿）+ `BAKE_VERIFY=1`（读回校验）+ `SLOT_PICK=score`（081 交付，针页余量 2→7）+ `ASM_INTERLEAVE=0` + 全 CUDA 图（GSQ 侧）+ prefix caching + `VLLM_SERVER_DEV_MODE=1`（清缓存路由）。

| 批 | 栈 / 臂 | 扫什么 | 档位 | 每档 boot | 出口判据 |
|---|---|---|---|---|---|
| **A** | GSQ `serve_gsq_kvmem_load1456.cmd`（旋钮 `S082_KVMB` / `S082_MBT` / `S082_WSMB`；`SW_WINDOW=131,072`、`--max-model-len 200,704`） | **GPU 池下压阶梯**（目标 = 阶段 3 原命题"缩池换余量"对冲 049 悬崖） | 3.4e9（基准，容量 131k 档已知）→ **2.6e9 → 2.0e9 → 1.6e9 → 1.2e9** | ≥3 + 分型；基准档可复用 077/078/083 读数但要同臂重跑 | 8k decode 不掉出带（GSQ 图带 ~74-90 参照 093 与生产 116-125 分开报）；needle 三深度全 HIT；页账 / `baked copies` / read-back 0 mismatch 逐字复现；`prod_headroom_check` 被挤 MiB 下降 = 收益侧记账 |
| **B** | A 的同臂复制成 `_w098304.cmd` / `_w065536.cmd`（**新文件名，不覆盖旧证据**） | **视窗 W × 池 的二维下界**（W 越小越省 GPU，但窗外召回余量越小） | W ∈ {131,072 / 98,304 / 65,536} × 池 ∈ {2.6e9 / 1.6e9}，mbt 随页长同步（页长 = cdiv 后重算，**先算再 boot**） | 双 boot（换 AOT 键） | 窗外针（token 125,370 系）在缩 W 后仍答出；**记录 `rank_by_logit` 与时间序槽位余量**（079：针页是 86 不是 88，槽按时间序填，余量只剩 2 ⇒ 缩 W/缩 TOPN 会直接吃掉余量） |
| **C** | Orca `serve_orcasaq2_029_kvmem100.cmd`（旋钮 `S100_KVMB` / `S100_ML` / `S100_MBT` / `S100_WSMB` / `S100_LOAD` / `S100_SLOTPOLICY` / `S100_PAGES`；`SW_WINDOW=65,536`） | **池 × L**（Orca 侧悬崖与 KVMem 的相互作用） | 池 {8e8 / 1.0e9 / **1.4e9 = 093 实测安全上限**} × L {16,384 / 49,152}（093 已证 eager 之外 L 到 49k 无独立速度效应，须在 KVMem 臂复验） | ≥3 + 分型 | 悬崖位置是否因 KVMem 的 host 侧开销而**左移**（1.4e9 是纯 serve 的读数，KVMem 臂多了锁页区 ⇒ 预期更早）；needle 三深度 + 装配 landed（101 口径 `matches 51,264` 随池变化要重算页账，别套 GSQ 数） |
| **D** | C 的同臂 | **EXL3 手压池的重建工作区警告**（100 实测记的雷） | 压到 0.8e8 后跑一次 LOAD 腿 | 1-2 | 若装配失败，**先确认工作区是否还在**（100：EXL3 手压池须留重建工作区）；这是阴性腿的合法性检查（必守 35①：阴性腿要先证明 boot 走得到目标代码） |
| **E** | A/C 各一腿 | **`SLOT_POLICY=lru` 是否成为缩池前提**（第 5 条卡点） | legacy vs lru 同池同 L 对照，同 boot 同态（必守 36③） | ≥3 ×2 | lru 下 A2 腿 landed（103 口径 matches 51,264）且 legacy 对照复现 miss ⇒ 结论 = "要缩池必须先开 lru" |

**不在本矩阵内（明确排除）**：`VLLM_KVMEM_VIEWPORT=1` 大视窗路线（068-071 已判死并回滚，勿重开）、检索粒度/`TOPN` 寻优（062 判 NO-GO + 079 记的耦合陷阱）、GPU 化烘焙、任何生产 launcher 改动。

---

## 4 出手前必须算的账（纸面先过，别用 boot 试错）

- **每请求需求** `need(L) = a + b·L`（必守 34）：GSQ 在 ssm bf16 + G=8 下 attn block 1456、统一页 1,677,312 B、`bytes_per_block` 13,418,496；Orca 侧**用引擎自己的横幅重读**（089 的 block 1456 是两支 `text_config` 逐字节相同的**结果**，不是可免验前提；换 W/池/spec 档都要重新读）。
- **容量式** `blocks_per_req = cdiv(L,block)×cdiv(16,G) + 4·cdiv(48,G) + sw·cdiv(5,G) ≤ num_blocks − 1`，其中 `sw = cdiv(2047+2×mbt, block)+1` ⇒ **mbt 与容量耦合**，压池时先看这条会不会先破。
- **`GPU KV cache size: N tokens` 是 `max_concurrency × max_model_len`，不是字节** ⇒ 禁止"池字节 ÷ N"（091 的假异常）。
- **pinned 预算表**：`WORKSPACE_MB` + 权威区 2.0 GiB + 快照区（`SNAPSHOT_KEEP × SNAPSHOT_EVERY_PAGES`）逐档列出来，和独显/主机内存一起对照（本机 **23.1 GiB RAM / 16 GB 独显**；跑过推理后独显余量仅 ~492-513 MiB）。

---

## 5 预算、安全与停手条件

- **每批 ≤6 支 boot**；一批读完再开下一批；臂与生产**一次只跑一个**（生产当前已停，保持停）。
- 改被图编译引用的文件 = 换 AOT 键 ⇒ 双 boot / watchdog，**禁止单次冷 boot 判回归**（必守 17）；只改 `kvmem_workspace/` 不触发（097/098 两证）。
- kill 后**先等独显回落 <800 MiB** 再起服（必守 18）；kill 一律 `tools/kill_vllm_orphans.ps1`，起服一律 `_tmp_line_b/boot_keep.py`（只吃 `.cmd`）。
- 任何"装不下/NO-GO"结论出手前，**先逐条 diff 同卡已跑通的 launcher**（必守 32）；结论限定到"这支臂"而不是"这张卡"。
- 判据没达成写 **INCOMPLETE**，不许用"机制走通"替代（必守 20）。
- 收尾**只记录生产状态、不恢复**（必守 30）。

---

## 6 交付与落档

- 交付形态永远是：**变体 arm（新文件名）并存、生产默认不动、新行为 env 门控（默认关）**。
- 每批一 commit：步骤文档四段式（107 = 立项与本计划；108 = 批 A …）+ 日志 + 提示词「状态一句话/头名」两处 + 详版新增纪律（若长出新的"必守 N"，先写详版再回提示词加一行）+ 台账 `prod029_logs/stepNN_boots.json` + 证据目录 `prod029_logs/{kvmem_pool,orca_pool}/…`。
- **待用户裁定的两件事**（不裁就按保守默认走）：
  1. **Orca 速度口径**：= 另开图兼容腿（要花 boot，且 100 只修了 eager 归一，图/fused 分支可能撞新东西）／还是限定为"eager 同口径相对不劣化"？**默认 = 后者**（少花 boot、不扩面）。
  2. **GSQ 侧是否同时补"纯 serve（不开 KVMem）的池下压曲线"**？035/048 只测过池**上探**，而 Orca 已证明池字节是速度开关 ⇒ 不做这条就没有"KVMem 换来了什么"的对照。**默认 = 做**（批 A 之前先跑 3 支纯 serve 压池，作为对照组）。
