# vllm-030win 执行计划 — KVMem 下的「GPU 池 × 上下文长度」扫描（GSQ + Orca，三判据：速度 / 智力 / 召回）

> **立项**：2026-10-06（步骤 107）。**指令原文**：「下一步是测试 kvmem 下，gsq 与 orca 的池的大小设置与上下文能拉到多少不影响正常使用（速度，智力方面，上下文召回）」。
> **v2 重写**：2026-10-18（步骤 108A）。v1 的批 A/C/D 矩阵经读码与纸面重算后**不可执行**（见 §0），本版改为**联合 (池, W) 可行域 + 双形态探针 + 粗扫/定档两段**。授权范围不变：仅池字节扫描 / 上下文上探 / 三判据验收；不含检索算法改良、参数寻优、烘焙 GPU 化、产品化、生产切换（GSQ 8080 与 Orca 8001 默认配置一律不动，交付仍是新臂 + env 门控）。
> **v1 原文与作废矩阵**保留在 §8，供追溯；冲突时以本版 §1-§7 为准。

---

## 0. v2 判定：v1 为什么不可执行

1. **GSQ 固定 W=131,072 / L=200,704 时，池降到 2.6e9 就起不来。**
   - 3.4e9 实测 `234,000 tokens / Maximum concurrency 1.17x`（077/078/083；日志 `prod029_logs/step078_b1_spec_repro.out.log:97`）。
   - 启动容量检查按「至少一个请求」算（`vllm/v1/core/kv_cache_utils.py:852-888`）；1.17x × 2.6/3.4 = 0.895x < 1 ⇒ 直接 `ValueError`。075 的拒起原文模板在 `实验步骤文档.md:961-963`。
   - 需求由**滑窗 W** 钉住，不由 `max-model-len` 钉住（075：L 从 262,144 降到 200,704，需求仍是 3.24 GiB）⇒ 低池必须**同时缩 W**，只缩 L 无效。
2. **Orca 批 C 的池 8e8/1.0e9/1.4e9 在 W=65,536 下同样低于 1 并发。**
   - 3.1e9 实测 `276,187 tokens / 2.11x`（100/101；`prod029_logs/stepe1_k1a_arm.out.log:69`）。
   - 1.4e9 折算约 0.95x，1.0e9/8e8 更低；批 D 的 0.8e8 更不可能。W=65,536 的低池地板约 1.6e9。
3. **Orca 批 C 的 L=16,384/49,152 < W=65,536，滑窗淘汰根本不触发。**
   - `single_type_kv_cache_manager.py:685-722,1124-1159`：`skipped <= 0` 直接返回；增量存页仍会发生（`manager.py:600-706`），但淘汰/装配/视窗召回路径测不到。
   - 要测 KVMem 交互，必须 **L > W**（用 L=131,072）或把 W 压到 L 以下。
4. **方法学缺陷一：anchor 的 needle 固定在 85% 深度，单深度。**
   - `tools/anchor_longctx.py:45-55,112-123`。L=200,704、W=131,072 时 85% = 170,598，落在滑窗内 ⇒ 测的是 recency，不是 KVMem 召回。
   - 窗外召回只能用 `tools/kvmem_viewport_probe.py`（074/078 协议），且需要 **VIEWPORT=1**；装配臂（VIEWPORT=0）的 066 头注释明确写「窗外针仍答不出」。
5. **方法学缺陷二：8k anchor 不触发淘汰**，量到的只是「KVMem 开着但空闲」的开销；页步/入库开销必须用长请求 + `tools/kvmem_time_budget.py`。
6. **臂与计划不一致**：GSQ load1456 臂未设 `VLLM_SERVER_DEV_MODE=1`（无法确定性清 prefix cache，必守 36②）；两臂默认 `VLLM_KVMEM_SLOT_PICK=time` 而计划要求 `score`（081 交付，针页余量 2→7）。
7. **AOT 机制记错**：页长公式是 `16*cdiv(mamba_raw, 16*1152)`（`vllm/platforms/interface.py:881-949`），只依赖 spec N 与模型几何，**不依赖 W**。旧注释「W=163,072 → 页 1424；W=131,072 → 页 1456」是把无 spec 臂与 +spec 臂的差异误记成 W 的效应。GSQ +spec N=2 在所有 W 都是页 1456、`mbt >= 1458`；Orca 无 spec 页 1424。
   - 改 L 会触发首编（093：每次 L 变更首编 3.29/3.29/3.255 tok/s）；只改 W 是否重编未被证明。跑臂时以日志 `Directly load AOT compilation` / `Compiling a graph` 判定，首编 boot 不入样本。

**调整方向**：纸面先走联合 (池, W, L, mbt) 可行域 → 粗扫定位边界（每点 2 boot）→ 定档点 ≥3 boot 全判据。粗扫只定位，不作速度断言。

---

## 1. 三判据与「可用上下文」

| 判据 | 主口径 | 工具 | 门限 / 红线 |
|---|---|---|---|
| **速度** | ①8k decode anchor 中位；②90% L 档 decode；③长档冷 prefill TTFT；④KVMem 页步中位（装配/入库开销） | `tools/anchor_longctx.py`（`--model` 切 Orca）+ `tools/step090_orca_speed.py`（Orca 带外 1 Hz 分型）+ `tools/kvmem_time_budget.py` + `tools/prod_headroom_check.ps1` | 粗扫单 boot 只记状态；**速度断言只在定档点、≥3 boot + 分型**（必守 7/25）；报「慢」前先报池字节（必守 33）；并发倍数不是健康证据（090 的 2.65x 比 1.13x 慢 16 倍）；首编 boot 不入样本（必守 17，093） |
| **智力** | **needle 多深度 = 主判**；PPL 只粗筛；多轮连贯性 | `tools/kvmem_viewport_probe.py`（窗外）+ `tools/kvmem_assembly_probe.py`（装配）+ `tools/orca_nvfp4_probe.py`（Orca `--depths`）+ `tools/anchor_ppl.py` | PPL 非位精确（同配置两 boot 差 1e-3 ⇒ 不得当判等，必守 20）；Orca 探针 `hit` 字段不可信（必守 31①）；`build_prompt` 无视 `--tokens`（必守 31②）；思考型输出预算 ≥64 token |
| **召回** | ①**窗外 needle 答出**（074/078 协议，**VIEWPORT=1**）；②装配 landed（`matches`/`issued`）；③读回校验 0 mismatch；④命中率必须报采样次数 | `tools/kvmem_viewport_probe.py` + `tools/kvmem_assembly_probe.py` + `tools/kvmem_stage_coverage_test.py` + `VLLM_KVMEM_BAKE_VERIFY=1` | 窗外 MISS 归「召回边界」，不归「智力失败」；装配没触发先读 `asm-miss` 诊断行（101）；native prefix cache 必须确定性清除（探针 `--reset-prefix-cache` + 臂 `VLLM_SERVER_DEV_MODE=1`，必守 36①②）；视窗探针之前不许插任何会入库的节拍请求（081 b5） |

**可用上下文**（本计划的交付主指标，必须单列）：

> 在给定 (池, W) 下，**in-window needle**（anchor 85%）与 **out-of-window needle**（viewport 中段 0.45/0.55/0.65）全 HIT、8k 速度在带内、装配 read-back 0 mismatch 的**最大 L**。

它通常小于 `max-model-len`。**接受长度**（引擎能收多长 prompt）与**可用上下文**（针全答得出）是两个数，报告里必须分开；不得用接受长度冒充可用上下文。

**双形态探针**（v2 的核心修正）：

| 形态 | 环境 | 测什么 | 说明 |
|---|---|---|---|
| main | `VIEWPORT=0` + `LOAD=1` | 速度 + 装配（`matches`/`issued`/`landed`/read-back）+ 窗内针 | 装配臂不能答窗外针（066 头注释） |
| recall | `VIEWPORT=1` + `LOAD=0` | 窗外针（074/078 协议） | 072+ **单坐标**视窗；与 068-071 双坐标大视窗（已判死回滚）不是同一路线，别混为一谈 |

---

## 2. 前置卡点（v2，全部在 Phase 0 处理）

1. **联合可行域**：任何点估算并发 < 1.05 不许直接 boot；起不来就把 W 降一档（98,304→94,208、65,536→61,440、49,152→45,056），边界记台账。公式与表见 §4。
2. **Orca 的 L 必须 > W**：主扫用 L=131,072 / W=65,536；低池边界用 W=49,152 / L=98,304。L=16,384/49,152 的腿删除（不触发淘汰）。
3. **页长不随 W 变**：GSQ +spec N=2 页 1456、mbt≥1458；Orca 无 spec 页 1424、mbt=1456（≥页长即可）。不再按 W 重算 mbt；仍按「首编 boot 不入样本」执行。
4. **pin host ≠ GPU 池**：`WORKSPACE_MB`（现 5120）+ 权威区 2.0 GiB + 快照区（`SNAPSHOT_KEEP × SNAPSHOT_EVERY_PAGES`）都是锁页 host，吃 WDDM 提交预算（必守 27，083）。池压到 1.0e9 以下若 boot OOM，先查 pinned 横幅与 driver 级报错（没有 `Tried to allocate` 就不是 vLLM 容量账），旋钮 = `SNAPSHOT_EVERY_PAGES`/`SNAPSHOT_KEEP`。
5. **缩池可能逼出 103 的 authority 先满驱逐**：批 E 的 legacy/lru 对照保留；「只有开 lru 才缩得下去」本身即结论。`ASM_INTERLEAVE` 扫描期一律关，末尾最多一支对照。
6. **臂修复**：新臂补 `VLLM_SERVER_DEV_MODE=1`；`SLOT_PICK` 默认 `score`；新臂文件名不覆盖旧证据；`.cmd` 必须 CRLF（`tools/make_kvmem108_arms.py` 生成并断言）。
7. **Orca 速度口径**：eager 同口径相对不劣化（默认）；不得与 Orca 生产 FULL+DFlash2 的 78-87 tok/s 直接比。另开图兼容腿属扩面，需单独授权。

---

## 3. 变量矩阵（v2）

**固定项（main 臂）**：`WORKSPACE=1` + `RAWK=1` + `AUTHORITY=1` + `LOAD=1` + `BAKE_VERIFY=1` + `SLOT_PICK=score` + `ASM_INTERLEAVE=0` + 全 CUDA 图（GSQ）+ prefix caching + `VLLM_SERVER_DEV_MODE=1`。
**recall 臂**：同上但 `VIEWPORT=1` + `LOAD=0`，其余不变。

| 批 | 栈 / 臂 | 扫什么 | 档位 | boot 预算 | 出口判据 |
|---|---|---|---|---|---|
| **108A** | 零 boot | Phase 0：纸面重算 + 新臂 + runner + 离线回归网 + 止损缺口 | — | 0 | 论文表无 <1.05 的点；三条臂 `--check` 过；`kvmem_capture_op_test.py` 仅 2 项既存 FAIL；page.py apply↔revert 往返过 |
| **108B-A1** | GSQ KVMem main + recall | 联合前沿上段 | 3.4e9/W=131,072；2.6e9/W=98,304；2.0e9/W=65,536 | 6（每点 recall + speed 各 1） | 起服横幅与纸面并发一致（±5%）；recall 点：窗内针 HIT + 窗外 3 深度 HIT/MISS 记边界 + 页步台账；speed 点：8k 中位 + 90% L TTFT/decode |
| **108B-A2** | 同上 | 下段/边界 | 1.6e9/W=49,152（必做，**speed-only**：视窗预留 32,768+16,384+sink 已超 W ⇒ 0 个检索页，recall 结构性 INCOMPLETE）；1.2e9/W=32,768（可选，同 speed-only） | 2 | 8k 中位 + 90% L TTFT/decode + headroom；recall 写 INCOMPLETE（不是 MISS） |
| **108B-A3** | GSQ 纯 serve `serve_gsq_serve108.cmd` | 同池对照 | 3.4e9/L=163,072（复用 046/105 读数）；2.6e9/L=117,000；2.0e9/L=85,000 | 4（两个新点各 compile+sample） | 同池同 L 下 8k/90% L 对照；回答「KVMem 换来了什么」：KVMem 接受 200,704 但窗外靠检索，纯 serve 接受较短 L 且全程精确 |
| **108C** | Orca KVMem main + recall | 池 × L（eager） | 3.1e9/W=65,536/L=131,072；2.0e9 同；1.8e9 同（拒起则 1.9e9） | 6 | 起服横幅 ±10%（公式对 Orca 约低 10%）；L>W 触发淘汰；窗外针 3 深度；速度只做 eager 相对不劣化 |
| **108D** | 定档点 | 三判据全量 | GSQ 与 Orca 各一个推荐点 | 6（每栈 3） | 8k 中位 + 分型；90% L decode；冷 prefill TTFT；页步中位；窗外 3 深度；装配 `matches`/`issued`/`landed` + read-back 0 mismatch；headroom |
| **108E**（可选） | A/D 同臂 | lru vs legacy | 同池同 L | 6（3×2） | legacy 失败 + lru landed ⇒ 「缩池须先开 lru」；两者都失败 ⇒ 与 lru 无关 |
| **108F**（可选） | 1.4e9/W=49,152/L=98,304 | Orca 低池边界 | — | 2 | 只在需要证明低池可行性时做 |

**108B-A1 实测修订（2026-10-18，写入计划供后续批次执行）**：

- **一个 viewport 深度一支干净 boot**。同一 boot 内连跑多深度会被 workspace 残留污染：一次被中断的 ingest 留下 54 页轨迹后，三腿 0.45/0.55/0.65 全 MISS，且每次 rewrite 的 page table 只有 54 页（078 干净态 = 136 页）；干净 boot 单深度立即恢复 136 页并 HIT。多深度的「可用上下文」判据须每深度一支干净 boot（A1 只采信 0.65；0.45/0.55 挪到 108D）。
- **LOAD=1 的 WSMB = 4096**。WSMB 5120 + 快照区 3.00 GiB 在本机当前 pinned 状态装不下（40×1,677,312 B ×8/×12 失败，随后 `KVBlockZeroer` OOM；无 `Tried to allocate` = 必守 27）。WSMB 4096 三点全部 boot 成功（160 槽 > 200k prompt 的 137 页）。
- **A1 三点结果**：3.4e9/W131072/p55 / 2.6e9/W98304/p32 / 2.0e9/W65536/p10，横幅 1.17/1.13/1.17x（纸面 ±5% 内），0.65 针全 HIT（TTFT 106.3/63.3/28.5 s，baked 55/32/10，read-back 0 mismatch），8k 单 boot 77.95/92.43/115.29，177,959 档 steady 59.12/77.69/106.33，headroom 515/1,075/1,456 MiB。8k 差值不作池因果（83-129 连续谱）；90% L 趋势与窗口缩小一致，待 108D ≥3 boot。
- **低 W 结构地板**：`pages = floor((W − 32,768 − 16,384)/1,456) − 1`；W=49,152 ⇒ 0 个检索页（且 0 页时压缩窗 + 生成预留 = 50,608 > 49,152）⇒ A2 的 recall 结构性不可用。

**108B-A2 + 108C 实测修订（2026-10-18）**：

- **A2 只跑 speed**：W=49,152 的 096 派生式给 0 个检索页（且 0 页时压缩窗 + 生成预留 = 50,608 > 49,152）⇒ recall 结构性 INCOMPLETE。A2 读数 = 横幅 1.16x、8k 107.41、177,959 档 TTFT 166.6 s / steady 101.19、headroom 1,840 MiB；比 2.0e9 只多 384 MiB，不作推荐点。
- **Orca 可用池 ≤2.0e9**：3.1e9 的横幅 2.11x 与 2.0e9 的 1.36x（纸面 +11.5%）、1.8e9 的 1.22x（纸面 +11%）都以横幅为准；3.1e9 在 KVMem 锁页区叠加后处于池悬崖上方——117,314 档冷 TTFT **1119.6 s** vs 2.0e9 的 **120.7 s**（9.3×），steady 18.92 vs 18.00（同带）；2.0e9/1.8e9 的视窗 recall 都有效 HIT（10 槽、baked 10、read-back 0、TTFT 25.8 s）。**Orca 的 ~18-19 tok/s = eager no-spec 基线，不随池变**；结论限定 eager no-spec，不与生产 FULL+DFlash2 78-87 比。
- **Orca recall 的有效性前置**：flush 必须把 ingest 的原生块顶掉（日志必须出现 `rewritten` + `baked`）。3.1e9 的池 236 块 > ingest+flush 164 块 ⇒ flush 不驱逐、serve 原生命中、针（76.2k）在原生 65,536 窗内 ⇒ 探针 HIT 是 recency，不采信。C 点的 recall 只在 2.0e9/1.8e9 有效。
- **headroom 工具的 KVMem 假阳性**：`prod_headroom_check.ps1` 的 `evicted weight pages = engine shared − offload 8192 − 106` 把 workspace + 快照 + 权威区（~9.8-10.7 GiB pinned）算成被驱逐权重；C1/C2/C3 的 DEGRADED 都不成立。**判 KVMem 臂的池悬崖只看 prefill 时间/首请求 TTFT**；真正的新分配仍按 049 看独显余量。
- **Orca 的 90% L 档用 117,965**（L=131,072），不能照抄 GSQ 的 180,633（会超 L 被拒）。

**108D 召回侧（2026-10-18，前半完成）**：

- **2.6e9/W=98,304/pages=32 = 三深度全 HIT**：0.45/0.55/0.65 均答出 77349（TTFT 57.0/60.6/63.3 s，rewrite 64,432 = sink 1,456 + 32×1,456 + recent 16,384，baked 32、read-back 64/0）⇒ **可用上下文 = 200,704（本组三深度）**。**全深度需求的推荐点 = 2.6e9/W98,304/p32。**
- **2.0e9/W=65,536/pages=10 = 0.55/0.65 HIT、0.45 MISS**：0.45 的针页在 serve top-64 排 **11**，10 个槽差 1 位（真检索边界；`finish=length` 是伴随现象，针页未进槽）；0.55/0.65 的针页 rank 5。⇒ **低池下界只保 0.55 以上，可用上下文 < 200,704**（最大 L 未扫，不宣称）；代价 = 窗口 65,536（vs 98,304）、headroom +384 MiB、ingest 204 s（vs 225 s）。
- **一个深度一支干净 boot** 的规则在两点都再次成立；0.65 直接复用 A1 的干净 boot 读数（2.0e9 的 `s108_a1_w65k_r`、2.6e9 的 `s108_a1_w98k_r`）。
- **speed ≥3 boot + 分型（GSQ 2.6e9/2.0e9）+ Orca 2.0e9 ≥3 boot = 下一批**（本步按用户暂停令停在召回侧；以上全部是单 boot 状态，不作速度断言）。

**不在本矩阵内**：`VLLM_KVMEM_VIEWPORT=1` 的 **068-071 双坐标大视窗**（已判死回滚）、检索粒度/`TOPN` 寻优（062 NO-GO + 079 耦合陷阱）、GPU 化烘焙、任何生产 launcher 改动。072+ 单坐标视窗的 recall 腿**在**本矩阵内（074/078 协议）。

---

## 4. 出手前必须算的账

**页长**：`page = 16 * cdiv(mamba_raw, 16 * 1152)`。
- GSQ +spec N=2：`mamba_raw = 1,675,264 B` → `16*cdiv(1,675,264, 18,432) = 1,456`；统一页 `1,677,312 B`；`bytes_per_block(G=8) = 13,418,496 B`；`mbt >= 1,458`。
- Orca 无 spec：`mamba_raw = 1,634,304 B` → `1,424`；统一页 `1,640,448 B`；`bytes_per_block = 13,123,584 B`；`mbt >= 1,424`（现用 1,456）。
- **W 不进入 page 公式**。旧「W→页长→AOT 键」的说法作废；AOT 只按日志 cache hit/miss 判定。

**容量式**（滑窗组）：`sw_blocks = cdiv(min(W-1+2*mbt, L), page) + 1`；
`blocks_per_req = cdiv(16,G) * sw_blocks + 4 * cdiv(48,G)`；
`num_blocks = floor(pool_bytes / bytes_per_block)`；
要求 `blocks_per_req <= num_blocks - 1`；并发 = `(num_blocks-1)/blocks_per_req`。
**纯 serve** 把 `sw_blocks` 换成 `cdiv(L,page)`（无滑窗），mamba 块同式，`sw` 取 4（mbt=1024 时 `cdiv(2047+2*1024,page)+1`）。

**GSQ KVMem 前沿（纸面，boot 横幅确认；公式对 3.4e9 实测 1.17x 误差约 2%）**

| 池 | W | L | mbt | sw_blocks | blocks_per_req | 估算并发 | 处置 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3.4e9 | 131,072 | 200,704 | 1,458 | 94 | 212 | 1.19（实测 1.17） | 基准 |
| 2.6e9 | 98,304 | 200,704 | 1,458 | 71 | 166 | 1.16 | A1 |
| 2.0e9 | 65,536 | 200,704 | 1,458 | 49 | 122 | 1.21 | A1 |
| 1.6e9 | 49,152 | 200,704 | 1,458 | 37 | 98 | 1.20 | A2 |
| 1.2e9 | 32,768 | 200,704 | 1,458 | 26 | 76 | 1.16 | 可选边界 |

**GSQ 纯 serve 对照（L 随池下调；公式对 3.4e9 实测 1.004x 吻合）**

| 池 | L | blocks_per_req | 估算并发 | 处置 |
|---:|---:|---:|---:|---|
| 3.4e9 | 163,072 | 252 | 1.00（实测 1.004） | 基准，复用 |
| 2.6e9 | 117,000 | 190 | 1.01 | A3，compile+sample |
| 2.0e9 | 85,000 | 146 | 1.01 | A3，compile+sample |
| 1.6e9 | 63,000 | 116 | 1.02 | 可选 |

**Orca KVMem 前沿（纸面；公式对 3.1e9 实测 2.11x 约低 10%，以横幅为准）**

| 池 | W | L | mbt | blocks_per_req | 估算并发 | 处置 |
|---:|---:|---:|---:|---:|---:|---|
| 3.1e9 | 65,536 | 131,072 | 1,456 | 124 | 1.90（实测 2.11） | 基准 |
| 2.0e9 | 65,536 | 131,072 | 1,456 | 124 | 1.22 | C |
| 1.8e9 | 65,536 | 131,072 | 1,456 | 124 | 1.10 | C，拒起则 1.9e9 |
| 1.4e9 | 49,152 | 98,304 | 1,456 | 100 | 1.05 | F 可选边界 |

**pinned 预算表**：`WORKSPACE_MB` + 权威区 2.0 GiB + 快照区（`SNAPSHOT_KEEP × SNAPSHOT_EVERY_PAGES`）逐档列出，与本机 23.1 GiB RAM / 16 GB 独显对照；跑过推理后独显余量仅 ~492-513 MiB（049）。

**Orca 池悬崖参照（093，非 KVMem）**：1.4e9 = 87.2 tok/s；1.5e9 = 64.6；1.6e9 = 15.6；1.72e9 = 4.8-5.5。KVMem 臂多了锁页区，悬崖可能左移；这是 C 的判据之一。

---

## 5. 预算、安全与停手条件

- 每子批 ≤6 支 boot；一批读完再开下一批；臂与生产一次只跑一个（生产当前停，保持停）。
- 改 L/mbt = 换 AOT 键 ⇒ 首编 boot 只作编译检查，第二支才采样；只改 W 是否重编以日志判定。禁止单次冷 boot 判回归（必守 17）。
- kill 一律 `tools/kill_vllm_orphans.ps1`，起服一律 `_tmp_line_b/boot_keep.py`；kill 后等独显 <800 MiB 再起（必守 18）。
- 任何「装不下/NO-GO」结论出手前，先逐条 diff 同卡已跑通的 launcher（必守 32）；结论限定到「这支臂」。
- 判据没达成写 **INCOMPLETE**，不许用「机制走通」替代（必守 20）。
- **Kill criterion**：3.4e9/W=131,072 基准点若 8k 掉出带或窗外针 MISS，立即停扫、写 NO-GO（KVMem 在基准点就不保正常使用）；若低池相对同池纯 serve 没有「更大可用上下文或更多 headroom」的净收益，也停扫。
- 新 27B 发布或 2026-10-30 止损触发：立即停扫，按《资产清单》§5 T0-T3 执行。
- 收尾只记录生产状态、不恢复（必守 30）。

---

## 6. 交付与落档

- 交付形态：三条新臂并存、生产默认不动、新行为 env 门控（默认关）：
  - `tools/serve_gsq_kvmem108_frontier.cmd`（main/recall 双形态）
  - `tools/serve_orcasaq2_029_kvmem108.cmd`（main/recall 双形态）
  - `tools/serve_gsq_serve108.cmd`（纯 serve 对照）
  - 生成器 `tools/make_kvmem108_arms.py`（CRLF + 替换命中数断言）
  - 编排器 `tools/kvmem108_scan.py`（boot / anchor / recall / collect / sample / stop）
  - 补丁工具 `tools/patch_flashinfer_page.py`（§10 gap 1）
  - scratch 归档 `tools/scratch/`（§10 gap 2）
- 每子批一 commit：步骤文档四段式 + 日志 + 提示词「状态一句话/头名」+ 详版新纪律（若有）+ 台账 `prod029_logs/step108_boots.json` + 证据目录 `prod029_logs/{kvmem_pool,orca_pool,step108}/…`。
- 定档结论表列：池、W、最大接受 L、精确注意力窗口、可用上下文、8k 中位、90% decode、页步、装配页账、headroom、GO/NO-GO/INCOMPLETE。
- Orca 结论必须带限定：eager no-spec KVMem 臂，不是生产 FULL+DFlash2 口径。

---

## 7. 开放项与默认裁定

- **Orca 速度口径**：eager 同口径相对不劣化（默认）；图兼容腿需单独授权。
- **GSQ 纯 serve 对照**：做（默认，批 A3）。
- **recall 腿的视窗页数**：以 W 为上限套 096 派生式 `pages = floor((W − 32,768 − 16,384)/1,456) − 1`，对 W=131,072 逐位复现 55（head = (1+pages)×1456）；W=98,304→32（head 48,048）、W=65,536→10（head 16,016）。**W=49,152 及以下 ⇒ 0 个检索页 ⇒ 只报 main/speed，recall 写 INCOMPLETE**（不猜、不把 0 页当 MISS）。一个深度一支干净 boot。
- **lru/legacy**：只在缩池撞 authority 先满时做；不预设结论。
- **止损缺口**：§10 gap 1（page.py apply/revert 工具）与 gap 2（scratch 归档）已在 Phase 0 收口；gap 3（vision）/gap 5（真新 checkpoint）不在本计划范围。

---

## 8. 附录：v1 矩阵（已作废，仅追溯）

v1 的批 A 是「GSQ 3.4e9→2.6/2.0/1.6/1.2e9，固定 W=131,072 / L=200,704，每档 ≥3 boot」；批 B 是「W 131,072/98,304/65,536 × 池 2.6e9/1.6e9」；批 C 是「Orca 池 8e8/1.0e9/1.4e9 × L 16,384/49,152」；批 D 是「Orca 0.8e8 LOAD 腿」；批 E 是 lru/legacy。
作废原因逐条见 §0：固定 W 在低池拒起（A）、W×池未联合（B）、Orca 低池拒起且 L<W 不触发淘汰（C/D）、每档 ≥3 boot 与每批 ≤6 boot 的预算冲突、窗外召回需要 VIEWPORT=1 而 v1 固定 VIEWPORT=0、页长机制记错。
