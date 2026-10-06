# vllm-030win — 换新底座资产清单 + 止损执行预案（阶段 5 收口件）

> **定位**：《vllm-030win-迁移计划.md》§4 阶段 5 的明文交付物——"止损或完成后都出交接文档，含**下个 27b 换新底座时可直接复用的资产清单**：锚点脚本、迁移源 diff、验证脚本"。该交付物此前全仓从未成档，本文补齐。
> **成档**：2026-10-06（步骤 106）。**零 boot、零显存、零代码改动**，纯盘点；所有条目都给**路径 + 用途 + 换底座时怎么用**，不重复各专项文档的机制推导。
> **触发**：①新 27B 发布（提前止损）②2026-10-30 到期止损 ③只是想清楚"家底有什么"。三种都从本文 §5 或 §1 进。
> **与其它文档的分工**：《进度文档.md》§三 = 路径速查表（在哪）；《切换与回退预案.md》= **上一次**（0.27→0.29）切换的操作预案；本文 = **下一次**（新模型 + 新底座）的资产账与动作序列；纪律全文仍是《铁律与现场详版.md》（「必守 N」= 该文件同编号）。

---

## 0 一句话结论

**代码资产 = 一棵树 + 一段 git 历史**（记录仓 `vllm/` 权威树，迁移源 diff 全在 commit 里）；**判据资产 = 脚本族**（锚点 / 探针 / 离线单测 / watchdog / headroom）；**配方资产 = 3 支点定的 launcher + 1 条零人写数字的泛用链（092）**；**知识资产 = 必守 1-36 + 四份设计权威**。
止损日的正确动作**不是**"把这套东西升级到新版本"，而是**换新底座后用同一批脚本对打同一批锚点**——拍板已定：**不建版本跟进机制**。

---

## 1 资产 A：锚点与验证脚本族（换底座后立刻要用的"裁判"）

### 1.1 锚点口径与数据包

| 资产 | 位置 | 用途 / 换底座时怎么用 |
|---|---|---|
| 锚点采集协议 | `docs/锚点采集协议.md` | 唯一口径（探针集、中位数、多次采样报数）。**新底座采"对照锚"必须同口径**，否则读数不可比 |
| 锚点数据包（14 份 json） | `docs/锚点数据包/` + 全量副本 `G:\qwen3.8model\vllm-030win-锚点数据包\` | **旧栈（0.27.1）锚值 = 回退锚数据**：`ppl_anchor_overlay0271.json` / `lineA_nospec_*` / `lineB_spec_n2_*` / `lineB_acc_*` / `smoke_s3_s6_*`。复现即回退成功（《切换与回退预案.md》§4）；新底座的第一次验收就是拿这份对打 |
| PPL 工具 | `tools/anchor_ppl.py` + `tools/run_anchor_ppl*.cmd`（**10 支**：`run_anchor_ppl` = 旧 overlay 栈、`_base029` 系 7 支 = 基线 / batch3b / batch4 / batch5_nvfp4_r4 / lineBfix_r5 / rerun036 / rerun036b、`_kvgroup_G8(_b)` = 036 复锚） | **PPL 判据已降级**（同配置两 boot 差 1e-3 ⇒ 非位精确，主判 = needle 多深度，必守 20）；换底座仍要跑它，用途是"量化域没被搬坏"的粗筛 |
| 长上下文锚 | `tools/anchor_longctx.py`（`--model` 可切 Orca）+ `tools/run_anchor_a5.cmd` / `run_anchor_nospec.cmd` / `run_anchor_arm.sh` | 8k / 32k / 90% 档 decode + needle；**watchdog 的 8k 探针与它同口径** |
| 环境契约模板 | 任一支 `tools/serve_gsq_*` / `tools/run_anchor_*`（必守 12） | 新底座起服**一律抄这些 launcher**，不要手搓命令行 |

### 1.2 运维与判据脚本（换底座后必须原样可用）

| 脚本 | 用途 | 关键口径 |
|---|---|---|
| `tools/prod_watchdog.ps1` | 起服 + 判快（慢则杀重启，上限 3） | 8k 探针 ×3 中位 **≥105** 判快；**配置变更后阈值须重标定**；退出码 0/2/3/4；进度看 `prod029_logs\watchdog.log`（别接管道 tail） |
| `tools/prod_headroom_check.ps1` | 显存余量体检 | 主判 = **引擎共享 − 8,298 MiB**；0=OK / >100=WARN / >250=DEGRADED；服务停着时报 `service down` |
| `tools/pdh_gpu_engine.py` | PDH per-process GPU 采样 | **否证"他占客户端"只认 PDH**（`--query-compute-apps` 看不见 WDDM 图形/拷贝客户端 = 必守 26④） |
| `tools/kill_vllm_orphans.ps1` | 按 CommandLine 匹配杀（`cli.main serve\|multiprocessing.spawn\|spawn_main`） | **Path 过滤必漏**；链式 bash→内联 PowerShell 必崩 ⇒ 一律文件版（必守 10） |
| `tools/soak_prod.py` / `tools/soak_orca.py` | 长稳（判定按 `finish_reason` 语义，052 已修假阳性） | 定档验收的"长稳/多轮"腿；池值上探类挂账的前置判据 |
| `tools/sync_venv.py` | `vllm/` ↔ venv **逐字节一致性**（保留各文件自身行尾） | 必守 9；**现役基线 = 2736 个 `.py` 全一致**（2026-10-06 复测 OK）。新底座建好 venv 后第一件事 = 跑它确认基线 |
| `G:\qwen3.8model\_tmp_line_b\boot_keep.py` | 起服包装（`cmd.exe /c <launcher>`） | **只吃 `.cmd`，喂 `.ps1` 会静默不起服**（080 踩过） |
| `_tmp_line_b\ctx_arm.py` / `mk_arm.py` / `mk_nospec.py` | 变体臂生成→boot→容量/显存→多档探针→杀 | **不入 git**，属临时 scratch——换底座前若还要用，先手工收进 `tools/`（见 §10 缺口） |

### 1.3 KVMem / Orca 探针与离线单测（**回归网，零显存**）

换底座最省钱的用法：`26` 支 `tools/kvmem_*.py` 与 `tools/profile_card/*` 里的大部分**都能在 CPU / 离线跑**，先在新底座上跑绿离线网，再谈 boot。

- **离线单测（无 GPU 即可判）**：`kvmem_remat_test.py`（35 项，重物化往返 + 误差尺）/ `kvmem_viewport_test.py`（16）/ `kvmem_capture_op_test.py`（2 项既存 FAIL，属 KVMem 线挂账）/ `kvmem_index_test.py` / `kvmem_slot_pick_test.py`（63）/ `kvmem_slot_policy_test.py`（103 的 lru/legacy）/ `kvmem_asm_interleave_test.py`（104 交插）/ `kvmem_record_nosync_test.py` / `kvmem_timing_unit_test.py` / `kvmem_stage_coverage_test.py`（逐槽×逐组覆盖）/ `kvmem_hooks_test.py`（22，钩子注册表）/ `tools/profile_card/verify_kvmem_defaults.py`（**096 回归锚：KVMem config 派生链必须逐字复现历史 GSQ 默认值**）。
- **在线探针**：`kvmem_assembly_probe.py`（装配主探针，含 `--reset-prefix-cache` / `--nonce` / `ab` 子命令 = 102 同 boot 同态 A/B 的载体）/ `kvmem_ws_probe.py` / `kvmem_viewport_probe.py` / `kvmem_k3_probe|replay|analyze|k3c_analyze` / `kvmem_window_control.py` / `kvmem_boot_fingerprint.py`（带外 1 Hz，**逐 PID** CPU/RSS）/ `kvmem_time_budget.py` + `kvmem_trace_budget.py`（计时与 kernel 级台账）/ `kvmem_bimodal082.py`（驻留分型）。
- **Orca 线**：`orca_nvfp4_probe.py`（chat + needle 三深度；注意 `hit` 字段口径 = 必守 31①）/ `step087_o3_meta_probe.py`（**meta-device 零显存门禁盘点：新模型接 DFlash2 之前先跑它算三个数**，必守 29）/ `step085_orca_boot.py`（臂驱动 `next/probe/stop/show`，扩了 `d2`/`d24`）/ `step090_orca_speed.py`（带外分型采样）/ `step093_o_sweep.py`（O 线扫档 = 池-速度悬崖与 083 曲线的取证工具）。
- **105 定档后新增**：`tools/serve_orcasaq2_prod029_dflash2.cmd`（Orca 投机定档，见 §3.1）。

---

## 2 资产 B：代码资产（"迁移源 diff"= git 历史，不在别处）

### 2.1 权威树

- **记录仓 `G:\qwen3.8model\vllm-030win-git\vllm\` = 0.29.0 底座 + 全部移植**，5,106 文件 / **2736 个 `.py` 与 venv 逐字节同步**。
- **唯一远程 = `github.com/deathwing475/vllm-030win`（main）**；底座仓 `vllm-029base-git` 无远端、HEAD `cd5e784`，只作工作区/归档，**其中 commit 必须立即 cherry-pick 进记录仓**（必守 3）。
- ⇒ 换新底座时"要搬哪些代码"的完整答案 = `git log --oneline`（现 168 个 commit）+ 下面四条锚点链，**不需要再去找旧的 overlay 工作区**。

### 2.2 自研迁移五批（九功能域；源 = `vllm-overlay` tag `freeze-migration-20260925`，45 文件 +2,937/−252）

| 批 | 功能域 | 记录仓步骤 / 底座 commit |
|---|---|---|
| 批 1 | 协议 / GC 冻结 / hybrid 逃生门（GC 挂 `CudaGraphManager.capture` = 0.29 架构点） | 步骤 012 / `57188ef` |
| 批 2 | KV 域 + `multi_turboquant_kv`（**KV 域 4 文件零搬运**：0.29 已吸收） | 步骤 013 / `f56c964` |
| 批 3 | 量化域：`inc/*` 低位注入 + 码本 embedding + `torch_wna16`（**humming 系 9 文件零搬运**） | 步骤 014 / `7d4c0e1` |
| 批 3b | PIECEWISE 经 custom-op 边界恢复（humming×dynamo） | 步骤 015 / `5fe078a` |
| 批 4 | 投机解码全套（DFlash2 / DFlash v1 / 支撑件；XQA 策略反转） | 步骤 016 / `76bec8b` |
| 批 5 | nvfp4 / flashinfer SM120 + **flashinfer 0.6.11→0.6.18.post1** | 步骤 017 / `78db8a6` |
| 回归修复 | INC 派发错位 / `HummingLinearMethod` 过 custom-op / embedding 两行 | `6fc2108` + `e9d534d` + `7bb4ce0` |

**免搬清单（覆盖性判定，别重演）**：`docs/vllm-030win-调研-改动清点.md` + 《交接文档.md》§4。**方法论 = 先做"底座是否已含该改动"的覆盖性判定，再决定搬不搬**——本战役靠它省掉约 30 文件 / 130 行回搬。**换底座时第一步重做的仍然是这个判定，而不是重放 diff**。

### 2.3 0.30 甄选了哪些、哪些判死（直接继承，不必重研）

- **已移植 2 项**：`#54782`（PIECEWISE 图不可用**抛错**而非静默乱码的安全网，commit `6b014de`，A7 收档）+ `#55341` 的**运行期**（`warmup_kernels` 提前到 `capture_model()` 之前 + capture 后 `lock_workspace()`，修"workspace resize 释放静态 graph buffer"的真实顺序缺陷，commit `cd5e784`，A1 收档）。同批评估后 NO-GO：`#54646`（89% 已覆盖）、`#54557`（启动提速不划算，瓶颈=权重 I/O）、**`#54794` autotune 缓存实测为空**（顺带排除 PPL 漂移的 autotune 嫌疑）、`#56908`（WSL 专属）、`#54660`（本栈不走那两条路径）、`#55095`（依赖未启用的 adaptive verification）。
- **覆盖率自检法**：把 PR diff 的 `+` 行拿到底座逐行找（`#54782` 移植后 17/17、`#54646` 89%、`#54794` 37%）⇒ **0.29 底座已含大量 0.30 改动**。
- **NO-GO 结论可跨底座继承**（除非依赖变）：D 组 + C6 全 NO-GO（flashinfer 的 SM120 GDN = **CuTe DSL**，而 `nvidia-cutlass-dsl` 元数据 POSIX/Linux-only）；C8 不适用（本栈走 prefix-cache 的 `--kv-offloading-size`，不经 tiering）。**铁律：平台专项类移植（CuTe DSL / DeepGEMM / CUTLASS）先查依赖在目标 OS 上是否存在。**

### 2.4 修复回填五连 + 布局修复（**换底座最容易整批丢掉的东西**）

| commit | 修的是什么 | 新底座上如何复验 |
|---|---|---|
| `d7cdb91` | **cuMemcpyBatchAsync 驱动级缺陷**（非默认流即毒化上下文）⇒ `swap_blocks_batch` Windows 分支收口为 ctypes 逐条 `cuMemcpyAsync` | 32k 二轮 TTFT 21.0→3.1 s 是其证据；新底座先查上游是否已修 |
| `3498ef1` | **AOT 缓存校验尾换行非对称**（`inspect.getsource` 补 `\n` vs `open().read()` 不补）⇒ 每 boot 重编译 ⇒ 生成码漂移 | 同产物两 boot ±0.4% |
| `3cc6c64` | 草稿图化解耦（decode −13.3%） | 步长台账 |
| `8723e6c` | GDN 包装减脂 | 同上 |
| `f0bff05` | **`VLLM_KV_GROUP_SIZE` 分组开关**（容量 +18.5%） | 横幅 `layers-per-group 5 -> 8` |
| 084（engine core resolve 处） | **nvfp4 KV 布局泛化修复**：专用 kernel 假设 head-major，默认 LBNHC 下**静默乱码零 ERROR** | 日志 `preferring LBHNC`；GSQ 靠 `VLLM_KV_CACHE_LAYOUT=HND` 恰好踩对 |

### 2.5 KVMem 全套（泛用化 094-100 完成态）

- **模块**：`vllm/v1/kvmem_workspace/`（11 文件 / **6,273 行**）= `capture / codec / config / groups / hooks / index / manager / metadata / remat / worker / __init__`。
  - `config.py` = 096 的**派生链**（knob 从 profile card 推导，无 card 时逐字节回退到历史常量，`verify_kvmem_defaults.py` 钉死）；
  - `codec.py` = 097 的 **PageCodec 注册表**（页几何 + 量化器从 `remat.py` 提出；未实测布局宁拒不算）；
  - `hooks.py` = 098 的**模型钩子注册表**（`qwen3_next.py` 只剩一行 `model_hooks_for(type(self).__name__)` ⇒ 任何复用 `Qwen3NextAttention` 的新模型**零改动自动覆盖**）。
- **门控**：实测 grep `vllm/` 里 **`VLLM_KVMEM_*` 共 36 个键**，全部**默认关**（交付纪律：新行为必须 env 门控 + 变体 bat 并存 + 生产默认不动）。含 103 的 `VLLM_KVMEM_SLOT_POLICY`（legacy/lru）与 104 的 `VLLM_KVMEM_ASM_INTERLEAVE`。
- **模型侧**：`vllm/model_executor/models/qwen3_next.py` 两钩子 + eager 分支的 `[3,T]→[T]` positions 归一（100，Orca EXL3 相容性）。
- **设计权威**：《调研-KVMem虚拟化KV工作区.md》（§12.x = 机制与实测）+《调研-KVMem泛用化.md》（四工作面 + §5.7 交插）。
- **边界提醒**：KVMem 线**已冻结**（2026-10-03 阶段 1 出口达成），泛用化改造是唯一已授权例外且已收官 ⇒ 新底座**不默认重启 KVMem**（§6）。

### 2.6 第三方包内补丁（**venv 资产，不在 `vllm/` 树里 ⇒ sync_venv 管不到**）

| 补丁 | 落在哪 | apply / revert |
|---|---|---|
| page 校验缓存化（**gap 三胜之一 −1.3 ms**） | `flashinfer/page.py`（marker `vllm-030win patch`） | **revert 只在 `G:\qwen3.8model\_tmp_line_b\revert_page_patch.py`，未入 git** ⇒ §10 缺口 #1 |
| `seq_lens_cpu` 同步消除（−2.55 ms） | vllm 树内 | `tools/apply_seq_lens_cpu_patch.py`（含 revert，必守 14） |
| 077 调度器一次性诊断 / 079 计时 / 080 record 同步 / 081 slot_pick / 066 mamba ext / KV 分组 | 各 apply 脚本，**全部默认关 + 实测可逆** | `tools/apply_sched_trace_step077.py`、`apply_kvmem_timing_step079.py`、`apply_kvmem_record_sync_step080.py`、`apply_kvmem_slot_pick_step081.py`、`apply_kvmem_mamba_ext_step066.py`、`apply_kv_group_size_patch.py` |
| humming 八件套垫片 | `tools/shims/`（**`SHIMS.md` 是索引**）：`humming_device/forward_dynamo/nvrtc/ops_utils.patched.py` + `mapped_file.h.win32` + `build_devinfo.cmd` + `run_ninja.cmd` + `prebuild_nvrtc4.py` + `qwen3_5.py.embed-patched` | **重装环境必重放**（步骤 010 的"七层洋葱"） |
| 钉快档 shim | `tools/pin_shim/sitecustomize.py`（`PYTHONPATH` 注入；`VLLM_DBG_TRACE/MIN/PIN` + `PASSES=2` + `SKIP_SHARED=1`） | 删 launcher 4 行 `set` 即回退 |
| EXL3 空共享模块容忍 | `tools/orcasaq2_sitecustomize/s085_empty_shared_patch.py`（`ORCA_EXL3_ALLOW_EMPTY_SHARED=1`，默认关） | **任何 EXL3 checkpoint 的 MTP/EAGLE 都会撞它**（必守 28③） |
| 上游 PR 材料 | `tools/upstream_pr/`（`0001-mapped-file-win32.patch` + `0002-ops-utils-msvc-buildflags.patch` + README） | **outward-facing，等用户发话**（§6） |

### 2.7 插件资产

- **vllm-exl3 0.5.0**（AGPL，不复制进 Apache 树）：`tools/install_vllm_exl3.py` / `tools/revert_vllm_exl3.py`（`.orig` 恢复）——通用注册层已归档、被下面专用路径替代。
- **OrcaSAQ2 专用插件**：`tools/install_orcasaq2.py` / `revert_orcasaq2.py`（ExLlamaV3 1.5.3 native extension 编译成功，插件 commit `7ddffb1`）。
- **判死留档**：`G:\vllm-exl3-0.5.0` 的通用路径与 EXL3 起 MTP/EAGLE 的"起不了"结论都已被 085 推翻/替代（详版 §C）——**换底座时别重跑这条弯路**。

---

## 3 资产 C：launcher 配方与定档配置

### 3.1 三支"已定档"的（可直接抄进新底座）

| 档 | launcher | 配方（实测要点） |
|---|---|---|
| **GSQ 生产** | `tools/serve_gsq_prod029_n2.cmd`（模型目录入口 shim = `Qwen3.8-27B-3Bit-GSQ\run_dflash2_n2_base029.cmd`） | DFlash2 gptq3c **N=2** + nvfp4 KV + **手动池 3.4e9** + FULL_AND_PIECEWISE（无 compilation-config）+ prefix-cache + offload 8G + `--max-num-seqs 1` + `VLLM_KV_GROUP_SIZE=8` + `--mamba-ssm-cache-dtype bfloat16` + **L=163,072** + mbt 1024 + util 0.922 + pin_shim 三钉 ⇒ 引擎自报 **163,719 tokens**，8k 快档 116-125 |
| **Orca 投机定档（105）** | `tools/serve_orcasaq2_prod029_dflash2.cmd` | **090 配方写死**：池 **8e8** / L=**16,384** / nvfp4 / DFlash2 gptq3c N=2 / FULL 图（无 compilation-config）+ capture 3 + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` / 五把容量钥匙 / util 0.922 / mbt 1024 / `--max-num-seqs 1` / 端口 8001 / `OMP_NUM_THREADS=8` / `PYTHONPATH` 同时挂 `orcasaq2_sitecustomize`（`ORCA_EXL3_ALLOW_EMPTY_SHARED=1` 必开）+ `pin_shim`（但**不开** `VLLM_DBG_TRACE/MIN/PIN` = 092 方向性阴性）。健康带 9 支 74.38-89.83；**GSQ 生产默认不动** |
| **泛用入口（092）** | `tools/serve_generic.cmd` + `tools/profile_card/` + `profile/*.json` | 人类参数只有 `SGEN_MODEL / L / TIER / FAMILY / SPEC / POOL / PORT`，其余全部派生或实测（§8） |

**并存变体（生产默认均未改）**：GSQ `_cb2`（草稿码本 2bit，容量 165,175、接受率 61.91%）、`_vis_cpu`（vision 塔驻内存 CPU 跑，图片整轮 13.5→5.1 s，**boot 期易撞余量悬崖**）；Orca `nvfp4 / mtp / nomtp_12k / nvfp4_mtp(_graph) / dflash2 / nvfp4_dflash2(_prodcap) / kvmem100`；KVMem `ws163k_graph / load1456 / codec097 / viewport07x`。共 **43 支 `serve_*.cmd`**（含冒烟/对照臂）。

### 3.2 环境契约八条（**新 venv 必逐条重放，缺一必炸或静默失败**；权威 =《交接文档.md》§2）

1. `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`（torch/humming 现场编译要 cl+ninja）。
2. `HOME`/`USERPROFILE` = **`C:\fi`**（humming 缓存键的一部分；不一致 → 重建 → 炸）。
3. `CUDA_HOME`/`CUDA_PATH` = **`C:\PROGRA~1\NVIDIA~2\CUDA\v13.3`**（**8.3 短路径 + 反斜杠**）。
4. `LIB` 加 CUDA `lib\x64`；`TMP`/`TEMP` 重定向 G 盘；**勿删 `G:\qwen3.8model\_tmp_anchors`**（PPL runner 的 TMP，删了 LINK LNK1104）。
5. zmq 29550 孤儿先清（launcher 已内置）。
6. 模板 = `tools/run_anchor_ppl_base029.cmd` / `tools/serve_gsq_base029*.cmd` / `serve_minicpm5_2b_dspark_base029.cmd`。
7. **cwd 铁律**：绝不在记录仓目录下起 python（会被仓内 `vllm/` 树捕获，炸 `_C_stable_libtorch`，**极像环境坏**）⇒ 一律 `cd G:\qwen3.8model`。
8. **flashinfer 0.6.18.post1**：从冻结生产 venv **整目录拷贝**（纯 py JIT 包）+ `FLASHINFER_WORKSPACE_BASE=C:/fw`（**必须正斜杠**，0.6.18 把 `\f` 吃成 form-feed → WinError 123）+ `FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi`（缺了 LNK2019）。旧 0.6.11 备份在 venv `flashinfer_0611_backup/`。
   - 附：**venv 里 `flashinfer/page.py` 是 LF、`gdn_attn.py` 是 CRLF** ⇒ 打补丁必须逐文件核行尾（必守 4 同族）。

### 3.3 profile / platform card（现成两张，可当模板）

- `profile/Qwen3.8-27B-3Bit-GSQ.json`（9.2 KB）、`profile/qwen3.8exl3.json`（6.5 KB）、`tools/profile_card/platform_card.json`（环境契约 + 平台读数）。
- 每个值带 `source ∈ {derived, measured, human-copied, platform}`；**生成器用到 `human-copied` 必打 WARN**（抄来的永远是债，不许伪装成结论 = 必守 34 + 设计档 §5）。
- 轨 A 双锚点已逐位复现（GSQ 163,719 / Orca 157,910）⇒ 换底座后这两张卡就是"派生链还活着"的自检件。

---

## 4 资产 D：机制知识（写不进脚本的那部分）

### 4.1 换底座最容易被重新踩的六条（全文 = 详版「必守」同编号）

| # | 一句话 | 为什么换底座会重犯 |
|---|---|---|
| 必守 32 | 写"装不下/容量 NO-GO"之前，**先逐条 diff 同卡已跑通的生产 launcher** | 新模型 + 新底座必然先撞容量墙（088→089 的整课） |
| 必守 34 | `need(L) = a + b·L`；`GPU KV cache size: N = max_concurrency × max_model_len`，**不是字节** | 版本升级可能改 `N` 的定义 ⇒ 守卫每次复核 |
| 必守 33 | **池字节 = 速度开关**（自动池给得多反而 −16×）；池安全上限 1.4e9 | 新底座的 auto sizing 目标函数不会变好 |
| 必守 17 | 改被图编译引用的文件 = **换 AOT 键** ⇒ 双 boot 或 watchdog；只改 `kvmem_workspace/` 不触发 | 新底座 AOT 缓存全新，首 boot 必"首编深塌"（16.8-18.7 tok/s 假象） |
| 必守 27 | **pinned host 大区吃 WDDM 提交预算** ⇒ `LOAD=1 boot OOM` 先查 pinned 横幅（无 "Tried to allocate"） | 新底座换 allocator 也一样 |
| 必守 35/36 | 阴性腿先证明 boot 走得到目标代码；**装配类实验的 native prefix cache 必须确定性清除**（`--reset-prefix-cache` + `VLLM_SERVER_DEV_MODE=1`） | 新底座的 prefix cache 行为会变，命中量逐 boot 漂移 0%↔98.4% 赌不得 |

### 4.2 设计权威四档（本文不复述机制）

《调研-KVMem虚拟化KV工作区》（§12.x）｜《调研-KVMem泛用化》（四工作面 + §5.7）｜《调研-新模型全自动适配》（§2 钥匙全集 14 条含 `human-copied` 账 / §5 守卫 / §9 顺序）｜《调研-029whl与030差异面》+《调研-030功能菜单》（砍量顺序与 NO-GO 台账）。

### 4.3 已判死（换底座也不重开；全文 = 详版 §C）

FULL 图挂死根治（WONTFIX_WITH_ROOT_CAUSE）/ 草稿 1bit 全族（047）/ E 轨池值回填（048）/ 草稿权重再压缩（043/051 收官）/ target 权重 PIN（049）/ 上下文提长与池值上探（035/040）/ KVMem 早期页偏置修法（062 NO-GO）/ KVMem 双坐标不重烘焙路线（068-071 整体回滚）/ 修拷贝形态提速 / "去掉 record 同步就提速" / 修提交-完成握手与去看驱动帧 / 用 autotune 窗判态 / 用 `--query-compute-apps` 否证他占 / nvfp4 + LBNHC 乱码（已泛化修复）/ "EXL3 起不了 MTP/EAGLE"（085 定罪已绕）/ "Orca 需要单独写 DFlash2 target adapter"（087 推翻）/ "用 bf16 的 z-lab 草稿直接 boot 16 GB 卡"（087 算术判死）。

### 4.4 关键数字（新底座要拿来对照的基准，不是结论）

GSQ：容量 **163,719** / 8k 快档 **116-125**（判快门 105）/ 146k 档 107.73 / 引擎专用 15,497.8 + 共享 8,298 = **恒等式 ≡ 23,796 MiB** / 独显余量 boot 后 ~795 MiB、跑过推理 ~492-513 MiB / 页 1456 / block 1456 / G=8 / 池 3.4e9 硬上限。
Orca：权重 11.43-12.55 GiB（无投机/MTP/DFlash2）/ 定档池 18,589 / 8k anchor 74.38-89.83 / needle 三深度 12,208 token / 接受率 0.48-0.49（310/265 drafts 口径）。
KVMem：页 1,640,448 B（89 chunks × 18,432 B，kernel block 16）/ remat 误差尺 **0.15039064100710628**（逐位锚）/ 装配 skip 51,264 / 收益 省 71.9±3.3 s、TTFT −31.5%（102）/ 交插 −24%（104，记录不作主结论）。

---

## 5 止损执行预案（新 27B 发布日 / 10-30 到期日的动作序列）

| 时刻 | 动作 | 判据 / 注意 |
|---|---|---|
| **T0 当天，30 分钟内（零风险、零显存）** | ①停战役：不再开新实验步骤；②把"新 27B 已到 / 止损触发"写进《交接提示词.md》§8 与详版 §D；③**生产不动**（GSQ 0.29 栈继续跑，新模型**不在生产端口起**）；④§6 冻结项逐条确认状态并搬进《交接收档存档.md》 | 收尾仍按必守 30：**只记录状态，不恢复环境** |
| **T1 当天** | 核对回退锚（本文 §7 清单，含 `git tag` / venv 存在性）；确认记录仓 clean 且与 `origin/main` 同步；`sync_venv.py` 基线留档 | 家底必须先能自证完整，再谈新底座 |
| **T2 新底座起步（≤3 天窗口）** | 按 **§8 新模型适配起步路径** 跑：platform card + `build_card.py` → `serve_generic.cmd` → `autoprobe.py`；**先跑离线单测网（§1.3）再 boot** | 判据顺序 = 加载能起 → 全 CUDA 图兼容 → needle 多深度 → 速度分型（≥3 boot） |
| **T3 判定** | 新栈总分 ≥ 现役 ⇒ 择时切换（走《切换与回退预案.md》同型演练：三线锚对打 + 演练 + 切换）；**否则留在 0.29 栈，新模型只作实验变体** | 特性级取舍：拖后腿的特性砍掉不进生产；砍完仍 < 现役 ⇒ 止损回现役并报告（迁移计划 §1 兜底流程） |

**硬规则（拍板已定，不再讨论）**：
1. **不建版本跟进机制**——止损 = 直接拥抱新底座，不做 0.31/0.32 的滚动跟进。
2. **阶段 3 的 NO-GO 结论直接继承**（CuTe DSL Linux-only 这类不会因为换模型而变）。
3. **交付形态永远是**：变体 bat 并存、**生产默认不动**、新行为必须 env 门控（默认关）。
4. 上游 PR 若在止损时仍未提交 ⇒ 随止损登记作废一次（材料留 `tools/upstream_pr/`，不追）。

---

## 6 冻结项清单（解冻 = 用户单独授权；止损日整表封存）

| 项 | 当前状态 | 解冻条件 | 出处 |
|---|---|---|---|
| KVMem 阶段 2/3 完整版（缩池治悬崖 / 生产切换） | **冻结**（阶段 1 出口 083 达成） | 硬停止条款：单独授权，重新进入亦然 | 详版「硬停止条款」+《调研-KVMem虚拟化KV工作区》 |
| 099 manager 代表组 / 混合页大小泛化 | 未做（按需另授权） | 单独授权 | 《调研-KVMem泛用化》§1.2-C |
| Orca 池 **1.4e9 上探档** | 定档用 8e8；1.4e9 挂账 | **≥3 boot + soak 复测**才可启用 | 步骤 105 / 详版 ⭐105 |
| MTP 图档池-悬崖画像 | 挂账 | 需 boot 预算 | 步骤 105 |
| `VLLM_KV_GROUP_SIZE=8` 敏感性（4/6/16） | card 内标 `human-copied` | 单独授权 | 《调研-新模型全自动适配》§7 |
| tool-call 段（Orca 定档未验 parser） | 定档**不加** | 需一次验收 | 步骤 105 |
| nvfp4 × 多步 draft × ≳10K 长 prompt 停摆 | **NEGATIVE 已记账，机制未定罪** | 用户授权单独立步 | 必守 28⑤ / 步骤 086 |
| spec × 装配 equality | 冻结后再授权项 | 用户授权 | 详版 §C / 设计档 |
| 上游 PR（mapped_file Win32 + ops/utils MSVC flags） | **材料就绪，outward-facing** | 用户发话 | 详版 §A4 |
| 余量监控接入日常 / watchdog 增强 | 未做（改生产脚本行为） | 用户点头 | 详版 §A2/A3 |
| A7 vision 131k/155k 锚 + vision 面切换 | **vision 仍在 0.27 栈**（`run_qwen_vision_offload_157696_best.cmd`） | 新底座适配时一并裁 | 《切换与回退预案.md》§6 |

---

## 7 回退锚核对（**2026-10-06 实测**）

| 锚 | 核对方式 | 结果 |
|---|---|---|
| 0.27.1 冻结 venv `G:\qwen3.8model\vllm-win` | `Scripts\python.exe` 存在 + `Lib\site-packages\vllm-0.27.1.dist-info` 在位 | **完好**（cp313 / 0.27.1 / torch 2.13；只读，回退当锚） |
| 迁移源 overlay tag | `git -C nvfp4-win-experiment\vllm-overlay tag -l` | **`freeze-migration-20260925` 在位**；HEAD `2c51ebc`（C1 instrumentation 留在 branch，默认关） |
| 底座仓 | `git -C vllm-029base-git log -1` | HEAD `cd5e784`（**无远端**，全部 commit 已 cherry-pick 进记录仓） |
| 记录仓 | `git status --short --branch` + `git rev-list --count HEAD` | `main` clean、与 `origin/main` 同步、168 commit |
| 代码树 ↔ venv | `python tools/sync_venv.py` | `OK: all 2736 repo .py files match the venv` |
| 生产 | `nvidia-smi` + `prod_headroom_check.ps1` | 独显 **48 MiB / 0%**，`engine python process not found (service down?)` ⇒ **生产停**（必守 30 只记录） |
| 旧栈入口 shim | `Qwen3.8-27B-3Bit-GSQ\run_dflash2_n2.cmd`（0.27 栈）与 `_base029.cmd`（现役）并存 | 未删，可直起 |

**明确未核（别当成已核）**：①**没跑 0.27.1 venv 的 `import vllm` 冒烟**（不动只读锚；真回退当天才做，判据 = `python -c "import vllm; print(vllm.__version__)"` 输出 0.27.1）；②`_tmp_line_b/revert_page_patch.py` 只确认脚本在位，**没做 apply↔revert 往返复验**（§10 缺口 #1）。
**只回配置、不换栈**的两步链（次序不可反）：`tools/apply_prod_ssm_bf16_step046.py revert` → `tools/apply_prod_kvgroup_step036.py revert`（036 的 revert 断言 `max-model-len == 144432`，046 未回滚时必失败）。

---

## 8 新模型适配起步路径（**零人写数字**；092 交付链）

1. **判形态**：HF safetensors / GGUF / EXL3？量化家族？有无 vision 包装？（EXL3 + 投机 ⇒ 先备好 `ORCA_EXL3_ALLOW_EMPTY_SHARED=1`；GGUF 非本栈形态）。
2. **建 card（零 boot、零 GPU）**：`python tools/profile_card/build_card.py <model_dir> [--draft <dir>] [--name <card>] [--group-size G] [--ssm-dtype DT]` → `profile/<name>.json`。
3. **派生候选 + 生成 launcher**：`set "SGEN_MODEL=<card或目录>" && set "SGEN_L=<长度>" && tools\serve_generic.cmd`（`TIER=speed|capacity|auto`，speed 档 = 手压 `need(L)×1.2`）。**生成器对每个 `human-copied` 值打 WARN ⇒ WARN 就是要还的债**。
4. **一次自证 boot**：`python tools/profile_card/autoprobe.py --launch <生成的.cmd> --card profile/<name>.json --port 8001`——轨 A（config 算）× 轨 B（引擎横幅）**不一致就非零退出并拒绝生成配置**，不许"人肉挑一个看起来对的"。
5. **接投机（DFlash2）之前先算三个数**（必守 29）：`python tools/step087_o3_meta_probe.py …` ⇒ `len(target_layer_ids)×target_hidden == draft.fc 宽`、`mask_token_id < vocab`、`max(aux) < num_hidden_layers`；**只能走 V2 runner**（判据 = 日志 `Using V2 Model Runner`）。
6. **验收四件（顺序即优先级）**：①能起 + 全 CUDA 图兼容（`--enforce-eager` 慢 4× ⇒ eager 过拟合不算过，必守 21）；②needle 多深度 + **≳10K 长请求**（必守 28⑤）；③速度断言 **≥3 boot + 分型**（`utilization.memory`/`power.draw`/温度/`pcie.link.*`，必守 7/25）；④长稳（`soak_*.py`）+ 余量体检（`prod_headroom_check.ps1`）。
7. **任何"装不下"先 diff 已跑通 launcher**（必守 32）；**判据没达成写 INCOMPLETE**，不许用"机制走通"替代。

**已知未验（继承，别当能力承诺）**：《调研-新模型全自动适配》§7——真"收支持"新 checkpoint 的**全流程仍 INCOMPLETE**（本机候选全是纯 GGUF）；`--language-model-only` 在纯 causal-LM 上是否 no-op 未实测；auto sizing 与驻留悬崖方向相反的机制未定罪。

---

## 9 复用判定表（换底座时每件资产怎么办）

| 资产 | 能否直接带走 | 前置动作 / 风险 |
|---|---|---|
| 锚点数据包 + 采集协议 | **直接带走** | 新栈锚另存**新文件名**，不覆盖旧证据（收尾自检项） |
| `tools/*.py` 探针与单测 | **直接带走** | 依赖 `vllm/` 内部符号的（`verify_kvmem_defaults` / `step087_o3_meta_probe`）在新版本可能漂，**先跑离线**再 boot |
| `prod_watchdog.ps1` / `prod_headroom_check.ps1` | 带走但**须重标定** | 阈值 105 与共享基线 8,298 MiB 都是**这一版配置 + 这张卡**的读数（必守 19 的教训：换臂就变） |
| `vllm/` 权威树 | **不整树带走，重做覆盖性判定** | 逐功能域查新底座是否已含（§2.2 方法）；自研独有的（`kvmem_workspace/`、`multi_turboquant`、`inc/*`、`torch_wna16`、码本 embedding）才搬 |
| KVMem 全套（6,273 行 / 36 门控键） | 结构可搬，**默认不重启该线** | 解冻需授权；搬之前先跑 `verify_kvmem_defaults.py` 证派生链活着 |
| 修复回填五连 + nvfp4 布局修复 | **逐条查上游是否已修** | 已修就删本地补丁（"port from 0.30"的三处本地补丁当初就是这么收敛的） |
| `tools/shims/` 八件套 | 视 humming 版本而定 | 重装 humming 必重放；`mapped_file.h.win32` / MSVC flags 已在上游 PR 材料里 |
| launcher 三支定档 + 43 支变体 | 配方带走、**参数全重测** | L/池/util/mbt 都是这张卡这份底座的实测值；`mbt ≥ 页长 + num_spec` 这类不变式才可直接继承 |
| profile/platform card | **重生成** | card 里的 `measured` 段属旧底座；`platform_card.json` 的读数是这张卡的现场 |
| 机制知识（必守 1-36 + 四档设计权威） | **价值最高的部分，直接带走** | 换模型/底座时最先读的仍是这四档 + §C 已判死清单 |

---

## 10 资产缺口（本步如实记录，不做未授权的补救）

1. **flashinfer `page.py` 补丁的 revert 脚本在 git 之外**（`_tmp_line_b/revert_page_patch.py`）⇒ 违反"补丁必须配 revert"的可核性口径（必守 14）。补救 = 把它收进 `tools/` 并补 apply↔revert 往返复验——**属新增改动，须用户授权**（本步是纯文档，未动代码）。
2. **`_tmp_line_b\` 里的 arm 生成器（`ctx_arm.py`/`mk_arm.py`/`mk_nospec.py`）与 `boot_keep.py` 不入 git**，而 §5/§8 的动作序列依赖它们 ⇒ 换底座若 scratch 被清，需按 §1.2 重写。同样未入 git：`o86_stair.py`、`o88_acc_sample.py`、`roundG1_097.py` 等各步 runner（台账 json 在 `prod029_logs/`，脚本本身不在仓内）。
3. **vision 面从未切到 0.29 栈**（A7 锚未采）⇒ 新底座适配时 vision 只能当"无锚起点"，别照抄本文 §4.4 的 GSQ 数字。
4. **`kvmem_capture_op_test.py` 2 项既存 FAIL**（097 记账、非该步引入）⇒ KVMem 线解冻前它是脏的回归网。
5. 泛用链的**真新 checkpoint 全流程未验**（§8 末段）⇒ "零人写数字"目前是双模型复现层面的验收，不是跨模型承诺。

---

## 11 本文的维护口径

- 换底座每次动作序列跑完，把**实际读数与差异**追加到《实验步骤文档.md》对应步骤，并在本文 §5 表下加一行指针（**不复述细节**）。
- 新判死 → 详版 §C；新纪律 → 详版「必守」续编号；**提示词只加一行指针**（《任务收尾规范.md》§3 体积硬线 12 KB）。
- 本文允许长的部分 = 清单与判定表；**不允许**变成机制推导文档（那是四档设计权威的职责）。
