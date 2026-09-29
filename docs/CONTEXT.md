# CONTEXT — DFlash2 混合精度量化项目术语表

（只收术语与边界，不收实现细节与决策记录——那些在 ADR 与交接文档里。）

- **臂（arm）**：一次测量配置 = 一个草稿检查点 + 其启动器。同协议（9 固定 prompt × 3 重复）下与其他臂可比。
- **配方（recipe）**：各模块量化位宽 × 量化算法的组合（如 gptq3c = qkv/fc 3bit GPTQ + kproj 2bit GPTQ + …）。
- **工作点（working point）**：一个 (接受率, 体积) 二元组及其检查点；"最佳工作点"指验收线下的最优臂。
- **接受率**：accepted / draft_tokens，草稿提议 token 的被接受比例。最终仲裁指标。
- **判据（gate）**：预注册的验收条件；与事后观察严格区分。
- **断崖**：位宽降低一档时接受率非线性崩塌（如 RTN kproj 2bit 的 −21pp）。断崖归属量化算法而非位宽本身——GPTQ 可修复。
- **干净版 / 泄漏版**：标定集与测试集零重叠的产物为干净版；历史上标定集误用测试 prompt 的产物为泄漏版（仅留作对照，不作依据）。
- **混杂效应**：早期对照实验同时变过两个变量（标定重叠性 + 标定量/多样性），其差值不是"泄漏效应"，措辞禁用。
- **测量口径**：吞吐/占用以 decode step 时长分布（流式逐 token 计时）为准；nvidia-smi utilization 不作判据（有假卡前科）。

---

## vLLM 0.30-on-Windows 迁移（2026-09-25 立项）

- **底座（base）**：自研改动的承载树，即目标版本的 Windows 构建（如 SystemPanic 的 vllm-win 0.29.0 whl）。换底座 = 换承载树，不换自研内容。
- **迁移源（migration source）**：自研改动的唯一出处 = vllm-overlay 开发头；生产副本只是"基线安装 + 补丁"的冻结物，不是迁移源。
- **锚点（anchor）**：迁移前在现役栈采集的对照数据包（PPL 逐位、接受率、tok/s、TTFT、显存账、needle）；一切验收以它为参照系，判等用逐位/统计指标，不用单条文本。
- **功能域（feature domain）**：自研改动按用途划分的迁移批次单位（量化域、投机解码域、KV 域等）。
- **三方合并 / 直搬 / 收敛**：同一文件合并"上游基线 + 上游增量 + 自研增量"为三方合并；上游无对应物的自研文件为直搬；本地补丁与上游同源时删本地补丁用上游为收敛。
- **一次性迁移**：本次迁移只服务当前 Qwen3.8-27B 工作负载，不建立版本跟进机制；下个 27B 模型出现时直接换新底座，不为它预留兼容。
- **硬承诺 / 增值**：阶段 0–2（生产等价 + 切换）是硬承诺永不砍；0.30 甄选是增值，做多少赚多少，砍尾不砍底。
- **止损（kill switch）**：2026-10-30（或新 27B 发布日，以先到者为准）未完成项一律冻结转新模型适配；另一触发条件=砍完特性总分仍低于现役。
- **B12X**：SM120 专属注意力路径的代号（0.30 起为 `--attention-backend B12X_ATTN`）；与本地 B12X spike 战役同源。注意与 DFlash2 的 extend 实验（曾慢 18% NO-GO）区分——那是 spike 结论，不是 B12X 后端本身的结论。

---

## KVMem 虚拟化 KV 工作区（2026-09-29 立项，步骤 056；步骤 057 阶段 1a 开工）

- **工作区（workspace）**：一次 agent 轨迹的逻辑上下文总量（含存放在 host 的部分），上限 **262,144 token**。与"模型一次调用能共同注意多少 token"是两件不同的事。
- **视窗（execution view / 工作集）**：某一时刻真正驻留 GPU、被注意力读到的 token 集合；上界 = `sink + 检索槽 + recent + query + 生成预留`。
- **页（page）**：本栈的分页粒度 = **1456 token**（混合模型 mamba/attn 页对齐强制；`bytes_per_block(G=8) = 13,418,496`）。**搬运与分配的最小粒度**。
- **子块（sub-block）**：检索**索引**粒度 = 128 token（一页 11 个子块）。索引精度与搬运粒度刻意解耦。
- **检索槽（retrieval slot）**：视窗里专用于放被选中历史页的固定数量槽位（N 个）；槽位的**位置**在请求开始前定死，选择只决定槽里的**内容**。
- **固定槽位布局（fixed slot layout）**：本方案的位置策略 —— `[sink S | 检索槽 N | recent R | query q | 生成预留 g]`。**关键不变式**：query 位置 `B = S+N+R` 与"最终选了哪些块"无关 ⇒ 不需要重 prefill。
- **identity 态**：workspace ≤ `budget_max` 时的运行态（整段全驻留、零检索、零重烘焙）。正确性判据 = 与 KVMem-off **逐 token 一致**（**identity canary**）。
- **Mean-K**：一个 (层, 子块, KV head) 的 **RoPE 前** K 的均值，检索索引的单元。索引与 query 同在"内容帧" ⇒ 打分不需要任何逆旋转。
- **raw K**：RoPE **前**的 K（post-`k_norm`），位置无关。本方案只对**旋转的 64 维**（`partial_rotary_factor=0.25`）存 fp16 作为重建权威。
- **重物化（rematerialization）**：把一个 host 页按目标槽位位置**单次**重烘焙 RoPE 后写回 GPU。本方案**禁止 delta re-RoPE**（每次从 raw K 重建 ⇒ 零累积漂移）。
- **budget_max / 生成预留（gen_reserve）**：视窗内"历史可用"与"本回合输出可用"的二分；生成预留是单次生成的**硬上限**（超出须优雅失败）。
- **冷启动**：本请求的**新增** token 数超过池容量（无前缀缓存的首轮长 prompt）⇒ 需要 **prefill 期 stage-out**。
- **有界 prefill（步骤 057 落地的机制）**：给 `full_attention` 层一个逐层滑窗 `per_layer_sliding_window`（`VLLM_KVMEM_SW_WINDOW=N` 门控），使单请求 KV 块需求从"∝ prompt 长度"变成"∝ 窗口"，从而让**超池长 prompt 能 prefill 进来**。**⚠️ 它不是工作区存储**——滑出窗口的历史 KV 是被**丢掉**的（offload 只存滑窗可达那几块）⇒ 工作区 store 另需 copy-before-free 钩子。
- **轨迹键（trajectory key）**：`hash(root task message) + hash(current query)`；prompt 缩水 > max(1024, 1%) 判为新/被压缩轨迹 ⇒ 冷启动。
- **被否的两条路线（勿重开）**：(A) 论文/QW3 的"重 prefill query"（要 GDN 状态快照 + 带 #43 同族退化路径）；(B) 参考实现的"原始位置 + 洞 + KQ mask"（vLLM 的 mask 路径只支持 fp16/bf16 KV，而生产 KV 是 nvfp4）。

---

## OrcaSAQ2 EXL3 集成（2026-09-29，步骤 059）

- **OrcaSAQ2 插件**：`Continuum-AI-Corp/OrcaSAQ2-kernel` 的 vLLM plugin，专门解析 OrcaSAQ2 的 `tensor_storage`、mixed-bit、mul1、int8 embedding 和 EXL3 lm_head；步骤 059 使用 commit `7ddffb1`。
- **原生扩展**：`G:\\exllamav3-master` 编译出的 ExLlamaV3 1.5.3 Windows CUDA extension；目标 venv 中扩展文件为 `exllamav3_ext.cp312-win_amd64.pyd`。
- **验证结论**：Qwen3_5ForCausalLM OrcaSAQ2 checkpoint 在 vLLM 0.29、TP1、eager、16K、GPU utilization 0.88 下权重加载 11.46 GiB；8001 服务 ready；chat 请求返回 4 且 finish=stop。
- **运行约束**：FlashInfer Windows JIT 需要 `FLASHINFER_WORKSPACE_BASE=C:/fw` 和 `FLASHINFER_EXTRA_LDFLAGS=-LG:/qwen3.8model/vllm-win029/Lib/site-packages/tvm_ffi/lib -ltvm_ffi`；checkpoint 原始索引名为 `model.safetensors.index (1).json`，服务目录需有标准别名 `model.safetensors.index.json`。
- **回滚**：`tools/revert_orcasaq2.py` 卸载 Orca/ExLlamaV3 并安全移除生成的索引别名；不触碰生产 GSQ。

---

## 外置 EXL3 插件集成（2026-09-29，步骤 058）

- **vllm-exl3**：外置的 `--quantization exl3` 插件，不是本仓 `vllm/` 源码的一部分；本步使用 `G:\vllm-exl3-0.5.0`，许可证为 AGPL-3.0-only，记录仓继续保持 Apache-2.0。
- **注册层通过**：在 `G:\\qwen3.8model\\vllm-win029` 中 entry point 可发现，`register()` 可重复调用，`Exl3Config` 已注册；这只证明插件能接入当前 vLLM 0.29 API，不证明模型可服务。
- **原生资格**：`exllamav3`、`exllamav3_ext`、`vllm_exl3_c` 三者均存在才可讨论原生 CUDA；步骤 058 三者均缺失，`runtime_diagnostics()` 报 `native_available=false`，因此原生 GPU/端到端资格未通过。
- **模型边界**：当前生产 Qwen3.8-27B GSQ 是 compressed-tensors WNA16 int3 + DFlash2，不是 EXL3 routed-MoE checkpoint；不能把插件安装状态写成 GSQ 已启用 EXL3。
- **回滚**：`tools/revert_vllm_exl3.py` 恢复 Qwen4Exp `.orig/.orig2` 并卸载外置插件；验证必须从 `G:\\qwen3.8model` cwd 执行，避免记录仓源码树遮蔽 venv。
