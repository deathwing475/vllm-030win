# 步骤 082 执行计划：KVMem 移植验收收口

> **路线性质**：本步骤是 GSQ 栈 KVMem 阶段 1 的移植验收，不是 KVMem 长期优化立项。阶段 1 出口完成后冻结 KVMem 线并回到原项目主线；未经用户单独授权，不继续做检索、参数、性能、长稳或产品化优化。

## 一、协议

### 082-A：同 launch 配置双峰风险定性

目的：为移植验收排除或记录 ingest 性能风险，不寻找 KVMem 优化方案。

- 固定同一个 kernel 名、grid、block、输入 shape、`M`、batch、步型、页长 1456、`VLLM_KVMEM_RECORD_NOSYNC=1`、图模式和 AOT 缓存状态。
- 优先选择步骤 081 已显示慢调用位次稳定的 humming WNA16 GEMM。
- 每条件至少 3 个独立 boot；每 boot 至少 20 个同形调用。若预算迫使降级，最低 2 boot + 100 个同形调用，并明确标注降级。
- 条件至少包含：基线、降低但仍可启动的 KV 竞争、独立诊断臂降低 workspace/authority 竞争、去掉一个明确 GPU 常驻竞争者。若改变 KVMem 请求路径，必须标为机制隔离而不是纯驻留 A/B。
- 输出调用 P50/P90/P95、快慢簇比例、慢位次比例、device busy/空隙、dedicated/shared、功耗/时钟/温度/PCIe、PDH（若有）、调度完整性、`compiling_n=0`、capture drain `0+0`、ERROR/TB、needle 和视窗指纹。
- 结论仅允许：`驻留因果支持`、`驻留被否`、`证据不足`。不得把不同 M、不同步型、不同 kernel 名集合差异混成结论。

### 082-B：score 多深度、多 nonce 正确性网格

目的：完成阶段 1 正确性验收，不做召回率优化。

- 深度：0.40、0.55、0.70、0.85；每格至少 2 次有效 serve，报告分子/分母和实际 `n`。
- 每个 boot 使用新 dump 目录；完成 ingest/flush 后立即 probe，探针前禁止任何会写入 authority/workspace 的节拍请求。
- 使用 `--serve-ignore-eos` 或足够输出预算；只认 `recent_tokens=16384` 为视窗分支，`32768` 一律记原生分支。
- 每格记录 score/time rank、`in_selected`、baked 槽数、逐组覆盖、read-back、TTFT、finish_reason、needle 文本、错误和全图判据。
- 不得用单点 b6 外推召回率；失败写 `INCOMPLETE`，不通过调 TOPN 或重开早期页偏置路线“调到通过”。

### 082-C：PDH GPU Engine 补充

目的：独立补充“其他 WDDM 客户端”归因取证，不阻塞 082-A/B，不形成长期运维项目。

- 优先正确枚举 `GPU Engine` object/counter，按 PID 和 `engtype` 记录。
- pywin32 arity 或 `typeperf` 继续受阻时，如实记录“工具未完成/仍未否证”；不能用 `nvidia-smi --query-compute-apps` 否证 WDDM 图形/拷贝客户端。

## 二、阶段 1 出口

当前必验：

- identity canary；
- 紧预算 needle；
- 40/55/70/85% 多深度 needle；
- 冷 262K transcript；
- 多轨迹/串台；
- 1456 页长装配探针回归；
- 082-A 的三值风险结论；
- 082-B 的有效网格结果。

阶段 1 之后暂停、须用户授权：

- KVMem 检索算法改良或修早期页偏置；
- N/R/VIEWPORT 参数继续穷举；
- GPU 化烘焙、批量重 RoPE、进一步 kernel 优化；
- KVMem 长稳、生产化、缩池换余量、生产切换。

## 三、结果模板

- 082-A：条件 / boot 数 / 同形调用数 / 双峰统计 / 结论（支持、否定或不足）。
- 082-B：深度 / 有效采样数 / HIT / MISS / 视窗指纹 / 覆盖与 read-back / 结论。
- 082-C：PDH 状态（发现客户端、仍未否证或工具未完成）。
- 路线状态：`阶段1未完成`、`阶段1验收完成并回主线` 或 `需用户授权后才能继续`。

## 四、回主线闸门

阶段 1 出口达成后：保留 KVMem 变体、生产默认不动、冻结 KVMem 线；下一步恢复 `docs/orcasaq2后续推进计划.md` 的 Orca NVFP4 → MTP → Orca 专用 DFlash2 adapter/draft → Orca/KVMem。任何重新进入 KVMem 优化必须由用户单独授权。
