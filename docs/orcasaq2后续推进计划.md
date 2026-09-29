# OrcaSAQ2 后续推进计划（Windows vLLM 0.29）

> 建立日期：2026-09-29
>
> 当前状态：步骤 059 已完成 OrcaSAQ-2-27B 的 Windows 原生 CUDA + vLLM 0.29 端到端 smoke。该计划只描述后续验证顺序，不代表已经执行。

## 1. 当前基线

已验证成功的独立变体：

- 模型：`G:\qwen3.8model\qwen3.8exl3`
- 来源：`orcarouter/OrcaSAQ-2-27B`
- 架构：`Qwen3_5ForCausalLM`
- 量化：EXL3/QTIP mixed-bit，`mul1`，平均 3.21 bpw
- vLLM：0.29.0 Windows 自研栈
- ExLlamaV3：1.5.3，Windows native CUDA extension 已编译
- Orca 插件：`Continuum-AI-Corp/OrcaSAQ2-kernel`，commit `7ddffb1`
- 已验证配置：TP1、eager、`max-model-len=16384`、`gpu-memory-utilization=0.88`、单序列、端口 8001
- 已验证结果：权重加载 11.46 GiB，KV cache 21,845 tokens，chat 返回 `4` 且 `finish_reason=stop`
- 生产 GSQ：8080 入口、launcher、池值、DFlash2、KVMem 均未改；后续所有 Orca 实验必须保持变体入口，不覆盖生产配置。

## 2. 已作出的路线决定

### 2.1 先 NVFP4 KV

这是第一优先级，因为它是对当前 Orca 基线最小的变量变化，而且 vLLM 0.29/FlashInfer 已有 NVFP4 KV 支持：

```text
--kv-cache-dtype nvfp4
```

先不启用 MTP、KVMem、CUDA graph 或 Orca DFlash2，保持：

- eager
- TP1
- 单序列
- 16K max model len
- GPU utilization 0.88

必须重新测量 Orca 自己的 hybrid page 对齐。**不能套用 GSQ 的 1456-token 页账**：当前 Orca 基线日志显示 attention block size 为 784，GDN/mamba 页对齐是另一套数字。

### 2.2 再验证 checkpoint 自带的 Qwen3.5 MTP

Orca checkpoint 自带：

- `mtp_bits=4`
- `mtp` 权重
- `Qwen3_5MTP` 运行时已存在于 vLLM 0.29
- checkpoint README 推荐 `qwen3_next_mtp`

因此下一条 speculative 路线是：

```json
{"method":"qwen3_next_mtp","num_speculative_tokens":2}
```

MTP 是 Orca 自带的模型能力，和当前 GSQ 的 DFlash2 草稿模型不是同一条链。

### 2.3 DFlash2 也纳入，但必须做 Orca 专用适配

用户明确要求 Orca 也要 DFlash2，因此 DFlash2 不再列为冻结项；但它不能直接复用 GSQ 的 `dflash2\\gptq3c`。

现有 DFlash2 接线主要面向 Qwen3/DFlash2 target：

- `qwen3_dflash2.py` 的 `DFlash2Qwen3ForCausalLM`；
- DFlash2 draft 的 `dflash_config`、mask token、grouped-conv 和非因果层约定；
- 当前 GSQ 的 draft checkpoint 与 target 结构。

Orca 是 `Qwen3_5ForCausalLM`，自带的是 Qwen3.5 MTP，不是 DFlash2 draft。Orca DFlash2 工作流必须新增两个独立资产：

1. **Orca target adapter**：确认/实现 `Qwen3_5ForCausalLM` 的 DFlash target 接口，覆盖 GDN 状态、full-attention 层、位置和非因果 draft 约定；
2. **Orca-compatible DFlash2 draft checkpoint**：现有 GSQ `dflash2\\gptq3c` 不能直接拿来给 Orca 用。若没有匹配 draft，先做适配可行性与 checkpoint 来源门，不伪造 DFlash2 已支持。

DFlash2 的顺序是：先在 Orca auto KV、eager、无 MTP 环境完成 target/draft 单步接口和 logits 对齐，再叠 NVFP4 KV、MTP 或 CUDA graph。DFlash2 和 Orca 自带 MTP 是两条独立 speculative 路线，不能同时作为第一轮变量。

### 2.4 再复用 KVMem 有界 prefill

当前 `Qwen3_5ForCausalLM` 复用 `Qwen3NextAttention`，因此现有：

```text
VLLM_KVMEM_SW_WINDOW=N
```

理论上可以作用于 Orca 的 16 个 `full_attention` 层。

但本阶段只复用“有界 prefill”机制，不宣称完整 KVMem。完整工作区仍缺：

- copy-before-free
- trajectory/page key
- host workspace store
- raw-K 捕获
- Mean-K 索引
- page retrieval
- fixed-slot rematerialization

KVMem Orca 实验必须先在 NVFP4 KV 单独通过后进行，否则无法区分 KV dtype 和滑窗问题。

## 3. 分阶段执行计划

### 阶段 A：Orca NVFP4 KV smoke

变体入口：`tools/serve_orcasaq2_029_nvfp4.cmd`

仅比步骤 059 改一项：

```text
--kv-cache-dtype nvfp4
```

验收：

1. 服务启动成功；
2. 日志明确显示 NVFP4 KV/FlashInfer FA2 路径；
3. 记录 attention block、mamba page、统一 page、KV tokens；
4. 短 chat 正常 stop；
5. identity prompt 与 BF16/auto 基线输出一致到 token 级或记录明确漂移；
6. 8K needle 命中；
7. 失败时保存完整 JIT/linker 日志，不改 venv 源码直接绕过。

判定：

- **GO**：NVFP4 boot + chat + needle 全过；
- **NO-GO**：NVFP4 路径在 Windows/SM120 或 Orca geometry 上失败，保留 auto/bf16 基线；
- 性能不作为阶段 A 的硬门。

### 阶段 B：Orca MTP smoke

前提：阶段 A 不要求 GO；MTP 可在 auto/bf16 基线上先做。

变体入口：`tools/serve_orcasaq2_029_mtp.cmd`

配置：

```json
{"method":"qwen3_next_mtp","num_speculative_tokens":2}
```

先保持 eager，不叠 NVFP4，避免一次改变两个变量。

验收：

1. MTP draft 权重加载成功；
2. target/draft 结构无 shape、dtype、hidden-state 错误；
3. 8K chat 正常 stop；
4. 接受率从 `/metrics` 记录；
5. 与 MTP off 同 prompt 做 3 轮交替；
6. needle/短 reasoning 正确；
7. 显存与 KV 容量记录，不把 Orca 数字和 GSQ 数字混用。

判定：

- **GO**：MTP 正确性通过且接受率可测；
- **NO-GO**：draft wiring 或 Qwen3.5 MTP 接口失败；回到 MTP off。

### 阶段 C：Orca DFlash2 target/draft 适配门

用户要求 DFlash2，因此在 MTP smoke 后正式推进 Orca 专用 DFlash2，而不是把 GSQ draft 硬接过来。

第一阶段不启动完整服务，先做静态和单步门：

1. 盘点当前 `qwen3_dflash2.py` 对 target config、`dflash_config`、GDN state、full-attention layer types、mask token、grouped-conv 的假设；
2. 确认 Orca 是否有可用的 DFlash2 draft checkpoint；没有则记录为外部资产阻塞，不伪造结果；
3. 为 `Qwen3_5ForCausalLM` 设计 target adapter，不改生产模型类，优先独立变体模块；
4. target/draft 单步 hidden/logits 对齐，先用 auto KV + eager + 无 MTP；
5. 通过后才做真实 speculative server、接受率、needle 和长稳。

DFlash2 验收门：

- target/draft 权重加载无缺失；
- 单步 target logits 与非 speculative 基线在容差内；
- 接受率可测且生成正确；
- GDN state、非因果 draft attention、mask/位置语义无异常；
- 不改变生产 GSQ launcher。

### 阶段 D：Orca + NVFP4 + MTP 组合

只有阶段 A、B 各自通过才做该组合。

验收重点：

- MTP 接受率是否因 NVFP4 发生系统性变化；
- KV 容量是否仍能满足请求；
- FlashInfer NVFP4 与 MTP 的 graph/eager 交互；
- 8K/32K needle；
- 至少 3 次交替 boot；
- 首编缓存与 WDDM 余量悬崖分开记录。

### 阶段 E：Orca 有界 prefill / KVMem 前半边

前提：阶段 A 通过，优先使用 NVFP4 KV；若 A NO-GO，则先在 auto/bf16 做机制 smoke，并明确不是最终路线。

先做：

```text
VLLM_KVMEM_SW_WINDOW=32768
--max-model-len 262144
```

暂时关闭 MTP 和 CUDA graph，只验证滑窗机制。

验收：

1. 冷 200K+ prompt 不 OOM、不永久卡队列；
2. 窗内 needle 命中；
3. 窗外 needle 作为正控失败；
4. Orca 自己的 page/容量数字记录；
5. 无前缀缓存冷请求；
6. 明确记录“滑窗丢历史 KV”，不把它写成完整工作区存储。

之后才考虑 Orca 版 copy-before-free/KVMem workspace。阶段 D 不改变生产 GSQ 的 KVMem 关闭默认。

## 4. 每阶段共同纪律

- 所有 Orca 变体使用独立端口和独立 `.cmd`，生产仍保留 8080 原入口。
- 每次改模型/kv/spec 参数都新建变体，不覆盖已验证启动器。
- 先 kill，再等 `nvidia-smi memory.used < 800 MiB`，再启动下一臂。
- A/B 交替，至少 3 boot；首编 boot 不作为性能结论。
- 先验证正确性，再测性能；阶段 A/D 性能只记录不作硬门。
- 任何 venv 源码补丁必须配套 revert；优先使用进程级 `sitecustomize`，不要直接改生产 venv。
- 不把 Orca 的 784 block/page 账和 GSQ 的 1456 block/page 账混写。
- 不把通用 `vllm-exl3` 和 Orca `orcasaq2` 同时安装到同一个服务环境并依赖注册顺序。
- DFlash2 必须接入 Orca 专用 adapter 和匹配 draft；在两者未通过单步对齐前，不启动完整 speculative service。

## 5. 预期交付

- `tools/serve_orcasaq2_029_nvfp4.cmd`
- `tools/serve_orcasaq2_029_mtp.cmd`
- `tools/serve_orcasaq2_029_nvfp4_mtp.cmd`（仅在 A/B 通过后创建）
- `tools/serve_orcasaq2_029_kvmem_sw32k.cmd`（仅在 D 阶段开始时创建）
- 每个变体对应新的实验步骤和日志条目
- 生产 GSQ launcher 和 KVMem 默认关闭保持不变

## 6. 当前明确结论

当前已知：

- Orca + ExLlamaV3 native CUDA：GO
- Orca + vLLM 0.29 eager/auto KV：GO
- Orca + NVFP4 KV：未测，不宣称
- Orca + MTP：未测，优先于 DFlash2
- Orca + KVMem 完整工作区：未实现
- Orca + KVMem 有界滑窗：代码路径理论可复用，未测
- Orca + DFlash2：正式列入后续阶段 C；当前尚无匹配 adapter/draft 证据，先做资产门和单步 logits 对齐
