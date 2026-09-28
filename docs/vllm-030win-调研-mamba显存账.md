# vllm-030win 调研 — mamba（GDN 状态）显存账与回收方式（步骤 045）

> 2026-09-28。用户提案立项：「mamba（GDN 状态）直接多拿走 0.47 GiB 显存，能否拿回来、用什么方式拿回来（例如用其他结构代替 mamba 之类）」。本文 = 账本 + 机制 + 候选方式盘点 + 实测验证。主流程记录见《实验步骤文档.md》步骤 045。

## 1. 实测账本（全部源码 + 运行时闭环）

**构成**（Qwen3.8-27B：64 层 = 48 GDN（linear_attention）+ 16 full attention；草稿 5 层 sw）：

| 项 | 值 | 来源 |
|---|---|---|
| conv state（每层每份） | 51,200 元素 × bf16 = **102,400 B** | `MambaStateShapeCalculator.gated_delta_net_state_shape(1,16,48,128,128,4,2)` 运行时调用；窗口维度 = `conv_kernel(4)−1+num_spec(2)` = 5 —— **N 不只加份数，还进 conv 窗口** |
| ssm state（每层每份） | 786,432 元素（=48 头×128×128）× **fp32** = **3,145,728 B** | 同上调用返回 `dtypes=(bfloat16, float32)` |
| 每层 page（raw → 统一后） | 3,248,128 → **3,262,464 B**（pad 0.44%） | 生产日志两条 mamba 行互证（`Padding mamba page size by 0.44%`）；3,248,128 = 2832×1152/1.004413 精确反推 |
| 每份全模型状态 | 48 × 3,248,128 = **148.68 MiB** | 手算 |
| align 份数 | **4 = 2 + N(2) + checkpoint(0)** | `MambaSpec.max_memory_usage_bytes`（`kv_cache_interface.py:889`）：公式硬编码 `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`，不读 `max_concurrent_batches`；GDN 路径 checkpoint 块默认 0（仅 Kimi-K3 KDA/flashkda 为 1，`abstract.py:71`） |
| **mamba 总占用** | **物理 594.75 MiB**；池记账 597.4 MiB（G=8：6 组 × 4 blocks × 26,099,712 B） | kvdump 实测每组恒 4 blocks × 8 层 × page |
| 容量耦合 | 每请求 24/129 blocks（18.6%）；容量摊销 3,111 B/token（034 单价分解的 10.5%） | 容量公式 + kvdump |

**0.491GB（034 记录）口径勘误**：该数字无法追溯出字节级来源（当时无 kvdump 明细佐证），判为公式估算的中间口径。本轮权威账 = 上表实测。用户提案的「0.47 GiB」与 594.75 MiB 总占同量级，账以下表为准。

## 2. 机制链（为什么是 fp32）

1. **模型声明**：`config.json` 的 `text_config.mamba_ssm_dtype = "float32"` —— 模型作者为 GDN 递归状态的数值稳定性显式声明 fp32。
2. **引擎接线**：`vllm/model_executor/models/config.py` 的 `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config` —— `--mamba-ssm-cache-dtype auto` 时读取该字段落 fp32；用户显式传值则覆盖（打一条 warning，官方支持路径）。
3. **参数类型**：`MambaDType = Literal["auto","float32","float16","bfloat16"]`（`config/cache.py:69`）—— 没有 fp8 选项；fp16/bf16 是官方支持的两个降档。
4. **fused kernel 白名单**：`qwen_gdn_linear_attn.py:90` `FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)` —— **bf16 state 走 fused GDN decode kernel；fp16 不在白名单**（会触发 `_fused_gdn_decode_unsupported_reason` 掉非 fused 路径）。conv state 恒 bf16（跟随 model dtype）不受影响。
5. **page 统一的联动**（容量收益的主通道）：`platforms/interface.py` 的 `_align_hybrid_block_size` 选 attention block_size 使 `block_size × attn_page_1tok(1,152 B) ≥ mamba_page`。ssm 降 bf16 → mamba page raw 3,248,128 → 1,675,264 B → 最小 block 2832 → **1456**（=16×cdiv(1,675,264,18,432)，kernel 对齐 base 16）→ 统一 page 1,677,312 B → `bytes_per_block`(G=8) 26,099,712 → **13,418,496** → `num_blocks = ⌊3.4e9/13,418,496⌋` 130 → **253**。

## 3. 候选方式矩阵

| # | 方式 | 显存收益 | 容量收益 | 代价/风险 | 判定 |
|---|---|---|---|---|---|
| A | **ssm fp32→bf16**（`--mamba-ssm-cache-dtype bfloat16`） | 物理 −273.1 MiB（594.75→321.65） | **上限 144,432→163,072（+12.9%）**（num_blocks 130→253） | 精度（模型声明 fp32 的动机 = 递归状态长程累积误差）；block_size 减半对 prefix 粒度/元数据的影响 | **✅ 实测 PASS（步骤 045），见 §4** |
| A' | ssm→fp16 + stochastic rounding | 同 A（2B） | 同 A | **fp16 不在 fused 白名单 → 掉非 fused 路径**，大概率性能崩 | 备选不推 |
| B1 | N=2→1 | 省 1 份 = 148.7 MiB | +8.6%（041） | 吞吐 −27%（041 实测） | NO-GO（已判） |
| B2 | `mamba_cache_mode=none` | 省 1 份 | +6%（041） | 语义 = prefix caching 关闭，与生产多轮对话冲突 | NO-GO（已判） |
| B3 | `mamba_cache_mode=all` | — | — | Qwen3_5 构造函数硬断言不支持（`qwen3_5.py:314`） | 不可用 |
| B4 | 关 `async_scheduling` | **0**（份数公式硬编码 "2"，不读它） | sw 组 3→2 ≈ +1% | 关闭双批重叠的调度性能风险 | 低赔率不立项 |
| C | **结构替代**（换掉 GDN） | — | — | 48 层 GDN 是训练权重的一部分，换结构 = 换模型重训；超 10-30 止损线 | **不可行**（论证结案） |
| D | 打破 page 统一假设 | pad 仅 0.44%（A 后 0.12%） | ~0 | 源码注释自陈 non-trivial（fragmentation）；034 已记 | 不立项 |
| E | **池值联动回填**（A 通过后） | 0（守恒转移） | 理论再 +7%（163k→~178k） | 035 的池值敏感性在总显存守恒下未证；boot 压力窗（044 机制链）变化 | 二阶段独立 A/B |

**结论**：唯一值得实测的是 A（必要时叠加 E）。mamba 状态中可回收部分的三分账：fp32 超额 288 MiB（A 可回收）、投机槽 2 份 297.4 MiB（性能换来，不可回收）、async 双批 2 份中的 1 份冗余（公式硬编码，不可回收）。

## 4. 实测验证（2026-09-28，驱动 `_tmp_line_b/ssm_ab.py` + `mk_ssm_arm.py`）

### 4.1 冒烟臂（kvdump + bf16，1 boot）

- **预测全中**：`attn_block 2832→1456`、`pad 0.44%→0.12%`、`bytes_per_block 26,099,712→13,418,496`、`num_blocks 130→253`、mamba 组恒 4 blocks、full 组 2×100、sw=4（cdiv(4095,1456)+1）、`blocks_per_req = 228 ≤ 252`（null block 约束）。
- 引擎自报容量（池 3.4e9 / max_len 144,432 不变）：**160,268 tokens**（现役 145,551，+10.1%）。
- **fused kernel 零回退**：`fused CUDA kernel requires` 警告 0 次；`Using Triton/FLA GDN prefill kernel` 正常；dtype 覆盖走官方 warning（`Qwen3.5 model specifies mamba_ssm_dtype='float32' ... Using the user-specified value`）。
- 上限公式：`2·cdiv(L,1456) + 28 ≤ 252` ⇒ `cdiv ≤ 112` ⇒ **L ≤ 163,072**（预测，待实测验证）。

### 4.2 A/B 性能门（fp32 vs bf16 交替 ×3，8k 探针 ×3 中位，`ssm_ab1`）

| 臂 | 8k steady ×3 | boot 中位 |
|---|---|---|
| p32a/b/c（fp32 现役） | 118.61/118.49/122.69 · 122.62/117.17/120.53 · 123.15/127.21/127.17 | 118.61 / 120.53 / 127.17 |
| bf16a/b/c | 126.62/115.50/123.40 · 129.98/120.55/118.79 · 125.50/128.65/120.91 | 123.40 / 120.55 / 125.50 |

- bf16 全部 18 样本落在 042 谱高带（115.50-129.98），fp32 对照 117.17-127.21 —— **同带、无回归**（bf16 中位略高 ~2 tok/s，不作宣称，谱内噪声）。
- needle **18/18 命中**；boot_s 全部 61-68s（缓存命中，同 L 已编译）。

### 4.3 容量上限 + 双判据（`ssm_cap1/bisect/cap2`）

- **L=163,072 被引擎接受**，自报容量 **163,719 tokens**（现役 145,551，**+12.5%**）；needle 全中。
- **146k 档（90% of 163,072，判据②）= 103.6-104.8 tok/s** ✓（门槛 70）；**8k = 116.9/125.0（中位）** ✓（门槛 85，判据①）——163k 配置缓存 boot 双判据全过。
- **⭐首编深塌发现（新随机源，机制挂账）**：任何 `max-model-len`/dtype 变更后的**第一次 boot**（AOT 重编译，boot ~145s）8k steady 全部 **16.8-18.7 tok/s**（4/4 复现：L=163,072/160,000/155,000/159,000），**同配置第二次 boot（缓存命中，boot ~61s）恢复正常带**（116.9-125.0）。深塌态 `/metrics` 定性：投机**在工作**（pos1 接受率 ~77%、accepted 1087）、`num_preemptions=0`、队列正常 ⇒ **每步开销真实暴涨（步长 ~103ms）**，非投机失效。嫌疑 = triton/flashinfer autotune-miss（呼应官方 #53436）。**生产应用：改 L/dtype 后的切换流程必须 boot 两次（第一次建缓存、第二次进生产），或 watchdog 探针兜底**。
- **044 残谱预测命中**：163k 缓存 boot 3 次中 1 次抽到 86.7（低带 83-99）——`83-99 残谱再现 = target 权重页 demote 成分` 的挂账预测成立，与配置无关（同配置其余 boot 116.9-125.0）。

### 4.4 长稳（`soak_ssm163_r1`，bf16 + L=163,072）

20 条混合请求 **15 PASS + 5 FAIL**；5 个 FAIL（decode2k_1/2、wiki8k/32k、wiki8k_repeat）**全部为已知假阳性**（`n_tokens>500` 阈值误判早停、finish=stop 自然结束，与 036 长稳完全同构，代码债第五次）。关键指标全绿：needle 8k（113.7/119.7）、32k（105.2/117.2/110.3）、**100k（111.1）**全中；reasoning（595/682 字符）与 tool-call（args_ok）正常；`health_after=true`。

### 4.5 PPL 门（双臂同轮对照，nvfp4 口径 / max-model-len 57344 / eager）

| 比对 | max \|Δmean_nll\|（8 条） |
|---|---|
| fp32 同轮复测 vs 历史锚 `lineBfix_r5` | 1.27e-3（= 036 记录的 boot 间漂移带） |
| **bf16 vs fp32 同轮对照** | **1.25e-3 —— 落在同一漂移带内** |

8 条全部 < 3e-3 容差 ⇒ **bf16 ssm state 无系统性数值劣化**（kernel 内部累计仍 fp32，state 只是存储精度）。

### 4.6 验收总判定

**方案 A（`--mamba-ssm-cache-dtype bfloat16`）全门 PASS**：性能同带（A/B 6 boot）、needle 全中、PPL 漂移带内、fused kernel 保留、长稳健康。**L 定版候选 163,072**（容量 163,719，+12.5%；判据①②过）。**生产 launcher 本轮未切换**（保守：等用户择时；切换流程含双 boot 规避首编深塌）。

## 5. 判据与回退

- 正确性主判 = needle 多深度全中；PPL 容差 `|Δmean_nll| < 3e-3` 多采样（PPL 非位精确，036）；接受率带 ±1pp（B1 协议，043）。
- 性能判 = 交替 A/B ≥3 boot、8k steady 不破带、64k prefill 无回归。
- 双判据（041）= 8k ≥85 且 90% max_len 档 ≥70。
- 回退 = launcher 删 `--mamba-ssm-cache-dtype bfloat16` 一行（零代码，天然幂等）。
