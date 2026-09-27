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
