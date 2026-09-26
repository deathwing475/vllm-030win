# vllm-030win — Windows 原生 vLLM 引擎（Qwen3.8-27B + DFlash2 投机解码）

基于 **vLLM v0.27.1** 的 Windows 移植 + 自研增强，面向 **ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ**（compressed_tensors WNA16 int3）在消费级 Blackwell 卡（RTX 50 系，SM120）上的高性能推理。本仓库同时是「vLLM 0.30-on-Windows 迁移」战役的记录仓（见 `docs/`）。

English: A Windows-native vLLM v0.27.1 build with custom enhancements (DFlash2 speculative decoding, NVFP4 KV cache on SM120, WNA16 int3 quantization), targeting the Qwen3.8-27B-3Bit-GSQ model on consumer Blackwell GPUs. See `docs/` for the migration-to-0.30 campaign records. Upstream README: [README-UPSTREAM.md](README-UPSTREAM.md).

## 迁移战役状态（2026-09-26 更新）

> 把现役 0.27.1 自研栈迁到 SystemPanic vLLM 0.29 底座 + 甄选 0.30 功能。deadline 2026-10-30（新 27B 发布即止损），硬承诺 10-18 前完成生产等价+切换。

| 阶段 | 状态 | 说明 |
|---|---|---|
| 阶段 0 锚点+底座 | ✅ 收官 | PPL 主锚/长上下文锚/无草稿基线；cp312 venv；底座 git 化 |
| 阶段 1 底座冒烟 | ✅ 收官 | S1 GSQ int3 裸加载 + S2 PPL 对锚 \|Δ\|≤0.0025；S3-S6 冒烟全过（A6/A8 锚补采）；两个结构缺口定性（投机解码/图模式在纯底座不可用，分别挂批 4/3b） |
| 阶段 2 自研迁移 | 🟡 **批 1-3 ✅（5 批中 3）** | 批1 协议/GC/hybrid（`57188ef`）、批2 TQ+KV（`f56c964`，KV 域零搬运）、批3 量化域（`7d4c0e1`，PPL 重锚 8/8 全绿）；批 3b = PIECEWISE 恢复（custom-op 方案已定）；批 4 投机解码、批 5 nvfp4/SM120 待做 |
| 阶段 3 0.30 甄选 | ⚪ 切换后滚动 | A 组+B7+D 组+C1/C6/C8，砍尾 C8→C1/C6→D |
| 阶段 4 回归+切换 | ⚪ 目标 10-18 | 旧 venv 冻结只读 |

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
