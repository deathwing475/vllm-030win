# tools/scratch — 实验驱动归档（只作可复现副本，不是维护入口）

这些文件是 `G:\qwen3.8model\_tmp_line_b\` 的**逐字节副本**，2026-10-18 步骤 108 Phase 0
收口《换新底座资产清单与止损执行预案》§10 gap 2（scratch 不入 git，清理后需重写）。
它们保留原路径语义（硬编码 `_tmp_line_b` / `prod029_logs`），**不要在仓库里直接改**；
现役扫描入口是 `tools/kvmem108_scan.py`。

| 文件 | 原路径 | SHA-256 |
|---|---|---|
| `boot_keep.py` | `_tmp_line_b/boot_keep.py` | `be3f2b0fec17149d47d90b130577ae6a7ae791b6852850977ae75738e19a1969` |
| `ctx_arm.py` | `_tmp_line_b/ctx_arm.py` | `6461abdf2ce0d4ae1a0645b3d1c1293e542601682c81529d948c8f4a90b847e7` |
| `mk_arm.py` | `_tmp_line_b/mk_arm.py` | `1ce31d3a473ffd61de229a6bdf5c5cbcf084f80dfe8009163b1d35bfda76cebf` |
| `mk_nospec.py` | `_tmp_line_b/mk_nospec.py` | `042d683e4fe0083d42890fb14dae4500a763b41a61a1e1cef3d5a5ea1ad6b9f1` |
| `roundB_082.py` | `_tmp_line_b/roundB_082.py` | `14abde7a1beccbc13bd4598aeeb428c37a9b9a7eff767093aa015b83a7ad0fd7` |
| `roundE1_082.py` | `_tmp_line_b/roundE1_082.py` | `007cc4cfa06cda441a382a21b774d15fe38ac73aeb79355f62edd0c2daf58d20` |

用途：
- `boot_keep.py`：`cmd.exe /d /c <launcher>` 脱机起服（只吃 `.cmd`，cwd 固定 `G:\qwen3.8model`）；现役路径仍是 `_tmp_line_b\boot_keep.py`（必守 2/3 的原文）。
- `ctx_arm.py` / `mk_arm.py` / `mk_nospec.py`：从生产 launcher 派生 G/L/N/MBT 变体并跑 anchor；步骤 108 的新臂改用 `tools/make_kvmem108_arms.py`。
- `roundB_082.py` / `roundE1_082.py`：082 的多深度视窗与 083 的五项出口编排骨架；步骤 108 的召回腿复用其“每深度 reset + 单 boot”思路。

已收进仓库、替代 scratch 的维护版：
- `tools/make_kvmem108_arms.py`：步骤 108 两条臂的生成器（CRLF 断言 + 替换命中数断言）。
- `tools/kvmem108_scan.py`：boot / anchor / recall / collect / sample / stop 编排。
- `tools/patch_flashinfer_page.py`：flashinfer `page.py` 补丁的 apply / revert / verify（§10 gap 1）。

复现检查：

```bash
cd G:/qwen3.8model/vllm-030win-git
sha256sum tools/scratch/*.py
```

六个哈希必须与上表一致；不一致说明副本被改过，以 `_tmp_line_b` 的原件为准。
