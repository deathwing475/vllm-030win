# humming-windows 0.1.15 + 0.29 底座 Windows 垫片清单（步骤 010 战利品）

> 这些是让 GSQ int3 在 0.29 底座跑通所做的第三方包修复/移植。全部为通用修复，
> 未来给 SystemPanic/humming-windows 提 PR 或复装环境时直接照此重放。
> 应用位置 = `G:\qwen3.8model\vllm-win029\Lib\site-packages\...`；生产运行依赖它们。

| # | 文件（仓库内） | 应用到 | 内容 | 性质 |
|---|---|---|---|---|
| 1 | `mapped_file.h.win32` | `humming/csrc/launcher/mapped_file.h` | mmap → Win32 CreateFileMapping 双实现（**含 NOMINMAX**，否则 windows.h 的 min/max 宏毒害 torch 头） | 源码移植 |
| 2 | `humming_ops_utils.patched.py` | `humming/ops/utils.py` | `extra_ldflags` GNU 旗标翻译成 `cuda.lib/c10_cuda.lib/torch_cuda.lib` + `/LIBPATH:CUDA\lib\x64`；`extra_cflags` 加 `/Zc:__cplusplus /Zc:preprocessor`（治 torch 重写 build.ninja） | 构建参数修复 |
| 3 | `humming_nvrtc.patched.py` | `humming/utils/nvrtc.py` | `_select_nvrtc_lib` 加 `nvrtc*.dll` glob；候选目录加 `bin/x64`、`bin`（Windows 只有 `nvrtc64_*.dll`，无 `libnvrtc.so`） | 定位器修复 |
| 4 | `humming_device.patched.py` | `humming/utils/device.py` | `extension_name` Windows 下用 `.pyd`（importlib 不认 `.so` 后缀） | 加载器修复 |
| 5 | `build_devinfo.cmd` | 手工构建 device_info 扩展 | clang++ 命令（去掉 -fPIC、补 python libs 路径），产物进 `C:\fi\.humming\cache\device_info\3a3f1e33f47a4caa\_device_info.abi3.{so,pyd}` | 一次性构建 |
| 6 | `prebuild_nvrtc4.py` | 移植 nvrtc_compile 辅助工具 | 用 humming 自己的哈希函数算出缓存路径，把旧缓存的独立 exe（与 Python ABI 无关）拷入 `C:\fi\.humming\cache\nvrtc_compile\<hash>\nvrtc_compile`（**无扩展名**！） | 一次性移植 |
| 7 | `run_ninja.cmd` | 手动构建 humming_launcher | vcvars64 + python libs 进 LIB 后跑 ninja | 调试工具 |
| 8 | `qwen3_5.py.embed-patched` | `vllm/model_executor/models/qwen3_5.py` | quantized embedding 2 行（`quant_config`+`prefix` 进 VocabParallelEmbedding）——底座 commit `7bb4ce0` | vllm 补丁 |

**通用教训**：humming-kernels 0.1.15（SystemPanic humming-windows@v0.1.15）**并未真正移植 Windows**——
csrc 里 dlfcn.h/sys/mman.h 照用、构建器 GNU 旗标、`.so` 命名。凡在 Windows 上重装此包，
这 8 件都要重放。缓存在 `C:\fi\.humming\cache\`（HOME=C:\fi 是环境契约的一部分）。
