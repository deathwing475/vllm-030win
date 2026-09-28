# Upstream PR bundle — SystemPanic/humming-windows @ v0.1.15

Two independent, general fixes extracted from the Windows bring-up of
humming-kernels 0.1.15 (steps 010/R1 of the vLLM 0.30-on-Windows migration;
see `../shims/SHIMS.md` for the full eight-shim list). Both are candidates
for upstream PRs — **submitting them is a user-gated action**; this directory
only holds the submission-ready material.

## PR 1 — `0001-mapped-file-win32.patch`

**File:** `humming/csrc/launcher/mapped_file.h`

The header is POSIX-only (`sys/mman.h`), so `humming_launcher` cannot compile
on Windows. The patch keeps the POSIX implementation untouched and adds a
`_WIN32` branch with the same semantics via `CreateFileW` +
`CreateFileMappingW` + `MapViewOfFile`:

- UTF-8 → wide-char conversion for the path (CJK-safe).
- `NOMINMAX` + `WIN32_LEAN_AND_MEAN` before `<windows.h>` — without
  `NOMINMAX`, windows.h's `min`/`max` macros poison torch/MSVC headers
  downstream.
- Error paths close every handle and report the Win32 error code.
- Handles are closed eagerly (file and mapping handles are only needed for
  the mapping lifetime, not the view's).

**Verification:** this exact byte content is what production builds and runs
on Windows (RTX 5070 Ti, CUDA 13.3, torch 2.11+cu130) — the launcher is
JIT-built with this header on every cache-miss boot of the vLLM 0.29-based
production stack.

## PR 2 — `0002-ops-utils-msvc-buildflags.patch`

**File:** `humming/ops/utils.py` (`init_humming_launcher`)

Two GNU-only assumptions break the JIT build of `humming_launcher` on
Windows (MSVC):

1. `extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"]` — GNU-style `-l`
   flags never reach `link.exe`, so the CUDA/torch import libraries are not
   linked at all.
2. Hand-patching flags into the generated `build.ninja` does not survive:
   torch **rewrites build.ninja on every invocation**, so the fix has to live
   at the `extra_ldflags`/`extra_cflags` level in `ops/utils.py` itself.

The patch adds a `sys.platform == "win32"` branch that translates the
libraries to MSVC form (`cuda.lib c10_cuda.lib torch_cuda.lib`) and derives
`/LIBPATH` from the same `cuda_env` that supplied the headers
(`cuda_env["path"] + lib/x64`), falling back to torch's default library
paths when the directory does not exist. `/Zc:__cplusplus` +
`/Zc:preprocessor` are appended to `extra_cflags` — required for torch's
C++17 headers under cl's default permissive mode. The POSIX branch is
byte-identical to upstream.

**Verification:** build + load verified on the dev machine (2026-09-29):
the flag block was executed verbatim from this patch's result file,
`humming_launcher` JIT-compiled from the wheel's own `launcher.cpp` into a
scratch build dir against CUDA 13.3 / torch 2.11.0+cu130, and the produced
DLL loads (all symbols resolve). Note `filter_cuda_paths` resolved to the
pip wheel environment (`nvidia/cu13`) whose `lib/x64` carries `cuda.lib`;
`c10_cuda.lib`/`torch_cuda.lib` come from torch's own default library path.

**Known delta vs the locally-shimmed variant:** the in-venv shim
(`../shims/humming_ops_utils.patched.py`, machine-specific by design)
hard-codes `/LIBPATH:C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\lib\x64` (system
toolkit). The PR version derives the path from `cuda_env` instead — more
self-consistent (headers and libs come from the same resolved environment)
and portable, but the hard-coded form is the one that has months of
production mileage; the derived form passed the build+load check above.

## How to apply / verify

```bat
git clone --branch v0.1.15 https://github.com/SystemPanic/humming-windows
cd humming-windows
git am|apply <patch>          ; or: git apply 0001-...patch
```

Sanity check on Windows: `python -c "import humming;
humming.ops.utils.init_humming_launcher()"` should JIT-build
`humming_launcher` without linker errors.

## Provenance

- Pristine v0.1.15 sources were pulled from the upstream tag and verified
  byte-exact against the installed wheel's `RECORD` sha256 (CRLF-normalized).
- Patches are LF; application via `git apply` preserves that.
