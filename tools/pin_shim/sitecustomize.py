# -*- coding: utf-8 -*-
"""vLLM 加载/编译递归树分支打点器（debug 启动器用）。

由 PYTHONPATH 指向本目录自动加载；仅当 VLLM_DBG_TRACE=1 时启用打点。
打点目标（分支走向 + 产物哈希 + 耗时）：
  1. vllm.compilation.decorators._try_load_aot_compiled_fn   (AOT 载入分支)
  2. vllm.compilation.decorators._verify_source_unchanged
  3. vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper
     .compile / .aot_compile / .call                          (现场编译分支)
  4. vllm.compilation.backends.VllmBackend.call               (piecewise 编译驱动)
  5. vllm.compilation.piecewise_backend.PiecewiseBackend
     .init / .call            (逐 piece：graph 路由 vs compiled_runnables 路由)
  6. vllm.compilation.codegen.generate_execution_code /
     compile_execution_fn     (缝合代码哈希)
  7. vllm.compilation.caching.VllmSerializableFunction
     .finalize_loading / reconstruct_serializable_fn_from_mega_artifact
  8. torch.compiler.load_compiled_function
  9. torch._inductor.codecache.PyCodeCache.load                (inductor 产物装载)
日志：pydbg/logs/{TAG}_pid{PID}.log，逐行刷新。
"""
import hashlib
import os
import sys
import threading
import time
import traceback

if os.environ.get("VLLM_DBG_TRACE") == "1":
    _MIN = os.environ.get("VLLM_DBG_MIN") == "1"
    if not _MIN:
        import faulthandler
        try:
            # 每 120s 把全线程栈打进 stderr（进程死锁时直接看卡点）
            faulthandler.dump_traceback_later(120, repeat=True)
        except Exception:
            pass
    import ctypes
    import torch
    import torch.nn as nn
    _T0 = time.perf_counter()
    _TAG = os.environ.get("VLLM_DBG_TAG", "boot")
    _LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(_LOGDIR, exist_ok=True)
    _LOG = os.path.join(_LOGDIR, "%s_pid%d.log" % (_TAG, os.getpid()))
    _LOCK = threading.Lock()
    _COUNTS = {}

    def log(evt, **kv):
        try:
            parts = ["T=%9.3f" % (time.perf_counter() - _T0), evt]
            for k, v in kv.items():
                parts.append("%s=%s" % (k, v))
            line = " ".join(parts)
            with _LOCK:
                with open(_LOG, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    def sha(s):
        if s is None:
            return "none"
        return hashlib.sha1(str(s).encode("utf-8", "replace")).hexdigest()[:12]

    def capped(name, limit=60):
        n = _COUNTS.get(name, 0) + 1
        _COUNTS[name] = n
        return n <= limit or n % 200 == 0

    def wrap(mod, fname, label, kind="call"):
        fn = getattr(mod, fname, None)
        if fn is None or getattr(fn, "_dbg_wrapped", False):
            return
        import functools

        if kind == "classmethod":
            orig = fn.__func__

            @functools.wraps(orig)
            def w(cls, *a, **k):
                t = time.perf_counter()
                log(label + ".enter", cls=cls.__name__)
                try:
                    r = orig(cls, *a, **k)
                    log(label + ".exit", ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                        ret=type(r).__name__)
                    return r
                except Exception as e:
                    log(label + ".raise", ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                        err=repr(e)[:160])
                    raise

            setattr(mod, fname, classmethod(w))
        else:
            @functools.wraps(fn)
            def w(*a, **k):
                t = time.perf_counter()
                log(label + ".enter", na=len(a), nk=len(k))
                try:
                    r = fn(*a, **k)
                    if capped(label):
                        log(label + ".exit",
                            ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                            ret=type(r).__name__)
                    return r
                except Exception as e:
                    log(label + ".raise", ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                        err=repr(e)[:160])
                    raise

            setattr(mod, fname, w)
        getattr(mod, fname)._dbg_wrapped = True

    def wrap_method(cls, mname, label):
        fn = getattr(cls, mname, None)
        if fn is None or getattr(fn, "_dbg_wrapped", False):
            return
        import functools

        orig = fn
        if isinstance(orig, (classmethod, staticmethod)):
            orig = orig.__func__

        @functools.wraps(orig)
        def w(self, *a, **k):
            t = time.perf_counter()
            if capped(label):
                log(label + ".enter",
                    submod=getattr(self, "submod_name", "-"),
                    idx=getattr(self, "piecewise_compile_index", "-"),
                    graph=self.graph is not None if hasattr(self, "graph") else "-",
                    runnables=bool(getattr(self, "compiled_runnables", None)),
                    na=len(a))
            try:
                r = orig(self, *a, **k)
                if capped(label):
                    log(label + ".exit",
                        ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                        submod=getattr(self, "submod_name", "-"),
                        ret=type(r).__name__)
                return r
            except Exception as e:
                log(label + ".raise",
                    ms="%.1f" % ((time.perf_counter() - t) * 1e3),
                    err=repr(e)[:160])
                raise

        w._dbg_wrapped = True
        setattr(cls, mname, w)

    _CUDART = None
    _PT_N = [0]

    def _ptr_type(t):
        """cudaPointerGetAttributes 内存类型：0=Unregistered 1=Host 2=Device 3=Managed。"""
        global _CUDART
        import ctypes
        _PT_N[0] += 1
        n = _PT_N[0]
        dbg = n <= 30
        try:
            if dbg:
                log("PT.enter", n=n)
            if _CUDART is None:
                if dbg:
                    log("PT.build_cudart", n=n)
                k32 = ctypes.WinDLL("kernel32")
                k32.GetModuleHandleW.restype = ctypes.c_void_p
                k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
                h = k32.GetModuleHandleW("cudart64_13.dll")
                if dbg:
                    log("PT.handle", n=n, h=str(h))
                if h:
                    dll = ctypes.WinDLL("cudart64_13.dll", handle=h)
                else:
                    dll = ctypes.CDLL(
                        os.path.join(os.path.dirname(torch.__file__),
                                     "lib", "cudart64_13.dll"))
                class _Attr(ctypes.Structure):
                    _fields_ = [("type", ctypes.c_int),
                                ("device", ctypes.c_int),
                                ("devicePointer", ctypes.c_void_p),
                                ("hostPointer", ctypes.c_void_p)]
                fn = dll.cudaPointerGetAttributes
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.POINTER(_Attr), ctypes.c_void_p]
                _CUDART = (fn, _Attr)
                if dbg:
                    log("PT.built", n=n)
            fn, Attr = _CUDART
            if dbg:
                log("PT.before_call", n=n, ptr=str(t.data_ptr()))
            a = Attr()
            rc = fn(ctypes.byref(a), t.data_ptr())
            if dbg:
                log("PT.after_call", n=n, rc=rc, typ=a.type)
            return "%d" % a.type if rc == 0 else "rc%d" % rc
        except Exception as e:
            return "err:%s" % type(e).__name__

    def census_and_force(model, tag):
        """参数/缓冲指针属性普查；VLLM_DBG_FORCE_GPU=1 时把非 Device 的拷回显存。"""
        pass  # imports hoisted at module top
        force = os.environ.get("VLLM_DBG_FORCE_GPU") == "1"
        moved, total, nondev = 0, 0, 0
        seen = set()
        for name, mod in model.named_modules():
            for pname, p in list(mod.named_parameters(recurse=False)) + \
                    list(mod.named_buffers(recurse=False)):
                if id(p) in seen:
                    continue
                seen.add(id(p))
                total += 1
                if total <= 30 or total % 50 == 0:
                    log("CENSUS_PROGRESS", tag=tag, total=total,
                        at=name + "." + pname, nbytes=p.numel() * p.element_size())
                pt = _ptr_type(p)
                if pt != "2":
                    nondev += 1
                    log("PLACE", tag=tag, name=name + "." + pname,
                        nbytes=p.numel() * p.element_size(), ptr=pt)
                    if force and isinstance(p, nn.Parameter):
                        try:
                            new = torch.empty_like(p.data, device="cuda:0")
                            new.copy_(p.data)
                            p.data = new
                            moved += 1
                            log("FORCE_MOVED", tag=tag, name=name + "." + pname,
                                nbytes=p.numel() * p.element_size(),
                                after=_ptr_type(p))
                        except Exception as e:
                            log("FORCE_FAIL", tag=tag, name=name + "." + pname,
                                err=repr(e)[:120])
                    elif force:
                        try:
                            new = torch.empty_like(p, device="cuda:0")
                            new.copy_(p)
                            setattr(mod, pname, new)
                            moved += 1
                            log("FORCE_MOVED", tag=tag, name=name + "." + pname,
                                nbytes=p.numel() * p.element_size(),
                                after=_ptr_type(new))
                        except Exception as e:
                            log("FORCE_FAIL", tag=tag, name=name + "." + pname,
                                err=repr(e)[:120])
        log("CENSUS", tag=tag, total=total, nondev=nondev, moved=moved,
            force=force)

    def _mem():
        """显存快照：free=驱动专用显存余量，alloc=torch 记账。
        落 sysmem 的块 signature = alloc 升而 free 不降（两计数器背离）。"""
        try:
            free, total = torch.cuda.mem_get_info()
            return ("free=%.0f alloc=%.0f resv=%.0f"
                    % (free / 2**20, torch.cuda.memory_allocated() / 2**20,
                       torch.cuda.memory_reserved() / 2**20))
        except Exception:
            return "mem=err"

    def wrap_load_dflash(orig, *a, **k):
        t = time.perf_counter()
        log("load_dflash_model.enter", mem=_mem())
        r = orig(*a, **k)
        log("load_dflash_model.exit",
            ms="%.1f" % ((time.perf_counter() - t) * 1e3), mem=_mem())
        if os.environ.get("VLLM_DBG_PIN") == "1":
            try:
                n, moved, bytes_ = 0, 0, 0
                for name, p in r.named_parameters():
                    n += 1
                    if p.numel() * p.element_size() < 1 << 20:
                        continue
                    # 瞬态已释放（free 回升），此时新分配落专用显存；
                    # 逐层搬（峰值 2x 单层），绝不满量克隆（force-all 毒）。
                    new = torch.empty_like(p.data, device="cuda:0")
                    new.copy_(p.data)
                    p.data = new
                    moved += 1
                    bytes_ += p.numel() * p.element_size()
                log("PIN_MOVED", moved=moved, params=n, mb=bytes_ // 2**20,
                    mem=_mem())
            except Exception as e:
                log("PIN_FAIL", err=repr(e)[:160])
        return r

    def wrap_repack(self, orig, *a, **k):
        """每层 repack 前后 mem 快照——丢层（落 sysmem）定位。"""
        t = time.perf_counter()
        name = getattr(self, "layer_config", None)
        log("repack.enter", submod=str(getattr(self, "_layer_config_str", ""))[:60],
            mem=_mem())
        r = orig(self, *a, **k)
        log("repack.exit",
            ms="%.1f" % ((time.perf_counter() - t) * 1e3), mem=_mem())
        return r

    def wrap_aot_mem(orig, *a, **k):
        t = time.perf_counter()
        log("aot_mem.enter", mem=_mem())
        r = orig(*a, **k)
        log("aot_mem.exit",
            ms="%.1f" % ((time.perf_counter() - t) * 1e3), mem=_mem())
        return r

    _LAUNCH_N = [0]

    def _census_args(args, tag):
        import torch
        out = []

        def walk(x, depth=0):
            if isinstance(x, torch.Tensor):
                out.append(("t", tuple(x.shape), x.numel() * x.element_size(),
                            _ptr_type(x)))
            elif isinstance(x, (list, tuple)) and depth < 3:
                for y in x:
                    walk(y, depth + 1)

        walk(args)
        return out

    def wrap_launch(orig, *a, **k):
        _LAUNCH_N[0] += 1
        n = _LAUNCH_N[0]
        if n <= 24 or n % 200 == 0:
            try:
                items = _census_args(a, "launch")
                log("LAUNCH_ARGS", n=n,
                    args=" | ".join("shape=%s bytes=%d ptr=%s" % (s, b, p)
                                    for _, s, b, p in items[:6]))
            except Exception:
                log("LAUNCH_ARGS_FAIL", n=n,
                    tb=traceback.format_exc()[-200:])
        return orig(*a, **k)

    def _patch_module(mod_name, jobs):
        """patch 一个已导入模块上的目标；零 import（避免与主线程导入竞态）。"""
        mod = sys.modules.get(mod_name)
        if mod is None:
            return 0
        n = 0
        for kind, obj_name, m_name, label in jobs:
            try:
                if kind == "custom":
                    fn = getattr(mod, obj_name, None)
                    if fn is not None and not getattr(fn, "_dbg_wrapped", False):
                        import functools

                        @functools.wraps(fn)
                        def w(*a, **k):
                            return wrap_load_dflash(fn, *a, **k)

                        w._dbg_wrapped = True
                        setattr(mod, obj_name, w)
                elif kind == "custom2":
                    fn = getattr(mod, obj_name, None)
                    if fn is not None and not getattr(fn, "_dbg_wrapped", False):
                        import functools

                        @functools.wraps(fn)
                        def w(*a, **k):
                            return wrap_launch(fn, *a, **k)

                        w._dbg_wrapped = True
                        setattr(mod, obj_name, w)
                elif kind == "memfn":
                    fn = getattr(mod, obj_name, None)
                    if fn is not None and not getattr(fn, "_dbg_wrapped", False):
                        import functools

                        @functools.wraps(fn)
                        def w(*a, **k):
                            return wrap_aot_mem(fn, *a, **k)

                        w._dbg_wrapped = True
                        setattr(mod, obj_name, w)
                elif kind == "repack":
                    cls = getattr(mod, obj_name, None)
                    fn = getattr(cls, m_name, None) if cls is not None else None
                    if fn is not None and not getattr(fn, "_dbg_wrapped", False):
                        import functools

                        orig = fn

                        @functools.wraps(orig)
                        def w(self, *a, **k):
                            return wrap_repack(self, orig, *a, **k)

                        w._dbg_wrapped = True
                        setattr(cls, m_name, w)
                elif kind == "fn":
                    wrap(mod, obj_name, label)
                else:
                    cls = getattr(mod, obj_name, None)
                    if cls is not None:
                        wrap_method(cls, m_name, label)
                n += 1
            except Exception:
                log("PATCH_ERR", mod=mod_name, target=obj_name + "." + m_name,
                    tb=traceback.format_exc()[-240:])
        return n

    JOBS = {
        "vllm.compilation.decorators": [
            ("memfn", "_try_load_aot_compiled_fn", "", "load_aot"),
            ("fn", "_verify_source_unchanged", "", "verify_src"),
        ],
        "vllm.model_executor.layers.quantization.humming": [
            ("repack", "HummingLinearMethod", "process_weights_after_loading",
             "repack"),
        ],
        "vllm.compilation.wrapper": [
            ("cls", "TorchCompileWithNoGuardsWrapper", "compile", "wrapper.compile"),
            ("cls", "TorchCompileWithNoGuardsWrapper", "aot_compile",
             "wrapper.aot_compile"),
            ("cls", "TorchCompileWithNoGuardsWrapper", "__call__", "wrapper.call"),
        ],
        "vllm.compilation.backends": [
            ("cls", "VllmBackend", "__call__", "vllm_backend.call"),
        ],
        "vllm.compilation.piecewise_backend": [
            ("cls", "PiecewiseBackend", "__init__", "piece.init"),
            ("cls", "PiecewiseBackend", "__call__", "piece.call"),
        ],
        "vllm.compilation.codegen": [
            ("fn", "generate_execution_code", "", "gen_exec_code"),
            ("fn", "compile_execution_fn", "", "compile_exec_fn"),
        ],
        "vllm.compilation.caching": [
            ("cls", "VllmSerializableFunction", "finalize_loading",
             "finalize_loading"),
            ("fn", "reconstruct_serializable_fn_from_mega_artifact", "",
             "reconstruct_mega"),
        ],
        "torch.compiler": [
            ("fn", "load_compiled_function", "", "torch.load_compiled_fn"),
        ],
        "torch._inductor.codecache": [
            ("cls", "PyCodeCache", "load", "inductor.pycode_load"),
        ],
        "vllm.v1.worker.gpu.spec_decode.dflash.utils": [
            ("custom", "load_dflash_model", "", "load_dflash_model"),
        ],
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator": [
            ("custom", "load_dflash_model", "", "load_dflash_model"),
        ],
        "humming.ops.launcher": [
            ("custom2", "launch_kernel", "", "humming.launch_kernel"),
        ],
    }

    def _watch():
        try:
            log("WATCH_START", py=sys.executable, argv=" ".join(sys.argv)[:160])
            jobs_map = JOBS
            if _MIN:
                jobs_map = {k: v for k, v in JOBS.items() if "dflash" in k}
            deadline = time.time() + 1200
            done = set()
            while time.time() < deadline and len(done) < len(jobs_map):
                for mod_name, mod_jobs in jobs_map.items():
                    if mod_name in done:
                        continue
                    if mod_name in sys.modules:
                        try:
                            _patch_module(mod_name, mod_jobs)
                            done.add(mod_name)
                            log("PATCHED", mod=mod_name)
                        except Exception:
                            log("PATCH_FAIL", mod=mod_name,
                                tb=traceback.format_exc()[-240:])
                time.sleep(0.3)
            log("WATCH_DONE", patched=sorted(done))
        except Exception:
            log("WATCH_CRASH", tb=traceback.format_exc()[-400:])

    threading.Thread(target=_watch, daemon=True).start()
