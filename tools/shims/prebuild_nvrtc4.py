import json, os, shutil
from pathlib import Path
import humming.utils.jit as jit_utils
from humming.utils.nvrtc import _find_nvrtc_lib_dir, may_build_nvrtc_compile_binary

mod = __import__('humming.utils.nvrtc', fromlist=['x'])
src_path = os.path.abspath(os.path.join(os.path.dirname(mod.__file__), '..', 'csrc', 'nvrtc_compile.cpp'))
lib_dir, lib_path, cuda_env = _find_nvrtc_lib_dir()
src_hash = jit_utils.hash_path_content(src_path, releative=True)
include_paths = list(cuda_env["include_paths"])
env_signature = json.dumps({"lib_dir": lib_dir, "lib_path": lib_path,
                            "include_paths": include_paths, "path": cuda_env["path"]},
                           sort_keys=True, ensure_ascii=False)
env_signature += jit_utils.get_native_platform_signature()
full_hash = jit_utils.hash_to_hex(src_hash + "$$" + env_signature)
build_dir = Path(jit_utils.get_humming_cache_dir()) / "nvrtc_compile" / full_hash
build_dir.mkdir(parents=True, exist_ok=True)
dst = build_dir / "nvrtc_compile"
if not dst.exists():
    shutil.copy2(r"C:\fi\.humming\cache\nvrtc_compile\1b56ebe95fa3ab0b\nvrtc_compile.exe", dst)
    print("copied ->", dst)
p = may_build_nvrtc_compile_binary()
print("may_build ->", p, "| exists:", os.path.exists(p))
