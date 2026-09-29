# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-030win patch (step 060): KVMem workspace configuration.

Every knob is read from the environment, matching the ``VLLM_KVMEM_SW_WINDOW``
convention the step 057 patch established. Unset values keep the KVMem arm on
its documented defaults; the whole subsystem stays inert unless
``VLLM_KVMEM_WORKSPACE`` is armed, which is checked in the core KV cache
manager.
"""

import os

DEFAULT_WORKSPACE_MB = 3072
DEFAULT_TRAJECTORY_PREFIX_TOKENS = 512
DEFAULT_MAX_WORKSPACE_TOKENS = 262144


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def workspace_host_bytes() -> int:
    """Host bytes the workspace may occupy (the pinned store's budget)."""
    return _env_int("VLLM_KVMEM_WORKSPACE_MB", DEFAULT_WORKSPACE_MB) << 20


def trajectory_prefix_tokens() -> int:
    """How many leading prompt tokens identify a trajectory.

    The client resends the whole conversation, so the first tokens are the root
    task message: they stay fixed while the conversation grows, and differ
    between conversations. That is the design's "hash(root task message
    identity)" without touching the client.
    """
    return _env_int(
        "VLLM_KVMEM_TRAJ_PREFIX", DEFAULT_TRAJECTORY_PREFIX_TOKENS
    )


def max_workspace_tokens() -> int:
    """Workspace capacity ceiling in tokens (design §7.2: 262,144)."""
    return _env_int("VLLM_KVMEM_WORKSPACE_TOKENS", DEFAULT_MAX_WORKSPACE_TOKENS)


def roundtrip_selftest() -> bool:
    """Copy each stored page back and compare it byte for byte (K1 exit)."""
    return bool(int(os.environ.get("VLLM_KVMEM_SELFTEST", "0")))


def dump_dir() -> str | None:
    """Directory for the page-table / occupancy dump, or None."""
    raw = os.environ.get("VLLM_KVMEM_DUMP", "").strip()
    return raw or None
