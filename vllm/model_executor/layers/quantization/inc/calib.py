# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibration Hessian capture for GPTQ (DFlash2 plan v2, C1).

Enabled only when ``DFLASH2_CALIB_DIR`` is set; otherwise nothing here runs.
For every ``LinearBase`` submodule of the draft model a forward_pre_hook
buffers the layer input ``x`` ([n, K], rows = samples) and periodically
accumulates the running Hessian ``H += (2/N) * X^T X`` in fp32 **on the CPU**
(the fc layer alone is K=25600 -> 2.44 GiB). The running convention matches
``gptq_utils.accumulate_hessian`` exactly (gptq_quantize is invariant to the
overall scale of H, so either normalisation works).

Env vars:
  DFLASH2_CALIB_DIR        output directory; setting it enables collection
  DFLASH2_CALIB_LAYERS     comma-separated name substrings to include (default all);
                           a leading '!' excludes instead (exclusions win)
  DFLASH2_CALIB_FLUSH_ROWS rows buffered per X^T X pass (default 512)
  DFLASH2_CALIB_SAVE_EVERY flushes between disk checkpoints (default 30;
                           a hard-killed process loses at most this much)

Design constraints honoured (plan v2 section 2): real joint forwards only
(hooks live on the running draft model -- never offline), no batch-size
filtering (draft forwards are always tiny), module typing not name prefixes.
"""
from __future__ import annotations

import atexit
import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

_COLLECTORS: dict[str, "HessianCollector"] = {}
_SAVE_EVERY = 30


class HessianCollector:
    """Running (2/n) X^T X accumulator for one layer's inputs, CPU fp32.

    K is taken from the first observed input (lazy): the module geometry
    (`input_size`, `weight.shape`) can disagree with what actually reaches
    the forward (fused/derived Linear wrappers), and a wrong K here used to
    kill the engine inside the first flush.
    """

    def __init__(self, name: str, out_dir: str, flush_rows: int):
        self.name = name
        self.k = -1
        self.out_dir = out_dir
        self.flush_rows = flush_rows
        self.H: torch.Tensor | None = None
        self.n_seen = 0
        self.buf: list[torch.Tensor] = []
        self.buf_rows = 0
        self.flushes = 0
        self.mismatched = 0
        self.dropped = 0
        self.path = os.path.join(out_dir, name.replace(".", "__") + ".pt")

    def add(self, x: torch.Tensor) -> None:
        x = x.detach()
        # Draft block-diffusion feeds all-mask rows through the stack; the
        # attention softmax makes those rows NaN (they are discarded at the
        # output, so generation is fine) and one NaN row poisons H forever.
        # Keep only rows that are real inputs in every column.
        finite = torch.isfinite(x).all(dim=-1)
        if not bool(finite.all()):
            self.dropped += int((~finite).sum())
            x = x[finite]
            if x.numel() == 0:
                return
        if x.shape[-1] != self.k:
            if self.k < 0:
                self.k = x.shape[-1]
                self.H = torch.zeros(self.k, self.k, dtype=torch.float32)
            else:
                self.mismatched += 1
                if self.mismatched <= 3:
                    logger.warning(
                        "calib: %s got x with K=%d but collector K=%d "
                        "(%d skipped so far)",
                        self.name, x.shape[-1], self.k, self.mismatched,
                    )
                return
        self.buf.append(x.to("cpu", torch.float32))
        self.buf_rows += x.shape[0]
        if self.buf_rows >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf or self.H is None:
            return
        x = torch.cat(self.buf, dim=0)
        self.buf.clear()
        self.buf_rows = 0
        n = x.shape[0]
        # same running convention as gptq_utils.accumulate_hessian
        self.H *= self.n_seen / (self.n_seen + n)
        self.n_seen += n
        x = (2.0 / self.n_seen) ** 0.5 * x
        self.H += x.t() @ x
        self.flushes += 1
        if self.flushes % _SAVE_EVERY == 0:
            self.save()

    def save(self) -> None:
        self.flush()
        if self.H is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        torch.save({"H": self.H, "n_seen": self.n_seen, "name": self.name,
                    "dropped": self.dropped}, tmp)
        os.replace(tmp, self.path)


def _save_all() -> None:
    for c in _COLLECTORS.values():
        try:
            c.save()
        except Exception:  # noqa: BLE001 -- atexit must not raise
            logger.exception("calib: failed to save %s", c.name)


def install_calib_hooks(root: torch.nn.Module) -> int:
    """Attach collectors to every LinearBase under *root*. Returns count."""
    from vllm.model_executor.layers.linear import LinearBase

    out_dir = os.environ["DFLASH2_CALIB_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    raw = [s for s in os.environ.get("DFLASH2_CALIB_LAYERS", "").split(",") if s]
    want = [s for s in raw if not s.startswith("!")]
    deny = [s[1:] for s in raw if s.startswith("!")]
    flush_rows = int(os.environ.get("DFLASH2_CALIB_FLUSH_ROWS", "512"))
    global _SAVE_EVERY
    _SAVE_EVERY = int(os.environ.get("DFLASH2_CALIB_SAVE_EVERY", "30"))

    n_installed = 0
    for name, mod in root.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        if any(s in name for s in deny):
            continue
        if want and not any(s in name for s in want):
            continue
        col = HessianCollector(name, out_dir, flush_rows)
        _COLLECTORS[name] = col

        def pre_hook(module, args, _name=name):
            x = args[0] if args else None
            if isinstance(x, torch.Tensor) and x.ndim >= 2:
                _COLLECTORS[_name].add(x.reshape(-1, x.shape[-1]))

        mod.register_forward_pre_hook(pre_hook)
        n_installed += 1

    atexit.register(_save_all)
    logger.info(
        "calib: capturing %d LinearBase modules -> %s (layers filter=%s)",
        n_installed,
        out_dir,
        raw or "all",
    )
    return n_installed
