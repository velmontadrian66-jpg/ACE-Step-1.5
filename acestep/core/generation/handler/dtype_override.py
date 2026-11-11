"""Optional user override of the CUDA model dtype via ``ACESTEP_DTYPE``."""

import os
from typing import Optional

import torch
from loguru import logger

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_cuda_dtype_override() -> Optional[torch.dtype]:
    """Return the dtype requested through ``ACESTEP_DTYPE``, if any.

    Pre-Ampere GPUs (e.g. Tesla T4) run in float16 by default, which can
    overflow and produce NaN latents. Setting ``ACESTEP_DTYPE=float32`` forces
    full precision for the DiT and VAE.

    Returns:
        The requested ``torch.dtype``, or ``None`` when the variable is unset
        or holds an unknown value.
    """
    raw = os.environ.get("ACESTEP_DTYPE", "").strip().lower()
    if not raw:
        return None
    dtype = _DTYPE_MAP.get(raw)
    if dtype is None:
        logger.warning(f"[dtype_override] Unknown ACESTEP_DTYPE={raw!r}; ignoring it.")
    return dtype
