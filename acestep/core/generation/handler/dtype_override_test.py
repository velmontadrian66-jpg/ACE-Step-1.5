"""Tests for the ``ACESTEP_DTYPE`` CUDA dtype override."""

import os
import unittest
from unittest.mock import patch

import torch

from acestep.core.generation.handler.dtype_override import resolve_cuda_dtype_override


class ResolveCudaDtypeOverrideTests(unittest.TestCase):
    """Behavior of ``resolve_cuda_dtype_override``."""

    def test_returns_float32_when_requested(self):
        """``ACESTEP_DTYPE=float32`` forces full precision."""
        with patch.dict(os.environ, {"ACESTEP_DTYPE": "float32"}):
            self.assertIs(resolve_cuda_dtype_override(), torch.float32)

    def test_accepts_short_alias_case_insensitive(self):
        """Short aliases such as ``FP16`` are accepted."""
        with patch.dict(os.environ, {"ACESTEP_DTYPE": " FP16 "}):
            self.assertIs(resolve_cuda_dtype_override(), torch.float16)

    def test_returns_none_when_unset(self):
        """Without the variable, automatic dtype selection is preserved."""
        env = {k: v for k, v in os.environ.items() if k != "ACESTEP_DTYPE"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(resolve_cuda_dtype_override())

    def test_returns_none_for_unknown_value(self):
        """Unknown values are ignored rather than raising."""
        with patch.dict(os.environ, {"ACESTEP_DTYPE": "int4"}):
            self.assertIsNone(resolve_cuda_dtype_override())


if __name__ == "__main__":
    unittest.main()
