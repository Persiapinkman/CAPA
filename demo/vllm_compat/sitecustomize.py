"""Runtime compatibility patches for local vLLM serving.

This file is loaded automatically when its directory is added to PYTHONPATH.
It keeps the project environment unchanged while smoothing small API
differences between vLLM 0.10.x and the local Transformers development build.
"""

from __future__ import annotations

try:
    import torch

    torch.backends.cudnn.enabled = False
except Exception:
    pass

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
            return list(getattr(self, "all_special_tokens", []) or [])

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended  # type: ignore[attr-defined]
except Exception:
    pass
