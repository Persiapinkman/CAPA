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

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    if not getattr(Qwen3_5Config, "_capa_text_config_fallback", False):
        _orig_getattr = getattr(Qwen3_5Config, "__getattr__", None)

        def _qwen35_getattr(self, name):  # type: ignore[no-untyped-def]
            text_config = self.__dict__.get("text_config")
            if text_config is not None and hasattr(text_config, name):
                return getattr(text_config, name)
            if _orig_getattr is not None:
                return _orig_getattr(self, name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        Qwen3_5Config.__getattr__ = _qwen35_getattr  # type: ignore[method-assign]
        Qwen3_5Config._capa_text_config_fallback = True  # type: ignore[attr-defined]
except Exception:
    pass
