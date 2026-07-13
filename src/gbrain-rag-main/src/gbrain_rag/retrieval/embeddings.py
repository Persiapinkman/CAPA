from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from gbrain_rag.core.config import Settings, get_settings
from gbrain_rag.core.text import text_tokens

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "hashing", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in text_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row_idx, bucket] += sign
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str, model_path: Path, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.model = SentenceTransformer(str(model_path), device=self.device).eval()
        self.dim = int(self.model.get_sentence_embedding_dimension())
        prompts = getattr(self.model, "prompts", None) or {}
        self.query_kwargs = {"prompt_name": "query"} if "query" in prompts else {}

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            **self.query_kwargs,
        )
        return np.asarray(embeddings, dtype=np.float32)


class EmbeddingManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cache: dict[tuple[str, str], EmbeddingBackend] = {}

    def get(self, model_name: str | None = None, backend_name: str | None = None) -> EmbeddingBackend:
        model = model_name or self.settings.EMBEDDING_MODEL
        backend = (backend_name or self.settings.EMBEDDING_BACKEND).lower()
        key = (backend, model)
        if key in self._cache:
            return self._cache[key]

        if backend == "hashing":
            result: EmbeddingBackend = HashingEmbeddingBackend(model_name=model, dim=self.settings.HASHING_DIM)
            self._cache[key] = result
            return result

        if backend in {"sentence-transformers", "sentence_transformers", "auto"}:
            model_path = self.settings.model_path_for(model)
            try:
                result = SentenceTransformerBackend(
                    model_name=model,
                    model_path=model_path,
                    device=self.settings.EMBEDDING_DEVICE,
                )
                self._cache[key] = result
                return result
            except Exception as exc:
                if backend != "auto" or not self.settings.ALLOW_HASHING_FALLBACK:
                    raise
                logger.warning(
                    "Falling back to hashing embeddings because %s could not load from %s: %s",
                    model,
                    model_path,
                    exc,
                )
                result = HashingEmbeddingBackend(model_name=model, dim=self.settings.HASHING_DIM)
                self._cache[key] = result
                return result

        raise ValueError(f"Unsupported embedding backend: {backend_name}")


def _resolve_device(device: str | None) -> str:
    requested = str(device or "cuda").strip() or "cuda"
    if requested.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA was requested for embeddings but is unavailable; falling back to CPU.")
                return "cpu"
            if requested == "cuda":
                free_by_device: list[tuple[int, int]] = []
                for idx in range(torch.cuda.device_count()):
                    try:
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
                    except Exception:
                        continue
                    free_by_device.append((int(free_bytes), idx))
                if free_by_device:
                    _free, idx = max(free_by_device)
                    return f"cuda:{idx}"
        except Exception as exc:
            logger.warning("Could not check CUDA availability for embeddings (%s); falling back to CPU.", exc)
            return "cpu"
    return requested


@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
