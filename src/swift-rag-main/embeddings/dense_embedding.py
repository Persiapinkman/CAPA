from typing import Any, Dict, List
import torch
from llama_index.core.embeddings import BaseEmbedding
# from pydantic import PrivateAttr
from llama_index.core.bridge.pydantic import PrivateAttr
from PIL import Image
from sentence_transformers import SentenceTransformer
#from config import *
#from cfg.setup_cfg import CONFIG as cfg
# from cfg.config import CONFIG as cfg


class DenseEmbeddings(BaseEmbedding):
    _model: Any = PrivateAttr()
    _instruction: str = PrivateAttr()
    _model_name: str = PrivateAttr()
    _cutof_length: int = PrivateAttr()
    _query_encode_kwargs: Dict[str, Any] = PrivateAttr()
    _image_dummy_query: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = 'moka-ai/m3e-base',
        embed_batch_size: int = 256,
        device: str = 'cuda:0',
        **kwargs: Any,
        ) -> None:
        #self._model = INSTRUCTOR(model_name)
        self._model_name = model_name
        self._cutof_length = 2048
        self._query_encode_kwargs = {}
        self._image_dummy_query = "Describe this image for retrieval."
        try:
            self._model = SentenceTransformer(model_name, device=device).eval()
        except ValueError as exc:
            if "Due to a serious vulnerability issue in `torch.load`" in str(exc):
                raise RuntimeError(
                    "模型加载失败：当前 transformers 版本禁止在 torch<2.6 的环境中通过 "
                    f"`torch.load` 加载非 safetensors 权重。模型路径: {model_name}，"
                    f"当前 torch 版本: {torch.__version__}。请升级 torch 至 >=2.6，"
                    "或为该模型提供 safetensors 权重文件。"
                ) from exc
            raise
        prompts = getattr(self._model, "prompts", None) or {}
        if "query" in prompts:
            self._query_encode_kwargs["prompt_name"] = "query"
        super().__init__(model_name=model_name, embed_batch_size=embed_batch_size, **kwargs)


    @classmethod
    def class_name(cls) -> str:
        return "dense_embedding"

    def get_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _get_query_embedding(self, query: str, normalize: bool = True) -> List[float]:
        with torch.no_grad():
            embedding = self._model.encode(
                query[0:self._cutof_length],
                normalize_embeddings=normalize,
                **self._query_encode_kwargs,
            ).tolist()
        return embedding

    def _get_text_embedding(self, text: str, normalize: bool = True) -> List[float]:
        with torch.no_grad():
            embedding = self._model.encode(text[0:self._cutof_length], normalize_embeddings=normalize).tolist()
        return embedding

    def _get_text_embeddings(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self._model.encode([text[0:self._cutof_length] for text in texts], normalize_embeddings=normalize).tolist()
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_image_embedding(self, image_path: str, normalize: bool = True) -> List[float]:
        """Get image embedding.

        This path is intentionally lightweight and keeps compatibility with the
        current SentenceTransformer-based model loading strategy.
        """
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            # NOTE:
            # Some ST wrappers infer the pipeline from the first element.
            # Supplying [text, image] keeps image encoding functional for this model.
            embeddings = self._model.encode(
                [self._image_dummy_query, image],
                normalize_embeddings=normalize,
                **self._query_encode_kwargs,
            )

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        if not isinstance(embeddings, list) or len(embeddings) < 2:
            raise RuntimeError(
                f"Image embedding failed for {image_path}: unexpected encode output."
            )
        return embeddings[-1]

    def get_image_embedding_batch(
        self,
        image_paths: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """Batch image embedding.

        Uses per-image calls to keep robustness across model wrapper variants.
        """
        return [
            self._get_image_embedding(image_path=path, normalize=normalize)
            for path in image_paths
        ]
