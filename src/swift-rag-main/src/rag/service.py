import json
import threading
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQuery
from pymilvus import DataType, MilvusClient

from embeddings.dense_embedding import DenseEmbeddings
from embeddings.embedding_builder import build_embedding_model
from node_parser.utils import get_node_with_embedding
from retriever.fuse_retriever import rrf_fuse_results
from doc_loader.mineru_loader import mineru_chunking

from src.core.logging import get_logger
from src.core.config import get_settings
from src.api.schemas import (
    AdelaChatRequest,
    ChunkingEmbeddingRequest,
    ChunkNodeWithEmbedding,
    ChunkNodeWithScore,
    RetrievingRequest,
    TableChatRequest,
    TableMatchedRow,
)
from src.rag.utils import chunk_into_small_nodes, process_search_results
from src.rag.bm25 import BM25Retriever
from src.rag.embedding_artifacts import EmbeddingArtifactStore
from src.rag.adela_dataset import export_adela_records_to_jsonl
from vector_stores.fuse_milvus import (
    FusedMilvusVectorStore,
    node_to_milvus_dict,
    preprocess_for_insert,
)

logger = get_logger(__name__)
settings = get_settings()


class RAGService:
    def __init__(
        self,
        use_local_milvus: Optional[Dict[str, Any]] = None,
        embedding_model_names: Optional[List[str]] = None,
        skip_failed_embedding_models: bool = False,
    ):
        # 从配置中获取embedding模型配置
        self.embedding_model_cfgs = settings.INIT_EMBEDDING_MODEL_CONFIGS
        self.embedding_model_cfg_map = {
            cfg["model_name"]: cfg for cfg in self.embedding_model_cfgs
        }
        self.embedding_model_names = embedding_model_names
        self.skip_failed_embedding_models = skip_failed_embedding_models
        self.embedding_models: Dict[str, DenseEmbeddings] = {}
        self.embedding_model_errors: Dict[str, str] = {}
        self.embedding_model_dims: Dict[str, int] = {}
        self._embedding_model_init_lock = threading.RLock()
        self._embedding_query_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._vector_store_locks: Dict[str, threading.Lock] = {}
        self._vector_store_locks_guard = threading.Lock()
        self._retrieval_runtime_lock = threading.Lock()
        initial_model_names = (
            self.embedding_model_names
            if self.embedding_model_names is not None
            else settings.DEFAULT_SINGLE_EMBEDDING_MODEL
        )
        self._init_embedding_models(initial_model_names)

        # 初始化tokenizer用于切分文本，noqa
        self.tokenizer = self.embedding_models[
            list(self.embedding_models.keys())[0]
        ]._model.tokenizer.tokenize

        # 初始化向量数据库
        self.embedding_field = "vector_data"
        self.use_local_milvus = use_local_milvus
        self.vector_store = None
        self.vector_store_dim: Optional[int] = None
        self.bm25_retriever = BM25Retriever(
            tokenizer=self._tokenize_text_thread_safe,
            embedding_field=self.embedding_field,
            resolve_vector_store_target=self._resolve_vector_store_target,
            get_vector_store_lock=self._get_vector_store_lock,
            k1=settings.BM25_K1,
            b=settings.BM25_B,
            candidate_limit=settings.BM25_CANDIDATE_LIMIT,
        )
        if self.use_local_milvus:
            self.milvus_uri = use_local_milvus.get("uri", settings.LOCAL_VECTOR_DB_URI)
            self.overwrite_vector_store = use_local_milvus.get("overwrite", settings.OVERWRITE_VECTOR_STORE)
            self.collection_name = use_local_milvus.get("collection_name", settings.LOCAL_COLLECTION_NAME)
        self.table_rows_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.table_embeddings_cache: Dict[str, np.ndarray] = {}
        self.adela_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_artifact_store = EmbeddingArtifactStore(
            settings.EMBEDDING_ARTIFACTS_DIR
        )

    def _init_vector_stores_settings(self, vector_dim: int):
        """初始化向量存储设置"""
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True
        )
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10240)
        schema.add_field(
            field_name="index_id", datatype=DataType.VARCHAR, max_length=256
        )
        schema.add_field(
            field_name="index_text", datatype=DataType.VARCHAR, max_length=20480
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(
            field_name="doc_name", datatype=DataType.VARCHAR, max_length=256
        )
        schema.add_field(
            field_name="embedding_model", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(
            field_name=self.embedding_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dim,
        )
        self.custom_schema = schema

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=self.embedding_field,
            index_type="FLAT",
            metric_type="IP",
        )

        self.custom_index_params = index_params

    def _init_vector_stores(self, vector_dim: int):
        """初始化向量存储"""
        # todo: 提供milvus查询表结构信息的接口，get_all_nodes等
        start = time.time()
        self._init_vector_stores_settings(vector_dim)
        self.vector_store = FusedMilvusVectorStore(
            uri=self.milvus_uri,
            overwrite=self.overwrite_vector_store,
            collection_name=self.collection_name,
            embedding_field=self.embedding_field,
            dim=vector_dim,  # 失效参数
            schema=self.custom_schema,
            index_params=self.custom_index_params,
        )
        self.vector_store_dim = vector_dim
        logger.info(f"Initializing vector store took: {time.time() - start:.2f} seconds")

    def _load_embedding_model(self, model_name: str):
        with self._embedding_model_init_lock:
            if model_name in self.embedding_models:
                return
            cfg = self.embedding_model_cfg_map.get(model_name)
            if cfg is None:
                message = f"Embedding model config not found: {model_name}"
                if self.skip_failed_embedding_models:
                    self.embedding_model_errors[model_name] = message
                    logger.warning(message)
                    return
                raise ValueError(message)

            logger.info(f"Loading embedding model: {cfg['model_name']} from {cfg['model_path']}")
            start = time.time()
            try:
                embedding_model = build_embedding_model(
                    model_name_or_path=cfg["model_path"],
                    batch_size=cfg["batchsize"],
                    device=cfg["device"],
                )
                self.embedding_models[cfg["model_name"]] = embedding_model
                self.embedding_model_dims[cfg["model_name"]] = embedding_model.get_dimension()
                logger.info(
                    "Embedding model %s loaded, dim=%s, time=%.2fs",
                    cfg["model_name"],
                    self.embedding_model_dims[cfg["model_name"]],
                    time.time() - start,
                )
            except Exception as exc:
                if not self.skip_failed_embedding_models:
                    raise
                self.embedding_model_errors[cfg["model_name"]] = str(exc)
                logger.warning(
                    "Skipping embedding model %s because initialization failed: %s",
                    cfg["model_name"],
                    exc,
                )

    def _init_embedding_models(self, model_names: Optional[List[str]] = None):
        """初始化embedding模型"""
        requested_model_names = model_names or []
        if self.embedding_model_names is not None:
            disallowed_model_names = set(requested_model_names) - set(self.embedding_model_names)
            if disallowed_model_names:
                missing_names = ", ".join(sorted(disallowed_model_names))
                raise ValueError(
                    f"Embedding models not enabled in this service instance: {missing_names}"
                )

        for model_name in requested_model_names:
            try:
                self._load_embedding_model(model_name)
            except Exception as exc:
                if not self.skip_failed_embedding_models:
                    raise
                self.embedding_model_errors[model_name] = str(exc)
                logger.warning(
                    "Skipping embedding model %s because initialization failed: %s",
                    model_name,
                    exc,
                )

        if not self.embedding_models:
            raise RuntimeError(
                "No embedding models were initialized successfully. "
                f"Requested models: {requested_model_names}. "
                f"Errors: {self.embedding_model_errors}"
            )

    def _ensure_embedding_models_loaded(self, model_names: List[str]):
        self._init_embedding_models(model_names)

    def _get_query_embedding_thread_safe(self, model_name: str, query: str) -> List[float]:
        if model_name not in self.embedding_models:
            raise ValueError(f"Invalid embedding model: {model_name}")
        with self._embedding_query_lock:
            return self.embedding_models[model_name].get_query_embedding(query)

    def _tokenize_text_thread_safe(self, text: str) -> List[str]:
        with self._tokenizer_lock:
            return self.tokenizer(text)

    def _get_vector_store_lock(self, target_key: str) -> threading.Lock:
        with self._vector_store_locks_guard:
            lock = self._vector_store_locks.get(target_key)
            if lock is None:
                lock = threading.Lock()
                self._vector_store_locks[target_key] = lock
            return lock

    def _get_embedding_dims(self, model_names: List[str]) -> Dict[str, int]:
        self._ensure_embedding_models_loaded(model_names)
        return {model_name: self.embedding_model_dims[model_name] for model_name in model_names}

    def _ensure_single_vector_dimension(self, model_names: List[str], scene: str) -> int:
        dims = self._get_embedding_dims(model_names)
        unique_dims = sorted(set(dims.values()))
        if len(unique_dims) > 1:
            dim_desc = ", ".join(f"{name}={dim}" for name, dim in dims.items())
            raise ValueError(
                f"{scene} 当前不支持混用不同向量维度的 embedding 模型: {dim_desc}。"
                "当前实现只有一个 `vector_data` 字段，请为不同维度模型使用独立 collection / 库，"
                "或改造成多向量字段方案。"
            )
        return unique_dims[0]

    def _ensure_local_vector_store(self, model_names: List[str]):
        if not self.use_local_milvus:
            return

        vector_dim = self._ensure_single_vector_dimension(
            model_names,
            scene="本地 Milvus 向量库",
        )
        if self.vector_store is None:
            self._init_vector_stores(vector_dim)
            return

        if self.vector_store_dim != vector_dim:
            raise ValueError(
                "当前本地 Milvus collection 的向量维度为 "
                f"{self.vector_store_dim}，但本次请求模型维度为 {vector_dim}。"
                "不同维度模型不能写入或查询同一个 `vector_data` 字段。"
            )

    def _get_collection_vector_dim(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
    ) -> int:
        schema = milvus_client.describe_collection(collection_name)
        for field in schema.get("fields", []):
            if field.get("name") != self.embedding_field:
                continue

            params = field.get("params", {}) or {}
            dim = params.get("dim")
            if dim is None:
                break
            return int(dim)

        raise ValueError(
            f"未能从 collection `{collection_name}` 的字段 `{self.embedding_field}` 中读取向量维度。"
        )

    def _ensure_collection_vector_dim_matches(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        model_names: List[str],
        uri: str,
    ) -> int:
        model_dim = self._ensure_single_vector_dimension(model_names, scene="检索")
        collection_dim = self._get_collection_vector_dim(milvus_client, collection_name)
        if collection_dim != model_dim:
            model_desc = ", ".join(
                f"{model_name}={self.embedding_model_dims[model_name]}"
                for model_name in model_names
            )
            raise ValueError(
                "向量维度不匹配："
                f"当前 collection `{collection_name}` ({uri}) 的字段 `{self.embedding_field}` 维度是 {collection_dim}，"
                f"但请求的 embedding 模型维度是 {model_desc}。"
                "这通常表示你正在用新 embedding 模型查询旧库。"
                "请重新用该模型离线入库到新的 Milvus DB，或把请求里的 `uri` / `collection_name` 切到与该模型对应的库。"
            )
        return collection_dim

    def _normalize_vector_store_target(
        self,
        target: Any,
    ) -> Tuple[str, str]:
        if hasattr(target, "uri") and hasattr(target, "collection_name"):
            return str(target.uri), str(target.collection_name)
        if isinstance(target, dict):
            return str(target["uri"]), str(target["collection_name"])
        raise ValueError(f"Invalid vector store config: {target}")

    def _resolve_vector_store_target(
        self,
        request: RetrievingRequest,
        model_name: str,
    ) -> Tuple[str, str]:
        request_vector_store_configs = getattr(request, "vector_store_configs", None)
        if request_vector_store_configs and model_name in request_vector_store_configs:
            return self._normalize_vector_store_target(
                request_vector_store_configs[model_name]
            )

        return request.uri, request.collection_name

    def _embed_nodes_for_model(
        self,
        nodes: List[IndexNode],
        model_name: str,
    ) -> List[IndexNode]:
        """Embed nodes for one model with block-level modality routing."""
        embedding_model = self.embedding_models[model_name]
        text_nodes: List[IndexNode] = []
        image_nodes: List[IndexNode] = []
        for node in nodes:
            content_type = str(node.metadata.get("content_type", "text")).lower()
            image_path = node.metadata.get("image_path")
            if content_type == "image" and image_path:
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        embedded_by_id: Dict[str, IndexNode] = {}
        if text_nodes:
            embedded_text_nodes = get_node_with_embedding(
                text_nodes,
                embedding_model=embedding_model,
                show_progress=True,
            )
            for node in embedded_text_nodes:
                embedded_by_id[node.node_id] = node

        if image_nodes:
            if hasattr(embedding_model, "get_image_embedding_batch"):
                image_paths = [str(node.metadata.get("image_path")) for node in image_nodes]
                image_embeddings = embedding_model.get_image_embedding_batch(image_paths)
                for node, embedding in zip(image_nodes, image_embeddings):
                    result = node.copy()
                    result.embedding = embedding
                    embedded_by_id[result.node_id] = result
            else:
                # Fallback: convert image block into text and use text encoder.
                fallback_nodes: List[IndexNode] = []
                for node in image_nodes:
                    fallback_node = node.copy()
                    if not (fallback_node.text or "").strip():
                        image_name = Path(
                            str(fallback_node.metadata.get("image_path", ""))
                        ).name
                        fallback_node.text = f"[image] {image_name}"
                    fallback_nodes.append(fallback_node)
                logger.warning(
                    "Model %s does not support image embedding directly; fallback to text embedding for %s image blocks.",
                    model_name,
                    len(fallback_nodes),
                )
                embedded_image_nodes = get_node_with_embedding(
                    fallback_nodes,
                    embedding_model=embedding_model,
                    show_progress=True,
                )
                for node in embedded_image_nodes:
                    embedded_by_id[node.node_id] = node

        return [embedded_by_id[node.node_id] for node in nodes if node.node_id in embedded_by_id]

    def chunking_embedding(
        self, request: ChunkingEmbeddingRequest, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ChunkNodeWithEmbedding]:
        """处理文档，生成chunks和embeddings"""
        logger.info(f"Starting document processing: {request.doc_name}, Document ID: {request.doc_id}")

        # 1. 输入检查
        if request.input_type not in ["autopdf", "raw", "autopdf_chenggong", "json_list", "markdown", "mineru", "pdf_blocks"]:
            logger.error(f"Unsupported input type: {request.input_type}")
            raise ValueError(f"Invalid input type: {request.input_type}")

        self._ensure_embedding_models_loaded(request.embedding_models)
        for embedding_model in request.embedding_models:
            if embedding_model not in self.embedding_models:
                # TODO: 尝试初始化embedding模型
                logger.error(f"Embedding model not found: {embedding_model}")
                raise ValueError(f"Invalid embedding model: {embedding_model}")

        if self.use_local_milvus:
            self._ensure_local_vector_store(request.embedding_models)

        # 针对mineru的chunking_embedding通道
        if request.input_type == 'mineru':
            logger.info(f"Starting document chunking: {request.doc_name}")
            from tqdm import tqdm
            import pandas as pd
            tqdm.pandas()

            chunks = pd.DataFrame(json.loads(request.text))
            process_chunks = mineru_chunking(chunks, doc_name=request.doc_name, doc_id=request.doc_id)

            # 向量化
            process_chunks['embeddings'] = process_chunks['embedding_text'].progress_apply(lambda text:
            [
            {"model": embedding_model,
            "embedding": self.embedding_models[embedding_model]._get_text_embedding(text)}
            for embedding_model in request.embedding_models
            ]
            )

            del process_chunks['embedding_text']
            result = [
                ChunkNodeWithEmbedding.model_validate(entry)
                for entry in process_chunks.to_dict(orient="records")
            ]
            artifact_path = self.embedding_artifact_store.save_document_embeddings(
                doc_id=request.doc_id,
                doc_name=request.doc_name,
                input_type=request.input_type,
                embedding_models=request.embedding_models,
                index_nodes=result,
            )
            logger.info("Saved document embedding artifact to %s", artifact_path)
            return result


        # 2. 加载文档并切分
        logger.info(f"Starting document chunking: {request.doc_name}")
        start_time = time.time()
        small_nodes = chunk_into_small_nodes(
            request.input_type,
            request.text,
            request.doc_name,
            request.doc_id,
            self.tokenizer,
        )
        if metadata:
            for node in small_nodes:
                node.metadata.update(metadata)
        logger.info(f"Document chunking completed, generated {len(small_nodes)} small nodes, time: {time.time() - start_time:.2f} seconds")

        # 3. 并行计算多个embedding
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"Starting embeddings calculation, using models: {request.embedding_models}")
        start_time = time.time()
        embeded_small_nodes: Dict[str, List[IndexNode]] = {}
        with ThreadPoolExecutor(max_workers=len(request.embedding_models)) as executor:
            futures = [
                executor.submit(
                    self._embed_nodes_for_model,
                    small_nodes,
                    model_name,
                )
                for model_name in request.embedding_models
            ]
            for model_name, future in zip(request.embedding_models, futures):
                embeded_small_nodes[model_name] = future.result()
                logger.info(f"Model {model_name} completed embedding calculation")

        logger.info(f"All embeddings calculation completed, time: {time.time() - start_time:.2f} seconds")

        if self.use_local_milvus:
            # 若使用本地向量数据库，则自动将数据插入本地向量数据库
            logger.info(f"Inserting data into local vector database: {self.milvus_uri}")
            start_time = time.time()
            # TODO: 调整了表结构，同一id将对应到多个记录（与向量模型数目相同），需要修改！！！
            insert_list, _ = self.vector_store.add(embeded_small_nodes)
            logger.info(f"Data insertion completed, time: {time.time() - start_time:.2f} seconds")
        else:
            insert_list, _ = preprocess_for_insert(embeded_small_nodes)

        result = [ChunkNodeWithEmbedding.model_validate(entry) for entry in insert_list]
        artifact_path = self.embedding_artifact_store.save_document_embeddings(
            doc_id=request.doc_id,
            doc_name=request.doc_name,
            input_type=request.input_type,
            embedding_models=request.embedding_models,
            index_nodes=result,
        )
        logger.info("Saved document embedding artifact to %s", artifact_path)
        logger.info(f"Document processing completed: {request.doc_name}, generated {len(result)} X {len(request.embedding_models)} nodes with embeddings")
        return result

    def retrieving(self, request: RetrievingRequest) -> List[ChunkNodeWithScore]:
        """根据query查询向量数据库"""
        with self._retrieval_runtime_lock:
            logger.info(
                "Starting retrieval request processing: '%s', method=%s",
                request.query,
                request.retrieval_method,
            )

            if request.retrieval_method == "bm25":
                return self.bm25_retriever.retrieve(
                    request=request,
                    use_local_milvus=self.use_local_milvus,
                    local_uri=getattr(self, "milvus_uri", None),
                    local_collection_name=getattr(self, "collection_name", None),
                )

            if request.retrieval_method == "hybrid":
                vector_results = self._vector_retrieving_nodes(request)
                bm25_chunks = self.bm25_retriever.retrieve(
                    request=request,
                    use_local_milvus=self.use_local_milvus,
                    local_uri=getattr(self, "milvus_uri", None),
                    local_collection_name=getattr(self, "collection_name", None),
                )
                bm25_nodes = [self._chunk_to_node_with_score(chunk) for chunk in bm25_chunks]

                final_results = rrf_fuse_results(
                    [vector_results, bm25_nodes],
                    similarity_top_k=request.top_k or 5,
                )
                logger.info(
                    "Hybrid retrieval completed: vector=%s, bm25=%s, fused=%s",
                    len(vector_results),
                    len(bm25_nodes),
                    len(final_results),
                )
                return self._node_scores_to_chunks(final_results, request)

            return self._node_scores_to_chunks(self._vector_retrieving_nodes(request), request)

    def _chunk_to_node_with_score(self, chunk: ChunkNodeWithScore) -> NodeWithScore:
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("index_text", chunk.index_text)
        metadata.setdefault("md5_name", chunk.doc_id)
        metadata.setdefault("file_name", chunk.doc_name)
        node = IndexNode(
            id_=chunk.id,
            text=chunk.text,
            metadata=metadata,
            index_id=chunk.index_id,
            start_char_idx=metadata.get("start_char_idx"),
            end_char_idx=metadata.get("end_char_idx"),
        )
        return NodeWithScore(node=node, score=chunk.score)

    def _node_scores_to_chunks(
        self,
        node_scores: List[NodeWithScore],
        request: RetrievingRequest,
    ) -> List[ChunkNodeWithScore]:
        similarity_threshold = request.similarity_threshold if request.similarity_threshold is not None else 0.5
        retrieve_results: List[ChunkNodeWithScore] = []
        for entry in node_scores:
            entry_dict = node_to_milvus_dict(entry.node)
            entry_dict["score"] = entry.score
            if entry.score > similarity_threshold:
                retrieve_results.append(ChunkNodeWithScore.model_validate(entry_dict))
        return retrieve_results

    def _vector_retrieving_nodes(self, request: RetrievingRequest) -> List[NodeWithScore]:
        """向量检索并返回 NodeWithScore 列表。"""
        top_k = request.top_k or 5

        self._ensure_embedding_models_loaded(request.embedding_models)

        query_embeddings = []
        for model_name in request.embedding_models:
            if model_name not in self.embedding_models:
                logger.error(f"Embedding model not found: {model_name}")
                raise ValueError(f"Invalid embedding model: {model_name}")

            logger.info(f"Calculating query embedding using model {model_name}")
            query_embeddings.append(
                self._get_query_embedding_thread_safe(model_name, request.query)
            )

        if self.use_local_milvus:
            self._ensure_local_vector_store(request.embedding_models)
            logger.info("Using local vector database for retrieval")
            vector_stores = {
                "__local__": self.vector_store,
            }
            model_targets = {
                model_name: "__local__" for model_name in request.embedding_models
            }
        else:
            vector_stores: Dict[str, FusedMilvusVectorStore] = {}
            model_targets: Dict[str, str] = {}
            for model_name in request.embedding_models:
                uri, collection_name = self._resolve_vector_store_target(
                    request,
                    model_name,
                )
                target_key = f"{uri}::{collection_name}"
                model_targets[model_name] = target_key
                if target_key not in vector_stores:
                    with self._get_vector_store_lock(target_key):
                        logger.info(
                            "Connecting to external vector database for model %s: %s, collection: %s",
                            model_name,
                            uri,
                            collection_name,
                        )
                        vector_stores[target_key] = FusedMilvusVectorStore(
                            uri=uri,
                            overwrite=False,
                            collection_name=collection_name,
                            dim=1024,  # 失效参数
                            schema=None,
                            index_params=None,
                            embedding_field=self.embedding_field,
                        )
                        self._ensure_collection_vector_dim_matches(
                            vector_stores[target_key]._milvusclient,
                            collection_name,
                            [model_name],
                            uri,
                        )

        # 执行搜索
        start_time = time.time()
        results = []
        for query_embedding, model_name in zip(
            query_embeddings, request.embedding_models
        ):
            target_key = model_targets[model_name]
            vector_store = vector_stores[target_key]
            logger.info(f"Executing vector retrieval using model: {model_name}")
            if request.filter is None or request.filter == "":
                filter_expr = f"embedding_model == '{model_name}'"
            else:
                filter_expr = f"embedding_model == '{model_name}' and {request.filter}"
            with self._get_vector_store_lock(target_key):
                result = vector_store.query(
                    VectorStoreQuery(
                        query_embedding=query_embedding,
                        similarity_top_k=top_k,
                    ),
                    anns_field=self.embedding_field,
                    filter_expr=filter_expr,
                )
            result = process_search_results(result)
            logger.info(f"Model {model_name} retrieved {len(result)} results")
            results.append(result)

        # 搜索结果后处理
        logger.info("Starting fusion of multi-model retrieval results")
        final_results: List[NodeWithScore] = rrf_fuse_results(
            results, similarity_top_k=top_k
        )
        logger.info(f"Retrieval completed, time: {time.time() - start_time:.2f} seconds, returning {len(final_results)} results")
        return final_results

    def _load_table_rows(self, data_path: str) -> List[Dict[str, Any]]:
        resolved_path = str(Path(data_path).resolve())
        if resolved_path in self.table_rows_cache:
            return self.table_rows_cache[resolved_path]

        path = Path(resolved_path)
        if not path.exists():
            raise ValueError(f"Table data file not found: {data_path}")

        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                rows.append(json.loads(raw))

        self.table_rows_cache[resolved_path] = rows
        logger.info("Loaded %s table rows from %s", len(rows), resolved_path)
        return rows

    def _ensure_adela_rows_file(
        self,
        data_path: str,
        source_dir: Optional[str],
        searchable_fields: List[str],
    ) -> str:
        resolved_data_path = str(Path(data_path).resolve())
        resolved_data_file = Path(resolved_data_path)
        records_csv_path = Path(settings.ADELA_RELEASE_RECORDS_CSV_PATH).resolve()

        need_regenerate = not resolved_data_file.exists()
        if not need_regenerate and records_csv_path.exists():
            if records_csv_path.stat().st_mtime > resolved_data_file.stat().st_mtime:
                need_regenerate = True
            elif not self._adela_rows_file_compatible(
                resolved_data_file,
                required_fields=["model_name", "label_list", "source_path"],
            ):
                need_regenerate = True

        if not need_regenerate:
            return resolved_data_path

        if not source_dir:
            raise ValueError(
                f"ADELA data file not found: {resolved_data_path}."
                "Please provide `source_dir` or generate the JSONL file first."
            )

        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            raise ValueError(
                f"ADELA data file not found: {resolved_data_path}, and source_dir does not exist: {source_path}"
            )

        row_count = export_adela_records_to_jsonl(
            input_dir=source_path,
            output_path=resolved_data_file,
            searchable_fields=searchable_fields,
            records_csv_path=records_csv_path if records_csv_path.exists() else None,
        )
        self.table_rows_cache.pop(resolved_data_path, None)
        for cache_key in list(self.table_embeddings_cache.keys()):
            if cache_key.startswith("adela::") and f"::{resolved_data_file}" in cache_key:
                self.table_embeddings_cache.pop(cache_key, None)
        logger.info(
            "Generated adela JSONL file %s from %s, rows=%s",
            resolved_data_path,
            source_path,
            row_count,
        )
        return resolved_data_path

    def _adela_rows_file_compatible(
        self,
        data_path: Path,
        required_fields: List[str],
    ) -> bool:
        if not data_path.exists():
            return False

        try:
            with data_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    return all(field in row for field in required_fields)
        except Exception:
            return False
        return False

    def _load_adela_metadata_payload(self, source_path: Optional[str]) -> Dict[str, Any]:
        if not source_path:
            return {}
        resolved = str(Path(source_path).resolve())
        if resolved in self.adela_metadata_cache:
            return self.adela_metadata_cache[resolved]

        path = Path(resolved)
        if not path.exists():
            self.adela_metadata_cache[resolved] = {}
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        self.adela_metadata_cache[resolved] = payload
        return payload

    def _build_adela_reference_url(self, did: Any) -> Optional[str]:
        if did in (None, ""):
            return None
        did_text = str(did).strip()
        if not did_text:
            return None
        return settings.ADELA_DEPLOYMENT_URL_TEMPLATE.format(did=did_text)

    def _build_row_search_text(
        self,
        row: Dict[str, Any],
        searchable_fields: List[str],
    ) -> str:
        parts = []
        for field in searchable_fields:
            value = row.get(field)
            if value:
                parts.append(f"{field}: {value}")
        return "\n".join(parts)

    def _tokenize_table_text(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            return [
                token.lower()
                for token in self._tokenize_text_thread_safe(text)
                if token.strip()
            ]
        except Exception:
            return [token.lower() for token in text.split() if token.strip()]

    def _score_table_row_keyword(
        self,
        query: str,
        query_tokens: List[str],
        row: Dict[str, Any],
        searchable_fields: List[str],
    ) -> Tuple[float, List[str]]:
        query_lower = query.lower().strip()
        query_token_set = set(query_tokens)
        field_scores: List[float] = []
        matched_fields: List[str] = []

        for field in searchable_fields:
            value = row.get(field)
            if value is None:
                continue

            field_text = str(value).strip()
            if not field_text:
                continue

            field_lower = field_text.lower()
            field_tokens = set(self._tokenize_table_text(field_text))

            score = 0.0
            if query_lower and query_lower in field_lower:
                score = max(score, 1.0)
            if field_lower and len(field_lower) >= 2 and field_lower in query_lower:
                score = max(score, 0.85)
            if query_token_set and field_tokens:
                overlap = len(query_token_set & field_tokens) / max(len(query_token_set), 1)
                score = max(score, overlap)

            if score > 0:
                matched_fields.append(field)
                field_scores.append(score)

        if not field_scores:
            return 0.0, []

        avg_score = sum(field_scores) / len(field_scores)
        max_score = max(field_scores)
        final_score = min(1.0, max_score * 0.7 + avg_score * 0.3)
        return final_score, matched_fields

    def _get_table_embeddings(
        self,
        data_path: str,
        rows: List[Dict[str, Any]],
        searchable_fields: List[str],
        model_name: str,
        artifact_namespace: str = "tables",
    ) -> np.ndarray:
        cache_key = (
            f"{artifact_namespace}::{Path(data_path).resolve()}"
            f"::{model_name}::{'|'.join(searchable_fields)}"
        )
        cached = self.table_embeddings_cache.get(cache_key)
        if cached is not None:
            return cached

        row_ids = [str(row["row_id"]) for row in rows]
        file_cached = self.embedding_artifact_store.load_table_embeddings(
            data_path=data_path,
            model_name=model_name,
            searchable_fields=searchable_fields,
            row_ids=row_ids,
            artifact_namespace=artifact_namespace,
        )
        if file_cached is not None:
            self.table_embeddings_cache[cache_key] = file_cached
            logger.info(
                "Loaded %s embedding artifact from file for %s with model %s, rows=%s",
                artifact_namespace,
                data_path,
                model_name,
                len(rows),
            )
            return file_cached

        self._ensure_embedding_models_loaded([model_name])
        if model_name not in self.embedding_models:
            raise ValueError(f"Invalid embedding model: {model_name}")

        texts = [self._build_row_search_text(row, searchable_fields) for row in rows]
        embeddings = self.embedding_models[model_name].get_text_embedding_batch(texts)
        matrix = np.asarray(embeddings, dtype=np.float32)
        self.table_embeddings_cache[cache_key] = matrix
        artifact_path = self.embedding_artifact_store.save_table_embeddings(
            data_path=data_path,
            model_name=model_name,
            searchable_fields=searchable_fields,
            row_ids=row_ids,
            matrix=matrix,
            artifact_namespace=artifact_namespace,
        )
        logger.info(
            "Built %s embedding cache for %s with model %s, rows=%s, artifact=%s",
            artifact_namespace,
            data_path,
            model_name,
            len(rows),
            artifact_path,
        )
        return matrix

    def _score_table_rows_vector(
        self,
        request: TableChatRequest,
        rows: List[Dict[str, Any]],
        searchable_fields: List[str],
        artifact_namespace: str = "tables",
    ) -> Dict[str, float]:
        if len(request.embedding_models) != 1:
            raise ValueError(
                "Table vector retrieval currently requires exactly one embedding model."
            )

        model_name = request.embedding_models[0]
        self._ensure_embedding_models_loaded([model_name])
        query_embedding = np.asarray(
            self._get_query_embedding_thread_safe(model_name, request.query),
            dtype=np.float32,
        )
        row_embeddings = self._get_table_embeddings(
            request.data_path,
            rows,
            searchable_fields,
            model_name,
            artifact_namespace=artifact_namespace,
        )

        row_norms = np.linalg.norm(row_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        denom = np.maximum(row_norms * query_norm, 1e-8)
        similarities = np.dot(row_embeddings, query_embedding) / denom

        return {
            row["row_id"]: float(score)
            for row, score in zip(rows, similarities.tolist())
        }

    def table_chat_retrieving(
        self,
        request: TableChatRequest,
        artifact_namespace: str = "tables",
    ) -> List[TableMatchedRow]:
        with self._retrieval_runtime_lock:
            rows = self._load_table_rows(request.data_path)
            searchable_fields = request.searchable_fields or settings.TABLE_SEARCHABLE_FIELDS
            return_fields = request.return_fields or settings.TABLE_RETURN_FIELDS
            query_tokens = self._tokenize_table_text(request.query)
            threshold = request.similarity_threshold if request.similarity_threshold is not None else 0.0

            keyword_scores: Dict[str, float] = {}
            matched_field_map: Dict[str, List[str]] = {}
            for row in rows:
                score, matched_fields = self._score_table_row_keyword(
                    request.query,
                    query_tokens,
                    row,
                    searchable_fields,
                )
                keyword_scores[row["row_id"]] = score
                matched_field_map[row["row_id"]] = matched_fields

            vector_scores: Dict[str, float] = {}
            if request.retrieval_method in {"vector", "hybrid"}:
                vector_scores = self._score_table_rows_vector(
                    request=request,
                    rows=rows,
                    searchable_fields=searchable_fields,
                    artifact_namespace=artifact_namespace,
                )

            ranked_rows: List[TableMatchedRow] = []
            for row in rows:
                row_id = row["row_id"]
                keyword_score = keyword_scores.get(row_id, 0.0)
                vector_score = vector_scores.get(row_id, 0.0)
                if request.retrieval_method == "keyword":
                    final_score = keyword_score
                elif request.retrieval_method == "vector":
                    final_score = vector_score
                else:
                    final_score = keyword_score * 0.45 + vector_score * 0.55

                if final_score < threshold:
                    continue

                entity = {"row_id": row_id}
                for field in return_fields:
                    entity[field] = row.get(field)

                ranked_rows.append(
                    TableMatchedRow(
                        row_id=row_id,
                        score=round(float(final_score), 6),
                        matched_fields=matched_field_map.get(row_id, []),
                        entity=entity,
                    )
                )

            ranked_rows.sort(key=lambda item: item.score, reverse=True)
            top_k = request.top_k or len(ranked_rows)
            return ranked_rows[:top_k]

    def adela_chat_retrieving(
        self,
        request: AdelaChatRequest,
    ) -> List[TableMatchedRow]:
        searchable_fields = request.searchable_fields or settings.ADELA_SEARCHABLE_FIELDS
        data_path = self._ensure_adela_rows_file(
            data_path=request.data_path,
            source_dir=request.source_dir,
            searchable_fields=searchable_fields,
        )

        desired_return_fields = request.return_fields or settings.ADELA_RETURN_FIELDS
        final_return_fields = list(
            dict.fromkeys(
                [
                    *desired_return_fields,
                    "reference",
                    "model_name",
                    "name",
                    "platform",
                    "rid",
                    "did",
                    "label_list",
                    "labels",
                    "source_file",
                    "model_info",
                    "benchmark_info",
                ]
            )
        )
        internal_return_fields = list(dict.fromkeys([*final_return_fields, "source_path"]))

        table_request = TableChatRequest(
            query=request.query,
            retrieval_method=request.retrieval_method,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            data_path=data_path,
            searchable_fields=searchable_fields,
            return_fields=internal_return_fields,
            embedding_models=request.embedding_models,
            llm_config=request.llm_config,
        )
        rows = self.table_chat_retrieving(table_request, artifact_namespace="adela")

        enriched_rows: List[TableMatchedRow] = []
        for row in rows:
            entity = dict(row.entity or {})
            payload = self._load_adela_metadata_payload(entity.get("source_path"))
            if payload:
                if entity.get("model_info") in (None, ""):
                    entity["model_info"] = payload.get("model_info")
                if entity.get("benchmark_info") in (None, ""):
                    entity["benchmark_info"] = payload.get("benchmark_info")
                if entity.get("name") in (None, ""):
                    entity["name"] = payload.get("name")
                if entity.get("model_name") in (None, ""):
                    entity["model_name"] = payload.get("name")
                if entity.get("platform") in (None, ""):
                    entity["platform"] = payload.get("platform")
                if entity.get("rid") in (None, ""):
                    entity["rid"] = payload.get("rid")
                if entity.get("did") in (None, ""):
                    entity["did"] = payload.get("did")

            if entity.get("model_name") in (None, "") and entity.get("name") not in (None, ""):
                entity["model_name"] = entity.get("name")
            if entity.get("name") in (None, "") and entity.get("model_name") not in (None, ""):
                entity["name"] = entity.get("model_name")
            entity["reference"] = self._build_adela_reference_url(entity.get("did"))

            output_entity = {field: entity.get(field) for field in final_return_fields}
            enriched_rows.append(
                TableMatchedRow(
                    row_id=row.row_id,
                    score=row.score,
                    matched_fields=row.matched_fields,
                    entity=output_entity,
                )
            )
        return enriched_rows


    def get_text_embedding(self, text: str, model_name: str) -> List[float]:
        """获取单个文本的向量表示"""
        self._ensure_embedding_models_loaded([model_name])
        if model_name not in self.embedding_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        return self.embedding_models[model_name].get_text_embedding(text)

    def get_text_embedding_batch(self, texts: List[str], model_name: str) -> List[List[float]]:
        """批量获取文本的向量表示"""
        self._ensure_embedding_models_loaded([model_name])
        if model_name not in self.embedding_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        return self.embedding_models[model_name].get_text_embedding_batch(texts)

    def export_all_nodes(self) -> List[ChunkNodeWithScore]:
        """导出所有节点"""

        vector_store = self.vector_store

        # 执行搜索
        collection = vector_store._collection
        # 查询所有数据
        results = collection.query(
            expr="",  # 这个表达式会匹配所有记录，假设 id 都是非负的
            output_fields=["id", "text", "index_id", "index_text", "metadata", "doc_id", "doc_name", "embedding_model"],  # 返回除vector_data外的所有字段
            limit=16384  # 设置足够大的限制以获取所有数据，milvus限制了取值范围[1, 16384]
        )
        # results = collection.query(expr="",output_fields=["id", "text", "index_id", "index_text", "metadata", "doc_id", "doc_name", "embedding_model"],limit=16384)
        # import ipdb; ipdb.set_trace()

        return results
