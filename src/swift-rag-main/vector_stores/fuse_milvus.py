import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.schema import BaseNode, IndexNode, TextNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    MILVUS_ID_FIELD,
    AnnSearchRequest,
    BaseSparseEmbeddingFunction,
    Collection,
    DataType,
    IndexManagement,
    LoadState,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
    _to_milvus_filter,
    get_default_sparse_embedding_function,
    logger,
)
from pymilvus.milvus_client.index import IndexParams
from pymilvus.orm.collection import CollectionSchema


def node_to_milvus_dict(node: BaseNode) -> dict:
    """将节点转换为字典，用于插入milvus"""
    node_metadata = node.metadata or {}
    data_dict = {}
    data_dict["id"] = node.node_id
    data_dict["text"] = node.text
    data_dict["index_id"] = node.index_id
    data_dict["index_text"] = node_metadata.get("index_text", node.text)

    # 以下是node.metadata的键(header, heading, page_label, start_page, end_page, start_char_idx, end_char_idx, content_type, extra_metadata)

    data_dict["metadata"] = deepcopy(node_metadata)
    data_dict["metadata"]["start_char_idx"] = node.start_char_idx
    data_dict["metadata"]["end_char_idx"] = node.end_char_idx
    # data_dict["metadata"].pop("doc_id")
    # data_dict["metadata"].pop("doc_name")
    data_dict["metadata"].pop("md5_name", None)
    data_dict["metadata"].pop("file_name", None)
    data_dict["metadata"].pop("index_text", None)
    if "rrf_score" in data_dict["metadata"]:
        data_dict["metadata"].pop("rrf_score")

    # data_dict["doc_id"] = node.metadata.get("doc_id", None)
    # data_dict["doc_name"] = node.metadata.get("doc_name", None)

    data_dict["doc_id"] = node_metadata.get("md5_name", node_metadata.get("doc_id", None))
    data_dict["doc_name"] = node_metadata.get("file_name", node_metadata.get("doc_name", None))

    return data_dict


def milvus_dict_to_node(data_dict: dict) -> IndexNode:
    """将milvus字典转换为节点"""
    metadata = data_dict.get("metadata", {})
    if type(metadata) == str:
        metadata = json.loads(metadata) # noqa！！！
    metadata["index_text"] = data_dict.get("index_text", None)
    metadata["md5_name"] = data_dict.get("doc_id", None)
    metadata["file_name"] = data_dict.get("doc_name", None)

    node = IndexNode(
        # id_=data_dict["id"],
        id_=data_dict["id"].split("_EMB_")[0],  # 从id中去掉_EMB_和模型名
        text=data_dict["text"],
        metadata=metadata,
        index_id=data_dict["index_id"],
        start_char_idx=metadata.get("start_char_idx", None),
        end_char_idx=metadata.get("end_char_idx", None),
    )
    return node


def preprocess_for_insert(
    nodes: Dict[str, List[BaseNode]]
) -> Tuple[List[IndexNode], List[str]]:
    insert_list = []
    insert_ids = []

    # Process that data we are going to insert
    for node_list in zip(*nodes.values()):
        # >>>
        # entry = node_to_metadata_dict(node)
        # entry[MILVUS_ID_FIELD] = node.node_id
        # entry[self.embedding_field] = node.embedding
        # <<<
        assert all(node.node_id == node_list[0].node_id for node in node_list)
        entry = node_to_milvus_dict(node_list[0])
        entry["embeddings"] = []
        for node, model_name in zip(node_list, nodes.keys()):
            entry["embeddings"].append(
                {"model": model_name, "embedding": node.embedding}
            )
        # >>>

        insert_ids.append(node_list[0].node_id)
        insert_list.append(entry)

    return insert_list, insert_ids


class FusedMilvusVectorStore(MilvusVectorStore):
    """自定义的Milvus向量存储类，支持两个embedding模型"""

    custom_schema: Optional[CollectionSchema] = None
    custom_index_params: Optional[IndexParams] = None

    def __init__(
        self,
        uri: str = "./milvus_llamaindex.db",
        token: str = "",
        collection_name: str = "llamacollection",
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "IP",
        consistency_level: str = "Strong",
        overwrite: bool = False,
        text_key: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        index_config: Optional[dict] = None,
        search_config: Optional[dict] = None,
        collection_properties: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_sparse: bool = False,
        sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None,
        hybrid_ranker: str = "RRFRanker",
        hybrid_ranker_params: dict = {},
        index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS,
        schema: Optional[CollectionSchema] = None,  # 新增参数
        index_params: Optional[IndexParams] = None,  # 新增参数
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super(MilvusVectorStore, self).__init__(
            collection_name=collection_name,
            dim=dim,
            embedding_field=embedding_field,
            doc_id_field=doc_id_field,
            consistency_level=consistency_level,
            overwrite=overwrite,
            text_key=text_key,
            output_fields=output_fields or [],
            index_config=index_config if index_config else {},
            search_config=search_config if search_config else {},
            collection_properties=collection_properties,
            batch_size=batch_size,
            enable_sparse=enable_sparse,
            sparse_embedding_function=sparse_embedding_function,
            hybrid_ranker=hybrid_ranker,
            hybrid_ranker_params=hybrid_ranker_params,
            index_management=index_management,
        )

        # Select the similarity metric
        similarity_metrics_map = {
            "ip": "IP",
            "l2": "L2",
            "euclidean": "L2",
            "cosine": "COSINE",
        }
        self.similarity_metric = similarity_metrics_map.get(
            similarity_metric.lower(), "L2"
        )
        # Connect to Milvus instance
        self._milvusclient = MilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )
        # Delete previous collection if overwriting
        if overwrite and collection_name in self.client.list_collections():
            self._milvusclient.drop_collection(collection_name)

        # Create the collection if it does not exist
        self.custom_schema = schema  # 新增参数
        self.custom_index_params = index_params  # 新增参数
        if collection_name not in self.client.list_collections():
            if dim is None:
                raise ValueError("Dim argument required for collection creation.")
            if self.enable_sparse is False:
                self._milvusclient.create_collection(
                    collection_name=collection_name,
                    dimension=dim,
                    primary_field_name=MILVUS_ID_FIELD,
                    vector_field_name=embedding_field,
                    id_type="string",
                    metric_type=self.similarity_metric,
                    max_length=65_535,
                    consistency_level=consistency_level,
                    schema=self.custom_schema,  # 新增参数
                    index_params=self.custom_index_params,  # 新增参数
                )
            else:
                try:
                    _ = DataType.SPARSE_FLOAT_VECTOR
                except Exception as e:
                    logger.error(
                        "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                    )
                    raise NotImplementedError(
                        "Hybrid retrieval requires Milvus 2.4.0 or later."
                    ) from e
                self._create_hybrid_index(collection_name)

        self._collection = Collection(collection_name, using=self._milvusclient._using)
        # if self.custom_index_params is None:  # 如果index_params为None，则创建索引
            # self._create_index_if_required()

        # Set properties
        if collection_properties:
            if self._milvusclient.get_load_state(collection_name) == LoadState.Loaded:
                self._collection.release()
                self._collection.set_properties(properties=collection_properties)
                self._collection.load()
            else:
                self._collection.set_properties(properties=collection_properties)

        self.enable_sparse = enable_sparse
        if self.enable_sparse is True and sparse_embedding_function is None:
            logger.warning("Sparse embedding function is not provided, using default.")
            self.sparse_embedding_function = get_default_sparse_embedding_function()
        elif self.enable_sparse is True and sparse_embedding_function is not None:
            self.sparse_embedding_function = sparse_embedding_function
        else:
            pass

        logger.debug(f"Successfully created a new collection: {self.collection_name}")

        # # 打印collection的schema
        # print(
        #     self._milvusclient.describe_collection(collection_name=self.collection_name)
        # )

    def add(self, nodes: Dict[str, List[BaseNode]], **add_kwargs: Any) -> Tuple[List[IndexNode], List[str]]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        insert_list, insert_ids = preprocess_for_insert(nodes)
        tmp_insert_list = deepcopy(insert_list)

        insert_list_X_embedding_model = []
        for entry in insert_list:
            embeddings = entry.pop("embeddings")
            for embedding in embeddings:
                tmp_entry = deepcopy(entry)
                tmp_entry["id"] = f"{entry['id']}_EMB_{embedding['model']}" # 多个向量模型的情况下会写入多条数据，因此ID要加模型名称后缀，防止ID冲突
                tmp_entry["embedding_model"] = embedding["model"]
                tmp_entry[self.embedding_field] = embedding["embedding"]
                insert_list_X_embedding_model.append(tmp_entry)
        insert_list = insert_list_X_embedding_model

        # Insert the data into milvus
        for insert_batch in iter_batch(insert_list, self.batch_size):
            self._collection.insert(insert_batch)
        if add_kwargs.get("force_flush", False):
            self._collection.flush()
        # if self.custom_index_params is None:  # 如果index_params为None，则创建索引
            # self._create_index_if_required()
        logger.debug(
            f"Successfully inserted embeddings into: {self.collection_name} "
            f"Num Inserted: {len(insert_list)}"
        )
        return tmp_insert_list, insert_ids

    def query(
        self,
        query: VectorStoreQuery,
        anns_field: Optional[str] = None,
        filter_expr: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode == VectorStoreQueryMode.DEFAULT:
            pass
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if self.enable_sparse is False:
                raise ValueError(f"QueryMode is HYBRID, but enable_sparse is False.")
        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = []
        output_fields = ["*"]

        # Parse the filter

        if query.filters is not None or "milvus_scalar_filters" in kwargs:
            expr.append(
                _to_milvus_filter(
                    query.filters,
                    (
                        kwargs["milvus_scalar_filters"]
                        if "milvus_scalar_filters" in kwargs
                        else None
                    ),
                )
            )

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        outputs_limited = False
        if query.output_fields is not None:
            output_fields = query.output_fields
            outputs_limited = True
        elif len(self.output_fields) > 0:
            output_fields = [*self.output_fields]
            outputs_limited = True

        # Add the text key to output fields if necessary
        if self.text_key and self.text_key not in output_fields and outputs_limited:
            output_fields.append(self.text_key)

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = f" and ".join(expr)

        if filter_expr is not None:
            string_expr = filter_expr

        # Perform the search
        if query.mode == VectorStoreQueryMode.DEFAULT:
            # Perform default search
            res = self._milvusclient.search(
                collection_name=self.collection_name,
                data=[query.query_embedding],
                filter=string_expr,
                limit=query.similarity_top_k,
                output_fields=output_fields,
                search_params=self.search_config,
                anns_field=anns_field,  # 新增参数
            )
            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )

            nodes = []
            similarities = []
            ids = []
            # Parse the results
            for hit in res[0]:
                if not self.text_key:
                    # >>>
                    # node = metadata_dict_to_node(
                    #     {
                    #         "_node_content": hit["entity"].get("_node_content", None),
                    #         "_node_type": hit["entity"].get("_node_type", None),
                    #     }
                    # )
                    # <<<
                    node = milvus_dict_to_node(hit["entity"])
                    # >>>
                else:
                    try:
                        text = hit["entity"].get(self.text_key)
                    except Exception:
                        raise ValueError(
                            "The passed in text_key value does not exist "
                            "in the retrieved entity."
                        )

                    metadata = {
                        key: hit["entity"].get(key) for key in self.output_fields
                    }
                    node = TextNode(text=text, metadata=metadata)

                nodes.append(node)
                similarities.append(hit["distance"])
                ids.append(hit["id"])

        else:
            # Perform hybrid search
            sparse_emb = self.sparse_embedding_function.encode_queries(
                [query.query_str]
            )[0]
            sparse_search_params = {"metric_type": "IP"}

            sparse_req = AnnSearchRequest(
                data=[sparse_emb],
                anns_field=self.sparse_embedding_field,
                param=sparse_search_params,
                limit=query.similarity_top_k,
                expr=string_expr,  # Apply metadata filters to sparse search
            )

            dense_search_params = {
                "metric_type": self.similarity_metric,
                "params": self.search_config,
            }
            dense_emb = query.query_embedding
            dense_req = AnnSearchRequest(
                data=[dense_emb],
                anns_field=self.embedding_field,
                param=dense_search_params,
                limit=query.similarity_top_k,
                expr=string_expr,  # Apply metadata filters to dense search
            )
            ranker = None

            if WeightedRanker is None or RRFRanker is None:
                logger.error(
                    "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                )
                raise ValueError(
                    "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                )
            if self.hybrid_ranker == "WeightedRanker":
                if self.hybrid_ranker_params == {}:
                    self.hybrid_ranker_params = {"weights": [1.0, 1.0]}
                ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
            elif self.hybrid_ranker == "RRFRanker":
                if self.hybrid_ranker_params == {}:
                    self.hybrid_ranker_params = {"k": 60}
                ranker = RRFRanker(self.hybrid_ranker_params["k"])
            else:
                raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")

            res = self._collection.hybrid_search(
                [dense_req, sparse_req],
                rerank=ranker,
                limit=query.similarity_top_k,
                output_fields=output_fields,
            )

            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )

            nodes = []
            similarities = []
            ids = []
            # Parse the results
            for hit in res[0]:
                if not self.text_key:
                    node = metadata_dict_to_node(
                        {
                            "_node_content": hit.entity.get("_node_content"),
                            "_node_type": hit.entity.get("_node_type"),
                        }
                    )
                else:
                    try:
                        text = hit.entity.get(self.text_key)
                    except Exception:
                        raise ValueError(
                            "The passed in text_key value does not exist "
                            "in the retrieved entity."
                        )

                    metadata = {key: hit.entity.get(key) for key in self.output_fields}
                    node = TextNode(text=text, metadata=metadata)

                nodes.append(node)
                similarities.append(hit.distance)
                ids.append(hit.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
