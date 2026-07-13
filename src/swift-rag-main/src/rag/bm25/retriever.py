import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from pymilvus import MilvusClient

from src.api.schemas import ChunkNodeWithScore, RetrievingRequest
from src.core.logging import get_logger
from vector_stores.fuse_milvus import milvus_dict_to_node, node_to_milvus_dict

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 keyword retriever over Milvus scalar query results."""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        embedding_field: str,
        resolve_vector_store_target: Callable[[RetrievingRequest, str], Tuple[str, str]],
        get_vector_store_lock: Callable[[str], Any],
        k1: float,
        b: float,
        candidate_limit: int,
    ):
        self._tokenizer = tokenizer
        self._embedding_field = embedding_field
        self._resolve_vector_store_target = resolve_vector_store_target
        self._get_vector_store_lock = get_vector_store_lock
        self._k1 = k1
        self._b = b
        self._candidate_limit = candidate_limit

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [token for token in self._tokenizer(text) if token and token.strip()]

    def _build_filter_expr(self, request: RetrievingRequest) -> str:
        clauses: List[str] = []
        if request.filter:
            clauses.append(f"({request.filter})")

        if request.embedding_models:
            # Use one model to avoid duplicate rows in multi-model indexing.
            model_name = request.embedding_models[0].replace("'", "\\'")
            clauses.append(f"embedding_model == '{model_name}'")

        return " and ".join(clauses)

    def _resolve_target(
        self,
        request: RetrievingRequest,
        use_local_milvus: Optional[Dict[str, Any]],
        local_uri: Optional[str],
        local_collection_name: Optional[str],
    ) -> Tuple[str, str]:
        if use_local_milvus:
            if not local_uri or not local_collection_name:
                raise ValueError("Local Milvus is enabled but uri/collection_name is missing")
            return local_uri, local_collection_name

        if request.embedding_models:
            return self._resolve_vector_store_target(request, request.embedding_models[0])

        return request.uri, request.collection_name

    def _query_candidates(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        filter_expr: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        output_fields = [
            "id",
            "text",
            "index_id",
            "index_text",
            "metadata",
            "doc_id",
            "doc_name",
            "embedding_model",
        ]
        return milvus_client.query(
            collection_name=collection_name,
            filter=filter_expr or "",
            output_fields=output_fields,
            limit=limit or self._candidate_limit,
        )

    def retrieve(
        self,
        request: RetrievingRequest,
        use_local_milvus: Optional[Dict[str, Any]],
        local_uri: Optional[str],
        local_collection_name: Optional[str],
    ) -> List[ChunkNodeWithScore]:
        top_k = request.top_k or 5
        similarity_threshold = request.similarity_threshold or 0.0

        uri, collection_name = self._resolve_target(
            request,
            use_local_milvus,
            local_uri,
            local_collection_name,
        )
        logger.info(
            "Executing BM25 retrieval: uri=%s, collection=%s, embedding_field=%s",
            uri,
            collection_name,
            self._embedding_field,
        )

        target_key = f"{uri}::{collection_name}"
        with self._get_vector_store_lock(target_key):
            milvus_client = MilvusClient(uri=uri)
            candidates = self._query_candidates(
                milvus_client=milvus_client,
                collection_name=collection_name,
                filter_expr=self._build_filter_expr(request),
            )
        if not candidates:
            logger.info("BM25 retrieval found no candidates")
            return []

        query_tokens = self._tokenize(request.query)
        if not query_tokens:
            logger.info("BM25 retrieval found empty query tokens")
            return []

        tokenized_docs: List[List[str]] = [
            self._tokenize(str(doc.get("text", ""))) for doc in candidates
        ]
        valid_lengths = [len(tokens) for tokens in tokenized_docs if len(tokens) > 0]
        if not valid_lengths:
            logger.info("BM25 retrieval found no tokenized documents")
            return []

        avg_doc_len = sum(valid_lengths) / len(valid_lengths)
        query_terms = list(dict.fromkeys(query_tokens))
        total_docs = len(tokenized_docs)

        doc_freq: Dict[str, int] = {}
        for term in query_terms:
            doc_freq[term] = sum(1 for tokens in tokenized_docs if term in set(tokens))

        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        for tokens, candidate in zip(tokenized_docs, candidates):
            if not tokens:
                scored_candidates.append((0.0, candidate))
                continue

            term_counts = Counter(tokens)
            bm25_score = 0.0
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + self._k1 * (1 - self._b + self._b * (len(tokens) / avg_doc_len))
                bm25_score += idf * (tf * (self._k1 + 1) / denominator)
            scored_candidates.append((bm25_score, candidate))

        max_score = max(score for score, _ in scored_candidates)
        normalized = [
            ((score / max_score) if max_score > 0 else 0.0, candidate)
            for score, candidate in scored_candidates
        ]
        normalized.sort(key=lambda item: item[0], reverse=True)

        retrieve_results: List[ChunkNodeWithScore] = []
        for score, candidate in normalized:
            if len(retrieve_results) >= top_k:
                break
            if score <= similarity_threshold:
                continue

            node = milvus_dict_to_node(candidate)
            entry_dict = node_to_milvus_dict(node)
            entry_dict["score"] = score
            retrieve_results.append(ChunkNodeWithScore.model_validate(entry_dict))

        logger.info("BM25 retrieval completed, returning %s results", len(retrieve_results))
        return retrieve_results
