from typing import List

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
# from global_logger import logger

def run_queries(query, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for i, retriever in enumerate(retrievers):
        retrieve_nodes = retriever.retrieve(query)
        # logger.info(f'fuse retriever_{i} retrieve {len(retrieve_nodes)} nodes')
        tasks.append(retrieve_nodes)

    results_dict = {}
    #for i, (query, query_result) in enumerate(zip(queries, tasks)):
    results_dict[(query, 0)] = tasks

    return results_dict

def rrf_fuse_results(result:List[List[NodeWithScore]], similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    #for nodes_with_scores in results_dict.values():
    for nodes_with_scores in result:
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            if text not in text_to_node or node_with_score.score > text_to_node[text].score:
                text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].node.metadata['rrf_score'] = score
        reranked_nodes[-1].node.excluded_embed_metadata_keys.append('rrf_score')
        reranked_nodes[-1].node.excluded_llm_metadata_keys.append('rrf_score')

    if similarity_top_k > 0:
        return reranked_nodes[:similarity_top_k]
    else:
        return reranked_nodes

def simple_merge_result(result:List[List[NodeWithScore]], similarity_top_k: int = 2):

    reranked_nodes: List[NodeWithScore] = []
    for nodes_with_scores in result:
        reranked_nodes.extend(nodes_with_scores)

    if similarity_top_k > 0:
        return reranked_nodes[:similarity_top_k]
    else:
        return reranked_nodes

class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = -1,
        fuse_strategy='simple_merge'
    ) -> None:
        #import pdb;pdb.set_trace()
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._fuse_strategy = fuse_strategy
        # logger.info(f'using fuse retriever, fuse strategy is {fuse_strategy}')
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        #results = run_queries(query_bundle, self._retrievers)
        #import pdb;pdb.set_trace()
        results = []
        for i, retriever in enumerate(self._retrievers):
            results.append(retriever.retrieve(query_bundle.query_str))
        if self._fuse_strategy == 'rrf':
            final_results = rrf_fuse_results(
                results, similarity_top_k=self._similarity_top_k
            )
        elif self._fuse_strategy == 'simple_merge':
            final_results = simple_merge_result(
                results, similarity_top_k=self._similarity_top_k
            )

        return final_results