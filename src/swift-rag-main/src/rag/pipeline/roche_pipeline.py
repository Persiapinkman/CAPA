import os
from tqdm import tqdm

from doc_loader.roche_loader import get_md5
from src.api.schemas import ChunkingEmbeddingRequest, RetrievingRequest
from src.rag.service import RAGService

service = RAGService(
    use_local_milvus={
        "uri": "/app/data_source/embedding_artifacts/documents/milvus_roche.db",
        # "overwrite": False,
        "overwrite": False, # 离线入库时设置为True，在线检索时设置为False
        "collection_name": "llamacollection",
    }
)

# # 离线入库（示例）
# parse_path = "/app/data_source"
# for file_name in tqdm(os.listdir(parse_path)):
#     file_path = os.path.join(parse_path, file_name)
#     if not os.path.exists(file_path):
#         continue
#     request = ChunkingEmbeddingRequest(
#         text=open(file_path).read(),
#         input_type="raw",
#         doc_id=get_md5(file_name),
#         doc_name=file_name,
#         embedding_models=["EvoQwen2.5-VL-Retriever-3B-v1"],
#     )
#     small_nodes = service.chunking_embedding(request)
#     # small_nodes = service.chunking_embedding(request, metadata={"content_type": "caption"})
#     print(f"chunking_embedding {file_name} done")

# 在线检索
nodes = service.retrieving(
    RetrievingRequest(
        query="如何在软件中切换到cobas link界面？",
        top_k=5,
        embedding_models=["EvoQwen2.5-VL-Retriever-3B-v1"],
        filter="doc_name == 'cobas pro 用户培训指导手册 e801 v1'",
        # filter="metadata['page_label'] == 16",    # 用metadata过滤
    )
)
