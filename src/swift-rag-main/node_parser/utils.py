from llama_index.core.schema import BaseNode, ImageDocument, ImageNode, MetadataMode, NodeRelationship, TextNode
from llama_index.core import Document
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.utils import embed_nodes


def get_node_with_embedding(
    nodes:List[BaseNode],
    embedding_model: BaseEmbedding, show_progress=True
    ):
    """embed nodes manually"""
    id_to_embed_map = embed_nodes(
        nodes, embedding_model, show_progress=show_progress
    )

    results = []
    for node in nodes:
        embedding = id_to_embed_map[node.node_id]
        result = node.copy()
        result.embedding = embedding
        results.append(result)
    return results


'''from embeddings.embedding_builder import build_embedding_model
from cfg.setup_cfg import CONFIG as cfg

def _get_node_with_embedding(
    nodes:List[BaseNode],
    gpu_id=None
    ):
    """embed nodes manually"""
    embedding_model = build_embedding_model(
        model_name_or_path=cfg.embed.model_path,
        batch_size=cfg.embed.batchsize,
        device=f'cuda:{gpu_id}'
    )

    id_to_embed_map = embed_nodes(
        nodes, embedding_model, show_progress=True
    )

    results = []
    for node in nodes:
        embedding = id_to_embed_map[node.node_id]
        result = node.copy()
        result.embedding = embedding
        results.append(result)
    return results

from multiprocessing import Pool
def get_node_with_embedding_mul(
    nodes:List[BaseNode],
    embedding_model: BaseEmbedding
    ):
    """embed nodes manually"""
    import pdb;pdb.set_trace()
    workers=1
    pool = Pool(workers)
    length = len(nodes)
    res = []
    for i in range(workers):
        p1 = length * i // workers
        p2 = length * (i + 1) // workers
        #args = [p1, p2, image_root, workers, essos_plg, cls_plg, image_root, json_root]
        #print(f"{p1},{p2}")
        nodes_split = nodes[p1:p2]
        argss = [nodes_split,i]
        r = pool.apply_async(func=_get_node_with_embedding, args=argss) #(func=cal_rouge, args=argss)
        res.append(r)
    pool.close()
    pool.join()
    embed_nodes = []
    for r in res:
        tmp = r.get()
        embed_nodes.extend(tmp)

    return embed_nodes'''
'''import multiprocessing as mp
def get_node_with_embedding_mul(
    nodes:List[BaseNode],
    embedding_model: BaseEmbedding
    ):
    """embed nodes manually"""
    import pdb;pdb.set_trace()
    ctx = mp.get_context("spawn")
    workers=4
    length = len(nodes)
    res = []
    pool = ctx.Pool(processes=workers)
    for i in range(workers):
        p1 = length * i // workers
        p2 = length * (i + 1) // workers
        #args = [p1, p2, image_root, workers, essos_plg, cls_plg, image_root, json_root]
        #print(f"{p1},{p2}")
        nodes_split = nodes[p1:p2]
        argss = [nodes_split,embedding_model,i]
        r = pool.apply_async(func=get_node_with_embedding, args=argss) #(func=cal_rouge, args=argss)
        res.append(r)

    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出

    embed_nodes = []
    for r in res:
        tmp = r.get()
        embed_nodes.extend(tmp)

    return embed_nodes'''

import concurrent.futures
import copy
import torch
def get_node_with_embedding_mul(
    nodes:List[BaseNode],
    embedding_model: BaseEmbedding
    ):
    """embed nodes manually"""
    #import pdb;pdb.set_trace()
    workers=min(torch.cuda.device_count(),4)
    length = len(nodes)
    res = []
    embed_nodes = []
    embedding_models = []
    for i in range(workers):
        if embedding_model._model.device.index == i:
            source_id = i
            embedding_models.append(embedding_model)
        else:
            embedding_model_i = copy.deepcopy(embedding_model)
            embedding_model_i._model.to(f'cuda:{i}')
            embedding_models.append(embedding_model_i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:

        for i in range(workers):
            p1 = length * i // workers
            p2 = length * (i + 1) // workers
            #args = [p1, p2, image_root, workers, essos_plg, cls_plg, image_root, json_root]
            #print(f"{p1},{p2}")
            nodes_split = nodes[p1:p2]
            argss = [nodes_split,i]
            r = executor.submit(get_node_with_embedding, nodes_split, embedding_models[i])
            res.append(r)
        for f in concurrent.futures.as_completed(res):
            embed_nodes.extend(f.result())

    for emd in embedding_models:
        if emd._model.device.index == source_id:
            continue
        emd._model.to('cpu')
        del emd

    torch.cuda.empty_cache()

    return embed_nodes