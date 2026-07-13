from typing import Callable, List, Optional, cast, Dict
from text_splitter.zh_text_splitter import ZHSentenceSplitter
from llama_index.core.node_parser import NodeParser
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import BaseNode,IndexNode,MetadataMode,NodeRelationship
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
import os
from tqdm import tqdm
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode,IndexNode,MetadataMode
from node_parser.utils import get_node_with_embedding,get_node_with_embedding_mul
# from global_logger import logger
from multiprocessing import Pool
import random

class HierarchicalParser:
    def __init__(
        self,
        chunk_sizes:List[int]=[128,512],
        tokenizer=None,
        embedding_model=None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        chunk_sizes.sort()

        self.chunk_sizes = chunk_sizes
        self.tokenizer=tokenizer
        self.embed_model = embedding_model
        self.callback_manager=callback_manager
        sub_splitters = [
            ZHSentenceSplitter(
                chunk_size=c,
                chunk_overlap=0,
                include_prev_next_rel=True,
                include_metadata=True,
                tokenizer=self.tokenizer,
                callback_manager=self.callback_manager
                ) for c in self.chunk_sizes
        ]
        self.sub_node_parsers = {
            'small': sub_splitters[0],
            'big': sub_splitters[1],
        }

    def _embed_nodes_mul(self,all_nodes,embed_model=None):
        chunk_128_index_nodes = [it for it in all_nodes if 'chunk128' in it.id_]
        chunk_512_index_nodes = [it for it in all_nodes if 'chunk128' not in it.id_]
        #from utils import get_node_with_embedding
        if embed_model:
            chunk_128_index_nodes = get_node_with_embedding_mul(chunk_128_index_nodes,embedding_model=embed_model)
        else:
            chunk_128_index_nodes = get_node_with_embedding_mul(chunk_128_index_nodes,embedding_model=self.embed_model)
        all_nodes = chunk_512_index_nodes + chunk_128_index_nodes
        return all_nodes

    def _embed_nodes(self,all_nodes,embed_model=None):
        chunk_128_index_nodes = [it for it in all_nodes if 'chunk128' in it.id_]
        chunk_512_index_nodes = [it for it in all_nodes if 'chunk128' not in it.id_]
        #from utils import get_node_with_embedding
        if embed_model:
            chunk_128_index_nodes = get_node_with_embedding(chunk_128_index_nodes,embedding_model=embed_model)
        else:
            chunk_128_index_nodes = get_node_with_embedding(chunk_128_index_nodes,embedding_model=self.embed_model)
        all_nodes = chunk_512_index_nodes + chunk_128_index_nodes
        return all_nodes

    @staticmethod
    def static_embed_nodes_mp(all_nodes,embed_model=None):
        import copy
        _all_nodes = copy.deepcopy(all_nodes)
        chunk_128_index_nodes = [it for it in _all_nodes if 'chunk128' in it.id_]
        chunk_512_index_nodes = [it for it in _all_nodes if 'chunk128' not in it.id_]
        # logger.info(f'loading {len(all_nodes)} nodes, in which {len(chunk_128_index_nodes)} chunk128 nodes need embed')
        for n in chunk_128_index_nodes:
            n.embedding=None
        #from utils import get_node_with_embedding
        chunk_128_index_nodes = get_node_with_embedding_mul(chunk_128_index_nodes,embedding_model=embed_model)
        _all_nodes = chunk_512_index_nodes + chunk_128_index_nodes
        return _all_nodes

    @staticmethod
    def static_embed_nodes(all_nodes,embed_model=None):
        import copy
        _all_nodes = copy.deepcopy(all_nodes)
        chunk_128_index_nodes = [it for it in _all_nodes if 'chunk128' in it.id_]
        chunk_512_index_nodes = [it for it in _all_nodes if 'chunk128' not in it.id_]
        # logger.info(f'loading {len(all_nodes)} nodes, in which {len(chunk_128_index_nodes)} chunk128 nodes need embed')
        for n in chunk_128_index_nodes:
            n.embedding=None
        #from utils import get_node_with_embedding
        chunk_128_index_nodes = get_node_with_embedding(chunk_128_index_nodes,embedding_model=embed_model)
        _all_nodes = chunk_512_index_nodes + chunk_128_index_nodes
        return _all_nodes

    def parse_nodes_mul(
        self,
        documents:List[Document]=None,
    )->List[BaseNode]:
        #import pdb;pdb.set_trace()
        #print(self)
        all_nodes = []
        chunk_512_parser = self.sub_node_parsers['big']
        chunk_128_parser = self.sub_node_parsers['small']
        chunk_func = self.hierarchy_chunking

        workers=10
        pool = Pool(workers)
        length = len(documents)
        random.shuffle(documents) ##打乱文档，避免长文档聚集在一个子进程
        res = []
        for i in range(workers):

            p1 = length * i // workers
            p2 = length * (i + 1) // workers
            #args = [p1, p2, image_root, workers, essos_plg, cls_plg, image_root, json_root]
            #print(f"{p1},{p2}")
            argss = (documents[p1:p2],chunk_512_parser,chunk_128_parser)
            r = pool.apply_async(func=chunk_func, args=argss)
            res.append(r)
        pool.close()
        pool.join()
        all_nodes = []
        for r in res:
            tmp = r.get()
            all_nodes.extend(tmp)
        return all_nodes

    def parse_nodes(
        self,
        documents:List[Document]=None,
        #chunk_512_parser:SimpleNodeParser=None,
        #chunk_128_parser:SimpleNodeParser=None
    )->List[BaseNode]:
        #import pdb;pdb.set_trace()
        all_nodes = []
        chunk_512_parser = self.sub_node_parsers['big']
        chunk_128_parser = self.sub_node_parsers['small']
        all_nodes = self.hierarchy_chunking(documents,chunk_512_parser,chunk_128_parser)
        '''for document in tqdm(documents):
            doc_id = document.doc_id
            chunk_512s = chunk_512_parser.get_nodes_from_documents([document])

            for idx in range(len(chunk_512s)):
                next_idx = f'doc_{doc_id}_chunk512_{idx+1}'
                prev_idx = f'doc_{doc_id}_chunk512_{idx-1}'
                cur_idx = f'doc_{doc_id}_chunk512_{idx}'
                if NodeRelationship.PREVIOUS in chunk_512s[idx].relationships:
                    chunk_512s[idx].relationships[NodeRelationship.PREVIOUS].node_id = prev_idx
                if NodeRelationship.NEXT in chunk_512s[idx].relationships:
                    chunk_512s[idx].relationships[NodeRelationship.NEXT].node_id = next_idx
                chunk_512s[idx].id_ = cur_idx

            for idx_512,chunk_512 in enumerate(chunk_512s):
                chunk_128s = chunk_128_parser.get_nodes_from_documents([chunk_512])
                for idx_128 in range(len(chunk_128s)):
                    next_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128+1}'
                    prev_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128-1}'
                    cur_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128}'

                    if NodeRelationship.PREVIOUS in chunk_128s[idx_128].relationships:
                        chunk_128s[idx_128].relationships[NodeRelationship.PREVIOUS].node_id = prev_idx
                    if NodeRelationship.NEXT in chunk_128s[idx_128].relationships:
                        chunk_128s[idx_128].relationships[NodeRelationship.NEXT].node_id = next_idx
                    chunk_128s[idx_128].id_ = cur_idx

                sub_inodes = [
                    IndexNode.from_text_node(sn, chunk_512.node_id) for sn in chunk_128s
                ]
                all_nodes.extend(sub_inodes)
                #chunk_128_index_nodes.extend(sub_inodes)
                original_node = IndexNode.from_text_node(chunk_512, chunk_512.node_id)
                all_nodes.append(original_node)'''
        #all_nodes = self.embed_nodes(all_nodes)
        return all_nodes

    @staticmethod
    def hierarchy_chunking(
        documents,chunk_big_parser,chunk_small_parser
    ):
        chunk_512_parser = chunk_big_parser
        chunk_128_parser = chunk_small_parser

        nodes = []
        for document in tqdm(documents):
            doc_id = document.doc_id
            chunk_512s = chunk_512_parser.get_nodes_from_documents([document])

            for idx in range(len(chunk_512s)):
                next_idx = f'doc_{doc_id}_chunk512_{idx+1}'
                prev_idx = f'doc_{doc_id}_chunk512_{idx-1}'
                cur_idx = f'doc_{doc_id}_chunk512_{idx}'
                if NodeRelationship.PREVIOUS in chunk_512s[idx].relationships:
                    chunk_512s[idx].relationships[NodeRelationship.PREVIOUS].node_id = prev_idx
                if NodeRelationship.NEXT in chunk_512s[idx].relationships:
                    chunk_512s[idx].relationships[NodeRelationship.NEXT].node_id = next_idx
                chunk_512s[idx].id_ = cur_idx

            for idx_512,chunk_512 in enumerate(chunk_512s):
                chunk_128s = chunk_128_parser.get_nodes_from_documents([chunk_512])
                for idx_128 in range(len(chunk_128s)):
                    next_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128+1}'
                    prev_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128-1}'
                    cur_idx = f'doc_{doc_id}_chunk512_{idx_512}_chunk128_{idx_128}'

                    if NodeRelationship.PREVIOUS in chunk_128s[idx_128].relationships:
                        chunk_128s[idx_128].relationships[NodeRelationship.PREVIOUS].node_id = prev_idx
                    if NodeRelationship.NEXT in chunk_128s[idx_128].relationships:
                        chunk_128s[idx_128].relationships[NodeRelationship.NEXT].node_id = next_idx
                    chunk_128s[idx_128].id_ = cur_idx

                sub_inodes = [
                    IndexNode.from_text_node(sn, chunk_512.node_id) for sn in chunk_128s
                ]
                nodes.extend(sub_inodes)
                #chunk_128_index_nodes.extend(sub_inodes)
                original_node = IndexNode.from_text_node(chunk_512, chunk_512.node_id)
                nodes.append(original_node)
        return nodes
