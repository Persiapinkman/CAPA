# coding:utf-8
import json
import os
import time

import jieba
import requests
from tqdm import tqdm
from typing import Any, List

#from llama_index.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import *
from sentence_transformers.util import cos_sim

class DenseEmbeddings():
    _model: Any = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = EMBEDDING_PATH,
        embed_batch_size: int = 256,
        device: str = 'cuda'
        ) -> None:
        #self._model = INSTRUCTOR(model_name)
        self._model = SentenceTransformer(model_name,device=device)

    @classmethod
    def class_name(cls) -> str:
        return "dense_embedding"

    def _get_text_embedding(self, text: str) -> float:
        if EMBEDDING_NORM:
            embedding = self._model.encode(text, normalize_embeddings=True)
        else:
            embedding = self._model.encode(text)
        return embedding

    def _get_cos_similarity(self, text1: str, text2: str) -> float:
        if EMBEDDING_NORM:
            embedding1 = self._model.encode(text1, normalize_embeddings=True)
            embedding2 = self._model.encode(text2, normalize_embeddings=True)
        else:
            embedding1 = self._model.encode(text1)
            embedding2 = self._model.encode(text2)
        result = cos_sim(embedding1, embedding2)
        return result.detach().cpu().numpy().squeeze()








if __name__ == '__main__':

    embedding_model = DenseEmbeddings(model_name=EMBEDDING_PATH)
    with open("result.jsonl") as fr:
        results = fr.readlines()


    flag = 0
    #for result in tqdm(results):
    for result in results:
        line = json.loads(result)
        flag = flag + 1
        #if flag > 20 and flag < 25:
        #if flag > 13 and flag < 17:
        #while True:
        #if flag > 13 and flag < 17:
        #if flag == 21 or flag == 32 or flag == 40 or flag == 45:
        if True:
            print(line["question"])
            cos_score = embedding_model._get_cos_similarity(line["answer"], line["端对端结果"])

            #final_score =
            #if score >= 0.0 and score <= 5.0:
            print(cos_score)
            #if avg_score < 2.2:
            #    final_score = 0
            #elif avg_score < 3.1:
            #    final_score = 1
            #else:
            #    final_score = 2
            #print(final_score)

            line["AI_Score"] = float(cos_score)
            #line["AI_Score"] = final_score
            with open("rating_result_test_bert.jsonl", "a") as fw:
                fw.write(json.dumps(line, ensure_ascii=False) + "\n")
