# coding:utf-8
import json
#import pdb

import jieba
#import lightgbm as lgb
import nltk
import numpy as np
#import torch
import transformers
#from meteor_score import single_meteor_score, single_meteor_score_no_sorted
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
#from tqdm import tqdm


def score_by_TF(sentence_1, sentence_2):
    # 计算feature
    input_1 = " ".join(jieba.cut(sentence_1))
    set_1 = set(input_1)
    input_2 = " ".join(jieba.cut(sentence_2))
    set_2 = set(input_2)
    rouge = Rouge()

    scores = rouge.get_scores(input_1, input_2)
    rouge_score = scores[0]["rouge-1"]["r"]

    bleu = sentence_bleu([input_1], input_2)

    #meteor = single_meteor_score(set_1, set_2)

    return bleu, rouge_score
