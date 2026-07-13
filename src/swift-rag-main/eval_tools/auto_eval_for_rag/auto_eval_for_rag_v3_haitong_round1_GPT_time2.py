# coding:utf-8
import json
import os
import time
from numpy import *
import jieba
import requests
from tqdm import tqdm
from bertSimilarity.bertBasedSimilarity import DenseEmbeddings
from LLMSimilarity.LLMBasedSimilarity import score_by_LLM
from TFSimilarity.TFBasedSimilarity import score_by_TF
from config import *
os.environ["TOKENIZERS_PARALLELISM"]="false"

def auto_rating(result_file, rating_file):
    #with open(READ_JSON) as fr:
    with open(result_file) as fr:
        results = fr.readlines()


    flag = 0
    #for result in results:
    for result in tqdm(results):
        line = json.loads(result)
        gt_answer = line["answer"]
        #chat_answer = line["chat_answer"]
        chat_answer = line["端对端结果"]
        if not chat_answer:
            chat_answer = "空"
        #flag = flag + 1
        #if flag > 20 and flag < 25:
        #if flag > 22 and flag < 30:
        #if flag > 300:
        #if flag > 10393:
        if True:
            print(line["question"])
            #compute bleu and rouge
            #bleu_score, rouge_score = score_by_TF(line["answer"], line["端对端结果"])
            bleu_score, rouge_score = score_by_TF(gt_answer, chat_answer)
            #compute cos_similarity
            #cos_similarity = embedding_model._get_cos_similarity(line["answer"], line["端对端结果"])
            cos_similarity = embedding_model._get_cos_similarity(gt_answer, chat_answer)
            if cos_similarity < 0.8:
                cos_score = 0.5
            elif cos_similarity < 0.9:
                cos_score = 1.5
            elif cos_similarity < 0.93:
                cos_score = 2.5
            elif cos_similarity < 0.96:
                cos_score = 3.5
            else:
                cos_score = 4.5
            print("cos_score: %.2f" % cos_score)

            if COS_THRESHOLD and (cos_score >= 3.5):
                llm_score = 4.5
            else:
                #middle_score = [0.0, 0.0]
                middle_score = []
                #while True:
                infer_time = 0
                #while (infer_time < GPT_ITERATION):
                while (infer_time < 2):
                    try:
                        #result = score_by_LLM(line["question"], line["answer"], line["端对端结果"])
                        result = score_by_LLM(line["question"], gt_answer, chat_answer)
                        str_score = result.split("：")[-1]
                        str_score = str_score.replace(' ', '')
                        str_score = str_score.replace('。', '')
                        str_score = str_score.replace('分', '')
                        #llm_score = float(str_score)
                        score = float(str_score)
                        #if llm_score >= 0.0 and llm_score <= 5.0:
                        if score >= 0.0 and score <= 5.0:
                            #print("iterative llm score-%d : %.2f" % (infer_time, score))
                            #middle_score[infer_time] = score
                            middle_score.append(score)
                            infer_time = infer_time + 1
                            #llm_score = score
                            #break
                    except Exception as e:
                        print(e)
                        time.sleep(1)
                        pass

                #llm_score = (middle_score[0] + middle_score[1]) * 0.5
                llm_score = mean(middle_score)

            print("LLM score: %.2f" % llm_score)
            #avg_score = llm_score * 0.7 + cos_score * 0.3
            avg_score = llm_score * 0.8 + cos_score * 0.2
            print("Avg score: %.2f" % avg_score)
            #final_score =
            #if score >= 0.0 and score <= 5.0:
            if avg_score < 2.0:
                final_score = 0
            elif avg_score < 3.0:
                final_score = 1
            else:
                final_score = 2
            print("0-1-2 score: %d" % final_score)
            print("bleu score: %.2f" % bleu_score)
            print("rouge score: %.2f" % rouge_score)

            line["bleu_score"] = bleu_score
            line["rouge_score"] = rouge_score
            line["cos_sim"] = float(cos_similarity)
            line["bert_Score"] = cos_score
            line["llm_Score"] = llm_score
            line["avg_Score"] = avg_score
            line["AI_Score"] = final_score
            #with open("rating_v7_round2_result_7K_24_05_01.jsonl", "a") as fw:
            #with open("rating_v8_ht_70_24_05_01.jsonl", "a") as fw:
            #with open("rating_v8_round1_result_7K_24_05_02.jsonl", "a") as fw:
            #with open("rating_v9_round1_result_7K_24_05_06.jsonl", "a") as fw:
            #with open("rating_v9_round2_result_7K_24_05_07.jsonl", "a") as fw:
            #with open("rating_v9_500_m3e_norerank_result_7K_24_05_08.jsonl", "a") as fw:
            #with open("rating_v9_500_m3e_rerank_result_7K_24_05_09.jsonl", "a") as fw:
            #with open("rating_v9_500_bge_norerank_result_7K_24_05_09.jsonl", "a") as fw:
            #with open("rating_v9_rgb_0510_round1_jsonl_chat_date_20240514.jsonl", "a") as fw:
            #with open(WRITE_JSON, "a") as fw:
            with open(rating_file, "a") as fw:
                fw.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == '__main__':

    embedding_model = DenseEmbeddings(model_name=EMBEDDING_PATH)
    #with open("result.jsonl") as fr:
    #with open("result_7K_24_04_29.jsonl") as fr:
    #with open("result_7K_round2_24_04_30.jsonl") as fr:
    #with open("result_7K_round2_24_04_30.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_m3e_norerank.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_m3e_rerank.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_bge_norerank.jsonl") as fr:
    #with open("noindent_1.2_compliance_bert_102b_m3e_norerank.jsonl") as fr:
    #with open("rgb_0510_round1_jsonl_chat_date_20240514.jsonl") as fr:
    #IN_DIR = "rgb_510_results"
    #JSON_NAME = "rgb_510_910B_240524_jsonl_chat.jsonl"
    ##IN_DIR = "haitong_v6_date_0607_results"
    IN_DIR = "haitong_v6_date_0611_results"
    #IN_DIR = "rgb_420_mini_results"
    #IN_DIR = "haitong_v6_results"
    #JSON_NAME = "exp_v1.1_one_mini_piccolo-large-zh-v2_result.jsonl"
    #JSON_NAME = "exp_v1.1_one_mini_infgrad_stella-base-zh-v3-1792d_result.jsonl"
    #JSON_NAMES = ["exp_v1.1_one_mini_infgrad_stella-base-zh-v3-1792d_result.jsonl", "exp_v1.1_one_mini_infgrad_stella-mrl-large-zh-v3.5-1792d_result.jsonl", "exp_v1.1_one_mini_bce-embedding-base_v1_result.jsonl", "exp_v1.1_one_mini_Pristinenlp_alime-embedding-large-zh_result.jsonl", "exp_v1.1_one_mini_bge-m3_result.jsonl"]
    #JSON_NAMES = ["sense-rag-v1.1_one_general_align_result.jsonl"]
    '''
    JSON_NAMES = [
        "baseline.jsonl",
        "baseline_simq.jsonl",
        "bge-large-zh-v1.5_result.jsonl",
        "bge-m3_result.jsonl",
        "bge-m3_result_similarq.jsonl",
        "gte-large-zh_result.jsonl",
        "m3e-base_result.jsonl",
        "Pristinenlp_alime-embedding-large-zh_result.jsonl"
    ]
    JSON_NAMES = [
        "bge-large-zh-v1.5+bge-m3_rrf_top10_result.jsonl",
        "bge-large-zh-v1.5+bge-m3_rrf_top10_result_similarq.jsonl",
        "bge-large-zh-v1.5+bge-m3_rrf_top5_result.jsonl",
        "bge-large-zh-v1.5+bge-m3_rrf_top5_result_similarq.jsonl",
        "bge-large-zh-v1.5+bge-m3_simple_merge_result.jsonl",
        "bge-large-zh-v1.5+bge-m3_simple_merge_result_similarq.jsonl",
        "bge-large-zh-v1.5+infgrad_stella-mrl-large-zh-v3.5-1792d_simple_merge_result.jsonl",
        "bge-large-zh-v1.5+infgrad_stella-mrl-large-zh-v3.5-1792d_simple_merge_result_similarq.jsonl",
    ]
    '''
    JSON_NAMES = ["bge-large-zh-v1.5+bge-m3_simple_merge_result_similarq.jsonl"]
    #OUT_DIR = "rating_rgb_510_results"
    for JSON_NAME in JSON_NAMES:
        OUT_DIR = "rating_" + IN_DIR
        READ_JSON = IN_DIR + "/" + JSON_NAME
        #WRITE_JSON = OUT_DIR + "/" + "rating_" + JSON_NAME
        WRITE_JSON = OUT_DIR + "/" + "rating_GPT_iter_2_" + JSON_NAME
        auto_rating(READ_JSON, WRITE_JSON)
