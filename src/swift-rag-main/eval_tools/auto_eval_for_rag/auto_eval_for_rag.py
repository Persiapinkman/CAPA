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

#def auto_rating(result_file, rating_file):
def auto_rating(args):
    #with open(READ_JSON) as fr:
    with open(args.read_jsonl_file) as fr:
        results = fr.readlines()


    flag = 0
    llm_Scores=[]
    final_Score=[]
    bleu_Scores=[]
    rouge_Scores=[]
    cos_Scores=[]

    #for result in results:
    for result in tqdm(results):
        line = json.loads(result)
        gt_answer = str(line["answer"])
        #chat_answer = line["chat_answer"]
        chat_answer = str(line["端对端结果"])
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
                while (infer_time < args.GPT_iter):
                    try:
                        #result = score_by_LLM(line["question"], line["answer"], line["端对端结果"])
                        result = score_by_LLM(line["question"], gt_answer, chat_answer, args.openai_key, args.openai_endpoint,args.llm_name)
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

            bleu_Scores.append(bleu_score)
            rouge_Scores.append(rouge_score)
            cos_Scores.append(float(cos_similarity))
            llm_Scores.append(llm_score)
            final_Score.append(final_score)
            #with open(rating_file, "a") as fw:
            with open(args.write_jsonl_file, "a") as fw:
                fw.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("mean bleu_score score: %f" % mean(bleu_Scores))
    print("mean rouge_score score: %f" % mean(rouge_Scores))
    print("mean cos_score score: %f" % mean(cos_Scores))
    print("mean 0-1-2 score: %f" % mean(final_score))
    print("mean llm_Score score: %f" % mean(llm_Scores))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='auto eval for rag')
    parser.add_argument('--GPT_iter', type=int, default=2,
                        help='the iteration num for calling GPT4')
    parser.add_argument('--openai_key', type=str, default=None,
                        help='the OPENAI_KEY for AZURE-GPT4')
    parser.add_argument('--openai_endpoint', type=str, default=None,
                        help='the OPENAI_ENDPOINT for AZURE-GPT4')
    parser.add_argument('--llm_name', type=str, default='gpt4o_ptu',
                        help='the OPENAI_ENDPOINT for AZURE-GPT4')
    parser.add_argument('--read_jsonl_file', type=str, default='example_eval.jsonl',
                        help='the read-jsonl-file for rating')
    parser.add_argument('--write_jsonl_file', type=str, default='rating_example_eval.jsonl',
                        help='the marked result')
    args = parser.parse_args()
    print(args)
    if os.path.exists(args.write_jsonl_file):
        os.remove(args.write_jsonl_file)
    embedding_model = DenseEmbeddings(model_name=EMBEDDING_PATH)
    auto_rating(args)
