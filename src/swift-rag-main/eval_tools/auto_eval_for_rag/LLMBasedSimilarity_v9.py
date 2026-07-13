# coding:utf-8
import json
import os
import time
from numpy import *
import jieba
import requests
from tqdm import tqdm
from remote_generation import remote_generate
from bertBasedSimilarity import DenseEmbeddings
from config import *
os.environ["TOKENIZERS_PARALLELISM"]="false"

def prompter_generation(question, answer_gt, answer_rag):

    base_prompter = """<|im_start|>user\n分析过程：
1. 作为一个文本相似度分打分专家，针对给定的问题，请对参考答案和AI答案，两个文本内容进行相似度比较，进行打分, 同时打分围绕这2个纬度进行思考判断，一方面针对AI答案内容是否有效理解问题意图、解决问题，另外一方面，将参考答案内容与AI答案内容对比，重点分析语义描述上的差异。
2. 在理解给定的问题方面，打分需要基于以下打分标准评估完毕后给一个1~5的分数：
    a. 0-1分：对于给定的问题，AI答案未能正确理解题目要求，偏离了题目主题。
    b. 1-2分：对于给定的问题，AI答案对题目要求有所理解, 但理解的不够全面。
    c. 2-3分：对于给定的问题，AI答案能够正确理解题目要求。
    d. 3-4分：对于给定的问题，AI答案完全理解题目要求，能够全面、清晰地回答问题。
    e. 4-5分：对于给定的问题，AI答案对题目要求的理解明显超出预期。
3. 在对比参考答案与AI答案方面，打分需要基于以下打分标准评估完毕后给一个1~5的分数：
    a. 0-1分：与参考答案中明确的观点相比，AI答案描述的内容差异很大，甚至可能存在严重的错误或偏差。
    b. 1-2分：与参考答案中明确的观点相比，AI答案并没有给出近似的结论，可能存在一些错误或不完全准确。
    c. 2-3分：与参考答案相比，AI答案在描述上给出了近似的观点，但可能存在一些不够清晰或不完整的地方，或者缺乏细节或具体支持。
    d. 3-4分：与参考答案相比，AI答案准确性高，逻辑性强，表述清晰。
    e. 4-5分：与参考答案相比，AI答案除了能够准确无误回答问题，还能够深入挖掘问题的内涵，提出独特见解，在表达、逻辑结构等方面也达到了极高水平，完美地满足了题目的所有要求。
4. 只需要输出最终得分，不需要多余的分析内容。根据上面2个纬度分别打一次分然后取平均分，分数可以保留2位小数，得分请按照以下格式来输出：综合得分：x。例如综合得分为2.1分，那么输出：综合得分：2.1 \n问题：{question}。\n参考答案：{answer_gt}。\n AI答案：{answer_rag}。\n<|im_end|>\n<|im_start|>assistant\n"""

    message = base_prompter.format(question=question, answer_gt=answer_gt, answer_rag = answer_rag)
    return message

def prompter_generation2(question, answer_gt, answer_rag):

    base_prompter = """<|im_start|>user\n分析过程：
一、 作为一个文本相似度分打分专家，请对句子1和句子2，两个文本内容进行相似度比较，进行打分, 将句子1的内容与句子2的内容对比，重点分析语义描述上的差异。
二、 打分需要基于以下打分标准评估完毕后给一个1~5的分数：
    a. 1分：与句子1中明确的观点相比，句子2描述了严重的错误或偏差的观点, 逻辑性一般。
    b. 2分：与句子1中明确的结论观点相比，句子2并没有给出匹配的观点，缺乏细节或具体支持。
    c. 3分：与句子1相比，句子2描述了近似的观点，但存在一些不够清晰或不完整的地方。
    d. 4分：与句子1相比，句子2准确性高，逻辑性强，表述清晰，论证严密。
    e. 5分：与句子1相比，句子2除了不仅完全准确，在表达、逻辑结构等方面也达到了极高水平。
三、 只需要输出最终得分，不需要多余的分析内容。根据上面5个等级进行打分，分数可以保留2位小数，得分请按照以下格式来输出：综合得分：x。例如综合得分为2.1分，那么输出：综合得分：2.1。\n句子1：{answer_gt}。\n 句子2：{answer_rag}。\n<|im_end|>\n<|im_start|>assistant\n"""

    message = base_prompter.format(answer_gt=answer_gt, answer_rag = answer_rag)
    return message

def prompter_generation3(question, answer_gt, answer_rag):


    #example1_query = ""
    example1_gt = "不算。股票发行采用代销方式，代销期限届满，向投资者出售的股票数量未达到拟公开发行股票数量百分之七十的，为发行失败。发行人应当按照发行价并加算银行同期存款利息返还股票认购人。"
    example1_rag = "根据提供的信息，无法判断证券公司代销证券时向投资者出售了预期数量百分之六十的股票是否算发行成功。"
    example1_score = "0.8"

    example2_gt = "在每一会计年度结束之日起四个月内，报送并公告年度报告，其中的年度财务会计报告应当经符合本法规定的会计师事务所审计。"
    example2_rag = "上市公司的年报需要在每个会计年度结束后的四个月内报送。"
    example2_score = "3.5"

    example3_gt = "根据《关于证券违法行为人财产优先用于承担民事赔偿责任有关事项的规定》，证监会行政处罚委员会办公室具体负责接收投资者申请材料。"
    example3_rag = "根据《关于证券违法行为人财产优先用于承担民事赔偿责任有关事项的规定》，您应该将申请材料提交给中国证券监督管理委员会（以下简称证监会）。具体的接收部门是证监会行政处罚委员会办公室。"
    example3_score = "4.3"



    base_prompter = """<|im_start|>user\n分析过程：
一、 作为一个文本相似度分打分专家，请对描述1和描述2，两个文本内容进行相似度比较，进行打分, 将描述1的内容与描述2的内容对比，重点分析语义描述上的差异。
二、 打分需要基于以下打分标准评估完毕后给一个1~5的分数：
    a. 0分到1分：与描述1的内容或观点相比，描述2表达了比较大偏差的观点, 逻辑性一般。
    b. 1-2分：与描述1中内容或观点相比，描述2并没有给出近似的观点，缺乏细节或具体支持。
    c. 2-3分：与描述1中内容或观点相比，描述2虽然给出了近似的观点，但存在一些不够清晰或不完整的地方。
    d. 3-4分：与描述1中内容或观点相比，描述2中的表达准确性高，表述清晰。
    e. 4-5分：与描述1中内容或观点相比，描述2不仅表述完全准确，在表达、逻辑结构等方面也达到了极高水平。
三、 只需要输出最终得分，不需要多余的分析内容。根据上面5个等级进行打分，分数可以保留2位小数，得分请按照以下格式来输出：综合得分：x。例如综合得分为2.1分，那么输出：综合得分：2.1。\n描述1：{answer_gt}。\n 描述2：{answer_rag}。\n<|im_end|>\n<|im_start|>assistant\n"""

    fewshot_prompter = """<|im_start|>user\n分析过程：
一、 作为一个文本相似度分打分专家，请对描述1和描述2，两个文本内容进行相似度比较，进行打分, 将描述1的内容与描述2的内容对比，重点分析语义描述上的差异。
二、 打分需要基于以下打分标准评估完毕后给一个1~5的分数：
    a. 0分到1分：与描述1的内容或观点相比，描述2表达了比较大偏差的观点, 逻辑性一般。
    b. 1-2分：与描述1中内容或观点相比，描述2并没有给出近似的观点，缺乏细节或具体支持。
    c. 2-3分：与描述1中内容或观点相比，描述2虽然给出了近似的观点，但存在一些不够清晰或不完整的地方。
    d. 3-4分：与描述1中内容或观点相比，描述2中的表达准确性高，逻辑性强，表述清晰，论证严密。
    e. 4-5分：与描述1中内容或观点相比，描述2不仅表述完全准确，在表达、逻辑结构等方面也达到了极高水平。
三、 只需要输出最终得分，不需要多余的分析内容。根据上面5个等级进行打分，分数可以保留2位小数，得分请按照以下格式来输出：综合得分：x。例如综合得分为2.1分，那么输出：综合得分：2.1。\n描述1：{answer_gt}。\n 描述2：{answer_rag}。\n<|im_end|>\n<|im_start|>assistant\n综合得分：{score}<|im_end|>\n"""

    fewshot1 = fewshot_prompter.format(answer_gt=example1_gt, answer_rag = example1_rag, score = example1_score)
    fewshot2 = fewshot_prompter.format(answer_gt=example2_gt, answer_rag = example2_rag, score = example2_score)
    fewshot3 = fewshot_prompter.format(answer_gt=example3_gt, answer_rag = example3_rag, score = example3_score)
    #message = base_prompter.format(answer_gt=answer_gt, answer_rag = answer_rag)
    message = fewshot1 + fewshot2 + fewshot3 + base_prompter.format(answer_gt=answer_gt, answer_rag = answer_rag)
    return message




#def score_by_LLM(question, gt, answer, model_name='nova_sensechatv5'):
#def score_by_LLM(question, gt, answer, model_name='azure_gpt4'):
def score_by_LLM(question, gt, answer, model_name='azure_gpt4_32k'):
    message = prompter_generation(question, gt, answer)
    #message = prompter_generation3(question, gt, answer)
    #message = prompter_generation3(question, gt, answer)
    #print(message)
    response = remote_generate(model_name)(message=message)
    return response




if __name__ == '__main__':

    embedding_model = DenseEmbeddings(model_name=EMBEDDING_PATH)
    #with open("result.jsonl") as fr:
    #with open("result_7K_24_04_29.jsonl") as fr:
    #with open("result_7K_round2_24_04_30.jsonl") as fr:
    #with open("result_7K_round2_24_04_30.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_m3e_norerank.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_m3e_rerank.jsonl") as fr:
    #with open("noindent_1.1_compliance_bert_102b_bge_norerank.jsonl") as fr:
    with open("noindent_1.2_compliance_bert_102b_m3e_norerank.jsonl") as fr:
        results = fr.readlines()


    flag = 0
    #for result in results:
    for result in tqdm(results):
        line = json.loads(result)
        flag = flag + 1
        #if flag > 20 and flag < 25:
        #if flag > 13 and flag < 17:
        #while True:
        #if flag > 13 and flag < 17:
        #if flag == 21 or flag == 32 or flag == 40 or flag == 45:
        #if flag == 6 or flag == 14 or flag == 21 or flag == 33 or flag == 34:
        #if flag == 5 or flag == 21 or flag == 32 or flag == 40 or flag == 45:
        #if flag > 2500:
        #if flag > 4307:
        if True:
            print(line["question"])
            #compute cos_similarity
            cos_similarity = embedding_model._get_cos_similarity(line["answer"], line["端对端结果"])
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
                while (infer_time < GPT_ITERATION):
                    try:
                        result = score_by_LLM(line["question"], line["answer"], line["端对端结果"])
                        str_score = result.split("：")[-1]
                        str_score = str_score.replace(' ', '')
                        str_score = str_score.replace('。', '')
                        str_score = str_score.replace('分', '')
                        #llm_score = float(str_score)
                        score = float(str_score)
                        #if llm_score >= 0.0 and llm_score <= 5.0:
                        if score >= 0.0 and score <= 5.0:
                            print("iterative llm score-%d : %.2f" % (infer_time, score))
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
            print(final_score)

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
            with open("rating_v9_500_1.2_norerank_result_7K_24_05_09.jsonl", "a") as fw:
                fw.write(json.dumps(line, ensure_ascii=False) + "\n")
