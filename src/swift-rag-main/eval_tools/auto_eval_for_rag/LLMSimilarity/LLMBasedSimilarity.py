# coding:utf-8
import json
import os
import time
from numpy import *
import jieba
import requests
from tqdm import tqdm
from .remote_generation import remote_generate
#from bertBasedSimilarity import DenseEmbeddings
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
def score_by_LLM(question, gt, answer, openai_key, openai_endpoint, model_name='azure_gpt4_32k'):
    message = prompter_generation(question, gt, answer)
    #message = prompter_generation3(question, gt, answer)
    #message = prompter_generation3(question, gt, answer)
    #print(message)
    response = remote_generate(model_name)(message=message,openai_key = openai_key, openai_endpoint = openai_endpoint)
    return response
