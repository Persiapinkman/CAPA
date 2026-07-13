#Release Notes

-发布日期: 2024.5.17
-版本号: dev-v0.9

<div align=center> <img src="./figs/eval_methods.png" width=600/> </div>

#**快速运行**
1. 将脚本lauch_auto_eval.sh中的oAI的**key/endpoint, read/write file 填好**。
2. **运行./launch_auto_eval.sh**

#整体方案介绍及采用的模型配置
- 目前支持端到端结果的自动打分，整体方案包括了Word-based、Embedding Models和LLM三个工具进行自动打分。
- Word-based方法目前利用了BLEU和ROUGE两个指标，根据两个句子中的词频进行计算，BLEU侧重于衡量两个句子单词级别的准确性和精确匹配程度，更偏向于Precision，而ROUGE侧重于衡量信息完整性和涵盖程度，更偏向于Recall，两个指标也都是0到1，数值越大越好。
- Embedding模型目前利用m3e进行embedding抽取，采用Cosine Similarity计算语义相似度，输出结果为0到1分，数值结果越大代表语义越相似。
- LLM打分利用GPT4-32K模型，打分结果为0-5分, prompt为：

<div align=center> <img src="./figs/prompt_img.png" width=600/> </div>
