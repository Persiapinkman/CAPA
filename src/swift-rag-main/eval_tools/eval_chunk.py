import pandas as pd
import argparse
import ast
from typing import List, Dict, Any, Optional


def read_xlsx_to_dict_list(
    file_path: str,
    sheet_name: Optional[str] = None,
    sheet_index: int = 0
) -> List[Dict[str, Any]]:
    """
    读取指定的xlsx文件，将表格数据转换为字典列表

    参数:
        file_path: xlsx文件路径
        sheet_name: 表格名称，如果提供则优先使用
        sheet_index: 表格索引，默认为0（第一个表格）

    返回:
        包含表格数据的字典列表，每行数据转换为一个字典
    """
    try:
        # 如果提供了sheet_name，则使用sheet_name，否则使用sheet_index
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_index)

        # 只保留"问题"不为空的行
        df = df.dropna(subset=['问题'])

        # 将DataFrame转换为字典列表
        result = df.to_dict(orient='records')

        print(f"成功从文件 {file_path} 读取了 {len(result)} 行数据")
        return result
    except Exception as e:
        print(f"读取xlsx文件时出错: {str(e)}")
        return []


def parse_search_list(data):
    """解析search_list字段"""
    for data_item in data:
        search_list_str = data_item['search_list']
        try:
            search_list = ast.literal_eval(search_list_str)
            data_item['search_list'] = search_list
        except Exception as e:
            print(f"解析search_list时出错: {str(e)}")
            print(data_item['search_list'])
    return data


def evaluate_retrieval(data, source_field='问题来源段落截取', retrieval_field='search_list', match_method='substring', threshold=0.8):
    """
    评估RAG检索阶段的召回效果，使用最长公共子串或最长公共子序列进行匹配

    参数:
        data: 包含标注数据的列表
        source_field: 标注的原文段落字段名
        retrieval_field: 检索结果字段名
        match_method: 匹配方法，'substring'表示最长公共子串，'subsequence'表示最长公共子序列
        threshold: 最长匹配比例阈值，大于该值视为命中

    返回:
        包含评估结果的字典
    """
    print(f"开始评估检索效果...使用{match_method}方法，阈值为{threshold}")

    total_queries = 0
    hit_count = 0

    def find_longest_common_substring(s1, s2):
        """查找两个字符串的最长公共子串"""
        if not s1 or not s2:
            return "", 0

        m = len(s1)
        n = len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        end_pos = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i

        if max_length == 0:
            return "", 0

        common_substring = s1[end_pos - max_length:end_pos]
        return common_substring, max_length

    def find_longest_common_subsequence(s1, s2):
        """查找两个字符串的最长公共子序列"""
        if not s1 or not s2:
            return "", 0

        m = len(s1)
        n = len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # 重建最长公共子序列
        lcs_length = dp[m][n]
        if lcs_length == 0:
            return "", 0

        lcs = [""] * lcs_length
        i, j = m, n
        index = lcs_length - 1

        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                lcs[index] = s1[i-1]
                i -= 1
                j -= 1
                index -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return "".join(lcs), lcs_length

    # 根据选择的方法确定匹配函数
    match_function = find_longest_common_substring if match_method == 'substring' else find_longest_common_subsequence

    for i, item in enumerate(data):
        # 跳过没有标注或检索结果的项
        if not item.get(source_field) or not item.get(retrieval_field):
            print(f"跳过没有标注或检索结果的项: {item}")
            continue

        total_queries += 1
        source_text = str(item[source_field]).strip()
        retrieved_texts = item[retrieval_field]

        hit = False
        best_match_length = 0
        best_match_source_ratio = 0
        best_common_match = ""
        best_match_retrieved_text = "未命中目标文件"

        # 对每个召回结果进行处理
        for retrieved_text in retrieved_texts:
            # 只考虑匹配到对应文件的检索结果
            if retrieved_text['docname'] not in item['问题文件来源']:
                continue

            retrieved_text = str(retrieved_text['indextext']).strip()

            # 预处理原始文本和检索文本
            source_text_processed = source_text.replace('\\n', '').replace('\n', '').replace(' ', '').strip()
            retrieved_text_processed = retrieved_text.replace('\\n', '').replace('\n', '').replace(' ', '').strip()

            # 使用选定的匹配方法进行匹配
            common_match, match_length = match_function(source_text_processed, retrieved_text_processed)

            # 计算匹配比例
            source_ratio = match_length / len(source_text_processed) if len(source_text_processed) > 0 else 0

            # 找到多个匹配中，匹配比例最大的
            if match_length > best_match_length:
                best_match_length = match_length
                best_common_match = common_match
                best_match_source_ratio = source_ratio
                best_match_retrieved_text = retrieved_text_processed

        # 使用传入的阈值判断是否命中
        if best_match_source_ratio > threshold:
            hit = True
            hit_count += 1

        detailed_result = {
            "是否命中": 1 if hit else 0,
            "最匹配的检索结果": best_match_retrieved_text,
            "最长匹配": best_common_match,
            "最长匹配比例": best_match_source_ratio,
            "最长匹配长度": best_match_length,
        }

        item.update(detailed_result)

    # 计算召回率
    recall = hit_count / total_queries if total_queries > 0 else 0

    print(f"评估完成: 总查询数 {total_queries}, 命中数 {hit_count}, 召回率 {recall:.2%}")
    return data


def print_missed_queries(evaluation_results, limit=30):
    """打印未命中的查询"""
    print("\n未命中的查询:")
    for result in sorted(evaluation_results, key=lambda x: x['最长匹配比例'])[:limit]:
        print(f"第{result['序号']}个问题:")
        print(f"问题: {result['问题']}")
        print(f"原文段落: {result['问题来源段落截取']}")
        print(f"最匹配的检索结果: {result['最匹配的检索结果']}")
        print(f"最长匹配: {result['最长匹配']}")
        print(f"最长匹配长度: {result['最长匹配长度']}")
        print(f"最长匹配比例: {result['最长匹配比例']:.2%}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='评估RAG检索阶段的召回效果')
    parser.add_argument('--file_path', type=str, default="eval_tools/data/呈贡RAG层测试集v0.1.xlsx",
                        help='xlsx文件路径')
    parser.add_argument('--sheet_name', type=str, default="20250401 internlm 20b chat",
                        help='表格名称')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='最长匹配比例阈值，大于该值视为命中')
    parser.add_argument('--output_file', type=str, default="eval_tools/data/eval_results.xlsx",
                        help='输出结果文件路径')
    args = parser.parse_args()

    # 读取数据
    data = read_xlsx_to_dict_list(
        file_path=args.file_path,
        sheet_name=args.sheet_name
    )

    # 解析search_list
    data = parse_search_list(data)

    # 执行评估 - 使用最长公共子序列和命令行参数中的阈值
    evaluation_results = evaluate_retrieval(data, match_method='subsequence', threshold=args.threshold)

    # 将结果转换为DataFrame并导出到Excel
    df_results = pd.DataFrame(evaluation_results)
    df_results.to_excel(args.output_file, index=False)
    print(f"评估结果已保存到 {args.output_file}")

    # # 输出未命中的查询，帮助分析问题
    # print_missed_queries(evaluation_results)


if __name__ == "__main__":
    main()