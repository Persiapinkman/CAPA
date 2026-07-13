import os
import json
import re
import pandas as pd

from llama_index.core import Document
import hashlib
from text_splitter.text_splitter2 import RecursiveCharacterTextSplitter2

# 指定路径（可通过 AUTOPDF_ROOT_PATH 覆盖）
root_path = os.getenv("AUTOPDF_ROOT_PATH", "/app/data_source")

def get_all_folders(path=root_path):
    # 返回路径下的所有文件夹
    if os.path.exists(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        return []

def get_md5(text):
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    md5_val="_"+md5.hexdigest()
    return md5_val


def clean_text(text):
    # 定义常见的不可见字符和特殊空格
    invisible_chars = [
        '\u2003',  # EM SPACE
        '\u3000',  # IDEOGRAPHIC SPACE
        '\u2002',  # EN SPACE
        '\u2009',  # THIN SPACE
        '\u00A0',  # NO-BREAK SPACE
        '\u200B',  # ZERO WIDTH SPACE
        '\uFEFF',  # ZERO WIDTH NO-BREAK SPACE (BOM)
    ]

    # 去除 Unicode 控制字符
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)

    # 去除不可见空格类字符
    for ch in invisible_chars:
        text = text.replace(ch, ' ')

    # 替换多个连续空格为一个
    text = re.sub(r'\s+', ' ', text)

    return text.replace(' ','').replace('·',' ').strip()


def check_alternating_pattern(data, fixed_value, min_repeat=5):
    """
    判断 fixed_value 是否与其他内容交替出现超过 min_repeat 次。

    参数：
        data: list[str] 输入列表
        fixed_value: str 要检查的固定值
        min_repeat: int 最小交替次数阈值

    返回：
        bool：是否满足交替出现次数大于等于 min_repeat
    """
    count = 0
    i = 0
    while i < len(data) - 1:
        if data[i] != fixed_value and data[i + 1] == fixed_value:
            count += 1
            i += 2  # 跳过这一组
        else:
            i += 1  # 没命中，继续往下滑动
    return count >= min_repeat

def merge_small_all_text_into_next(content):
    # 创建副本以避免修改原始数据
    content = content.copy().reset_index(drop=True)
    indices_to_drop = []  # 记录需要删除的行索引
    total_len = len(content)

    # 定义辅助函数，将值转换为列表
    def to_list(x):
        if isinstance(x, list):
            return x
        elif pd.isna(x):
            return []
        else:
            return [x]

    for idx in range(total_len):
        current_row = content.loc[idx]
        current_len = current_row['all_text_len']

        if current_len >= 40:
            continue  # 如果长度足够，跳过该行

        # 检查是否有下一行可以合并
        if idx + 1 < total_len:
            next_row = content.loc[idx + 1]

            # 合并 'all_text'
            merged_text = ((current_row['all_text'] or '') + '\n\n' + (next_row['all_text'] or '')).strip()
            content.at[idx + 1, 'all_text'] = merged_text

            # 合并 'all_img_paths'
            if 'all_img_paths' in content.columns:
                current_img_paths = to_list(current_row['all_img_paths'])
                next_img_paths = to_list(next_row['all_img_paths'])
                combined_img_paths = current_img_paths + next_img_paths
                content.at[idx + 1, 'all_img_paths'] = combined_img_paths

            # 合并 'header'，避免重复
            if (current_row['header'] == next_row['header']) or (str(current_row['header']).strip() == str(next_row['header']).strip()):
                # 如果相同，保留一个即可
                merged_header = next_row['header']
            else:
                # 如果不同，按顺序合并
                merged_header = ((current_row['header'] or '') + ' ' + (next_row['header'] or '')).strip()
            content.at[idx + 1, 'header'] = merged_header
            # 合并 'heading'，避免重复
            if (current_row['heading'] == next_row['heading']) or (str(current_row['heading']).strip() == str(next_row['heading']).strip()):
                merged_heading = next_row['heading']
            else:
                merged_heading = ((current_row['heading'] or '') + ' ' + (next_row['heading'] or '')).strip()
            content.at[idx + 1, 'heading'] = merged_heading

            # 更新 'all_text_len'
            content.at[idx + 1, 'all_text_len'] = len(merged_text)

            # 标记当前行以便删除
            indices_to_drop.append(idx)
        elif idx - 1 >= 0:
            # 如果没有下一行，合并到上一行
            prev_row = content.loc[idx - 1]

            # 合并 'all_text'
            merged_text = ((prev_row['all_text'] or '') + '\n\n' + (current_row['all_text'] or '')).strip()
            content.at[idx - 1, 'all_text'] = merged_text

            # 合并 'all_img_paths'
            if 'all_img_paths' in content.columns:
                prev_img_paths = to_list(prev_row['all_img_paths'])
                current_img_paths = to_list(current_row['all_img_paths'])
                combined_img_paths = prev_img_paths + current_img_paths
                content.at[idx - 1, 'all_img_paths'] = combined_img_paths

            # 合并 'header'，避免重复
            if (prev_row['header'] == current_row['header']) or (str(prev_row['header']).strip() == str(current_row['header']).strip()):
                merged_header = prev_row['header']
            else:
                merged_header = ((prev_row['header'] or '') + ' ' + (current_row['header'] or '')).strip()
            content.at[idx - 1, 'header'] = merged_header

            # 合并 'heading'，避免重复
            if (prev_row['heading'] == current_row['heading']) or (str(prev_row['heading']).strip() == str(current_row['heading']).strip()):
                merged_heading = prev_row['heading']
            else:
                merged_heading = ((prev_row['heading'] or '') + ' ' + (current_row['heading'] or '')).strip()
            content.at[idx - 1, 'heading'] = merged_heading

            # 更新 'all_text_len'
            content.at[idx - 1, 'all_text_len'] = len(merged_text)
            # 标记当前行以便删除
            indices_to_drop.append(idx)
        else:
            # 如果既没有下一行也没有上一行，无法合并
            continue

    # 删除已合并的行
    content = content.drop(index=indices_to_drop).reset_index(drop=True)
    return content


from bs4 import BeautifulSoup
import json

def html_table_to_markdown_str(html_content):
    """
    将 HTML 内容中的表格部分转换为 Markdown 格式字符串，前后添加 <table> 标签区分，其余非表格部分保持原样。

    参数:
        html_content (str): 包含 HTML 内容的字符串。

    返回:
        str: 混合后的字符串，其中表格为 Markdown 格式并加上 <table> 标签，其他部分保持原样。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = []

    for element in soup.contents:
        if element.name == 'table':  # 如果是表格，转换为 Markdown
            headers = [header.get_text(strip=True) for header in element.find_all('tr')[0].find_all('td')]
            # 添加 Markdown 表头
            markdown_table = '|' + '|'.join(headers) + '|\n'
            markdown_table += '|' + '|'.join(['---'] * len(headers)) + '|\n'  # 表头分隔线

            for row in element.find_all('tr')[1:]:
                cells = row.find_all('td')
                row_data = [cell.get_text(strip=True) for cell in cells]
                # 添加行数据到 Markdown 表格
                markdown_table += '|' + '|'.join(row_data) + '|\n'

            # 添加 <table> 标签前后区分表格部分
            result.append(f"<table>\n{markdown_table}</table>\n")
        else:  # 非表格部分直接添加
            result.append(str(element))

    return ''.join(result)


def get_chunk_512_from_content(file_name,file_path=None,input_type='filename',doc_id=None):
    # 新增input_type参数，支持filename、dict两种输入方式，兼容罗氏代码的处理
    if input_type == 'filename':
        if file_path is None:
            file_path = os.path.join(root_path, file_name, file_name+'.json')
        data = json.load(open(file_path, 'r'))
    elif input_type == 'json_str':
        data = json.loads(file_path)  # 此时file_path实际是json字符串，根据autopdf解析的json文件读取而来
    else:
        raise ValueError(f"Invalid input type: {input_type}")

    content_list = data['content_list']

    content = pd.DataFrame([json_chunk for content in content_list for json_chunk in content['content']
                   if not json_chunk.update({'page_id': content['page_id'],
                                             'page_width': content['page_width'],
                                             'page_height': content['page_height'],
                                             'others': content['others']})])

    content['file_name'] = file_name
    content['md5_name'] = doc_id if doc_id else get_md5(file_name)
    content['text'] = content['text'].fillna('')
    content['text'] = content['text'].apply(lambda text: re.sub(r'(\.\s*){2,}', ' ', text))
    content['text'] = content['text'].apply(lambda text: html_table_to_markdown_str(text))
    content['text_len'] = content['text'].apply(lambda text: len(text))

    # 提取 header 中的 text，如果没有则返回 None
    content['header'] = content['others'].apply(
        lambda others: next((clean_text(item['text']) for item in others if item.get('type') == 'header'), None)
    )

    content['header'] = content.apply(lambda row: row['header'].replace(str(row['page_id']+1),'').strip()
                                      if pd.notnull(row['header']) else None, axis=1)

    try:
        page_header = content.drop_duplicates('page_id')

        if page_header.header.value_counts().index.tolist() != []:
            most_common_header = page_header.header.value_counts().index[0]
            header_condition = check_alternating_pattern(page_header['header'].tolist(), most_common_header)

            if page_header.header.value_counts().max() / page_header.shape[0] > 0.3 and header_condition:
                print(page_header.header.value_counts().max(), page_header.shape[0],
                      page_header.header.value_counts().max() / page_header.shape[0])
                print('process_header')
                content['header'] = content['header'].apply(
                    lambda header: None if pd.notnull(header) and header == most_common_header else header
                )
    except Exception as e:
        print(f"[WARN] header processing skipped due to error: {e}")

    # 填充非 `header` 类型行的 `header`，让它们与最近的 `header` 对应
    content['header'] = content['header'].ffill()

    # 生成 heading
    content['heading'] = content.apply(lambda row: clean_text(row['text']) if row['type']=='heading' else None, axis=1)
    # 填充非 `heading` 类型行的 `heading`，让它们与最近的 `heading` 对应
    content['heading'] = content['heading'].ffill()

    ##############################################
    # groupby header and heading, 把all_image_paths和all_text填充好

    # Step 1: Create the group_key
    def create_group_key(row):
        header = row['header']
        heading = row['heading']
        page_id = row['page_id']

        if pd.notna(header) and pd.notna(heading):
            return ('header', header, 'heading', heading)
        elif pd.notna(header):
            return ('header', header, 'page_id', page_id)
        elif pd.notna(heading):
            return ('heading', heading)
        else:
            return ('no_header_no_heading', 'page_id', page_id)

    content['group_key'] = content.apply(create_group_key, axis=1)

    # Step 2: Aggregate img_path
    if 'img_path' in content.columns:
        content['img_path'] = content['img_path'].str.split('/').str[-1]

        img_path_series = (
            content.dropna(subset=['img_path'])
                .groupby('group_key')['img_path']
                .apply(list)
        )

        content['all_img_paths'] = content['group_key'].map(img_path_series)

    # Step 3: Aggregate text
    all_text_series = content.groupby('group_key')['text'].apply(
        lambda x: '\n\n'.join(str(item) for item in x if pd.notna(item) and item != '')
    )

    content['all_text'] = content['group_key'].map(all_text_series)

    content['all_text_len'] = content['all_text'].apply(lambda x: len(x) if pd.notna(x) else 0)

    # Step 4: Aggregate start_page and end_page
    page_agg = content.groupby('group_key').agg(
        start_page=('page_id', 'min'),
        end_page=('page_id', 'max')
    )

    # Map the results back to the original DataFrame
    content['start_page'] = content['group_key'].map(page_agg['start_page'])
    content['end_page'] = content['group_key'].map(page_agg['end_page'])

    ##############################################
    # 如果all_text小于40字，就优先合并给下面header相同的all_text中，如果header不相同就合并到上面的all_text中，all_image_paths也是一样
    chunk_512_table = content.drop_duplicates(['group_key','all_text']).reset_index(drop=True)
    chunk_512_table = merge_small_all_text_into_next(chunk_512_table)

    ##############################################
    # groupby header and heading, 把page_start和page_end，这个是后面为了多模态直接看多少页的pdf用

    ##############################################

    chunk_512_table['id_'] = chunk_512_table.apply(lambda row: f"doc_{row['md5_name']}_chunk512_{row.name}", axis=1)

    return split_large_chunks(chunk_512_table)


def split_large_chunks(chunk_512: pd.DataFrame, max_chunk_size: int = 1024, threshold: int = 6000) -> pd.DataFrame:
    """
    对 chunk_512 中 all_text_len 大于阈值的行进行切分处理，每段最大不超过 max_chunk_size。

    参数：
        chunk_512: 原始 chunk 表
        max_chunk_size: 每段允许的最大字符数（默认 1024）
        threshold: 超过该长度才触发切分（默认 6000）

    返回：
        处理后的 DataFrame，所有段的 all_text_len 均小于等于 max_chunk_size
    """

    # Step 1: 分成两部分
    chunk_to_split = chunk_512[chunk_512['all_text_len'] > threshold].copy()
    chunk_rest = chunk_512[chunk_512['all_text_len'] <= threshold].copy()

    # Step 2: 切分 all_text 为多个段
    chunk_to_split['split_text'] = chunk_to_split['all_text'].apply(lambda x: RecursiveCharacterTextSplitter2._split_text_by_size(x, max_size=max_chunk_size))

    # Step 3: explode
    chunk_exploded = chunk_to_split.explode('split_text').reset_index(drop=True)

    # Step 4: 更新文本和长度
    chunk_exploded['all_text'] = chunk_exploded['split_text']
    chunk_exploded['all_text_len'] = chunk_exploded['all_text'].apply(len)
    chunk_exploded = chunk_exploded.drop(columns=['split_text'])

    # Step 5: 合并两部分
    chunk_512_updated = pd.concat([chunk_rest, chunk_exploded], ignore_index=True)

    # Step 6: 更新 id_
    chunk_512_updated['id_'] = chunk_512_updated.apply(lambda row: f"doc_{row['md5_name']}_chunk512_{row.name}", axis=1)

    return chunk_512_updated
