import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def merge_small_texts(df: pd.DataFrame, min_len: int = 32) -> pd.DataFrame:
    df = df.copy()
    # 1) 连续短正文（≤32，且非标题）批量合并
    cond = (df['type'].eq('text')) & (df['text_len']<=min_len) & (df['text_level'].isna())
    sub = df[cond].copy().reset_index()                     # 保留原行号在 index 列
    sub['grp'] = (sub['index'] != sub['index'].shift()+1).cumsum()  # 连续性分组（用 Series.shift）

    merged = sub.groupby('grp').agg({
        'type': 'first',
        'text': lambda x: '\n'.join(x),
        'page_idx': 'first',
        'text_len': 'sum'
    }).reset_index(drop=True)

    # 每组用首行原索引作为回填位置
    merged_index = sub.groupby('grp')['index'].first().values
    merged = merged.set_index(pd.Index(merged_index, name='index'))

    out = df.drop(index=sub['index']).copy()
    out = pd.concat([out, merged]).sort_index()

    # 2) 孤立短行（<16 且非标题）向上合并
    out = out.reset_index()  # 恢复原序
    to_drop = []
    for i in range(1, len(out)):
        if (out.at[i, 'type'] == 'text') and pd.isna(out.at[i, 'text_level']) and (out.at[i, 'text_len'] < min_len/2):
            # 合并到上一行
            out.at[i-1, 'text'] = f"{out.at[i-1, 'text']}\n{out.at[i, 'text']}"
            out.at[i-1, 'text_len'] = (out.at[i-1, 'text_len'] or 0) + (out.at[i, 'text_len'] or 0)
            to_drop.append(i)
    if to_drop:
        out = out.drop(index=to_drop)

    return out.drop(columns=['index']).reset_index(drop=True)


def split_big_text(df: pd.DataFrame, max_len: int = 6000) -> pd.DataFrame:
    def _split(s: str) -> list:
        s = "" if pd.isna(s) else str(s)
        n = len(s)
        if n <= max_len:
            return [s]
        out, i = [], 0
        while i < n:
            # 第一次尝试
            end = min(i + max_len, n)
            window = s[i:end]
            cut = -1
            for d in ("\n\n", "\n", "。", ". ", "；"):    # 优先级：双换行 > 单换行 > 中文句号 > 中文分号
                pos = window.rfind(d)
                if pos != -1:
                    cut = i + pos + len(d)
                    break

            # 如果没找到，就向后多看128字符
            if cut == -1 and end < n:
                extra_end = min(end + 128, n)
                window2 = s[i:extra_end]
                for d in ("\n\n", "\n", "。", ". ", "；", "，", ", "):    # 优先级：双换行 > 单换行 > 中文句号 > 中文分号 > 中文逗号
                    pos = window2.rfind(d)
                    if pos != -1:
                        cut = i + pos + len(d)
                        break

            # 如果还是没找到就硬切
            if cut == -1 or cut <= i:
                cut = end

            out.append(s[i:cut].strip())
            i = cut
        return [t for t in out if t]

    rows = []
    for _, r in df.iterrows():
        if r.get("type") == "text":
            parts = _split(r.get("text"))
            if len(parts) > 1:                   # 只有超长才会被拆
                for p in parts:
                    nr = r.copy()
                    nr["text"] = p
                    nr["text_len"] = len(p)
                    rows.append(nr)
                continue
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def merge_cross_page_short(df: pd.DataFrame, max_len: int = 128, end_char=("。",".")) -> pd.DataFrame:
    out, drop = df.copy(), []
    idx = out.index.to_list()
    for k in range(len(idx)-1):
        i, j = idx[k], idx[k+1]
        a, b = out.loc[i], out.loc[j]
        if (
            a.get("type")=="text" and b.get("type")=="text"           # 仅文本
            and pd.isna(a.get("text_level")) and pd.isna(b.get("text_level"))  # 非标题
            and pd.notna(a.get("page_idx")) and pd.notna(b.get("page_idx"))
            and a["page_idx"] + 1 == b["page_idx"]                    # 跨页相邻
            and ((a.get("text_len", 0) < max_len) or (b.get("text_len", 0) < max_len))
            and (not str(a.get("text","")).rstrip().endswith(end_char))
        ):
            out.at[i, "text"] = f"{str(a.get('text',''))}\n{str(b.get('text',''))}"
            out.at[i, "text_len"] = (a.get("text_len", 0) or 0) + (b.get("text_len", 0) or 0)
            drop.append(j)
    if drop:
        out = out.drop(index=drop)
    return out.reset_index(drop=True)


def to_seq(x):
        # 列表/元组/ndarray：逐个转字符串并过滤空/None/nan
        if isinstance(x, (list, tuple, np.ndarray)):
            return [s for s in (str(y).strip() for y in x)
                    if s and s.lower() not in ('nan', 'none')]
        # 标量：能转成 str 就用，转不了就丢
        try:
            s = str(x).strip()
        except Exception:
            return []
        return [s] if s and s.lower() not in ('nan', 'none') else []


def fill_image_text(df: pd.DataFrame) -> pd.DataFrame:
    # 这里只是为了把image的caption和footnote加到text里去
    df = df.copy()
    mask = df['type'].eq('image')
    df.loc[mask, 'text'] = df.loc[mask].apply(
        lambda r: '\n'.join(
            to_seq(r['text']) +
            to_seq(r.get("image_caption", "")) +
            to_seq(r.get("image_footnote", ""))
        ).strip(),
        axis=1
    )
    df.loc[mask, 'text_len'] = df.loc[mask, 'text'].str.len()
    return df


def add_group_text(df: pd.DataFrame) -> pd.DataFrame:
    # 按照标题合并在一起
    df = df.reset_index(drop=True).copy()

    # 每遇到一个 text_level 非空就开新组
    mask = df['type'].ne('table')   # 只处理非表格行
    grp = (df.loc[mask, 'text_level'].notna()).cumsum()
    df.loc[mask, 'grp'] = grp

    group_texts = df[mask].groupby('grp')['text'].apply(lambda s: '\n'.join(s.dropna().astype(str)))
    df.loc[mask, 'group_text'] = df.loc[mask, 'grp'].map(group_texts)
    df.loc[mask, 'group_text_len'] = df.loc[mask, 'group_text'].str.len()

    return df.drop(columns=['grp'], errors='ignore')


def merge_small_groups(df: pd.DataFrame, min_len: int = 128,
                       text_col: str = "group_text", len_col: str = "group_text_len",
                       sep: str = "\n") -> pd.DataFrame:
    # 把group_text小于128字的再合并在一起
    df = df.copy().reset_index(drop=True)
    mask = df['type'].ne('table')   # 只处理非表格行
    idxs = df.index[mask]

    i = 0
    while i < len(idxs):
        cur = idxs[i]
        if df.at[cur, len_col] < min_len:
            j = i
            while j + 1 < len(idxs) and df.at[idxs[j + 1], len_col] < min_len:
                j += 1
            lines = []
            for k in idxs[i:j + 1]:
                for line in str(df.at[k, text_col]).split(sep):
                    if line not in lines:
                        lines.append(line)
            merged_text = sep.join(lines)
            merged_len = len(merged_text)
            df.loc[idxs[i:j + 1], text_col] = merged_text
            df.loc[idxs[i:j + 1], len_col] = merged_len
            i = j + 1
        else:
            i += 1

    return df


def add_heading(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    # 遇到 text_level 非空时，用当前 text 作为新的 heading
    df["heading"] = df["text"].where(df["text_level"].notna())
    df["heading"] = df["heading"].ffill()
    return df


def add_header(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    n, i, cur = len(out), 0, None
    header = []
    tl = out["text_level"]

    while i < n:
        if pd.notna(tl.iat[i]):                      # 进入一段连续的“标题”(text_level非空)
            j = i
            while j < n and pd.notna(tl.iat[j]):
                j += 1
            k = j - i
            if k == 2:                               # 连续2个：header=第一个标题
                cur = str(out.at[i, "text"])
            elif k >= 3:                             # 连续≥3个：header=所有标题用\n拼接
                cur = "\n".join(out.loc[i:j-1, "text"].dropna().astype(str))
            # k==1：不更新，沿用之前的header
            header.extend([cur] * k)                 # 标题行本身也赋当前header
            i = j
        else:
            header.append(cur)                       # 正文/其他行继承最近header
            i += 1

    out["header"] = header
    return out


def merge_doc_text(df: pd.DataFrame, max_sum: int = 2000,
                   src_col: str = "text", dst_col: str = "group_text",
                   dst_len_col: str = "group_text_len", sep: str = "\n\n") -> pd.DataFrame:
    # 整个文档字数少 (2000字以内) 的话可以全部合并在一起
    out = df.copy()
    if out["text_len"].sum() <= max_sum:
        items = [s for s in out[src_col].dropna().astype(str) if s.strip() and s.lower() not in ("nan","none")]
        merged = sep.join(dict.fromkeys(items))  # 去重且保序
        out[dst_col] = merged
        out[dst_len_col] = len(merged)
    return out


def split_big_groups(df: pd.DataFrame, max_len: int = 6000,
                     text_col: str = "text", len_col: str = "text_len",
                     gcol: str = "group_text", glen: str = "group_text_len",
                     sep: str = "\n") -> pd.DataFrame:
    # group_text超过6000字需要处理下，控制最大给模型的上下文长度为6000
    """
    Pipeline 说明：
    1) 找到所有分组头：type=='text' 且 text_level 非空。
    2) 若该头行的 group_text_len <= max_len：跳过。
    3) 否则取该头到下一个头(不含)为一组，忽略其中的 table 行，仅用 text 行按 text_len 从前往后累加，
       每段累加不超过 max_len 视为一个子组（chunk）。
    4) 对每个子组：把该子组里所有“text行”的 group_text 改为子组合并文本（按 sep 拼接），
       group_text_len 改为其长度；表格行保持不变。
    """
    out = df.copy()
    is_header = out["type"].eq("text") & out["text_level"].notna()
    hdr_idxs = out.index[is_header].tolist()
    hdr_idxs.append(len(out))  # 方便确定最后一组的边界

    for hpos in range(len(hdr_idxs) - 1):
        start, end = hdr_idxs[hpos], hdr_idxs[hpos + 1]
        # 只处理超阈值的组
        if pd.isna(out.at[start, glen]) or out.at[start, glen] <= max_len:
            continue

        # 该组内的“文本行”索引（忽略表格等）
        chunk_rows = [i for i in range(start, end) if out.at[i, "type"] == "text"]

        # 按 text_len 累加切块
        chunks, cur, cur_sum = [], [], 0
        for i in chunk_rows:
            l = int(out.at[i, len_col] or 0)
            if cur and cur_sum + l > max_len:
                chunks.append(cur); cur, cur_sum = [i], l
            else:
                cur.append(i); cur_sum += l
        if cur: chunks.append(cur)

        # 组装 & 回填本块的 group_text / group_text_len（仅作用于 text 行）
        for part in chunks:
            merged = sep.join(out.loc[part, text_col].fillna("").astype(str)).strip()
            out.loc[part, gcol] = merged
            out.loc[part, glen] = len(merged)

    return out.reset_index(drop=True)


def clean_table_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:  # 过滤空行
                rows.append("\t".join(cells))
        return "\n".join(rows).strip()
    except Exception:
        # 如果不是 HTML 或解析失败，直接返回原始文本
        return str(text) if pd.notna(text) else ""


def fill_table_text(df: pd.DataFrame, max_len: int =2048, clean_table: bool = True) -> pd.DataFrame:
    """
    填充 table 类型的 text / group_text：
    - 如果 table_body > max_len，则先清洗 HTML 表格
    - 拼接 header + heading + text + table_caption + table_footnote + table_body
    """
    df = df.copy()
    mask = df["type"].eq("table")

    def build_text(r):
        body = r["table_body"]
        # 如果 table_body 超过 max_len，就先清洗 HTML
        if clean_table:
            if isinstance(body, str) and len(body) > max_len:
                body = clean_table_html(body)
        return "\n".join(
            to_seq(r["header"]) +
            to_seq(r["heading"]) +
            to_seq(r["text"]) +
            to_seq(r.get("table_caption", "")) +
            to_seq(r.get("table_footnote", "")) +
            to_seq(body)
        ).strip()

    # 更新 text
    df.loc[mask, "text"] = df.loc[mask].apply(build_text, axis=1)
    df.loc[mask, "text_len"] = df.loc[mask, "text"].str.len()

    # group_text 和 text 保持一致
    df.loc[mask, "group_text"] = df.loc[mask, "text"]
    df.loc[mask, "group_text_len"] = df.loc[mask, "text_len"]

    return df


def llamaindex_format_chunks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[(df['text'].notna()) & (df['text']!='') & (df['text_level'].isna())].reset_index()

    # 1. 重命名 page_idx -> page_label
    df = df.rename(columns={"page_idx": "page_label", "group_text": "index_text"})

    # 2. 构建 embedding_text
    df["embedding_text"] = df.apply(
        lambda row: f"doc_name: {row['doc_name']}\n\nheader: {row['header']}\n\nheading: {row['heading']}\n\n{row['text']}",
        axis=1
    )

    # 5. 根据 doc_id + index_text 分组，为每个唯一的 index_text 生成唯一 group_id
    df["group_id"] = df.groupby(["doc_id", "index_text"], sort=False).ngroup()

    # 6. 生成 index_id：基于唯一的 group_text
    df["index_id"] = df.apply(
        lambda r: f"doc_{r['doc_id']}_chunk512_{r['group_id']}",
        axis=1
    )

    # 7. 在每个 index_id 内编号 chunk128
    df["chunk_id"] = df.groupby("index_id", sort=False).cumcount()

    # 8. 生成 id：每条记录对应一个 chunk128
    df["id"] = df.apply(
        lambda r: f"{r['index_id']}_chunk128_{r['chunk_id']}",
        axis=1
    )

    # 7. 构建 metadata
    df["metadata"] = df.apply(
        lambda r: {
            "header": r["header"],
            "heading": r["heading"],
            "page_label": r["page_label"]
        },
        axis=1
    )

    # 8. 删除中间临时列
    use_cols = ["embedding_text", "id", "text", "index_id", "index_text", "metadata", "doc_id", "doc_name"]
    df = df[use_cols]

    return df


def mineru_chunking(df: pd.DataFrame, doc_name: str, doc_id: str) -> pd.DataFrame:
    '''
    输入是 pd.DataFrame(content_list_json)
    # https://opendatalab.github.io/MinerU/zh/reference/output_files/#content_listjson
    输出是处理好后的pd.DataFrame
    '''
    chunks = df.copy()

    chunks['text_len'] = chunks['text'].str.len()

    # 只处理文本（包含image的caption）

    chunks = merge_small_texts(chunks, min_len=32)

    chunks = split_big_text(chunks, max_len=6000)

    chunks = merge_cross_page_short(chunks, max_len=128)

    chunks = fill_image_text(chunks)

    chunks = add_group_text(chunks)

    chunks = merge_small_groups(chunks, min_len=128)

    chunks = add_heading(chunks)

    chunks = add_header(chunks)

    chunks = merge_doc_text(chunks, max_sum=2000)

    chunks = split_big_groups(chunks, max_len=6000)

    # 只处理table

    chunks = fill_table_text(chunks, max_len=2048)

    # 切分小块，准备开始向量化的文本（还是不管table）

    chunks = split_big_text(chunks, max_len=128)

    chunks['doc_name'] = doc_name

    chunks['doc_id'] = doc_id

    return llamaindex_format_chunks(chunks)


def autopdf_to_mineru(content_list: list) -> list:
    df = pd.DataFrame([json_chunk for content in content_list for json_chunk in content['content']
                            if not json_chunk.update({'page_id': content['page_id'],
                                                      'others': content['others']})])

    df = df.rename(columns={"page_id": "page_idx", "heading_level": "text_level",
                            "img_caption": "image_caption", "img_footnote": "image_footnote"})

    df.loc[df["type"] == "table", "table_body"] = df.loc[df["type"] == "table", "text"]

    df.loc[df["type"] == "heading", "type"] = "text"

    return df.to_dict(orient="records")