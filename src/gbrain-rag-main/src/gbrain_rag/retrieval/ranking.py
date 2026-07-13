import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from gbrain_rag.core.text import text_tokens


def cosine(a, b) -> float:
    # Embeddings are normalized by both backends; this helper is intentionally
    # tiny for numpy arrays and list-like vectors.
    return float(a @ b)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[tuple[str, float]]],
    *,
    weights: Iterable[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    weight_list = list(weights or [])
    for list_idx, ranked in enumerate(ranked_lists):
        weight = weight_list[list_idx] if list_idx < len(weight_list) else 1.0
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] += weight / (k + rank)
    return dict(scores)


def bm25_scores(query: str, docs: dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    query_terms = list(dict.fromkeys(text_tokens(query)))
    if not query_terms or not docs:
        return {}

    tokenized = {doc_id: text_tokens(text) for doc_id, text in docs.items()}
    lengths = {doc_id: len(tokens) for doc_id, tokens in tokenized.items()}
    non_empty_lengths = [length for length in lengths.values() if length > 0]
    if not non_empty_lengths:
        return {}

    avg_len = sum(non_empty_lengths) / len(non_empty_lengths)
    total_docs = len(tokenized)
    dfs: dict[str, int] = {}
    for term in query_terms:
        dfs[term] = sum(1 for tokens in tokenized.values() if term in set(tokens))

    raw_scores: dict[str, float] = {}
    for doc_id, tokens in tokenized.items():
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = dfs.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * len(tokens) / max(avg_len, 1e-9))
            score += idf * (tf * (k1 + 1) / denom)
        if score > 0:
            raw_scores[doc_id] = score

    if not raw_scores:
        return {}
    max_score = max(raw_scores.values()) or 1.0
    return {doc_id: score / max_score for doc_id, score in raw_scores.items()}
