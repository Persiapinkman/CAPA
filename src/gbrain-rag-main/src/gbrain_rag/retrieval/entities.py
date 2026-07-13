import re
from collections import Counter

from gbrain_rag.core.text import normalize_text


_MODEL_RE = re.compile(
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9_./+-]{3,}\.(?:model|onnx|pt|pth|safetensors))"
)
_VERSION_RE = re.compile(r"\b[vV]?\d+(?:\.\d+){1,3}(?:[-_][A-Za-z0-9]+)?\b")
_OID_RE = re.compile(r"\b[a-fA-F0-9]{32,64}\b")
_DID_RID_RE = re.compile(r"\b(?:did|rid|deployment_id|release_id)[:=\s_-]*(\d{4,})\b", re.I)
_PLATFORM_RE = re.compile(
    r"\b(?:cuda\d+(?:\.\d+)?-trt\d+(?:\.\d+)?-[a-z0-9]+-[A-Za-z0-9]+|acl-[A-Za-z0-9]+-[a-z0-9]+|cpu-[A-Za-z0-9]+-[a-z0-9]+)\b",
    re.I,
)


def extract_entities(text: str | None, *, max_entities: int = 36) -> list[tuple[str, str]]:
    """Extract deterministic entities for graph-assisted retrieval.

    This intentionally avoids LLM calls; it mirrors gbrain's "zero LLM calls"
    auto-linking idea for the concrete model-release domain.
    """

    raw = normalize_text(text)
    if not raw:
        return []

    entities: list[tuple[str, str]] = []
    for match in _MODEL_RE.finditer(raw):
        entities.append(("model", match.group("model")))
    for match in _VERSION_RE.finditer(raw):
        entities.append(("version", match.group(0)))
    for match in _OID_RE.finditer(raw):
        entities.append(("oid", match.group(0)))
    for match in _DID_RID_RE.finditer(raw):
        entities.append(("deployment", match.group(1)))
    for match in _PLATFORM_RE.finditer(raw):
        entities.append(("platform", match.group(0)))

    # Domain phrases that often answer short Chinese queries better than raw
    # vector search alone.
    phrase_patterns = [
        ("field", r"(输入|输出|阈值|推荐阈值|追加数据|优化点|标签|负责人|更新时间|推荐配置)"),
        ("scene", r"([\u4e00-\u9fff]{2,12}(?:检测|识别|分类|属性|特征|质量|模型|项目|场景))"),
    ]
    for kind, pattern in phrase_patterns:
        for match in re.finditer(pattern, raw):
            entities.append((kind, match.group(1)))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, name in entities:
        clean = normalize_text(name)
        if not clean:
            continue
        key = (kind, clean.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((kind, clean))
        if len(deduped) >= max_entities:
            break
    return deduped


def entity_names(text: str | None) -> list[str]:
    return [name for _, name in extract_entities(text)]


def entity_overlap_score(query_entities: list[str], chunk_entities: list[str]) -> float:
    if not query_entities or not chunk_entities:
        return 0.0
    q = Counter(name.lower() for name in query_entities)
    c = Counter(name.lower() for name in chunk_entities)
    overlap = sum(min(q[name], c[name]) for name in q.keys() & c.keys())
    return overlap / max(len(q), 1)
