import hashlib
import re
from collections.abc import Iterable


_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def stable_id(*parts: object, length: int = 24) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:length]


def text_tokens(text: str | None) -> list[str]:
    """Tokenize Chinese/internal model text without external segmenters.

    The mix of ASCII technical tokens, Chinese single characters, and Chinese
    bigrams works well for model names, versions, OIDs, and short CJK queries.
    """

    raw = str(text or "").lower()
    tokens = [match.group(0) for match in _ASCII_WORD_RE.finditer(raw)]
    cjk_chars = [match.group(0) for match in _CJK_RE.finditer(raw)]
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return [token for token in tokens if token.strip()]


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        cut = -1
        for marker in ("\n\n", "\n", "。", "；", ". ", ", ", "，"):
            pos = window.rfind(marker)
            if pos >= max(120, int(chunk_size * 0.45)):
                cut = start + pos + len(marker)
                break
        if cut <= start:
            cut = end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut >= len(text):
            break
        start = max(cut - overlap, start + 1)
    return chunks


def compact_json_text(items: Iterable[tuple[str, object]], max_value_len: int = 2400) -> str:
    lines = []
    for key, value in items:
        if value in (None, "", [], {}):
            continue
        value_text = str(value)
        if len(value_text) > max_value_len:
            value_text = value_text[: max_value_len - 3] + "..."
        lines.append(f"{key}: {value_text}")
    return "\n".join(lines)
