from pathlib import Path
from typing import Any

from gbrain_rag.core.config import get_settings
from gbrain_rag.core.text import split_text, stable_id
from gbrain_rag.core.types import Chunk


def make_chunk(
    *,
    doc_id: str,
    doc_name: str,
    source_type: str,
    text: str,
    index_text: str | None = None,
    block_type: str = "text",
    page_label: int | str | None = None,
    title: str | None = None,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    ordinal: int = 0,
) -> Chunk:
    chunk_id = stable_id(doc_id, source_type, block_type, page_label, ordinal, text[:200])
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        source_type=source_type,
        title=title,
        text=text.strip(),
        index_text=(index_text or text).strip(),
        block_type=block_type,
        page_label=page_label,
        source_path=source_path,
        metadata=metadata or {},
    )


def text_to_chunks(
    *,
    doc_id: str,
    doc_name: str,
    source_type: str,
    text: str,
    title: str | None = None,
    source_path: str | None = None,
    page_label: int | str | None = None,
    block_type: str = "text",
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    parts = split_text(
        text,
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        overlap=chunk_overlap or settings.CHUNK_OVERLAP,
    )
    return [
        make_chunk(
            doc_id=doc_id,
            doc_name=doc_name,
            source_type=source_type,
            text=part,
            index_text=text,
            block_type=block_type,
            page_label=page_label,
            title=title,
            source_path=source_path,
            metadata={**(metadata or {}), "part": idx},
            ordinal=idx,
        )
        for idx, part in enumerate(parts)
    ]


def doc_id_for_path(path: Path, base_dir: Path | None = None) -> str:
    try:
        rel = path.relative_to(base_dir) if base_dir else path
    except ValueError:
        rel = path
    stat = path.stat()
    return stable_id(rel.as_posix(), stat.st_size, int(stat.st_mtime))
