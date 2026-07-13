from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from gbrain_rag.core.config import get_settings
from gbrain_rag.core.text import normalize_text
from gbrain_rag.core.types import Chunk
from gbrain_rag.ingest.chunker import make_chunk, text_to_chunks
from gbrain_rag.retrieval.aspects import classify_chunk_aspects, chunk_section_type

logger = logging.getLogger(__name__)

_MODEL_ARTIFACT_RE = re.compile(r"\.(?:model|onnx|pt|pth|safetensors)\b", re.I)
_HEX_OID_RE = re.compile(r"^[0-9a-f\s]{24,}$", re.I)
_PLATFORM_RE = re.compile(r"(?:cuda|trt|acl|ascend|fp16|fp32|int8|t4|p4|l4|cpu|nart)", re.I)
_FEATURE_DIM_RE = re.compile(r"^\d{2,5}$")


def _compact_cell(value: Any) -> str:
    return normalize_text(value)


def _compact_label(value: Any) -> str:
    return re.sub(r"\s+", "", _compact_cell(value)).lower()


def _clean_record_value(value: Any) -> str:
    text = _compact_cell(value)
    if _MODEL_ARTIFACT_RE.search(text) or _HEX_OID_RE.match(text) or _PLATFORM_RE.search(text):
        text = re.sub(r"(?<=[A-Za-z0-9_.:-])\s+(?=[A-Za-z0-9_.:-])", "", text)
    return text


def _clean_cell(value: Any) -> str:
    return normalize_text(value).replace("|", "\\|")


def table_to_markdown(table: list[list[Any]]) -> str:
    rows = [[_clean_cell(cell) for cell in row] for row in table if row and any(_clean_cell(c) for c in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _find_header_columns(rows: list[list[str]]) -> dict[str, int]:
    aliases = {
        "model_name": ("模型名称", "modelname", "模型"),
        "component_type": ("组件类型", "component"),
        "oid": ("oid",),
        "platform": ("平台", "platform"),
        "feature_dim": ("特征维度", "输出特征维度", "featuredim", "dimension", "维度"),
    }
    columns: dict[str, int] = {}
    for row in rows[:10]:
        for idx, cell in enumerate(row):
            label = _compact_label(cell)
            if not label:
                continue
            for key, candidates in aliases.items():
                if key in columns:
                    continue
                if any(_compact_label(candidate) in label for candidate in candidates):
                    columns[key] = idx
    return columns


def _value_near(
    row: list[str],
    col: int | None,
    *,
    radius: int = 1,
    predicate: Any | None = None,
) -> str:
    if col is None:
        return ""
    start = max(0, col - radius)
    end = min(len(row), col + radius + 1)
    candidates: list[tuple[int, int, str]] = []
    for idx in range(start, end):
        value = _clean_record_value(row[idx])
        if not value:
            continue
        if predicate is not None and not predicate(value):
            continue
        candidates.append((abs(idx - col), idx, value))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][2]


def _is_model_artifact(value: str) -> bool:
    text = value.lower()
    return bool(_MODEL_ARTIFACT_RE.search(text)) or "_nart_" in text


def _is_component(value: str) -> bool:
    return _compact_label(value).replace("textencoder", "text_encoder") in {"senu", "text_encoder", "text_encode"}


def _normalize_component(value: str) -> str:
    label = _compact_label(value)
    if label in {"textencoder", "text_encode", "text_encoder"} or label.startswith("text_encode"):
        return "text_encoder"
    return value


def _is_oid(value: str) -> bool:
    return bool(_HEX_OID_RE.match(value)) and not _FEATURE_DIM_RE.match(value)


def _is_platform(value: str) -> bool:
    return bool(_PLATFORM_RE.search(value)) and not _is_model_artifact(value) and not _is_oid(value)


def _is_feature_dim(value: str) -> bool:
    return bool(_FEATURE_DIM_RE.match(value))


def _looks_like_family_label(value: str) -> bool:
    if not value or _is_model_artifact(value):
        return False
    label = _compact_label(value)
    if label in {"模型名称", "组件类型", "oid", "平台", "特征维度"}:
        return False
    return len(value) <= 40 and not _is_oid(value) and not _is_platform(value)


def table_to_structured_text(table: list[list[Any]]) -> str:
    """Flatten sparse PDF tables into searchable row records.

    pdfplumber often extracts merged-cell release tables as many mostly-empty
    columns. The Markdown table keeps the visual shape, while these records keep
    row-level bindings such as model name -> platform -> feature dimension.
    """

    rows = [[_compact_cell(cell) for cell in row] for row in table if row and any(_compact_cell(c) for c in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    columns = _find_header_columns(rows)
    if not {"model_name", "component_type", "platform", "feature_dim"} & set(columns):
        return ""

    current = {
        "model_family": "",
        "component_type": "",
        "platform": "",
        "feature_dim": "",
    }
    records: list[str] = []
    for row in rows:
        labels = {_compact_label(cell) for cell in row if _compact_label(cell)}
        if labels & {"模型名称", "组件类型", "oid", "平台", "特征维度"}:
            continue

        model = _value_near(row, columns.get("model_name"), radius=1, predicate=_is_model_artifact)
        raw_model_slot = _value_near(row, columns.get("model_name"), radius=1)
        if raw_model_slot and not model and _looks_like_family_label(raw_model_slot):
            current["model_family"] = raw_model_slot

        component = _value_near(row, columns.get("component_type"), radius=1, predicate=_is_component)
        platform = _value_near(row, columns.get("platform"), radius=2, predicate=_is_platform)
        feature_dim = _value_near(row, columns.get("feature_dim"), radius=1, predicate=_is_feature_dim)
        if component:
            current["component_type"] = _normalize_component(component)
        if platform:
            current["platform"] = platform
        if feature_dim:
            current["feature_dim"] = feature_dim

        if not model:
            continue

        oid = _value_near(row, columns.get("oid"), radius=2, predicate=_is_oid)
        fields = [
            ("模型族", current["model_family"]),
            ("模型名称", model),
            ("组件类型", current["component_type"]),
            ("OID", oid),
            ("平台", current["platform"]),
            ("特征维度", current["feature_dim"]),
        ]
        record = "\n".join(f"{key}: {value}" for key, value in fields if value)
        if record:
            records.append(record)

    if not records:
        return ""
    return "表格结构化行:\n" + "\n\n".join(records)


def table_to_search_text(table: list[list[Any]]) -> str:
    parts = [table_to_structured_text(table), table_to_markdown(table)]
    return "\n\n".join(part for part in parts if part)


def _attach_aspect_metadata(chunk: Chunk) -> Chunk:
    aspects = classify_chunk_aspects(chunk)
    chunk.metadata = {
        **(chunk.metadata or {}),
        "aspects": list(aspects),
        "section_type": chunk_section_type(chunk, aspects),
    }
    return chunk


def _extract_text_with_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF parsing requires pdfplumber, PyMuPDF, or pypdf.") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_idx, text))
    return pages


def _extract_text_with_fitz(pdf_path: Path) -> dict[int, str]:
    try:
        import fitz
    except Exception:
        return {}

    page_text: dict[int, str] = {}
    doc = fitz.open(str(pdf_path))
    try:
        for idx, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks") or []
            blocks = sorted(blocks, key=lambda item: (round(item[1] / 8), item[0]))
            lines = [str(block[4]).strip() for block in blocks if len(block) >= 5 and str(block[4]).strip()]
            if lines:
                page_text[idx] = "\n".join(lines)
    finally:
        doc.close()
    return page_text


def extract_pdf_chunks(
    pdf_path: Path,
    *,
    doc_id: str,
    doc_name: str,
    source_type: str = "document",
    source_path: str | None = None,
) -> list[Chunk]:
    """Extract PDF chunks with first-class table preservation.

    pdfplumber tables become Markdown table chunks. Text comes from pdfplumber
    layout mode with a PyMuPDF block fallback, then pypdf as last resort.
    """

    settings = get_settings()
    source_path = source_path or str(pdf_path)
    chunks: list[Chunk] = []
    fitz_text = _extract_text_with_fitz(pdf_path)

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                table_texts: list[str] = []
                try:
                    tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "intersection_tolerance": 5,
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                        }
                    )
                except Exception:
                    tables = []

                for table_idx, table in enumerate(tables or []):
                    if (
                        not table
                        or len(table) < settings.PDF_TABLE_MIN_ROWS
                        or max((len(row or []) for row in table), default=0) < settings.PDF_TABLE_MIN_COLS
                    ):
                        continue
                    md = table_to_search_text(table)
                    if not md:
                        continue
                    table_text = f"PDF表格: {doc_name} 第{page_idx}页 表{table_idx + 1}\n\n{md}"
                    table_texts.append(table_text)
                    chunks.append(
                        _attach_aspect_metadata(
                            make_chunk(
                                doc_id=doc_id,
                                doc_name=doc_name,
                                source_type=source_type,
                                text=table_text,
                                index_text=table_text,
                                block_type="table",
                                page_label=page_idx,
                                title=f"{doc_name} p{page_idx} table {table_idx + 1}",
                                source_path=source_path,
                                metadata={"table_index": table_idx + 1},
                                ordinal=10_000 + page_idx * 100 + table_idx,
                            )
                        )
                    )

                text = ""
                try:
                    text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
                except Exception:
                    text = ""
                text = text.strip() or fitz_text.get(page_idx, "")
                if text:
                    index_text = "\n\n".join([text, *table_texts]).strip()
                    page_chunks = text_to_chunks(
                        doc_id=doc_id,
                        doc_name=doc_name,
                        source_type=source_type,
                        text=text,
                        title=f"{doc_name} p{page_idx}",
                        source_path=source_path,
                        page_label=page_idx,
                        block_type="text",
                        metadata={"table_count_on_page": len(table_texts)},
                    )
                    for chunk in page_chunks:
                        chunk.index_text = index_text
                        _attach_aspect_metadata(chunk)
                    chunks.extend(page_chunks)
    except Exception as exc:
        logger.warning("pdfplumber failed for %s; falling back to pypdf: %s", pdf_path, exc)
        for page_idx, text in _extract_text_with_pypdf(pdf_path):
            fallback_chunks = text_to_chunks(
                doc_id=doc_id,
                doc_name=doc_name,
                source_type=source_type,
                text=text,
                title=f"{doc_name} p{page_idx}",
                source_path=source_path,
                page_label=page_idx,
                block_type="text",
            )
            chunks.extend(_attach_aspect_metadata(chunk) for chunk in fallback_chunks)

    if not chunks and fitz_text:
        for page_idx, text in fitz_text.items():
            fallback_chunks = text_to_chunks(
                doc_id=doc_id,
                doc_name=doc_name,
                source_type=source_type,
                text=text,
                title=f"{doc_name} p{page_idx}",
                source_path=source_path,
                page_label=page_idx,
                block_type="text",
            )
            chunks.extend(_attach_aspect_metadata(chunk) for chunk in fallback_chunks)

    return chunks
