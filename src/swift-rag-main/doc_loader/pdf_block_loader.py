import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TABLE_LINE_PATTERN = re.compile(r"\S+\s{2,}\S+")


def _get_pdf_reader_cls():
    try:
        from pypdf import PdfReader as reader_cls  # type: ignore
        return reader_cls
    except Exception:
        try:
            from PyPDF2 import PdfReader as reader_cls  # type: ignore
            return reader_cls
        except Exception as exc:
            raise RuntimeError(
                "无法解析 PDF：未安装 pypdf 或 PyPDF2，请安装其中一个依赖。"
            ) from exc


def _is_table_like_line(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    if "|" in line or "\t" in line:
        return True
    return bool(_TABLE_LINE_PATTERN.search(line))


def _split_text_and_table_blocks(page_text: str) -> Tuple[List[str], List[str]]:
    lines = (page_text or "").splitlines()
    groups: List[Tuple[str, List[str]]] = []
    current_kind: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        kind = "table" if _is_table_like_line(line) else "text"
        if current_kind is None:
            current_kind = kind
            current_lines = [line]
            continue

        if kind == current_kind:
            current_lines.append(line)
            continue

        groups.append((current_kind, current_lines))
        current_kind = kind
        current_lines = [line]

    if current_kind is not None and current_lines:
        groups.append((current_kind, current_lines))

    text_blocks: List[str] = []
    table_blocks: List[str] = []
    for kind, group_lines in groups:
        merged = "\n".join(line for line in group_lines if line.strip()).strip()
        if not merged:
            continue
        if kind == "table":
            table_blocks.append(merged)
        else:
            text_blocks.append(merged)

    return text_blocks, table_blocks


def _build_index_text(text_blocks: List[str], table_blocks: List[str]) -> str:
    all_parts = [*text_blocks, *table_blocks]
    if not all_parts:
        return ""
    return "\n\n".join(all_parts).strip()


def _safe_stem(file_path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", file_path.stem)


def extract_pdf_blocks(
    file_path: Path,
    doc_id: str,
    image_output_root: Path,
) -> List[Dict[str, Any]]:
    """Extract page-level text/table blocks and image blocks from a PDF.

    The output is a JSON-serializable list and can be fed into `input_type=pdf_blocks`.
    """
    reader_cls = _get_pdf_reader_cls()
    reader = reader_cls(str(file_path))
    blocks: List[Dict[str, Any]] = []
    image_output_root.mkdir(parents=True, exist_ok=True)

    file_name = file_path.name
    safe_stem = _safe_stem(file_path)

    for page_idx, page in enumerate(reader.pages):
        page_label = page_idx + 1
        page_text = (page.extract_text() or "").strip()
        text_blocks, table_blocks = _split_text_and_table_blocks(page_text)
        index_text = _build_index_text(text_blocks, table_blocks)

        for block_idx, text in enumerate(text_blocks):
            block_id = f"doc_{doc_id}_page_{page_label}_text_{block_idx}"
            blocks.append(
                {
                    "id": block_id,
                    "type": "text",
                    "text": text,
                    "page_label": page_label,
                    "index_text": index_text or text,
                    "doc_name": file_name,
                }
            )

        for block_idx, text in enumerate(table_blocks):
            block_id = f"doc_{doc_id}_page_{page_label}_table_{block_idx}"
            blocks.append(
                {
                    "id": block_id,
                    "type": "table",
                    "text": text,
                    "page_label": page_label,
                    "index_text": index_text or text,
                    "doc_name": file_name,
                }
            )

        images = list(getattr(page, "images", []) or [])
        for image_idx, image_file in enumerate(images):
            image_name = getattr(image_file, "name", f"img_{image_idx}.png")
            suffix = Path(image_name).suffix.lower() or ".png"
            image_rel = (
                Path(safe_stem)
                / f"page_{page_label:04d}"
                / f"img_{image_idx:04d}{suffix}"
            )
            image_abs = image_output_root / image_rel
            image_abs.parent.mkdir(parents=True, exist_ok=True)

            saved = False
            pil_image = getattr(image_file, "image", None)
            if pil_image is not None:
                try:
                    pil_image.save(str(image_abs))
                    saved = True
                except Exception:
                    saved = False

            if not saved:
                image_data = getattr(image_file, "data", None)
                if image_data:
                    image_abs.write_bytes(image_data)
                    saved = True

            if not saved:
                continue

            width = None
            height = None
            if pil_image is not None and hasattr(pil_image, "size"):
                width, height = pil_image.size

            block_id = f"doc_{doc_id}_page_{page_label}_image_{image_idx}"
            blocks.append(
                {
                    "id": block_id,
                    "type": "image",
                    "text": "",
                    "page_label": page_label,
                    "index_text": index_text,
                    "image_path": str(image_abs.resolve()),
                    "image_name": image_name,
                    "image_width": width,
                    "image_height": height,
                    "doc_name": file_name,
                }
            )

    return blocks
