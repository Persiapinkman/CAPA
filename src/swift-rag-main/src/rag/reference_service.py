import csv
from pathlib import Path
from typing import Dict, List, Optional

from src.api.schemas import ChunkNodeWithScore, ReferenceItem
from src.core.config import get_settings


settings = get_settings()


def _normalize_doc_name(doc_name: str) -> str:
    normalized = (doc_name or "").strip().replace("\\", "/")
    return normalized.lower()


def _basename(doc_name: str) -> str:
    normalized = (doc_name or "").strip().replace("\\", "/")
    return Path(normalized).name


class PDFReferenceStore:
    def __init__(self, mapping_path: Optional[str] = None) -> None:
        self.mapping_path = Path(mapping_path or settings.PDF_REFERENCE_MAPPING_PATH)
        self._mapping: Dict[str, Optional[str]] = {}
        self._last_mtime: Optional[float] = None

    def _ensure_loaded(self) -> None:
        if not self.mapping_path.exists():
            self._mapping = {}
            self._last_mtime = None
            return

        current_mtime = self.mapping_path.stat().st_mtime
        if self._last_mtime == current_mtime:
            return

        mapping: Dict[str, Optional[str]] = {}
        with self.mapping_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pdf_file = (row.get("pdf_file") or "").strip()
                if not pdf_file:
                    continue
                url = (row.get("url") or "").strip() or None
                normalized_full = _normalize_doc_name(pdf_file)
                normalized_base = _normalize_doc_name(_basename(pdf_file))
                mapping[normalized_full] = url
                mapping[normalized_base] = url

        self._mapping = mapping
        self._last_mtime = current_mtime

    def resolve_references(
        self, retrieved_chunks: List[ChunkNodeWithScore]
    ) -> List[ReferenceItem]:
        self._ensure_loaded()

        references: List[ReferenceItem] = []
        seen = set()
        for chunk in retrieved_chunks:
            raw_doc_name = chunk.doc_name or chunk.metadata.get("file_name") or ""
            if not raw_doc_name:
                continue

            reference = self._resolve_reference_item(raw_doc_name)
            normalized_base = _normalize_doc_name(reference.doc_name)
            if normalized_base in seen:
                continue

            references.append(reference)
            seen.add(normalized_base)

        return references

    def resolve_doc_names(self, doc_names: List[str]) -> List[ReferenceItem]:
        self._ensure_loaded()

        references: List[ReferenceItem] = []
        seen = set()
        for raw_doc_name in doc_names:
            if not raw_doc_name:
                continue

            reference = self._resolve_reference_item(raw_doc_name)
            normalized_base = _normalize_doc_name(reference.doc_name)
            if normalized_base in seen:
                continue

            references.append(reference)
            seen.add(normalized_base)

        return references

    def _resolve_reference_item(self, raw_doc_name: str) -> ReferenceItem:
        display_name = _basename(raw_doc_name) or raw_doc_name
        normalized_full = _normalize_doc_name(raw_doc_name)
        normalized_base = _normalize_doc_name(display_name)

        url = self._mapping.get(normalized_full)
        if url is None:
            url = self._mapping.get(normalized_base)

        return ReferenceItem(doc_name=display_name, url=url)
