import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.api.schemas import ChunkNodeWithEmbedding


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingArtifactStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.documents_dir = self.base_dir / "documents"
        self.tables_dir = self.base_dir / "tables"
        self.adela_dir = self.base_dir / "adela"

        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.adela_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(self, value: str, default: str = "unknown") -> str:
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
        return sanitized[:80] or default

    def _short_hash(self, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _resolve_structured_dir(self, artifact_namespace: str) -> Path:
        namespace = self._slugify(artifact_namespace, default="tables")
        if namespace == "tables":
            return self.tables_dir
        if namespace == "adela":
            return self.adela_dir
        custom_dir = self.base_dir / namespace
        custom_dir.mkdir(parents=True, exist_ok=True)
        return custom_dir

    def _structured_artifact_type(self, artifact_namespace: str) -> str:
        namespace = self._slugify(artifact_namespace, default="tables")
        if namespace == "tables":
            return "table_embeddings"
        return f"{namespace}_embeddings"

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def _atomic_write_npy(self, path: Path, matrix: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.save(handle, matrix)
        tmp_path.replace(path)

    def _atomic_write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        tmp_path.replace(path)

    def save_document_embeddings(
        self,
        doc_id: str,
        doc_name: str,
        input_type: str,
        embedding_models: List[str],
        index_nodes: List[ChunkNodeWithEmbedding],
    ) -> Path:
        model_tag = "__".join(self._slugify(model_name) for model_name in sorted(set(embedding_models)))
        doc_name_slug = self._slugify(Path(doc_name).stem or doc_name)
        file_stem = f"{self._slugify(doc_id)}__{doc_name_slug}__{model_tag or 'unknown_model'}"

        jsonl_path = self.documents_dir / f"{file_stem}.jsonl"
        meta_path = self.documents_dir / f"{file_stem}.meta.json"

        self._atomic_write_jsonl(
            jsonl_path,
            (node.model_dump(mode="json") for node in index_nodes),
        )

        metadata = {
            "artifact_type": "document_embeddings",
            "created_at": _utc_now_iso(),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "input_type": input_type,
            "embedding_models": embedding_models,
            "node_count": len(index_nodes),
            "jsonl_path": str(jsonl_path),
        }
        self._atomic_write_text(
            meta_path,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )
        return jsonl_path

    def resolve_table_embedding_paths(
        self,
        data_path: str,
        model_name: str,
        searchable_fields: List[str],
        artifact_namespace: str = "tables",
    ) -> Tuple[Path, Path]:
        resolved_data_path = Path(data_path).resolve()
        field_hash = self._short_hash(searchable_fields)
        source_hash = self._short_hash(str(resolved_data_path))
        artifact_dir = self._resolve_structured_dir(artifact_namespace)
        stem = (
            f"{self._slugify(resolved_data_path.stem)}"
            f"__{self._slugify(model_name)}"
            f"__fields_{field_hash}"
            f"__src_{source_hash}"
        )
        return (
            artifact_dir / f"{stem}.npy",
            artifact_dir / f"{stem}.meta.json",
        )

    def load_table_embeddings(
        self,
        data_path: str,
        model_name: str,
        searchable_fields: List[str],
        row_ids: List[str],
        artifact_namespace: str = "tables",
    ) -> Optional[np.ndarray]:
        matrix_path, meta_path = self.resolve_table_embedding_paths(
            data_path=data_path,
            model_name=model_name,
            searchable_fields=searchable_fields,
            artifact_namespace=artifact_namespace,
        )
        if not matrix_path.exists() or not meta_path.exists():
            return None

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            source_path = Path(data_path).resolve()
            source_stat = source_path.stat()
            namespace = self._slugify(artifact_namespace, default="tables")
            expected = {
                "artifact_type": self._structured_artifact_type(namespace),
                "data_path": str(source_path),
                "model_name": model_name,
                "searchable_fields": searchable_fields,
                "row_ids": row_ids,
                "data_file_size": source_stat.st_size,
                "data_file_mtime_ns": source_stat.st_mtime_ns,
            }
            for key, expected_value in expected.items():
                if metadata.get(key) != expected_value:
                    return None
            metadata_namespace = metadata.get("artifact_namespace")
            if metadata_namespace is not None and metadata_namespace != namespace:
                return None
            if metadata_namespace is None and namespace != "tables":
                return None

            matrix = np.load(matrix_path)
            if matrix.shape[0] != len(row_ids):
                return None
            return np.asarray(matrix, dtype=np.float32)
        except Exception:
            return None

    def save_table_embeddings(
        self,
        data_path: str,
        model_name: str,
        searchable_fields: List[str],
        row_ids: List[str],
        matrix: np.ndarray,
        artifact_namespace: str = "tables",
    ) -> Path:
        matrix_path, meta_path = self.resolve_table_embedding_paths(
            data_path=data_path,
            model_name=model_name,
            searchable_fields=searchable_fields,
            artifact_namespace=artifact_namespace,
        )
        source_path = Path(data_path).resolve()
        source_stat = source_path.stat()
        namespace = self._slugify(artifact_namespace, default="tables")

        metadata = {
            "artifact_type": self._structured_artifact_type(namespace),
            "artifact_namespace": namespace,
            "created_at": _utc_now_iso(),
            "data_path": str(source_path),
            "model_name": model_name,
            "searchable_fields": searchable_fields,
            "row_ids": row_ids,
            "matrix_shape": list(matrix.shape),
            "matrix_dtype": str(matrix.dtype),
            "matrix_path": str(matrix_path),
            "data_file_size": source_stat.st_size,
            "data_file_mtime_ns": source_stat.st_mtime_ns,
        }

        self._atomic_write_npy(matrix_path, np.asarray(matrix, dtype=np.float32))
        self._atomic_write_text(
            meta_path,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )
        return matrix_path
