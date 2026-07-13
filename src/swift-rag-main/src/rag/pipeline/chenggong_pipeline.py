import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from doc_loader.roche_loader import get_md5
from src.api.schemas import ChunkingEmbeddingRequest
from src.core.config import get_settings
from src.rag.service import RAGService
from src.rag.utils import read_excel_to_dict_list

settings = get_settings()

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".json", ".jsonl", ".xlsx"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_data_source_dir(custom_dir: Optional[str] = None) -> Path:
    candidates: List[Path] = []

    if custom_dir:
        candidates.append(Path(custom_dir))

    env_data_source = os.getenv("DATA_SOURCE_DIR")
    if env_data_source:
        candidates.append(Path(env_data_source))

    candidates.extend(
        [
            Path("/app/data_source"),
            get_project_root() / "data_source",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "未找到可用的数据源目录。请创建 data_source 目录，或通过 DATA_SOURCE_DIR 指定目录。"
    )


def resolve_milvus_uri() -> str:
    env_uri = os.getenv("DATA_SOURCE_MILVUS_URI")
    if env_uri:
        uri_path = Path(env_uri)
    else:
        # 默认跟随主配置，统一使用 EvoQwen 2048 维向量库
        uri_path = Path(settings.DATA_SOURCE_VECTOR_DB_URI)

    uri_path.parent.mkdir(parents=True, exist_ok=True)
    return str(uri_path)


def sanitize_model_name(model_name: str) -> str:
    """将模型名转换为可用于文件名的短标识。"""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return sanitized or "unknown_model"


def build_model_tag(embedding_models: List[str]) -> str:
    unique_models = sorted(set(embedding_models))
    return "__".join(sanitize_model_name(model_name) for model_name in unique_models)


def resolve_vector_store_target(
    model_name: str,
    requested_embedding_models: List[str],
) -> Tuple[str, str]:
    env_uri = os.getenv("DATA_SOURCE_MILVUS_URI")
    collection_name = os.getenv("DATA_SOURCE_COLLECTION_NAME", "llamacollection")
    if env_uri and len(requested_embedding_models) == 1:
        uri_path = Path(env_uri)
        uri_path.parent.mkdir(parents=True, exist_ok=True)
        return str(uri_path), collection_name

    target = settings.DATA_SOURCE_VECTOR_STORE_CONFIGS.get(model_name)
    if target is not None:
        return target["uri"], target["collection_name"]

    # 未显式配置的模型，自动生成按模型区分的库文件，避免不同模型写入冲突
    project_root = get_project_root()
    model_slug = sanitize_model_name(model_name)
    fallback_uri = (
        project_root
        / "data_source"
        / "embedding_artifacts"
        / "documents"
        / f"milvus_data_source_{model_slug}.db"
    )
    fallback_uri.parent.mkdir(parents=True, exist_ok=True)
    return str(fallback_uri), collection_name


def resolve_result_csv_path(embedding_models: List[str]) -> Path:
    env_result_dir = os.getenv("DATA_SOURCE_RESULT_DIR")
    if env_result_dir:
        result_dir = Path(env_result_dir)
    else:
        result_dir = Path("/app/results")
        if not result_dir.exists():
            result_dir = get_project_root() / "results"

    result_dir.mkdir(parents=True, exist_ok=True)
    model_tag = build_model_tag(embedding_models)
    filename = f"data_source_embedding_report__{model_tag}.csv"
    return (result_dir / filename).resolve()


def collect_source_files(base_dir: Path) -> List[Path]:
    files = [
        file_path
        for file_path in base_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def extract_pdf_text(file_path: Path) -> str:
    reader_cls = None

    try:
        from pypdf import PdfReader as reader_cls  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader as reader_cls  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "无法解析PDF：未安装 pypdf 或 PyPDF2，请先安装其中一个依赖。"
            ) from exc

    reader = reader_cls(str(file_path))
    page_texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            page_texts.append(text)

    merged_text = "\n\n".join(page_texts).strip()
    if not merged_text:
        raise ValueError("PDF抽取结果为空")
    return merged_text


def build_autopdf_json_from_pdf(file_path: Path) -> str:
    """Build the AutoPDF-style JSON shape expected by the `autopdf` chunker."""
    reader_cls = None

    try:
        from pypdf import PdfReader as reader_cls  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader as reader_cls  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "无法解析PDF：未安装 pypdf 或 PyPDF2，请先安装其中一个依赖。"
            ) from exc

    reader = reader_cls(str(file_path))
    content_list = []
    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        media_box = getattr(page, "mediabox", None)
        page_width = float(getattr(media_box, "width", 0) or 0)
        page_height = float(getattr(media_box, "height", 0) or 0)
        content_list.append(
            {
                "page_id": page_idx,
                "page_width": page_width,
                "page_height": page_height,
                "others": [],
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        )

    if not content_list:
        raise ValueError("PDF抽取结果为空")

    return json.dumps({"content_list": content_list}, ensure_ascii=False)


def build_chunking_request(
    file_path: Path,
    data_source_dir: Path,
    embedding_models: List[str],
) -> Tuple[ChunkingEmbeddingRequest, str]:
    suffix = file_path.suffix.lower()
    relative_name = file_path.relative_to(data_source_dir).as_posix()
    doc_id = get_md5(relative_name)

    if suffix == ".pdf":
        text = build_autopdf_json_from_pdf(file_path)
        input_type = "autopdf"
    elif suffix == ".md":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        input_type = "markdown"
    elif suffix in {".txt", ".jsonl"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        input_type = "raw"
    elif suffix == ".xlsx":
        text = str(read_excel_to_dict_list(str(file_path)))
        input_type = "json_list"
    elif suffix == ".json":
        raw_json = file_path.read_text(encoding="utf-8", errors="ignore")
        json_obj = json.loads(raw_json)
        if isinstance(json_obj, list):
            text = str(json_obj)
            input_type = "json_list"
        elif isinstance(json_obj, dict):
            text = raw_json
            input_type = "autopdf"
        else:
            raise ValueError("JSON文件内容既不是list也不是dict，无法自动识别入库方式")
    else:
        raise ValueError(f"不支持的文件后缀: {suffix}")

    if not text.strip():
        raise ValueError("文件内容为空，跳过")

    request = ChunkingEmbeddingRequest(
        text=text,
        input_type=input_type,
        doc_id=doc_id,
        doc_name=relative_name,
        embedding_models=embedding_models,
    )
    return request, input_type


def offline_embedding(
    data_source_dir: Optional[str] = None,
    overwrite: bool = True,
    embedding_models: Optional[List[str]] = None,
) -> Path:
    requested_embedding_models = embedding_models or settings.DEFAULT_EMBEDDING_MODELS
    source_dir = resolve_data_source_dir(data_source_dir)
    result_csv = resolve_result_csv_path(requested_embedding_models)

    print(f"数据源目录: {source_dir}")

    source_files = collect_source_files(source_dir)
    print(f"发现待处理文件数量: {len(source_files)}")

    models_by_target: Dict[Tuple[str, str], List[str]] = {}
    for model_name in requested_embedding_models:
        target = resolve_vector_store_target(model_name, requested_embedding_models)
        models_by_target.setdefault(target, []).append(model_name)

    reports = []
    for (milvus_uri, collection_name), model_group in models_by_target.items():
        print(f"向量库路径: {milvus_uri}")
        print(f"向量库集合: {collection_name}")
        print(f"当前入库模型: {model_group}")

        service = RAGService(
            use_local_milvus={
                "uri": milvus_uri,
                "overwrite": overwrite,
                "collection_name": collection_name,
            },
            embedding_model_names=model_group,
            skip_failed_embedding_models=True,
        )
        loaded_models = list(service.embedding_models.keys())
        skipped_embedding_models = [
            model_name
            for model_name in model_group
            if model_name not in loaded_models
        ]

        print(f"实际加载的向量模型: {loaded_models}")
        if skipped_embedding_models:
            print(f"跳过不可用向量模型: {skipped_embedding_models}")
            for model_name in skipped_embedding_models:
                print(f"- {model_name}: {service.embedding_model_errors.get(model_name, '未知错误')}")

        for file_path in tqdm(source_files):
            relative_path = file_path.relative_to(source_dir).as_posix()
            report = {
                "file": relative_path,
                "embedding_models": ",".join(loaded_models),
                "vector_db_uri": milvus_uri,
                "status": "failed",
                "input_type": "",
                "error": "",
            }

            try:
                request, input_type = build_chunking_request(
                    file_path=file_path,
                    data_source_dir=source_dir,
                    embedding_models=loaded_models,
                )
                service.chunking_embedding(request)
                report["status"] = "success"
                report["input_type"] = input_type
            except Exception as exc:
                report["error"] = str(exc)

            reports.append(report)

    report_df = pd.DataFrame(
        reports,
        columns=["file", "embedding_models", "vector_db_uri", "status", "input_type", "error"],
    )
    report_df.to_csv(result_csv, index=False, encoding="utf-8")

    success_count = int((report_df["status"] == "success").sum())
    failed_count = int((report_df["status"] == "failed").sum())
    print(f"入库完成，成功: {success_count}，失败: {failed_count}")
    print(f"处理报告已保存: {result_csv}")

    return result_csv


if __name__ == "__main__":
    offline_embedding()
