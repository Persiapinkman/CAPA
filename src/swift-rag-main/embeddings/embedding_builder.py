from pathlib import Path

from embeddings.dense_embedding import DenseEmbeddings


def _build_local_model_error(model_path: Path) -> str:
    hints = [
        f"本地向量模型目录不可用: {model_path}",
    ]

    if not model_path.exists():
        hints.append("目录不存在，请先准备本地模型文件，或将配置中的 model_path 改为正确路径。")
    elif not any(model_path.iterdir()):
        hints.append("目录是空的，当前没有任何模型文件。")
    else:
        hints.append("目录中缺少有效的 HuggingFace / sentence-transformers 模型配置文件。")

    if model_path.name == "bge-m3":
        hints.append("请将 `bge-m3` 模型文件下载到该目录，或在 `src/core/config.py` 中修改 `model_path`。")
    elif model_path.name == "EvoQwen2.5-VL-Retriever-3B-v1":
        hints.append("请将 `EvoQwen2.5-VL-Retriever-3B-v1` 模型文件下载到该目录，或在 `src/core/config.py` 中修改 `model_path`。")
    elif model_path.name == "EvoQwen2.5-VL-Retriever-7B-v1":
        hints.append("请将 `EvoQwen2.5-VL-Retriever-7B-v1` 模型文件下载到该目录，或在 `src/core/config.py` 中修改 `model_path`。")

    hints.append("期望至少存在 `config.json` 或 `modules.json` 等模型配置文件。")
    return " ".join(hints)


def _validate_model_source(model_name_or_path):
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        return

    if model_path.is_dir():
        if (not any(model_path.iterdir())
            or (not (model_path / "config.json").exists() and not (model_path / "modules.json").exists())):
            raise FileNotFoundError(_build_local_model_error(model_path))


def build_embedding_model(model_name_or_path,batch_size,device):
    _validate_model_source(model_name_or_path)
    embedding_model = DenseEmbeddings(
        model_name=model_name_or_path,
        embed_batch_size=batch_size,
        device=device
        )
    return embedding_model
