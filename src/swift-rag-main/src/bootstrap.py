import os
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except (OSError, zipfile.BadZipFile):
        return False


def configure_nltk_data() -> None:
    """Point NLTK at a safe cache and skip corrupted user-level archives."""
    safe_nltk_dir = (
        Path(sys.prefix)
        / "lib"
        / f"python{os.sys.version_info.major}.{os.sys.version_info.minor}"
        / "site-packages"
        / "llama_index"
        / "core"
        / "_static"
        / "nltk_cache"
    )

    if not safe_nltk_dir.exists():
        safe_nltk_dir = PROJECT_ROOT / ".cache" / "nltk_data"
        safe_nltk_dir.mkdir(parents=True, exist_ok=True)

    os.environ["NLTK_DATA"] = str(safe_nltk_dir)

    import nltk

    sanitized_paths = [str(safe_nltk_dir)]
    for raw_path in nltk.data.path:
        path = Path(raw_path)
        if path == safe_nltk_dir:
            continue

        punkt_zip = path / "tokenizers" / "punkt.zip"
        punkt_tab_zip = path / "tokenizers" / "punkt_tab.zip"
        if (punkt_zip.exists() and not _is_valid_zip(punkt_zip)) or (
            punkt_tab_zip.exists() and not _is_valid_zip(punkt_tab_zip)
        ):
            continue

        sanitized_paths.append(raw_path)

    nltk.data.path = sanitized_paths
