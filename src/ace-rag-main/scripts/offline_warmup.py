from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ace_rag.ace.service import AceService
from ace_rag.api.schemas import QueryRequest
from ace_rag.core.config import get_settings
from ace_rag.playbook.store import PlaybookStore
from ace_rag.v2_client.client import V2Client


async def run(path: Path) -> None:
    settings = get_settings()
    service = AceService(
        store=PlaybookStore(settings.PLAYBOOK_DB_PATH),
        v2_client=V2Client(settings.V2_BASE_URL, settings.V2_TIMEOUT_SECONDS),
        seed_path=settings.PLAYBOOK_SEED_PATH,
        auto_import_seed=settings.AUTO_IMPORT_SEED,
    )
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        query = line.strip()
        if not query or query.startswith("#"):
            continue
        response = await service.query(QueryRequest(query=query))
        print(json.dumps({"line": line_no, "query": query, "run_id": response.run_id}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline ACE warmup questions through v3.")
    parser.add_argument("questions", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.questions))


if __name__ == "__main__":
    main()
