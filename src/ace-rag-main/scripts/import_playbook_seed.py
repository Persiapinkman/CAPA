from __future__ import annotations

import argparse
from pathlib import Path

from ace_rag.core.config import get_settings
from ace_rag.playbook.seed import load_seed_items
from ace_rag.playbook.store import PlaybookStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Import ACE Playbook seed items.")
    parser.add_argument("--seed", type=Path, default=settings.PLAYBOOK_SEED_PATH)
    parser.add_argument("--db", type=Path, default=settings.PLAYBOOK_DB_PATH)
    args = parser.parse_args()

    store = PlaybookStore(args.db)
    count = store.import_items(load_seed_items(args.seed))
    print(f"Imported {count} playbook items into {args.db}")


if __name__ == "__main__":
    main()
