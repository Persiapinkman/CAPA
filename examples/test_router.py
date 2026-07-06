#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import router


def main() -> None:
    repo_root = ROOT
    parser = argparse.ArgumentParser(description="Test demo router output")
    parser.add_argument("--text", required=True, help="User input text")
    parser.add_argument("--image", default="", help="Optional image path")
    parser.add_argument(
        "--model",
        default="",
        help="Optional router model override (default: demo.router.DEMO_ROUTE_MODEL / env DEMO_ROUTE_MODEL)",
    )
    args = parser.parse_args()

    image_path = ""
    if args.image.strip():
        p = Path(args.image).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        image_path = str(p)

    result = router.choose_route_with_fallback(
        text=args.text,
        image_path=image_path or None,
        model=(args.model.strip() or None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
