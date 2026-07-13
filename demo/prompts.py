"""Compatibility import for prompt builders moved to :mod:`capa.prompts`."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capa.prompts import *  # noqa: F401,F403,E402
