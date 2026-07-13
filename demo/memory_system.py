"""Compatibility import for memory components moved to :mod:`capa.memory`."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capa.memory import *  # noqa: F401,F403,E402
