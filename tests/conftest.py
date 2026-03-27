"""Pytest configuration for the fin-portfolio test suite.

Adds the project root to sys.path so that `from services.xxx import ...`
and `from shared.xxx import ...` resolve correctly when tests run from
the `tests/` subdirectory.
"""

import sys
from pathlib import Path

# Project root = parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
