"""Pytest configuration: make the local rtmlib checkout importable."""

import sys
from pathlib import Path

# Ensure the repository root (containing the rtmlib package) takes priority
# over any installed copy of rtmlib in site-packages.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
