"""Pytest configuration for Brian2 integration tests."""
from __future__ import annotations

import sys
from pathlib import Path

from brian2 import prefs


# Ensure the project package is importable when running the test suite.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Use the fast NumPy runtime to avoid repeated C++ compilation during tests.
prefs.codegen.target = "numpy"
