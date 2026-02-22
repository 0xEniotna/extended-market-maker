from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Keep tests deterministic: do not load developer-local MM_* values.
os.environ["ENV"] = "env.test"
for key in list(os.environ.keys()):
    if key.startswith("MM_"):
        os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def _restore_environment() -> None:
    """Prevent environment mutations from leaking across tests."""
    snapshot = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
