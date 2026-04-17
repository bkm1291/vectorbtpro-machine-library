from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_REPO_ROOT = Path("/home/benji/trade_scanner2")
SRC_ROOT = REPO_ROOT / "src"
LEGACY_TEST_ENV_VAR = "VBTPRO_ENABLE_LEGACY_TESTS"


def _inject_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


_inject_path(SRC_ROOT)


def legacy_repo_tests_enabled() -> bool:
    return os.environ.get(LEGACY_TEST_ENV_VAR) == "1"


def ensure_legacy_repo_path() -> bool:
    if not LEGACY_REPO_ROOT.exists():
        return False
    _inject_path(LEGACY_REPO_ROOT)
    return True


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "legacy_compat: requires explicit opt-in and the legacy /home/benji/trade_scanner2 checkout",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if legacy_repo_tests_enabled():
        return
    marker = pytest.mark.skip(reason=f"requires {LEGACY_TEST_ENV_VAR}=1")
    for item in items:
        if "legacy_compat" in item.keywords:
            item.add_marker(marker)
