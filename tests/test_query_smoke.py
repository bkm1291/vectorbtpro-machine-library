from __future__ import annotations

import json
import subprocess
import sys

from conftest import REPO_ROOT


def test_local_query_smoke_by_module_id() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "query_vbtpro_machine_library.py"),
        "--module-id",
        "machine_library",
    ]
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["query_kind"] == "module_id"


def test_local_query_smoke_by_external_repo_path() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "query_vbtpro_machine_library.py"),
        "--repo-path",
        str(REPO_ROOT / "external/vectorbt.pro/vectorbtpro/generic/splitting/base.py"),
    ]
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["query_kind"] == "repo_path"
    assert payload["routes"]


def test_local_query_smoke_by_symbol() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "query_vbtpro_machine_library.py"),
        "--symbol",
        "SplitterCV",
    ]
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["query_kind"] == "symbol"
    assert payload["rows"]
