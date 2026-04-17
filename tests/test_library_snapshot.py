from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
from conftest import REPO_ROOT


def test_local_machine_library_rebuild_then_validate_only_passes() -> None:
    rebuild_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_vbtpro_machine_library.py"),
    ]
    validate_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_vbtpro_machine_library.py"),
        "--validate-only",
    ]
    subprocess.run(rebuild_cmd, check=True, cwd=REPO_ROOT)
    subprocess.run(validate_cmd, check=True, cwd=REPO_ROOT)


def test_machine_library_lock_rejects_parallel_validate_only(tmp_path) -> None:
    script_path = REPO_ROOT / "scripts" / "build_vbtpro_machine_library.py"
    spec = importlib.util.spec_from_file_location("build_vbtpro_machine_library", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "machine_library.sqlite"

    with module.machine_library_lock(db_path, "rebuild"):
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--db-path",
                str(db_path),
                "--validate-only",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    assert proc.returncode == 2
    assert "refuse to run validate_only in parallel" in proc.stdout
