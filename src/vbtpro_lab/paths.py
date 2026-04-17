from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"
ARTIFACT_SPECS_DIR = REPO_ROOT / "artifacts" / "specs"
LIBRARY_DIR = REPO_ROOT / "reports" / "strategy_factory"
SCRIPTS_DIR = REPO_ROOT / "scripts"
