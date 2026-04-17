from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import ARTIFACT_SPECS_DIR, CONTRACTS_DIR, LIBRARY_DIR, REPO_ROOT


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=None)
def repo_manifest() -> dict[str, Any]:
    return read_yaml(REPO_ROOT / "repo_manifest.yaml")


@lru_cache(maxsize=None)
def authority_contract() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "authority_contract.yaml")


@lru_cache(maxsize=None)
def stage_registry() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "stage_registry.yaml")


@lru_cache(maxsize=None)
def module_registry() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "module_registry.yaml")


@lru_cache(maxsize=None)
def artifact_registry() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "artifact_registry.yaml")


@lru_cache(maxsize=None)
def promotion_gates() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "promotion_gates.yaml")


@lru_cache(maxsize=None)
def runtime_config_schema_contract() -> dict[str, Any]:
    return read_yaml(CONTRACTS_DIR / "runtime_config_schema_contract.yaml")


@lru_cache(maxsize=None)
def runtime_config_bundle() -> dict[str, Any]:
    return read_yaml(ARTIFACT_SPECS_DIR / "vbtpro_lab_runtime_config_bundle_20260409.yaml")


def runtime_config_path(config_name: str) -> Path:
    bundle = runtime_config_bundle()
    try:
        return Path(bundle["runtime_config_documents"][config_name])
    except KeyError as exc:
        raise KeyError(f"Unknown runtime config document: {config_name}") from exc


@lru_cache(maxsize=None)
def runtime_config_document(config_name: str) -> dict[str, Any]:
    path = runtime_config_path(config_name)
    return read_yaml(path)


@lru_cache(maxsize=None)
def baseline_spec() -> dict[str, Any]:
    return read_yaml(ARTIFACT_SPECS_DIR / "gold_baseline_bootstrap_20260409.yaml")


@lru_cache(maxsize=None)
def local_build_registry() -> dict[str, Any]:
    return read_yaml(LIBRARY_DIR / "vbtpro_lab_build_registry_20260409.yaml")


def module_ids() -> list[str]:
    return [entry["module_id"] for entry in module_registry()["modules"]]
