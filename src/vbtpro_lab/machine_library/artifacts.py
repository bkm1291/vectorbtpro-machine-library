from __future__ import annotations

from .models import MODULE_OUTPUTS
from .registry import compiled_library_paths


def declared_output_artifacts() -> tuple[str, ...]:
    return MODULE_OUTPUTS


def reserved_artifact_paths() -> dict[str, str]:
    return compiled_library_paths()


def local_registry_snapshot_path() -> str:
    return compiled_library_paths()["build_manifest_path"]


def local_query_contract_path() -> str:
    return compiled_library_paths()["query_examples_path"]


def build_event_log_path() -> str:
    return compiled_library_paths()["build_events_path"]
