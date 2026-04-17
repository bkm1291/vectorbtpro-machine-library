from __future__ import annotations

from .models import MODULE_OUTPUTS
from .registry import compiled_library_paths, query_spec, required_output_artifacts


def validate_machine_library_output_contract() -> None:
    assert required_output_artifacts() == MODULE_OUTPUTS


def validate_machine_library_query_contract() -> None:
    assert "text" in query_spec().query_kinds
    assert "text_like" not in query_spec().query_kinds


def validate_machine_library_paths_shape() -> None:
    paths = compiled_library_paths()
    assert paths["db_path"].endswith(".sqlite")
    assert paths["build_manifest_path"].endswith(".yaml")
    assert paths["coverage_dashboard_path"].endswith(".yaml")
    assert paths["query_examples_path"].endswith(".yaml")
    assert paths["build_events_path"].endswith(".yaml")
    assert paths["build_registry_path"].endswith(".yaml")
