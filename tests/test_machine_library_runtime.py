from __future__ import annotations

from pathlib import Path

import yaml

from vbtpro_lab.machine_library.artifacts import build_event_log_path, local_query_contract_path, local_registry_snapshot_path
from vbtpro_lab.machine_library.runtime import (
    build_registry_record,
    lookup_registered_artifact,
    module_registry_record,
    query_machine_library,
    update_build_registry_module_status,
)


def test_runtime_query_module_id_matches_contract_shape() -> None:
    payload = query_machine_library("module_id", "machine_library")
    assert payload["query_kind"] == "module_id"
    assert payload["module_records"]
    assert payload["module_records"][0]["module_id"] == "machine_library"
    assert payload["stage_links"] == [{"stage_id": "S8_shadow_parity_and_promotion_gate"}]


def test_runtime_lookup_helpers_resolve_registered_contracts() -> None:
    assert lookup_registered_artifact("validation_contract").endswith("contracts/validation_contract.yaml")
    assert module_registry_record()["build_status"] == "built_validated"
    assert build_registry_record()["build_status"] == "built_validated"
    assert local_registry_snapshot_path().endswith("vbtpro_machine_library_build_manifest_20260409.yaml")
    assert local_query_contract_path().endswith("vbtpro_machine_library_query_examples_20260409.yaml")
    assert build_event_log_path().endswith("vbtpro_lab_build_events_20260409.yaml")


def test_registered_runtime_config_artifacts_are_queryable_by_artifact_path() -> None:
    artifact_path = "/home/benji/vectorbtpro-machine-library/artifacts/specs/runtime_config/timeframe_registry.yaml"
    payload = query_machine_library("artifact_path", artifact_path)
    assert payload["query_kind"] == "artifact_path"
    assert payload["rows"]
    assert any(row["artifact_kind"] == "registered_artifact" for row in payload["rows"])
    assert any(row["role"] == "timeframe_registry" for row in payload["rows"])


def test_build_event_active_summary_is_queryable_without_runtime_surface_promotion() -> None:
    artifact_path = "/home/benji/vectorbtpro-machine-library/reports/strategy_factory/vbtpro_lab_build_events_active_summary_20260415.yaml"
    payload = query_machine_library("artifact_path", artifact_path)
    assert payload["query_kind"] == "artifact_path"
    assert payload["rows"]
    assert any(row["artifact_kind"] == "note_surface" for row in payload["rows"])
    assert any(row["role"] == "build_event_active_summary" for row in payload["rows"])
    assert not any(row["artifact_kind"] == "runtime_artifact_surface" for row in payload["rows"])


def test_external_vectorbtpro_library_index_is_queryable() -> None:
    artifact_path = "/home/benji/vectorbtpro-machine-library/artifacts/specs/control_plane/vbtpro_external_vectorbtpro_library_index_20260416.yaml"
    payload = query_machine_library("artifact_path", artifact_path)
    assert payload["query_kind"] == "artifact_path"
    assert payload["rows"]
    assert any(row["artifact_kind"] == "registered_artifact" for row in payload["rows"])


def test_external_vectorbtpro_repo_path_is_queryable() -> None:
    repo_path = "/home/benji/vectorbtpro-machine-library/external/vectorbt.pro/vectorbtpro/generic/splitting/base.py"
    payload = query_machine_library("repo_path", repo_path)
    assert payload["query_kind"] == "repo_path"
    assert payload["routes"]
    assert any(row["target_path"] == repo_path for row in payload["routes"])


def test_external_vectorbtpro_symbol_is_queryable() -> None:
    payload = query_machine_library("symbol", "SplitterCV")
    assert payload["query_kind"] == "symbol"
    assert payload["rows"]
    assert any(row["symbol_name"] == "SplitterCV" for row in payload["rows"])
    assert any(row["kind"] == "class" for row in payload["rows"])
    assert payload["rows"][0]["source_kind"] == "external_runtime_export"


def test_text_query_prefers_external_symbol_rows() -> None:
    payload = query_machine_library("text", "SplitterCV")
    assert payload["query_kind"] == "text"
    assert payload["rows"]
    assert payload["rows"][0]["source_kind"] == "external_runtime_export"


def test_build_registry_status_update_records_event(tmp_path: Path) -> None:
    registry_path = tmp_path / "build_registry.yaml"
    event_log_path = tmp_path / "build_events.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "modules": [
                    {
                        "module_id": "machine_library",
                        "build_status": "active_next",
                        "blockers": [],
                    }
                ]
            },
            sort_keys=False,
        )
    )

    result = update_build_registry_module_status(
        module_id="machine_library",
        build_status="built_validated",
        summary="machine library runtime wiring landed",
        stage_id="S8_shadow_parity_and_promotion_gate",
        blockers=[],
        registry_path=registry_path,
        event_log_path=event_log_path,
    )

    updated_registry = yaml.safe_load(registry_path.read_text())
    updated_event_log = yaml.safe_load(event_log_path.read_text())

    assert result["module_record"]["build_status"] == "built_validated"
    assert updated_registry["modules"][0]["build_status"] == "built_validated"
    assert updated_event_log["events"][0]["event_kind"] == "build_status_updated"
    assert updated_event_log["events"][0]["new_build_status"] == "built_validated"
