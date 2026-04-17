from __future__ import annotations

from pathlib import Path

from vbtpro_lab.paths import LIBRARY_DIR

from .models import MODULE_OUTPUTS, MachineLibraryQuerySpec, MachineLibrarySnapshotSpec


def compiled_library_paths() -> dict[str, str]:
    return {
        "db_path": str(LIBRARY_DIR / "vbtpro_machine_library_20260409.sqlite"),
        "build_manifest_path": str(LIBRARY_DIR / "vbtpro_machine_library_build_manifest_20260409.yaml"),
        "coverage_dashboard_path": str(LIBRARY_DIR / "vbtpro_library_coverage_dashboard_20260409.yaml"),
        "query_examples_path": str(LIBRARY_DIR / "vbtpro_machine_library_query_examples_20260409.yaml"),
        "build_events_path": str(LIBRARY_DIR / "vbtpro_lab_build_events_20260409.yaml"),
        "build_registry_path": str(LIBRARY_DIR / "vbtpro_lab_build_registry_20260409.yaml"),
    }


def required_output_artifacts() -> tuple[str, ...]:
    return MODULE_OUTPUTS


def snapshot_spec() -> MachineLibrarySnapshotSpec:
    return MachineLibrarySnapshotSpec()


def query_spec() -> MachineLibraryQuerySpec:
    return MachineLibraryQuerySpec()


def compiled_library_files_exist() -> bool:
    required = compiled_library_paths()
    return all(
        Path(required[key]).exists()
        for key in (
            "db_path",
            "build_manifest_path",
            "coverage_dashboard_path",
            "query_examples_path",
            "build_events_path",
            "build_registry_path",
        )
    )
