from __future__ import annotations

from dataclasses import dataclass


MODULE_ID = "machine_library"
PRIMARY_STAGE_ID = "S8_shadow_parity_and_promotion_gate"
MODULE_OUTPUTS = ("local_registry_snapshot", "local_query_contract")


@dataclass(frozen=True, slots=True)
class MachineLibrarySnapshotSpec:
    module_id: str = MODULE_ID
    stage_id: str = PRIMARY_STAGE_ID
    output_artifacts: tuple[str, ...] = MODULE_OUTPUTS


@dataclass(frozen=True, slots=True)
class MachineLibraryQuerySpec:
    module_id: str = MODULE_ID
    query_kinds: tuple[str, ...] = (
        "artifact_path",
        "alias",
        "map_id",
        "module_id",
        "notebook_surface",
        "registry_id",
        "repo_path",
        "risk_id",
        "stage_id",
        "topic",
        "truth_topic",
        "text",
    )
