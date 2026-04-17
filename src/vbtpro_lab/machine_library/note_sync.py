from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from vbtpro_lab.paths import ARTIFACT_SPECS_DIR, LIBRARY_DIR


BUILD_REGISTRY_PATH = LIBRARY_DIR / "vbtpro_lab_build_registry_20260409.yaml"
NOTES_PLAN_PATH = ARTIFACT_SPECS_DIR / "vbtpro_lab_reset_rebuild_notes_plan_20260409.yaml"
CONTROL_PLANE_DIR = ARTIFACT_SPECS_DIR / "control_plane"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return {}
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _norm_paths(values: Iterable[str | Path]) -> list[str]:
    return sorted({str(Path(value)) for value in values})


def _note_exists(path: str | Path) -> bool:
    path_obj = Path(path)
    return path_obj.exists() and path_obj.suffix in {".yaml", ".yml"}


def _is_note_surface_path(path: str | Path, *, specs_root: Path, control_plane_dir: Path) -> bool:
    path_obj = Path(path)
    if not _note_exists(path_obj):
        return False
    if path_obj.parent == control_plane_dir:
        return True
    if str(path_obj).startswith(str(specs_root / "runtime_config")):
        return False
    if str(path_obj).startswith(str(specs_root)):
        return True
    try:
        payload = _read_yaml(path_obj)
    except Exception:
        return False
    artifact_type = str(payload.get("artifact_type", "")).lower()
    return any(token in artifact_type for token in ("plan", "note", "review", "audit", "control_plane"))


def _discover_explicit_live_notes(
    *,
    specs_root: Path,
    control_plane_dir: Path,
    explicit_paths: Iterable[str | Path] = (),
) -> list[str]:
    discovered: list[str] = []
    for value in explicit_paths:
        path = Path(value)
        if not _is_note_surface_path(path, specs_root=specs_root, control_plane_dir=control_plane_dir) or path.parent == control_plane_dir:
            continue
        discovered.append(str(path))
    return _norm_paths(discovered)


def _discover_control_plane_notes(
    *,
    control_plane_dir: Path,
    explicit_paths: Iterable[str | Path] = (),
) -> list[str]:
    discovered = [str(path) for path in sorted(control_plane_dir.glob("*.y*ml")) if path.exists()]
    for value in explicit_paths:
        path = Path(value)
        if _note_exists(path) and path.parent == control_plane_dir:
            discovered.append(str(path))
    return _norm_paths(discovered)


def sync_repo_truth_note_surfaces(
    *,
    build_registry_path: str | Path | None = None,
    notes_plan_path: str | Path | None = None,
    specs_root: str | Path | None = None,
    control_plane_dir: str | Path | None = None,
    explicit_note_paths: Iterable[str | Path] = (),
    explicit_artifact_paths: Iterable[str | Path] = (),
    explicit_runtime_artifact_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    build_registry_file = Path(build_registry_path) if build_registry_path is not None else BUILD_REGISTRY_PATH
    notes_plan_file = Path(notes_plan_path) if notes_plan_path is not None else NOTES_PLAN_PATH
    specs_dir = Path(specs_root) if specs_root is not None else ARTIFACT_SPECS_DIR
    control_dir = Path(control_plane_dir) if control_plane_dir is not None else CONTROL_PLANE_DIR

    build_registry = _read_yaml(build_registry_file)
    notes_plan = _read_yaml(notes_plan_file)

    note_hygiene = build_registry.setdefault("note_hygiene", {})
    raw_live = [str(Path(value)) for value in note_hygiene.get("active_live_note_surfaces", [])]
    raw_control = [str(Path(value)) for value in note_hygiene.get("active_control_plane_surfaces", [])]
    raw_runtime_artifacts = [str(Path(value)) for value in note_hygiene.get("active_runtime_artifact_surfaces", [])]

    current_live = [
        value
        for value in raw_live
        if _is_note_surface_path(value, specs_root=specs_dir, control_plane_dir=control_dir)
    ]
    current_control = [
        value
        for value in raw_control
        if _is_note_surface_path(value, specs_root=specs_dir, control_plane_dir=control_dir)
    ]
    current_runtime_artifacts = [
        str(Path(value))
        for value in raw_runtime_artifacts
        if Path(value).exists()
    ]

    preserved_live = [str(Path(value)) for value in current_live if Path(value).parent != control_dir]
    explicit_runtime_artifacts = _norm_paths(
        [
            value
            for value in explicit_runtime_artifact_paths
            if Path(value).exists()
        ]
    )
    final_runtime_artifacts = _norm_paths([*current_runtime_artifacts, *explicit_runtime_artifacts])

    discovered_live = _discover_explicit_live_notes(
        specs_root=specs_dir,
        control_plane_dir=control_dir,
        explicit_paths=explicit_note_paths,
    )
    discovered_control = _discover_control_plane_notes(
        control_plane_dir=control_dir,
        explicit_paths=explicit_note_paths,
    )

    final_live = _norm_paths([*preserved_live, *discovered_live, str(notes_plan_file)])
    final_control = _norm_paths(discovered_control)

    authoritative_control = {
        str(Path(value))
        for value in notes_plan.get("authoritative_note_surfaces", {}).values()
        if str(value).startswith(str(control_dir))
    }
    final_support = [path for path in final_control if path not in authoritative_control]

    changed = False
    if raw_live != final_live:
        note_hygiene["active_live_note_surfaces"] = final_live
        changed = True
    if raw_control != final_control:
        note_hygiene["active_control_plane_surfaces"] = final_control
        changed = True
    if raw_runtime_artifacts != final_runtime_artifacts:
        note_hygiene["active_runtime_artifact_surfaces"] = final_runtime_artifacts
        changed = True
    if notes_plan.get("active_regime_control_plane_support_surfaces", []) != final_support:
        notes_plan["active_regime_control_plane_support_surfaces"] = final_support
        changed = True

    if changed:
        _write_yaml(build_registry_file, build_registry)
        _write_yaml(notes_plan_file, notes_plan)

    return {
        "changed": changed,
        "build_registry_path": str(build_registry_file),
        "notes_plan_path": str(notes_plan_file),
        "active_live_note_surfaces": final_live,
        "active_control_plane_surfaces": final_control,
        "active_runtime_artifact_surfaces": final_runtime_artifacts,
        "verification_only_artifact_paths": _norm_paths(
            [
                value
                for value in explicit_artifact_paths
                if Path(value).exists()
            ]
        ),
        "active_regime_control_plane_support_surfaces": final_support,
    }
