from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from vbtpro_lab.contracts import artifact_registry, local_build_registry, module_registry
from vbtpro_lab.paths import LIBRARY_DIR

from .models import MODULE_ID
from .registry import compiled_library_paths


DEFAULT_QUERY_LIMIT = 25
BUILD_REGISTRY_PATH = LIBRARY_DIR / "vbtpro_lab_build_registry_20260409.yaml"
BUILD_EVENT_LOG_PATH = LIBRARY_DIR / "vbtpro_lab_build_events_20260409.yaml"

TEXT_SOURCE_KIND_PRIORITY = {
    "external_runtime_export": 0,
    "external_repo_symbol": 1,
    "external_repo_file": 2,
    "repo_lookup": 3,
    "artifact": 4,
    "topic_route": 5,
    "truth_route": 6,
    "build_module": 7,
    "stage_record": 8,
    "truth_risk": 9,
    "replacement_record": 10,
    "code_map": 11,
    "query_alias": 12,
    "note_body": 20,
}


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return {}
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def default_db_path() -> str:
    return compiled_library_paths()["db_path"]


def build_registry_path() -> str:
    return str(BUILD_REGISTRY_PATH)


def build_event_log_path() -> str:
    return str(BUILD_EVENT_LOG_PATH)


def registered_artifact_paths() -> dict[str, str]:
    return {
        entry["artifact_role"]: entry["path"]
        for entry in artifact_registry()["registered_artifacts"]
    }


def lookup_registered_artifact(artifact_role: str) -> str:
    try:
        return registered_artifact_paths()[artifact_role]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact role: {artifact_role}") from exc


def module_registry_record(module_id: str = MODULE_ID) -> dict[str, Any]:
    for entry in module_registry()["modules"]:
        if entry["module_id"] == module_id:
            return entry
    raise KeyError(f"Unknown module_id: {module_id}")


def build_registry_record(module_id: str = MODULE_ID) -> dict[str, Any]:
    for entry in local_build_registry()["modules"]:
        if entry["module_id"] == module_id:
            return entry
    raise KeyError(f"Unknown module_id: {module_id}")


def open_library(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def text_source_kind_case_sql(alias: str = "source_kind") -> str:
    cases = " ".join(
        f"WHEN '{kind}' THEN {priority}"
        for kind, priority in TEXT_SOURCE_KIND_PRIORITY.items()
    )
    return f"CASE {alias} {cases} ELSE 50 END"


def simple_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return rows_to_dicts(conn.execute(sql, params))


def query_repo_path(conn: sqlite3.Connection, repo_path: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "repo_path",
        "routes": simple_rows(
            conn,
            """
            SELECT * FROM lookup_by_repo_path
            WHERE repo_path = ?
            ORDER BY priority_rank ASC, COALESCE(line_hint, 999999) ASC, target_path ASC
            LIMIT ?
            """,
            (repo_path, limit),
        ),
        "stage_links": simple_rows(
            conn,
            "SELECT DISTINCT stage_id, source_kind, source_id FROM repo_stage_links WHERE repo_path = ? ORDER BY stage_id, source_kind LIMIT ?",
            (repo_path, limit),
        ),
        "risk_links": simple_rows(
            conn,
            "SELECT DISTINCT risk_id, source_kind, severity FROM repo_truth_links WHERE repo_path = ? ORDER BY risk_id LIMIT ?",
            (repo_path, limit),
        ),
        "candidate_links": simple_rows(
            conn,
            "SELECT DISTINCT candidate_name, source_kind, source_id FROM repo_candidate_links WHERE repo_path = ? ORDER BY candidate_name LIMIT ?",
            (repo_path, limit),
        ),
    }


def query_symbol(conn: sqlite3.Connection, symbol: str, limit: int) -> dict[str, Any]:
    source_rank = "CASE WHEN source_kind = 'external_runtime_export' THEN 0 WHEN source_kind = 'external_repo_symbol' THEN 1 ELSE 5 END"
    exact_name_rank = "CASE WHEN symbol_name = ? THEN 0 WHEN qualname = ? THEN 0 WHEN qualname LIKE ? THEN 1 ELSE 2 END"
    return {
        "query_kind": "symbol",
        "rows": simple_rows(
            conn,
            """
            SELECT source_kind, symbol_name, kind, qualname, repo_path, target_path, line_number, module_name, section, priority_rank
            FROM lookup_by_symbol
            WHERE symbol_name = ?
               OR qualname = ?
               OR qualname LIKE ?
               OR symbol_name LIKE ?
            ORDER BY """
            + source_rank
            + """, """
            + exact_name_rank
            + """, priority_rank ASC, line_number ASC, repo_path ASC
            LIMIT ?
            """,
            (symbol, symbol, f"%.{symbol}", f"%{symbol}%", symbol, symbol, f"%.{symbol}", limit),
        ),
    }


def query_stage_id(conn: sqlite3.Connection, stage_id: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "stage_id",
        "stage_records": simple_rows(
            conn,
            "SELECT * FROM stage_records WHERE stage_id = ? LIMIT ?",
            (stage_id, limit),
        ),
        "linked_notes": simple_rows(
            conn,
            "SELECT * FROM lookup_by_stage_id WHERE stage_id = ? ORDER BY target_path, section LIMIT ?",
            (stage_id, limit),
        ),
        "repo_paths": simple_rows(
            conn,
            "SELECT DISTINCT repo_path, source_kind, source_id FROM repo_stage_links WHERE stage_id = ? ORDER BY repo_path LIMIT ?",
            (stage_id, limit),
        ),
        "truth_risks": simple_rows(
            conn,
            "SELECT DISTINCT risk_id FROM stage_truth_links WHERE stage_id = ? ORDER BY risk_id LIMIT ?",
            (stage_id, limit),
        ),
    }


def query_risk_id(conn: sqlite3.Connection, risk_id: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "risk_id",
        "risk_records": simple_rows(
            conn,
            "SELECT * FROM truth_risk_records WHERE risk_id = ? LIMIT ?",
            (risk_id, limit),
        ),
        "linked_notes": simple_rows(
            conn,
            "SELECT * FROM lookup_by_risk_id WHERE risk_id = ? ORDER BY target_path LIMIT ?",
            (risk_id, limit),
        ),
        "repo_paths": simple_rows(
            conn,
            "SELECT DISTINCT repo_path, source_kind, severity FROM repo_truth_links WHERE risk_id = ? ORDER BY repo_path LIMIT ?",
            (risk_id, limit),
        ),
        "stage_scope": simple_rows(
            conn,
            "SELECT DISTINCT stage_id FROM stage_truth_links WHERE risk_id = ? ORDER BY stage_id LIMIT ?",
            (risk_id, limit),
        ),
    }


def query_registry_id(conn: sqlite3.Connection, registry_id: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "registry_id",
        "registry_records": simple_rows(
            conn,
            "SELECT * FROM replacement_records WHERE registry_id = ? LIMIT ?",
            (registry_id, limit),
        ),
        "repo_paths": simple_rows(
            conn,
            "SELECT DISTINCT repo_path, candidate_name, source_kind FROM repo_candidate_links WHERE source_id = ? ORDER BY repo_path, candidate_name LIMIT ?",
            (registry_id, limit),
        ),
    }


def query_map_id(conn: sqlite3.Connection, map_id: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "map_id",
        "map_records": simple_rows(
            conn,
            "SELECT * FROM code_map_records WHERE map_id = ? LIMIT ?",
            (map_id, limit),
        ),
        "repo_paths": simple_rows(
            conn,
            "SELECT DISTINCT repo_path, stage_id, source_kind FROM repo_stage_links WHERE source_id = ? ORDER BY repo_path, stage_id LIMIT ?",
            (map_id, limit),
        ),
        "candidates": simple_rows(
            conn,
            "SELECT DISTINCT repo_path, candidate_name FROM repo_candidate_links WHERE source_id = ? ORDER BY candidate_name, repo_path LIMIT ?",
            (map_id, limit),
        ),
    }


def query_module_id(conn: sqlite3.Connection, module_id: str, limit: int) -> dict[str, Any]:
    return {
        "query_kind": "module_id",
        "module_records": simple_rows(
            conn,
            "SELECT * FROM build_module_records WHERE module_id = ? LIMIT ?",
            (module_id, limit),
        ),
        "stage_links": simple_rows(
            conn,
            "SELECT DISTINCT stage_id FROM module_stage_links WHERE module_id = ? ORDER BY stage_id LIMIT ?",
            (module_id, limit),
        ),
    }


def run_query(conn: sqlite3.Connection, kind: str, value: str, limit: int) -> dict[str, Any]:
    if kind == "artifact_path":
        return {
            "query_kind": "artifact_path",
            "rows": simple_rows(
                conn,
                "SELECT * FROM lookup_by_artifact_path WHERE artifact_path = ? LIMIT ?",
                (value, limit),
            ),
        }
    if kind == "repo_path":
        return query_repo_path(conn, value, limit)
    if kind == "symbol":
        return query_symbol(conn, value, limit)
    if kind == "topic":
        return {
            "query_kind": "topic",
            "rows": simple_rows(
                conn,
                "SELECT * FROM lookup_by_topic WHERE topic = ? ORDER BY COALESCE(line_hint, 999999), target_path LIMIT ?",
                (value, limit),
            ),
        }
    if kind == "truth_topic":
        return {
            "query_kind": "truth_topic",
            "rows": simple_rows(
                conn,
                "SELECT * FROM lookup_by_truth_topic WHERE truth_topic = ? ORDER BY COALESCE(line_hint, 999999), target_path LIMIT ?",
                (value, limit),
            ),
        }
    if kind == "notebook_surface":
        return {
            "query_kind": "notebook_surface",
            "rows": simple_rows(
                conn,
                "SELECT * FROM lookup_by_notebook_surface WHERE surface_name = ? ORDER BY target_path LIMIT ?",
                (value, limit),
            ),
        }
    if kind == "stage_id":
        return query_stage_id(conn, value, limit)
    if kind == "risk_id":
        return query_risk_id(conn, value, limit)
    if kind == "registry_id":
        return query_registry_id(conn, value, limit)
    if kind == "map_id":
        return query_map_id(conn, value, limit)
    if kind == "module_id":
        return query_module_id(conn, value, limit)
    if kind == "text":
        text_order_sql = text_source_kind_case_sql("sd.source_kind")
        try:
            rows = simple_rows(
                conn,
                """
                SELECT sd.source_kind, sd.primary_key, sd.secondary_key, sd.path, sd.section
                FROM search_documents_fts fts
                JOIN search_documents sd ON sd.doc_id = fts.rowid
                WHERE search_documents_fts MATCH ?
                ORDER BY """
                + text_order_sql
                + """, LENGTH(sd.path), sd.path
                LIMIT ?
                """,
                (value, limit),
            )
            if rows:
                return {"query_kind": "text", "rows": rows}
        except sqlite3.OperationalError:
            pass
        fallback_order_sql = text_source_kind_case_sql("source_kind")
        return {
            "query_kind": "text",
            "rows": simple_rows(
                conn,
                """
                SELECT source_kind, primary_key, secondary_key, path, section
                FROM search_documents
                WHERE body_text LIKE ?
                ORDER BY """
                + fallback_order_sql
                + """, LENGTH(path), path
                LIMIT ?
                """,
                (value, limit),
            )
        }
    raise ValueError(f"Unsupported query kind: {kind}")


def query_alias(conn: sqlite3.Connection, alias: str, limit: int) -> dict[str, Any]:
    alias_rows = simple_rows(
        conn,
        "SELECT * FROM query_aliases WHERE alias = ? ORDER BY alias_id LIMIT ?",
        (alias, limit),
    )
    resolved = []
    for row in alias_rows:
        resolved.append(
            {
                "alias": row,
                "result": run_query(conn, row["target_kind"], row["target_value"], limit),
            }
        )
    return {"query_kind": "alias", "rows": alias_rows, "resolved": resolved}


def query_machine_library(
    query_kind: str,
    value: str,
    *,
    limit: int = DEFAULT_QUERY_LIMIT,
    db_path: str | None = None,
) -> dict[str, Any]:
    with open_library(db_path) as conn:
        if query_kind == "alias":
            return query_alias(conn, value, limit)
        return run_query(conn, query_kind, value, limit)


def query_build_meta(*, db_path: str | None = None) -> dict[str, Any]:
    with open_library(db_path) as conn:
        return {
            "query_kind": "build_meta",
            "rows": simple_rows(conn, "SELECT key, value_json FROM build_meta ORDER BY key", ()),
        }


def query_from_namespace(args: Any) -> dict[str, Any]:
    limit = args.limit
    if args.alias:
        return query_machine_library("alias", args.alias, limit=limit, db_path=args.db_path)
    for arg_name, kind in [
        ("artifact_path", "artifact_path"),
        ("repo_path", "repo_path"),
        ("symbol", "symbol"),
        ("topic", "topic"),
        ("truth_topic", "truth_topic"),
        ("notebook_surface", "notebook_surface"),
        ("stage_id", "stage_id"),
        ("risk_id", "risk_id"),
        ("registry_id", "registry_id"),
        ("map_id", "map_id"),
        ("module_id", "module_id"),
        ("text", "text"),
    ]:
        value = getattr(args, arg_name)
        if value:
            return query_machine_library(kind, value, limit=limit, db_path=args.db_path)
    return query_build_meta(db_path=args.db_path)


def load_build_events(path: str | Path | None = None) -> dict[str, Any]:
    event_path = Path(path) if path is not None else BUILD_EVENT_LOG_PATH
    if not event_path.exists():
        return {
            "artifact_id": "vbtpro_lab_build_events_20260409",
            "artifact_version": 1,
            "artifact_type": "build_event_log",
            "status": "active_next",
            "created_on": "2026-04-09",
            "owner": "codex",
            "events": [],
        }
    return _read_yaml(event_path)


def append_build_event(
    *,
    module_id: str,
    stage_id: str,
    event_kind: str,
    summary: str,
    new_build_status: str | None = None,
    details: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    event_path = Path(path) if path is not None else BUILD_EVENT_LOG_PATH
    payload = load_build_events(event_path)
    events = payload.setdefault("events", [])
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    event = {
        "event_id": f"BLE{len(events) + 1:03d}",
        "module_id": module_id,
        "stage_id": stage_id,
        "event_kind": event_kind,
        "timestamp_utc": timestamp,
        "summary": summary,
    }
    if new_build_status is not None:
        event["new_build_status"] = new_build_status
    if details:
        event["details"] = details
    events.append(event)
    _write_yaml(event_path, payload)
    return event


def update_build_registry_module_status(
    *,
    module_id: str,
    build_status: str,
    summary: str,
    stage_id: str,
    blockers: list[str] | None = None,
    registry_path: str | Path | None = None,
    event_log_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(registry_path) if registry_path is not None else BUILD_REGISTRY_PATH
    payload = _read_yaml(path)
    modules = payload.get("modules", [])
    for entry in modules:
        if entry["module_id"] != module_id:
            continue
        previous_status = entry["build_status"]
        entry["build_status"] = build_status
        if blockers is not None:
            entry["blockers"] = blockers
        _write_yaml(path, payload)
        event = append_build_event(
            module_id=module_id,
            stage_id=stage_id,
            event_kind="build_status_updated",
            summary=summary,
            new_build_status=build_status,
            details={"previous_build_status": previous_status},
            path=event_log_path,
        )
        return {"module_record": entry, "event": event}
    raise KeyError(f"Unknown module_id: {module_id}")
