#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import subprocess
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vbtpro_lab.machine_library.note_sync import sync_repo_truth_note_surfaces


DEFAULT_DB = ROOT / "reports/strategy_factory/vbtpro_machine_library_20260409.sqlite"
DEFAULT_MANIFEST = ROOT / "reports/strategy_factory/vbtpro_machine_library_build_manifest_20260409.yaml"
DEFAULT_EXAMPLES = ROOT / "reports/strategy_factory/vbtpro_machine_library_query_examples_20260409.yaml"
DEFAULT_DASHBOARD = ROOT / "reports/strategy_factory/vbtpro_library_coverage_dashboard_20260409.yaml"
MASTER_INDEX = ROOT / "reports/strategy_factory/vbtpro_research_master_index_20260408.yaml"
TOPIC_ROUTER = ROOT / "reports/strategy_factory/vbtpro_topic_router_20260408.yaml"
REPO_PATH_ROUTER = ROOT / "reports/strategy_factory/vbtpro_repo_path_router_20260408.yaml"
TRUTH_ROUTER = ROOT / "reports/strategy_factory/vbtpro_truth_honesty_router_20260408.yaml"
NOTEBOOK_ROUTER = ROOT / "reports/strategy_factory/vbtpro_notebook_script_test_index_20260408.yaml"
STAGE_CONTRACTS = ROOT / "reports/strategy_factory/vbtpro_native_pipeline_stage_contracts_20260409.yaml"
TRUTH_RISK_MATRIX = ROOT / "reports/strategy_factory/vbtpro_truth_risk_test_matrix_20260409.yaml"
REPLACEMENT_REGISTRY = ROOT / "reports/strategy_factory/vbtpro_repo_replacement_registry_20260409.yaml"
CODE_LEVEL_MAP = ROOT / "reports/strategy_factory/vbtpro_repo_code_level_map_20260409.yaml"
ALIAS_REGISTRY = ROOT / "reports/strategy_factory/vbtpro_query_alias_registry_20260409.yaml"
LAB_BUILD_REGISTRY = ROOT / "reports/strategy_factory/vbtpro_lab_build_registry_20260409.yaml"
RESET_REBUILD_NOTES_PLAN = ROOT / "artifacts/specs/vbtpro_lab_reset_rebuild_notes_plan_20260409.yaml"
BATCH_RECORDS = ROOT / "reports/strategy_factory/vbtpro_capability_batch_records_20260408.yaml"
COMPLETION_CONTRACT = ROOT / "reports/strategy_factory/vbtpro_library_completion_contract_20260409_v2.yaml"
LAB_MODULE_REGISTRY = ROOT / "contracts/module_registry.yaml"
ARTIFACT_REGISTRY = ROOT / "contracts/artifact_registry.yaml"
EXTERNAL_VECTORBTPRO_LIBRARY_INDEX = (
    ROOT / "artifacts/specs/control_plane/vbtpro_external_vectorbtpro_library_index_20260416.yaml"
)
EXTERNAL_VECTORBTPRO_REPO_ROOT = ROOT / "external/vectorbt.pro"
EXTERNAL_VECTORBTPRO_SOURCE_ROOT = EXTERNAL_VECTORBTPRO_REPO_ROOT / "vectorbtpro"
LEGACY_REPO_ROOT = Path("/home/benji/trade_scanner2")
LEGACY_DESKTOP_VECTORBTPRO_ROOT = Path("/home/benji/Desktop/vectorbt.pro-main")
LEGACY_EXTERNAL_VECTORBTPRO_ROOT = Path("/media/benji/INDIA_SCRATCH/mnq-lab/external/vectorbt.pro")


class MachineLibraryLockError(RuntimeError):
    pass


def localize_path_string(value: str) -> str:
    if not value.startswith("/"):
        return value
    path = Path(value)
    if value.startswith(str(LEGACY_REPO_ROOT)):
        try:
            candidate = ROOT / path.relative_to(LEGACY_REPO_ROOT)
        except ValueError:
            candidate = None
        if candidate is not None and candidate.exists():
            return str(candidate)
    for old_root in (LEGACY_DESKTOP_VECTORBTPRO_ROOT, LEGACY_EXTERNAL_VECTORBTPRO_ROOT):
        if value.startswith(str(old_root)):
            try:
                candidate = EXTERNAL_VECTORBTPRO_REPO_ROOT / path.relative_to(old_root)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.exists():
                return str(candidate)
    return value


def localize_yaml_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: localize_yaml_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [localize_yaml_paths(value) for value in payload]
    if isinstance(payload, str):
        return localize_path_string(payload)
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    return localize_yaml_paths(payload)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def external_vectorbtpro_repo_files() -> list[Path]:
    files = sorted(EXTERNAL_VECTORBTPRO_SOURCE_ROOT.rglob("*.py"))
    for relative_path in ("README.md", "pyproject.toml"):
        candidate = EXTERNAL_VECTORBTPRO_REPO_ROOT / relative_path
        if candidate.exists():
            files.append(candidate)
    return files


def external_vectorbtpro_relative_path(path: Path) -> str:
    return str(path.relative_to(EXTERNAL_VECTORBTPRO_REPO_ROOT))


def external_vectorbtpro_section(path: Path) -> str:
    relative = path.relative_to(EXTERNAL_VECTORBTPRO_REPO_ROOT)
    if len(relative.parts) >= 2 and relative.parts[0] == "vectorbtpro":
        return relative.parts[1]
    return relative.parts[0]


def external_vectorbtpro_module_name(path: Path) -> str | None:
    if path.suffix != ".py":
        return None
    relative = path.relative_to(EXTERNAL_VECTORBTPRO_REPO_ROOT)
    module_parts = list(relative.with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts) if module_parts else None


def external_vectorbtpro_python_modules() -> list[tuple[str, Path]]:
    modules: list[tuple[str, Path]] = []
    for path in external_vectorbtpro_repo_files():
        if path.suffix != ".py":
            continue
        module_name = external_vectorbtpro_module_name(path)
        if module_name:
            modules.append((module_name, path))
    return modules


def external_vectorbtpro_symbol_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix != ".py":
        return []
    module_name = external_vectorbtpro_module_name(path)
    if not module_name:
        return []
    relative_path = external_vectorbtpro_relative_path(path)
    section = external_vectorbtpro_section(path)
    try:
        tree = ast.parse(safe_read_text(path), filename=str(path))
    except SyntaxError:
        return []

    records: list[dict[str, Any]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            kind: str | None = None
            if isinstance(child, ast.ClassDef):
                kind = "class"
            elif isinstance(child, ast.AsyncFunctionDef):
                kind = "async_function"
            elif isinstance(child, ast.FunctionDef):
                kind = "function"
            if kind is not None:
                qual_parts = [module_name, *parents, child.name]
                records.append(
                    {
                        "symbol_name": child.name,
                        "kind": kind,
                        "qualname": ".".join(qual_parts),
                        "repo_path": str(path),
                        "relative_path": relative_path,
                        "line_number": getattr(child, "lineno", 1),
                        "module_name": module_name,
                        "section": section,
                    }
                )
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return records


def external_vectorbtpro_runtime_export_payload() -> dict[str, Any]:
    modules = [module_name for module_name, _ in external_vectorbtpro_python_modules()]
    if not modules:
        return {"records": [], "failures": []}
    script = """
import importlib
import inspect
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
module_names = sys.argv[2:]
sys.path.insert(0, str(repo_root))
records = []
failures = []

def classify(obj):
    if inspect.isclass(obj):
        return "class"
    if inspect.iscoroutinefunction(obj):
        return "async_function"
    if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if callable(obj):
        return "callable"
    return None

for module_name in module_names:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        failures.append(
            {
                "module_name": module_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        continue
    export_names = getattr(module, "__all__", None)
    if not isinstance(export_names, (list, tuple, set)):
        export_names = [name for name in vars(module) if not name.startswith("_")]
    seen = set()
    for export_name in export_names:
        if not isinstance(export_name, str) or export_name in seen:
            continue
        seen.add(export_name)
        try:
            obj = getattr(module, export_name)
        except Exception:
            continue
        kind = classify(obj)
        if kind is None:
            continue
        object_module = getattr(obj, "__module__", module.__name__)
        object_qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", export_name))
        export_qualname = f"{module.__name__}.{export_name}"
        try:
            source_path = inspect.getsourcefile(obj) or inspect.getfile(obj)
        except Exception:
            source_path = getattr(module, "__file__", None)
        if source_path is None:
            continue
        source_path = str(Path(source_path).resolve())
        if not source_path.startswith(str(repo_root)):
            continue
        try:
            _, line_number = inspect.getsourcelines(obj)
        except Exception:
            line_number = 1
        if module.__name__.startswith("vectorbtpro.") and module.__name__.count(".") >= 1:
            section = module.__name__.split(".")[1]
        else:
            section = module.__name__.split(".")[0]
        records.append(
            {
                "symbol_name": export_name,
                "kind": kind,
                "qualname": export_qualname,
                "repo_path": source_path,
                "line_number": int(line_number),
                "module_name": module.__name__,
                "section": section,
                "object_module": object_module,
                "object_qualname": object_qualname,
            }
        )

print(json.dumps({"records": records, "failures": failures}, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(EXTERNAL_VECTORBTPRO_REPO_ROOT), *modules],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    records = payload.get("records") or []
    records.sort(key=lambda row: (row["qualname"], row["line_number"], row["repo_path"]))
    payload["records"] = records
    return payload


def temp_db_path_for(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}.tmp")


def lock_path_for(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}.lock")


@contextmanager
def machine_library_lock(db_path: Path, operation: str):
    lock_path = lock_path_for(db_path)
    ensure_parent(lock_path)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MachineLibraryLockError(
                f"machine-library lock is busy; refuse to run {operation} in parallel"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(json.dumps({"operation": operation, "db_path": str(db_path)}, sort_keys=True))
        lock_file.flush()
        try:
            yield lock_path
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def priority_rank(priority: str | None) -> int:
    mapping = {"highest": 0, "high": 1, "medium": 2, "low": 3, None: 9}
    return mapping.get(priority, 9)


def insert_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO build_meta(key, value_json) VALUES (?, ?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def read_meta(conn: sqlite3.Connection, key: str) -> dict[str, Any]:
    row = conn.execute("SELECT value_json FROM build_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return {}
    return json.loads(row[0])


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        DROP TABLE IF EXISTS build_meta;
        DROP TABLE IF EXISTS source_files;
        DROP TABLE IF EXISTS artifacts;
        DROP TABLE IF EXISTS quick_start_routes;
        DROP TABLE IF EXISTS topic_routes;
        DROP TABLE IF EXISTS topic_repo_targets;
        DROP TABLE IF EXISTS repo_path_routes;
        DROP TABLE IF EXISTS repo_support_sections;
        DROP TABLE IF EXISTS truth_routes;
        DROP TABLE IF EXISTS truth_repo_targets;
        DROP TABLE IF EXISTS notebook_surfaces;
        DROP TABLE IF EXISTS notebook_note_refs;
        DROP TABLE IF EXISTS notebook_repo_targets;
        DROP TABLE IF EXISTS stage_records;
        DROP TABLE IF EXISTS truth_risk_records;
        DROP TABLE IF EXISTS replacement_records;
        DROP TABLE IF EXISTS code_map_records;
        DROP TABLE IF EXISTS build_module_records;
        DROP TABLE IF EXISTS query_aliases;
        DROP TABLE IF EXISTS repo_stage_links;
        DROP TABLE IF EXISTS repo_truth_links;
        DROP TABLE IF EXISTS repo_candidate_links;
        DROP TABLE IF EXISTS stage_truth_links;
        DROP TABLE IF EXISTS module_stage_links;
        DROP TABLE IF EXISTS lookup_by_repo_path;
        DROP TABLE IF EXISTS lookup_by_topic;
        DROP TABLE IF EXISTS lookup_by_truth_topic;
        DROP TABLE IF EXISTS lookup_by_notebook_surface;
        DROP TABLE IF EXISTS lookup_by_artifact_path;
        DROP TABLE IF EXISTS lookup_by_stage_id;
        DROP TABLE IF EXISTS lookup_by_risk_id;
        DROP TABLE IF EXISTS lookup_by_alias;
        DROP TABLE IF EXISTS external_repo_symbols;
        DROP TABLE IF EXISTS lookup_by_symbol;
        DROP TABLE IF EXISTS note_section_index;
        DROP TABLE IF EXISTS search_documents;
        DROP TABLE IF EXISTS search_documents_fts;

        CREATE TABLE build_meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE source_files (
            path TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            mtime REAL NOT NULL,
            bytes INTEGER NOT NULL
        );
        CREATE TABLE artifacts (
            artifact_path TEXT NOT NULL,
            artifact_kind TEXT,
            role TEXT,
            priority TEXT,
            line_hint INTEGER,
            section TEXT,
            source_group TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_artifacts_path ON artifacts(artifact_path);
        CREATE INDEX idx_artifacts_kind ON artifacts(artifact_kind);
        CREATE INDEX idx_artifacts_priority ON artifacts(priority);

        CREATE TABLE quick_start_routes (
            route_group TEXT NOT NULL,
            route_name TEXT NOT NULL,
            ord INTEGER NOT NULL,
            path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_qsr_route ON quick_start_routes(route_group, route_name);

        CREATE TABLE topic_routes (
            topic TEXT NOT NULL,
            goal TEXT,
            path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_topic_routes ON topic_routes(topic);

        CREATE TABLE topic_repo_targets (
            topic TEXT NOT NULL,
            repo_target TEXT NOT NULL
        );
        CREATE INDEX idx_topic_repo_targets ON topic_repo_targets(topic, repo_target);

        CREATE TABLE repo_path_routes (
            repo_group TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            first_note TEXT NOT NULL,
            first_section_line INTEGER
        );
        CREATE INDEX idx_repo_path_routes ON repo_path_routes(repo_path);

        CREATE TABLE repo_support_sections (
            repo_path TEXT NOT NULL,
            support_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_repo_support_sections ON repo_support_sections(repo_path);

        CREATE TABLE truth_routes (
            truth_topic TEXT NOT NULL,
            goal TEXT,
            path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_truth_routes ON truth_routes(truth_topic);

        CREATE TABLE truth_repo_targets (
            truth_topic TEXT NOT NULL,
            repo_target TEXT NOT NULL
        );
        CREATE INDEX idx_truth_repo_targets ON truth_repo_targets(truth_topic, repo_target);

        CREATE TABLE notebook_surfaces (
            surface_name TEXT NOT NULL,
            route_kind TEXT NOT NULL,
            artifact_path TEXT NOT NULL
        );
        CREATE INDEX idx_notebook_surfaces ON notebook_surfaces(surface_name, route_kind);

        CREATE TABLE notebook_note_refs (
            surface_name TEXT NOT NULL,
            note_path TEXT NOT NULL
        );
        CREATE INDEX idx_notebook_note_refs ON notebook_note_refs(surface_name, note_path);

        CREATE TABLE notebook_repo_targets (
            surface_name TEXT NOT NULL,
            repo_target TEXT NOT NULL
        );
        CREATE INDEX idx_notebook_repo_targets ON notebook_repo_targets(surface_name, repo_target);

        CREATE TABLE stage_records (
            stage_id TEXT PRIMARY KEY,
            stage_kind TEXT,
            purpose TEXT,
            cpu_role TEXT,
            gpu_role TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE truth_risk_records (
            risk_id TEXT PRIMARY KEY,
            blocking_severity TEXT,
            failure_mode TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_truth_risk_severity ON truth_risk_records(blocking_severity);

        CREATE TABLE replacement_records (
            registry_id TEXT PRIMARY KEY,
            adoption_mode TEXT,
            risk_level TEXT,
            current_role TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_replacement_adoption ON replacement_records(adoption_mode);

        CREATE TABLE code_map_records (
            map_id TEXT PRIMARY KEY,
            primary_path TEXT,
            adoption_mode TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_code_map_primary_path ON code_map_records(primary_path);

        CREATE TABLE build_module_records (
            module_id TEXT PRIMARY KEY,
            module_path TEXT,
            build_status TEXT,
            rollout_phase TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX idx_build_module_status ON build_module_records(build_status);

        CREATE TABLE query_aliases (
            alias TEXT NOT NULL,
            alias_id TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            target_value TEXT NOT NULL,
            description TEXT
        );
        CREATE INDEX idx_query_aliases_alias ON query_aliases(alias);

        CREATE TABLE repo_stage_links (
            repo_path TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL
        );
        CREATE INDEX idx_repo_stage_links ON repo_stage_links(repo_path, stage_id);

        CREATE TABLE repo_truth_links (
            repo_path TEXT NOT NULL,
            risk_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            severity TEXT
        );
        CREATE INDEX idx_repo_truth_links ON repo_truth_links(repo_path, risk_id);

        CREATE TABLE repo_candidate_links (
            repo_path TEXT NOT NULL,
            candidate_name TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL
        );
        CREATE INDEX idx_repo_candidate_links ON repo_candidate_links(repo_path, candidate_name);

        CREATE TABLE stage_truth_links (
            stage_id TEXT NOT NULL,
            risk_id TEXT NOT NULL
        );
        CREATE INDEX idx_stage_truth_links ON stage_truth_links(stage_id, risk_id);

        CREATE TABLE module_stage_links (
            module_id TEXT NOT NULL,
            stage_id TEXT NOT NULL
        );
        CREATE INDEX idx_module_stage_links ON module_stage_links(module_id, stage_id);

        CREATE TABLE lookup_by_repo_path (
            repo_path TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_key TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT,
            priority_rank INTEGER NOT NULL
        );
        CREATE INDEX idx_lookup_repo_path ON lookup_by_repo_path(repo_path, priority_rank, source_kind);

        CREATE TABLE lookup_by_topic (
            topic TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_lookup_topic ON lookup_by_topic(topic, source_kind);

        CREATE TABLE lookup_by_truth_topic (
            truth_topic TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_lookup_truth_topic ON lookup_by_truth_topic(truth_topic, source_kind);

        CREATE TABLE lookup_by_notebook_surface (
            surface_name TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_lookup_notebook_surface ON lookup_by_notebook_surface(surface_name, source_kind);

        CREATE TABLE lookup_by_artifact_path (
            artifact_path TEXT NOT NULL,
            artifact_kind TEXT,
            role TEXT,
            priority TEXT
        );
        CREATE INDEX idx_lookup_artifact_path ON lookup_by_artifact_path(artifact_path);

        CREATE TABLE lookup_by_stage_id (
            stage_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_lookup_stage_id ON lookup_by_stage_id(stage_id, source_kind);

        CREATE TABLE lookup_by_risk_id (
            risk_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_hint INTEGER,
            section TEXT
        );
        CREATE INDEX idx_lookup_risk_id ON lookup_by_risk_id(risk_id, source_kind);

        CREATE TABLE lookup_by_alias (
            alias TEXT NOT NULL,
            alias_id TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            target_value TEXT NOT NULL
        );
        CREATE INDEX idx_lookup_alias ON lookup_by_alias(alias);

        CREATE TABLE external_repo_symbols (
            symbol_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            qualname TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            module_name TEXT NOT NULL,
            section TEXT NOT NULL
        );
        CREATE INDEX idx_external_repo_symbols_name ON external_repo_symbols(symbol_name);
        CREATE INDEX idx_external_repo_symbols_qualname ON external_repo_symbols(qualname);
        CREATE INDEX idx_external_repo_symbols_repo_path ON external_repo_symbols(repo_path, line_number);

        CREATE TABLE external_runtime_exports (
            symbol_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            qualname TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            module_name TEXT NOT NULL,
            section TEXT NOT NULL,
            object_module TEXT NOT NULL,
            object_qualname TEXT NOT NULL
        );
        CREATE INDEX idx_external_runtime_exports_name ON external_runtime_exports(symbol_name);
        CREATE INDEX idx_external_runtime_exports_qualname ON external_runtime_exports(qualname);
        CREATE INDEX idx_external_runtime_exports_repo_path ON external_runtime_exports(repo_path, line_number);

        CREATE TABLE lookup_by_symbol (
            source_kind TEXT NOT NULL,
            symbol_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            qualname TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            target_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            module_name TEXT NOT NULL,
            section TEXT NOT NULL,
            priority_rank INTEGER NOT NULL
        );
        CREATE INDEX idx_lookup_symbol_name ON lookup_by_symbol(symbol_name, source_kind, priority_rank, line_number);
        CREATE INDEX idx_lookup_symbol_qualname ON lookup_by_symbol(qualname, source_kind, priority_rank, line_number);

        CREATE TABLE note_section_index (
            note_path TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            indent INTEGER NOT NULL,
            key_name TEXT NOT NULL
        );
        CREATE INDEX idx_note_section_index ON note_section_index(note_path, key_name);

        CREATE TABLE search_documents (
            doc_id INTEGER PRIMARY KEY,
            source_kind TEXT NOT NULL,
            primary_key TEXT NOT NULL,
            secondary_key TEXT,
            path TEXT NOT NULL,
            section TEXT,
            body_text TEXT NOT NULL
        );
        """
    )
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE search_documents_fts USING fts5(
                body_text, path, section, source_kind, primary_key, secondary_key,
                content='search_documents', content_rowid='doc_id'
            );
            """
        )
        conn.execute("INSERT INTO search_documents_fts(search_documents_fts) VALUES ('rebuild')")
    except sqlite3.OperationalError:
        pass


def add_source_files(conn: sqlite3.Connection, sources: dict[Path, str]) -> None:
    for path, role in sources.items():
        stat = path.stat()
        conn.execute(
            "INSERT INTO source_files(path, role, sha256, mtime, bytes) VALUES (?, ?, ?, ?, ?)",
            (str(path), role, sha256_file(path), stat.st_mtime, stat.st_size),
        )


def extract_note_sections(path: Path) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    pattern = re.compile(r"^(\s*)([A-Za-z0-9_]+):\s*$")
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        match = pattern.match(line)
        if match:
            rows.append((str(path), line_no, len(match.group(1)), match.group(2)))
    return rows


def load_master_index(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    artifact_catalog = data["artifact_catalog"]
    for source_group in ("primary_authority_refs", "vbtpro_research_refs"):
        for entry in artifact_catalog.get(source_group, []):
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_path, artifact_kind, role, priority, line_hint, section, source_group, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["artifact_path"],
                    entry.get("artifact_kind"),
                    entry.get("role"),
                    entry.get("priority"),
                    entry.get("line_hint"),
                    entry.get("section"),
                    source_group,
                    json.dumps(entry, sort_keys=True),
                ),
            )
            conn.execute(
                "INSERT INTO lookup_by_artifact_path(artifact_path, artifact_kind, role, priority) VALUES (?, ?, ?, ?)",
                (
                    entry["artifact_path"],
                    entry.get("artifact_kind"),
                    entry.get("role"),
                    entry.get("priority"),
                ),
            )

    for route_group, route_map in data["quick_start_routes"]["if_you_want"].items():
        for ord_idx, item in enumerate(route_map.get("read_order", []), start=1):
            conn.execute(
                "INSERT INTO quick_start_routes(route_group, route_name, ord, path, line_hint, section) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "if_you_want",
                    route_group,
                    ord_idx,
                    item["path"],
                    item.get("line_hint"),
                    item.get("section"),
                ),
            )
    return data


def load_topic_router(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for topic, entry in data["topics"].items():
        for ref in entry.get("primary_refs", []):
            conn.execute(
                "INSERT INTO topic_routes(topic, goal, path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (topic, entry.get("goal"), ref["path"], ref.get("line_hint"), ref.get("section")),
            )
            conn.execute(
                "INSERT INTO lookup_by_topic(topic, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (topic, "primary_ref", ref["path"], ref.get("line_hint"), ref.get("section")),
            )
        for repo_target in entry.get("repo_targets", []):
            conn.execute(
                "INSERT INTO topic_repo_targets(topic, repo_target) VALUES (?, ?)",
                (topic, repo_target),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_target, "topic", topic, str(path), 1, topic, 1),
            )
    return data


def load_repo_path_router(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for repo_group, entry in data["repo_groups"].items():
        for file_entry in entry.get("files", []):
            repo_path = file_entry["repo_path"]
            conn.execute(
                "INSERT INTO repo_path_routes(repo_group, repo_path, first_note, first_section_line) VALUES (?, ?, ?, ?)",
                (repo_group, repo_path, file_entry["first_note"], file_entry.get("first_section_line")),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_path,
                    "first_note",
                    repo_group,
                    file_entry["first_note"],
                    file_entry.get("first_section_line"),
                    None,
                    0,
                ),
            )
            for support in file_entry.get("supporting_sections", []):
                conn.execute(
                    "INSERT INTO repo_support_sections(repo_path, support_path, line_hint, section) VALUES (?, ?, ?, ?)",
                    (repo_path, support["path"], support.get("line_hint"), support.get("section")),
                )
                conn.execute(
                    """
                    INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        repo_path,
                        "support_section",
                        repo_group,
                        support["path"],
                        support.get("line_hint"),
                        support.get("section"),
                        2,
                    ),
                )
    return data


def load_truth_router(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for truth_topic, entry in data["topics"].items():
        for ref in entry.get("primary_refs", []):
            conn.execute(
                "INSERT INTO truth_routes(truth_topic, goal, path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (truth_topic, entry.get("goal"), ref["path"], ref.get("line_hint"), ref.get("section")),
            )
            conn.execute(
                "INSERT INTO lookup_by_truth_topic(truth_topic, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (truth_topic, "primary_ref", ref["path"], ref.get("line_hint"), ref.get("section")),
            )
        for repo_target in entry.get("repo_targets", []):
            conn.execute(
                "INSERT INTO truth_repo_targets(truth_topic, repo_target) VALUES (?, ?)",
                (truth_topic, repo_target),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_target, "truth_topic", truth_topic, str(path), 1, truth_topic, 1),
            )
    return data


def load_notebook_router(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for surface_name, entry in data["surfaces"].items():
        for route_kind in ("primary_artifacts", "planned_surfaces"):
            for artifact_path in entry.get(route_kind, []):
                conn.execute(
                    "INSERT INTO notebook_surfaces(surface_name, route_kind, artifact_path) VALUES (?, ?, ?)",
                    (surface_name, route_kind, artifact_path),
                )
                conn.execute(
                    "INSERT INTO lookup_by_notebook_surface(surface_name, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                    (surface_name, route_kind, artifact_path, None, None),
                )
        for note_path in entry.get("primary_note_refs", []):
            conn.execute(
                "INSERT INTO notebook_note_refs(surface_name, note_path) VALUES (?, ?)",
                (surface_name, note_path),
            )
            conn.execute(
                "INSERT INTO lookup_by_notebook_surface(surface_name, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (surface_name, "primary_note_ref", note_path, None, None),
            )
        for repo_target in entry.get("repo_targets", []):
            conn.execute(
                "INSERT INTO notebook_repo_targets(surface_name, repo_target) VALUES (?, ?)",
                (surface_name, repo_target),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_target, "notebook_surface", surface_name, str(path), 1, surface_name, 1),
            )
    return data


def load_stage_contracts(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for stage_id in data.get("stage_order", []):
        entry = data["stages"][stage_id]
        conn.execute(
            """
            INSERT INTO stage_records(stage_id, stage_kind, purpose, cpu_role, gpu_role, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                stage_id,
                entry.get("stage_kind"),
                entry.get("purpose"),
                entry.get("cpu_gpu_role", {}).get("cpu_role"),
                entry.get("cpu_gpu_role", {}).get("gpu_role"),
                json.dumps(entry, sort_keys=True),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_stage_id(stage_id, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
            (stage_id, "stage_contract", str(path), 1, stage_id),
        )
        for repo_target in entry.get("primary_repo_targets", []):
            conn.execute(
                "INSERT INTO repo_stage_links(repo_path, stage_id, source_kind, source_id) VALUES (?, ?, ?, ?)",
                (repo_target, stage_id, "stage_contract", stage_id),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_target, "stage_contract", stage_id, str(path), 1, stage_id, 1),
            )
    return data


def load_truth_risk_matrix(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for entry in data.get("risk_matrix", []):
        risk_id = entry["risk_id"]
        conn.execute(
            """
            INSERT INTO truth_risk_records(risk_id, blocking_severity, failure_mode, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                risk_id,
                entry.get("blocking_severity"),
                entry.get("failure_mode"),
                json.dumps(entry, sort_keys=True),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_risk_id(risk_id, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
            (risk_id, "truth_risk", str(path), 1, risk_id),
        )
        for stage_id in entry.get("stage_scope", []):
            conn.execute(
                "INSERT INTO stage_truth_links(stage_id, risk_id) VALUES (?, ?)",
                (stage_id, risk_id),
            )
            conn.execute(
                "INSERT INTO lookup_by_stage_id(stage_id, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (stage_id, "truth_risk", str(path), 1, risk_id),
            )
        for repo_target in entry.get("repo_targets", []):
            conn.execute(
                "INSERT INTO repo_truth_links(repo_path, risk_id, source_kind, severity) VALUES (?, ?, ?, ?)",
                (repo_target, risk_id, "truth_risk", entry.get("blocking_severity")),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_target, "truth_risk", risk_id, str(path), 1, risk_id, 1),
            )
    return data


def load_replacement_registry(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for entry in data.get("entries", []):
        registry_id = entry["registry_id"]
        conn.execute(
            """
            INSERT INTO replacement_records(registry_id, adoption_mode, risk_level, current_role, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                registry_id,
                entry.get("adoption_mode"),
                entry.get("risk_level"),
                entry.get("current_role"),
                json.dumps(entry, sort_keys=True),
            ),
        )
        for repo_path in entry.get("repo_paths", []):
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_path, "replacement_registry", registry_id, str(path), 1, registry_id, 1),
            )
            for candidate in entry.get("vbt_candidates", []):
                conn.execute(
                    "INSERT INTO repo_candidate_links(repo_path, candidate_name, source_kind, source_id) VALUES (?, ?, ?, ?)",
                    (repo_path, candidate, "replacement_registry", registry_id),
                )
    return data


def load_code_level_map(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for entry in data.get("entries", []):
        map_id = entry["map_id"]
        repo_paths = []
        if entry.get("path"):
            repo_paths.append(entry["path"])
        repo_paths.extend(entry.get("paths", []))
        primary_path = repo_paths[0] if repo_paths else None
        conn.execute(
            """
            INSERT INTO code_map_records(map_id, primary_path, adoption_mode, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                map_id,
                primary_path,
                entry.get("adoption_mode"),
                json.dumps(entry, sort_keys=True),
            ),
        )
        for repo_path in repo_paths:
            conn.execute(
                """
                INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_path, "code_map", map_id, str(path), 1, map_id, 1),
            )
            for stage_id in entry.get("stage_ids", []):
                conn.execute(
                    "INSERT INTO repo_stage_links(repo_path, stage_id, source_kind, source_id) VALUES (?, ?, ?, ?)",
                    (repo_path, stage_id, "code_map", map_id),
                )
            for candidate in entry.get("vbt_candidates", []):
                conn.execute(
                    "INSERT INTO repo_candidate_links(repo_path, candidate_name, source_kind, source_id) VALUES (?, ?, ?, ?)",
                    (repo_path, candidate, "code_map", map_id),
                )
        for stage_id in entry.get("stage_ids", []):
            conn.execute(
                "INSERT INTO lookup_by_stage_id(stage_id, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (stage_id, "code_map", str(path), 1, map_id),
            )
    return data


def load_alias_registry(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    data = read_yaml(path)
    for entry in data.get("entries", []):
        conn.execute(
            """
            INSERT INTO query_aliases(alias, alias_id, target_kind, target_value, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry["alias"],
                entry["alias_id"],
                entry["target_kind"],
                entry["target_value"],
                entry.get("description"),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_alias(alias, alias_id, target_kind, target_value) VALUES (?, ?, ?, ?)",
            (entry["alias"], entry["alias_id"], entry["target_kind"], entry["target_value"]),
        )
    return data


def load_external_vectorbtpro_repo(conn: sqlite3.Connection, index_path: Path) -> dict[str, Any]:
    data = read_yaml(index_path)
    if not EXTERNAL_VECTORBTPRO_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Missing external vectorbtpro source root: {EXTERNAL_VECTORBTPRO_SOURCE_ROOT}")
    for file_path in external_vectorbtpro_repo_files():
        repo_path = str(file_path)
        relative_path = external_vectorbtpro_relative_path(file_path)
        section = external_vectorbtpro_section(file_path)
        repo_group = f"external_vectorbtpro::{section}"
        conn.execute(
            "INSERT INTO repo_path_routes(repo_group, repo_path, first_note, first_section_line) VALUES (?, ?, ?, ?)",
            (repo_group, repo_path, str(index_path), 1),
        )
        conn.execute(
            """
            INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (repo_path, "external_repo_file", relative_path, repo_path, 1, relative_path, 0),
        )
        conn.execute(
            """
            INSERT INTO lookup_by_repo_path(repo_path, source_kind, source_key, target_path, line_hint, section, priority_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (repo_path, "external_repo_index", repo_group, str(index_path), 1, section, 1),
        )
        for symbol in external_vectorbtpro_symbol_records(file_path):
            conn.execute(
                """
                INSERT INTO external_repo_symbols(symbol_name, kind, qualname, repo_path, line_number, module_name, section)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol["symbol_name"],
                    symbol["kind"],
                    symbol["qualname"],
                    symbol["repo_path"],
                    symbol["line_number"],
                    symbol["module_name"],
                    symbol["section"],
                ),
            )
            conn.execute(
                """
                INSERT INTO lookup_by_symbol(source_kind, symbol_name, kind, qualname, repo_path, target_path, line_number, module_name, section, priority_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "external_repo_symbol",
                    symbol["symbol_name"],
                    symbol["kind"],
                    symbol["qualname"],
                    symbol["repo_path"],
                    symbol["repo_path"],
                    symbol["line_number"],
                    symbol["module_name"],
                    symbol["section"],
                    0,
                ),
            )
    runtime_export_payload = external_vectorbtpro_runtime_export_payload()
    for export in runtime_export_payload.get("records", []):
        conn.execute(
            """
            INSERT INTO external_runtime_exports(symbol_name, kind, qualname, repo_path, line_number, module_name, section, object_module, object_qualname)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                export["symbol_name"],
                export["kind"],
                export["qualname"],
                export["repo_path"],
                export["line_number"],
                export["module_name"],
                export["section"],
                export["object_module"],
                export["object_qualname"],
            ),
        )
        conn.execute(
            """
            INSERT INTO lookup_by_symbol(source_kind, symbol_name, kind, qualname, repo_path, target_path, line_number, module_name, section, priority_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "external_runtime_export",
                export["symbol_name"],
                export["kind"],
                export["qualname"],
                export["repo_path"],
                export["repo_path"],
                export["line_number"],
                export["module_name"],
                export["section"],
                0,
            ),
        )
    insert_meta(conn, "external_vectorbtpro_runtime_export_payload", runtime_export_payload)
    return data


def load_lab_build_registry(conn: sqlite3.Connection, build_path: Path, module_path: Path) -> dict[str, Any]:
    data = read_yaml(build_path)
    module_data = read_yaml(module_path)
    module_paths = {
        entry["module_id"]: entry.get("module_path") or entry.get("path")
        for entry in module_data.get("modules", [])
    }
    for entry in data.get("modules", []):
        module_id = entry["module_id"]
        module_abs_path = module_paths.get(module_id, entry.get("module_path"))
        payload = dict(entry)
        payload["module_path"] = module_abs_path
        conn.execute(
            """
            INSERT INTO build_module_records(module_id, module_path, build_status, rollout_phase, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                module_id,
                module_abs_path,
                entry.get("build_status"),
                entry.get("rollout_phase"),
                json.dumps(payload, sort_keys=True),
            ),
        )
        for stage_id in entry.get("primary_stage_ids", []):
            conn.execute(
                "INSERT INTO module_stage_links(module_id, stage_id) VALUES (?, ?)",
                (module_id, stage_id),
            )
            conn.execute(
                "INSERT INTO lookup_by_stage_id(stage_id, source_kind, target_path, line_hint, section) VALUES (?, ?, ?, ?, ?)",
                (stage_id, "lab_build_registry", str(build_path), 1, module_id),
            )
    return data


def load_active_note_surfaces(
    conn: sqlite3.Connection,
    *,
    build_registry_path: Path,
    notes_plan_path: Path,
) -> set[Path]:
    note_paths: set[Path] = set()
    candidates: list[tuple[Path, str, str]] = []

    def add_candidate(path_value: str | None, role: str, source_group: str) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if not path.exists() or path.suffix not in {".yaml", ".yml"}:
            return
        candidates.append((path, role, source_group))
        note_paths.add(path)

    build_registry = read_yaml(build_registry_path)
    note_hygiene = build_registry.get("note_hygiene", {})
    for path_value in note_hygiene.get("active_live_note_surfaces", []):
        add_candidate(path_value, "active_live_note", "build_registry")
    for path_value in note_hygiene.get("active_control_plane_surfaces", []):
        add_candidate(path_value, "active_control_plane_note", "build_registry")
    add_candidate(note_hygiene.get("archived_note_index"), "archived_note_index", "build_registry")

    seen_runtime_artifacts: set[str] = set()
    for path_value in note_hygiene.get("active_runtime_artifact_surfaces", []):
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        artifact_path = str(path)
        if artifact_path in seen_runtime_artifacts:
            continue
        seen_runtime_artifacts.add(artifact_path)
        payload = {
            "artifact_path": artifact_path,
            "artifact_kind": "runtime_artifact_surface",
            "role": "active_runtime_artifact",
            "priority": "high",
            "line_hint": 1,
            "section": "active_runtime_artifact",
            "source_group": "build_registry",
        }
        conn.execute(
            """
            INSERT INTO artifacts(
                artifact_path, artifact_kind, role, priority, line_hint, section, source_group, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_path,
                "runtime_artifact_surface",
                "active_runtime_artifact",
                "high",
                1,
                "active_runtime_artifact",
                "build_registry",
                json.dumps(payload, sort_keys=True),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_artifact_path(artifact_path, artifact_kind, role, priority) VALUES (?, ?, ?, ?)",
            (artifact_path, "runtime_artifact_surface", "active_runtime_artifact", "high"),
        )

    notes_plan = read_yaml(notes_plan_path)
    for role, path_value in notes_plan.get("authoritative_note_surfaces", {}).items():
        add_candidate(path_value, role, "notes_plan")
    for path_value in notes_plan.get("active_regime_control_plane_support_surfaces", []):
        add_candidate(path_value, "active_regime_control_plane_support", "notes_plan")

    seen_paths: set[str] = set()
    for path, role, source_group in candidates:
        artifact_path = str(path)
        if artifact_path in seen_paths:
            continue
        seen_paths.add(artifact_path)
        payload = {
            "artifact_path": artifact_path,
            "artifact_kind": "note_surface",
            "role": role,
            "priority": "high",
            "line_hint": 1,
            "section": role,
            "source_group": source_group,
        }
        conn.execute(
            """
            INSERT INTO artifacts(
                artifact_path, artifact_kind, role, priority, line_hint, section, source_group, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_path,
                "note_surface",
                role,
                "high",
                1,
                role,
                source_group,
                json.dumps(payload, sort_keys=True),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_artifact_path(artifact_path, artifact_kind, role, priority) VALUES (?, ?, ?, ?)",
            (artifact_path, "note_surface", role, "high"),
        )
    return note_paths


def load_registered_artifacts(conn: sqlite3.Connection, artifact_registry_path: Path) -> dict[str, Any]:
    data = read_yaml(artifact_registry_path)
    for entry in data.get("registered_artifacts", []):
        artifact_path = entry["path"]
        artifact_kind = "registered_artifact"
        role = entry.get("artifact_role")
        priority = "high"
        payload = {
            "artifact_path": artifact_path,
            "artifact_kind": artifact_kind,
            "role": role,
            "priority": priority,
            "section": role,
            "source_group": "artifact_registry",
        }
        conn.execute(
            """
            INSERT INTO artifacts(
                artifact_path, artifact_kind, role, priority, line_hint, section, source_group, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_path,
                artifact_kind,
                role,
                priority,
                1,
                role,
                "artifact_registry",
                json.dumps(payload, sort_keys=True),
            ),
        )
        conn.execute(
            "INSERT INTO lookup_by_artifact_path(artifact_path, artifact_kind, role, priority) VALUES (?, ?, ?, ?)",
            (artifact_path, artifact_kind, role, priority),
        )
    return data


def build_search_documents(conn: sqlite3.Connection, note_paths: set[Path]) -> None:
    docs: list[tuple[str, str, str | None, str, str | None, str]] = []
    query_specs = [
        ("SELECT artifact_path, artifact_kind, role, priority FROM artifacts", "artifact", 4),
        ("SELECT topic, goal, path, section FROM topic_routes", "topic_route", 4),
        ("SELECT truth_topic, goal, path, section FROM truth_routes", "truth_route", 4),
        ("SELECT repo_path, source_kind, source_key, target_path, section FROM lookup_by_repo_path", "repo_lookup", 5),
        ("SELECT surface_name, source_kind, target_path FROM lookup_by_notebook_surface", "notebook_surface", 3),
        ("SELECT stage_id, stage_kind, purpose FROM stage_records", "stage_record", 3),
        ("SELECT risk_id, blocking_severity, failure_mode FROM truth_risk_records", "truth_risk", 3),
        ("SELECT registry_id, adoption_mode, current_role FROM replacement_records", "replacement_record", 3),
        ("SELECT map_id, primary_path, adoption_mode FROM code_map_records", "code_map", 3),
        ("SELECT module_id, build_status, rollout_phase FROM build_module_records", "build_module", 3),
        ("SELECT alias, target_kind, target_value FROM query_aliases", "query_alias", 3),
    ]
    source_path_indexes = {
        "artifact": 0,
        "topic_route": 2,
        "truth_route": 2,
        "repo_lookup": 3,
        "notebook_surface": 2,
    }
    for sql, source_kind, width in query_specs:
        for row in conn.execute(sql):
            values = list(row)
            primary_key = values[0]
            secondary_key = values[1] if width > 1 else None
            path = None
            if source_kind in source_path_indexes:
                path = values[source_path_indexes[source_kind]]
            if source_kind == "query_alias":
                path = str(ALIAS_REGISTRY)
            elif source_kind == "build_module":
                path = str(LAB_BUILD_REGISTRY)
            elif source_kind == "stage_record":
                path = str(STAGE_CONTRACTS)
            elif source_kind == "truth_risk":
                path = str(TRUTH_RISK_MATRIX)
            elif source_kind == "replacement_record":
                path = str(REPLACEMENT_REGISTRY)
            elif source_kind == "code_map":
                path = str(CODE_LEVEL_MAP)
            if not path:
                path = str(ROOT)
            section = primary_key
            body = " ".join(str(v) for v in values if v)
            docs.append((source_kind, str(primary_key), None if secondary_key is None else str(secondary_key), str(path), section, body))

    for note_path in sorted(note_paths):
        docs.append(("note_body", str(note_path), None, str(note_path), None, safe_read_text(note_path)))

    for repo_file in external_vectorbtpro_repo_files():
        relative_path = external_vectorbtpro_relative_path(repo_file)
        docs.append(
            (
                "external_repo_file",
                relative_path,
                external_vectorbtpro_section(repo_file),
                str(repo_file),
                relative_path,
                safe_read_text(repo_file),
            )
        )
        for symbol in external_vectorbtpro_symbol_records(repo_file):
            docs.append(
                (
                    "external_repo_symbol",
                    symbol["qualname"],
                    symbol["kind"],
                    symbol["repo_path"],
                    symbol["qualname"],
                    " ".join(
                        [
                            symbol["symbol_name"],
                            symbol["qualname"],
                            symbol["kind"],
                            symbol["module_name"],
                            symbol["section"],
                            symbol["relative_path"],
                        ]
                    ),
                )
            )

    runtime_export_payload = read_meta(conn, "external_vectorbtpro_runtime_export_payload")
    for export in runtime_export_payload.get("records", []):
        docs.append(
            (
                "external_runtime_export",
                export["qualname"],
                export["kind"],
                export["repo_path"],
                export["qualname"],
                " ".join(
                    [
                        export["symbol_name"],
                        export["qualname"],
                        export["kind"],
                        export["module_name"],
                        export["section"],
                        export["object_module"],
                        export["object_qualname"],
                        export["repo_path"],
                    ]
                ),
            )
        )

    for doc in docs:
        conn.execute(
            "INSERT INTO search_documents(source_kind, primary_key, secondary_key, path, section, body_text) VALUES (?, ?, ?, ?, ?, ?)",
            doc,
        )
    try:
        conn.execute("INSERT INTO search_documents_fts(search_documents_fts) VALUES ('rebuild')")
    except sqlite3.OperationalError:
        pass


def count_batch_records(path: Path) -> int:
    data = read_yaml(path)
    total = 0
    for batch in data.get("batches", {}).values():
        total += len(batch.get("records", []))
    return total


def validate_core(conn: sqlite3.Connection) -> dict[str, Any]:
    required_tables = [
        "artifacts",
        "quick_start_routes",
        "topic_routes",
        "repo_path_routes",
        "truth_routes",
        "notebook_surfaces",
        "stage_records",
        "truth_risk_records",
        "replacement_records",
        "code_map_records",
        "build_module_records",
        "query_aliases",
        "external_repo_symbols",
        "external_runtime_exports",
        "lookup_by_repo_path",
        "lookup_by_symbol",
        "lookup_by_topic",
        "lookup_by_truth_topic",
        "lookup_by_notebook_surface",
        "lookup_by_artifact_path",
        "lookup_by_stage_id",
        "lookup_by_risk_id",
        "lookup_by_alias",
        "repo_stage_links",
        "repo_truth_links",
        "repo_candidate_links",
        "stage_truth_links",
        "module_stage_links",
        "search_documents",
    ]
    counts = {}
    for table in required_tables:
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    external_repo_like = f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%"
    external_repo_metrics = {
        "repo_path_routes": conn.execute(
            "SELECT COUNT(*) FROM repo_path_routes WHERE repo_path LIKE ?",
            (external_repo_like,),
        ).fetchone()[0],
        "lookup_by_repo_path": conn.execute(
            "SELECT COUNT(*) FROM lookup_by_repo_path WHERE repo_path LIKE ?",
            (external_repo_like,),
        ).fetchone()[0],
        "search_documents": conn.execute(
            "SELECT COUNT(*) FROM search_documents WHERE path LIKE ?",
            (external_repo_like,),
        ).fetchone()[0],
        "symbol_records": conn.execute(
            "SELECT COUNT(*) FROM external_repo_symbols WHERE repo_path LIKE ?",
            (external_repo_like,),
        ).fetchone()[0],
        "runtime_export_records": conn.execute(
            "SELECT COUNT(*) FROM external_runtime_exports WHERE repo_path LIKE ?",
            (external_repo_like,),
        ).fetchone()[0],
    }
    return {
        "required_tables": required_tables,
        "row_counts": counts,
        "external_vectorbtpro_metrics": external_repo_metrics,
        "all_nonzero": all(counts[t] > 0 for t in required_tables)
        and all(value > 0 for value in external_repo_metrics.values()),
    }


def build_examples(db_path: Path) -> dict[str, Any]:
    import subprocess

    script = ROOT / "scripts/query_vbtpro_machine_library.py"
    examples = [
        {"name": "alias_splitters", "args": ["--alias", "splitters"]},
        {"name": "topic_splitters", "args": ["--topic", "splitters_cv_and_no_leak"]},
        {
            "name": "artifact_path_external_vectorbtpro_index",
            "args": ["--artifact-path", str(EXTERNAL_VECTORBTPRO_LIBRARY_INDEX)],
        },
        {
            "name": "repo_vbtpro_splitter",
            "args": ["--repo-path", str(EXTERNAL_VECTORBTPRO_SOURCE_ROOT / "generic/splitting/base.py")],
        },
        {
            "name": "repo_vbtpro_from_signals",
            "args": ["--repo-path", str(EXTERNAL_VECTORBTPRO_SOURCE_ROOT / "portfolio/nb/from_signals.py")],
        },
        {
            "name": "repo_vbtpro_polygon",
            "args": ["--repo-path", str(EXTERNAL_VECTORBTPRO_SOURCE_ROOT / "data/custom/polygon.py")],
        },
        {
            "name": "repo_vbtpro_mcp_server",
            "args": ["--repo-path", str(EXTERNAL_VECTORBTPRO_SOURCE_ROOT / "mcp_server.py")],
        },
        {"name": "symbol_splittercv", "args": ["--symbol", "SplitterCV"]},
        {
            "name": "symbol_from_signals_nb",
            "args": ["--symbol", "vectorbtpro.portfolio.nb.from_signals.from_signals_nb"],
        },
        {"name": "text_splittercv", "args": ["--text", "SplitterCV"]},
        {"name": "text_from_signals", "args": ["--text", "from_signals"]},
        {"name": "stage_split_truth", "args": ["--stage-id", "S1_split_planning_and_membership_truth"]},
        {"name": "risk_split_membership", "args": ["--risk-id", "TR001_split_membership_drift"]},
        {"name": "module_machine_library", "args": ["--module-id", "machine_library"]},
        {
            "name": "artifact_path_timeframe_registry",
            "args": ["--artifact-path", str(ROOT / "artifacts/specs/runtime_config/timeframe_registry.yaml")],
        },
        {
            "name": "artifact_path_ml_gpu_index",
            "args": [
                "--artifact-path",
                str(ROOT / "artifacts/specs/control_plane/vbtpro_ml_gpu_vectorbtpro_index_20260416.yaml"),
            ],
        },
        {
            "name": "artifact_path_blackwell_ml_master_note",
            "args": [
                "--artifact-path",
                str(ROOT / "artifacts/specs/control_plane/vbtpro_blackwell_ml_master_note_20260416.yaml"),
            ],
        },
        {
            "name": "artifact_path_upstream_gpu_ml_source_index",
            "args": [
                "--artifact-path",
                str(ROOT / "artifacts/specs/control_plane/vbtpro_upstream_gpu_ml_source_index_20260416.yaml"),
            ],
        },
        {"name": "text_blackwell", "args": ["--text", "blackwell"]},
        {"name": "text_gpu_ml_vectorbtpro", "args": ["--text", "gpu ml vectorbtpro"]},
        {"name": "text_patsim", "args": ["--text", "PATSIM"]},
        {"name": "text_trendlb", "args": ["--text", "TRENDLB"]},
        {"name": "artifact_path_upstream_gpu_ml_triple_check_note", "args": ["--artifact-path", str(ROOT / "artifacts/specs/control_plane/vbtpro_upstream_gpu_ml_triple_check_note_20260416.yaml")]},
    ]
    out = {
        "artifact_id": "vbtpro_machine_library_query_examples_20260409",
        "artifact_version": 1,
        "artifact_type": "query_examples",
        "status": "generated",
        "created_on": datetime.now(timezone.utc).isoformat(),
        "owner": "codex",
        "db_path": str(db_path),
        "source_refs": [
            str(ROOT / "scripts/query_vbtpro_machine_library.py"),
            str(ROOT / "scripts/build_vbtpro_machine_library.py"),
        ],
        "examples": [],
    }
    for ex in examples:
        cmd = [sys.executable, str(script), "--db-path", str(db_path), "--format", "json", *ex["args"]]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out["examples"].append(
            {
                "name": ex["name"],
                "args": ex["args"],
                "result": json.loads(proc.stdout),
            }
        )
    return out


def build_coverage_dashboard(conn: sqlite3.Connection, path: Path, build_time: str) -> dict[str, Any]:
    ensure_parent(path)
    repo_route_count = conn.execute("SELECT COUNT(*) FROM repo_path_routes").fetchone()[0]
    external_vectorbtpro_repo_route_count = conn.execute(
        "SELECT COUNT(*) FROM repo_path_routes WHERE repo_path LIKE ?",
        (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
    ).fetchone()[0]
    external_vectorbtpro_search_document_count = conn.execute(
        "SELECT COUNT(*) FROM search_documents WHERE path LIKE ?",
        (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
    ).fetchone()[0]
    external_vectorbtpro_symbol_count = conn.execute(
        "SELECT COUNT(*) FROM external_repo_symbols WHERE repo_path LIKE ?",
        (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
    ).fetchone()[0]
    external_vectorbtpro_runtime_export_count = conn.execute(
        "SELECT COUNT(*) FROM external_runtime_exports WHERE repo_path LIKE ?",
        (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
    ).fetchone()[0]
    mapped_repo_count = conn.execute(
        """
        SELECT COUNT(DISTINCT r.repo_path)
        FROM repo_path_routes r
        JOIN repo_stage_links s ON s.repo_path = r.repo_path
        """
    ).fetchone()[0]
    build_status_counts = Counter(
        row[0] for row in conn.execute("SELECT build_status FROM build_module_records")
    )
    completion_contract = read_yaml(COMPLETION_CONTRACT)
    normalized_record_count = (
        conn.execute("SELECT COUNT(*) FROM stage_records").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM truth_risk_records").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM replacement_records").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM code_map_records").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM build_module_records").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM query_aliases").fetchone()[0]
    )
    batch_record_count = count_batch_records(BATCH_RECORDS)
    normalization_proxy_pct = round(
        normalized_record_count / (normalized_record_count + batch_record_count), 4
    ) if (normalized_record_count + batch_record_count) else 0.0
    minimum_thresholds = completion_contract["completion_profiles"]["minimum_complete"]["thresholds"]
    gold_thresholds = completion_contract["completion_profiles"]["gold_complete"]["thresholds"]
    repo_mapping_pct = round(mapped_repo_count / repo_route_count, 4) if repo_route_count else 0.0
    measured_surfaces = {
        "repo_mapping_threshold": repo_mapping_pct,
        "normalization_threshold_proxy": normalization_proxy_pct,
        "truth_risk_count_surface": conn.execute("SELECT COUNT(*) FROM truth_risk_records").fetchone()[0],
        "query_alias_count_surface": conn.execute("SELECT COUNT(*) FROM query_aliases").fetchone()[0],
    }
    minimum_measured_checks = {
        "repo_mapping_threshold": repo_mapping_pct >= minimum_thresholds["repo_mapping_threshold"]["target"],
        "normalization_threshold_proxy": normalization_proxy_pct >= minimum_thresholds["normalization_threshold"]["target"],
    }
    gold_measured_checks = {
        "repo_mapping_threshold": repo_mapping_pct >= gold_thresholds["repo_mapping_threshold"]["target"],
        "normalization_threshold_proxy": normalization_proxy_pct >= gold_thresholds["normalization_threshold"]["target"],
    }
    minimum_remaining_gaps = {
        "repo_mapping_threshold": round(
            max(0.0, minimum_thresholds["repo_mapping_threshold"]["target"] - repo_mapping_pct), 4
        ),
        "normalization_threshold_proxy": round(
            max(0.0, minimum_thresholds["normalization_threshold"]["target"] - normalization_proxy_pct), 4
        ),
    }
    gold_remaining_gaps = {
        "repo_mapping_threshold": round(
            max(0.0, gold_thresholds["repo_mapping_threshold"]["target"] - repo_mapping_pct), 4
        ),
        "normalization_threshold_proxy": round(
            max(0.0, gold_thresholds["normalization_threshold"]["target"] - normalization_proxy_pct), 4
        ),
    }
    normalization_targets_remaining: list[str] = []
    if batch_record_count > 0:
        normalization_targets_remaining.append(
            "promote_more_capability_batch_records_into_canonical_registries"
        )
    if gold_remaining_gaps["normalization_threshold_proxy"] > 0.0:
        normalization_targets_remaining.append(
            "extend_repo_code_level_map_toward_function_level_coverage"
        )
        normalization_targets_remaining.append(
            "add_more_query_aliases_for_frequent_build_and_truth questions"
        )
    if any(status != "built_validated" for status in build_status_counts):
        normalization_targets_remaining.append(
            "add_vbtpro_lab_module_validation_events_into_build_registry"
        )
    dashboard = {
        "artifact_id": "vbtpro_library_coverage_dashboard_20260409",
        "artifact_version": 1,
        "artifact_type": "coverage_dashboard",
        "status": "generated",
        "created_on": build_time,
        "owner": "codex",
        "db_path": str(DEFAULT_DB),
        "compiled_counts": {
            "artifact_count": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            "search_document_count": conn.execute("SELECT COUNT(*) FROM search_documents").fetchone()[0],
            "stage_record_count": conn.execute("SELECT COUNT(*) FROM stage_records").fetchone()[0],
            "truth_risk_count": conn.execute("SELECT COUNT(*) FROM truth_risk_records").fetchone()[0],
            "replacement_count": conn.execute("SELECT COUNT(*) FROM replacement_records").fetchone()[0],
            "code_map_count": conn.execute("SELECT COUNT(*) FROM code_map_records").fetchone()[0],
            "build_module_count": conn.execute("SELECT COUNT(*) FROM build_module_records").fetchone()[0],
            "query_alias_count": conn.execute("SELECT COUNT(*) FROM query_aliases").fetchone()[0],
            "repo_stage_link_count": conn.execute("SELECT COUNT(*) FROM repo_stage_links").fetchone()[0],
            "repo_truth_link_count": conn.execute("SELECT COUNT(*) FROM repo_truth_links").fetchone()[0],
            "repo_candidate_link_count": conn.execute("SELECT COUNT(*) FROM repo_candidate_links").fetchone()[0],
            "external_vectorbtpro_repo_route_count": external_vectorbtpro_repo_route_count,
            "external_vectorbtpro_search_document_count": external_vectorbtpro_search_document_count,
            "external_vectorbtpro_symbol_count": external_vectorbtpro_symbol_count,
            "external_vectorbtpro_runtime_export_count": external_vectorbtpro_runtime_export_count,
        },
        "coverage_metrics": {
            "repo_path_router_count": repo_route_count,
            "repo_stage_mapped_count": mapped_repo_count,
            "repo_stage_mapping_pct": repo_mapping_pct,
            "batch_record_intake_count": batch_record_count,
            "normalized_record_count": normalized_record_count,
            "normalization_proxy_pct": normalization_proxy_pct,
            "external_vectorbtpro_repo_route_count": external_vectorbtpro_repo_route_count,
            "external_vectorbtpro_search_document_count": external_vectorbtpro_search_document_count,
            "external_vectorbtpro_symbol_count": external_vectorbtpro_symbol_count,
            "external_vectorbtpro_runtime_export_count": external_vectorbtpro_runtime_export_count,
        },
        "completion_profiles": {
            "minimum_complete": {
                "thresholds": minimum_thresholds,
                "current_measurements": measured_surfaces,
                "measured_checks": minimum_measured_checks,
                "remaining_gaps": minimum_remaining_gaps,
                "meets_all_currently_measured_thresholds": all(minimum_measured_checks.values()),
            },
            "gold_complete": {
                "thresholds": gold_thresholds,
                "current_measurements": measured_surfaces,
                "measured_checks": gold_measured_checks,
                "remaining_gaps": gold_remaining_gaps,
                "meets_all_currently_measured_thresholds": all(gold_measured_checks.values()),
            },
        },
        "measurement_status": {
            "directly_measured_now": completion_contract["measurement_contract"]["directly_measured_now"],
            "not_yet_directly_measured": completion_contract["measurement_contract"]["not_yet_directly_measured"],
        },
        "new_repo_build_status": {
            "module_status_counts": dict(sorted(build_status_counts.items())),
            "active_next_modules": [
                row[0]
                for row in conn.execute(
                    "SELECT module_id FROM build_module_records WHERE build_status = 'active_next' ORDER BY module_id"
                )
            ],
        },
        "normalization_targets_remaining": normalization_targets_remaining,
        "source_refs": [
            str(COMPLETION_CONTRACT),
            str(LAB_BUILD_REGISTRY),
            str(BATCH_RECORDS),
            str(ROOT / "scripts/build_vbtpro_machine_library.py"),
        ],
    }
    path.write_text(yaml.safe_dump(dashboard, sort_keys=False))
    return dashboard


def do_build(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    temp_db_path = temp_db_path_for(db_path)
    manifest_path = Path(args.manifest_path)
    examples_path = Path(args.examples_path)
    dashboard_path = Path(args.dashboard_path)
    ensure_parent(db_path)
    ensure_parent(manifest_path)
    ensure_parent(examples_path)
    ensure_parent(dashboard_path)

    sources = {
        MASTER_INDEX: "master_index",
        TOPIC_ROUTER: "topic_router",
        REPO_PATH_ROUTER: "repo_path_router",
        TRUTH_ROUTER: "truth_router",
        NOTEBOOK_ROUTER: "notebook_router",
        STAGE_CONTRACTS: "stage_contracts",
        TRUTH_RISK_MATRIX: "truth_risk_matrix",
        REPLACEMENT_REGISTRY: "replacement_registry",
        CODE_LEVEL_MAP: "code_level_map",
        ALIAS_REGISTRY: "alias_registry",
        LAB_BUILD_REGISTRY: "lab_build_registry",
        RESET_REBUILD_NOTES_PLAN: "reset_rebuild_notes_plan",
        LAB_MODULE_REGISTRY: "lab_module_registry",
        ARTIFACT_REGISTRY: "artifact_registry",
        BATCH_RECORDS: "batch_records",
        EXTERNAL_VECTORBTPRO_LIBRARY_INDEX: "external_vectorbtpro_library_index",
    }

    if args.validate_only:
        try:
            with machine_library_lock(db_path, "validate_only"):
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    validation = validate_core(conn)
                finally:
                    conn.close()
        except MachineLibraryLockError as exc:
            print(str(exc))
            raise SystemExit(2) from exc
        print(json.dumps(validation, indent=2, sort_keys=True))
        if not validation["all_nonzero"]:
            raise SystemExit(1)
        return

    try:
        with machine_library_lock(db_path, "rebuild"):
            sync_repo_truth_note_surfaces(
                build_registry_path=LAB_BUILD_REGISTRY,
                notes_plan_path=RESET_REBUILD_NOTES_PLAN,
            )
            if temp_db_path.exists():
                temp_db_path.unlink()
            conn = sqlite3.connect(temp_db_path)
            create_schema(conn)
            add_source_files(conn, sources)

            load_master_index(conn, MASTER_INDEX)
            load_topic_router(conn, TOPIC_ROUTER)
            load_repo_path_router(conn, REPO_PATH_ROUTER)
            load_truth_router(conn, TRUTH_ROUTER)
            load_notebook_router(conn, NOTEBOOK_ROUTER)
            load_stage_contracts(conn, STAGE_CONTRACTS)
            load_truth_risk_matrix(conn, TRUTH_RISK_MATRIX)
            load_replacement_registry(conn, REPLACEMENT_REGISTRY)
            load_code_level_map(conn, CODE_LEVEL_MAP)
            load_alias_registry(conn, ALIAS_REGISTRY)
            load_lab_build_registry(conn, LAB_BUILD_REGISTRY, LAB_MODULE_REGISTRY)
            load_external_vectorbtpro_repo(conn, EXTERNAL_VECTORBTPRO_LIBRARY_INDEX)
            active_note_paths = load_active_note_surfaces(
                conn,
                build_registry_path=LAB_BUILD_REGISTRY,
                notes_plan_path=RESET_REBUILD_NOTES_PLAN,
            )
            load_registered_artifacts(conn, ARTIFACT_REGISTRY)

            note_paths = set()
            for table, col in [
                ("artifacts", "artifact_path"),
                ("quick_start_routes", "path"),
                ("topic_routes", "path"),
                ("truth_routes", "path"),
                ("repo_path_routes", "first_note"),
                ("repo_support_sections", "support_path"),
                ("lookup_by_notebook_surface", "target_path"),
            ]:
                for (note_path,) in conn.execute(f"SELECT DISTINCT {col} FROM {table}"):
                    if note_path and note_path.endswith((".yaml", ".yml")) and Path(note_path).exists():
                        note_paths.add(Path(note_path))
            note_paths.update(active_note_paths)
            note_paths.update(
                {
                    MASTER_INDEX,
                    TOPIC_ROUTER,
                    REPO_PATH_ROUTER,
                    TRUTH_ROUTER,
                    NOTEBOOK_ROUTER,
                    STAGE_CONTRACTS,
                    TRUTH_RISK_MATRIX,
                    REPLACEMENT_REGISTRY,
                    CODE_LEVEL_MAP,
                    ALIAS_REGISTRY,
                    LAB_BUILD_REGISTRY,
                    ARTIFACT_REGISTRY,
                    BATCH_RECORDS,
                    EXTERNAL_VECTORBTPRO_LIBRARY_INDEX,
                }
            )

            for note_path in sorted(note_paths):
                if not note_path.exists():
                    continue
                for row in extract_note_sections(note_path):
                    conn.execute(
                        "INSERT INTO note_section_index(note_path, line_no, indent, key_name) VALUES (?, ?, ?, ?)",
                        row,
                    )

            build_search_documents(conn, {p for p in note_paths if p.exists()})
            build_time = datetime.now(timezone.utc).isoformat()
            validation = validate_core(conn)
            insert_meta(conn, "build_time_utc", build_time)
            insert_meta(conn, "source_inputs", {str(k): v for k, v in sources.items()})
            insert_meta(conn, "validation", validation)
            conn.commit()
            conn.close()

            temp_db_path.replace(db_path)

            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

            dashboard = build_coverage_dashboard(conn, dashboard_path, build_time)
            manifest = {
                "artifact_id": "vbtpro_machine_library_build_manifest_20260409",
                "artifact_version": 1,
                "artifact_type": "build_manifest",
                "status": "generated",
                "created_on": build_time,
                "owner": "codex",
                "db_path": str(db_path),
                "dashboard_path": str(dashboard_path),
                "source_inputs": [str(p) for p in sources],
                "validation": validation,
                "quick_start_route_count": conn.execute("SELECT COUNT(*) FROM quick_start_routes").fetchone()[0],
                "topic_route_count": conn.execute("SELECT COUNT(*) FROM topic_routes").fetchone()[0],
                "repo_route_count": conn.execute("SELECT COUNT(*) FROM repo_path_routes").fetchone()[0],
                "external_vectorbtpro_repo_route_count": conn.execute(
                    "SELECT COUNT(*) FROM repo_path_routes WHERE repo_path LIKE ?",
                    (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
                ).fetchone()[0],
                "external_vectorbtpro_search_document_count": conn.execute(
                    "SELECT COUNT(*) FROM search_documents WHERE path LIKE ?",
                    (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
                ).fetchone()[0],
                "external_vectorbtpro_symbol_count": conn.execute(
                    "SELECT COUNT(*) FROM external_repo_symbols WHERE repo_path LIKE ?",
                    (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
                ).fetchone()[0],
                "external_vectorbtpro_runtime_export_count": conn.execute(
                    "SELECT COUNT(*) FROM external_runtime_exports WHERE repo_path LIKE ?",
                    (f"{EXTERNAL_VECTORBTPRO_SOURCE_ROOT}%",),
                ).fetchone()[0],
                "truth_route_count": conn.execute("SELECT COUNT(*) FROM truth_routes").fetchone()[0],
                "notebook_surface_count": conn.execute("SELECT COUNT(DISTINCT surface_name) FROM notebook_surfaces").fetchone()[0],
                "note_section_index_count": conn.execute("SELECT COUNT(*) FROM note_section_index").fetchone()[0],
                "normalized_counts": dashboard["compiled_counts"],
                "source_refs": [str(p) for p in sources] + [str(ROOT / "scripts/build_vbtpro_machine_library.py")],
            }
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

            examples = build_examples(db_path)
            examples_path.write_text(yaml.safe_dump(examples, sort_keys=False))

            summary = {
                "db_path": str(db_path),
                "manifest_path": str(manifest_path),
                "examples_path": str(examples_path),
                "dashboard_path": str(dashboard_path),
                "validation": validation,
                "artifact_count": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            conn.close()
    except MachineLibraryLockError as exc:
        print(str(exc))
        raise SystemExit(2) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--examples-path", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--dashboard-path", default=str(DEFAULT_DASHBOARD))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    do_build(parse_args())
