# AGENTS.md

## Scope

This repo is the standalone sqlite-backed VBT PRO machine library rooted at
`/home/benji/vectorbtpro-machine-library`.

It is the active machine-library authority. Use this repo as the only machine-library authority.

## Read First

1. `README.md`
2. `repo_manifest.yaml`
3. `reports/strategy_factory/vbtpro_machine_library_build_manifest_20260409.yaml`
4. `reports/strategy_factory/vbtpro_machine_library_query_examples_20260409.yaml`
5. `artifacts/specs/control_plane/vbtpro_external_vectorbtpro_library_index_20260416.yaml`

## Query Surfaces

- `scripts/build_vbtpro_machine_library.py`: rebuild sqlite library
- `scripts/query_vbtpro_machine_library.py`: query repo-path, symbol, alias, topic, and text surfaces
- `reports/strategy_factory/vbtpro_machine_library_20260409.sqlite`: compiled lookup DB

## Write Boundaries

- Keep this repo focused on the machine-library layer, not wider runtime code.
- Prefer updating build/query/runtime support and generated manifests together.
- If source-of-truth routing changes, rebuild the sqlite library and refresh the generated reports in `reports/strategy_factory/`.

## Validation

- Rebuild: `python scripts/build_vbtpro_machine_library.py`
- Validate only: `python scripts/build_vbtpro_machine_library.py --validate-only`
- Smoke query: `python scripts/query_vbtpro_machine_library.py --symbol SplitterCV`
