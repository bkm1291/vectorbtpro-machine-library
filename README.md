# vectorbtpro-machine-library

Standalone sqlite-backed machine library for the vendored `vectorbt.pro` source tree.

Contents:
- `external/vectorbt.pro/`: repo-owned VBT PRO source used for file, symbol, and runtime-export indexing
- `reports/strategy_factory/`: generated sqlite DB, routers, manifests, dashboards, and query examples
- `scripts/`: rebuild and query entrypoints
- `src/vbtpro_lab/machine_library/`: runtime support used by the builder and query CLI

Notes:
- This repo is separate from `vbtpro-lab` and replaces the old frozen `trade_scanner2`-era export.
- The sqlite DB is rebuilt locally from the vendored `vectorbt.pro` checkout and the machine-library routing surfaces committed here.
- Internal artifact names still preserve `vbtpro_lab_*` compatibility labels where the existing schema depends on them.
