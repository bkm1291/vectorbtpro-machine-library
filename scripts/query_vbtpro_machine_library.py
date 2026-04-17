#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vbtpro_lab.machine_library.runtime import default_db_path, query_from_namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=default_db_path())
    parser.add_argument("--artifact-path")
    parser.add_argument("--repo-path")
    parser.add_argument("--symbol")
    parser.add_argument("--topic")
    parser.add_argument("--truth-topic")
    parser.add_argument("--notebook-surface")
    parser.add_argument("--stage-id")
    parser.add_argument("--risk-id")
    parser.add_argument("--registry-id")
    parser.add_argument("--map-id")
    parser.add_argument("--module-id")
    parser.add_argument("--alias")
    parser.add_argument("--text")
    parser.add_argument("--format", choices=["json", "table"], default="json")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = query_from_namespace(args)
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    elif "rows" in result:
        for row in result["rows"]:
            print(" | ".join(f"{k}={v}" for k, v in row.items()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
