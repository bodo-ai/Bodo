# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Aggregate query profile logs:
    usage: python -m bodo.utils.query_profile_collector_aggregator <dir>
"""
import argparse
import json
import sys
from pathlib import Path

from bodo.utils.aggregate_query_profiles import aggregate

# The pragma: no cover comments are used to skip coverage because this is just a
# wraper around the functionality in __init__.py, which is covered by tests.


def main(argv: list[str]):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--print", dest="print", default=False, action="store_true")
    args = parser.parse_args(argv[1:])
    assert args.dir.is_dir(), f"'{args.dir}' is not a directory."

    logs = []
    for path in args.dir.iterdir():
        if not path.stem.startswith("query_profile"):
            continue
        with path.open() as f:
            data = json.load(f)
        logs.append(data)

    aggregated = json.dumps(aggregate(logs), indent=4)
    if args.print:
        print(aggregated)
    with open(args.dir / "aggregated.json", "w") as f:
        f.write(aggregated)
        print(
            f"Aggregated logs written to {args.dir / 'aggregated.json'}",
            file=sys.stderr,
        )


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
