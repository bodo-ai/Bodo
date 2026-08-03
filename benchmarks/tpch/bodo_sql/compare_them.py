#!/usr/bin/env python3
"""
Compare three Parquet files row-wise and report differences.

Usage:
  python compare_parquets.py todd_pandas.pq todd_bodo_cpu.pq todd_bodo_gpu.pq
Options:
  --sort-by col1,col2    : comma-separated columns to sort by before comparing (aligns rows)
  --max-diffs N          : stop after reporting N differing rows (default 1000)
  --show-values          : show full row values for differing rows (default True)
  --engine ENGINE        : parquet engine for pandas (pyarrow or fastparquet). Default: pyarrow
"""

import argparse
import sys

import numpy as np
import pandas as pd


def read_parquet(path: str, engine: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine=engine).sort_values("S_NAME", ascending=True)


def align_and_prepare(
    dfs: list[pd.DataFrame], sort_by: list[str]
) -> tuple[list[pd.DataFrame], list[str]]:
    # Ensure same columns set and order
    all_cols = []
    for df in dfs:
        print("---------------")
        print(df.columns)
        print(df)
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)
    # Reindex columns in each df to the union (missing columns filled with NaN)
    dfs_reindexed = [df.reindex(columns=all_cols) for df in dfs]

    # Optionally sort by key columns to align rows deterministically
    if sort_by:
        for i, df in enumerate(dfs_reindexed):
            # If any sort column missing, raise
            missing = [c for c in sort_by if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Sort-by columns {missing} not present in file {i + 1}"
                )
            dfs_reindexed[i] = df.sort_values(by=sort_by, kind="mergesort").reset_index(
                drop=True
            )
    else:
        # Reset index to simple integer index for row-by-row comparison
        dfs_reindexed = [df.reset_index(drop=True) for df in dfs_reindexed]

    return dfs_reindexed, all_cols


def compare_rows(dfs: list[pd.DataFrame], cols: list[str], max_diffs: int):
    len(dfs)
    lengths = [len(df) for df in dfs]
    max_len = max(lengths)

    diffs = []
    only_in = []
    for i in range(max_len):
        # gather row values for each file (or None if file shorter)
        rows = []
        for df in dfs:
            if i < len(df):
                rows.append(df.iloc[i])
            else:
                rows.append(None)

        # If any file missing this row, record presence differences
        if any(r is None for r in rows):
            present = [r is not None for r in rows]
            only_in.append((i, present))
            if len(diffs) + len(only_in) >= max_diffs:
                break
            continue

        # Compare values column by column
        differing_cols = []
        for c in cols:
            vals = []
            for r in rows:
                v = r[c]
                # Normalize NaN for comparison
                if pd.isna(v):
                    vals.append(np.nan)
                else:
                    vals.append(v)
            # All equal if all pairwise equal considering NaN equality
            equal = True
            first = vals[0]
            for v in vals[1:]:
                # treat NaN == NaN
                if pd.isna(first) and pd.isna(v):
                    continue
                if pd.isna(first) != pd.isna(v):
                    equal = False
                    break
                if not pd.isna(first) and first != v:
                    equal = False
                    break
            if not equal:
                differing_cols.append((c, vals))
        if differing_cols:
            diffs.append((i, differing_cols))
        if len(diffs) + len(only_in) >= max_diffs:
            break
    return diffs, only_in, lengths


def print_summary(paths: list[str], lengths: list[int], diffs, only_in):
    print("Files compared:")
    for i, p in enumerate(paths):
        print(f"  [{i}] {p}  (rows: {lengths[i]})")
    print()

    if not diffs and not only_in:
        print("No differences found (row-by-row comparison).")
        return

    if only_in:
        print(
            "Rows present in some files but not others (index, presence vector [file0,file1,file2]):"
        )
        for idx, present in only_in:
            print(f"  row {idx}: {present}")
        print()

    if diffs:
        print(f"Differing rows (showing up to reported limit): {len(diffs)}")
        for idx, differing_cols in diffs:
            print(f"\nRow index: {idx}")
            print("Columns that differ:")
            for col, vals in differing_cols:
                # Format values for each file
                val_strs = []
                for v in vals:
                    if pd.isna(v):
                        val_strs.append("NaN")
                    else:
                        val_strs.append(repr(v))
                print(f"  {col}:")
                for fi, vs in enumerate(val_strs):
                    print(f"    [{fi}] {vs}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare three Parquet files row-by-row and report differences."
    )
    parser.add_argument(
        "files", nargs=3, help="Three parquet files to compare (order matters)."
    )
    parser.add_argument(
        "--sort-by",
        default="",
        help="Comma-separated columns to sort by before comparing (align rows).",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=1000,
        help="Maximum number of differing rows to report.",
    )
    parser.add_argument(
        "--engine",
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Parquet engine for pandas.",
    )
    args = parser.parse_args()

    paths = args.files
    sort_by = [s for s in (args.sort_by.split(",") if args.sort_by else []) if s]

    try:
        dfs = [read_parquet(p, engine=args.engine) for p in paths]
    except Exception as e:
        print(f"Error reading parquet files: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        dfs_aligned, cols = align_and_prepare(dfs, sort_by)
    except Exception as e:
        print(f"Error preparing dataframes: {e}", file=sys.stderr)
        sys.exit(3)

    diffs, only_in, lengths = compare_rows(dfs_aligned, cols, args.max_diffs)
    print_summary(paths, lengths, diffs, only_in)


if __name__ == "__main__":
    main()
