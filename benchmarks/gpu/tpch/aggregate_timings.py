"""
Aggregate timings from multiple runs of the TPC-H benchmark and save the average timings to a new CSV file.
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timings_csv",
        type=str,
        default="timings.csv",
        help="Path to CSV file containing individual timing runs. The CSV file is expected to have columns: implementation, n_gpus, scale_factor, params, time_seconds.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="timings_agg.csv",
        help="Path to CSV file where aggregated timings will be saved. The aggregated timings will contain the average time_seconds for each combination of implementation, n_gpus, scale_factor, and params.",
    )
    args = parser.parse_args()

    agg_df = (
        pd.read_csv(args.timings_csv)
        .groupby(
            by=["implementation", "n_gpus", "scale_factor", "storage_type", "params"],
            as_index=False,
            dropna=False,
        )["time_seconds"]
        .mean()
    )
    agg_df = agg_df.pivot(
        index=["storage_type", "scale_factor", "n_gpus"],
        columns=["implementation", "params"],
        values="time_seconds",
    )
    agg_df.columns = [
        f"{impl}[{params}]" if pd.notna(params) else impl
        for impl, params in agg_df.columns
    ]
    agg_df = agg_df.reset_index()
    agg_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
