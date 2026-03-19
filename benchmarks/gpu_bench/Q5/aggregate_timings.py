"""
Usage:
    python aggregate_timings.py --timings_csv <path_to_timings_csv> --output_csv <path_to_output_csv>
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timings_csv",
        type=str,
        default="timings.csv",
        help="Path to CSV file containing individual timing runs. The CSV file is expected to have columns: implementation, n_gpus, scale_factor, extras, time_seconds.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="timings_agg.csv",
        help="Path to CSV file where aggregated timings will be saved. The aggregated timings will contain the average time_seconds for each combination of implementation, n_gpus, scale_factor, and extras.",
    )
    args = parser.parse_args()

    agg_df = (
        pd.read_csv(args.timings_csv)
        .groupby(
            by=["implementation", "n_gpus", "scale_factor", "storage_type", "extras"],
            as_index=False,
        )["time_seconds"]
        .mean()
    )
    agg_df = agg_df.pivot(
        index=["storage_type", "scale_factor", "n_gpus"],
        columns=["implementation", "extras"],
        values="time_seconds",
    ).reset_index()
    agg_df.columns = [
        f"{impl}[{extra}]" if extra else impl for impl, extra in agg_df.columns
    ]
    agg_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
