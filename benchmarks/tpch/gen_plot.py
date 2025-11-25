import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_section(lines, backend):
    rows = []

    # BODO has "QXX Execution time (s): ..."
    re_bodo = re.compile(r"Q(\d+)\s+Execution time \(s\):\s+([0-9.]+)")
    # BODO total
    re_bodo_total = re.compile(r"Total query execution time \(s\):\s+([0-9.]+)")

    # POLARS has repeated "Query execution time (s): ..."
    re_polars = re.compile(r"Query execution time \(s\):\s+([0-9.]+)")
    re_polars_total = re.compile(r"Total query execution time \(s\):\s+([0-9.]+)")

    qnum = 1

    for line in lines:
        line = line.strip()

        if backend == "bodo" or backend == "pyspark":
            m = re_bodo.search(line)
            if m:
                query = f"q{int(m.group(1))}"
                time = float(m.group(2))
                rows.append((backend, query, time))
                continue

            m = re_bodo_total.search(line)
            if m:
                rows.append((backend, "total", float(m.group(1))))

        elif backend == "polars":
            m = re_polars.search(line)
            if m:
                query = f"q{qnum}"
                time = float(m.group(1))
                rows.append((backend, query, time))
                qnum += 1
                continue

            m = re_polars_total.search(line)
            if m:
                rows.append((backend, "total", float(m.group(1))))

    return rows


def get_idx(text, pattern):
    try:
        return text.index(pattern)
    except ValueError:
        return -1


def parse_file(path):
    text = Path(path).read_text().splitlines()

    # split into backend sections
    bodo_start = text.index("### BODO ###")
    polars_start = text.index("### POLARS ###")
    pyspark_start = text.index("### PYSPARK ###")

    bodo_lines = text[bodo_start:polars_start]
    polars_lines = text[polars_start:pyspark_start]
    pyspark_lines = text[pyspark_start:]

    rows = []
    rows += parse_section(bodo_lines, "bodo")
    rows += parse_section(polars_lines, "polars")
    rows += parse_section(pyspark_lines, "pyspark")

    return rows


def plot_results():
    # Load CSV
    df = pd.read_csv("queries.csv")

    # Remove the "total" row for the per-query chart
    df_queries = df[df["query"] != "total"].copy()
    df_totals = df[df["query"] == "total"].copy()

    pivot = df_queries.pivot(index="query", columns="backend", values="time")

    pivot = pivot.reindex(sorted(pivot.index, key=lambda q: int(q[1:])))
    pivot = pivot[["polars", "bodo", "pyspark"]]

    # Plot query times
    _, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pivot.index))
    width = 0.5

    for i, backend in enumerate(pivot.columns):
        print(f"{backend} total time: {pivot[backend].sum():.2f} s")
        ax.bar(
            x - width / 2 + i * width / len(pivot.columns),
            pivot[backend],
            width / len(pivot.columns),
            label=backend,
        )

    ax.set_xlabel("Query")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("TPC-H Query Execution Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend()

    plt.savefig("tpch_query_comparison.png")

    # Plot totals
    colors = ["skyblue", "lightgreen", "salmon", "orange", "purple", "cyan"]
    df_totals.plot.bar(
        x="backend",
        y="time",
        legend=False,
        title="Total Execution Time Comparison",
        ylabel="Execution Time (s)",
        xlabel="Backend",
        color=colors[: len(df_totals)],
    )
    plt.savefig("tpch_total_comparison.png")


def main():
    input_path = "out.txt"
    output_path = "queries.csv"

    rows = parse_file(input_path)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "query", "time"])
        for backend, query, time in rows:
            writer.writerow([backend, query, time])

    print(f"Wrote {len(rows)} rows to {output_path}")
    plot_results()


if __name__ == "__main__":
    main()
