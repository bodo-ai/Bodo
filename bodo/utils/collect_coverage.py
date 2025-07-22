"""
This script scans a list of predefined Pandas API documentation URLs (from `bodo.utils.pandas_coverage_tracking`)
and determines whether each API is currently supported by Bodo's Pandas backend.

For each API:
- It instantiates a minimal sample object (e.g., Series, DataFrame) based on the API category.
- It attempts to access the attribute or method.
- It monitors for `BodoLibFallbackWarning` warnings or exceptions to infer support status.
- It generates a tab-separated report listing:
    - API category
    - Full method or attribute name
    - Whether it is supported (Boolean)
    - Link to the Pandas documentation

Usage:
    Run the script with the environment variable `BODO_PANDAS_FALLBACK=0` to prevent silent fallbacks:
        BODO_PANDAS_FALLBACK=0 python3 collect_coverage.py

Output:
    A TSV file named `bodo_coverage_report.csv` (with tab delimiter) summarizing support status for all tested APIs.
"""

import csv
import os
import warnings

import pandas

import bodo.pandas as pd
import bodo.utils.pandas_coverage_tracking as tracker
from bodo.pandas.utils import BodoLibFallbackWarning

urls = tracker.PANDAS_URLS
output_path = "bodo_coverage_report.csv"


def get_sample(name):
    if name.startswith("Series."):
        if name.startswith("Series.str"):
            return pd.Series([""])
        if name.startswith("Series.dt"):
            return pd.Series(
                pandas.date_range("20010827 01:08:27", periods=1, freq="MS")
            )
        return pd.Series([])
    elif name.startswith("DataFrame."):
        return pd.DataFrame({"A": []})
    elif name.startswith("DataFrameGroupBy."):
        return pd.DataFrame({"A": [], "B": []}).groupby("B")
    elif name.startswith("SeriesGroupBy."):
        return pd.DataFrame({"A": [], "B": []}).groupby("B")["A"]
    else:
        return pd


def recursive_getattr(sample, name):
    parts = name.split(".")
    module = sample
    for part in parts:
        module = getattr(module, part)


def get_prefix(attr):
    if "." not in attr:
        return "", attr
    return attr[: attr.index(".") + 1], attr[attr.index(".") + 1 :]


def collect(key):
    coverage = []
    url = urls[key]
    for attr in tracker.get_pandas_apis_from_url(url):
        link = (
            f"https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.{attr}.html"
            if attr.startswith("Styler")
            else f"https://pandas.pydata.org/docs/reference/api/pandas.{attr}.html"
        )
        prefix, body = get_prefix(attr)
        if prefix not in [
            "Series.",
            "DataFrame.",
            "DataFrameGroupBy.",
            "SeriesGroupBy.",
            "",
        ]:
            coverage.append([attr, False, link])
            continue
        sample = get_sample(attr)
        name = body
        supported = False
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                recursive_getattr(sample, name)
                supported = True
            except Exception:
                pass
        if record:
            fallback_warnings = [
                w for w in record if issubclass(w.category, BodoLibFallbackWarning)
            ]
            if fallback_warnings:
                supported = False
        coverage.append([attr, supported, link])
    return coverage


if __name__ == "__main__":
    res = {}
    # The script catches both warnings and exceptions to tell if a method is not supported.
    # Ideally it would be best to only watch for warnings, but due to silent fallbacks of top-level methods
    # we turn off BODO_PANDAS_FALLBACK and catch exceptions raised for unsupported methods.
    assert os.environ["BODO_PANDAS_FALLBACK"] == "0", (
        "Execute script with command >> BODO_PANDAS_FALLBACK=0 Python3 path-to-file/collect_coverage.py"
    )
    for key in urls:
        res[key] = globals()["collect"](key)

    with open(output_path, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["Category", "Method", "Supported", "Link"])

        for key in res:
            infolist = res[key]
            if not infolist:
                continue
            for entry in infolist:
                writer.writerow([key, entry[0], entry[1], entry[2]])

    print(f"Compatibility report written to: {os.path.abspath(output_path)}")
