import csv
import os
import warnings

import pandas

import bodo.pandas as pd
import bodo.utils.pandas_coverage_tracking as tracker
from bodo.pandas.utils import BodoLibFallbackWarning

urls = tracker.PANDAS_URLS
output_path = "bodo_compat_report.csv"


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
        if body.startswith("_") or body.startswith("__"):
            continue
        if prefix not in [
            "Series.",
            "DataFrame.",
            "DataFrameGroupBy.",
            "SeriesGroupBy.",
            "",
        ]:
            coverage.append([attr, "NO", link])
            continue
        sample = get_sample(attr)
        name = body
        supported = "NO"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                recursive_getattr(sample, name)
                supported = "YES"
            except Exception:
                pass
        if record:
            fallback_warnings = [
                w for w in record if issubclass(w.category, BodoLibFallbackWarning)
            ]
            if fallback_warnings:
                supported = "NO"
        coverage.append([attr, supported, link])
    return coverage


res = {}
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
