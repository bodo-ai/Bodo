import csv
import os
import warnings

# import pandas as pd
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
            return pd.Series(pd.date_range("20010827 01:08:27", periods=1, freq="MS"))
        return pd.Series([])

    elif name.startswith("DataFrame."):
        return pd.DataFrame({"A": []})
    elif name.startswith("DataFrameGroupBy."):
        return pd.DataFrame({"A": [], "B": []}).groupby("B")
    elif name.startswith("SeriesGroupBy."):
        return pd.DataFrame({"A": [], "B": []}).groupby("B")["A"]


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
    count = 0
    url = urls[key]
    for attr in tracker.get_pandas_apis_from_url(url):
        prefix, body = get_prefix(attr)
        print(prefix)
        if (
            prefix
            not in ["Series.", "DataFrame.", "DataFrameGroupBy.", "SeriesGroupBy."]
            or body.startswith("_")
            or body.startswith("__")
        ):
            continue
        print(attr)
        sample = get_sample(attr)
        name = attr[attr.index(".") + 1 :]
        supported = False
        print(attr)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                recursive_getattr(sample, name)
                supported = True
            except Exception as e:
                print(e)
                pass
        if record:
            fallback_warnings = [
                w for w in record if issubclass(w.category, BodoLibFallbackWarning)
            ]
            if fallback_warnings:
                supported = False
        if supported:
            count += 1
        coverage.append([attr, supported])
    return coverage


res = {}
for key in urls:
    res[key] = globals()["collect"](key)

# Write results to CSV
with open(output_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Category", "Method", "Supported"])

    for key in res:
        infolist = res[key]
        if not infolist:
            continue
        for entry in infolist:
            writer.writerow([key, entry[0], entry[1]])

print(f"Compatibility report written to: {os.path.abspath(output_path)}")
