"""Tests for dictionary-encoded string reads in the DataFrame library Iceberg
path (BODO_TEST_ICEBERG_DICT_ENCODE=auto/1/0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

import bodo.pandas as bpd
from bodo.tests.utils import _test_equal

pytestmark = [pytest.mark.iceberg]

N_ROWS = 20000


@pytest.fixture
def str_table(tmp_path):
    """Create a local Iceberg table with a low-cardinality string column
    (should be dict-encoded by the heuristic), a high-cardinality string
    column (should not) and an integer column."""
    warehouse = tmp_path / "warehouse"
    data_dir = warehouse / "data"
    data_dir.mkdir(parents=True)

    pdf = _get_pandas_data()
    pa_table = pa.Table.from_pandas(pdf)
    n = len(pdf)
    for i in range(2):
        sub = pa_table.slice(i * n // 2, n // 2)
        pq.write_table(sub, data_dir / f"f{i}.parquet")

    from bodo.io.iceberg.catalog.dir import DirCatalog

    catalog = DirCatalog("test", warehouse=str(warehouse))
    catalog.create_namespace("t")
    schema = Schema(
        NestedField(1, "LOW", StringType(), required=False),
        NestedField(2, "HIGH", StringType(), required=False),
        NestedField(3, "VAL", LongType(), required=False),
    )
    table = catalog.create_table("t.str_table", schema=schema)
    table.add_files([str(data_dir / f"f{i}.parquet") for i in range(2)])
    return str(warehouse)


def _get_pandas_data():
    return pd.DataFrame(
        {
            "LOW": [f"v{i % 20}" for i in range(N_ROWS)],
            "HIGH": [str(i) for i in range(N_ROWS)],
            "VAL": np.arange(N_ROWS),
        }
    )


def _read_table(location):
    return bpd.read_iceberg("t.str_table", location=location)


def test_dict_encode_auto_selection(str_table):
    """The auto heuristic dict-encodes low-cardinality string columns but not
    high-cardinality ones."""
    df = _read_table(str_table)
    assert str(df["LOW"].dtype).startswith("dictionary<values=string")
    assert df["HIGH"].dtype == pd.ArrowDtype(pa.large_string())

    # Values match Pandas (sort by the unique VAL column since row order
    # across ranks differs from Pandas)
    df = df.sort_values("VAL").reset_index(drop=True)
    pdf = _get_pandas_data().sort_values("VAL").reset_index(drop=True)
    _test_equal(df, pdf, check_dtype=False)


def test_dict_encode_knobs(str_table, monkeypatch):
    """Knob=1 dict-encodes all string columns, knob=0 disables dict
    encoding."""
    monkeypatch.setenv("BODO_TEST_ICEBERG_DICT_ENCODE", "1")
    df = _read_table(str_table)
    assert str(df["HIGH"].dtype).startswith("dictionary<")

    monkeypatch.setenv("BODO_TEST_ICEBERG_DICT_ENCODE", "0")
    df = _read_table(str_table)
    assert df["LOW"].dtype == pd.ArrowDtype(pa.large_string())
    assert df["HIGH"].dtype == pd.ArrowDtype(pa.large_string())


def test_dict_encode_ops(str_table):
    """Groupby, filter, sort and string methods work on dict-encoded
    columns and match Pandas."""
    df = _read_table(str_table)
    pdf = _get_pandas_data()

    # Groupby on the dict column (sort both sides since groupby output order
    # differs across ranks)
    res = df.groupby("LOW", as_index=False)["VAL"].sum().sort_values("LOW")
    expected = pdf.groupby("LOW", as_index=False)["VAL"].sum().sort_values("LOW")
    _test_equal(res, expected, reset_index=True)

    # Filter on the dict column
    res = df[df["LOW"] == "v3"]
    expected = pdf[pdf["LOW"] == "v3"]
    _test_equal(res, expected, sort_output=True, reset_index=True)

    # Sort by the dict column
    res = df.sort_values(["LOW", "VAL"]).head(50)
    expected = pdf.sort_values(["LOW", "VAL"]).head(50)
    _test_equal(res, expected, sort_output=True, reset_index=True)

    # String method on the dict column
    res = df["LOW"].str.upper()
    expected = pdf["LOW"].str.upper()
    _test_equal(res, expected, sort_output=True, reset_index=True)


def test_dict_encode_join_mismatch(str_table):
    """Joining a dict-encoded Iceberg scan with a plain string in-memory
    DataFrame works in both probe/build directions and matches Pandas."""
    df = _read_table(str_table)
    pdf = _get_pandas_data()
    ref = pd.DataFrame(
        {"LOW": [f"v{i}" for i in range(10)], "RATE": [i / 10 for i in range(10)]}
    )
    ref_bodo = bpd.from_pandas(ref)

    # Dict-encoded scan on the probe side, string build side
    res = df.merge(ref_bodo, on="LOW", how="inner")
    expected = pdf.merge(ref, on="LOW", how="inner")
    _test_equal(res, expected, sort_output=True, reset_index=True)

    # Dict-encoded scan on the build side, string probe side
    res = ref_bodo.merge(df, on="LOW", how="inner")
    expected = ref.merge(pdf, on="LOW", how="inner")
    _test_equal(res, expected, sort_output=True, reset_index=True)
