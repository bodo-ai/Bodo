"""
Test example functions that mix SQL and Python inside
JIT functions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.
import pandas as pd

import bodosql
from bodo.tests.utils import check_func


def test_count_head(datapath, memory_leak_check):
    """
    Check that computing an aggregation is compatible with
    a function like head that modify the index.

    This bug was spotted by Anudeep while trying to breakdown
    a query into components.

    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"t1": bodosql.TablePath(filename, "parquet")})
        df = bc.sql("select count(B) as cnt from t1")
        return df.head()

    filename = datapath("sample-parquet-data/no_index.pq")
    read_df = pd.read_parquet(filename)
    count = read_df.B.count()
    expected_output = pd.DataFrame({"cnt": count}, index=pd.Index([0]))
    check_func(impl, (filename,), py_output=expected_output, is_out_distributed=False)
