"""
Test using BodoSQL inside Bodo JIT functions
"""
import bodosql
import pandas as pd

# Copyright (C) 2022 Bodo Inc. All rights reserved.
import bodo


def test_demo1(datapath, memory_leak_check):
    """test for demo1.py"""

    @bodo.jit
    def f(ss_file, i_file):
        # NOTE: in the actual demo, we don't have the limits on the number of rows
        # However, we run out of memory on the CI machine, so we limit the number of rows
        df1 = pd.read_parquet(ss_file).head(100000)
        df2 = pd.read_parquet(i_file)
        bc = bodosql.BodoSQLContext({"store_sales": df1, "item": df2})
        sale_items = bc.sql(
            "select * from store_sales join item on store_sales.ss_item_sk=item.i_item_sk"
        )
        count = sale_items.groupby("ss_customer_sk")["i_class_id"].agg(
            lambda x: (x == 1).sum()
        )
        return count

    ss_file = datapath("tpcxbb-test-data/store_sales")
    i_file = datapath("tpcxbb-test-data/item")
    S = f(ss_file, i_file)
    # just basic sanity check
    # TODO(ehsan): compare results with SparkSQL?
    assert isinstance(S, pd.Series) and S.name == "i_class_id" and len(S) > 0


def test_df_type_error():
    """Tests that bodosql inlining correctly handles a preceding df setitem"""

    @bodo.jit()
    def small_reproducer(df):
        df["test"] = pd.Series([123] * 10)
        bc = bodosql.BodoSQLContext({"df1": df})
        df2 = bc.sql("""select test from df1""")

    df = pd.DataFrame({"a": [1] * 10})

    small_reproducer(df)
