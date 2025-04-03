"""
Tests dataframe library frontend (no triggering of execution).
"""

import bodo.pandas as pd


def test_read_join_filter_proj(datapath):
    df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    df2 = pd.read_parquet(datapath("dataframe_library/df2.parquet"))
    df3 = df1.merge(df2, on="A")
    df3 = df3[df3.A > 3]
    df3["B", "C"]
