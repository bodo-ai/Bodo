# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Unittests for objmode blocks
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_type_register():
    """test bodo.register_type() including error checking"""
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)
    bodo.register_type("my_type1", df_type1)

    def impl():
        with bodo.objmode(df="my_type1"):
            df = pd.DataFrame({"A": [1, 2, 5]})
        return df

    check_func(impl, (), is_out_distributed=False)

    # error checking
    with pytest.raises(BodoError, match="type name 'my_type1' already exists"):
        bodo.register_type("my_type1", df_type1)
    with pytest.raises(BodoError, match="type name should be a string"):
        bodo.register_type(3, df_type1)
    with pytest.raises(BodoError, match="type value should be a valid data type"):
        bodo.register_type("mt", 3)


def test_type_check():
    """test type checking for objmode output values"""

    # A is specified as int but return value has strings
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)

    def impl():
        with bodo.objmode(df=df_type1):
            df = pd.DataFrame({"A": ["abc", "bc"]})
        return df

    with pytest.raises(BodoError, match="Invalid objmode data type specified"):
        bodo.jit(impl)()


def test_df_dist_fix():
    """test fixing dist attribute of DataFrameType if it doesn't match"""

    # dist defaults to REP in typeof but metadata for df2 specifies 1D_Var
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)

    def impl():
        df2 = pd.DataFrame({"A": np.arange(10)})
        with bodo.objmode(df=df_type1):
            df = df2[["A"]]
        return df

    check_func(impl, (), only_1D=True)


def test_df_index_fix():
    """test dropping numeric index from objmode output dataframe if necessary"""

    # Index defaults to RangeIndex
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df_type1 = bodo.typeof(df1)

    def impl():
        with bodo.objmode(df=df_type1):
            df = pd.DataFrame({"A": np.arange(10)}, index=np.arange(10) + 1)
        return df

    check_func(impl, (), reset_index=True, only_seq=True)


def test_df_type_class():
    """test dropping numeric index from objmode output dataframe if necessary"""

    def impl():
        with bodo.objmode(df=bodo.DataFrameType):
            df = pd.DataFrame({"A": np.arange(10)}, index=np.arange(10) + 1)
        return df

    with pytest.raises(
        BodoError, match="objmode type annotations require full data types"
    ):
        bodo.jit(impl)()


def test_df_index_name_fix():
    """test dropping index name from objmode output dataframe if necessary"""

    df_type1 = bodo.typeof(pd.DataFrame({"B": [1.1, 2.2, 3.2]}, index=[0, 1, 2]))

    def impl():
        df2 = pd.DataFrame({"A": np.arange(10), "B": np.ones(10)})
        with bodo.objmode(df=df_type1):
            df = df2.groupby("A").sum()
        return df

    check_func(
        impl,
        (),
        additional_compiler_arguments={"distributed": False},
        reset_index=True,
        only_seq=True,
    )


def test_reflected_list():
    """make sure specifying reflected list data type doesn't fail"""

    t = bodo.typeof([1, 2, 3])

    def impl():
        with bodo.objmode(a=t):
            a = [1, 2, 3]
        return a

    check_func(impl, (), only_seq=True)


def test_df_table_format():
    """test handling table_format mismatch in df type"""

    # user specified type has table_format=False
    n_cols = max(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD, 1)
    df_type = bodo.DataFrameType(
        tuple(bodo.int64[::1] for _ in range(n_cols)),
        bodo.RangeIndexType(),
        tuple(i for i in range(n_cols)),
    )

    def impl():
        with bodo.objmode(df=df_type):
            df = pd.DataFrame({i: [1, 2, 3] for i in range(n_cols)})
        return df

    check_func(impl, (), only_seq=True)


def test_df_column_order():
    """test handling column order mismatch in df type"""

    # user specified type doesn't match output's column order
    df1 = pd.DataFrame(
        {
            "D": pd.date_range("2017-01-03", periods=3),
            "C": ["a", "ab", "cd"],
            "A": [1, 2, 3],
            "B": [1.1, 1.1, 2.2],
        }
    )
    df_type = bodo.typeof(df1)

    def impl():
        with bodo.objmode(df=df_type):
            df = pd.DataFrame(
                {
                    "C": ["a", "ab", "cd"],
                    "A": [1, 2, 3],
                    "D": pd.date_range("2017-01-03", periods=3),
                    "B": [1.1, 1.1, 2.2],
                }
            )
        return df

    check_func(impl, (), reorder_columns=True, only_seq=True)


def test_scalar_cast():
    """make sure objmode works if only minor scalar type cast is needed"""

    # int to int
    def impl1():
        with bodo.objmode(a="uint32"):
            a = 4
        return a

    # float to float
    def impl2():
        with bodo.objmode(a="float32"):
            a = 4.1
        return a

    # float to int
    def impl3():
        with bodo.objmode(a="int32"):
            a = 4.0000001
        return a

    # int to float
    def impl4():
        with bodo.objmode(a="float64"):
            a = 4
        return a

    check_func(impl1, (), only_seq=True)
    check_func(impl2, (), only_seq=True)
    check_func(impl3, (), only_seq=True)
    check_func(impl4, (), only_seq=True)
