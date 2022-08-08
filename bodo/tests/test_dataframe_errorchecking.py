import re
import string
from decimal import Decimal

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.fixture(params=bodo.hiframes.pd_dataframe_ext.dataframe_unsupported)
def df_unsupported_method(request):
    return request.param


@pytest.fixture(params=bodo.hiframes.pd_dataframe_ext.dataframe_unsupported_attrs)
def df_unsupported_attr(request):
    return request.param


@pytest.mark.slow
def test_df_filter_err_check(memory_leak_check):
    """
    Tests df.filter with the 'items' and 'like' args both passed.
    """

    df = pd.DataFrame(
        np.array(([1, 2, 3], [4, 5, 6])),
        index=["mouse", "rabbit"],
        columns=["one", "two", "three"],
    )

    def test_multiple_args(df):
        return df.filter(items=["one", "three"], like="bla")

    def test_variable_like_arg(df, like):
        return df.filter(like=like)

    def test_non_list_items(df):
        return df.filter(items=7)

    def test_invalid_axis_str(df):
        return df.filter(axis="one", like="bla")

    def test_invalid_axis_int(df):
        return df.filter(axis=12, like="bla")

    def test_no_kwd_args(df):
        return df.filter()

    def test_axis_unsupporteed_err_int(df):
        return df.filter(axis=0, like="bla")

    def test_axis_unsupporteed_err_str(df):
        return df.filter(axis="index", like="bla")

    with pytest.raises(
        BodoError,
        match="keyword arguments `items`, `like`, and `regex` are mutually exclusive",
    ):
        bodo.jit(test_multiple_args)(df)

    with pytest.raises(
        BodoError,
        match="argument 'like' must be a constant string",
    ):
        bodo.jit(test_variable_like_arg)(df, "bla")

    with pytest.raises(
        BodoError,
        match="argument 'items' must be a list of constant strings",
    ):
        bodo.jit(test_non_list_items)(df)

    with pytest.raises(
        BodoError,
        match=re.escape(
            'DataFrame.filter(): keyword arguments `axis` must be either "index" or "columns" if string'
        ),
    ):
        bodo.jit(test_invalid_axis_str)(df)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.filter(): keyword arguments `axis` must be either 0 or 1 if integer"
        ),
    ):
        bodo.jit(test_invalid_axis_int)(df)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.filter(): one of keyword arguments `items`, `like`, and `regex` must be supplied"
        ),
    ):
        bodo.jit(test_no_kwd_args)(df)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.filter(): filtering based on index is not supported."
        ),
    ):
        bodo.jit(test_axis_unsupporteed_err_int)(df)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.filter(): filtering based on index is not supported."
        ),
    ):
        bodo.jit(test_axis_unsupporteed_err_str)(df)


@pytest.mark.slow
def test_df_iat_getitem_nonconstant(memory_leak_check):
    """
    Tests DataFrame.iat getitem when the column index isn't a constant.
    """

    def test_impl(idx):
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.iat[0, idx[0]]

    with pytest.raises(
        BodoError,
        match="DataFrame.iat getitem: column index must be a constant integer",
    ):
        bodo.jit(test_impl)([0])


@pytest.mark.slow
def test_df_iat_getitem_str(memory_leak_check):
    """
    Tests DataFrame.iat getitem when the row index isn't an integer.
    """

    def test_impl():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.iat["a", 0]

    with pytest.raises(
        BodoError,
        match="DataFrame.iat: iAt based indexing can only have integer indexers",
    ):
        bodo.jit(test_impl)()


@pytest.mark.slow
def test_df_iat_setitem_nonconstant(memory_leak_check):
    """
    Tests DataFrame.iat setitem when the column index isn't a constant.
    """

    def test_impl(idx):
        df = pd.DataFrame({"A": np.random.randn(10)})
        df.iat[0, idx[0]] = 2
        return df

    with pytest.raises(
        BodoError,
        match="DataFrame.iat setitem: column index must be a constant integer",
    ):
        bodo.jit(test_impl)([0])


@pytest.mark.slow
def test_df_iat_setitem_str(memory_leak_check):
    """
    Tests DataFrame.iat setitem when the row index isn't an integer.
    """

    def test_impl():
        df = pd.DataFrame({"A": np.random.randn(10)})
        df.iat["a", 0] = 2
        return df

    with pytest.raises(
        BodoError,
        match="DataFrame.iat: iAt based indexing can only have integer indexers",
    ):
        bodo.jit(test_impl)()


@pytest.mark.slow
def test_df_iat_setitem_immutable_array(memory_leak_check):
    """
    Tests DataFrame.iat setitem with an immutable array.
    """

    def test_impl(df):
        df.iat[0, 0] = [1, 2, 2]
        return df

    df = pd.DataFrame({"A": [[1, 2, 3], [2, 1, 1], [1, 2, 3], [2, 1, 1]]})

    with pytest.raises(
        BodoError,
        match="DataFrame setitem not supported for column with immutable array type .*",
    ):
        bodo.jit(test_impl)(df)


def test_df_sample_error(memory_leak_check):
    def test_impl(df):
        return df.sample(n=10, frac=0.5)

    df = pd.DataFrame({"A": [[1, 2, 3], [2, 1, 1], [1, 2, 3], [2, 1, 1]]})

    with pytest.raises(
        BodoError,
        match="only one of n and frac option can be selected",
    ):
        bodo.jit(test_impl)(df)


def test_df_info_unsupported(memory_leak_check):
    def test_impl(df):
        return df.info(verbose=True)

    df = pd.DataFrame({"A": [[1, 2, 3], [2, 1, 1], [1, 2, 3], [2, 1, 1]]})

    with pytest.raises(
        BodoError,
        match="verbose parameter only supports default value",
    ):
        bodo.jit(test_impl)(df)


@pytest.fixture
def df_fillna():
    return pd.DataFrame({"A": [1, np.nan, 2]})


@pytest.mark.slow
def test_df_fillna_limit_error(df_fillna, memory_leak_check):
    match = "limit parameter only supports default value None"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: df_fillna.fillna(0, limit=2))()


@pytest.mark.slow
def test_df_fillna_downcast_error(df_fillna, memory_leak_check):
    df_fillna["A"] = df_fillna["A"].astype(np.float64)
    match = "downcast parameter only supports default value None"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: df_fillna.fillna(0, downcast="infer"))()


def test_df_fillna_axis_error(df_fillna, memory_leak_check):
    match = "'axis' argument not supported"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: df_fillna.fillna(0, axis=1))()


@pytest.mark.slow
def test_df_fillna_dict_value_error(df_fillna, memory_leak_check):
    message = "Cannot use value type DictType"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_fillna.fillna({"A": 3}))()


def test_df_fillna_both_value_method_args(df_fillna, memory_leak_check):
    message = re.escape("DataFrame.fillna(): Cannot specify both 'value' and 'method'.")
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_fillna.fillna(value=1, method="ffill"))()


def test_df_fillna_neither_value_method_args(df_fillna, memory_leak_check):
    message = re.escape("DataFrame.fillna(): Must specify one of 'value' and 'method'.")
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_fillna.fillna())()


@pytest.fixture
def df_replace():
    return pd.DataFrame({"A": np.arange(12)})


@pytest.mark.slow
def test_df_replace_inplace_error(df_replace):
    message = "inplace parameter only supports default value False"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_replace.replace(1, 100, inplace=True))()


@pytest.mark.slow
def test_df_replace_limit_error(df_replace):
    message = "limit parameter only supports default value None"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_replace.replace(1, 100, limit=1))()


@pytest.mark.slow
def test_df_replace_regex_error(df_replace):
    message = "regex parameter only supports default value False"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_replace.replace(1, 100, regex=True))()


@pytest.mark.slow
def test_df_replace_pad_error(df_replace):
    message = "method parameter only supports default value pad"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: df_replace.replace(1, 100, method=None))()


@pytest.mark.slow
def test_df_rename_errors(memory_leak_check):
    """
    Tests BodoErrors from DataFrame.rename.
    """

    def test_impl1():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(index=[0])

    def test_impl2():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(level=0)

    def test_impl3():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(errors="raise")

    def test_impl4():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(inplace=None)

    def test_impl5():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename({"A": "B"}, columns={"A": "B"})

    def test_impl6():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename({"A": "B"})

    def test_impl7(cols):
        d = {"A": cols[0]}
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(d, axis=1)

    def test_impl8(cols):
        d = {"A": cols[0]}
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(columns=d)

    def test_impl9():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename(columns={"A": "B"}, axis=1)

    def test_impl10():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename()

    def test_impl11():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.rename({"A": "B"}, axis=0)

    unsupported_arg_err_msg = "DataFrame.rename.* parameter only supports default value"

    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl1)()
    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl2)()
    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl3)()
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: 'inplace' keyword only supports boolean constant assignment",
    ):
        bodo.jit(test_impl4)()
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: Cannot specify both 'mapper' and 'columns'",
    ):
        bodo.jit(test_impl5)()
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: 'mapper' only supported with axis=1",
    ):
        bodo.jit(test_impl6)()
    with pytest.raises(
        BodoError,
        match="'mapper' argument to DataFrame.rename.* should be a constant dictionary",
    ):
        bodo.jit(test_impl7)(["B", "C"])
    with pytest.raises(
        BodoError,
        match="'columns' argument to DataFrame.rename.* should be a constant dictionary",
    ):
        bodo.jit(test_impl8)(["B", "C"])
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: Cannot specify both 'axis' and 'columns'",
    ):
        bodo.jit(test_impl9)()
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: must pass columns either via 'mapper' and 'axis'=1 or 'columns'",
    ):
        bodo.jit(test_impl10)()
    with pytest.raises(
        BodoError,
        match="DataFrame.rename.*: 'mapper' only supported with axis=1",
    ):
        bodo.jit(test_impl11)()


@pytest.mark.slow
def test_df_set_index_errors(memory_leak_check):
    """
    Tests BodoErrors from DataFrame.set_index.
    """

    def test_impl1():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.set_index("A", inplace=True)

    def test_impl2():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.set_index("A", append=True)

    def test_impl3():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.set_index("A", verify_integrity=True)

    def test_impl4():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.set_index(["A"])

    unsupported_arg_err_msg = (
        "DataFrame.set_index.* parameter only supports default value"
    )

    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl1)()
    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl2)()
    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl3)()
    with pytest.raises(
        BodoError,
        match="DataFrame.set_index.*: 'keys' must be a constant string",
    ):
        bodo.jit(test_impl4)()


@pytest.mark.slow
def test_df_reset_index_errors(memory_leak_check):
    """
    Tests BodoErrors from DataFrame.rename_index.
    """

    def test_impl1():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.reset_index(col_fill="*")

    def test_impl2():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.reset_index(col_level=1)

    def test_impl3():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.reset_index(drop=None)

    def test_impl4():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.reset_index(level=1)

    def test_impl5():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.reset_index(inplace=None)

    unsupported_arg_err_msg = (
        "DataFrame.reset_index.* parameter only supports default value"
    )

    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl1)()
    with pytest.raises(
        BodoError,
        match=unsupported_arg_err_msg,
    ):
        bodo.jit(test_impl2)()
    with pytest.raises(
        BodoError,
        match="DataFrame.reset_index.*: 'drop' parameter should be a constant boolean value",
    ):
        bodo.jit(test_impl3)()
    with pytest.raises(
        BodoError,
        match="DataFrame.reset_index.*: only dropping all index levels supported",
    ):
        bodo.jit(test_impl4)()
    with pytest.raises(
        BodoError,
        match="DataFrame.reset_index.*: 'inplace' parameter should be a constant boolean value",
    ):
        bodo.jit(test_impl5)()


@pytest.mark.slow
def test_df_head_errors(memory_leak_check):
    def impl():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.head(5.0)

    with pytest.raises(BodoError, match="DataFrame.head.*: 'n' must be an Integer"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_tail_errors(memory_leak_check):
    def impl():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.tail(5.0)

    with pytest.raises(BodoError, match="Dataframe.tail.*: 'n' must be an Integer"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_drop_errors(memory_leak_check):
    def impl1():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(index=[0, 1])

    def impl2():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(level=0)

    def impl3():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(errors="warn")

    def impl4():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(inplace=None)

    def impl5():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(labels="A", columns="A")

    def impl6():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(labels="A")

    def impl7(labels):
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(labels=labels[0], axis=1)

    def impl8(labels):
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(columns=labels[0], axis=1)

    def impl9():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop(columns=["A", "C"], axis=1)

    def impl10():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop()

    unsupported_arg_err_msg = "DataFrame.drop.* parameter only supports default value"
    const_err_msg = "constant list of columns expected for labels in DataFrame.drop.*"
    col = ["A"]
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl2)()
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl3)()
    with pytest.raises(
        BodoError,
        match="DataFrame.drop.*: 'inplace' parameter should be a constant bool",
    ):
        bodo.jit(impl4)()
    with pytest.raises(
        BodoError, match="Dataframe.drop.*: Cannot specify both 'labels' and 'columns'"
    ):
        bodo.jit(impl5)()
    with pytest.raises(BodoError, match="DataFrame.drop.*: only axis=1 supported"):
        bodo.jit(impl6)()
    with pytest.raises(BodoError, match=const_err_msg):
        bodo.jit(impl7)(col)
    with pytest.raises(BodoError, match=const_err_msg):
        bodo.jit(impl8)(col)
    with pytest.raises(
        BodoError, match="DataFrame.drop.*: column C not in DataFrame columns .*"
    ):
        bodo.jit(impl9)()
    with pytest.raises(
        BodoError,
        match="DataFrame.drop.*: Need to specify at least one of 'labels' or 'columns'",
    ):
        bodo.jit(impl10)()


@pytest.mark.slow
def test_df_drop_duplicates_errors(memory_leak_check):
    def impl1():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop_duplicates(keep="last")

    def impl2():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        df.drop_duplicates(inplace=True)

    def impl3():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.drop_duplicates(ignore_index=True)

    unsupported_arg_err_msg = (
        "DataFrame.drop_duplicates.* parameter only supports default value"
    )
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl2)()
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl3)()


@pytest.mark.slow
def test_df_duplicated_errors(memory_leak_check):
    def impl1():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.duplicated(keep="last")

    def impl2():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.duplicated(subset=["A"])

    unsupported_arg_err_msg = (
        "DataFrame.duplicated.* parameter only supports default value"
    )
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl2)()


@pytest.mark.slow
def test_df_apply_all_args(memory_leak_check):
    """Test DataFrame.apply with unsupported and wrong arguments"""

    def test_axis(df):
        return df.apply(lambda r: r.A)

    def test_result_type(df):
        return df.apply(lambda r: r.A, result_type="expand", axis=1)

    def test_raw(df):
        return df.apply(lambda r: r.A, raw=True, axis=1)

    def test_np_func(df):
        return df.apply(np.abs, axis=1)

    df = pd.DataFrame({"A": [1, 2, -1, -3, -4, 0]})

    with pytest.raises(
        BodoError,
        match=r"Dataframe.apply\(\): only axis=1 supported for user-defined functions",
    ):
        bodo.jit(test_axis)(df)

    with pytest.raises(BodoError, match="parameter only supports default"):
        bodo.jit(test_result_type)(df)
        bodo.jit(test_raw)(df)

    with pytest.raises(BodoError, match="does not support built-in functions"):
        bodo.jit(test_np_func)(df)


@pytest.mark.slow
def test_dataframe_idxmax_unordered_cat(memory_leak_check):
    """Test that DataFrame.idxmax throws an appropriate error with an unordered
    Categorical Column"""

    def impl(df):
        return df.idxmax()

    df = pd.DataFrame({"A": pd.Categorical([1, 2, 5, None, 2], ordered=False)})

    match = "DataFrame.idxmax.*: categorical columns must be ordered"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_dataframe_idxmin_unordered_cat(memory_leak_check):
    """Test that DataFrame.idxmin throws an appropriate error with an unordered
    Categorical Column"""

    def impl(df):
        return df.idxmin()

    df = pd.DataFrame({"A": pd.Categorical([1, 2, 5, None, 2], ordered=False)})

    match = "DataFrame.idxmin.*: categorical columns must be ordered"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(df)


# --------------------- df.describe() ----------------------
@pytest.mark.slow
def test_describe_args(memory_leak_check):
    """Test df.describe with unsupported arguments"""

    def impl_percentiles(df):
        return df.describe(percentiles=[0.25, 0.5, 0.75])

    def impl_include(df):
        return df.describe(include="all")

    def impl_exclude(df):
        return df.describe(exclude=[np.number])

    def impl_datetime_is_numeric(df):
        return df.describe(datetime_is_numeric=False)

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
        }
    )

    err_msg = "parameter only supports default value"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_percentiles)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_include)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_exclude)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_datetime_is_numeric)(df)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                # Decimal
                "A": pd.Series(
                    [
                        Decimal("1.6"),
                        Decimal("-0.2"),
                        Decimal("44.2"),
                        np.nan,
                        Decimal("0"),
                    ]
                ),
                # timedelta
                "C": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
                # list
                "D": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4], [2]]),
                # tuple
                "E": pd.Series([(1, 2), (3,), (5, 4, 6), (-1, 3, 4), (2,)]),
                # Categorical
                "F": pd.Categorical([1, 2, 5, 5, 3], ordered=True),
                # boolean Array
                "H": [True, True, False, True, True],
                # string
                "I": ["ab", "cd", "ef", "gh", "mm"],
            }
        )
    ],
)
def test_describe_unsupported_types(df, memory_leak_check):
    """Test df.describe with its unsupported Bodo types"""

    def impl(df):
        return df.describe()

    err_msg = "only supports numeric columns"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_dataframe_iloc_getrow_invalid_col(memory_leak_check):
    """Test that DataFrame.iloc throws an appropriate error when a col index
    is out of bounds.
    """

    def impl1(df):
        return df.iloc[1, -1]

    def impl2(df):
        return df.iloc[1, 1]

    def impl3(df):
        return df.iloc[1, [0, 1]]

    def impl4(df):
        return df.iloc[1, [-1, 0]]

    def impl5(df):
        return df.iloc[1, [-1.0, 0]]

    def impl6(df, value):
        return df.iloc[1, value[0]]

    def impl7(df):
        values = list(pd.Series(np.array([0, 1, 2, 1, 2])).unique())
        return df.iloc[1, values]

    df = pd.DataFrame({"A": pd.Categorical([1, 2, 5, None, 2], ordered=False)})

    match = r"df.iloc: column integer must refer to a valid column number"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl2)(df)
    match = r"df.iloc: column list must be integers referring to a valid column number"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl3)(df)
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl4)(df)
    match = r"idx2 in df.iloc\[idx1, idx2\] should be a constant integer or constant list of integers"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl5)(df)
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl6)(df, [1, 2])
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl7)(df)


@pytest.mark.slow
def test_dataframe_iloc_bad_index(memory_leak_check):
    def impl1(df):
        return df.iloc["a", 0]

    def impl2(df):
        return df.iloc["a", 0:4]

    df = pd.DataFrame(
        {"A": np.random.randn(10)}, index=list(string.ascii_lowercase)[:10]
    )
    error_msg = "df.iloc\\[] getitem using .* not supported"
    with pytest.raises(
        BodoError,
        match=error_msg,
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError,
        match=error_msg,
    ):
        bodo.jit(impl2)(df)


def test_non_numba_err(memory_leak_check):
    def test():
        return x

    with pytest.raises(
        numba.TypingError,
        match="name 'x' is not defined",
    ):
        bodo.jit(test)()


@pytest.mark.slow
def test_df_head_too_many_args(memory_leak_check):
    """
    Test that a function split into typing and lowering still checks
    the correct number of arguments.
    """

    def impl(df):
        return df.head(3, 5)

    df = pd.DataFrame({"A": [1, 23, 4, 1, 1, 7]})
    with pytest.raises(
        BodoError,
        match="Too many arguments specified. Function takes 1 argument, but 2 were provided.",
    ):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_df_corr_repeat_kws(memory_leak_check):
    """
    Test that a function split into typing and lowering still checks
    that there are no kws that repeat an args.
    """

    def impl(df):
        return df.corr("pearson", method="kendall")

    df = pd.DataFrame({"A": [1, 23, 4, 1, 1, 7]})
    with pytest.raises(
        BodoError,
        match="multiple values for argument 'method'",
    ):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_df_corr_unknown_kws(memory_leak_check):
    """
    Test that a function split into typing and lowering still checks
    that there are no kws that are unknown.
    """

    def impl(df):
        return df.corr("pearson", unknown_arg=5)

    df = pd.DataFrame({"A": [1, 23, 4, 1, 1, 7]})
    with pytest.raises(
        BodoError,
        match="got an unexpected keyword argument 'unknown_arg'",
    ):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_astype_non_constant_string(memory_leak_check):
    """
    Checks that calling DataFrame.astype(str_value) with a string that
    is not a compile time constant will produce a reasonable BodoError.
    """

    def impl(df, type_str):
        return df.astype(type_str[0])

    df = pd.DataFrame({"A": [1, 2, 3, 4] * 10})
    type_str = ["uint64"]

    with pytest.raises(
        BodoError,
        match="DataFrame.astype\\(\\): 'dtype' when passed as string must be a constant value",
    ):
        bodo.jit(impl)(df, type_str)


@pytest.mark.slow
def test_concat_const_args(memory_leak_check):
    """
    make sure proper error is raised when axis/ignore_index arguments to pd.concat
    are not constant
    """

    def impl1(S1, S2, flag):
        axis = 0
        if flag:
            axis = 1
        return pd.concat((S1, S2), axis=axis)

    def impl2(S1, S2, flag):
        ignore_index = True
        if flag:
            ignore_index = False
        return pd.concat((S1, S2), ignore_index=ignore_index)

    S1 = pd.Series([1, 2, 3], name="A")
    S2 = pd.Series([3, 4, 5], name="B")

    with pytest.raises(
        BodoError,
        match=r"'axis' should be a constant integer",
    ):
        bodo.jit(impl1)(S1, S2, True)

    with pytest.raises(
        BodoError,
        match=r"'ignore_index' should be a constant boolean",
    ):
        bodo.jit(impl2)(S1, S2, True)


def test_df_getitem_non_const_columname_error(memory_leak_check):

    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        flag = len(df) > 10
        if flag:
            col = g(a)
        else:
            col = df.columns[0]
        return df[col]

    message = r"df\[\] getitem selecting a subset of columns requires providing constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(BodoError, match=message):
        f(df, "A")

    df = pd.DataFrame({1: [1, 2, 3]})
    with pytest.raises(BodoError, match=message):
        f(df, 1)


def test_df_getitem_non_const_columname_list_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        flag = len(df) > 10
        if flag:
            cols = g(a)
        else:
            cols = [df.columns[0]]
        return df[cols]

    message = r"df\[\] getitem using .* not supported. If you are trying to select a subset of the columns, you must provide the column names you are selecting as a constant. See https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(BodoError, match=message):
        f(df, ["A"])

    df = pd.DataFrame({1: [1, 2, 3]})
    with pytest.raises(BodoError, match=message):
        f(df, [1])


@pytest.mark.slow
def test_df_loc_getitem_non_const_columname_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        x = df.loc[0, g(a)]
        return x

    @bodo.jit
    def f2(df, a):
        return df.loc[1:, g(a)]

    df = pd.DataFrame({"A": np.arange(10)})
    # TODO: add when loc supports indexing with integer col names
    # df = pd.DataFrame({1: [1, 2, 3]})

    message = r"DataFrame.loc\[\] getitem \(location-based indexing\) requires constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
    with pytest.raises(BodoError, match=message):
        f(df, "A")
    with pytest.raises(BodoError, match=message):
        f2(df, "A")


def test_df_loc_getitem_non_const_columname_list_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        flag = len(df) > 10
        if flag:
            cols = g(a)
        else:
            cols = ["A"]
        x = df.loc[0, cols]
        return x

    @bodo.jit
    def f2(df, a):
        flag = len(df) > 10
        if flag:
            cols = g(a)
        else:
            cols = ["A"]
        return df.loc[1:, cols]

    df = pd.DataFrame({"A": np.arange(10)})
    # TODO: add when loc supports indexing with integer col names
    # df = pd.DataFrame({1: [1, 2, 3]})

    message = r"DataFrame.loc\[\] getitem \(location-based indexing\) using .* not supported yet. If you are trying to select a subset of the columns by passing a list of column names, that list must be a compile time constant. See https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    # TODO: re add this check when we support it in df.loc
    # with pytest.raises(BodoError, match=message):
    #     f(df, ["A"])
    with pytest.raises(BodoError, match=message):
        f2(df, ["A"])


@pytest.mark.slow
def test_df_iloc_getitem_non_const_columname_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        x = df.iloc[0, g(a)]
        return x

    @bodo.jit
    def f2(df, a):
        return df.iloc[1:, g(a)]

    df = pd.DataFrame({"A": np.arange(10)})

    message = r"idx2 in df.iloc\[idx1, idx2\] should be a constant integer or constant list of integers. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
    with pytest.raises(BodoError, match=message):
        f(df, 1)
    with pytest.raises(BodoError, match=message):
        f2(df, [1])


def test_df_iloc_getitem_non_const_slice_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        flag = len(df) > 10
        if flag:
            cols = g(a)
        else:
            cols = slice(0, 2)
        x = df.iloc[1:, cols]
        return x

    df = pd.DataFrame({"A": np.arange(10)})

    message = r"slice2 in df.iloc\[slice1,slice2\] should be constant. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    with pytest.raises(BodoError, match=message):
        f(df, slice(0, 1))


@pytest.mark.slow
def test_df_iat_getitem_non_const_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        x = df.iat[1, g(a)]
        return x

    df = pd.DataFrame({"A": np.arange(10)})

    message = r"DataFrame.iat getitem: column index must be a constant integer. For more informaton, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    with pytest.raises(BodoError, match=message):
        f(df, 0)


@pytest.mark.slow
def test_df_iat_setitem_non_const_error(memory_leak_check):
    g = bodo.jit(lambda a: a)

    @bodo.jit
    def f(df, a):
        df.iat[1, g(a)] = -1

    df = pd.DataFrame({"A": np.arange(10)})

    message = r"DataFrame.iat setitem: column index must be a constant integer. For more informaton, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."

    with pytest.raises(BodoError, match=message):
        f(df, 0)


@pytest.mark.slow
def test_df_plot_args(memory_leak_check):
    """
    Error checking for types/values of df.plot supported arguments
    """

    def impl1():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(x=1.2, y="B")

    err_msg = "x must be a constant column name, constant integer, or None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()

    err_msg = "x must be a constant column name, constant integer, or None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()

    def impl2():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(y="m")

    err_msg = "column not found"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)()

    def impl3():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(x="A", y=12)

    err_msg = "is out of bounds"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl3)()

    def impl4():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="pie")

    err_msg = "pie plot is not supported"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)()

    def impl5():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(figsize=10)

    err_msg = "figsize must be a constant numeric tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl5)()

    def impl6():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(title=True)

    err_msg = "title must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl6)()

    def impl7():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(legend="X")

    err_msg = "legend must be a boolean type"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl7)()

    def impl8():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(legend="X")

    err_msg = "legend must be a boolean type"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl8)()

    def impl9():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(xticks=3)

    err_msg = "xticks must be a constant tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl9)()

    def impl10():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(yticks=2)

    err_msg = "yticks must be a constant tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl10)()

    def impl11():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(fontsize=3.4)

    err_msg = "fontsize must be an integer"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl11)()

    def impl12():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(xlabel=10)

    err_msg = "xlabel must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl12)()

    def impl13():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(ylabel=10)

    err_msg = "ylabel must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl13)()

    def impl14():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter")

    err_msg = "requires an x and y column"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl14)()

    def impl15():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter", y="B")

    err_msg = "x column is missing."
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl15)()

    def impl16():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter", x="A")

    err_msg = "y column is missing."
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl16)()


def test_invalid_replace_col_data(memory_leak_check):
    """Checks that DataFrameType.replace_col_type throws a reasonable error
    if the column doesn't exist."""
    dummy_df = pd.DataFrame(
        {
            "A": [1],
            "B": ["a"],
        }
    )
    infered_dtype = bodo.typeof(dummy_df)
    with pytest.raises(
        ValueError,
        match="DataFrameType.replace_col_type replaced column must be found in the DataFrameType",
    ):
        new_dtype = infered_dtype.replace_col_type("C", bodo.string_array_type)


def test_dd_map_array_drop_subset(memory_leak_check):
    """
    Test drop_duplicates throws an appropriate error
    when a MapArray column is part of the subset.
    """

    def test_impl(df1):
        df1["B"] = df1["A"].apply(lambda val: {str(j): val for j in range(val)})
        df2 = df1.drop_duplicates(subset="B")
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4]})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Columns ['B'] have dictionary types which cannot be used to drop duplicates. Please consider using the 'subset' argument to skip these columns."
        ),
    ):
        bodo.jit(test_impl)(df1)


def test_dd_map_array_drop_all(memory_leak_check):
    """
    Test drop_duplicates throws an appropriate error
    when a MapArray column is part of the columns dropped
    without a subset.
    """

    def test_impl(df1):
        df1["B"] = df1["A"].apply(lambda val: {str(j): val for j in range(val)})
        df2 = df1.drop_duplicates()
        return df2

    df1 = pd.DataFrame({"A": [3, 3, 3, 1, 4]})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Columns ['B'] have dictionary types which cannot be used to drop duplicates. Please consider using the 'subset' argument to skip these columns."
        ),
    ):
        bodo.jit(test_impl)(df1)


@pytest.mark.slow
def test_df_unsupported_methods(df_unsupported_method):
    """tests that unsupported dataframe methods throw the expected error"""

    df_val = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    func_text = f"""
def impl(df):
    return df.{df_unsupported_method}()
"""

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]

    err_msg = re.escape(f"DataFrame.{df_unsupported_method}() not supported yet")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(df_val)


@pytest.mark.slow
def test_df_unsupported_atrs(df_unsupported_attr):
    """tests that unsupported dataframe attributes throw the expected error"""

    df_val = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    func_text = f"""
def impl(df):
    return df.{df_unsupported_attr}
"""

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]

    err_msg = f"DataFrame.{df_unsupported_attr} not supported yet"

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(df_val)


@pytest.mark.skip("TODO: throw the propper error messages for df.plot.x")
def test_df_plot_submethods():
    """tests that df.plot.x throws a propper error message"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    def impl_1(df):
        foo = df.plot.area()

    err_msg1 = f"pandas.Dataframe.plot (the attribute, not the bound function) not yet supported"
    with pytest.raises(
        BodoError,
        match=err_msg1,
    ):
        bodo.jit(impl_1)(df)


@pytest.mark.slow
def test_df_values_to_numpy_err():
    """Tests that df.values and df.to_numpy() throw a reasonable error message"""

    def impl_1():
        return pd.DataFrame({"A": ["hi"]}).to_numpy()

    def impl_2():
        return pd.DataFrame({"A": ["hi"]}).values

    err_msg1 = re.escape(
        "DataFrame.to_numpy(): only supported for dataframes containing numeric values"
    )
    err_msg2 = re.escape(
        "DataFrame.values: only supported for dataframes containing numeric values"
    )
    with pytest.raises(
        BodoError,
        match=err_msg1,
    ):
        bodo.jit(impl_1)()

    with pytest.raises(
        BodoError,
        match=err_msg2,
    ):
        bodo.jit(impl_2)()


@pytest.mark.slow
def test_df_abs_err():
    """Tests that df.abs() throws a reasonable error on non numeric/timedelta dfs"""

    def impl():
        foo = pd.DataFrame({"A": ["hi"]}).abs()

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.abs(): Only supported for numeric and Timedelta. Encountered array with dtype"
        ),
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_corr_err():
    """Tests that df.corr() throws a reasonable error for empty dataframe"""

    def impl():
        foo = pd.DataFrame().corr()

    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.corr(): requires non-empty dataframe"),
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_cov_err():
    """Tests that df.cov() throws a reasonable error for empty dataframe"""

    def impl():
        foo = pd.DataFrame().cov()

    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.cov(): requires non-empty dataframe"),
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_rolling_unsupported():
    def impl():
        df = pd.DataFrame({"B": [1, 2, 3, 4, 5, 6]}).rolling(1).kurt()

    err_msg = re.escape("pandas.core.window.rolling.Rolling.kurt() not supported yet")
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_error_message_truncates():
    """
    Checks that DataFrames with large number of columns have string representation truncated in error messages
    """

    @bodo.jit
    def impl(df):
        # non-existent column
        return df["qwerty"]

    num_cols_over_3 = 1000
    data_dict = {}
    for i in range(num_cols_over_3):
        for j in range(3):
            modulo = j % 3
            if modulo == 0:
                arr = np.arange(1000) * (i + 1)
            if modulo == 1:
                arr = np.arange(1000) * 0.5 * (i + 1)
            if modulo == 2:
                arr = [f"value{k}" for k in (np.arange(1000) * (i + 1))]
            data_dict[f"Column{(i * 3) + j}"] = arr
    large_df = pd.DataFrame(data_dict)

    with pytest.raises(BodoError) as bodo_err:
        impl(large_df)
    # 1000 is an arbitrary value to allow for small changes due
    # to internal changes for version updates.
    # Without truncation the length is > 10,000
    assert len(bodo_err.value.msg) < 1000


def test_df_first_last_invalid_index():
    def test_impl1(df):
        return df.first("15D")

    def test_impl2(df):
        return df.last("15D")

    df = pd.DataFrame({"A": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))

    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.first(): only supports a DatetimeIndex index"),
    ):
        bodo.jit(test_impl1)(df)

    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.last(): only supports a DatetimeIndex index"),
    ):
        bodo.jit(test_impl2)(df)


def test_df_first_last_invalid_offset():
    def test_impl1(df, off):
        return df.first(off)

    def test_impl2(df, off):
        return df.last(off)

    i = pd.date_range("2018-04-09", periods=10, freq="2D")
    df = pd.DataFrame({"A": pd.Series(np.arange(10))}, index=i)
    off = 2
    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.first(): 'offset' must be an string or DateOffset"),
    ):
        bodo.jit(test_impl1)(df, off)
    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.last(): 'offset' must be an string or DateOffset"),
    ):
        bodo.jit(test_impl2)(df, off)


def test_df_explode_invalid_cols():
    @bodo.jit
    def test_impl(df):
        return df.explode(["A", "B"])

    df = pd.DataFrame(
        {
            "A": [[0, 1, 2], [5], [], [3, 4]] * 3,
            "B": [1, 7, 2, 4] * 3,
            "C": [[1, 2, 3], np.nan, [], [1, 2]] * 3,
        }
    )
    with pytest.raises(
        BodoError,
        match=re.escape("DataFrame.explode(): columns must have array-like entries"),
    ):
        test_impl(df)

    df2 = pd.DataFrame(
        {
            "A": [[0, 1, 2], [5], [], [3, 4]] * 3,
            "B": [[1, 2], [3], [], [1, 2]] * 3,
            "C": [[1, 2, 3], np.nan, [], [1, 2]] * 3,
        }
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "DataFrame.explode(): columns must have matching element counts"
        ),
    ):
        test_impl(df2)


def test_to_parquet_int_colnames(memory_leak_check):
    """
    Test that non-string columns raises an error
    as expected.
    """

    @bodo.jit
    def test_impl(df):
        df.to_parquet("dummy.pq")

    df = pd.DataFrame(
        {
            1: np.arange(100),
            2: np.range(100, 200),
        }
    )
    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.to_parquet(): parquet must have string column names"
        ),
    ):
        test_impl(df)


def test_to_parquet_int_colnames(memory_leak_check):
    """
    Test that non-string columns output from
    pivot raises an error as expected.
    """

    @bodo.jit
    def test_impl(df):
        new_df = df.pivot(index="A", columns="B", values="C")
        new_df.to_parquet("test.pq")

    df = pd.DataFrame(
        {
            "A": np.arange(10),
            "B": np.arange(10),
            "C": np.arange(10),
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.to_parquet(): parquet must have string column names"
        ),
    ):
        test_impl(df)


def test_df_melt_not_common_value_columns():
    """
    Tests that df.melt with 'value_vars' columns of different types raises error as expected.
    """

    def test_impl(id_vars, value_vars):
        return df.melt(id_vars, value_vars)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [1.1, 2.3, 3.1],
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.melt(): columns selected in 'value_vars' must have a unifiable type."
        ),
    ):
        bodo.jit(test_impl)(["A"], ["B", "C"])


def test_df_melt_not_common_value_labels():
    """
    Tests that df.melt with 'value_vars' labels of different types raises error as expected.
    """

    def test_impl(id_vars, value_vars):
        return df.melt(id_vars, value_vars)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            3: [1.1, 2.3, 3.1],
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.melt(): column names selected for 'value_vars' must all share a common int or string type. Please convert your names to a common type using DataFrame.rename()"
        ),
    ):
        bodo.jit(test_impl)(["A"], ["B", 3])


def test_df_melt_invalid_cols(memory_leak_check):
    def test_impl(df, id_vars, value_vars):
        return df.melt(id_vars, value_vars)

    df = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": [1, 3, 5],
            "C": [2, 4, 6],
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame.melt(): currently empty 'value_vars' is unsupported"
        ),
    ):
        # TODO: works with "A", "A" since value_vars will remove those labels found in id_vars
        # however will not work with value_vars=[] due to Numba "ValueError: cannot compute
        # fingerprint of empty list"
        bodo.jit(test_impl)(df, ["A"], ["A"])

    with pytest.raises(BodoError, match="not found"):
        bodo.jit(test_impl)(df, ["D"], ["A"])

    with pytest.raises(BodoError, match="not found"):
        bodo.jit(test_impl)(df, ["A"], ["D"])


def test_series_df_comparison(memory_leak_check):
    """
    Test that comapring dataframe and series
    raises an error as expected.
    """

    @bodo.jit
    def test_impl(df, other):
        return df == other

    df = pd.DataFrame(
        {"cost": [250, 150, 100], "revenue": [100, 250, 300]}, index=["A", "B", "C"]
    )
    other = pd.Series([100, 250], index=["cost", "revenue"])
    with pytest.raises(
        BodoError, match="Comparison operation between Dataframe and Series"
    ):
        test_impl(df, other)

    with pytest.raises(
        BodoError, match="Comparison operation between Dataframe and Series"
    ):
        test_impl(other, df)
