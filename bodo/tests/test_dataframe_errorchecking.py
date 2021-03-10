import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


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


# TODO: Mark as slow after CI passes
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


# TODO: Mark as slow after CI passes
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


# TODO: Mark as slow after CI passes
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
def test_df_set_index_empty_dataframe(memory_leak_check):
    """
    Tests DataFrame.set_index that produces an empty DataFrame.
    """

    def test_impl():
        df = pd.DataFrame({"A": np.random.randn(10)})
        return df.set_index("A")

    with pytest.raises(
        BodoError,
        match="DataFrame.set_index.*: Not supported on single column DataFrames.",
    ):
        bodo.jit(test_impl)()


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

    with pytest.raises(BodoError, match="Dataframe.head.*: 'n' must be an Integer"):
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
        return df.drop_duplicates(subset=["A"])

    def impl3():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        df.drop_duplicates(inplace=True)

    def impl4():
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
    with pytest.raises(BodoError, match=unsupported_arg_err_msg):
        bodo.jit(impl4)()


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
