"""Test Bodo's Table data type"""

import io

import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import find_callname, find_const, guard  # noqa TID253

import bodo
from bodo.tests.test_table_column_del import _check_column_dels
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    dist_IR_contains,
)


@pytest.fixture
def table_value(request):
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import Table

    return Table(
        [
            np.ones(10),
            np.arange(10),
            np.array(["AB"] * 10),
            np.ones(10) * 3,
            np.arange(10) + 1,
            np.arange(10) + 2,
            np.array(["A B C"] * 10),
        ]
    )


def test_unbox(table_value, memory_leak_check):
    # just unbox
    def impl(t_arg):
        return True

    # unbox and box
    def impl2(t_arg):
        return t_arg

    check_func(impl, (table_value,), only_seq=True)
    check_func(impl2, (table_value,), only_seq=True)


def test_constant_lowering(table_value, memory_leak_check):
    """Test constant lowering for TableType"""

    def test_impl():
        return table_value

    check_func(test_impl, (), only_seq=True)


def test_logical_table(memory_leak_check):
    """Test converting a logical table to TableType"""
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import Table
    from bodo.tests.utils_jit import ColumnDelTestPipeline, SeriesOptTestPipeline
    from bodo.utils.utils import find_build_tuple, is_call_assign

    col_inds = bodo.utils.typing.MetaType((2, 3, 1))
    col_names = bodo.utils.typing.ColNamesMetaType(("C1", "C2", "C3"))

    def impl1(T, A):
        return bodo.hiframes.table.logical_table_to_table(T, (A,), col_inds, 3)

    # test logical_table_to_table() elimination when the input table portion is used
    def impl2(df, A):
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (A,),
            col_inds,
            df.shape[1],
        )
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df.index, col_names)
        return df2.C1.values

    # test logical_table_to_table() elimination when the extra array portion is used
    def impl3(df, A):
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (A,),
            col_inds,
            df.shape[1],
        )
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df.index, col_names)
        return df2.C2.values

    # test logical_table_to_table() input elimination when input table is dead
    def impl4(df, A):
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (A,),
            col_inds,
            df.shape[1],
        )
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df.index, col_names)
        # using a list in iloc keeps the output table and therefore
        # logical_table_to_table alive
        return df2.iloc[:, [1]].values

    # test logical_table_to_table() input elimination when extra arrs are all dead
    def impl5(df, A, B):
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (A, B),
            col_inds,
            df.shape[1],
        )
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df.index, col_names)
        return df2.iloc[:, [0]].values

    A0 = np.arange(4)
    A1 = np.array(["A", "B", "C", "D"], object)
    A2 = np.ones(4)
    A4 = np.array([4, 3, 2, 1], np.int64)
    T1 = Table([A0, A1, A2])
    T2 = Table(
        [
            A2,
            A4,
            A1,
        ]
    )
    T3 = (A0, A1, A2)
    check_func(impl1, (T1, A4), only_seq=True, py_output=T2)
    check_func(impl1, (T3, A4), only_seq=True, py_output=T2)

    df = pd.DataFrame({"A": A0, "B": A1, "C": A2})

    def _check_no_logical_table_to_table(impl, expected_output):
        """make sure IR of 'impl' after dead code elimination doesn't contain logical_table_to_table()"""
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        np.testing.assert_array_equal(bodo_func(df, A4), expected_output)
        f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert not dist_IR_contains(f_ir, "logical_table_to_table")

    _check_no_logical_table_to_table(impl2, A2)
    _check_no_logical_table_to_table(impl3, A4)

    # make sure extra arr argument is set to None if dead
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl5)
    np.testing.assert_array_equal(bodo_func(df, A4, A0).ravel(), A2)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    table_call_found = False
    for stmt in f_ir.blocks[0].body:
        if is_call_assign(stmt) and guard(find_callname, f_ir, stmt.value) == (
            "logical_table_to_table",
            "bodo.hiframes.table",
        ):
            for v in find_build_tuple(f_ir, stmt.value.args[1]):
                assert find_const(f_ir, v) is None
            table_call_found = True
            break

    assert table_call_found


def test_logical_table_to_table_dels(datapath, memory_leak_check):
    """
    Make sure table columns are deleted properly for logical_table_to_table() calls
    """
    from bodo.tests.utils_jit import ColumnDelTestPipeline

    filename = datapath("many_columns.parquet")
    col_inds = bodo.utils.typing.MetaType((2, 99, 11, 7))
    col_names = bodo.utils.typing.ColNamesMetaType(("C1", "C2", "C3", "C4"))

    def impl():
        df1 = pd.read_parquet(filename, dtype_backend="pyarrow")
        A = np.ones(len(df1))
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1),
            (A,),
            col_inds,
            df1.shape[1],
        )
        df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df1.index, col_names)
        return df2

    in_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output = pd.DataFrame(
        {
            "C1": in_df.iloc[:, 2],
            "C2": np.ones(len(in_df)),
            "C3": in_df.iloc[:, 11],
            "C4": in_df.iloc[:, 7],
        }
    )
    check_func(impl, (), py_output=py_output)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        columns_list = [f"Column{i}" for i in [2, 7, 11]]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        _check_column_dels(bodo_func, [[2, 11, 7]])


def test_table_shape_opt(datapath, table_value, memory_leak_check):
    """
    Make sure table.shape[1] is optimized out (used in BodoSQL)
    """
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    def impl(T):
        return T.shape[1]

    check_func(impl, (table_value,), py_output=len(table_value.arrays), only_seq=True)
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(table_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "shape")


def test_table_astype_copy_false_bug(memory_leak_check):
    """
    Make sure table.astype(copy=False) works correctly with column elimination.
    See [BSE-840]
    """
    from bodo.utils.typing import ColNamesMetaType

    global_9 = ColNamesMetaType(("A", "B"))

    @bodo.jit
    def inner(delta_table, delta_col_names):
        delta_index = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(delta_table), 1, None
        )
        delta_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (delta_table,), delta_index, delta_col_names
        )
        delta_df_casted = delta_df.astype(
            delta_df.dtypes, copy=False, _bodo_nan_to_str=False
        )
        return delta_df_casted

    def impl(T9):
        return inner(T9, global_9)

    arrs = [np.array([1], np.int64), np.array([1], np.int8)]
    T9 = bodo.hiframes.table.Table(arrs)
    py_output = pd.DataFrame({"A": [1], "B": np.array([1], np.int8)})
    check_func(impl, (T9,), py_output=py_output, only_seq=True)


def test_create_empty_table(memory_leak_check):
    """
    Tests the implementation of create empty table, which should create a table
    of length 0 with a specified type.
    """
    empty_table = bodo.hiframes.table.Table(
        [
            pd.array([], dtype="Int64"),
            pd.array([], dtype="Int32"),
            pd.array([], dtype="string"),
            pd.array([], dtype="boolean"),
            pd.array([], dtype="Int64"),
        ]
    )
    table_type = bodo.typeof(empty_table)

    def impl():
        return bodo.hiframes.table.create_empty_table(table_type)

    check_func(impl, (), py_output=empty_table)


def test_create_empty_table_len(memory_leak_check):
    """
    Tests the implementation of create empty table is defined to be length 0
    and can optimized out the create_empty_table call.
    """
    from bodo.tests.utils_jit import DistTestPipeline

    empty_table = bodo.hiframes.table.Table(
        [
            pd.array([], dtype="Int64"),
            pd.array([], dtype="Int32"),
            pd.array([], dtype="string"),
            pd.array([], dtype="boolean"),
            pd.array([], dtype="Int64"),
        ]
    )
    table_type = bodo.typeof(empty_table)

    def impl():
        return len(bodo.hiframes.table.create_empty_table(table_type))

    check_func(impl, (), py_output=0)
    # Check the IR for removing create_empty_table
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    bodo_func()
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "create_empty_table")
