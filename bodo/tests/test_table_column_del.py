import io
import math
import os
import shutil

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core import ir

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import ColumnDelTestPipeline, check_func, reduce_sum
from bodo.utils.utils import is_expr


def _check_column_dels(bodo_func, col_del_lists):
    """
    Helper functions to check for the col_del calls inserted
    into the IR in BodoTableColumnDelPass. Since we don't know
    the exact blocks + labels in the IR, we cannot pass a dictionary
    or expected structure.

    Instead we pass 'col_del_lists', which is a list of lists for just
    the that delete columns for each del_column. For example, If we know
    that one call should delete 1 then 3 and another should just delete 2,
    we pass [[1, 3], [2]]. We then verify that all elements of this list are
    encountered (and nothing outside this list).

    Note: We do not check the order in which the deletions are inserted
    within a block to simplify testing as some changes that could occur
    would be insignificant.
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    typemap = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_typemap"]
    # Ensure every input list is sorted
    col_del_lists = [list(sorted(x)) for x in col_del_lists]
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and is_expr(stmt.value, "call"):
                typ = typemap[stmt.value.func.name]
                if (
                    isinstance(typ, numba.core.types.functions.Function)
                    and typ.name == "Function(<intrinsic del_column>)"
                ):
                    col_nums = typemap[stmt.value.args[1].name].instance_type.meta
                    col_nums = list(sorted(col_nums))
                    removed = False
                    for i, col_list in enumerate(col_del_lists):
                        if col_list == col_nums:
                            to_pop = i
                            removed = True
                            break
                    assert removed, f"Unexpected Del Column list {col_nums}"
                    col_del_lists.pop(to_pop)
    assert (
        not col_del_lists
    ), f"Some expected del columns were not encountered: {col_del_lists}"


@pytest.fixture(params=["csv", "parquet"])
def file_type(request):
    """
    Fixture for the various file source supporting table types.
    """
    return request.param


def test_table_len(file_type, datapath, memory_leak_check):
    """
    Check that an operation that just returns a length
    and doesn't use any particular column still computes
    a correct result.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        return len(df)"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        if file_type == "parquet":
            # Parquet can determine length without loading any columns.
            loaded_columns = []
        else:
            loaded_columns = ["Column0"]
        check_logger_msg(stream, f"Columns loaded {loaded_columns}")


def test_table_filter_dead_columns(datapath, memory_leak_check):
    """
    Test table filter with no used column (just length)
    """
    filename = datapath(f"many_columns.parquet")

    def impl(idx):
        df = pd.read_parquet(filename)
        df2 = df[idx]
        return len(df2)

    idx = np.arange(len(pd.read_parquet(filename))) % 3 == 0
    check_func(impl, (idx,), only_seq=True)
    # NOTE: this needs distributed=False since args/returns don't force
    # sequential execution.
    check_func(
        impl,
        (slice(10),),
        only_seq=True,
        additional_compiler_arguments={"distributed": False},
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(distributed=False)(impl)
        bodo_func(idx)
        # Parquet can determine length without loading any columns.
        check_logger_msg(stream, "Columns loaded []")


def test_table_len_with_idx_col(datapath, memory_leak_check):
    """
    Check that an operation that just returns a length
    and doesn't use any particular column still computes
    a correct result.

    Manually verified that the index col is dead/removed
    """
    filename = datapath(f"many_columns.csv")

    def impl():
        df = pd.read_csv(filename, index_col="Column0")
        return len(df)

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        # Only load column 1 for length. We skip 0 because
        # its the index.
        check_logger_msg(stream, "Columns loaded ['Column1']")


def test_table_shape(file_type, datapath, memory_leak_check):
    """
    Check that an operation that just returns a shape
    and doesn't use any particular column still computes
    a correct result.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        return df.shape"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        if file_type == "parquet":
            # Parquet can determine length without loading any columns.
            loaded_columns = []
        else:
            loaded_columns = ["Column0"]
        check_logger_msg(stream, f"Columns loaded {loaded_columns}")


def test_table_del_single_block(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns with a single block.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        return (df["Column3"], df["Column37"], df["Column59"])"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[3], [37], [59]])
        check_logger_msg(stream, "Columns loaded ['Column3', 'Column37', 'Column59']")


def test_table_del_back(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove some after the first
    basic block.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        size = df["Column0"].mean()
        if size > 10000:
            n = 100
        else:
            n = 500
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[0], [3, 37, 59], [0, 1, 2]]
        else:
            delete_list = [[0], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream, "Columns loaded ['Column0', 'Column3', 'Column37', 'Column59']"
        )


def test_table_del_front(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove a column at the start
    of some successors.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        size = df["Column0"].mean()
        if size > 10000:
            n = df["Column0"].sum()
        else:
            n = df["Column3"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[0], [0], [3, 37, 59], [0, 1, 2]]
        else:
            delete_list = [[0], [0], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream, "Columns loaded ['Column0', 'Column3', 'Column37', 'Column59']"
        )


def test_table_del_front_back(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination where some columns
    can be removed at the end of basic blocks but
    others must be removed at the start of basic blocks
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        size = df["Column0"].mean() - df["Column3"].mean()
        if size > 0:
            n = df["Column0"].sum() + df["Column6"].sum()
        else:
            n = df["Column3"].sum() + df["Column9"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[9], [0], [6], [0, 6], [9], [3, 37, 59], [0, 1, 2]]
        else:
            delete_list = [[9], [0], [6], [0, 6], [9], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream,
            "Columns loaded ['Column0', 'Column3', 'Column6', 'Column9', 'Column37', 'Column59']",
        )


def test_table_useall_later_block(file_type, datapath, memory_leak_check):
    """
    Check that an operation that requires using all
    columns eventually doesn't eliminate any columns.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        size = df["Column0"].sum()
        if size < 1000:
            w = 10
        else:
            w = 100
        # Use objmode to force useall
        with bodo.objmode(n="int64"):
            n = df.shape[1]
        return n + w"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")


def test_table_useall_early_block(file_type, datapath, memory_leak_check):
    """
    Check that an operation that requires using all
    columns early doesn't eliminate any columns later.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        # Use objmode to force useall
        with bodo.objmode(n="int64"):
            n = df.shape[1]
        size = df["Column0"].sum()
        if size < 1000:
            w = 10
        else:
            w = 100
        return n + w"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")


def test_table_del_usecols(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination where a user has
    provided usecols as well
    """
    filename = datapath(f"many_columns.{file_type}")
    columns = [0, 1, 2, 3, 4, 5, 6, 9, 10, 37, 38, 52, 59, 67, 95, 96, 97, 98]
    if file_type == "csv":
        kwarg = "usecols"
    elif file_type == "parquet":
        kwarg = "columns"
        columns = [f"Column{i}" for i in columns]
    func_text = f"""def impl():
        df = pd.read_{file_type}(
            {filename!r},
            {kwarg}={columns},
        )
        size = df["Column0"].mean() - df["Column3"].mean()
        if size > 0:
            n = df["Column0"].sum() + df["Column6"].sum()
        else:
            n = df["Column3"].sum() + df["Column9"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # The column numbers here are remapped to their index in usecols.
        # Note 6 and 7 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[7], [0], [6], [0, 6], [7], [3, 9, 12], [0, 1, 2]]
        else:
            delete_list = [[7], [0], [6], [0, 6], [7], [3], [9], [12]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream,
            "Columns loaded ['Column0', 'Column3', 'Column6', 'Column9', 'Column37', 'Column59']",
        )


def test_table_set_table_columns(file_type, datapath, memory_leak_check):
    """
    Tests setting a table can still run dead column elimination.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        size = df["Column0"].mean() - df["Column3"].mean()
        if size > 0:
            n = df["Column0"].sum() + df["Column6"].sum()
        else:
            n = df["Column3"].sum() + df["Column9"].sum()
        df["Column37"] = np.arange(1000)
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[9], [0], [6], [0, 6], [9], [3, 37, 59], [3, 59], [0, 1, 2]]
        else:
            delete_list = [[9], [0], [6], [0, 6], [9], [37], [3, 59], [3], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream,
            "Columns loaded ['Column0', 'Column3', 'Column6', 'Column9', 'Column59']",
        )


def test_table_extra_column(file_type, datapath, memory_leak_check):
    """
    Tests that using a column outside the original column list doesn't have
    any issues when loading from a file.

    # TODO: Add code to ensure 0 columns were loaded
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        return df["Column99"].sum()"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    bodo_func()
    # We should only remove the column added by set_table_data
    _check_column_dels(bodo_func, [[99]])


def test_table_dead_var(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when a variable is dead in certain parts of
    control flow.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        n = df["Column0"].sum()
        if n > 100:
            return df["Column0"].mean()
        else:
            return 1.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0]])
        check_logger_msg(stream, "Columns loaded ['Column0']")


def test_table_for_loop(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        total = 0.0
        for _ in range(n):
            total += df["Column0"].sum()
        return total + df["Column3"].sum()"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_while_loop(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        total = 0.0
        while n > 0:
            total += df["Column0"].sum()
            n -= 1
        return total + df["Column3"].sum()"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_for_loop_branch(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        total = 0.0
        if len(df) > 900:
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()
        else:
            return 0.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_while_loop_branch(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        total = 0.0
        if len(df) > 900:
            while n > 0:
                total += df["Column0"].sum()
                n -= 1
            return total + df["Column3"].sum()
        else:
            return 0.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_loop_unroll(file_type, datapath, memory_leak_check):
    """
    Tests removing columns with a loop that
    requires unrolling.
    """
    filename = datapath(f"many_columns.{file_type}")
    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        total = 0.0
        columns = ["Column0", "Column3", "Column6"]
        for c in columns:
            total += df[c].sum()
        return total"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note: all values are separate because sum adds extra basic blocks
        _check_column_dels(bodo_func, [[0], [3], [6]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3', 'Column6']")


def test_table_return(file_type, datapath, memory_leak_check):
    """
    Tests that returning a table avoids dead column elimination.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        n = df["Column0"].sum()
        if n < 1000:
            n = df["Column3"].sum() - n
        else:
            n = df["Column6"].sum() - n
        return df, n"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")


# Tests involving an alias. Numba can sometimes optimize out aliases in user code
# so we include a set_table_data call at the front via df["Column99"] = np.arange(1000)
def test_table_len_alias(file_type, datapath, memory_leak_check):
    """
    Check that len only loads a single column when there is
    an alias
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        return len(df)"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        if file_type == "parquet":
            # Parquet can determine length without loading any columns.
            loaded_columns = []
        else:
            loaded_columns = ["Column0"]
        check_logger_msg(stream, f"Columns loaded {loaded_columns}")


def test_table_shape_alias(file_type, datapath, memory_leak_check):
    """
    Check that shape only loads a single column when there is
    an alias
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        return df.shape"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        if file_type == "parquet":
            # Parquet can determine length without loading any columns.
            loaded_columns = []
        else:
            loaded_columns = ["Column0"]
        check_logger_msg(stream, f"Columns loaded {loaded_columns}")


def test_table_del_single_block_alias(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns with a single block and an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        return df[["Column3", "Column37", "Column59"]]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            delete_list = [[3, 37, 59], [3, 37, 59]]
        else:
            delete_list = [[3, 37, 59], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(stream, "Columns loaded ['Column3', 'Column37', 'Column59']")


def test_table_del_back_alias(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove some after the first
    basic block.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        size = df["Column0"].mean()
        if size > 10000:
            n = 100
        else:
            n = 500
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[0, 3, 37, 59], [0], [3, 37, 59], [0, 1, 2]]
        else:
            delete_list = [[0, 3, 37, 59], [0], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream, "Columns loaded ['Column0', 'Column3', 'Column37', 'Column59']"
        )


def test_table_del_front_alias(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination with an alias that
    loads a subset of columns and can remove a column
    at the start of some successors.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        size = df["Column0"].mean()
        if size > 10000:
            n = df["Column0"].sum()
        else:
            n = df["Column3"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[0, 3, 37, 59], [0], [0], [3, 37, 59], [0, 1, 2]]
        else:
            delete_list = [[0, 3, 37, 59], [0], [0], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream, "Columns loaded ['Column0', 'Column3', 'Column37', 'Column59']"
        )


def test_table_del_front_back_alias(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination with an alias where
    some columns can be removed at the end of basic
    blocks but others must be removed at the start of
    basic blocks
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        size = df["Column0"].mean() - df["Column3"].mean()
        if size > 0:
            n = df["Column0"].sum() + df["Column6"].sum()
        else:
            n = df["Column3"].sum() + df["Column9"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [
                [0, 3, 6, 9, 37, 59],
                [9],
                [0],
                [6],
                [0, 6],
                [9],
                [3, 37, 59],
                [0, 1, 2],
            ]
        else:
            delete_list = [
                [0, 3, 6, 9, 37, 59],
                [9],
                [0],
                [6],
                [0, 6],
                [9],
                [3],
                [37],
                [59],
            ]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream,
            "Columns loaded ['Column0', 'Column3', 'Column6', 'Column9', 'Column37', 'Column59']",
        )


def test_table_useall_later_block_alias(file_type, datapath, memory_leak_check):
    """
    Check that an operation with an alias that requires
    using all columns eventually doesn't eliminate any columns.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        size = df["Column0"].sum()
        if size < 1000:
            w = 10
        else:
            w = 100
        # Use objmode to force useall
        with bodo.objmode(n="int64"):
            n = df.shape[1]
        return n + w"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [list(range(99))])
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")


def test_table_useall_early_block_alias(file_type, datapath, memory_leak_check):
    """
    Check that an operation with an alias that
    requires using all columns early doesn't
    eliminate any columns later.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        # Use objmode to force useall
        with bodo.objmode(n="int64"):
            n = df.shape[1]
        size = df["Column0"].sum()
        if size < 1000:
            w = 10
        else:
            w = 100
        return n + w"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [list(range(99))])
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")


def test_table_del_usecols_alias(file_type, datapath, memory_leak_check):
    """
    Test dead column elimination where a user has
    provided usecols as well + an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    columns = [0, 1, 2, 3, 4, 5, 6, 9, 10, 37, 38, 52, 59, 67, 95, 96, 97, 98]
    if file_type == "csv":
        kwarg = "usecols"
    elif file_type == "parquet":
        kwarg = "columns"
        columns = [f"Column{i}" for i in columns]
    func_text = f"""def impl():
        df = pd.read_{file_type}(
            {filename!r},
            {kwarg}={columns},
        )
        df["Column99"] = np.arange(1000)
        size = df["Column0"].mean() - df["Column3"].mean()
        if size > 0:
            n = df["Column0"].sum() + df["Column6"].sum()
        else:
            n = df["Column3"].sum() + df["Column9"].sum()
        return df[["Column3", "Column37", "Column59"]].head(np.int64(n))"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # The column numbers here are remapped to their index in usecols.
        # Note 6 and 7 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [
                [0, 3, 6, 7, 9, 12],
                [7],
                [0],
                [6],
                [0, 6],
                [7],
                [3, 9, 12],
                [0, 1, 2],
            ]
        else:
            delete_list = [
                [0, 3, 6, 7, 9, 12],
                [7],
                [0],
                [6],
                [0, 6],
                [7],
                [3],
                [9],
                [12],
            ]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(
            stream,
            "Columns loaded ['Column0', 'Column3', 'Column6', 'Column9', 'Column37', 'Column59']",
        )


def test_table_dead_var_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when a variable is dead in certain parts of
    control flow with an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        n = df["Column0"].sum()
        if n > 100:
            return df["Column0"].mean()
        else:
            return 1.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0], [0]])
        check_logger_msg(stream, "Columns loaded ['Column0']")


def test_table_for_loop_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        total = 0.0
        for _ in range(n):
            total += df["Column0"].sum()
        return total + df["Column3"].sum()"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_while_loop_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        total = 0.0
        while n > 0:
            total += df["Column0"].sum()
            n -= 1
        return total + df["Column3"].sum()"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_for_loop_branch_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch
    with an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        total = 0.0
        if len(df) > 900:
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()
        else:
            return 0.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_while_loop_branch_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch with
    an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl(n):
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        total = 0.0
        if len(df) > 900:
            while n > 0:
                total += df["Column0"].sum()
                n -= 1
            return total + df["Column3"].sum()
        else:
            return 0.0"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3']")


def test_table_loop_unroll_alias(file_type, datapath, memory_leak_check):
    """
    Tests removing columns with a loop that
    requires unrolling with an alias.
    """
    filename = datapath(f"many_columns.{file_type}")

    func_text = f"""def impl():
        df = pd.read_{file_type}({filename!r})
        df["Column99"] = np.arange(1000)
        total = 0.0
        columns = ["Column0", "Column3", "Column6"]
        for c in columns:
            total += df[c].sum()
        return total"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0, 3, 6], [0], [3], [6]])
        check_logger_msg(stream, "Columns loaded ['Column0', 'Column3', 'Column6']")


#### Index column tests
def test_table_del_single_block_pq_index(datapath, memory_leak_check):
    """
    Test dead column elimination works with a DataFrame that
    loads an index from parquet.
    TODO: add automatic check to ensure that the index is loaded
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl():
        df = pd.read_parquet(filename)
        return df[["Column3", "Column37", "Column59"]]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            delete_list = [[3, 37, 59]]
        else:
            delete_list = [[3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(stream, f"Columns loaded ['Column3', 'Column37', 'Column59']")


def test_table_del_single_block_pq_index_alias(datapath, memory_leak_check):
    """
    Test dead column elimination works with a DataFrame that
    loads an index from parquet with an alias.
    TODO: add automatic check to ensure that the index is loaded
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl():
        df = pd.read_parquet(filename)
        df["Column99"] = np.arange(1000)
        return df[["Column3", "Column37", "Column59"]]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 3:
            delete_list = [[3, 37, 59], [3, 37, 59]]
        else:
            delete_list = [[3, 37, 59], [3], [37], [59]]
        _check_column_dels(bodo_func, delete_list)
        check_logger_msg(stream, f"Columns loaded ['Column3', 'Column37', 'Column59']")


def test_table_dead_pq_index(datapath, memory_leak_check):
    """
    Test dead code elimination still works for unused indices.
    TODO: add automatic check to ensure that the index is marked as dead
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl(n):
        df = pd.read_parquet(filename)
        total = 0.0
        for _ in range(n):
            total += df["Column0"].sum()
        return total + df["Column3"].sum()

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0], [3]])
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column3']")


def test_table_dead_pq_index_alias(datapath, memory_leak_check):
    """
    Test dead code elimination still works for unused indices with an alias.
    TODO: add automatic check to ensure that the index is marked as dead
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl(n):
        df = pd.read_parquet(filename)
        df["Column99"] = np.arange(1000)
        total = 0.0
        for _ in range(n):
            total += df["Column0"].sum()
        return total + df["Column3"].sum()

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column3']")


def test_table_while_loop_alias_with_idx_col(datapath, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.
    TODO: add automatic check to ensure that the index is marked as dead
    """
    filename = datapath(f"many_columns.csv")

    def impl(n):
        df = pd.read_csv(filename)
        df["Column99"] = np.arange(1000)
        total = 0.0
        while n > 0:
            total += df["Column0"].sum()
            n -= 1
        return total + df["Column3"].sum()

    check_func(impl, (25,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3], [0], [3]])
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column3']")


def test_table_dead_pq_table(datapath, memory_leak_check):
    """
    Test dead code elimination still works on a table with a used index.
    TODO: add automatic check to ensure that the table is marked as dead
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl():
        df = pd.read_parquet(filename)
        df["Column99"] = np.arange(1000)
        return df.index

    check_func(impl, ())
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    bodo_func()
    _check_column_dels(bodo_func, [])


def test_table_dead_pq_table_alias(datapath, memory_leak_check):
    """
    Test dead code elimination still works on a table with a used index
    and an alias.
    TODO: add automatic check to ensure that the table is marked as dead
    """
    filename = datapath(f"many_columns_index.parquet")

    def impl():
        df = pd.read_parquet(filename)
        df["Column99"] = np.arange(1000)
        return df.index

    check_func(impl, ())
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    bodo_func()
    _check_column_dels(bodo_func, [])


def test_table_dead_csv(datapath, memory_leak_check):
    """
    Tests returning only the index succeeds.
    I've manually confirmed that the table variable is correctly marked as dead.
    TODO: add automatic check to ensure that the table is marked as dead
    """
    filename = datapath(f"many_columns.csv")

    def impl():
        df = pd.read_csv(filename, index_col="Column4")
        return df.index

    check_func(impl, ())
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    bodo_func()
    _check_column_dels(bodo_func, [])


def test_many_cols_to_parquet(datapath, memory_leak_check):
    """Tests df.to_parquet with many columns."""
    filename = datapath(f"many_columns.csv")
    try:

        def impl(source_filename, dest_filename):
            df = pd.read_csv(source_filename)
            df.to_parquet(dest_filename)

        def check_correctness(pandas_filename, bodo_filename):
            pandas_df = pd.read_parquet(pandas_filename)
            bodo_df = pd.read_parquet(bodo_filename)
            try:
                pd.testing.assert_frame_equal(
                    pandas_df, bodo_df, check_column_type=False
                )
                return 1
            except Exception:
                return 0

        pandas_filename = "pandas_out.pq"
        bodo_filename = "bodo_out.pq"
        impl(filename, pandas_filename)
        bodo.jit(impl)(filename, bodo_filename)
        passed = 1
        if bodo.get_rank() == 0:
            passed = check_correctness(pandas_filename, bodo_filename)
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()
    finally:
        if bodo.get_rank() == 0:
            os.remove(pandas_filename)
            shutil.rmtree(bodo_filename, ignore_errors=True)


def test_table_dead_csv(datapath, memory_leak_check):
    """
    Tests returning only the index succeeds.
    I've manually confirmed that the table variable is correctly marked as dead.
    TODO: add automatic check to ensure that the table is marked as dead
    """
    filename = datapath(f"many_columns.csv")

    def impl():
        df = pd.read_csv(filename, index_col="Column4")
        return df.index

    check_func(impl, ())
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    bodo_func()
    _check_column_dels(bodo_func, [])


@pytest.mark.parametrize("file_type", ["csv", "parquet"])
def test_table_column_del_past_setitem(memory_leak_check, datapath, file_type):
    """
    Tests that dead setitems on tables are correctly converted to set_table_column_null, which
    allows for column pruning at the IO node.
    """
    filename = datapath(f"many_columns.{file_type}")
    num_layers = 10

    func_text = ""
    func_text += "def impl():\n"
    func_text += f"    df = pd.read_{file_type}({filename!r})\n"
    for i in range(num_layers):
        func_text += f"    df['Column{i+1}'] = df['Column{i}']\n"

    func_text += f"    return df['Column{num_layers + 1}']\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "pd": pd}, loc_vars)
    impl = loc_vars["impl"]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (), check_dtype=False)
        check_logger_msg(stream, f"Columns loaded ['Column{num_layers + 1}']")


@pytest.mark.parametrize("num_layers", [3, pytest.param(10, marks=pytest.mark.slow)])
def test_table_column_filter_past_setitem(memory_leak_check, datapath, num_layers):
    """
    Tests that dead setitems on tables are correctly converted to set_table_column_null, which
    allows for column pruning and filtering at the IO node.
    """
    filename = datapath(f"many_columns.parquet")

    func_text = ""
    func_text += "def impl():\n"
    func_text += f"    df = pd.read_parquet({filename!r})\n"
    for i in range(num_layers):
        func_text += f"    df['Column{i+1}'] = df['Column{i}']\n"
    func_text += f"    df2 = df[df['Column96'] > 10]\n"
    # func_text += f"    df2 = df\n"
    func_text += f"    return df2['Column{num_layers + 1}']\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "pd": pd}, loc_vars)
    impl = loc_vars["impl"]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        check_func(impl, (), check_dtype=False, reset_index=True)
        # TODO: update filter pushdown to be able to push column 96
        check_logger_msg(
            stream, f"Columns loaded ['Column{num_layers + 1}', 'Column96']"
        )


def test_table_column_pruing_past_atype_setitem(datapath, memory_leak_check):
    """
    Tests that dead setitems on tables are correctly converted to set_table_column_null, which
    allows for column pruning at the IO node.

    This case checks that the setiem is converted even when using funtions that have no side
    effects.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        df["Column0"] = df["Column0"].astype(float)
        df["Column1"] = df["Column1"].astype(str)
        df["Column0"] = df["Column0"] + 1
        return df["Column3"]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    check_func(impl, (), check_dtype=False, reset_index=True)
    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func()
        check_logger_msg(stream, "Columns loaded ['Column3']")
        _check_column_dels(bodo_func, [[3], [3], [3], [3]])


def test_table_del_astype(datapath, memory_leak_check):
    """
    Test dead column elimination works with an astype
    cast.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        df1 = df.astype({"Column3": np.float32, "Column36": pd.Int8Dtype()})
        return df1[["Column3", "Column37", "Column59"]]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)()
        check_logger_msg(stream, "Columns loaded ['Column3', 'Column37', 'Column59']")


def test_table_nbytes_del_cols(datapath, memory_leak_check):
    """
    Test generate_table_nbytes properly creates del_column
    calls inside the IR. This function should require all
    columns to be loaded but it should be possible to delete
    every column that isn't used later once the operation
    finishes.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        total_bytes = df.memory_usage(index=False)
        if total_bytes.sum() > 1000:
            n = df["Column3"].sum()
        else:
            n = df["Column6"].sum()
        return n

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # df.memory_usage uses every column, but those should be possible
        # to delete in 1 block
        init_block_columns = list(sorted(set(range(99)) - {3, 6}))
        # The other blocks should then each decref the opposite column
        # at the start and then the used column
        _check_column_dels(bodo_func, [init_block_columns, [3], [6], [3], [6]])


def test_table_nbytes_ret_df(datapath, memory_leak_check):
    """
    Test generate_table_nbytes doesn't creates del_column
    calls inside the IR if we return a DataFrame with the
    same table.

    Because we return the DataFrame, this also tests that
    when the source of a table is a DataFrame that we do
    not prune columns.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        total_bytes = df.memory_usage(index=False)
        if total_bytes.sum() > 1000:
            df["Column0"] += 1
        else:
            df["Column0"] += 2
        return df

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # Since we return a DF no columns should be deleted.
        _check_column_dels(bodo_func, [])


def test_table_concat_del_cols(datapath, memory_leak_check):
    """
    Test table_concat properly creates del_column
    calls inside the IR. This function should enable
    pruning any columns it needs that isn't used later
    once the operation finishes.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        out_df = df.melt(
            id_vars=["Column2", "Column5"], value_vars=["Column0", "Column3"]
        )
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if (out_df["Column2"].nunique() + out_df["Column5"].nunique()) > 1000:
            n = df["Column3"].sum()
        else:
            n = df["Column6"].sum()
        return n

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        check_logger_msg(
            stream,
            f"Columns loaded ['Column0', 'Column2', 'Column3', 'Column5', 'Column6']",
        )
        # Melt uses but can then delete 0, 2 and 5.
        # The other blocks delete the columns used in both branch
        # options.
        _check_column_dels(bodo_func, [[0], [2], [5], [3], [6], [3], [6]])


def test_table_concat_ret_df(datapath, memory_leak_check):
    """
    Test table_concat doesn't creates del_column
    calls inside the IR if we return a DataFrame with the
    same table.

    Because we return the DataFrame, this also tests that
    when the source of a table is a DataFrame that we do
    not prune columns.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df = pd.read_parquet(filename)
        out_df = df.melt(
            id_vars=["Column2", "Column5"], value_vars=["Column0", "Column3"]
        )
        if (out_df["Column2"].nunique() + out_df["Column5"].nunique()) > 1000:
            df["Column0"] += 1
        else:
            df["Column0"] += 2
        return df

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have a source DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # Since we return a DF no columns should be deleted.
        _check_column_dels(bodo_func, [])


def test_table_set_data_reflection(memory_leak_check):
    """
    Tests that set_table_data works properly reflects columns
    back to the parent and won't attempt to delete columns.
    """

    def impl(df):
        df["G"] = np.arange(len(df))
        return df["A"].sum()

    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 20,
            "B": ["abc", "bc", "cd"] * 20,
            "C": [5, 6, 7] * 20,
            "D": [1.1, 2.2, 3.3] * 20,
        }
    )
    # add a bunch of columns to trigger table format
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df[f"F{i}"] = 11

    df2 = df.copy(deep=True)
    df2["G"] = np.arange(len(df))

    bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
    n = bodo_func(df)
    # Verify the output + reflection
    assert n == 120, "Incorrect sum computed"
    pd.testing.assert_frame_equal(df, df2, check_column_type=False)
    # Since the source is a DF, no columns should be deleted.
    _check_column_dels(bodo_func, [])


def test_table_astype_del_cols(datapath, memory_leak_check):
    """
    Test table_astype properly creates del_column
    calls inside the IR and that calling table prunes
    its columns as well.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.astype({"Column0": np.float32, "Column6": np.float32})
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2["Column6"]

    # Set check_dtype=False becuase the += upcasts
    # Column6 in Pandas despite always having an
    # int output.
    check_func(impl, (), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        check_logger_msg(
            stream,
            f"Columns loaded ['Column0', 'Column3', 'Column6']",
        )
        # In the first block df1 needs to load columns 0 and 6 for astype.
        # Column 0 is no longer needed by df1 after the operation. Similarly
        # df2 can remove column 0 after the getitem, so both delete the column
        # in the first block.
        _check_column_dels(bodo_func, [[0], [0], [3], [6], [3], [6], [6], [6]])


def test_table_astype_ret_df_input(datapath, memory_leak_check):
    """
    Test table_astype output creates del_column
    calls inside the IR if we return a DataFrame with the
    input table.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.astype({"Column0": np.float32, "Column6": np.float32})
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df1["Column0"] += n
        return df1

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have a source DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # Since we return we only delete the columns from df2.
        # Note: Column 6 will be pruned.
        _check_column_dels(bodo_func, [[0], list(range(99))])


def test_table_astype_ret_df_output(datapath, memory_leak_check):
    """
    Test table_astype output doesn't create del_column
    calls inside the IR if we return the result DataFrame.
    However, the del_column should still be created for the input.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.astype({"Column0": np.float32, "Column6": np.float32})
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have an output DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # We prune columns from df1 after the astype.
        init_block_columns = list(sorted(set(range(99)) - {3, 6}))
        _check_column_dels(
            bodo_func, [init_block_columns, [3], [6], [3], [6], list(range(99))]
        )


def test_table_astype_multiple_cols_different_cast(datapath, memory_leak_check):
    """
    Test table_astype properly handles the case where we're casting two
    columns of of the same type, to a different type.

    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        # All columns being casted are originally int
        df2 = df1.astype({"Column0": np.float32, "Column3": np.float64})
        return df2

    check_func(impl, (), check_dtype=False)


def test_filter_del_cols(datapath, memory_leak_check):
    """
    Test table_filter properly creates del_column
    calls inside the IR and that calling table prunes
    its columns as well.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.head()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2["Column6"]

    # Set check_dtype=False becuase the += upcasts
    # Column6 in Pandas despite always having an
    # int output.
    check_func(impl, (), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        check_logger_msg(
            stream,
            f"Columns loaded ['Column0', 'Column3', 'Column6']",
        )
        # In the first block df1 needs to load columns 0 and 6 for astype.
        # Column 0 is no longer needed by df1 after the operation. Similarly
        # df2 can remove column 0 after the getitem, so both delete the column
        # in the first block.
        _check_column_dels(bodo_func, [[0], [0], [3], [6], [3], [6], [6], [6]])


def test_filter_ret_df_input(datapath, memory_leak_check):
    """
    Test table_filter output creates del_column
    calls inside the IR if we return a DataFrame with the
    input table.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.head()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df1["Column0"] += n
        return df1

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have a source DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # Since we return we only delete the columns from df2.
        # Note: Column 6 will be pruned.
        _check_column_dels(bodo_func, [[0], list(range(99))])


def test_table_filter_ret_df_output(datapath, memory_leak_check):
    """
    Test table_filter output doesn't create del_column
    calls inside the IR if we return the result DataFrame.
    However, the del_column should still be created for the input.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.head()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have an output DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # We prune columns from df1 after the filter.
        init_block_columns = list(sorted(set(range(99)) - {3, 6}))
        _check_column_dels(
            bodo_func, [init_block_columns, [3], [6], [3], [6], list(range(99))]
        )


def test_table_mappable_del_cols(datapath, memory_leak_check):
    """
    Test generate_mappable_table_func properly creates del_column
    calls inside the IR and that calling table prunes
    its columns as well.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.copy()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2["Column6"]

    # Set check_dtype=False becuase the += upcasts
    # Column6 in Pandas despite always having an
    # int output.
    check_func(impl, (), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        check_logger_msg(
            stream,
            f"Columns loaded ['Column0', 'Column3', 'Column6']",
        )
        # In the first block df1 needs to load columns 0 and 6 for astype.
        # Column 0 is no longer needed by df1 after the operation. Similarly
        # df2 can remove column 0 after the getitem, so both delete the column
        # in the first block.
        _check_column_dels(bodo_func, [[0], [0], [3], [6], [3], [6], [6], [6]])


def test_table_mappable_ret_df_input(datapath, memory_leak_check):
    """
    Test generate_mappable_table_func output creates del_column
    calls inside the IR if we return a DataFrame with the
    input table.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.copy()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df1["Column0"] += n
        return df1

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have a source DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # Since we return we only delete the columns from df2.
        # Note: Column 6 will be pruned.
        _check_column_dels(bodo_func, [[0], list(range(99))])


def test_table_mappable_ret_df_output(datapath, memory_leak_check):
    """
    Test generate_mappable_table_func output doesn't create del_column
    calls inside the IR if we return the result DataFrame.
    However, the del_column should still be created for the input.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.copy()
        # Use both Column2 and Column5 so they don't get claimed by DCE.
        # These columns aren't part of table operations.
        if df2["Column0"].sum() > 1000.0:
            n = df1["Column3"].sum()
        else:
            n = df1["Column6"].sum()
        # TODO: Putting the df2["Column6"] inside the branch
        # fails because clobbering df2 results in a phi node
        # which prevents eliminating the intermediate
        # DataFrame.
        df2["Column6"] += n
        return df2

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded as we have an output DataFrame.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # We prune columns from df1 after the copy.
        init_block_columns = list(sorted(set(range(99)) - {3, 6}))
        _check_column_dels(
            bodo_func, [init_block_columns, [3], [6], [3], [6], list(range(99))]
        )


def test_sort_table_dels(datapath, memory_leak_check):
    """
    Make sure table columns are deleted properly for Sort nodes
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.sort_values(by=["Column9", "Column6", "Column13", "Column11"])
        return df2["Column11"], df2["Column8"], df2["Column3"], df1["Column1"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        columns_list = [f"Column{i}" for i in [1, 3, 6, 8, 9, 11, 13]]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # [3, 6, 8, 9, 11, 13] for input table is repeated since there is a table copy
        _check_column_dels(
            bodo_func, [[3, 6, 8, 9, 11, 13], [3, 6, 8, 9, 11, 13], [11], [8], [3], [1]]
        )


def test_groupby_table_dels(datapath, memory_leak_check):
    """
    Make sure table columns are deleted properly for Aggregate nodes
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.groupby(by=["Column9", "Column6", "Column13", "Column11"])[
            ["Column14", "Column3", "Column17", "Column8", "Column1"]
        ].count()
        return df2["Column8"].values, df2["Column3"].values, df1["Column1"].values

    check_func(impl, (), sort_output=True, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        columns_list = [f"Column{i}" for i in [1, 3, 6, 8, 9, 11, 13]]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        _check_column_dels(bodo_func, [[3, 6, 8, 9, 11, 13], [3], [1], [1]])


def test_groupby_table_dels_as_index_false(datapath, memory_leak_check):
    """
    Make sure table columns are deleted properly for Aggregate nodes when as_index=False
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.groupby(
            by=["Column9", "Column6", "Column13", "Column11"], as_index=False
        )[["Column14", "Column3", "Column17", "Column8", "Column1"]].count()
        return (
            df2["Column8"].values,
            df2["Column3"].values,
            df1["Column1"].values,
            df2["Column6"].values,
        )

    check_func(impl, (), sort_output=True, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        columns_list = [f"Column{i}" for i in [1, 3, 6, 8, 9, 11, 13]]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        _check_column_dels(bodo_func, [[3, 6, 8, 9, 11, 13], [7], [5], [1], [1]])


def test_two_column_dels(datapath, memory_leak_check):
    """
    Test that when deleting a large number of columns we
    delete columns in batches.
    """
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.memory_usage(index=False)
        return df2.sum() > 100.0

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # All columns should be loaded.
        columns_list = [f"Column{i}" for i in range(99)]
        check_logger_msg(stream, f"Columns loaded {columns_list}")
        # We prune columns from the table after memory usage in
        # 1 function call.
        columns_to_delete = list(range(99))
        _check_column_dels(bodo_func, [columns_to_delete])


def test_table_loc_column_subset_level1(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format via df.loc. This is useful to maintain projections,
    especially for BodoSQL.

    BodoSQL generated code may select a large subset of columns, even
    if uncommon in user generated code.
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded = [f"Column{i}" for i in range(n_cols)]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        return df.loc[:, {columns_loaded[::-1] * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(columns_loaded))
        _check_column_dels(bodo_func, [list(range(n_cols)), list(range(n_cols * 2))])


def test_table_loc_column_subset_level2(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format and multiple level of table_format subset via
    df.loc.

    This is used to ensure used_cols is properly updated at each step
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols_level_1 = max(1, bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD + 4)
    columns_loaded_level_1 = [f"Column{i}" for i in range(n_cols_level_1)]
    n_cols_level_2 = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded_level_2 = [f"Column{i}" for i in range(n_cols_level_2)]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        df1 = df.loc[:, {columns_loaded_level_1[::-1]}]
        return df1.loc[:, {columns_loaded_level_2 * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(columns_loaded_level_2))
        # df deletes columns in the original location, but df1
        # deletes them at the new remapped location.
        df_col_nums = list(range(n_cols_level_2))
        df1_col_nums = list(range(n_cols_level_1 - n_cols_level_2, n_cols_level_1))
        df2_col_nums = list(range(n_cols_level_2 * 2))
        # There is 1 del_columns for the original df, 2 for df1 (the subset
        # and the filter) and 1 for df2 (the subset)
        _check_column_dels(
            bodo_func, [df_col_nums, df1_col_nums, df1_col_nums, df2_col_nums]
        )


def test_table_iloc_column_subset_level1(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format via df.iloc. This is useful to maintain projections,
    especially for BodoSQL.

    BodoSQL generated code may select a large subset of columns, even
    if uncommon in user generated code.
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded = list(range(n_cols))
    cols_names_loaded = [f"Column{i}" for i in columns_loaded]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        return df.iloc[:, {columns_loaded[::-1] * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(cols_names_loaded))
        _check_column_dels(bodo_func, [list(range(n_cols)), list(range(n_cols * 2))])


def test_table_iloc_column_subset_level2(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format and multiple level of table_format subset via
    df.iloc.

    This is used to ensure used_cols is properly updated at each step
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols_level_1 = max(1, bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD + 4)
    columns_loaded_level_1 = list(range(n_cols_level_1))
    n_cols_level_2 = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded_level_2 = list(
        range(n_cols_level_1 - n_cols_level_2, n_cols_level_1)
    )
    cols_name_loaded_level_2 = [f"Column{i}" for i in range(n_cols_level_2)]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        df1 = df.iloc[:, {columns_loaded_level_1[::-1]}]
        return df1.iloc[:, {columns_loaded_level_2 * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(cols_name_loaded_level_2))
        # df deletes columns in the original location, but df1
        # deletes them at the new remapped location.
        df_col_nums = list(range(n_cols_level_2))
        df1_col_nums = list(range(n_cols_level_1 - n_cols_level_2, n_cols_level_1))
        df2_col_nums = list(range(n_cols_level_2 * 2))
        # There is 1 del_columns for the original df, 2 for df1 (the subset
        # and the filter) and 1 for df2 (the subset)
        _check_column_dels(
            bodo_func, [df_col_nums, df1_col_nums, df1_col_nums, df2_col_nums]
        )


def test_table_column_subset_level1(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format. This is useful to maintain projections,
    especially for BodoSQL.

    BodoSQL generated code may select a large subset of columns, even
    if uncommon in user generated code.
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded = [f"Column{i}" for i in range(n_cols)]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        return df[{columns_loaded[::-1] * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(columns_loaded))
        _check_column_dels(bodo_func, [list(range(n_cols))])


def test_table_column_subset_level2(datapath, memory_leak_check):
    """
    Tests selecting a subset of the columns (i.e. [["A", "B", "C"]])
    using table format and multiple level of table_format subset.

    This is used to ensure used_cols is properly updated at each step
    """
    filename = datapath("many_columns.parquet")
    # Load enough columns to ensure we get table format. We divide
    # by 2 to check for duplicate support.
    n_cols_level_1 = max(1, bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD + 4)
    columns_loaded_level_1 = [f"Column{i}" for i in range(n_cols_level_1)]
    n_cols_level_2 = max(1, math.ceil(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD / 2))
    columns_loaded_level_2 = [f"Column{i}" for i in range(n_cols_level_2)]

    # Note we reverse the columns to check reordering
    func_text = f"""def impl():
        df = pd.read_parquet({filename!r})
        df1 = df[{columns_loaded_level_1[::-1]}]
        return df1[{columns_loaded_level_2 * 2}]"""

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Check the columns were pruned
        check_logger_msg(stream, str(columns_loaded_level_2))
        # df deletes columns in the original location, but df1
        # deletes them at the new remapped location.
        df_col_nums = list(range(n_cols_level_2))
        df1_col_nums = list(range(n_cols_level_1 - n_cols_level_2, n_cols_level_1))
        _check_column_dels(bodo_func, [df_col_nums, df1_col_nums])


def test_merge_del_columns(datapath, memory_leak_check):
    # Verify that column pruning from the source and dead
    # column insertion works with Join.
    filename = datapath(f"many_columns.parquet")

    def impl():
        df1 = pd.read_parquet(filename)
        df2 = df1.merge(df1, on="Column0", how="inner", suffixes=("_x", "_y"))
        return df2[["Column1_x", "Column21_y"]]

    check_func(impl, (), sort_output=True, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column1', 'Column21']")
        check_logger_msg(stream, f"Output columns: ['Column1_x', 'Column21_y']")
        # We prune columns 0, 1, 21 from the read parquet and 1 and 119
        # from the join.
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 2:
            # threshold for if the df[[]] creates another table,
            # which adds deletions
            delete_list = [[1], [0, 21], [1, 119]]
        else:
            # [1] and [0, 21] are handled separately because each
            # is used separately with py_data_to_cpp_table
            delete_list = [[1], [0, 21], [1], [119]]
        _check_column_dels(bodo_func, delete_list)


def test_merge_del_columns_tuple(datapath, memory_leak_check):
    """
    Tests executing pd.merge with DataFrames that don't have table
    format still enable optimizations, such as removing columns.
    """
    if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD <= 10:
        # Only test this with tuple format.
        return

    # Verify that column pruning from the source and dead
    # column insertion works with Join.
    filename = datapath(f"many_columns.parquet")

    def impl1():
        df1 = pd.read_parquet(filename)
        df2 = df1[
            [
                "Column0",
                "Column1",
                "Column2",
                "Column3",
                "Column4",
                "Column5",
                "Column6",
                "Column7",
                "Column8",
                "Column9",
            ]
        ]
        df3 = df1.merge(df2, on="Column0", how="inner", suffixes=("_x", "_y"))
        return df3[["Column1_y", "Column21"]]

    def impl2():
        df1 = pd.read_parquet(filename)
        df2 = df1[
            [
                "Column0",
                "Column1",
                "Column2",
                "Column3",
                "Column4",
                "Column5",
                "Column6",
                "Column7",
                "Column8",
                "Column9",
            ]
        ]
        df3 = df2.merge(df1, on="Column0", how="inner", suffixes=("_x", "_y"))
        return df3[["Column1_x", "Column21"]]

    def impl3():
        df1 = pd.read_parquet(filename)
        df2 = df1[
            [
                "Column0",
                "Column1",
                "Column2",
                "Column3",
                "Column4",
                "Column5",
                "Column6",
                "Column7",
                "Column8",
                "Column9",
            ]
        ]
        df3 = df2.merge(df2, on="Column0", how="inner", suffixes=("_x", "_y"))
        return df3[["Column1_x", "Column2_y"]]

    check_func(impl1, (), sort_output=True, reset_index=True)
    check_func(impl2, (), sort_output=True, reset_index=True)
    check_func(impl3, (), sort_output=True, reset_index=True)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl1)
        bodo_func()
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column1', 'Column21']")
        check_logger_msg(stream, f"Output columns: ['Column21', 'Column1_y']")
        _check_column_dels(bodo_func, [[1], [0, 21], [21], [99]])

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl2)
        bodo_func()
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column1', 'Column21']")
        check_logger_msg(stream, f"Output columns: ['Column1_x', 'Column21']")
        _check_column_dels(bodo_func, [[1], [0, 21], [1], [30]])

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl3)
        bodo_func()
        check_logger_msg(stream, f"Columns loaded ['Column0', 'Column1', 'Column2']")
        check_logger_msg(stream, f"Output columns: ['Column1_x', 'Column2_y']")
        _check_column_dels(bodo_func, [[0], [1], [2], [1], [11]])


def test_parquet_tail(datapath, memory_leak_check):
    def impl():
        df = pd.read_parquet(filename)
        return len(df.tail(10000))

    filename = datapath(f"many_columns.parquet")
    check_func(impl, (), sort_output=True, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
        # There shouldn't be a need to load any columns.
        check_logger_msg(stream, f"Columns loaded []")
