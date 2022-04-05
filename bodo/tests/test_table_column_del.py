import os
import shutil

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core import ir

import bodo
from bodo.tests.utils import (
    ColumnDelTestPipeline,
    _create_many_column_file,
    _del_many_column_file,
    check_func,
    reduce_sum,
)
from bodo.utils.utils import is_expr


def _check_column_dels(bodo_func, col_del_lists):
    """
    Helper functions to check for the col_del calls inserted
    into the IR in BodoTableColumnDelPass. Since we don't know
    the exact blocks + labels in the IR, we cannot pass a dictionary
    or expected structure.

    Instead we pass 'col_del_lists', which is a list of lists for just
    the blocks that will delete columns for example. If we know that one
    block should delete 1 then 3 and another should just delete, we pass
    [[1, 3], [2]]. There may be many other blocks that don't remove any
    columns, but we just verify that all elements of this list are
    encountered (and nothing outside this list).
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    typemap = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_typemap"]
    for block in fir.blocks.values():
        block_del_cols = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and is_expr(stmt.value, "call"):
                typ = typemap[stmt.value.func.name]
                if (
                    isinstance(typ, numba.core.types.functions.Function)
                    and typ.name == "Function(<intrinsic del_column>)"
                ):
                    col_num = typemap[stmt.value.args[1].name].literal_value
                    block_del_cols.append(col_num)
        if block_del_cols:
            removed = False
            for i, col_list in enumerate(col_del_lists):
                if col_list == block_del_cols:
                    to_pop = i
                    removed = True
                    break
            assert removed, f"Unexpected Del Column list {block_del_cols}"
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


def test_table_len(file_type, memory_leak_check):
    """
    Check that an operation that just returns a length
    and doesn't use any particular column still computes
    a correct result.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            return len(df)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that just column 0 was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_filter_dead_columns(memory_leak_check):
    """
    Test table filter with no used column (just length)
    """
    try:
        _create_many_column_file("parquet")

        # no columns used (just length needed)
        def impl(idx):
            df = pd.read_parquet("many_columns.parquet")
            df2 = df[idx]
            return len(df2)

        idx = np.arange(len(pd.read_parquet("many_columns.parquet"))) % 3 == 0
        check_func(impl, (idx,), only_seq=True)
        # NOTE: this needs distributed=False since args/returns don't force
        # sequential execution.
        check_func(
            impl,
            (slice(10),),
            only_seq=True,
            additional_compiler_arguments={"distributed": False},
        )
    finally:
        _del_many_column_file(file_type)


def test_table_len_with_idx_col(memory_leak_check):
    """
    Check that an operation that just returns a length
    and doesn't use any particular column still computes
    a correct result.

    Manually verified that the index col is dead/removed
    """
    try:
        file_type = "csv"
        _create_many_column_file(file_type)

        def impl():
            df = pd.read_csv("many_columns.csv", index_col="Column0")
            return len(df)

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that just column 0 was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_shape(file_type, memory_leak_check):
    """
    Check that an operation that just returns a shape
    and doesn't use any particular column still computes
    a correct result.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            return df.shape"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that just column 0 was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_del_single_block(file_type, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns with a single block.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            return df[["Column3", "Column37", "Column59"]]"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_back(file_type, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove some after the first
    basic block.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            size = df["Column0"].mean()
            if size > 10000:
                n = 100
            else:
                n = 500
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 4 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_front(file_type, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove a column at the start
    of some successors.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            size = df["Column0"].mean()
            if size > 10000:
                n = df["Column0"].sum()
            else:
                n = df["Column3"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 4 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0], [0], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_front_back(file_type, memory_leak_check):
    """
    Test dead column elimination where some columns
    can be removed at the end of basic blocks but
    others must be removed at the start of basic blocks
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            size = df["Column0"].mean() - df["Column3"].mean()
            if size > 0:
                n = df["Column0"].sum() + df["Column6"].sum()
            else:
                n = df["Column3"].sum() + df["Column9"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        _check_column_dels(bodo_func, [[9, 0], [6], [0, 6], [9], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_useall_later_block(file_type, memory_leak_check):
    """
    Check that an operation that requires using all
    columns eventually doesn't eliminate any columns.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that all columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_useall_early_block(file_type, memory_leak_check):
    """
    Check that an operation that requires using all
    columns early doesn't eliminate any columns later.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that all columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_del_usecols(file_type, memory_leak_check):
    """
    Test dead column elimination where a user has
    provided usecols as well
    """
    try:
        _create_many_column_file(file_type)
        columns = [0, 1, 2, 3, 4, 5, 6, 9, 10, 37, 38, 52, 59, 67, 95, 96, 97, 98]
        if file_type == "csv":
            kwarg = "usecols"
        elif file_type == "parquet":
            kwarg = "columns"
            columns = [f"Column{i}" for i in columns]
        func_text = f"""def impl():
            df = pd.read_{file_type}(
                "many_columns.{file_type}",
                {kwarg}={columns},
            )
            size = df["Column0"].mean() - df["Column3"].mean()
            if size > 0:
                n = df["Column0"].sum() + df["Column6"].sum()
            else:
                n = df["Column3"].sum() + df["Column9"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # The column numbers here are remapped to their index in usecols.
        # Note 6 and 7 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        _check_column_dels(bodo_func, [[7, 0], [6], [0, 6], [7], [3, 9, 12]])
    finally:
        _del_many_column_file(file_type)


def test_table_set_table_columns(file_type, memory_leak_check):
    """
    Tests setting a table can still run dead column elimination.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            size = df["Column0"].mean() - df["Column3"].mean()
            if size > 0:
                n = df["Column0"].sum() + df["Column6"].sum()
            else:
                n = df["Column3"].sum() + df["Column9"].sum()
            df["Column37"] = np.arange(1000)
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        _check_column_dels(bodo_func, [[9, 0], [6], [0, 6], [9], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_extra_column(file_type, memory_leak_check):
    """
    Tests that using a column outside the original column list doesn't have
    any issues when loading from a file.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            return df["Column99"].sum()"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # We should only remove the column added by set_table_data
        _check_column_dels(bodo_func, [[99]])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_var(file_type, memory_leak_check):
    """
    Tests removing columns when a variable is dead in certain parts of
    control flow.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            n = df["Column0"].sum()
            if n > 100:
                return df["Column0"].mean()
            else:
                return 1.0"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 1 column was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0]])
    finally:
        _del_many_column_file(file_type)


def test_table_for_loop(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
            total = 0.0
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_while_loop(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
            total = 0.0
            while n > 0:
                total += df["Column0"].sum()
                n -= 1
            return total + df["Column3"].sum()"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_for_loop_branch(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_while_loop_branch(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_loop_unroll(file_type, memory_leak_check):
    """
    Tests removing columns with a loop that
    requires unrolling.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            total = 0.0
            columns = ["Column0", "Column3", "Column6"]
            for c in columns:
                total += df[c].sum()
            return total"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note: all values are separate because sum adds extra basic blocks
        _check_column_dels(bodo_func, [[0], [3], [6]])
    finally:
        _del_many_column_file(file_type)


def test_table_return(file_type, memory_leak_check):
    """
    Tests that returning a table avoids dead column elimination.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that all columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


# Tests involving an alias. Numba can sometimes optimize out aliases in user code
# so we include a set_table_data call at the front via df["Column99"] = np.arange(1000)
def test_table_len_alias(file_type, memory_leak_check):
    """
    Check that len only loads a single column when there is
    an alias
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            return len(df)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that just column 0 was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_shape_alias(file_type, memory_leak_check):
    """
    Check that shape only loads a single column when there is
    an alias
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            return df.shape"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that just column 0 was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_del_single_block_alias(file_type, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns with a single block and an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            return df[["Column3", "Column37", "Column59"]]"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_back_alias(file_type, memory_leak_check):
    """
    Test dead column elimination that loads a subset of
    columns and can remove some after the first
    basic block.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            size = df["Column0"].mean()
            if size > 10000:
                n = 100
            else:
                n = 500
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 4 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_front_alias(file_type, memory_leak_check):
    """
    Test dead column elimination with an alias that
    loads a subset of columns and can remove a column
    at the start of some successors.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            size = df["Column0"].mean()
            if size > 10000:
                n = df["Column0"].sum()
            else:
                n = df["Column3"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 4 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0], [0], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_front_back_alias(file_type, memory_leak_check):
    """
    Test dead column elimination with an alias where
    some columns can be removed at the end of basic
    blocks but others must be removed at the start of
    basic blocks
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            size = df["Column0"].mean() - df["Column3"].mean()
            if size > 0:
                n = df["Column0"].sum() + df["Column6"].sum()
            else:
                n = df["Column3"].sum() + df["Column9"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note 6 and 9 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        _check_column_dels(bodo_func, [[9, 0], [6], [0, 6], [9], [3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_useall_later_block_alias(file_type, memory_leak_check):
    """
    Check that an operation with an alias that requires
    using all columns eventually doesn't eliminate any columns.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that all columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_useall_early_block_alias(file_type, memory_leak_check):
    """
    Check that an operation with an alias that
    requires using all columns early doesn't
    eliminate any columns later.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that all columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # There shouldn't be any del_column calls
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_del_usecols_alias(file_type, memory_leak_check):
    """
    Test dead column elimination where a user has
    provided usecols as well + an alias.
    """
    try:
        _create_many_column_file(file_type)

        columns = [0, 1, 2, 3, 4, 5, 6, 9, 10, 37, 38, 52, 59, 67, 95, 96, 97, 98]
        if file_type == "csv":
            kwarg = "usecols"
        elif file_type == "parquet":
            kwarg = "columns"
            columns = [f"Column{i}" for i in columns]
        func_text = f"""def impl():
            df = pd.read_{file_type}(
                "many_columns.{file_type}",
                {kwarg}={columns},
            )
            df["Column99"] = np.arange(1000)
            size = df["Column0"].mean() - df["Column3"].mean()
            if size > 0:
                n = df["Column0"].sum() + df["Column6"].sum()
            else:
                n = df["Column3"].sum() + df["Column9"].sum()
            return df[["Column3", "Column37", "Column59"]].head(n)"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 6 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # The column numbers here are remapped to their index in usecols.
        # Note 6 and 7 are deleted on their own in the if statement because the
        # sum calls create a new basic block
        _check_column_dels(bodo_func, [[7, 0], [6], [0, 6], [7], [3, 9, 12]])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_var_alias(file_type, memory_leak_check):
    """
    Tests removing columns when a variable is dead in certain parts of
    control flow with an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 1 column was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[0]])
    finally:
        _del_many_column_file(file_type)


def test_table_for_loop_alias(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
            df["Column99"] = np.arange(1000)
            total = 0.0
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()"""

        local_vars = {}
        exec(func_text, globals(), local_vars)
        impl = local_vars["impl"]

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_while_loop_alias(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_for_loop_branch_alias(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch
    with an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_while_loop_branch_alias(file_type, memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop inside a branch with
    an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl(n):
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_loop_unroll_alias(file_type, memory_leak_check):
    """
    Tests removing columns with a loop that
    requires unrolling with an alias.
    """
    try:
        _create_many_column_file(file_type)

        func_text = f"""def impl():
            df = pd.read_{file_type}("many_columns.{file_type}")
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
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        # Note: all values are separate because sum adds extra basic blocks
        _check_column_dels(bodo_func, [[0], [3], [6]])
    finally:
        _del_many_column_file(file_type)


#### Index column tests
def test_table_del_single_block_pq_index(memory_leak_check):
    """
    Test dead column elimination works with a DataFrame that
    loads an index from parquet.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl():
            df = pd.read_parquet("many_columns.parquet")
            return df[["Column3", "Column37", "Column59"]]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns + index were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_del_single_block_pq_index_alias(memory_leak_check):
    """
    Test dead column elimination works with a DataFrame that
    loads an index from parquet with an alias.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl():
            df = pd.read_parquet("many_columns.parquet")
            df["Column99"] = np.arange(1000)
            return df[["Column3", "Column37", "Column59"]]

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns + index were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [[3, 37, 59]])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_pq_index(memory_leak_check):
    """
    Test dead code elimination still works for unused indices.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl(n):
            df = pd.read_parquet("many_columns.parquet")
            total = 0.0
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns and not the index were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_pq_index_alias(memory_leak_check):
    """
    Test dead code elimination still works for unused indices with an alias.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl(n):
            df = pd.read_parquet("many_columns.parquet")
            df["Column99"] = np.arange(1000)
            total = 0.0
            for _ in range(n):
                total += df["Column0"].sum()
            return total + df["Column3"].sum()

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns and not the index were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_while_loop_alias_with_idx_col(memory_leak_check):
    """
    Tests removing columns when using a column
    repeatedly in a for loop with an alias.

    I've manually confirmed that the index column is correctly marked as dead.
    TODO: add automatic check to insure that the index column is marked as dead
    """
    file_type = "csv"
    try:
        _create_many_column_file(file_type)

        def impl(n):
            df = pd.read_csv("many_columns.csv")
            df["Column99"] = np.arange(1000)
            total = 0.0
            while n > 0:
                total += df["Column0"].sum()
                n -= 1
            return total + df["Column3"].sum()

        check_func(impl, (25,))
        # TODO [BE-2440]: Add code to check that only 2 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func(25)
        _check_column_dels(bodo_func, [[0, 3]])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_pq_table(memory_leak_check):
    """
    Test dead code elimination still works on a table with a used index.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl():
            df = pd.read_parquet("many_columns.parquet")
            df["Column99"] = np.arange(1000)
            return df.index

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only the index was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_pq_table_alias(memory_leak_check):
    """
    Test dead code elimination still works on a table with a used index
    and an alias.
    """
    file_type = "parquet"
    try:
        _create_many_column_file(file_type, indexcol="Column77")

        def impl():
            df = pd.read_parquet("many_columns.parquet")
            df["Column99"] = np.arange(1000)
            return df.index

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only the index was loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_table_dead_csv(memory_leak_check):
    """
    Tests returning only the index succeeds.
    I've manually confirmed that the table variable is correctly marked as dead.
    TODO: add automatic check to insure that the table is marked as dead
    """
    file_type = "csv"
    try:
        _create_many_column_file(file_type)

        def impl():
            df = pd.read_csv("many_columns.csv", index_col="Column4")
            return df.index

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)


def test_many_cols_to_parquet(memory_leak_check):
    """Tests df.to_parquet with many columns."""
    file_type = "csv"
    try:
        _create_many_column_file(file_type)

        def impl(source_filename, dest_filename):
            df = pd.read_csv(source_filename)
            df.to_parquet(dest_filename)

        def check_correctness(pandas_filename, bodo_filename):
            pandas_df = pd.read_parquet(pandas_filename)
            bodo_df = pd.read_parquet(bodo_filename)
            try:
                pd.testing.assert_frame_equal(pandas_df, bodo_df)
                return 1
            except Exception:
                return 0

        pandas_filename = "pandas_out.pq"
        bodo_filename = "bodo_out.pq"
        impl("many_columns.csv", pandas_filename)
        bodo.jit(impl)("many_columns.csv", bodo_filename)
        passed = 1
        if bodo.get_rank() == 0:
            passed = check_correctness(pandas_filename, bodo_filename)
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()
    finally:
        if bodo.get_rank() == 0:
            os.remove(pandas_filename)
            shutil.rmtree(bodo_filename, ignore_errors=True)
        _del_many_column_file(file_type)


def test_table_dead_csv(memory_leak_check):
    """
    Tests returning only the index succeeds.
    I've manually confirmed that the table variable is correctly marked as dead.
    TODO: add automatic check to insure that the table is marked as dead
    """
    file_type = "csv"
    try:
        _create_many_column_file(file_type)

        def impl():
            df = pd.read_csv("many_columns.csv", index_col="Column4")
            return df.index

        check_func(impl, ())
        # TODO [BE-2440]: Add code to check that only 3 columns were loaded
        bodo_func = bodo.jit(pipeline_class=ColumnDelTestPipeline)(impl)
        bodo_func()
        _check_column_dels(bodo_func, [])
    finally:
        _del_many_column_file(file_type)
