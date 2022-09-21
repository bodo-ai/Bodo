"""Utility functions for testing such as check_func() that tests a function.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.
import io
import os
import random
import string
import time
import types as pytypes
import warnings
from contextlib import contextmanager
from decimal import Decimal
from enum import Enum
from typing import Dict
from uuid import uuid4

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from numba.core import ir, types
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import find_callname, guard
from numba.core.typed_passes import NopythonRewrites
from numba.core.untyped_passes import PreserveIR

import bodo
from bodo.utils.typing import BodoWarning, dtype_to_array_type
from bodo.utils.utils import (
    is_assign,
    is_distributable_tuple_typ,
    is_distributable_typ,
    is_expr,
)

# TODO: Include testing DBs for other systems: MSSQL, SQLite, ...
sql_user_pass_and_hostname = (
    "user:pass@localhost"
)
oracle_user_pass_and_hostname = "user:pass@localhost"


class InputDist(Enum):
    """
    Enum used to represent the various
    distributed analysis options for input
    data.
    """

    REP = 0
    OneD = 1
    OneDVar = 2


def count_array_REPs():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_REPs():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_OneDs():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_array_OneDs():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_parfor_OneD_Vars():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def count_array_OneD_Vars():
    from bodo.transforms.distributed_pass import Distribution

    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def dist_IR_contains(f_ir, *args):
    """check if strings 'args' are in function IR 'f_ir'"""
    with io.StringIO() as str_io:
        f_ir.dump(str_io)
        f_ir_text = str_io.getvalue()
    return sum([(s in f_ir_text) for s in args])


@bodo.jit
def get_rank():
    return bodo.libs.distributed_api.get_rank()


@bodo.jit(cache=True)
def get_start_end(n):
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(n, n_pes, rank)
    return start, end


@numba.njit
def reduce_sum(val):
    sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
    return bodo.libs.distributed_api.dist_reduce(val, np.int32(sum_op))


def check_func(
    func,
    args,
    is_out_distributed=None,
    sort_output=False,
    check_names=True,
    copy_input=False,
    check_dtype=True,
    reset_index=False,
    convert_columns_to_pandas=False,
    py_output=None,
    dist_test=True,
    check_typing_issues=True,
    additional_compiler_arguments=None,
    set_columns_name_to_none=False,
    reorder_columns=False,
    only_seq=False,
    only_1D=False,
    only_1DVar=False,
    check_categorical=False,
    atol=1e-08,
    rtol=1e-05,
    use_table_format=None,
    use_dict_encoded_strings=None,
):
    """test bodo compilation of function 'func' on arguments using REP, 1D, and 1D_Var
    inputs/outputs

    Rationale of the functionalities:
    - is_out_distributed: By default to None and is adjusted according to the REP/1D/1D_Var
    If the is_out_distributed is selected then it is hardcoded here.
    - sort_output: The order of the data produced may not match pandas (e.g. groupby/join).
    In that case, set sort_output=True
    - reset_index: The index produced by pandas may not match the one produced by Bodo for good
    reasons. In that case, set sort_output=True
    - convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
    list of strings or in general arrow data type. It typically fails at sorting. In that case
    using convert_columns_to_pandas=True will convert the columns to a string format which may help.
    - check_dtype: pandas may produce a column of float while bodo may produce a column of integers
    for some operations. Using check_dtype=False ensures that comparison is still possible.
    - py_output: Sometimes pandas has entirely lacking functionality and we need to put what output
    we expect to obtain.
    - additional_compiler_arguments: For some operations (like pivot_table) some additional compiler
    arguments are needed. These are keyword arguments to bodo.jit() passed as a dictionary.
    - set_columns_name_to_none: Some operation (like pivot_table) set a name to the list of columns.
    This is mostly for esthetic purpose and has limited support, therefore sometimes we have to set
    the name to None.
    - reorder_columns: The columns of the output have some degree of uncertainty sometimes (like pivot_table)
    thus a reordering operation is needed in some cases to make the comparison meaningful.
    - only_seq: Run just the sequential check.
    - only_1D: Run just the check on a 1D Distributed input.
    - only_1DVar: Run just the check on a 1DVar Distributed input.
    - check_categorical: Argument to pass to Pandas assert_frame_equals. We use this if we want to disable
    the check_dtype with a categorical input (as otherwise it will still raise an error).
    - atol: Argument to pass to Pandas assert equals functions. This argument will be used if
    floating point percision can vary due to differences in underlying floating point libraries.
    - rtol: Argument to pass to Pandas assert equals functions. This argument will be used if
    floating point percision can vary due to differences in underlying floating point libraries.
    - use_table_format: flag for loading dataframes in table format for testing.
    If None, tests both formats if input arguments have dataframes.
    - use_dict_encoded_strings: flag for loading string arrays in dictionary-encoded
    format for testing.
    If None, tests both formats if input arguments have string arrays.
    """
    run_seq, run_1D, run_1DVar = False, False, False
    if only_seq:
        if only_1D or only_1DVar:
            warnings.warn(
                "Multiple select only options specified, running only sequential."
            )
        run_seq = True
        dist_test = False
    elif only_1D:
        if only_1DVar:
            warnings.warn("Multiple select only options specified, running only 1D.")
        run_1D = True
    elif only_1DVar:
        run_1DVar = True
    else:
        run_seq, run_1D, run_1DVar = True, True, True

    n_pes = bodo.get_size()

    # avoid running sequential tests on multi-process configs to save time
    # is_out_distributed=False may lead to avoiding parallel runs and seq run is
    # necessary to capture the parallelism warning (see "w is not None" below)
    if (
        n_pes > 1
        and not numba.core.config.DEVELOPER_MODE
        and is_out_distributed is not False
    ):
        run_seq = False

    call_args = tuple(_get_arg(a, copy_input) for a in args)
    w = None

    # gives the option of passing desired output to check_func
    # in situations where pandas is buggy/lacks support
    if py_output is None:
        if convert_columns_to_pandas:
            call_args_mapped = tuple(convert_non_pandas_columns(a) for a in call_args)
            py_output = func(*call_args_mapped)
        else:
            py_output = func(*call_args)
    else:
        if convert_columns_to_pandas:
            py_output = convert_non_pandas_columns(py_output)
    if set_columns_name_to_none:
        py_output.columns.name = None
    if reorder_columns:
        py_output.sort_index(axis=1, inplace=True)

    saved_TABLE_FORMAT_THRESHOLD = bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    saved_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    try:
        # test table format for dataframes (non-table format tested below if flag is
        # None)
        if use_table_format is None or use_table_format:
            bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = 0

        # test dict-encoded string arrays if flag is set (dict-encoded tested below if
        # flag is None)
        if use_dict_encoded_strings:
            bodo.hiframes.boxing._use_dict_str_type = True

        # sequential
        if run_seq:
            w = check_func_seq(
                func,
                args,
                py_output,
                copy_input,
                sort_output,
                check_names,
                check_dtype,
                reset_index,
                convert_columns_to_pandas,
                additional_compiler_arguments,
                set_columns_name_to_none,
                reorder_columns,
                n_pes,
                check_categorical,
                atol,
                rtol,
            )
            # test string arguments as StringLiteral type also (since StringLiteral is
            # not a subtype of UnicodeType)
            if any(isinstance(a, str) for a in args):
                check_func_seq(
                    func,
                    args,
                    py_output,
                    copy_input,
                    sort_output,
                    check_names,
                    check_dtype,
                    reset_index,
                    convert_columns_to_pandas,
                    additional_compiler_arguments,
                    set_columns_name_to_none,
                    reorder_columns,
                    n_pes,
                    check_categorical,
                    atol,
                    rtol,
                    True,
                )

        # distributed test is not needed
        if not dist_test:
            return

        if is_out_distributed is None:
            # assume all distributable output is distributed if not specified
            is_out_distributed = is_distributable_typ(
                _typeof(py_output)
            ) or is_distributable_tuple_typ(_typeof(py_output))

        # skip 1D distributed and 1D distributed variable length tests
        # if no parallelism is found
        # and if neither inputs nor outputs are distributable
        if (
            w is not None  # if no parallelism is found
            and not is_out_distributed  # if output is not distributable
            and not any(
                is_distributable_typ(_typeof(a))
                or is_distributable_tuple_typ(_typeof(a))
                for a in args
            )  # if none of the inputs is distributable
        ):
            return  # no need for distributed checks

        if run_1D:
            check_func_1D(
                func,
                args,
                py_output,
                is_out_distributed,
                copy_input,
                sort_output,
                check_names,
                check_dtype,
                reset_index,
                check_typing_issues,
                convert_columns_to_pandas,
                additional_compiler_arguments,
                set_columns_name_to_none,
                reorder_columns,
                n_pes,
                check_categorical,
                atol,
                rtol,
            )

        if run_1DVar:
            check_func_1D_var(
                func,
                args,
                py_output,
                is_out_distributed,
                copy_input,
                sort_output,
                check_names,
                check_dtype,
                reset_index,
                check_typing_issues,
                convert_columns_to_pandas,
                additional_compiler_arguments,
                set_columns_name_to_none,
                reorder_columns,
                n_pes,
                check_categorical,
                atol,
                rtol,
            )
    finally:
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = saved_TABLE_FORMAT_THRESHOLD
        bodo.hiframes.boxing._use_dict_str_type = saved_use_dict_str_type

    # test non-table format case if there is any dataframe in input
    if use_table_format is None and any(
        isinstance(_typeof(a), bodo.DataFrameType) for a in args
    ):
        check_func(
            func,
            args,
            is_out_distributed,
            sort_output,
            check_names,
            copy_input,
            check_dtype,
            reset_index,
            convert_columns_to_pandas,
            py_output,
            dist_test,
            check_typing_issues,
            additional_compiler_arguments,
            set_columns_name_to_none,
            reorder_columns,
            only_seq,
            only_1D,
            only_1DVar,
            check_categorical,
            atol,
            rtol,
            use_table_format=False,
            use_dict_encoded_strings=use_dict_encoded_strings,
        )

    # test dict-encoded string type if there is any string array in input
    if use_dict_encoded_strings is None and any(
        _type_has_str_array(_typeof(a)) for a in args
    ):
        check_func(
            func,
            args,
            is_out_distributed,
            sort_output,
            check_names,
            copy_input,
            check_dtype,
            reset_index,
            convert_columns_to_pandas,
            py_output,
            dist_test,
            check_typing_issues,
            additional_compiler_arguments,
            set_columns_name_to_none,
            reorder_columns,
            only_seq,
            only_1D,
            only_1DVar,
            check_categorical,
            atol,
            rtol,
            # the default case use_table_format=None already tests
            # use_table_format=False above so we just test use_table_format=True for it
            use_table_format=True if use_table_format is None else use_table_format,
            use_dict_encoded_strings=True,
        )


def _type_has_str_array(t):
    """Return True if input type 't' has a string array component: string array,
    string Series, DataFrame with one or more string columns.

    Args:
        t (types.Type): input type

    Returns:
        bool: True if input type 't' has a string array component
    """
    return (
        (t == bodo.string_array_type)
        or (isinstance(t, bodo.SeriesType) and t.data == bodo.string_array_type)
        or (
            isinstance(t, bodo.DataFrameType)
            and any(a == bodo.string_array_type for a in t.data)
        )
    )


def check_func_seq(
    func,
    args,
    py_output,
    copy_input,
    sort_output,
    check_names,
    check_dtype,
    reset_index,
    convert_columns_to_pandas,
    additional_compiler_arguments,
    set_columns_name_to_none,
    reorder_columns,
    n_pes,
    check_categorical,
    atol,
    rtol,
    test_str_literal=False,
):
    """check function output against Python without manually setting inputs/outputs
    distributions (keep the function sequential)
    """
    kwargs = {"returns_maybe_distributed": False}
    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)
    bodo_func = bodo.jit(func, **kwargs)

    # type string inputs as literal
    if test_str_literal:
        # create a wrapper around function and call numba.literally() on str args
        args_str = ", ".join(f"a{i}" for i in range(len(args)))
        func_text = f"def wrapper({args_str}):\n"
        for i in range(len(args)):
            if isinstance(args[i], str):
                func_text += f"  numba.literally(a{i})\n"
        func_text += f"  return bodo_func({args_str})\n"
        loc_vars = {}
        exec(func_text, {"bodo_func": bodo_func, "numba": numba}, loc_vars)
        wrapper = loc_vars["wrapper"]
        bodo_func = bodo.jit(wrapper)

    call_args = tuple(_get_arg(a, copy_input) for a in args)
    # try to catch BodoWarning if no parallelism found
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always", message="No parallelism found for function", category=BodoWarning
        )
        bodo_out = bodo_func(*call_args)
        if convert_columns_to_pandas:
            bodo_out = convert_non_pandas_columns(bodo_out)
    if set_columns_name_to_none:
        bodo_out.columns.name = None
    if reorder_columns:
        bodo_out.sort_index(axis=1, inplace=True)
    passed = _test_equal_guard(
        bodo_out,
        py_output,
        sort_output,
        check_names,
        check_dtype,
        reset_index,
        check_categorical,
        atol,
        rtol,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes, "Sequential test failed"
    return w


def check_func_1D(
    func,
    args,
    py_output,
    is_out_distributed,
    copy_input,
    sort_output,
    check_names,
    check_dtype,
    reset_index,
    check_typing_issues,
    convert_columns_to_pandas,
    additional_compiler_arguments,
    set_columns_name_to_none,
    reorder_columns,
    n_pes,
    check_categorical,
    atol,
    rtol,
):
    """Check function output against Python while setting the inputs/outputs as
    1D distributed
    """
    # 1D distributed
    kwargs = {
        "all_args_distributed_block": True,
        "all_returns_distributed": is_out_distributed,
    }
    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)
    bodo_func = bodo.jit(func, **kwargs)
    dist_args = tuple(
        _get_dist_arg(a, copy_input, False, check_typing_issues) for a in args
    )
    bodo_output = bodo_func(*dist_args)
    if convert_columns_to_pandas:
        bodo_output = convert_non_pandas_columns(bodo_output)
    if is_out_distributed:
        bodo_output = _gather_output(bodo_output)
    if set_columns_name_to_none:
        bodo_output.columns.name = None
    if reorder_columns:
        bodo_output.sort_index(axis=1, inplace=True)

    # only rank 0 should check if gatherv() called on output
    passed = 1

    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
            check_categorical,
            atol,
            rtol,
        )

    n_passed = reduce_sum(passed)
    assert n_passed == n_pes, "Parallel 1D test failed"


def check_func_1D_var(
    func,
    args,
    py_output,
    is_out_distributed,
    copy_input,
    sort_output,
    check_names,
    check_dtype,
    reset_index,
    check_typing_issues,
    convert_columns_to_pandas,
    additional_compiler_arguments,
    set_columns_name_to_none,
    reorder_columns,
    n_pes,
    check_categorical,
    atol,
    rtol,
):
    """Check function output against Python while setting the inputs/outputs as
    1D distributed variable length
    """
    kwargs = {
        "all_args_distributed_varlength": True,
        "all_returns_distributed": is_out_distributed,
    }
    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)
    bodo_func = bodo.jit(func, **kwargs)
    dist_args = tuple(
        _get_dist_arg(a, copy_input, True, check_typing_issues) for a in args
    )
    bodo_output = bodo_func(*dist_args)
    if convert_columns_to_pandas:
        bodo_output = convert_non_pandas_columns(bodo_output)
    if is_out_distributed:
        bodo_output = _gather_output(bodo_output)
    if set_columns_name_to_none:
        bodo_output.columns.name = None
    if reorder_columns:
        bodo_output.sort_index(axis=1, inplace=True)

    passed = 1
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
            check_categorical,
            atol,
            rtol,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes, "Parallel 1D Var test failed"


def _get_arg(a, copy=False):
    if copy and hasattr(a, "copy"):
        return a.copy()
    return a


def _get_dist_arg(a, copy=False, var_length=False, check_typing_issues=True):
    """get distributed chunk for 'a' on current rank (for input to test functions)"""
    if copy and hasattr(a, "copy"):
        a = a.copy()

    if isinstance(a, pytypes.FunctionType):
        return a

    bodo_typ = bodo.typeof(a)
    if not (is_distributable_typ(bodo_typ) or is_distributable_tuple_typ(bodo_typ)):
        return a

    try:
        from bodosql import BodoSQLContext
        from bodosql.context_ext import BodoSQLContextType
    except ImportError:  # pragma: no cover
        BodoSQLContextType = None

    if BodoSQLContextType is not None and isinstance(bodo_typ, BodoSQLContextType):
        # Distribute each of the DataFrames.
        new_dict = {
            name: _get_dist_arg(df, copy, var_length, check_typing_issues)
            for name, df in a.tables.items()
        }
        return BodoSQLContext(new_dict, a.catalog)

    # PyArrow doesn't support shape
    l = len(a) if isinstance(a, pa.Array) else a.shape[0]

    start, end = get_start_end(l)
    # for var length case to be different than regular 1D in chunk sizes, add
    # one extra element to last processor
    if var_length and bodo.get_size() >= 2:
        if bodo.get_rank() == bodo.get_size() - 2:
            end -= 1
        if bodo.get_rank() == bodo.get_size() - 1:
            start -= 1

    if isinstance(a, (pd.Series, pd.DataFrame)):
        out_val = a.iloc[start:end]
    else:
        out_val = a[start:end]

    if check_typing_issues:
        _check_typing_issues(out_val)
    return out_val


def _test_equal_guard(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
    check_categorical=False,
    atol=1e-08,
    rtol=1e-05,
):
    # no need to avoid exceptions if running with a single process and hang is not
    # possible. TODO: remove _test_equal_guard in general when [BE-2223] is resolved
    if bodo.get_size() == 1:
        _test_equal(
            bodo_out,
            py_out,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
            check_categorical,
            atol,
            rtol,
        )
        return 1
    passed = 1
    try:
        _test_equal(
            bodo_out,
            py_out,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
            check_categorical,
            atol,
            rtol,
        )
    except Exception as e:
        print(e)
        passed = 0
    return passed


# We need to sort the index and values for effective comparison
def sort_series_values_index(S):
    S1 = S.sort_index()
    # pandas fails if all null integer column is sorted
    if S1.isnull().all():
        return S1
    return S1.sort_values(kind="mergesort")


def sort_dataframe_values_index(df):
    """sort data frame based on values and index"""
    if isinstance(df.index, pd.MultiIndex):
        # rename index in case names are None
        index_names = [f"index_{i}" for i in range(len(df.index.names))]
        list_col_names = df.columns.to_list() + index_names
        return df.rename_axis(index_names).sort_values(list_col_names, kind="mergesort")

    eName = "index123"
    list_col_names = df.columns.to_list() + [eName]
    return df.rename_axis(eName).sort_values(list_col_names, kind="mergesort")


def _test_equal(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
    check_categorical=False,
    atol=1e-08,
    rtol=1e-05,
):
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        csr_matrix = None

    # Bodo converts lists to array in array(item) array cases
    if isinstance(py_out, list) and isinstance(bodo_out, np.ndarray):
        py_out = np.array(py_out)

    if isinstance(py_out, pd.Series):
        if sort_output:
            py_out = sort_series_values_index(py_out)
            bodo_out = sort_series_values_index(bodo_out)
        if reset_index:
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.reset_index(inplace=True, drop=True)
        # we return typed extension arrays like StringArray for all APIs but Pandas
        # doesn't return them by default in all APIs yet.
        if py_out.dtype in (object, np.bool_):
            check_dtype = False
        # TODO: support freq attribute of DatetimeIndex/TimedeltaIndex
        pd.testing.assert_series_equal(
            bodo_out,
            py_out,
            check_names=check_names,
            check_categorical=check_categorical,
            check_dtype=check_dtype,
            check_index_type=False,
            check_freq=False,
            atol=atol,
            rtol=rtol,
        )
    elif isinstance(py_out, pd.Index):
        if sort_output:
            py_out = py_out.sort_values()
            bodo_out = bodo_out.sort_values()
        # avoid assert_index_equal() issues for ArrowStringArray comparison (exact=False
        # still fails for some reason)
        if bodo_out.dtype == pd.StringDtype("pyarrow"):
            bodo_out = bodo_out.astype(object)
        if isinstance(bodo_out, pd.MultiIndex):
            bodo_out = pd.MultiIndex(
                levels=[
                    v.values.to_numpy()
                    if isinstance(v.values, pd.arrays.ArrowStringArray)
                    else v
                    for v in bodo_out.levels
                ],
                codes=bodo_out.codes,
                names=bodo_out.names,
            )
        pd.testing.assert_index_equal(
            bodo_out,
            py_out,
            check_names=check_names,
            exact="equiv" if check_dtype else False,
            check_categorical=False,
        )
    elif isinstance(py_out, pd.DataFrame):
        if sort_output:
            py_out = sort_dataframe_values_index(py_out)
            bodo_out = sort_dataframe_values_index(bodo_out)
        if reset_index:
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.reset_index(inplace=True, drop=True)

        # We return typed extension arrays like StringArray for all APIs but Pandas
        # & Spark doesn't return them by default in all APIs yet.
        py_out_dtypes = py_out.dtypes.values.tolist()

        if object in py_out_dtypes or np.bool_ in py_out_dtypes:
            check_dtype = False

        pd.testing.assert_frame_equal(
            bodo_out,
            py_out,
            check_names=check_names,
            check_dtype=check_dtype,
            check_index_type=False,
            check_column_type=False,
            check_freq=False,
            check_categorical=check_categorical,
            atol=atol,
            rtol=rtol,
        )
    elif isinstance(py_out, np.ndarray):
        if sort_output:
            py_out = np.sort(py_out)
            bodo_out = np.sort(bodo_out)
        # use tester of Pandas for array of objects since Numpy doesn't handle np.nan
        # properly
        if py_out.dtype == np.dtype("O") and (
            bodo_out.dtype == np.dtype("O")
            or isinstance(
                bodo_out.dtype,
                (
                    pd.BooleanDtype,
                    pd.Int8Dtype,
                    pd.Int16Dtype,
                    pd.Int32Dtype,
                    pd.Int64Dtype,
                    pd.UInt8Dtype,
                    pd.UInt16Dtype,
                    pd.UInt32Dtype,
                    pd.UInt64Dtype,
                ),
            )
        ):
            # struct array needs special handling in nested case
            if len(py_out) > 0 and isinstance(py_out[0], dict):
                _test_equal_struct_array(bodo_out, py_out)
            else:
                pd.testing.assert_series_equal(
                    pd.Series(py_out),
                    pd.Series(bodo_out),
                    check_dtype=False,
                    atol=atol,
                    rtol=rtol,
                )
        else:
            # parallel reduction can result in floating point differences
            if py_out.dtype in (np.float32, np.float64):
                np.testing.assert_allclose(bodo_out, py_out, atol=atol, rtol=rtol)
            elif isinstance(bodo_out, pd.arrays.ArrowStringArray):
                pd.testing.assert_extension_array_equal(
                    bodo_out, pd.array(py_out, "string[pyarrow]")
                )
            else:
                np.testing.assert_array_equal(bodo_out, py_out)
    # check for array since is_extension_array_dtype() matches dtypes also
    elif pd.api.types.is_array_like(py_out) and pd.api.types.is_extension_array_dtype(
        py_out
    ):
        # bodo currently returns object array instead of pd StringArray
        if not pd.api.types.is_extension_array_dtype(bodo_out):
            bodo_out = pd.array(bodo_out)
        if sort_output:
            py_out = py_out[py_out.argsort()]
            bodo_out = bodo_out[bodo_out.argsort()]
        # We don't care about category order so sort always.
        if isinstance(py_out, pd.Categorical):
            py_out.categories = py_out.categories.sort_values()
        if isinstance(bodo_out, pd.Categorical):
            bodo_out.categories = bodo_out.categories.sort_values()
        pd.testing.assert_extension_array_equal(bodo_out, py_out, check_dtype=False)
    elif isinstance(py_out, csr_matrix):
        # https://stackoverflow.com/questions/30685024/check-if-two-scipy-sparse-csr-matrix-are-equal
        #
        # Similar to np.assert_array_equal we compare nan's like numbers,
        # so two nan's are considered equal. To detect nan's in sparse matrices,
        # we use the fact that nan's return False in all comparisons
        # according to IEEE-754, so nan is the only value that satisfies
        # `nan != nan` in a sparse matrix.
        # https://stackoverflow.com/questions/1565164/what-is-the-rationale-for-all-comparisons-returning-false-for-ieee754-nan-values
        #
        # Here, `(py_out != py_out).multiply(bodo_out != bodo_out)` counts the
        # number of instances where both values are nan, so the assertion
        # passes if in all the instances such that py_out != bodo_out, we also
        # know that both values are nan. Here, `.multiply()` performs logical-and
        # between nan instances of `py_out` and nan instances of `bodo_out`.
        assert (
            isinstance(bodo_out, csr_matrix)
            and py_out.shape == bodo_out.shape
            and (py_out != bodo_out).nnz
            == ((py_out != py_out).multiply(bodo_out != bodo_out)).nnz
        )
    # pyarrow array types
    elif isinstance(py_out, pa.Array):
        pd.testing.assert_series_equal(
            pd.Series(py_out),
            pd.Series(bodo_out),
            check_dtype=False,
            atol=atol,
            rtol=rtol,
        )
    elif isinstance(py_out, float):
        # avoid equality check since paralellism can affect floating point operations
        np.testing.assert_allclose(py_out, bodo_out, 1e-4)
    elif isinstance(py_out, tuple):
        assert len(py_out) == len(bodo_out)
        for p, b in zip(py_out, bodo_out):
            _test_equal(b, p, sort_output, check_names, check_dtype)
    elif isinstance(py_out, dict):
        _test_equal_struct(
            bodo_out,
            py_out,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
        )
    elif py_out is pd.NaT:
        assert py_out is bodo_out
    else:
        np.testing.assert_equal(bodo_out, py_out)


def _test_equal_struct(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
):
    """check struct/dict to be equal. checking individual elements separately since
    regular assertion cannot handle nested arrays properly
    """
    assert py_out.keys() == bodo_out.keys()
    for py_field, bodo_field in zip(py_out, bodo_out):
        _test_equal(
            bodo_field,
            py_field,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
        )


def _test_equal_struct_array(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
):
    """check struct arrays to be equal. checking individual elements separately since
    assert_series_equal() cannot handle nested case properly
    """
    assert len(py_out) == len(bodo_out)
    for i in range(len(py_out)):
        py_val = py_out[i]
        bodo_val = bodo_out[i]
        if pd.isna(py_val):
            assert pd.isna(bodo_val)
            continue
        _test_equal_struct(
            bodo_val,
            py_val,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
        )


def _gather_output(bodo_output):
    """gather bodo output from all processes. Uses bodo.gatherv() if there are no typing
    issues (e.g. empty object array). Otherwise, uses mpi4py's gather.
    """

    # don't gather scalar items of tuples since replicated
    if isinstance(bodo_output, tuple):
        return tuple(
            t if bodo.utils.typing.is_scalar_type(bodo.typeof(t)) else _gather_output(t)
            for t in bodo_output
        )

    try:
        _check_typing_issues(bodo_output)
        bodo_output = bodo.gatherv(bodo_output)
    except Exception as e:
        comm = MPI.COMM_WORLD
        bodo_output_list = comm.gather(bodo_output)
        if bodo.get_rank() == 0:
            if isinstance(bodo_output_list[0], np.ndarray):
                bodo_output = np.concatenate(bodo_output_list)
            else:
                bodo_output = pd.concat(bodo_output_list)

    return bodo_output


def _typeof(val):
    # Pandas returns an object array for .values or to_numpy() call on Series of
    # nullable int, which can't be handled in typeof. Bodo returns a nullable int array
    # see test_series_to_numpy[numeric_series_val3] and
    # test_series_get_values[series_val4]
    if (
        isinstance(val, np.ndarray)
        and val.dtype == np.dtype("O")
        and all(
            (isinstance(a, float) and np.isnan(a)) or isinstance(a, int) for a in val
        )
    ):
        # TODO: Should this check be fixed? It seems like all floats should
        # be an IntegerArray
        return bodo.libs.int_arr_ext.IntegerArrayType(bodo.int64)
    elif isinstance(val, pd.arrays.FloatingArray):
        # TODO: Add proper support for floating array
        # FloatingArrays are used somewhat extensively in Pandas >= 1.2.0
        # so we need to add further support. This code is fragile and
        # should not be considered reliable.
        return numba.core.types.Array(numba.core.types.float64, 1, "C")
    # TODO: add handling of Series with Float64 values here
    elif isinstance(val, pd.DataFrame) and any(
        [
            isinstance(val.iloc[:, i].dtype, pd.core.arrays.floating.FloatingDtype)
            for i in range(len(val.columns))
        ]
    ):
        col_typs = []
        for i in range(len(val.columns)):
            S = val.iloc[:, i]
            if isinstance(S.dtype, pd.core.arrays.floating.FloatingDtype):
                col_typs.append(typeof_pd_float_dtype(S.dtype))
            else:
                col_typs.append(bodo.hiframes.boxing._infer_series_arr_type(S).dtype)
        col_typs = (dtype_to_array_type(typ) for typ in col_typs)
        col_names = tuple(val.columns.to_list())
        index_typ = numba.typeof(val.index)
        return bodo.DataFrameType(col_typs, index_typ, col_names)
    elif isinstance(val, pd.Series) and isinstance(
        val.dtype, pd.core.arrays.floating.FloatingDtype
    ):
        return bodo.SeriesType(
            typeof_pd_float_dtype(val.dtype),
            index=numba.typeof(val.index),
            name_typ=numba.typeof(val.name),
        )
    elif isinstance(val, pytypes.FunctionType):
        # function type isn't accurate, but good enough for the purposes of _typeof
        return types.FunctionType(types.none())
    return bodo.typeof(val)


def typeof_pd_float_dtype(val):
    """
    Pandas 1.2.0 adds interpretting a nullable FloatArray
    This isn't supported yet in Bodo, so we convert these inputs
    to floating point values.
    """
    # Warning: This code is unstable and doesn't currently change
    # the actual underlying type. This is just used by _typeof inside
    # test utils for distirbuted testing. It should not be used in
    # actual Numba Code
    bitwidth = 8 * val.itemsize
    dtype = getattr(numba.types, "float{}".format(bitwidth))
    # TODO: Add a custom FloatingDType() inside Bodo.
    return dtype


def is_bool_object_series(S):
    if isinstance(S, pd.Series):
        S = S.values
    return (
        S.dtype == np.dtype("O")
        and bodo.hiframes.boxing._infer_ndarray_obj_dtype(S).dtype
        == numba.core.types.bool_
    )


def is_list_str_object_series(S):
    if isinstance(S, pd.Series):
        S = S.values
    return S.dtype == np.dtype("O") and bodo.hiframes.boxing._infer_ndarray_obj_dtype(
        S
    ).dtype == numba.core.types.List(numba.core.types.unicode_type)


def has_udf_call(fir):
    """returns True if function IR 'fir' has a UDF call."""
    for block in fir.blocks.values():
        for stmt in block.body:
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Global)
                and isinstance(stmt.value.value, numba.core.registry.CPUDispatcher)
            ):
                if (
                    stmt.value.value._compiler.pipeline_class
                    == bodo.compiler.BodoCompilerUDF
                ):
                    return True

    return False


class DeadcodeTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_join_deadcode_cleanup and test_csv_remove_col0_used_for_len
    with an additional PreserveIR pass then bodo_pipeline
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, NopythonRewrites)
        pipeline.finalize()
        return [pipeline]


class SeriesOptTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_series_apply_df_output with an additional PreserveIR pass
    after SeriesPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIRTypeMap, bodo.compiler.BodoSeriesPass)
        pipeline.finalize()
        return [pipeline]


class ParforTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_parfor_optimizations with an additional PreserveIR pass
    after ParforPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.ParforPass)
        pipeline.finalize()
        return [pipeline]


class ColumnDelTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_column_del_pass with an additional PreserveIRTypeMap pass
    after BodoTableColumnDelPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIRTypeMap, bodo.compiler.BodoTableColumnDelPass)
        pipeline.finalize()
        return [pipeline]


@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIRTypeMap(PreserveIR):
    """
    Extension to PreserveIR that also saves the typemap.
    """

    _name = "preserve_ir_typemap"

    def __init__(self):
        PreserveIR.__init__(self)

    def run_pass(self, state):
        PreserveIR.run_pass(self, state)
        state.metadata["preserved_typemap"] = state.typemap.copy()
        return False


class TypeInferenceTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in bodosql tests with an additional PreserveIR pass
    after BodoTypeInference. This is used to monitor the code being generated.
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.BodoTypeInference)
        pipeline.finalize()
        return [pipeline]


class DistTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline with an additional PreserveIR pass
    after DistributedPass
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.BodoDistributedPass)
        pipeline.finalize()
        return [pipeline]


class SeqTestPipeline(bodo.compiler.BodoCompiler):
    """
    Bodo sequential pipeline with an additional PreserveIR pass
    after LowerBodoIRExtSeq
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=False, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, bodo.compiler.LowerBodoIRExtSeq)
        pipeline.finalize()
        return [pipeline]


@register_pass(analysis_only=False, mutates_CFG=True)
class ArrayAnalysisPass(FunctionPass):
    _name = "array_analysis_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        array_analysis = numba.parfors.array_analysis.ArrayAnalysis(
            state.typingctx,
            state.func_ir,
            state.typemap,
            state.calltypes,
        )
        array_analysis.run(state.func_ir.blocks)
        state.func_ir._definitions = numba.core.ir_utils.build_definitions(
            state.func_ir.blocks
        )
        state.metadata["preserved_array_analysis"] = array_analysis
        return False


class AnalysisTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeline used in test_dataframe_array_analysis()
    additional ArrayAnalysis pass that preserves analysis object
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(ArrayAnalysisPass, bodo.compiler.BodoSeriesPass)
        pipeline.finalize()
        return [pipeline]


def check_timing_func(func, args):
    """Function for computing runtimes. First run is to get the code compiled and second
    run is to recompute with the compiled code"""
    bodo_func = bodo.jit(func)
    the_res1 = bodo_func(*args)
    t1 = time.time()
    the_res2 = bodo_func(*args)
    t2 = time.time()
    delta_t = round(t2 - t1, 4)
    print("Time:", delta_t, end=" ")
    assert True


def string_list_ent(x):
    if isinstance(x, (int, np.int64, float)):
        return str(x)
    if isinstance(x, dict):
        l_str = []
        for k in x:
            estr = '"' + str(k) + '": ' + string_list_ent(x[k])
            l_str.append(estr)
        return "{" + ", ".join(l_str) + "}"
    if isinstance(
        x, (list, np.ndarray, pd.arrays.IntegerArray, pd.arrays.ArrowStringArray)
    ):
        l_str = []
        for e_val in x:
            l_str.append(string_list_ent(e_val))
        return "[" + ",".join(l_str) + "]"
    if pd.isna(x):
        return "nan"
    if isinstance(x, str):
        return x
    if isinstance(x, Decimal):
        e_s = str(x)
        if e_s.find(".") != -1:
            f_s = e_s.strip("0").strip(".")
            return f_s
        return e_s
    print("Failed to find matching type")
    assert False


# The functionality below exist because in the case of column having
# a list-string or Decimal as data type, several functionalities are missing from pandas:
# For pandas and list of strings:
# ---sort_values
# ---groupby/join/drop_duplicates
# ---hashing
# For pandas and list of decimals:
# ---mean
# ---median
# ---var/std
#
# The solution:
# ---to transform columns of list string into columns of strings and therefore
#    amenable to sort_values.
# ---to transform columns of decimals into columns of floats
#
# Note: We cannot use df_copy["e_col_name"].str.join(',') because of unahashable list.
def convert_non_pandas_columns(df):
    if not isinstance(df, pd.DataFrame):
        return df

    df_copy = df.copy()
    # Manually invalidate the cached typing information.
    df_copy._bodo_meta = None
    list_col = df.columns.to_list()
    n_rows = df_copy[list_col[0]].size
    # Determine which columns have list of strings in them
    # Determine which columns have Decimals in them.
    col_names_list_string = []
    col_names_array_item = []
    col_names_arrow_array_item = []
    col_names_decimal = []
    for e_col_name in list_col:
        e_col = df[e_col_name]
        nb_list_string = 0
        nb_array_item = 0
        nb_arrow_array_item = 0
        nb_decimal = 0
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(
                e_ent,
                (
                    list,
                    np.ndarray,
                    pd.arrays.BooleanArray,
                    pd.arrays.IntegerArray,
                    pd.arrays.StringArray,
                    pd.arrays.ArrowStringArray,
                ),
            ):
                if len(e_ent) > 0:
                    if isinstance(e_ent[0], str):
                        nb_list_string += 1
                    if isinstance(
                        e_ent[0],
                        (int, float, np.int32, np.int64, np.float32, np.float64),
                    ):
                        nb_array_item += 1
                    for e_val in e_ent:
                        if isinstance(
                            e_val, (list, dict, np.ndarray, pd.arrays.IntegerArray)
                        ):
                            nb_arrow_array_item += 1
            if isinstance(e_ent, dict):
                nb_arrow_array_item += 1
            if isinstance(e_ent, Decimal):
                nb_decimal += 1
        if nb_list_string > 0:
            col_names_list_string.append(e_col_name)
        if nb_array_item > 0:
            col_names_array_item.append(e_col_name)
        if nb_arrow_array_item > 0:
            col_names_arrow_array_item.append(e_col_name)
        if nb_decimal > 0:
            col_names_decimal.append(e_col_name)
    for e_col_name in col_names_list_string:
        e_list_str = []
        e_col = df[e_col_name]

        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(
                e_ent,
                (list, np.ndarray, pd.arrays.StringArray, pd.arrays.ArrowStringArray),
            ):
                f_ent = [x if not pd.isna(x) else "None" for x in e_ent]
                e_str = ",".join(f_ent) + ","
                e_list_str.append(e_str)
            else:
                e_list_str.append(np.nan)
        df_copy[e_col_name] = e_list_str
    for e_col_name in col_names_array_item:
        e_list_str = []
        e_col = df[e_col_name]
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(
                e_ent,
                (
                    list,
                    np.ndarray,
                    pd.arrays.BooleanArray,
                    pd.arrays.IntegerArray,
                    pd.arrays.StringArray,
                    pd.arrays.ArrowStringArray,
                ),
            ):
                e_str = ",".join([str(x) for x in e_ent]) + ","
                e_list_str.append(e_str)
            else:
                e_list_str.append(np.nan)
        df_copy[e_col_name] = e_list_str
    for e_col_name in col_names_arrow_array_item:
        e_list_str = []
        e_col = df[e_col_name]
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            f_ent = string_list_ent(e_ent)
            e_list_str.append(f_ent)
        df_copy[e_col_name] = e_list_str
    for e_col_name in col_names_decimal:
        e_list_float = []
        e_col = df[e_col_name]
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(e_ent, Decimal):
                e_list_float.append(float(e_ent))
            else:
                e_list_float.append(np.nan)
        df_copy[e_col_name] = e_list_float
    return df_copy


# Mapping Between Numpy Dtype and Equivalent Pandas Extension Dtype
# Bodo returns the Pandas Dtype while other implementations like Pandas and Spark
# return the Numpy equivalent. Need to convert during testing.
np_to_pd_dtype: Dict[np.dtype, pd.api.extensions.ExtensionDtype] = {
    np.dtype(np.int8): pd.Int8Dtype,
    np.dtype(np.int16): pd.Int16Dtype,
    np.dtype(np.int32): pd.Int32Dtype,
    np.dtype(np.int64): pd.Int64Dtype,
    np.dtype(np.uint8): pd.UInt8Dtype,
    np.dtype(np.uint16): pd.UInt16Dtype,
    np.dtype(np.uint32): pd.UInt32Dtype,
    np.dtype(np.uint64): pd.UInt64Dtype,
    np.dtype(np.string_): pd.StringDtype,
    np.dtype(np.bool_): pd.BooleanDtype,
}


def sync_dtypes(py_out, bodo_out_dtypes):
    py_out_dtypes = py_out.dtypes.values.tolist()

    if any(
        isinstance(dtype, pd.api.extensions.ExtensionDtype) for dtype in bodo_out_dtypes
    ):
        for i, (py_dtype, bodo_dtype) in enumerate(zip(py_out_dtypes, bodo_out_dtypes)):

            if isinstance(bodo_dtype, np.dtype) and py_dtype == bodo_dtype:
                continue
            if (
                isinstance(py_dtype, pd.api.extensions.ExtensionDtype)
                and py_dtype == bodo_dtype
            ):
                continue
            elif (
                isinstance(bodo_dtype, pd.api.extensions.ExtensionDtype)
                and isinstance(py_dtype, np.dtype)
                and py_dtype in np_to_pd_dtype
                and np_to_pd_dtype[py_dtype]() == bodo_dtype
            ):
                py_out[py_out.columns[i]] = py_out[py_out.columns[i]].astype(bodo_dtype)
    return py_out


# This function allows to check the coherency of parallel output.
# That is, do we get the same result on one or on two or more MPI processes?
#
# No need for fancy stuff here. All output is unsorted. When we have the
# sort_values working for list_string column the use of conversion with col_names
# would be removed.
#
# Were the functionality of the list-string supported in pandas, we would not need
# any such function.
def check_parallel_coherency(
    func,
    args,
    sort_output=False,
    reset_index=False,
    additional_compiler_arguments=None,
):
    n_pes = bodo.get_size()

    # Computing the output in serial mode
    copy_input = True
    call_args_serial = tuple(_get_arg(a, copy_input) for a in args)
    kwargs = {"all_args_distributed_block": False, "all_returns_distributed": False}
    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)
    bodo_func_serial = bodo.jit(func, **kwargs)
    serial_output_raw = bodo_func_serial(*call_args_serial)
    serial_output_final = convert_non_pandas_columns(serial_output_raw)

    # If running on just one processor, nothing more is needed.
    if n_pes == 1:
        return

    # Computing the parallel input and output.
    kwargs = {"all_args_distributed_block": True, "all_returns_distributed": True}
    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)
    bodo_func_parall = bodo.jit(func, **kwargs)
    call_args_parall = tuple(
        _get_dist_arg(a, copy=True, var_length=False) for a in args
    )

    parall_output_raw = bodo_func_parall(*call_args_parall)
    parall_output_proc = convert_non_pandas_columns(parall_output_raw)
    # Collating the parallel output on just one processor.
    _check_typing_issues(parall_output_proc)
    parall_output_final = bodo.gatherv(parall_output_proc)

    # Doing the sorting. Mandatory here
    if sort_output:
        serial_output_final = sort_dataframe_values_index(serial_output_final)
        parall_output_final = sort_dataframe_values_index(parall_output_final)

    # reset_index if asked.
    if reset_index:
        serial_output_final.reset_index(drop=True, inplace=True)
        parall_output_final.reset_index(drop=True, inplace=True)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            pd.testing.assert_frame_equal(
                serial_output_final,
                parall_output_final,
                check_dtype=False,
                check_column_type=False,
            )
        except Exception as e:
            print(e)
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes, "Parallel test failed"


def gen_random_arrow_array_struct_int(span, n):
    e_list = []
    for _ in range(n):
        valA = random.randint(0, span)
        valB = random.randint(0, span)
        e_ent = {"A": valA, "B": valB}
        e_list.append(e_ent)
    return e_list


def gen_random_arrow_array_struct_list_int(span, n):
    e_list = []
    for _ in range(n):
        # We cannot allow empty block because if the first one is such type
        # gets found to be wrong.
        valA = [random.randint(0, span) for _ in range(random.randint(1, 5))]
        valB = [random.randint(0, span) for _ in range(random.randint(1, 5))]
        e_ent = {"A": valA, "B": valB}
        e_list.append(e_ent)
    return e_list


def gen_random_arrow_list_list_decimal(rec_lev, prob_none, n):
    def random_list_rec(rec_lev):
        if random.random() < prob_none:
            return None
        else:
            if rec_lev == 0:
                return Decimal(
                    str(random.randint(1, 10)) + "." + str(random.randint(1, 10))
                )
            else:
                return [
                    random_list_rec(rec_lev - 1) for _ in range(random.randint(1, 3))
                ]

    return [random_list_rec(rec_lev) for _ in range(n)]


def gen_random_arrow_list_list_int(rec_lev, prob_none, n):
    def random_list_rec(rec_lev):
        if random.random() < prob_none:
            return None
        else:
            if rec_lev == 0:
                return random.randint(0, 10)
            else:
                return [
                    random_list_rec(rec_lev - 1) for _ in range(random.randint(1, 3))
                ]

    return [random_list_rec(rec_lev) for _ in range(n)]


def gen_random_arrow_list_list_double(rec_lev, prob_none, n):
    def random_list_rec(rec_lev):
        if random.random() < prob_none:
            return None
        else:
            if rec_lev == 0:
                return 0.4 + random.randint(0, 10)
            else:
                return [
                    random_list_rec(rec_lev - 1) for _ in range(random.randint(1, 3))
                ]

    return [random_list_rec(rec_lev) for _ in range(n)]


def gen_random_arrow_struct_struct(span, n):
    e_list = []
    for _ in range(n):
        valA1 = random.randint(0, span)
        valA2 = random.randint(0, span)
        valB1 = random.randint(0, span)
        valB2 = random.randint(0, span)
        e_ent = {"A": {"A1": valA1, "A2": valA2}, "B": {"B1": valB1, "B2": valB2}}
        e_list.append(e_ent)
    return e_list


def gen_nonascii_list(num_strings):
    """
    Generate list of num_strings number of non-ASCII strings
    Non-ASCII Reference: https://rbutterworth.nfshost.com/Tables/compose/
    """

    list_non_ascii_strings = [
        "À È Ì",
        "Á É Í",
        "Â Ê Î",
        "Ã Ẽ Ĩ",
        "Ä Ë Ï",
        "Å Ů ẘ ẙ",
        "Ā Ē Ī",
        "Ă Ĕ Ĭ",
        "Ą Ę Į",
        "Ǎ Ě Ǐ",
        "Ɨ Ø ƀ",
        "Ȩ Ç Ḑ",
        "Ő Ű",
        "ǖ ǘ ǚ ǜ",
        "Æ æ",
        "Œ œ",
        "Ð ð",
        "Þ þ",
        "Ŋ ŋ",
        "ẞ ß",
        "ſ ſ",
        "İ ı",
        "ĸ",
        "ə",
        "ʻ",
        "♩ ♪ ♫ ♬ ♭ ♮ ♯",
        "ⁱ ⁿ ª º",
        "ʰ ʲ ˡ ʳ ˢ ʷ ˣ ʸ",
        "⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹",
        "⁽ ⁺ ⁼ ⁾",
        "₍ ₊ ₌ ₎",
        "₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉",
        "½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞",
        "± × ÷ √",
        "≠ ≤ ≥ ≡",
        "← → ⇒",
        "∴ ∵",
        "¦ ¬ ⋄",
        "° ∞ ‰",
        "µ ∅",
        "« »",
        "‹ ›",
        "“ ”",
        "‘ ’",
        "‚ „",
        "〞〝",
        "¡",
        "¿",
        "§",
        "¶",
        "…",
        "·",
        "¸",
        "–",
        "—",
        "©",
        "®",
        "℠",
        "™",
        "₥",
        "¢",
        "¤",
        "€",
        "£",
        "₡",
        "¥",
        "₦",
        "₨",
        "₩",
        "฿",
        "₫",
        "₠",
        "₣",
        "₤",
        "₧",
        "₢",
        "∞",
    ]

    return list_non_ascii_strings[0:num_strings]


def gen_random_list_string_array(option, n):
    """Generate a random array of list(string)
    option=1 for series with nullable values
    option=2 for series without nullable entries.
    """

    def rand_col_str(n):
        e_ent = []
        for _ in range(n):
            k = random.randint(1, 3)
            val = "".join(random.choices(["À", "B", "C"], k=k))
            e_ent.append(val)
        return e_ent

    def rand_col_l_str(n):
        e_list = []
        for _ in range(n):
            if random.random() < 0.1:
                e_ent = np.nan
            else:
                e_ent = rand_col_str(random.randint(1, 3))
            e_list.append(e_ent)
        return e_list

    def rand_col_l_str_none_no_first(n):
        e_list_list = []
        for _ in range(n):
            e_list = []
            for idx in range(random.randint(1, 4)):
                # The None on the first index creates some problems.
                if random.random() < 0.1 and idx > 0:
                    val = None
                else:
                    k = random.randint(1, 3)
                    val = "".join(random.choices(["À", "B", "C"], k=k))
                e_list.append(val)
            e_list_list.append(e_list)
        return e_list_list

    if option == 1:
        e_list = rand_col_l_str(n)
    if option == 2:
        e_list = rand_col_str(n)
    if option == 3:
        e_list = rand_col_l_str_none_no_first(n)
    return e_list


def gen_random_decimal_array(option, n):
    """Compute a random decimal series for tests
    option=1 will give random arrays with collision happening (guaranteed for n>100)
    option=2 will give random arrays with collision unlikely to happen
    """

    def random_str1():
        e_str1 = str(1 + random.randint(1, 8))
        e_str2 = str(1 + random.randint(1, 8))
        return Decimal(e_str1 + "." + e_str2)

    def random_str2():
        klen1 = random.randint(1, 7)
        klen2 = random.randint(1, 7)
        e_str1 = "".join([str(1 + random.randint(1, 8)) for _ in range(klen1)])
        e_str2 = "".join([str(1 + random.randint(1, 8)) for _ in range(klen2)])
        esign = "" if random.randint(1, 2) == 1 else "-"
        return Decimal(esign + e_str1 + "." + e_str2)

    if option == 1:
        e_list = [random_str1() for _ in range(n)]
    if option == 2:
        e_list = [random_str2() for _ in range(n)]
    return pd.Series(e_list)


def gen_random_string_binary_array(n, max_str_len=10, is_binary=False):
    """
    helper function that generates a random string array
    """
    random.seed(0)
    str_vals = []
    for _ in range(n):
        # store NA with 30% chance
        if random.random() < 0.3:
            str_vals.append(np.nan)
            continue

        k = random.randint(1, max_str_len)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=k))
        if is_binary:
            val = val.encode("utf-8")
        str_vals.append(val)

    # use consistent string array type with Bodo to avoid output comparison errors
    if not is_binary and bodo.libs.str_arr_ext.use_pd_string_array:
        return pd.array(str_vals, "string")
    return np.array(str_vals, dtype="object")  # avoid unichr dtype (TODO: support?)


def _check_typing_issues(val):
    """Raises an error if there is a typing issue for value 'val'.
    Runs bodo typing on value and converts warnings to errors.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            bodo.typeof(val)
        errors = comm.allgather(None)
    except Exception as e:
        # The typing issue typically occurs on only a subset of processes,
        # because the process got a chunk of data from Python that is empty
        # or that we cannot currently type correctly.
        # To avoid a hang, we need to notify every rank of the error.
        comm.allgather(e)
        raise
    for e in errors:
        if isinstance(e, Exception):
            raise e


def check_caching(
    impl,
    args,
    is_cached,
    input_dist,
    check_names=False,
    check_dtype=False,
    sort_output=False,
    copy_input=False,
    atol=1e-08,
    rtol=1e-05,
    reset_index=False,
    convert_columns_to_pandas=False,
    check_categorical=False,
    set_columns_name_to_none=False,
    reorder_columns=False,
    py_output=None,
    is_out_dist=True,
    args_already_distributed=False,
):
    """Test caching by compiling a BodoSQL function with
    cache=True, then running it again loading from cache.

    This function also tests correctness for the specified input distribution.

    impl: the function to compile
    args: arguments to pass to the function
    is_cached: true if we expect the function to already be cached, false if we do not.
    input_dist: The InputDist for the dataframe argumennts. This is used
        in the flags for compiling the function.
    """
    if py_output is None:
        py_output = impl(*args)

    # compile impl in the correct dist
    if not args_already_distributed and (
        input_dist == InputDist.OneD or input_dist == InputDist.OneDVar
    ):
        args = tuple(
            _get_dist_arg(
                a,
                copy=copy_input,
                var_length=(InputDist.OneDVar == input_dist),
                check_typing_issues=True,
            )
            for a in args
        )

    all_args_distributed_block = input_dist == InputDist.OneD
    all_args_distributed_varlength = input_dist == InputDist.OneDVar
    all_returns_distributed = input_dist != InputDist.REP and is_out_dist
    returns_maybe_distributed = input_dist != InputDist.REP and is_out_dist
    args_maybe_distributed = input_dist != InputDist.REP
    bodo_func = bodo.jit(
        cache=True,
        all_args_distributed_block=all_args_distributed_block,
        all_args_distributed_varlength=all_args_distributed_varlength,
        all_returns_distributed=all_returns_distributed,
        returns_maybe_distributed=returns_maybe_distributed,
        args_maybe_distributed=args_maybe_distributed,
    )(impl)

    # Add a barrier to reduce the odds of possible race condition
    # between ranks getting a cached implementation.
    bodo.barrier()
    bodo_output = bodo_func(*args)

    # correctness check, copied from the various check_func's
    if convert_columns_to_pandas:
        bodo_output = convert_non_pandas_columns(bodo_output)
    if returns_maybe_distributed:
        bodo_output = _gather_output(bodo_output)
    if set_columns_name_to_none:
        bodo_output.columns.name = None
    if reorder_columns:
        bodo_output.sort_index(axis=1, inplace=True)
    passed = 1
    if not returns_maybe_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output,
            py_output,
            sort_output,
            check_names,
            check_dtype,
            reset_index,
            check_categorical,
            atol,
            rtol,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size()

    bodo.barrier()

    # get signature of compiled function
    sig = bodo_func.signatures[0]

    if is_cached:
        # assert that it was loaded from cache
        assert bodo_func._cache_hits[sig] == 1
        assert bodo_func._cache_misses[sig] == 0
    else:
        # assert that it wasn't loaded from cache
        assert bodo_func._cache_hits[sig] == 0
        assert bodo_func._cache_misses[sig] == 1

    return bodo_output


def _check_for_io_reader_filters(bodo_func, node_class):
    """make sure a Connector node has filters set, and the filtering code in the IR
    is removed
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    read_found = False
    for stmt in fir.blocks[0].body:
        if isinstance(stmt, node_class):
            assert stmt.filters is not None
            read_found = True
        # filtering code has getitem which should be removed
        assert not (is_assign(stmt) and is_expr(stmt.value, "getitem"))

    assert read_found


def _ensure_func_calls_optimized_out(bodo_func, call_names):
    """
    Ensures the bodo_func doesn't contain any calls to functions
    that should be optimzed out.

    Note: Each call_name should be a tuple that matches the output of
    find_callname
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    typemap = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_typemap"]
    for _, block in fir.blocks.items():
        for stmt in block.body:
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "call"
            ):
                call_name = guard(find_callname, fir, stmt.value, typemap)
                assert (
                    call_name not in call_names
                ), f"{call_name} found in IR when it should be optimized out"


# We only run snowflake tests on Azure Pipelines because the Snowflake account credentials
# are stored there (to avoid failing on AWS or our local machines)
def get_snowflake_connection_string(db, schema, user=1):
    """
    Generates a common snowflake connection string. Some details (how to determine
    username and password) seem unlikely to change, whereas as some tests could require
    other details (db and schema) to change.
    """
    if user == 1:
        username = os.environ["SF_USER"]
        password = os.environ["SF_PASSWORD"]
    elif user == 2:
        username = os.environ["SF_USER2"]
        password = os.environ["SF_PASSWORD2"]
    else:
        raise ValueError("Invalid user")
    account = "bodopartner.us-east-1"
    warehouse = "DEMO_WH"
    conn = f"snowflake://{username}:{password}@{account}/{db}/{schema}?warehouse={warehouse}"
    return conn


@contextmanager
def create_snowflake_table(
    df: pd.DataFrame, base_table_name: str, db: str, schema: str
) -> str:
    """Creates a new table in Snowflake derived from the base table name
    and using the DataFrame. The name from the base name is modified to help
    reduce the likelihood of conflicts during concurrent tests.

    Returns the name of the table added to Snowflake.

    Args:
        df (pd.DataFrame): DataFrame to insert
        base_table_name (str): Base string for generating the table name.
        db (str): Name of the snowflake db.
        schema (str): Name of the snowflake schema

    Returns:
        str: The final table name.
    """
    try:
        comm = MPI.COMM_WORLD
        table_name = None
        if bodo.get_rank() == 0:
            unique_name = str(uuid4()).replace("-", "_")
            table_name = f"{base_table_name}_{unique_name}".lower()
            conn_str = get_snowflake_connection_string(db, schema)
            df.to_sql(
                table_name, conn_str, schema=schema, index=False, if_exists="replace"
            )
        table_name = comm.bcast(table_name)
        yield table_name
    finally:
        drop_snowflake_table(table_name, db, schema)


def drop_snowflake_table(table_name: str, db: str, schema: str):
    """Drops a table from snowflake with the given table_name.
    The db and schema are also provided to connect to Snowflake.

    Args:
        table_name (str): Table Name inside Snowflake.
        db (str): Snowflake database name
        schema (str): Snowflake schema name.
    """
    comm = MPI.COMM_WORLD
    drop_err = None
    if bodo.get_rank() == 0:
        try:
            conn_str = get_snowflake_connection_string(db, schema)
            pd.read_sql(f"drop table {table_name}", conn_str)
        except Exception as e:
            drop_err = e
    drop_err = comm.bcast(drop_err)
    if isinstance(drop_err, Exception):
        raise drop_err
