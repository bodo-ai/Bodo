# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Utility functions for testing such as check_func() that tests a function.
"""
import datetime
import io
import os
import random
import re
import string
import subprocess
import time
import types as pytypes
import warnings
from contextlib import contextmanager
from decimal import Decimal
from enum import Enum
from typing import Callable, Generator, Optional, TypeVar, Union
from urllib.parse import urlencode
from uuid import uuid4

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI
from numba.core import ir, types
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import build_definitions, find_callname, guard
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


def dist_IR_count(f_ir, func_name):
    """count how many times the string 'func_name' is in function IR 'f_ir'"""
    with io.StringIO() as str_io:
        f_ir.dump(str_io)
        f_ir_text = str_io.getvalue()
    return f_ir_text.count(func_name)


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
    distributed: Union[list[tuple[str, int]], bool, None] = None,
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
    only_1DVar=None,
    check_categorical=False,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    use_table_format=True,
    use_dict_encoded_strings=None,
    use_map_arrays: bool = False,
    convert_to_nullable_float=True,
) -> dict[str, Callable]:
    """test bodo compilation of function 'func' on arguments using REP, 1D, and 1D_Var
    inputs/outputs

    Rationale of the functionalities:
    - is_out_distributed: By default to None and is adjusted according to the REP/1D/1D_Var
    If the is_out_distributed is selected then it is hardcoded here.
    - distributed: Arguments expected to be distributed input. Default to None where all
    arguments are allowed to be distributed (for 1D and 1D Var tests)
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
    floating point precision can vary due to differences in underlying floating point libraries.
    - rtol: Argument to pass to Pandas assert equals functions. This argument will be used if
    floating point precision can vary due to differences in underlying floating point libraries.
    - use_table_format: flag for loading dataframes in table format for testing.
    If None, tests both formats if input arguments have dataframes.
    - use_dict_encoded_strings: flag for loading string arrays in dictionary-encoded
    format for testing.
    - use_map_arrays: Flag for forcing all input dict-object arrays to be unboxed as map arrays
    If None, tests both formats if input arguments have string arrays.
    - check_typing_issues: raise an error if there is a typing issue for input args.
    Runs bodo typing on arguments and converts warnings to errors.
    - convert_to_nullable_float: convert float inputs to nullable float if the global
    nullable float flag is on.
    """

    # We allow the environment flag BODO_TESTING_ONLY_RUN_1D_VAR to change the default
    # testing behavior, to test with only 1D_var. This environment variable is set in our
    # AWS PR CI environment
    if only_1DVar is None and not (only_seq or only_1D):
        only_1DVar = os.environ.get("BODO_TESTING_ONLY_RUN_1D_VAR", None) is not None

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
    # Ideally we would like to also restrict running parallel tests when we have a single rank,
    # but this can lead to test coverage issues when running with BODO_TESTING_ONLY_RUN_1D_VAR
    # on AWS PR CI, where we only run with just a single rank
    if (
        n_pes > 1
        and not numba.core.config.DEVELOPER_MODE
        and is_out_distributed is not False
        and distributed is not None
    ):
        run_seq = False

    # Avoid testing 1D on CI to run faster. It's not likely to fail independent of
    # 1D_Var.
    if not only_1D and not numba.core.config.DEVELOPER_MODE:
        run_1D = False

    # convert float input to nullable float to test new nullable float functionality
    if convert_to_nullable_float:
        args = _convert_float_to_nullable_float(args)
        py_output = _convert_float_to_nullable_float(py_output)

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

    # List of Output Bodo Functions
    bodo_funcs: dict[str, Callable] = {}

    saved_TABLE_FORMAT_THRESHOLD = bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    saved_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    saved_struct_size_limit = bodo.hiframes.boxing.struct_size_limit
    try:
        # test table format for dataframes (non-table format tested below if flag is
        # None)
        if use_table_format is None or use_table_format:
            bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = 0

        # test dict-encoded string arrays if flag is set (dict-encoded tested below if
        # flag is None)
        if use_dict_encoded_strings:
            bodo.hiframes.boxing._use_dict_str_type = True

        # Test all dict-like arguments as map arrays (no structs) if flag is set
        if use_map_arrays:
            bodo.hiframes.boxing.struct_size_limit = -1

        # sequential
        if run_seq:
            bodo_func, w = check_func_seq(
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
            bodo_funcs["seq"] = bodo_func

            # test string arguments as StringLiteral type also (since StringLiteral is
            # not a subtype of UnicodeType)
            if any(isinstance(a, str) for a in args):
                bodo_func, _ = check_func_seq(
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
                bodo_funcs["seq-strlit"] = bodo_func

        # distributed test is not needed
        if not dist_test:
            return bodo_funcs

        if is_out_distributed is None and py_output is not pd.NA:
            # assume all distributable output is distributed if not specified
            py_out_typ = _typeof(py_output)
            is_out_distributed = is_distributable_typ(
                py_out_typ
            ) or is_distributable_tuple_typ(py_out_typ)

        # skip 1D distributed and 1D distributed variable length tests
        # if no parallelism is found
        # and if neither inputs nor outputs are distributable
        if (
            w is not None  # if no parallelism is found
            and not is_out_distributed  # if output is not distributable
            and not distributed  # If some inputs are required to be distributable
            and not any(
                is_distributable_typ(_typeof(a))
                or is_distributable_tuple_typ(_typeof(a))
                for a in args
            )  # if none of the inputs is distributable
        ):
            return bodo_funcs  # no need for distributed checks

        if run_1D:
            bodo_func = check_func_1D(
                func,
                args,
                py_output,
                is_out_distributed,
                distributed,
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
            bodo_funcs["1D"] = bodo_func

        if run_1DVar:
            bodo_func = check_func_1D_var(
                func,
                args,
                py_output,
                is_out_distributed,
                distributed,
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
            bodo_funcs["1D_var"] = bodo_func

    finally:
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = saved_TABLE_FORMAT_THRESHOLD
        bodo.hiframes.boxing._use_dict_str_type = saved_use_dict_str_type
        bodo.hiframes.boxing.struct_size_limit = saved_struct_size_limit

    # test non-table format case if there is any dataframe in input
    if use_table_format is None and any(
        isinstance(_typeof(a), bodo.DataFrameType) for a in args
    ):
        inner_funcs = check_func(
            func,
            args,
            is_out_distributed,
            distributed,
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
            use_map_arrays=use_map_arrays,
            use_table_format=False,
            use_dict_encoded_strings=use_dict_encoded_strings,
            convert_to_nullable_float=convert_to_nullable_float,
        )
        bodo_funcs.update(
            {f"table-format-{name}": func for name, func in inner_funcs.items()}
        )

    # test dict-encoded string type if there is any string array in input
    if use_dict_encoded_strings is None and any(
        _type_has_str_array(_typeof(a)) for a in args
    ):
        inner_funcs = check_func(
            func,
            args,
            is_out_distributed,
            distributed,
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
            convert_to_nullable_float=convert_to_nullable_float,
            use_map_arrays=use_map_arrays,
        )
        bodo_funcs.update(
            {f"dict-encoding-{name}": func for name, func in inner_funcs.items()}
        )

    return bodo_funcs


def _convert_float_to_nullable_float(arg):
    """Convert float array/Series/Index/DataFrame to nullable float"""
    # tuple
    if isinstance(arg, tuple):
        return tuple(_convert_float_to_nullable_float(a) for a in arg)

    # Numpy float array
    if (
        isinstance(arg, np.ndarray)
        and arg.dtype in (np.float32, np.float64)
        and arg.ndim == 1
    ):
        return pd.array(arg)

    # Series/Index with float data
    if isinstance(arg, (pd.Series, pd.Index)) and arg.dtype in (np.float32, np.float64):
        return arg.astype("Float32" if arg.dtype == np.float32 else "Float64")

    # DataFrame float columns
    if isinstance(arg, pd.DataFrame) and any(
        a in (np.float32, np.float64) for a in arg.dtypes
    ):
        return pd.DataFrame(
            {c: _convert_float_to_nullable_float(arg[c]) for c in arg.columns}
        )

    return arg


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
) -> tuple[Callable, list[warnings.WarningMessage]]:
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
        # Avoid PyArrow failure in sorting dict-encoded string arrays
        if bodo_out.columns.dtype == pd.StringDtype("pyarrow"):
            bodo_out.columns = bodo_out.columns.astype(object)
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
    return bodo_func, w


def check_func_1D(
    func,
    args,
    py_output,
    is_out_distributed,
    distributed: Union[list[tuple[str, int]], bool, None],
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
) -> Callable:
    """Check function output against Python while setting the inputs/outputs as
    1D distributed
    """
    kwargs = {}
    if distributed is None:
        kwargs["all_returns_distributed"] = is_out_distributed
        kwargs["all_args_distributed_block"] = True
        dist_map = [True] * len(args)
    elif isinstance(distributed, bool):
        kwargs["distributed"] = distributed
        dist_map = [distributed] * len(args)
    else:
        kwargs["distributed"] = [a for a, _ in distributed]
        args_to_dist = {i for _, i in distributed}
        dist_map = [i in args_to_dist for i in range(len(args))]

    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)

    dist_args = tuple(
        _get_dist_arg(a, copy_input, False, check_typing_issues) if to_dist else a
        for a, to_dist in zip(args, dist_map)
    )

    bodo_func = bodo.jit(func, **kwargs)
    bodo_output = bodo_func(*dist_args)
    # NOTE: We need to gather the output before converting to pandas.
    # The rationale behind this:
    # Suppose we have two output arrays from two ranks: [[1]] (single element nested array), [None].
    # For bodo_output, if we convert to pandas first, [[1]] will become ["[1]" (string type)], and [None] will become [NaN (float type)]. After gathering, the result will be ["[1]" (string), NaN (float)]
    # For py_output, since it's predefined by the test writer, it will convert [[1], None] to pandas i.e. ["[1]", "nan"] (both are strings), which does not equal ["[1]" (string), NaN (float)].
    # Thus, asserting bodo_output = py_output will fail, which is not what we want in this case
    if is_out_distributed:
        bodo_output = _gather_output(bodo_output)
    if convert_columns_to_pandas:
        bodo_output = convert_non_pandas_columns(bodo_output)
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
    return bodo_func


def check_func_1D_var(
    func,
    args,
    py_output,
    is_out_distributed,
    distributed: Union[list[str], bool, None],
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
) -> Callable:
    """Check function output against Python while setting the inputs/outputs as
    1D distributed variable length
    """
    kwargs = {}
    if distributed is None:
        kwargs["all_returns_distributed"] = is_out_distributed
        kwargs["all_args_distributed_varlength"] = True
        dist_map = [True] * len(args)
    elif isinstance(distributed, bool):
        kwargs["distributed"] = distributed
        dist_map = [distributed] * len(args)
    else:
        kwargs["distributed"] = [a for a, _ in distributed]
        args_to_dist = {i for _, i in distributed}
        dist_map = [i in args_to_dist for i in range(len(args))]

    if additional_compiler_arguments != None:
        kwargs.update(additional_compiler_arguments)

    dist_args = tuple(
        _get_dist_arg(a, copy_input, True, check_typing_issues) if to_dist else a
        for a, to_dist in zip(args, dist_map)
    )

    bodo_func = bodo.jit(func, **kwargs)
    bodo_output = bodo_func(*dist_args)
    # NOTE: We need to gather the output before converting to pandas.
    # See note in check_func_1D for the rationale behind this.
    if is_out_distributed:
        bodo_output = _gather_output(bodo_output)
    if convert_columns_to_pandas:
        bodo_output = convert_non_pandas_columns(bodo_output)
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
    return bodo_func


def _get_arg(a, copy=False):
    if copy and hasattr(a, "copy"):
        return a.copy()
    return a


T = TypeVar("T", pytypes.FunctionType, pd.Series, pd.DataFrame)


def _get_dist_arg(
    a: T, copy: bool = False, var_length: bool = False, check_typing_issues: bool = True
) -> T:
    """Get distributed chunk for 'a' on current rank (for input to test functions)"""
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
    # one extra element to the second processor
    if var_length and bodo.get_size() >= 2 and l > bodo.get_size():
        if bodo.get_rank() == 0:
            end -= 1
        if bodo.get_rank() == 1:
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
    if S.index.dtype == pd.StringDtype("pyarrow"):
        S.index = S.index.astype("string")
    S1 = S.sort_index()
    # pandas fails if all null integer column is sorted
    if S1.isnull().all():
        return S1
    # Replace PyArrow strings with regular strings since Arrow doesn't support sort for
    # dict(large_string)
    if S1.dtype == pd.StringDtype("pyarrow"):
        S1 = S1.astype("string")
    return S1.sort_values(kind="mergesort")


def sort_dataframe_values_index(df):
    """sort data frame based on values and index"""
    if isinstance(df.index, pd.MultiIndex):
        # rename index in case names are None
        index_names = [f"index_{i}" for i in range(len(df.index.names))]
        list_col_names = [
            c
            for i, c in enumerate(df.columns)
            if not isinstance(df.dtypes.iloc[i], pd.ArrowDtype)
        ] + index_names
        return df.rename_axis(index_names).sort_values(list_col_names, kind="mergesort")

    eName = "index123"
    list_col_names = [
        c
        for i, c in enumerate(df.columns)
        if not isinstance(df.dtypes.iloc[i], pd.ArrowDtype)
    ] + [eName]
    if None in list_col_names:
        raise RuntimeError(
            "Testing error in sort_dataframe_values_index: None in column names"
        )

    # Sort only works on hashable datatypes
    # Thus we convert (non-hashable) list-like types to (hashable) tuples
    df = df.applymap(
        lambda x: tuple(x)
        if isinstance(x, (list, np.ndarray, pd.core.arrays.ExtensionArray))
        else x
    )
    return df.rename_axis(eName).sort_values(list_col_names, kind="mergesort")


def _to_pa_array(py_out, bodo_arr_type):
    """Convert object array to Arrow array with specified Bodo type"""
    arrow_type = bodo.io.helpers._numba_to_pyarrow_type(
        bodo_arr_type, use_dict_arr=True
    )[0]
    arrow_type_no_dict = bodo.io.helpers._numba_to_pyarrow_type(bodo_arr_type)[0]
    py_out = pa.array(py_out, arrow_type_no_dict)
    if arrow_type != arrow_type_no_dict:
        py_out = bodo.libs.array.convert_arrow_arr_to_dict(py_out, arrow_type)
    return py_out


def _test_equal(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
    check_categorical=False,
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> None:
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        csr_matrix = type(None)

    # Bodo converts lists to array in array(item) array cases
    if isinstance(py_out, list) and isinstance(bodo_out, np.ndarray):
        py_out = np.array(py_out)

    if isinstance(py_out, pd.Series):
        if isinstance(bodo_out.dtype, pd.ArrowDtype) and not isinstance(
            py_out.dtype, pd.ArrowDtype
        ):
            py_out = pd.Series(
                _to_pa_array(
                    py_out.map(lambda a: None if a is np.nan else a).values,
                    bodo.typeof(bodo_out.values),
                ),
                py_out.index,
                bodo_out.dtype,
                py_out.name,
            )
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
        # still fails for some reason).
        # Note: The pd.StringDtype must be the left to ensure we pick the correct operator.
        if pd.StringDtype("pyarrow") == bodo_out.dtype:
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
    # Convert object array py_out to Pandas PyArrow array to match Bodo
    # Avoid changing string arrays to avoid regressions in current infrastructure
    elif (
        isinstance(bodo_out, pd.arrays.ArrowExtensionArray)
        and not isinstance(bodo_out, pd.arrays.ArrowStringArray)
        and not isinstance(py_out, pd.arrays.ArrowExtensionArray)
    ):
        py_out = _to_pa_array(py_out, bodo.typeof(bodo_out))
        pd.testing.assert_series_equal(
            pd.Series(py_out),
            pd.Series(bodo_out),
            check_dtype=False,
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
            if py_out.dtype in (np.float32, np.float64, np.complex128, np.complex64):
                np.testing.assert_allclose(bodo_out, py_out, atol=atol, rtol=rtol)
            elif isinstance(bodo_out, pd.arrays.ArrowStringArray):
                pd.testing.assert_extension_array_equal(
                    bodo_out, pd.array(py_out, "string[pyarrow]")
                )
            elif isinstance(bodo_out, pd.arrays.FloatingArray):
                pd.testing.assert_extension_array_equal(
                    bodo_out, pd.array(py_out, bodo_out.dtype)
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
            py_out = pd.Categorical(py_out, categories=py_out.categories.sort_values())
        if isinstance(bodo_out, pd.Categorical):
            bodo_out = pd.Categorical(
                bodo_out, categories=bodo_out.categories.sort_values()
            )
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
    elif isinstance(py_out, (float, np.floating, np.complex128, np.complex64)):
        # avoid equality check since paralellism can affect floating point operations
        np.testing.assert_allclose(py_out, bodo_out, rtol=rtol, atol=atol)
    elif isinstance(py_out, tuple):
        assert len(py_out) == len(bodo_out)
        for p, b in zip(py_out, bodo_out):
            _test_equal(
                b, p, sort_output, check_names, check_dtype, rtol=rtol, atol=atol
            )
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
    # Bodo returns np.nan instead of pd.NA for nullable float data to avoid typing
    # issues
    elif py_out is pd.NA and np.isnan(bodo_out):
        pass
    elif isinstance(py_out, pd.CategoricalDtype):
        np.testing.assert_equal(bodo_out.categories.values, py_out.categories.values)
        assert bodo_out.ordered == py_out.ordered
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
            elif isinstance(bodo_output_list[0], pd.arrays.ArrowExtensionArray):
                pd.concat([pd.Series(a) for a in bodo_output_list]).values
            else:
                bodo_output = pd.concat(bodo_output_list)

    return bodo_output


def _typeof(val):
    # Pandas returns an object array for .values or to_numpy() call on Series of
    # nullable int/float, which can't be handled in typeof. Bodo returns a
    # nullable int/float array
    # see test_series_to_numpy[numeric_series_val3] and
    # test_series_get_values[series_val4]
    if (
        isinstance(val, np.ndarray)
        and val.dtype == np.dtype("O")
        and all(
            (isinstance(a, float) and np.isnan(a)) or isinstance(a, int) for a in val
        )
    ):
        return bodo.libs.int_arr_ext.IntegerArrayType(bodo.int64)
    elif isinstance(val, pd.arrays.FloatingArray):
        return bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)
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
                col_typs.append(
                    bodo.libs.float_arr_ext.typeof_pd_float_dtype(S.dtype, None)
                )
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
            bodo.libs.float_arr_ext.typeof_pd_float_dtype(val.dtype, None),
            index=numba.typeof(val.index),
            name_typ=numba.typeof(val.name),
        )
    elif isinstance(val, pytypes.FunctionType):
        # function type isn't accurate, but good enough for the purposes of _typeof
        return types.FunctionType(types.none())
    return bodo.typeof(val)


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
        pipeline.add_pass_after(PreserveIR, bodo.compiler.ParforPreLoweringPass)
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
        state.metadata["preserved_calltypes"] = state.calltypes.copy()
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

    # Avoid copy propagation so we don't delete variables used to
    # check array analysis.
    avoid_copy_propagation = True

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
    if isinstance(
        x, (int, np.int64, float, pd.Timestamp, datetime.date, datetime.time)
    ):
        return str(x)
    if isinstance(x, dict):
        l_str = []
        for k in x:
            estr = '"' + str(k) + '": ' + string_list_ent(x[k])
            l_str.append(estr)
        return "{" + ", ".join(l_str) + "}"
    if isinstance(
        x,
        (
            list,
            np.ndarray,
            pd.arrays.IntegerArray,
            pd.arrays.FloatingArray,
            pd.arrays.ArrowStringArray,
        ),
    ):
        l_str = []
        if all(isinstance(elem, tuple) for elem in x):
            return string_list_ent({k: v for k, v in x})
        for e_val in x:
            l_str.append(string_list_ent(e_val))
        return "[" + ",".join(l_str) + "]"
    if pd.isna(x):
        return "nan"
    if isinstance(x, str):
        return x
    if isinstance(x, Decimal):
        if x == Decimal("0"):
            return "0"
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
    col_names_bytes = []
    for e_col_name in list_col:
        e_col = df[e_col_name]
        nb_list_string = 0
        nb_array_item = 0
        nb_arrow_array_item = 0
        nb_decimal = 0
        nb_bytes = 0
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(
                e_ent,
                (
                    list,
                    np.ndarray,
                    pd.arrays.BooleanArray,
                    pd.arrays.IntegerArray,
                    pd.arrays.FloatingArray,
                    pd.arrays.StringArray,
                    pd.arrays.ArrowStringArray,
                ),
            ):
                if len(e_ent) > 0:
                    if isinstance(e_ent[0], str):
                        nb_list_string += 1
                    if isinstance(
                        e_ent[0],
                        (int, float, np.integer, np.floating),
                    ):
                        nb_array_item += 1
                    for e_val in e_ent:
                        if isinstance(
                            e_val,
                            (
                                list,
                                dict,
                                tuple,
                                Decimal,
                                datetime.time,
                                np.ndarray,
                                pd.arrays.IntegerArray,
                                pd.arrays.FloatingArray,
                            ),
                        ):
                            nb_arrow_array_item += 1
                else:
                    nb_array_item += 1
            if isinstance(e_ent, dict):
                nb_arrow_array_item += 1
            if isinstance(e_ent, Decimal):
                nb_decimal += 1
            if isinstance(e_ent, bytearray):
                nb_bytes += 1
        if nb_list_string > 0:
            col_names_list_string.append(e_col_name)
        elif nb_arrow_array_item > 0:
            col_names_arrow_array_item.append(e_col_name)
        elif nb_array_item > 0:
            col_names_array_item.append(e_col_name)
        elif nb_decimal > 0:
            col_names_decimal.append(e_col_name)
        elif nb_bytes > 0:
            col_names_bytes.append(e_col_name)
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
                    pd.arrays.FloatingArray,
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
    for e_col_name in col_names_bytes:
        # Convert bytearray to bytes
        df_copy[e_col_name] = df[e_col_name].apply(
            lambda x: bytes(x) if isinstance(x, bytearray) else x
        )
    return df_copy


# Mapping Between Numpy Dtype and Equivalent Pandas Extension Dtype
# Bodo returns the Pandas Dtype while other implementations like Pandas and Spark
# return the Numpy equivalent. Need to convert during testing.
np_to_pd_dtype: dict[np.dtype, pd.api.extensions.ExtensionDtype] = {
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


def gen_random_arrow_array_struct_int(span, n, return_map=False):
    random.seed(0)
    e_list = []
    for _ in range(n):
        valA = random.randint(0, span)
        valB = random.randint(0, span)
        e_ent = {"A": valA, "B": valB}
        e_list.append(e_ent)
    dtype = pd.ArrowDtype(
        pa.struct([pa.field("A", pa.int64()), pa.field("B", pa.int64())])
    )
    if return_map:
        dtype = pd.ArrowDtype(pa.map_(pa.large_string(), pa.int64()))
    S = pd.Series(e_list, dtype=dtype)
    return S


def gen_random_arrow_array_struct_list_int(span, n, return_map=False):
    random.seed(0)
    e_list = []
    for _ in range(n):
        # We cannot allow empty block because if the first one is such type
        # gets found to be wrong.
        valA = [random.randint(0, span) for _ in range(random.randint(1, 5))]
        valB = [random.randint(0, span) for _ in range(random.randint(1, 5))]
        e_ent = {"A": valA, "B": valB}
        e_list.append(e_ent)

    dtype = pd.ArrowDtype(
        pa.struct(
            [
                pa.field("A", pa.large_list(pa.int64())),
                pa.field("B", pa.large_list(pa.int64())),
            ]
        )
    )
    if return_map:
        dtype = pd.ArrowDtype(pa.map_(pa.large_string(), pa.large_list(pa.int64())))
    S = pd.Series(e_list, dtype=dtype)
    return S


def gen_random_arrow_list_list_decimal(rec_lev, prob_none, n):
    random.seed(0)

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
    random.seed(0)

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
    random.seed(0)

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


def gen_random_arrow_struct_struct(span, n, return_map=False):
    random.seed(0)
    e_list = []
    for _ in range(n):
        valA1 = random.randint(0, span)
        valA2 = random.randint(0, span)
        valB1 = random.randint(0, span)
        valB2 = random.randint(0, span)
        e_ent = {"A": {"A1": valA1, "A2": valA2}, "B": {"B1": valB1, "B2": valB2}}
        e_list.append(e_ent)

    dtype = pd.ArrowDtype(
        pa.struct(
            [
                pa.field(
                    "A",
                    pa.struct([pa.field("A1", pa.int64()), pa.field("A2", pa.int64())]),
                ),
                pa.field(
                    "B",
                    pa.struct([pa.field("B1", pa.int64()), pa.field("B2", pa.int64())]),
                ),
            ]
        )
    )
    if return_map:
        dtype = pd.ArrowDtype(
            pa.map_(pa.large_string(), pa.map_(pa.large_string(), pa.int64()))
        )
    S = pd.Series(e_list, dtype=dtype)
    return S


def gen_random_arrow_struct_string(string_size, n):
    random.seed(0)
    return [
        {
            "A": "".join(
                random.choices(string.ascii_uppercase + string.digits, k=string_size)
            ),
            "B": "".join(
                random.choices(string.ascii_uppercase + string.digits, k=string_size)
            ),
        }
        for _ in range(n)
    ]


def gen_random_arrow_array_struct_string(string_size, num_structs, n):
    return [gen_random_arrow_struct_string(string_size, num_structs) for _ in range(n)]


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
    ] * 2

    return list_non_ascii_strings[0:num_strings]


def gen_random_list_string_array(option, n):
    """Generate a random array of list(string)
    option=1 for series with nullable values
    option=2 for series without nullable entries.
    """
    random.seed(0)

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
        e_list = pd.Series(
            rand_col_l_str(n), dtype=pd.ArrowDtype(pa.large_list(pa.large_string()))
        ).values
    if option == 2:
        e_list = rand_col_str(n)
    if option == 3:
        e_list = pd.Series(
            rand_col_l_str_none_no_first(n),
            dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
        ).values
    return e_list


def gen_random_decimal_array(option, n):
    """Compute a random decimal series for tests
    option=1 will give random arrays with collision happening (guaranteed for n>100)
    option=2 will give random arrays with collision unlikely to happen
    """
    random.seed(0)

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
    input_dist: The InputDist for the dataframe arguments. This is used
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
        assert (
            bodo_func._cache_hits[sig] == 1
        ), "Expected a cache hit for function signature"
        assert (
            bodo_func._cache_misses[sig] == 0
        ), "Expected no cache miss for function signature"
    else:
        # assert that it wasn't loaded from cache
        assert (
            bodo_func._cache_hits[sig] == 0
        ), "Expected no cache hits for function signature"
        assert (
            bodo_func._cache_misses[sig] == 1
        ), "Expected a miss for function signature"

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
def get_snowflake_connection_string(
    db: str, schema: str, conn_params: Optional[dict[str, str]] = None, user: int = 1
) -> str:
    """
    Generates a common snowflake connection string. Some details (how to determine
    username and password) seem unlikely to change, whereas as some tests could require
    other details (db and schema) to change.
    """
    if user == 1:
        username = os.environ["SF_USERNAME"]
        password = os.environ["SF_PASSWORD"]
        account = "bodopartner.us-east-1"
    elif user == 2:
        username = os.environ["SF_USER2"]
        password = os.environ["SF_PASSWORD2"]
        account = "bodopartner.us-east-1"
    elif user == 3:
        username = os.environ["SF_AZURE_USER"]
        password = os.environ["SF_AZURE_PASSWORD"]
        account = "kl02615.east-us-2.azure"
    else:
        raise ValueError("Invalid user")

    params = {"warehouse": "DEMO_WH"} if conn_params is None else conn_params
    conn = (
        f"snowflake://{username}:{password}@{account}/{db}/{schema}?{urlencode(params)}"
    )
    return conn


def snowflake_cred_env_vars_present(user: int = 1) -> bool:
    """
    Simple function to check if environment variables for the
    snowflake credentials are set or not. Goes along with
    get_snowflake_connection_string.

    Args:
        user (int, optional): Same user definition as get_snowflake_connection_string.
            Defaults to 1.

    Returns:
        bool: Whether env vars are set or not
    """
    if user == 1:
        return ("SF_USERNAME" in os.environ) and ("SF_PASSWORD" in os.environ)
    elif user == 2:
        return ("SF_USER2" in os.environ) and ("SF_PASSWORD2" in os.environ)
    elif user == 3:
        return ("SF_AZURE_USER" in os.environ) and ("SF_AZURE_PASSWORD" in os.environ)
    else:
        raise ValueError("Invalid user")


@contextmanager
def create_snowflake_table(
    df: pd.DataFrame, base_table_name: str, db: str, schema: str
) -> Generator[str, None, None]:
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
    comm = MPI.COMM_WORLD
    table_name = None
    try:
        if bodo.get_rank() == 0:
            table_name = gen_unique_table_id(base_table_name)
            conn_str = get_snowflake_connection_string(db, schema)
            df.to_sql(
                table_name, conn_str, schema=schema, index=False, if_exists="replace"
            )
        table_name = comm.bcast(table_name)
        yield table_name
    finally:
        drop_snowflake_table(table_name, db, schema)


def gen_unique_table_id(base_table_name):
    unique_name = str(uuid4()).replace("-", "_")
    return f"{base_table_name}_{unique_name}".lower()


def drop_snowflake_table(table_name: str, db: str, schema: str) -> None:
    """Drops a table from snowflake with the given table_name.
    The db and schema are also provided to connect to Snowflake.

    Args:
        table_name: Table Name inside Snowflake.
        db: Snowflake database name
        schema: Snowflake schema name.
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


@contextmanager
def create_snowflake_table_from_select_query(
    query: str, base_table_name: str, db: str, schema: str
) -> Generator[str, None, None]:
    """Creates a new table in Snowflake derived from the base table name
    and using the given select query. The name from the base name is modified to help
    reduce the likelihood of conflicts during concurrent tests.

    Returns the name of the table added to Snowflake.

    Args:
        query (str): A valid Snowfalke SQL query that will be used to create the table.
            This will be a SELECT query as we wrap the query in a create or replace table as clause.
        base_table_name (str): Base string for generating the table name.
        db (str): Name of the snowflake db.
        schema (str): Name of the snowflake schema

    Returns:
        str: The final table name.
    """
    comm = MPI.COMM_WORLD
    table_name = None
    try:
        if bodo.get_rank() == 0:
            table_name = gen_unique_table_id(base_table_name)
            conn_str = get_snowflake_connection_string(db, schema)
            pd.read_sql(f"CREATE or REPLACE TABLE {table_name} as ({query})", conn_str)
        table_name = comm.bcast(table_name)
        yield table_name
    finally:
        drop_snowflake_table(table_name, db, schema)


def generate_comparison_ops_func(op, check_na=False):
    """
    Generates a comparison function. If check_na,
    then we are being called on a scalar value because Pandas
    can't handle NA values in the array. If so, we return None
    if either input is NA.
    """
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(a, b):\n"
    if check_na:
        func_text += f"  if pd.isna(a) or pd.isna(b):\n"
        func_text += f"    return None\n"
    func_text += f"  return a {op_str} b\n"
    loc_vars = {}
    exec(func_text, {"pd": pd}, loc_vars)
    return loc_vars["test_impl"]


def find_funcname_in_annotation_ir(annotation, desired_callname):
    """Finds a function call if it exists in the blocks stored
    in the annotation. Returns the name of the LHS variable
    defining the function if it exists and the variables
    defining the arguments with which the function was called.

    Args:
        annotation (TypeAnnotation): Dispatcher annotation
        contain the blocks and typemap.

    Returns:
        tuple(Name of the LHS variable, list[Name of argument variables])

    Raises Assertion Error if the desired_callname is not found.
    """
    # Generate a dummy IR to enable find_callname
    f_ir = ir.FunctionIR(
        annotation.blocks,
        False,
        annotation.func_id,
        ir.Loc("", 0),
        {},
        0,
        [],
    )
    # Update the definitions
    f_ir._definitions = build_definitions(f_ir.blocks)
    # Iterate over the IR
    for block in f_ir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign):
                callname = guard(find_callname, f_ir, stmt.value, annotation.typemap)
                if callname == desired_callname:
                    return stmt.value.func.name, [var.name for var in stmt.value.args]

    assert False, f"Did not find function {desired_callname} in the IR"


def find_nested_dispatcher_and_args(
    dispatcher, args, func_name, return_dispatcher=True
):
    """Finds a dispatcher in the IR and the arguments with which it was
    called for the given func_name (which matches the output of find_callname).

    Note: This code assumes that if return_dispatcher=True, then the new
    dispatch must be called with the Overload infrastructure. If any other infrastructure
    is used, for example the generated_jit infrastructure, then the given
    code may not work.

    Args:
        dispatcher (Dispatch): A numba/bodo dispatcher
        args (tuple(numba.core.types.Type)): Input tuple of Numba types
        func_name (tuple[str, str]): func_name to find.
        return_dispatcher (bool): Should we find and return the dispatcher + arguments?
            This is True when we are doing this as part of a multi-step traversal.

    Returns a tuple with the dispatcher and the arguments with which it was called.
    """
    sig = types.void(*args)
    cr = dispatcher.get_compile_result(sig)
    annotation = cr.type_annotation
    var_name, arg_names = find_funcname_in_annotation_ir(annotation, func_name)
    if return_dispatcher:
        typemap = annotation.typemap
        arg_types = tuple([typemap[name] for name in arg_names])
        # Find the dispatcher in the IR
        cached_info = typemap[var_name].templates[0]._impl_cache
        return cached_info[
            (numba.core.registry.cpu_target.typing_context, arg_types, ())
        ]


def nanoseconds_to_other_time_units(val, unit_str):
    supported_time_parts = [
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ]

    assert unit_str in supported_time_parts

    if unit_str == "hour":
        return val // ((10**9) * 3600)
    elif unit_str == "minute":
        return val // ((10**9) * 60)
    elif unit_str == "second":
        return val // (10**9)
    elif unit_str == "millisecond":
        return val // (10**6)
    elif unit_str == "microsecond":
        return val // (10**3)
    else:
        return val


def compose_decos(decos):
    def composition(func):
        for deco in reversed(decos):
            func = deco(func)
        return func

    return composition


def get_files_changed():
    """
    Returns a list of any files changed.
    """
    res = subprocess.run(["git", "diff", "--name-only", "develop"], capture_output=True)
    return res.stdout.decode("utf-8").strip().split("\n")


files_changed = get_files_changed()


def check_for_compiler_file_changes():
    """
    Function that returns if any of the files critical to the compiler pipeline were
    altered. If this is True, then a larger subset of tests will be run on CI.
    """
    core_compiler_files = {
        "bodo/transforms/distributed_pass.py",
        "bodo/transforms/dataframe_pass.py",
        "bodo/transforms/series_pass.py",
        "bodo/transforms/table_column_del_pass.py",
        "bodo/transforms/typing_pass.py",
        "bodo/transforms/untyped_pass.py",
        "bodo/utils/transform.py",
        "bodo/utils/typing.py",
    }
    for filename in files_changed:
        if filename in core_compiler_files:
            return True
    return False


compiler_files_were_changed = check_for_compiler_file_changes()


# Determine if we are re-running a test due to a flaky failure.
pytest_snowflake_is_rerunning = False


# Determine wether we want to re-run a test due to a flaky failure,
# and set the pytest_snowflake_is_rerunning flag.
def _pytest_snowflake_rerun_filter(err, *args):
    str_err = str(err)
    should_rerun = (
        "HTTP 503: Service Unavailable" in str_err
        or "HTTP 504: Gateway Timeout" in str_err
        or "Could not connect to Snowflake backend after " in str_err
        or "snowflake.connector.vendored.requests.exceptions.ReadTimeout" in str_err
    )
    if should_rerun:
        global pytest_snowflake_is_rerunning
        pytest_snowflake_is_rerunning = True
    return should_rerun


# This is for use as a decorator for a single test function.
# (@pytest_mark_snowflake)
pytest_mark_snowflake = compose_decos(
    (
        pytest.mark.snowflake,
        pytest.mark.skipif(
            "AGENT_NAME" not in os.environ, reason="requires Azure Pipelines"
        ),
        pytest.mark.flaky(max_runs=3, rerun_filter=_pytest_snowflake_rerun_filter),
    )
)

# This is for marking an entire test file
# (pytestmark = pytest_snowflake)
pytest_snowflake = [
    pytest.mark.snowflake,
    pytest.mark.skipif(
        "AGENT_NAME" not in os.environ, reason="requires Azure Pipelines"
    ),
    pytest.mark.flaky(max_runs=3, rerun_filter=_pytest_snowflake_rerun_filter),
]

# Flag to ignore the mass slowing of tests unless specific files are changed
ignore_slow_unless_changed = os.environ.get("BODO_IGNORE_SLOW_UNLESS_CHANGED", False)


def pytest_slow_unless_changed(features):
    """
    Uses the slow marker unless any of the changed files match any of the regex
    corresponding to the entries in the features input, or compiler pass files
    have been changed.

    The defined feature keywords:
        - groupby: any BodoSQL files relating to aggregation
        - window: any BodoSQL files relating window functions
        - joins: any BodoSQL files relating to joins
        - codegen: any in files relating to BodoSQL codegen
        - library: any files in bodo/libs
        - io: any files in bodo/io
        - hiframes: any files in bodo/hiframes
    """
    if ignore_slow_unless_changed:
        return []
    known_features = {
        "groupby": "BodoSQL/calcite_sql/bodosql-calcite-application/src/main/.*Agg.*",
        "window": "BodoSQL/calcite_sql/bodosql-calcite-application/src/main/.*Window.*",
        "join": "BodoSQL/calcite_sql/bodosql-calcite-application/src/main/.*Join.*",
        "codegen": "(BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/.*)|(BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/ir/.*)",
        "library": "bodo/libs/.*",
        "io": "bodo/io/.*",
        "hiframes": "bodo/hiframes/.*",
    }
    if len(features) == 0:
        raise Exception(f"Invalid features: {features}")
    patterns = []
    for feature in features:
        if feature not in known_features:
            raise Exception(f"Invalid features: {features}")
        patterns.append(f"({known_features[feature]})")
    p = re.compile("|".join(patterns), re.I)
    if compiler_files_were_changed or any([p.match(f) for f in files_changed]):
        return []
    return [pytest.mark.slow]


pytest_slow_unless_codegen = pytest_slow_unless_changed(["library", "codegen"])
pytest_slow_unless_groupby = pytest_slow_unless_changed(
    ["library", "codegen", "groupby"]
)
pytest_slow_unless_window = pytest_slow_unless_changed(["library", "codegen", "window"])
pytest_slow_unless_join = pytest_slow_unless_changed(["library", "codegen", "join"])


# This is for use as a decorator for a single test function.
# (@pytest_mark_pandas)
pytest_mark_pandas = compose_decos(
    (
        pytest.mark.skipif(
            not (
                compiler_files_were_changed
                or "AGENT_NAME" in os.environ
                or os.environ.get("NUMBA_DEVELOPER_MODE", False)
            ),
            reason="only runs in Azure Pipelines unless compiler files were changed",
        ),
        pytest.mark.pandas,
    )
)

# This is for marking an entire test file
# (pytestmark = pytest_pandas)
pytest_pandas = [
    pytest.mark.skipif(
        not (
            compiler_files_were_changed
            or "AGENT_NAME" in os.environ
            or os.environ.get("NUMBA_DEVELOPER_MODE", False)
        ),
        reason="only runs in Azure Pipelines unless compiler files were changed",
    ),
    pytest.mark.pandas,
]

# This is for marking an entire test file
# (pytestmark = pytest_ml)
pytest_ml = [pytest.mark.ml]

# This is for marking an entire test file
# (pytestmark = pytest_perf_regression)
pytest_perf_regression = [
    pytest.mark.skipif(
        "BODO_RUN_REGRESSION_TESTS" not in os.environ,
        reason="only runs per regression tests manually",
    ),
    pytest.mark.perf_regression,
]


@contextmanager
def temp_env_override(env_vars: dict[str, Optional[str]]):
    """Update the current environment variables with key-value pairs provided
    in a dictionary and then restore it after.

    Args
        env_vars (dict(str, str or None)): A dictionary of environment variables to set.
            A value of None indicates a variable should be removed.
    """

    def update_env_vars(env_vars):
        old_env_vars = {}
        for k, v in env_vars.items():
            if k in os.environ:
                old_env_vars[k] = os.environ[k]
            else:
                old_env_vars[k] = None

            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v
        return old_env_vars

    old_env = {}
    try:
        old_env = update_env_vars(env_vars)
        yield
    finally:
        update_env_vars(old_env)


@contextmanager
def set_broadcast_join(broadcast: bool):
    """
    Context manager for enabling/disabling broadcast join in a test.
    If broadcasting it set the threshold to 1 TB so it will always
    broadcast.

    Args:
        broadcast (bool): Should broadcast be allowed?
    """
    old_threshold = os.environ.get("BODO_BCAST_JOIN_THRESHOLD", None)
    try:
        if broadcast:
            # Set to a very high value of 1 GB
            os.environ["BODO_BCAST_JOIN_THRESHOLD"] = "1000000000"
        else:
            os.environ["BODO_BCAST_JOIN_THRESHOLD"] = "0"
        yield
    finally:
        if old_threshold is None:
            del os.environ["BODO_BCAST_JOIN_THRESHOLD"]
        else:
            os.environ["BODO_BCAST_JOIN_THRESHOLD"] = old_threshold


def nullable_float_arr_maker(L, to_null, to_nan):
    """
    Utility function for helping test cases to generate nullable floating
    point arrays that contain both NULL and NaN. Takes in a list of numbers,
    a list of indices that should be set to NULL, a list of indices that should
    be set to NaN, and outputs the corresponding floating point array.

    For example:
    nullable_float_arr_maker(list(range(10)), [1, 5, 9], [2, 3, 7])

    Outputs the following Series:
    0     0.0
    1    <NA>
    2     NaN
    3     NaN
    4     4.0
    5    <NA>
    6     6.0
    7     NaN
    8     8.0
    9    <NA>
    dtype: Float64
    """
    S = _nullable_float_arr_maker(L, to_null, to_nan)
    # Remove the bodo metadata. It improperly assigns
    # 1D_Var to the series which interferes with the test
    # functionality. Deleting the metadata sets it back to
    # the default of REP distribution.
    del S._bodo_meta
    return S


@bodo.jit(distributed=False)
def _nullable_float_arr_maker(L, to_null, to_nan):
    n = len(L)
    data_arr = np.empty(n, np.float64)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    A = bodo.libs.float_arr_ext.init_float_array(data_arr, nulls)
    for i in range(len(L)):
        if i in to_null:
            bodo.libs.array_kernels.setna(A, i)
        elif i in to_nan:
            A[i] = np.nan
        else:
            A[i] = L[i]
    return pd.Series(A)


def run_rank0(func: Callable, bcast_result: bool = True, result_default=None):
    """
    Utility function decorator to run a function on just rank 0
    but re-raise any Exceptions safely on all ranks.
    NOTE: 'func' must be a simple python function that doesn't require
    any synchronization.
    e.g. Using a bodo.jit function might be unsafe in this situation.
    Similarly, a function that uses any MPI collective
    operation would be unsafe and could result in a hang.

    Args:
        func: Function to run.
        bcast_result (bool, optional): Whether the function should be
            broadcasted to all ranks. Defaults to True.
        result_default (optional): Default for result. This is only
            useful in the bcase_result=False case. Defaults to None.
    """

    def inner(*args, **kwargs):
        comm = MPI.COMM_WORLD
        result = result_default
        err = None
        # Run on rank 0 and catch any exceptions.
        if comm.Get_rank() == 0:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                err = e
        # Synchronize and re-raise any exception on all ranks.
        err = comm.bcast(err)
        if isinstance(err, Exception):
            raise err
        # Broadcast the result to all ranks.
        if bcast_result:
            result = comm.bcast(result)
        return result

    return inner


def cast_dt64_to_ns(df):
    """Cast datetime64 to datetime64[ns] to match Bodo since Pandas 2 reads some
    Parquet files as datetime64[us/ms]
    """
    from pandas.api.types import is_datetime64_any_dtype

    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            if isinstance(df[c].dtype, pd.DatetimeTZDtype):
                df[c] = df[c].astype(pd.DatetimeTZDtype(tz=df[c].dtype.tz))
            else:
                df[c] = df[c].astype("datetime64[ns]")
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.astype("datetime64[ns]")
    return df
