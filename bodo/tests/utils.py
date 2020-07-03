"""Utility functions for testing such as check_func() that tests a function.
"""
# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import random
import string
import numba
from decimal import Decimal
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import NopythonRewrites
import bodo
from numba.core import types
from bodo.utils.typing import BodoWarning
import warnings
import time
from bodo.utils.utils import is_distributable_typ, is_distributable_tuple_typ


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


def dist_IR_contains(*args):
    return sum([(s in bodo.transforms.distributed_pass.fir_text) for s in args])


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
):
    """test bodo compilation of function 'func' on arguments using REP, 1D, and 1D_Var
    inputs/outputs
    """
    n_pes = bodo.get_size()

    call_args = tuple(_get_arg(a, copy_input) for a in args)
    w = None

    # gives the option of passing desired output to check_func
    # in situations where pandas is buggy/lacks support
    if py_output is None:
        if convert_columns_to_pandas:
            call_args_mapped = tuple(
                convert_list_string_decimal_columns(a) for a in call_args
            )
            py_output = func(*call_args_mapped)
        else:
            py_output = func(*call_args)

    # sequential
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
        n_pes,
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
            is_distributable_typ(_typeof(a)) or is_distributable_tuple_typ(_typeof(a))
            for a in args
        )  # if none of the inputs is distributable
    ):
        return  # no need for distributed checks

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
        convert_columns_to_pandas,
        n_pes,
    )

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
        convert_columns_to_pandas,
        n_pes,
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
    n_pes,
):
    """check function output against Python without manually setting inputs/outputs
    distributions (keep the function sequential)
    """
    bodo_func = bodo.jit(func)
    call_args = tuple(_get_arg(a, copy_input) for a in args)
    # try to catch BodoWarning if no parallelism found
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always", message="No parallelism found for function", category=BodoWarning
        )
        bodo_out = bodo_func(*call_args)
        if convert_columns_to_pandas:
            bodo_out = convert_list_string_decimal_columns(bodo_out)

    passed = _test_equal_guard(
        bodo_out, py_output, sort_output, check_names, check_dtype, reset_index
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes
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
    convert_columns_to_pandas,
    n_pes,
):
    """Check function output against Python while setting the inputs/outputs as
    1D distributed
    """
    # 1D distributed
    bodo_func = bodo.jit(
        all_args_distributed_block=True, all_returns_distributed=is_out_distributed
    )(func)
    dist_args = tuple(_get_dist_arg(a, copy_input) for a in args)
    bodo_output = bodo_func(*dist_args)
    if convert_columns_to_pandas:
        bodo_output = convert_list_string_decimal_columns(bodo_output)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    # only rank 0 should check if gatherv() called on output
    passed = 1
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output, py_output, sort_output, check_names, check_dtype, reset_index
        )

    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


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
    convert_columns_to_pandas,
    n_pes,
):
    """Check function output against Python while setting the inputs/outputs as
    1D distributed variable length
    """
    bodo_func = bodo.jit(
        all_args_distributed_varlength=True, all_returns_distributed=is_out_distributed
    )(func)
    dist_args = tuple(_get_dist_arg(a, copy_input, True) for a in args)
    bodo_output = bodo_func(*dist_args)
    if convert_columns_to_pandas:
        bodo_output = convert_list_string_decimal_columns(bodo_output)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    passed = 1
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output, py_output, sort_output, check_names, check_dtype, reset_index
        )
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def _get_arg(a, copy=False):
    if copy and hasattr(a, "copy"):
        return a.copy()
    return a


def _get_dist_arg(a, copy=False, var_length=False):
    if copy and hasattr(a, "copy"):
        a = a.copy()

    arg_typ = bodo.typeof(a)
    if not is_distributable_typ(arg_typ):
        return a

    start, end = get_start_end(len(a))
    # for var length case to be different than regular 1D in chunk sizes, add
    # one extra element to last processor
    if var_length and bodo.get_size() >= 2:
        if bodo.get_rank() == bodo.get_size() - 2:
            end -= 1
        if bodo.get_rank() == bodo.get_size() - 1:
            start -= 1

    if isinstance(a, (pd.Series, pd.DataFrame)):
        return a.iloc[start:end]
    return a[start:end]


def _test_equal_guard(
    bodo_out,
    py_out,
    sort_output=False,
    check_names=True,
    check_dtype=True,
    reset_index=False,
):
    passed = 1
    try:
        _test_equal(
            bodo_out, py_out, sort_output, check_names, check_dtype, reset_index
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
    if isinstance(df.index, pd.MultiIndex):
        list_col_names = df.columns.to_list() + [x for x in df.index.names]
        return df.sort_values(list_col_names, kind="mergesort")
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
):

    if isinstance(py_out, pd.Series):
        if sort_output:
            py_out = sort_series_values_index(py_out)
            bodo_out = sort_series_values_index(bodo_out)
        if reset_index:
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.reset_index(inplace=True, drop=True)
        # we return typed extension arrays like StringArray for all APIs but Pandas
        # doesn't return them by default in all APIs yet.
        if py_out.dtype in (np.object, np.bool_):
            check_dtype = False
        pd.testing.assert_series_equal(
            bodo_out, py_out, check_names=check_names, check_dtype=check_dtype
        )
    elif isinstance(py_out, pd.Index):
        if sort_output:
            py_out = py_out.sort_values()
            bodo_out = bodo_out.sort_values()
        pd.testing.assert_index_equal(bodo_out, py_out, check_names=check_names)
    elif isinstance(py_out, pd.DataFrame):
        if sort_output:
            py_out = sort_dataframe_values_index(py_out)
            bodo_out = sort_dataframe_values_index(bodo_out)
        if reset_index:
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.reset_index(inplace=True, drop=True)
        # we return typed extension arrays like StringArray for all APIs but Pandas
        # doesn't return them by default in all APIs yet.
        if (
            np.object in py_out.dtypes.values.tolist()
            or np.bool_ in py_out.dtypes.values.tolist()
        ):
            check_dtype = False
        pd.testing.assert_frame_equal(
            bodo_out, py_out, check_names=check_names, check_dtype=check_dtype
        )
    elif isinstance(py_out, np.ndarray):
        if sort_output:
            py_out = np.sort(py_out)
            bodo_out = np.sort(bodo_out)
        # use tester of Pandas for array of objects since Numpy doesn't handle np.nan
        # properly
        if py_out.dtype == np.dtype("O") and (
            bodo_out.dtype == np.dtype("O")
            or isinstance(bodo_out.dtype, pd.BooleanDtype)
        ):
            pd.testing.assert_series_equal(
                pd.Series(py_out), pd.Series(bodo_out), check_dtype=False
            )
        else:
            np.testing.assert_array_equal(bodo_out, py_out)
    # check for array since is_extension_array_dtype() matches dtypes also
    elif pd.api.types.is_array_like(py_out) and pd.api.types.is_extension_array_dtype(
        py_out
    ):
        # bodo currently returns np object array instead of pd StringArray
        if not pd.api.types.is_extension_array_dtype(bodo_out):
            bodo_out = pd.array(bodo_out)
        if sort_output:
            py_out = py_out[py_out.argsort()]
            bodo_out = bodo_out[bodo_out.argsort()]
        pd.testing.assert_extension_array_equal(bodo_out, py_out)
    elif isinstance(py_out, float):
        # avoid equality check since paralellism can affect floating point operations
        np.testing.assert_allclose(py_out, bodo_out, 1e-4)
    elif isinstance(py_out, tuple):
        assert len(py_out) == len(bodo_out)
        for p, b in zip(py_out, bodo_out):
            _test_equal(b, p, sort_output, check_names, check_dtype)
    else:
        assert bodo_out == py_out


def _typeof(val):
    # Pandas returns an object array for .values or to_numpy() call on Series of
    # nullable int, which can't be handled in typeof. Bodo returns a nullable int array
    # see test_series_to_numpy[numeric_series_val3] and
    # test_series_get_values[series_val4]
    if (
        isinstance(val, np.ndarray)
        and val.dtype == np.dtype("O")
        and all(
            (isinstance(a, np.float) and np.isnan(a)) or isinstance(a, int) for a in val
        )
    ):
        return bodo.libs.int_arr_ext.IntegerArrayType(bodo.int64)

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


class DeadcodeTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeleine used in test_join_deadcode_cleanup and test_csv_remove_col0_used_for_len
    additional PreserveIR pass then bodo_pipeline
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=False
        )
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, NopythonRewrites)
        pipeline.finalize()
        return [pipeline]


from numba.core.compiler_machinery import FunctionPass, register_pass


@register_pass(analysis_only=False, mutates_CFG=True)
class ArrayAnalysisPass(FunctionPass):
    _name = "array_analysis_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        array_analysis = numba.parfors.array_analysis.ArrayAnalysis(
            state.typingctx,
            state.func_ir,
            state.type_annotation.typemap,
            state.type_annotation.calltypes,
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
def convert_list_string_decimal_columns(df):
    if not isinstance(df, pd.DataFrame):
        return df

    df_copy = df.copy()
    list_col = df.columns.to_list()
    n_rows = df_copy[list_col[0]].size
    # Determine which columns have list of strings in them
    # Determine which columns have Decimals in them.
    col_names_list_string = []
    col_names_array_item = []
    col_names_decimal = []
    for e_col_name in list_col:
        e_col = df[e_col_name]
        nb_list_string = 0
        nb_array_item = 0
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
            if isinstance(e_ent, Decimal):
                nb_decimal += 1
        if nb_list_string > 0:
            col_names_list_string.append(e_col_name)
        if nb_array_item > 0:
            col_names_array_item.append(e_col_name)
        if nb_decimal > 0:
            col_names_decimal.append(e_col_name)
    for e_col_name in col_names_list_string:
        e_list_str = []
        e_col = df[e_col_name]
        for i_row in range(n_rows):
            e_ent = e_col.iat[i_row]
            if isinstance(e_ent, list):
                e_str = ",".join(e_ent) + ","
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
                ),
            ):
                e_str = ",".join([str(x) for x in e_ent]) + ","
                e_list_str.append(e_str)
            else:
                e_list_str.append(np.nan)
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
    func, args, sort_output=False, reset_index=False,
):
    n_pes = bodo.get_size()

    # Computing the output in serial mode
    copy_input = True
    call_args_serial = tuple(_get_arg(a, copy_input) for a in args)
    bodo_func_serial = bodo.jit(
        all_args_distributed_block=False, all_returns_distributed=False
    )(func)
    serial_output_raw = bodo_func_serial(*call_args_serial)
    serial_output_final = convert_list_string_decimal_columns(serial_output_raw)

    # If running on just one processor, nothing more is needed.
    if n_pes == 1:
        return

    # Computing the parallel input and output.
    bodo_func_parall = bodo.jit(
        all_args_distributed_block=True, all_returns_distributed=True
    )(func)
    call_args_parall = tuple(
        _get_dist_arg(a, copy=True, var_length=False) for a in args
    )

    parall_output_raw = bodo_func_parall(*call_args_parall)
    parall_output_proc = convert_list_string_decimal_columns(parall_output_raw)
    # Collating the parallel output on just one processor.
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
                serial_output_final, parall_output_final, check_dtype=False
            )
        except Exception as e:
            print(e)
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def compute_random_decimal_array(option, n):
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


def gen_random_string_array(n, max_str_len=10):
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
        str_vals.append(val)

    # use consistent string array type with Bodo to avoid output comparison errors
    if bodo.libs.str_arr_ext.use_pd_string_array:
        return pd.array(str_vals, "string")
    return np.array(str_vals, dtype="object")  # avoid unichr dtype (TODO: support?)
