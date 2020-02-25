# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import numba
from numba.untyped_passes import PreserveIR
from numba.typed_passes import NopythonRewrites
import bodo
from bodo.utils.typing import BodoWarning
import warnings
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
    py_output=None,
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
        py_output = func(*call_args)

    # sequential
    w = check_func_seq(
        func, args, py_output, copy_input, sort_output, check_names, check_dtype, n_pes
    )

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
        n_pes,
    )


def check_func_seq(
    func, args, py_output, copy_input, sort_output, check_names, check_dtype, n_pes
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

    passed = _test_equal_guard(
        bodo_out, py_output, sort_output, check_names, check_dtype
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
    n_pes,
):
    """Check function output against Python while setting the inputs/outputs as
    1D distributed
    """
    # 1D distributed
    bodo_func = bodo.jit(
        all_args_distributed=True, all_returns_distributed=is_out_distributed
    )(func)
    dist_args = tuple(_get_dist_arg(a, copy_input) for a in args)
    bodo_output = bodo_func(*dist_args)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    # only rank 0 should check if gatherv() called on output
    passed = 1
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output, py_output, sort_output, check_names, check_dtype
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
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    passed = 1
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_output, py_output, sort_output, check_names, check_dtype
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
    bodo_out, py_out, sort_output, check_names=True, check_dtype=True
):
    passed = 1
    try:
        _test_equal(bodo_out, py_out, sort_output, check_names, check_dtype)
    except Exception as e:
        print(e)
        passed = 0
    return passed


def _test_equal(
    bodo_out, py_out, sort_output=False, check_names=True, check_dtype=True
):

    if isinstance(py_out, pd.Series):
        if sort_output:
            # pandas fails if all null integer column is sorted
            if not py_out.isnull().all():
                py_out.sort_values(inplace=True)
                bodo_out.sort_values(inplace=True)
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
            py_out.sort_values(py_out.columns.to_list(), inplace=True)
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.sort_values(bodo_out.columns.to_list(), inplace=True)
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
    elif isinstance(py_out, pd.arrays.IntegerArray):
        if sort_output:
            py_out = py_out[py_out.argsort()]
            bodo_out = bodo_out[bodo_out.argsort()]
        pd.util.testing.assert_extension_array_equal(bodo_out, py_out)
    elif isinstance(py_out, float):
        # avoid equality check since paralellism can affect floating point operations
        np.testing.assert_allclose(py_out, bodo_out, 1e-4)
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
        and bodo.libs.str_arr_ext._infer_ndarray_obj_dtype(S) == numba.types.bool_
    )


def is_list_str_object_series(S):
    if isinstance(S, pd.Series):
        S = S.values
    return S.dtype == np.dtype("O") and bodo.libs.str_arr_ext._infer_ndarray_obj_dtype(
        S
    ) == numba.types.List(numba.types.unicode_type)


class DeadcodeTestPipeline(bodo.compiler.BodoCompiler):
    """
    pipeleine used in test_join_deadcode_cleanup and test_csv_remove_col0_used_for_len
    additional PreserveIR pass then bodo_pipeline
    """

    def define_pipelines(self):
        [pipeline] = self._create_bodo_pipeline(True)
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, NopythonRewrites)
        pipeline.finalize()
        return [pipeline]
