import pandas as pd
import numpy as np
import bodo


def count_array_REPs():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v==Distribution.REP for v in vals])


def count_parfor_REPs():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.REP for v in vals])


def count_parfor_OneDs():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD for v in vals])


def count_array_OneDs():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD for v in vals])


def count_parfor_OneD_Vars():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])


def count_array_OneD_Vars():
    from bodo.transforms.distributed_pass import Distribution
    vals = bodo.transforms.distributed_pass.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])


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


def check_func(func, args, is_out_distributed=None, sort_output=False,
                         check_names=True, copy_input=False, check_dtype=True):
    """test bodo compilation of function 'func' on arguments using REP and 1D
    inputs/outputs
    """
    call_args = tuple(_get_arg(a, copy_input) for a in args)
    py_output = func(*call_args)
    # sequential
    bodo_func = bodo.jit(func)
    call_args = tuple(_get_arg(a, copy_input) for a in args)
    _test_equal(bodo_func(*call_args), py_output, sort_output, check_names,
                                                                   check_dtype)

    if is_out_distributed is None:
        # assume all distributable output is distributed if not specified
        is_out_distributed = isinstance(py_output,
            (pd.Series, pd.Index, pd.DataFrame, np.ndarray,
            pd.arrays.IntegerArray))

    # 1D distributed
    bodo_func = bodo.jit(
        all_args_distributed=True,
        all_returns_distributed=is_out_distributed)(func)
    dist_args = tuple(_get_dist_arg(a, copy_input) for a in args)
    bodo_output = bodo_func(*dist_args)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    # only rank 0 should check if gatherv() called on output
    if not is_out_distributed or bodo.get_rank() == 0:
        _test_equal(bodo_output, py_output, sort_output, check_names,
                                                                 check_dtype)

    # 1D distributed variable length
    bodo_func = bodo.jit(
        all_args_distributed_varlength=True,
        all_returns_distributed=is_out_distributed)(func)
    dist_args = tuple(_get_dist_arg(a, copy_input, True) for a in args)
    bodo_output = bodo_func(*dist_args)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    if not is_out_distributed or bodo.get_rank() == 0:
        _test_equal(bodo_output, py_output, sort_output, check_names,
                                                                 check_dtype)


def _get_arg(a, copy=False):
    if copy and hasattr(a, 'copy'):
        return a.copy()
    return a


def _get_dist_arg(a, copy=False, var_length=False):
    if copy and hasattr(a, 'copy'):
        a = a.copy()

    arg_typ = bodo.typeof(a)
    if not bodo.utils.utils.is_distributable_typ(arg_typ):
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


def _test_equal(bodo_out, py_out, sort_output, check_names=True,
                                                             check_dtype=True):

    if isinstance(py_out, pd.Series):
        if sort_output:
            py_out.sort_values(inplace=True)
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.sort_values(inplace=True)
            bodo_out.reset_index(inplace=True, drop=True)
        pd.testing.assert_series_equal(
            bodo_out, py_out, check_names=check_names, check_dtype=check_dtype)
    elif isinstance(py_out, pd.Index):
        if sort_output:
            py_out = py_out.sort_values()
            bodo_out = bodo_out.sort_values()
        pd.testing.assert_index_equal(
            bodo_out, py_out, check_names=check_names)
    elif isinstance(py_out, pd.DataFrame):
        if sort_output:
            py_out.sort_values(py_out.columns.to_list(), inplace=True)
            py_out.reset_index(inplace=True, drop=True)
            bodo_out.sort_values(bodo_out.columns.to_list(), inplace=True)
            bodo_out.reset_index(inplace=True, drop=True)
        pd.testing.assert_frame_equal(
            bodo_out, py_out, check_names=check_names, check_dtype=check_dtype)
    elif isinstance(py_out, np.ndarray):
        if sort_output:
            py_out.sort()
            bodo_out.sort()
        np.testing.assert_array_equal(bodo_out, py_out)
    elif isinstance(py_out, pd.arrays.IntegerArray):
        pd.util.testing.assert_extension_array_equal(bodo_out, py_out)
    else:
        assert bodo_out == py_out
