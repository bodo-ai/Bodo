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


def test_func(func, args, is_out_distributed=None, sort_output=False):
    """test bodo compilation of function 'func' on arguments using REP and 1D
    inputs/outputs
    """
    py_output = func(*args)
    # sequential
    bodo_func = bodo.jit(func)
    _test_equal(bodo_func(*args), py_output, sort_output)

    if is_out_distributed is None:
        # assume all distributable output is distributed if not specified
        is_out_distributed = isinstance(py_output,
            (pd.Series, pd.Index, pd.DataFrame, np.ndarray))

    # 1D distributed
    bodo_func = bodo.jit(
        all_args_distributed=True,
        all_returns_distributed=is_out_distributed)(func)
    dist_args = tuple(_get_dist_arg(a) for a in args)
    bodo_output = bodo_func(*dist_args)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    if bodo.get_rank() == 0:
        _test_equal(bodo_output, py_output, sort_output)

    # 1D distributed variable length
    bodo_func = bodo.jit(
        all_args_distributed_varlength=True,
        all_returns_distributed=is_out_distributed)(func)
    dist_args = tuple(_get_dist_arg(a) for a in args)
    bodo_output = bodo_func(*dist_args)
    if is_out_distributed:
        bodo_output = bodo.gatherv(bodo_output)
    if bodo.get_rank() == 0:
        _test_equal(bodo_output, py_output, sort_output)


def _get_dist_arg(a):
    arg_typ = bodo.typeof(a)
    if not bodo.utils.utils.is_distributable_typ(arg_typ):
        return a

    start, end = get_start_end(len(a))
    if isinstance(a, (pd.Series, pd.DataFrame)):
        return a.iloc[start:end]
    return a[start:end]


def _test_equal(bodo_out, py_out, sort_output):
    if sort_output:
        bodo_out = np.sort(bodo_out)
        py_out = np.sort(py_out)

    if isinstance(py_out, pd.Series):
        pd.testing.assert_series_equal(bodo_out, py_out)
    elif isinstance(py_out, pd.Index):
        pd.testing.assert_index_equal(bodo_out, py_out)
    elif isinstance(py_out, pd.DataFrame):
        pd.testing.assert_frame_equal(bodo_out, py_out)
    elif isinstance(py_out, np.ndarray):
        np.testing.assert_array_equal(bodo_out, py_out)
    else:
        assert bodo_out == py_out
