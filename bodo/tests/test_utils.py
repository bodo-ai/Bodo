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


def test_func(func, args):
    """test bodo compilation of function 'func' on arguments using REP and 1D
    inputs/outputs
    """
    py_output = func(*args)
    # sequential
    bodo_func = bodo.jit(func)
    pd.testing.assert_series_equal(bodo_func(*args), py_output)

    # 1D distributed
    bodo_func = bodo.jit(
        all_args_distributed=True, all_returns_distributed=True)(func)
    dist_args = tuple(_get_dist_arg(a) for a in args)
    bodo_output = bodo_func(*dist_args)
    bodo_output = bodo.gatherv(bodo_output)
    if bodo.get_rank() == 0:
        pd.testing.assert_series_equal(bodo_output, py_output)


def _get_dist_arg(a):
    if not bodo.utils.utils.is_distributable_typ(bodo.typeof(a)):
        return a

    start, end = get_start_end(len(a))
    return a[start:end]
