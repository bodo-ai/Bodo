"""
Implements array kernels such as median and quantile.
"""
import numpy as np

import numba
from numba import types, cgutils
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from numba.targets.arrayobj import make_array

import bodo
from bodo.utils import _numba_to_c_type_map, unliteral_all

import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
from bodo.libs import quantile_alg
import llvmlite.binding as ll
ll.add_symbol('quantile_parallel', quantile_alg.quantile_parallel)
ll.add_symbol('nth_sequential', quantile_alg.nth_sequential)
ll.add_symbol('nth_parallel', quantile_alg.nth_parallel)


nth_sequential = types.ExternalFunction("nth_sequential",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))

nth_parallel = types.ExternalFunction("nth_parallel",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))


@numba.njit
def nth_element(arr, k, parallel=False):
    res = np.empty(1, arr.dtype)
    type_enum = bodo.libs.distributed_api.get_type_enum(arr)
    if parallel:
        nth_parallel(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    else:
        nth_sequential(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    return res[0]


sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value


@numba.njit
def median(arr, parallel=False):
    # similar to numpy/lib/function_base.py:_median
    # TODO: check return types, e.g. float32 -> float32
    n = len(arr)
    if parallel:
        n = bodo.libs.distributed_api.dist_reduce(n, np.int32(sum_op))
    k = n // 2

    # odd length case
    if n % 2 == 1:
        return nth_element(arr, k, parallel)

    v1 = nth_element(arr, k-1, parallel)
    v2 = nth_element(arr, k, parallel)
    return (v1 + v2) / 2



def quantile(A, q):  # pragma: no cover
    return 0


def quantile_parallel(A, q):  # pragma: no cover
    return 0


@infer_global(quantile)
@infer_global(quantile_parallel)
class QuantileType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in [2, 3]
        return signature(types.float64, *unliteral_all(args))


@lower_builtin(quantile, types.npytypes.Array, types.float64)
@lower_builtin(quantile_parallel, types.npytypes.Array, types.float64, types.intp)
def lower_dist_quantile(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    assert sig.args[0].ndim == 1

    arr = make_array(sig.args[0])(context, builder, args[0])
    local_size = builder.extract_value(arr.shape, 0)

    if len(args) == 3:
        total_size = args[2]
    else:
        # sequential case
        total_size = local_size

    call_args = [builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                 local_size, total_size, args[1], builder.load(typ_arg)]

    # array, size, total_size, quantile, type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                lir.DoubleType(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_parallel")
    return builder.call(fn, call_args)
