# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements a number of array kernels that handling casting functions for BodoSQL
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


def cast_float64(arr):  # pragma: no cover
    return


def cast_float64_util(arr):  # pragma: no cover
    return


def cast_float32(arr):  # pragma: no cover
    return


def cast_float32_util(arr):  # pragma: no cover
    return


# casting functions to be overloaded
# each tuple is (fn to overload, util to overload, name of fn)
cast_funcs_utils_names = (
    (cast_float64, cast_float64_util, "float64"),
    (cast_float32, cast_float32_util, "float32"),
)

# mapping from function name to equivalent numpy function
fname_to_equiv = {
    "float64": "np.float64",
    "float32": "np.float32",
}

# mapping from function name to desired out_dtype
fname_to_dtype = {
    "float64": types.Array(bodo.float64, 1, "C"),
    "float32": types.Array(bodo.float32, 1, "C"),
}


def create_cast_func_overload(func_name):
    def overload_cast_func(arr):
        if isinstance(arr, types.optional):
            return unopt_argument(
                f"bodo.libs.bodosql_array_kernels.cast_{func_name}", ["arr"], 0
            )

        func_text = "def impl(arr):\n"
        func_text += (
            f"  return bodo.libs.bodosql_array_kernels.cast_{func_name}_util(arr)"
        )

        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_cast_func


def create_cast_util_overload(func_name):
    def overload_cast_util(arr):
        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]
        scalar_text = f"res[i] = {fname_to_equiv[func_name]}(arg0)"

        out_dtype = fname_to_dtype[func_name]

        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
        )

    return overload_cast_util


def _install_cast_func_overloads(funcs_utils_names):
    for func, util, name in funcs_utils_names:
        overload(func)(create_cast_func_overload(name))
        overload(util)(create_cast_util_overload(name))


_install_cast_func_overloads(cast_funcs_utils_names)
