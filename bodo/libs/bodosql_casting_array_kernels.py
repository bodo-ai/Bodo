# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements a number of array kernels that handling casting functions for BodoSQL
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import BodoError


def cast_float64(arr):  # pragma: no cover
    return


def cast_float64_util(arr):  # pragma: no cover
    return


def cast_float32(arr):  # pragma: no cover
    return


def cast_float32_util(arr):  # pragma: no cover
    return


def cast_int64(arr):  # pragma: no cover
    return


def cast_int64_util(arr):  # pragma: no cover
    return


def cast_int32(arr):  # pragma: no cover
    return


def cast_int32_util(arr):  # pragma: no cover
    return


def cast_int16(arr):  # pragma: no cover
    return


def cast_int16_util(arr):  # pragma: no cover
    return


def cast_int8(arr):  # pragma: no cover
    return


def cast_int8_util(arr):  # pragma: no cover
    return


# casting functions to be overloaded
# each tuple is (fn to overload, util to overload, name of fn)
cast_funcs_utils_names = (
    (cast_float64, cast_float64_util, "float64"),
    (cast_float32, cast_float32_util, "float32"),
    (cast_int64, cast_int64_util, "int64"),
    (cast_int32, cast_int32_util, "int32"),
    (cast_int16, cast_int16_util, "int16"),
    (cast_int8, cast_int8_util, "int8"),
)

# mapping from function name to equivalent numpy function
fname_to_equiv = {
    "float64": "np.float64",
    "float32": "np.float32",
    "int64": "np.int64",
    "int32": "np.int32",
    "int16": "np.int16",
    "int8": "np.int8",
}

# mapping from function name to desired out_dtype
fname_to_dtype = {
    "float64": types.Array(bodo.float64, 1, "C"),
    "float32": types.Array(bodo.float32, 1, "C"),
    "int64": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
    "int32": bodo.libs.int_arr_ext.IntegerArrayType(types.int32),
    "int16": bodo.libs.int_arr_ext.IntegerArrayType(types.int16),
    "int8": bodo.libs.int_arr_ext.IntegerArrayType(types.int8),
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
        scalar_text = ""
        if func_name[:3] == "int" and not is_valid_boolean_arg(arr):
            if is_valid_int_arg(arr):
                scalar_text += "if arg0 < np.iinfo(np.int64).min or arg0 > np.iinfo(np.int64).max:\n"
                scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
                scalar_text += "else:\n"
                scalar_text += f"  res[i] = {fname_to_equiv[func_name]}(arg0)\n"
            else:
                # Note that not all integers are representable in float64 (e.g. 2**63 - 1), so we check
                # if string inputs are valid integers before proceeding with the cast.
                if is_valid_string_arg(arr):
                    scalar_text = "i_val = 0\n"
                    scalar_text += "f_val = np.float64(arg0)\n"
                    scalar_text += "is_valid = not (pd.isna(f_val) or np.isinf(f_val) or f_val < np.iinfo(np.int64).min or f_val > np.iinfo(np.int64).max)\n"
                    scalar_text += "is_int = (f_val % 1 == 0)\n"
                    scalar_text += "if not (is_valid and is_int):\n"
                    scalar_text += "  val = f_val\n"
                    scalar_text += "else:\n"
                    scalar_text += "  val = np.int64(arg0)\n"
                    scalar_text += "  i_val = np.int64(arg0)\n"
                else:
                    # must be a float
                    if not is_valid_float_arg(arr):
                        raise BodoError(
                            "only strings, floats, booleans, and ints can be cast to ints"
                        )
                    scalar_text += "val = arg0\n"
                    scalar_text += "is_valid = not(pd.isna(val) or np.isinf(val) or val < np.iinfo(np.int64).min or val > np.iinfo(np.int64).max)\n"
                    scalar_text += "is_int = (val % 1 == 0)\n"
                # We have to set the output to null because of overflow / underflow issues with large/small ints,
                # (note that snowflake supports up to 128 bit ints, which we currently cannot).
                scalar_text += "if not is_valid:\n"
                scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
                scalar_text += "else:\n"
                if is_valid_float_arg(arr):
                    scalar_text += "  i_val = np.int64(val)\n"
                # [BE-3819] We must cast to int64 first in order to avoid numba involving
                # float inputs, e.g. numba.jit(lambda: np.int32(-2234234254.0)) -> 0 while
                # numba.jit(lambda: np.int32(-2234234254)) -> 2060733042, as desired
                scalar_text += "  if not is_int:\n"
                scalar_text += (
                    "    ans = np.int64(np.sign(val) * np.floor(np.abs(val) + 0.5))\n"
                )
                scalar_text += "  else:\n"
                scalar_text += "    ans = i_val\n"
                if func_name == "int64":
                    scalar_text += f"  res[i] = ans\n"
                else:
                    scalar_text += f"  res[i] = {fname_to_equiv[func_name]}(ans)"
        else:
            scalar_text += f"res[i] = {fname_to_equiv[func_name]}(arg0)"

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
