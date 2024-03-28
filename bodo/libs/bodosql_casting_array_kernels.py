# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements a number of array kernels that handling casting functions for BodoSQL
"""

import numpy as np
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import BodoError, is_overload_none


def cast_float64(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_float64_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_float32(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_float32_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_int64(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_int64_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_int32(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_int32_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_int16(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_int16_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_int8(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_int8_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    return


def cast_boolean(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_char(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_date(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return arr


def cast_timestamp(ar, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_interval(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    return


def cast_interval_util(arr, dict_encoding_state, func_id):  # pragma: no cover
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
    (cast_boolean, None, "boolean"),
    (cast_char, None, "char"),
    (cast_date, None, "date"),
    (cast_timestamp, None, "timestamp"),
    (cast_interval, cast_interval_util, "interval"),
)

# mapping from function name to equivalent numpy function
fname_to_equiv = {
    "float64": "np.float64",
    "float32": "np.float32",
    "int64": "np.int64",
    "int32": "np.int32",
    "int16": "np.int16",
    "int8": "np.int8",
    "interval": "pd.to_timedelta",
}

# mapping from function name to desired out_dtype
fname_to_dtype = {
    "float64": bodo.libs.float_arr_ext.FloatingArrayType(types.float64),
    "float32": bodo.libs.float_arr_ext.FloatingArrayType(types.float32),
    "int64": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
    "int32": bodo.libs.int_arr_ext.IntegerArrayType(types.int32),
    "int16": bodo.libs.int_arr_ext.IntegerArrayType(types.int16),
    "int8": bodo.libs.int_arr_ext.IntegerArrayType(types.int8),
    "interval": np.dtype("timedelta64[ns]"),
}


def create_cast_func_overload(func_name):
    def overload_cast_func(arr, dict_encoding_state=None, func_id=-1):
        if isinstance(arr, types.optional):
            return unopt_argument(
                f"bodo.libs.bodosql_array_kernels.cast_{func_name}",
                ["arr", "dict_encoding_state", "func_id"],
                0,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

        func_text = "def impl(arr, dict_encoding_state=None, func_id=-1):\n"
        if func_name == "boolean":
            func_text += f"  return bodo.libs.bodosql_snowflake_conversion_array_kernels.to_boolean_util(arr, numba.literally(True), dict_encoding_state, func_id)\n"
        elif func_name == "char":
            # TODO(Yipeng): Correctly support semi-structured type cast_char for scalar & vector with is_scalar parameter
            func_text += f"  return bodo.libs.bodosql_snowflake_conversion_array_kernels.to_char_util(arr, None, None)\n"
        elif func_name == "date":
            func_text += f"  return bodo.libs.bodosql_snowflake_conversion_array_kernels.to_date_util(arr, None, dict_encoding_state, func_id)\n"
        elif func_name == "timestamp":
            func_text += f"  return bodo.libs.bodosql_snowflake_conversion_array_kernels.to_timestamp_util(arr, None, None, 0, dict_encoding_state, func_id)\n"
        else:
            func_text += f"  return bodo.libs.bodosql_array_kernels.cast_{func_name}_util(arr, dict_encoding_state, func_id)"

        loc_vars = {}
        exec(func_text, {"bodo": bodo, "numba": numba}, loc_vars)

        return loc_vars["impl"]

    return overload_cast_func


def create_cast_util_overload(func_name):
    def overload_cast_util(arr, dict_encoding_state, func_id):
        arg_names = ["arr", "dict_encoding_state", "func_id"]
        arg_types = [arr, dict_encoding_state, func_id]
        propagate_null = [True, False, False]
        scalar_text = ""
        if (
            func_name[:3] == "int"
            and func_name != "interval"
            and not is_valid_boolean_arg(arr)
        ):
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
        elif func_name == "interval":
            unbox_str = (
                "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
                if bodo.utils.utils.is_array_typ(arr, True)
                else ""
            )
            scalar_text += f"res[i] = {unbox_str}(pd.to_timedelta(arg0))"
        else:
            scalar_text += f"res[i] = {fname_to_equiv[func_name]}(arg0)"

        out_dtype = fname_to_dtype[func_name]

        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_cast_util


def _install_cast_func_overloads(funcs_utils_names):
    for func, util, name in funcs_utils_names:
        overload(func)(create_cast_func_overload(name))
        if name not in ("boolean", "char", "date", "timestamp"):
            overload(util)(create_cast_util_overload(name))


_install_cast_func_overloads(cast_funcs_utils_names)


@numba.generated_jit(nopython=True)
def round_to_int64(x):
    if isinstance(x, types.optional):
        return unopt_argument(
            f"bodo.libs.bodosql_casting_array_kernels.round_to_int64", ["x"], 0
        )

    def impl(x):
        # we can't use cast_int64(round(x, 0)), because round(x, 0) doesn't
        # handle cases where x is not in [INT64_MIN, INT64_MAX].
        rounded = bodo.libs.bodosql_array_kernels.round(x, 0)
        return bodo.libs.bodosql_casting_array_kernels.round_to_int64_util(x, rounded)

    return impl


@numba.generated_jit(nopython=True)
def round_to_int64_util(x, rounded_x):
    arg_names = ["x", "rounded_x"]
    arg_types = [x, rounded_x]
    propagate_null = [True, True]
    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

    imax = np.iinfo(np.int64).max
    imin = np.iinfo(np.int64).min
    extra_globals = {"imax": imax, "imin": imin}

    # Essentially the logic of cast_int64 for floats, but we use the rounded
    # value
    scalar_text = "is_valid = not(pd.isna(arg0) or np.isinf(arg0) or arg0 < imin or arg0 > imax)\n"
    scalar_text += "if not is_valid:\n"
    scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "else:\n"
    scalar_text += f"  res[i] = np.int64(arg1)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )
