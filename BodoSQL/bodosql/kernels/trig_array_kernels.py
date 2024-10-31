from numba.core import types
from numba.extending import overload

import bodo
import bodosql
from bodosql.kernels.array_kernel_utils import (
    gen_vectorized,
    unopt_argument,
    verify_int_float_arg,
)


def acos(arr):  # pragma: no cover
    # Dummy function used to install the overload
    return


def acosh(arr):  # pragma: no cover
    return


def asin(arr):  # pragma: no cover
    return


def asinh(arr):  # pragma: no cover
    return


def atan(arr):  # pragma: no cover
    return


def atanh(arr):  # pragma: no cover
    return


def atan2(arr0, arr1):  # pragma: no cover
    return


def cos(arr):  # pragma: no cover
    return


def cosh(arr):  # pragma: no cover
    return


def cot(arr):  # pragma: no cover
    return


def sin(arr):  # pragma: no cover
    return


def sinh(arr):  # pragma: no cover
    return


def tan(arr):  # pragma: no cover
    return


def tanh(arr):  # pragma: no cover
    return


def radians(arr):  # pragma: no cover
    return


def degrees(arr):  # pragma: no cover
    return


def acos_util(arr):  # pragma: no cover
    # Dummy function used for overload
    return


def acosh_util(arr):  # pragma: no cover
    return


def asin_util(arr):  # pragma: no cover
    return


def asinh_util(arr):  # pragma: no cover
    return


def atan_util(arr):  # pragma: no cover
    return


def atanh_util(arr):  # pragma: no cover
    return


def atan2_util(arr0, arr1):  # pragma: no cover
    return


def cos_util(arr):  # pragma: no cover
    return


def cosh_util(arr):  # pragma: no cover
    return


def cot_util(arr):  # pragma: no cover
    return


def sin_util(arr):  # pragma: no cover
    return


def sinh_util(arr):  # pragma: no cover
    return


def tan_util(arr):  # pragma: no cover
    return


def tanh_util(arr):  # pragma: no cover
    return


def radians_util(arr):  # pragma: no cover
    return


def degrees_util(arr):  # pragma: no cover
    return


funcs_utils_names = (
    (acos, acos_util, "ACOS"),
    (acosh, acosh_util, "ACOSH"),
    (asin, asin_util, "ASIN"),
    (asinh, asinh_util, "ASINH"),
    (atan, atan_util, "ATAN"),
    (atanh, atanh_util, "ATANH"),
    (atan2, atan2_util, "ATAN2"),
    (cos, cos_util, "COS"),
    (cosh, cosh_util, "COSH"),
    (cot, cot_util, "COT"),
    (sin, sin_util, "SIN"),
    (sinh, sinh_util, "SINH"),
    (tan, tan_util, "TAN"),
    (tanh, tanh_util, "TANH"),
    (radians, radians_util, "RADIANS"),
    (degrees, degrees_util, "DEGREES"),
)

double_arg_funcs = ("ATAN2",)


def create_trig_func_overload(func_name):
    if func_name not in double_arg_funcs:
        func_name = func_name.lower()

        def overload_func(arr):
            """Handles cases where func_name receives an optional argument and forwards
            to the appropriate version of the real implementation"""
            if isinstance(arr, types.optional):
                return unopt_argument(
                    f"bodosql.kernels.trig_array_kernels.{func_name}_util", ["arr"], 0
                )

            func_text = "def impl(arr):\n"
            func_text += (
                f"  return bodosql.kernels.trig_array_kernels.{func_name}_util(arr)"
            )
            loc_vars = {}
            exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

            return loc_vars["impl"]

    else:
        func_name = func_name.lower()

        def overload_func(arr0, arr1):
            """Handles cases where func_name receives optional arguments and forwards
            to the appropriate version of the real implementation"""
            args = [arr0, arr1]
            for i in range(2):
                if isinstance(args[i], types.optional):
                    return unopt_argument(
                        f"bodosql.kernels.{func_name}",
                        ["arr0", "arr1"],
                        i,
                    )

            func_text = "def impl(arr0, arr1):\n"
            func_text += f"  return bodosql.kernels.trig_array_kernels.{func_name}_util(arr0, arr1)"
            loc_vars = {}
            exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

            return loc_vars["impl"]

    return overload_func


def create_trig_util_overload(func_name):  # pragma: no cover
    """Creates an overload function to support trig functions on
       a string array representing a column of a SQL table

    Args:
        func_name: which trig function is being called (e.g. "ACOS")

    Returns:
        (function): a utility that takes in one argument and returns
        the appropriate trig function applied to the argument, where the
        argument could be an array/scalar/null.
    """
    if func_name not in double_arg_funcs:

        def overload_trig_util(arr):
            verify_int_float_arg(arr, func_name, "arr")

            arg_names = [
                "arr",
            ]
            arg_types = [arr]
            propagate_null = [True]
            scalar_text = ""
            if func_name == "ACOS":
                scalar_text += "res[i] = np.arccos(arg0)"
            elif func_name == "ACOSH":
                scalar_text += "res[i] = np.arccosh(arg0)"
            elif func_name == "ASIN":
                scalar_text += "res[i] = np.arcsin(arg0)"
            elif func_name == "ASINH":
                scalar_text += "res[i] = np.arcsinh(arg0)"
            elif func_name == "ATAN":
                scalar_text += "res[i] = np.arctan(arg0)"
            elif func_name == "ATANH":
                scalar_text += "res[i] = np.arctanh(arg0)"
            elif func_name == "COS":
                scalar_text += "res[i] = np.cos(arg0)"
            elif func_name == "COSH":
                scalar_text += "res[i] = np.cosh(arg0)"
            elif func_name == "COT":
                scalar_text += "res[i] = np.divide(1, np.tan(arg0))"
            elif func_name == "SIN":
                scalar_text += "res[i] = np.sin(arg0)"
            elif func_name == "SINH":
                scalar_text += "res[i] = np.sinh(arg0)"
            elif func_name == "TAN":
                scalar_text += "res[i] = np.tan(arg0)"
            elif func_name == "TANH":
                scalar_text += "res[i] = np.tanh(arg0)"
            elif func_name == "RADIANS":
                scalar_text += "res[i] = np.radians(arg0)"
            elif func_name == "DEGREES":
                scalar_text += "res[i] = np.degrees(arg0)"
            else:
                raise ValueError(f"Unknown function name: {func_name}")

            out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

            return gen_vectorized(
                arg_names, arg_types, propagate_null, scalar_text, out_dtype
            )

    else:

        def overload_trig_util(arr0, arr1):
            verify_int_float_arg(arr0, func_name, "arr0")
            verify_int_float_arg(arr1, func_name, "arr1")

            arg_names = ["arr0", "arr1"]
            arg_types = [arr0, arr1]
            propagate_null = [True, True]
            scalar_text = ""
            if func_name == "ATAN2":
                scalar_text += "res[i] = np.arctan2(arg0, arg1)\n"
            else:
                raise ValueError(f"Unknown function name: {func_name}")

            out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

            return gen_vectorized(
                arg_names,
                arg_types,
                propagate_null,
                scalar_text,
                out_dtype,
            )

    return overload_trig_util


def _install_trig_overload(funcs_utils_names):
    """Creates and installs the overloads for trig functions"""
    for func, util, func_name in funcs_utils_names:
        func_overload_impl = create_trig_func_overload(func_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_trig_util_overload(func_name)
        overload(util)(util_overload_impl)


_install_trig_overload(funcs_utils_names)
