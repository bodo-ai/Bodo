"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""

import numpy as np
import pandas as pd
import pytest

import bodosql
from bodo.tests.utils import check_func, pytest_slow_unless_codegen
from bodosql.kernels.array_kernel_utils import vectorized_sol

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen

test_arrs = [
    pd.Series([0, 1, -1, 0.5, -0.5, 0.3212, -0.78]),
    pd.Series(
        [
            0,
            1,
            -1,
            10000,
            -100000,
            20,
            -139,
        ]
    ),
    pd.Series([1, -1, 0.1, -0.1, 1234, pd.NA, -4321, 3], dtype=pd.Float32Dtype()),
    pd.Series([-1, 0, 1, None, 2], dtype=pd.Int32Dtype()),
]

single_arg_np_map = {
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atanh": "arctanh",
    "cos": "cos",
    "cosh": "cosh",
    "sin": "sin",
    "sinh": "sinh",
    "tan": "tan",
    "tanh": "tanh",
    "radians": "radians",
    "degrees": "degrees",
}
single_arg_np_list = list(single_arg_np_map.keys()) + ["cot"]
double_arg_np_map = {
    "atan2": "arctan2",
}
double_arg_np_list = list(double_arg_np_map.keys())


@pytest.mark.parametrize("arr", test_arrs)
@pytest.mark.parametrize("func", single_arg_np_list)
def test_trig_single_arg_funcs(arr, func, memory_leak_check):
    test_impl = "def impl(arr):\n"
    test_impl += f"  return pd.Series(bodosql.kernels.{func}(arr))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)

    # Simulates CONV on a single row
    scalar_impl = "def impl(elem):\n"
    if func == "cot":
        # COT doesn't have a 1 to 1 numpy mapping.
        scalar_impl += (
            "    return np.divide(1, np.tan(elem)) if not pd.isna(elem) else None"
        )
        # We need to avoid divide by 0 issue
        arr = arr.copy()
        arr[arr == 0] = 1
    else:
        scalar_impl += f"    return np.{single_arg_np_map[func]}(elem) if not pd.isna(elem) else None"
    scalar_vars = {}
    exec(scalar_impl, {"np": np, "pd": pd}, scalar_vars)

    conv_answer = vectorized_sol((arr,), scalar_vars["impl"], np.float64)
    check_func(
        impl_vars["impl"],
        (arr,),
        py_output=conv_answer,
        check_dtype=False,
        reset_index=True,
        # This test can output NaN, so we don't convert to nullable float
        # as this will be coerced to NA.
        convert_to_nullable_float=False,
    )


@pytest.mark.parametrize("func", single_arg_np_list)
def test_trig_single_arg_option(func, memory_leak_check):
    test_impl = "def impl(a, flag0):\n"
    test_impl += "  arg0 = a if flag0 else None\n"
    test_impl += f"  return bodosql.kernels.{func}(arg0)"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql}, impl_vars)

    for flag0 in [True, False]:
        if func == "cot":
            # COT doesn't have a 1 to 1 numpy mapping.
            answer = np.divide(1, np.tan(0.75)) if flag0 else None
        else:
            answer = eval(f"np.{single_arg_np_map[func]}(0.75)") if flag0 else None
        check_func(impl_vars["impl"], (0.75, flag0), py_output=answer)


@pytest.mark.parametrize(
    "arr1",
    [pd.Series(list(arr), dtype=arr.dtype) for arr in test_arrs],
)
@pytest.mark.parametrize(
    "arr0",
    [pd.Series(list(arr)[::-1], dtype=arr.dtype) for arr in test_arrs],
)
@pytest.mark.parametrize("func", double_arg_np_list)
def test_trig_double_arg_funcs(arr0, arr1, func, memory_leak_check):
    if len(arr0) != len(arr1):
        return
    test_impl = "def impl(arr0, arr1):\n"
    test_impl += f"  return pd.Series(bodosql.kernels.{func}(arr0, arr1))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)

    # Simulates trig func on a single row
    scalar_impl = "def impl(elem0, elem1):\n"
    scalar_impl += f"    return np.{double_arg_np_map[func]}(elem0, elem1) if not pd.isna(elem0) and not pd.isna(elem1) else None"
    scalar_vars = {}
    exec(scalar_impl, {"np": np, "pd": pd}, scalar_vars)

    trig_func_answer = vectorized_sol((arr0, arr1), scalar_vars["impl"], np.float64)
    check_func(
        impl_vars["impl"],
        (arr0, arr1),
        py_output=trig_func_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize("func", double_arg_np_list)
def test_trig_double_arg_option(func, memory_leak_check):
    test_impl = "def impl(a, b, flag0, flag1):\n"
    test_impl += "  arg0 = a if flag0 else None\n"
    test_impl += "  arg1 = b if flag1 else None\n"
    test_impl += f"  return bodosql.kernels.{func}(arg0, arg1)"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql}, impl_vars)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                eval(f"np.{double_arg_np_map[func]}(0.75, 0.5)")
                if flag0 and flag1
                else None
            )
            check_func(impl_vars["impl"], (0.75, 0.5, flag0, flag1), py_output=answer)
