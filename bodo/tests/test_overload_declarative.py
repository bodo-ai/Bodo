from types import NoneType

import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.hiframes
import bodo.hiframes.series_impl
from bodo.ir.argument_checkers import (
    BooleanScalarArgumentChecker,
    ConstantArgumentChecker,
    FloatScalarArgumentChecker,
    IntegerScalarArgumentChecker,
    NDistinctValueArgumentChecker,
    NumericScalarArgumentChecker,
    NumericSeriesBinOpChecker,
    OverloadArgumentsChecker,
    StringScalarArgumentChecker,
)
from bodo.ir.declarative_templates import overload_method_declarative
from bodo.tests.utils import run_rank0
from bodo.utils.typing import BodoError


def _val_to_string(val):
    """
    Embed *val* as it's constructor in a string.
    *val* can be a Series, Index, RangeIndex, numpy array,
    DataFrame, str, int, float or bool
    """
    if isinstance(val, pd.Series):
        elems_str = ",".join(map(_val_to_string, val.array))
        return f"pd.Series([{elems_str}])"
    if isinstance(val, pd.RangeIndex):
        return f"pd.RangeIndex(start={val.start}, stop={val.stop}, step={val.step})"
    if isinstance(val, pd.Index):
        elems_str = ",".join(map(str, val.array))
        return f"pd.Index([{elems_str}])"
    if isinstance(val, np.ndarray):
        return f"np.array({list(val)})"
    if isinstance(val, pd.DataFrame):
        df_dict = dict(val)
        for key, val in df_dict.items():
            df_dict[key] = list(val)
        return f"pd.DataFrame({df_dict})"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


@run_rank0
@pytest.mark.parametrize(
    "args, kwargs, expected_err_msg",
    [
        pytest.param((True, 0), {"arg3": 0}, None, id="no_error_message"),
        pytest.param(
            (True, None),
            {"arg3": 0, "arg4": 2},
            r"pd.Series.do_something\(\): arg4 parameter only supports default value 3",
            id="unsupported_default",
        ),
        pytest.param(
            (True, 9),
            {"arg3": 0},
            r"pd.Series.do_something\(\): Expected 'arg2' to be a compile time constant and must be None, 1, 0 or \"10\". Got: ",
            id="bad_const_value",
        ),
        pytest.param(
            (True, 0),
            {"arg3": "hello"},
            r"pd.Series.do_something\(\): Expected 'arg3' to be a constant Integer, Tuple or None. Got: ",
            id="bad_const_type",
        ),
    ],
)
def test_literal_argument_checkers(args, kwargs, expected_err_msg):
    """Test argument checkers check method"""
    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.SeriesType,
        "do_something",
        path="pd.Series.do_something",
        unsupported_args=["arg4"],
        description="This method does something",
        method_args_checker=OverloadArgumentsChecker(
            [
                NDistinctValueArgumentChecker("arg1", [True, False]),
                NDistinctValueArgumentChecker("arg2", [None, 1, 0, "10"]),
                ConstantArgumentChecker("arg3", [int, tuple, NoneType]),
            ]
        ),
    )
    def overload_series_do_something(S, arg1, arg2, arg3=0, arg4=3):
        def impl(S, arg1, arg2, arg3=0, arg4=3):
            return 1

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_do_something",
        overload_series_do_something,
    )

    try:
        func_text = (
            "def test_impl(S):\n"
            f"  return S.do_something({arguments_str}, {kwargs_str})\n"
        )
        locals = {}
        gbls = {}
        exec(func_text, gbls, locals)
        test_impl = locals["test_impl"]

        S = pd.Series([1, 2, 3])

        if expected_err_msg is None:
            # expect to compile successfully
            bodo.jit(test_impl)(S)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_do_something")


@run_rank0
@pytest.mark.parametrize(
    "use_constant",
    [
        pytest.param(True, id="use_constant"),
        pytest.param(False, id="no_constant"),
    ],
)
@pytest.mark.parametrize(
    "args, kwargs, expected_err_msg",
    [
        pytest.param(
            ("hello", np.nan), {"arg3": True, "arg4": 2}, None, id="no_error_message"
        ),
        pytest.param(
            (1, 0.0),
            {"arg3": True, "arg4": 2},
            r"pd.Series.do_something2\(\): Expected 'arg1' to be type String. Got:",
            id="str_error_message",
        ),
        pytest.param(
            ("hello", 0),
            {"arg3": True, "arg4": 2},
            r"pd.Series.do_something2\(\): Expected 'arg2' to be type Float. Got:",
            id="float_error_message",
        ),
        pytest.param(
            ("hello", 0.0),
            {"arg3": 23, "arg4": 2},
            r"pd.Series.do_something2\(\): Expected 'arg3' to be type Boolean. Got:",
            id="bool_error_message",
        ),
        pytest.param(
            ("hello", 0.0),
            {"arg3": True, "arg4": 2.71},
            r"pd.Series.do_something2\(\): Expected 'arg4' to be type Integer. Got:",
            id="int_error_message",
        ),
    ],
)
def test_primative_type_argument_checkers(args, kwargs, expected_err_msg, use_constant):
    """Test argument checkers check method"""

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.SeriesType,
        "do_something2",
        path="pd.Series.do_something2",
        unsupported_args=[],
        description="This method does something",
        method_args_checker=OverloadArgumentsChecker(
            [
                StringScalarArgumentChecker("arg1"),
                FloatScalarArgumentChecker("arg2"),
                BooleanScalarArgumentChecker("arg3"),
                IntegerScalarArgumentChecker("arg4"),
            ]
        ),
    )
    def overload_series_do_something2(S, arg1, arg2, arg3=False, arg4=3):
        def impl(S, arg1, arg2, arg3=False, arg4=3):
            return 1

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_do_something2",
        overload_series_do_something2,
    )

    try:
        locals = {}
        func_text = "def test_impl(S, arg1, arg2, arg3=False, arg4=3):\n"
        if use_constant:
            func_text += f"  return S.do_something2({arguments_str}, {kwargs_str})\n"
        else:
            func_text += "  return S.do_something2(arg1, arg2, arg3=arg3, arg4=arg4)\n"

        exec(func_text, {"nan": np.nan}, locals)
        test_impl = locals["test_impl"]

        S = pd.Series([1, 2, 3])

        if expected_err_msg is None:
            # expect to compile successfully
            bodo.jit(test_impl)(S, *args, **kwargs)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S, *args, **kwargs)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_do_something2")


@run_rank0
@pytest.mark.parametrize(
    "use_constant",
    [
        pytest.param(True, id="use_constant"),
        pytest.param(False, id="no_constant"),
    ],
)
@pytest.mark.parametrize(
    "other, fill_value, expected_err_msg",
    [
        pytest.param(
            pd.Index([1.0, np.nan, 3.0, 4.0, 5.0]),
            None,
            None,
            id="index",
        ),
        pytest.param(
            pd.RangeIndex(0, 5),
            None,
            None,
            id="range_index",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            None,
            None,
            id="list",
        ),
        pytest.param(
            (1, 2, 3, 4, 5),
            None,
            None,
            id="tuple",
            marks=pytest.mark.skip("TODO fix tuple-literal coersion to array"),
        ),
        pytest.param(
            np.array([1, 2, 3, 4, 5]),
            None,
            None,
            id="array",
        ),
        pytest.param(
            pd.Series([True, False, True, False, True]),
            False,
            None,
            id="series",
        ),
        pytest.param(
            6.0,
            None,
            None,
            id="other_scalar",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
            None,
            r"pd.Series.sub2\(\): Expected 'other' to be a numeric scalar or Series, Index, Array, List or Tuple with numeric data. Got:",
            id="other_ndim_err",
        ),
        pytest.param(
            pd.Series(["hi", "goodbye", "why", "2", "3"]),
            None,
            r"pd.Series.sub2\(\): Expected 'other' to be a numeric scalar or Series, Index, Array, List or Tuple with numeric data. Got:",
            id="other_dtype_err",
        ),
        pytest.param(
            pd.Index([1, 2, 3, 4, 5]),
            "hello",
            r"pd.Series.sub2\(\): Expected 'fill_value' to be type Integer, Float, Boolean or None. Got:",
            id="scalar_dtype_err",
        ),
    ],
)
def test_numeric_series_argument_checkers(
    other, fill_value, expected_err_msg, use_constant
):
    """Verify that the numeric argument checkers for Series methods work as expected using Series.sub"""
    args = (other,)
    kwargs = {"fill_value": fill_value}

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.SeriesType,
        "sub2",
        path="pd.Series.sub2",
        unsupported_args=[],
        description="This method adds",
        method_args_checker=OverloadArgumentsChecker(
            [
                NumericSeriesBinOpChecker("other"),
                NumericScalarArgumentChecker("fill_value"),
            ]
        ),
    )
    def overload_series_sub2(S, other, fill_value=None):
        def impl(S, other, fill_value=None):
            return S.sub(other, fill_value=fill_value)

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_sub2",
        overload_series_sub2,
    )

    try:
        S = pd.Series([1, 2, 3, 4, 5])

        locals = {}
        func_text = "def test_impl(S, other, fill_value):\n"
        if use_constant:
            func_text += f"  return S.sub2({arguments_str}, {kwargs_str})\n"
        else:
            func_text += "  return S.sub2(other, fill_value=fill_value)\n"

        exec(func_text, {"np": np, "nan": np.nan, "pd": pd}, locals)
        test_impl = locals["test_impl"]

        if expected_err_msg is None:
            # expect to compile successfully
            bodo.jit(test_impl)(S, *args, **kwargs)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S, *args, **kwargs)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_sub2")
