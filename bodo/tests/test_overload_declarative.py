from __future__ import annotations

from types import NoneType

import numpy as np
import pandas as pd
import pytest
from numba.core import types  # noqa TID253

import bodo
from bodo.tests.utils import check_func

pytestmark = pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="setattr doesn't change worker state",
)


def _val_to_string(val):
    """
    Embed `val` as it's constructor in a string.
    `val` can be a Series, Index, RangeIndex, numpy array,
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
            r"pd.Series.do_something\(\): 'arg2' must be a compile time constant and must be None, 1, 0 or \"10\"",
            id="bad_const_value",
        ),
        pytest.param(
            (True, 0),
            {"arg3": "hello"},
            r"pd.Series.do_something\(\): 'arg3' must be a constant Integer, Tuple or None",
            id="bad_const_type",
        ),
    ],
)
def test_literal_argument_checkers(args, kwargs, expected_err_msg):
    """Test argument checkers check method"""
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        ConstantArgumentChecker,
        NDistinctValueArgumentChecker,
        OverloadArgumentsChecker,
    )
    from bodo.ir.declarative_templates import overload_method_declarative
    from bodo.utils.typing import BodoError

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.types.SeriesType,
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
            r"pd.Series.do_something2\(\): 'arg1' must be a String. Got:",
            id="str_error_message",
        ),
        pytest.param(
            ("hello", 0),
            {"arg3": True, "arg4": 2},
            r"pd.Series.do_something2\(\): 'arg2' must be a Float. Got:",
            id="float_error_message",
        ),
        pytest.param(
            ("hello", 0.0),
            {"arg3": 23, "arg4": 2},
            r"pd.Series.do_something2\(\): 'arg3' must be a Boolean. Got:",
            id="bool_error_message",
        ),
        pytest.param(
            ("hello", 0.0),
            {"arg3": True, "arg4": 2.71},
            r"pd.Series.do_something2\(\): 'arg4' must be a Integer. Got:",
            id="int_error_message",
        ),
    ],
)
def test_primative_type_argument_checkers(args, kwargs, expected_err_msg, use_constant):
    """Test argument checkers check method"""
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        BooleanScalarArgumentChecker,
        FloatScalarArgumentChecker,
        IntegerScalarArgumentChecker,
        OverloadArgumentsChecker,
        StringScalarArgumentChecker,
    )
    from bodo.ir.declarative_templates import overload_method_declarative
    from bodo.utils.typing import BodoError

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.types.SeriesType,
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
            r"pd.Series.sub2\(\): 'other' must be a numeric scalar or Series, Index, Array, List or Tuple with numeric data",
            id="other_ndim_err",
        ),
        pytest.param(
            pd.Series(["hi", "goodbye", "why", "2", "3"]),
            None,
            r"pd.Series.sub2\(\): 'other' must be a numeric scalar or Series, Index, Array, List or Tuple with numeric data",
            id="other_dtype_err",
        ),
        pytest.param(
            pd.Index([1, 2, 3, 4, 5]),
            "hello",
            r"pd.Series.sub2\(\): 'fill_value' must be a Float, Integer or Boolean, or it can be None",
            id="scalar_dtype_err",
        ),
    ],
)
def test_numeric_series_argument_checkers(
    other, fill_value, expected_err_msg, use_constant
):
    """Verify that the numeric argument checkers for Series methods work as expected using Series.sub"""
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        NumericScalarArgumentChecker,
        NumericSeriesBinOpChecker,
        OptionalArgumentChecker,
        OverloadArgumentsChecker,
    )
    from bodo.ir.declarative_templates import overload_method_declarative
    from bodo.utils.typing import BodoError

    args = (other,)
    kwargs = {"fill_value": fill_value}

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.types.SeriesType,
        "sub2",
        path="pd.Series.sub2",
        unsupported_args=[],
        description="This method adds",
        method_args_checker=OverloadArgumentsChecker(
            [
                NumericSeriesBinOpChecker("other"),
                OptionalArgumentChecker(NumericScalarArgumentChecker("fill_value")),
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


@pytest.mark.parametrize(
    "S, arg1, arg2, expected_err_msg",
    [
        pytest.param(
            pd.Series([1, 2, 3, 4]),
            pd.Series(["1", "2", "3", "4"]),
            pd.Series([np.datetime64("2007-01-01T03:30")], dtype="datetime64[ns]"),
            None,
            id="no_errors",
        ),
        pytest.param(
            pd.Series(["this", "is", "a", "string"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.Series([np.datetime64("2007-01-01T03:30")], dtype="datetime64[ns]"),
            "'self' must be a Series of Float or Integer data. Got:",
            id="self_error",
        ),
        pytest.param(
            pd.Series([1, 2, 3, 4]),
            pd.Series(["1", "2", "3", "4"]),
            pd.Series(
                [np.timedelta64(i, "h") for i in range(5)], dtype="timedelta64[ns]"
            ),
            "'arg2' must be a Series of datetime64 data. Got:",
            id="datetimelike_series_error",
        ),
        pytest.param(
            pd.Series([1, 2, 3, 4]),
            pd.Series([[1, 2, 3], [2, 3, 4]]),
            pd.Series([np.datetime64("2007-01-01T03:30")], dtype="datetime64[ns]"),
            "'arg1' must be a Series of String data. Got:",
            id="string_series_error",
        ),
    ],
)
def test_series_self_argument_checkers(S, arg1, arg2, expected_err_msg):
    """Verify that the numeric argument checkers for Series methods work as expected using Series.sub"""
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        DatetimeLikeSeriesArgumentChecker,
        NumericSeriesArgumentChecker,
        OverloadArgumentsChecker,
        StringSeriesArgumentChecker,
    )
    from bodo.ir.declarative_templates import overload_method_declarative
    from bodo.utils.typing import BodoError

    @overload_method_declarative(
        bodo.types.SeriesType,
        "do_something3",
        path="pd.Series.do_something3",
        unsupported_args=[],
        description="This method adds",
        method_args_checker=OverloadArgumentsChecker(
            [
                NumericSeriesArgumentChecker("S", is_self=True),
                StringSeriesArgumentChecker("arg1"),
                DatetimeLikeSeriesArgumentChecker("arg2", type="datetime"),
            ]
        ),
    )
    def overload_series_do_something3(S, arg1, arg2):
        def impl(S, arg1, arg2):
            return 1

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_do_something3",
        overload_series_do_something3,
    )

    try:

        def test_impl(S, arg1, arg2):
            return S.do_something3(arg1, arg2)

        def test_impl2(S, arg1, arg2):
            return S.do_something3(arg1.str, arg2.dt)

        if expected_err_msg is None:
            # expect to compile successfully
            check_func(test_impl, (S, arg1, arg2), py_output=1)
            check_func(test_impl2, (S, arg1, arg2), py_output=1)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S, arg1, arg2)
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl2)(S, arg1, arg2)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_do_something3")


@pytest.mark.parametrize(
    "S, arg1, expected_err_msg",
    [
        pytest.param(pd.Series([5, 4, 3, 2, 1]), "int", None, id="no_errors"),
        pytest.param(
            pd.Series([5, 4, 3, 2, 1]),
            "float",
            "only accepts the constant value 'int' for Series of integer data",
            id="float_error",
        ),
        pytest.param(
            pd.Series([5.0, 4.0, 3.0, 2.0, 1.0]),
            "int",
            "only accepts the constant value 'float' for Series of float data",
            id="int_error",
        ),
    ],
)
def test_series_generic_argument_checkers(S, arg1, expected_err_msg):
    """
    Test that generic argument checker works. In this example, argument arg1 must be
    of of ('int', 'float'), S can be either a Series of integers or a Series of floats,
    and if S is a Series of ints then arg1 must be 'int' and if it is a Series of floats
    then arg1 must be 'float'
    """
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        GenericArgumentChecker,
        NumericSeriesArgumentChecker,
        OverloadArgumentsChecker,
    )
    from bodo.ir.declarative_templates import overload_method_declarative
    from bodo.utils.typing import BodoError, is_overload_const_str_equal

    def check_fn(context, arg_typ):
        series_type = context["self"]
        if isinstance(
            series_type.dtype, types.Integer
        ) and not is_overload_const_str_equal(arg_typ, "int"):
            return (
                arg_typ,
                "only accepts the constant value 'int' for Series of integer data",
            )
        elif isinstance(
            series_type.dtype, types.Float
        ) and not is_overload_const_str_equal(arg_typ, "float"):
            return (
                arg_typ,
                "only accepts the constant value 'float' for Series of float data",
            )
        return arg_typ, None

    def explain_fn(context):
        return 'only supports constant value "int" for Series of integer data and "float" for Series of float data.'

    @overload_method_declarative(
        bodo.types.SeriesType,
        "do_something4",
        path="pd.Series.do_something4",
        unsupported_args=[],
        description="This method adds",
        method_args_checker=OverloadArgumentsChecker(
            [
                NumericSeriesArgumentChecker("S", is_self=True),
                GenericArgumentChecker("arg1", check_fn, explain_fn),
            ]
        ),
    )
    def overload_series_do_something4(S, arg1):
        def impl(S, arg1):
            return 1

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_do_something4",
        overload_series_do_something4,
    )

    try:

        def test_impl(S):
            return S.do_something4(arg1)

        if expected_err_msg is None:
            # expect to compile successfully
            check_func(test_impl, (S,), py_output=1)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_do_something4")


@pytest.mark.parametrize(
    "S, expected_err_msg",
    [
        pytest.param(pd.Series(["1", "2", "3", "4", "5"]), None, id="no_errors"),
        pytest.param(
            pd.Series([1, 2, 3, 4]),
            "input must be a Series of String data",
            id="error",
        ),
    ],
)
def test_overload_attr(S, expected_err_msg):
    import bodo.decorators  # isort:skip # noqa
    import bodo.hiframes.series_impl
    from bodo.ir.argument_checkers import (
        OverloadAttributeChecker,
        StringSeriesArgumentChecker,
    )
    from bodo.ir.declarative_templates import overload_attribute_declarative
    from bodo.utils.typing import BodoError

    @overload_attribute_declarative(
        bodo.types.SeriesType,
        "some_attr",
        "pd.Series.some_attr",
        description="this is an attribute",
        arg_checker=OverloadAttributeChecker(StringSeriesArgumentChecker("S")),
        inline="always",
    )
    def overload_series_attr(S):
        def impl(S):
            return S.str.casefold()

        return impl

    setattr(
        bodo.hiframes.series_impl,
        "overload_series_attr",
        overload_series_attr,
    )

    try:

        def test_impl(S):
            return S.some_attr

        if expected_err_msg is None:
            # expect to compile successfully
            bodo.jit(test_impl)(S)
        else:
            with pytest.raises(BodoError, match=expected_err_msg):
                bodo.jit(test_impl)(S)

    finally:
        delattr(bodo.hiframes.series_impl, "overload_series_attr")
