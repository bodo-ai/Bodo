from types import NoneType

import pandas as pd
import pytest

import bodo
import bodo.hiframes
import bodo.hiframes.series_impl
from bodo.ir.argument_checkers import (
    ConstantArgumentChecker,
    NDistinctValueArgumentChecker,
    OverloadArgumentsChecker,
)
from bodo.ir.declarative_templates import overload_method_declarative
from bodo.tests.utils import run_rank0
from bodo.utils.typing import BodoError


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
            r"pd.Series.do_something\(\): Expected arg2 to be a compile time constant and must be None, 1, 0 or \"10\". Got: ",
            id="bad_const_value",
        ),
        pytest.param(
            (True, 0),
            {"arg3": "hello"},
            r"pd.Series.do_something\(\): Expected arg3 to be a compile time constant and must have type integer, tuple or none. Got: ",
            id="bad_const_type",
        ),
    ],
)
def test_literal_argument_checkers(args, kwargs, expected_err_msg):
    """Test argument checkers check method"""

    def _val_to_string(val):
        return f'"{val}"' if isinstance(val, str) else str(val)

    arguments_str = ", ".join(map(_val_to_string, args))
    kwargs_str = ", ".join(
        f"{key}={_val_to_string(value)}" for key, value in kwargs.items()
    )

    @overload_method_declarative(
        bodo.SeriesType,
        "do_something",
        path_name="pd.Series.do_something",
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
