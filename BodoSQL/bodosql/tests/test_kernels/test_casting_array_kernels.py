"""Test Bodo's array kernel utilities for casting"""

import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_string_dtype

import bodo
import bodosql
from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.utils import check_func, pytest_slow_unless_codegen
from bodosql.kernels.array_kernel_utils import vectorized_sol
from bodosql.tests.test_kernels.test_snowflake_conversion_array_kernels import (
    str_to_bool,
)

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(
            (pd.Series([1, 2, 0, 4, 5]),),
            id="int_array",
        ),
        pytest.param(
            (pd.Series([32.3, None, -2, 4.4, 5]),),
            id="small_float_array",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        32123123.32313423,
                        None,
                        -2234234254,
                        np.nan,
                        50782748792,
                        12379739850550389709,
                        -8454757700450211157,
                    ]
                ),
            ),
            id="large_float_array",
        ),
        pytest.param(
            (pd.Series([1, None, 3, None, 5], dtype="Int32"),),
            id="nullable_int32_array",
        ),
        pytest.param(
            (pd.Series([2**60, None, 3, None, 5], dtype="Int64"),),
            id="nullable_int64_array",
        ),
        pytest.param(
            (pd.Series([2**-150, 2**130, (2 - 2**-24) * (2**-128)], dtype="float64"),),
            id="float64_precision_array",
        ),
        pytest.param((pd.Series(["1", "2", "3", "4", "5"]),), id="str_int_array"),
        pytest.param(
            (
                pd.Series(
                    [
                        "1.1",
                        "3.5",
                        "nan",
                        "4.5",
                        "-inf",
                        "-8454757700450211157",
                        "12379739850550389709",
                    ]
                ),
            ),
            id="str_float_array",
        ),
        pytest.param(
            (pd.Series([True, False, True, False, True]),),
            id="bool_array",
        ),
        pytest.param(
            (pd.Series([True, False, None, False, True], dtype="boolean"),),
            id="nullable_bool_array",
        ),
        pytest.param(
            (11,),
            id="int32_scalar",
        ),
        pytest.param(
            (2**60,),
            id="int64_scalar",
        ),
        pytest.param(
            (14.0,),
            id="float32_scalar",
        ),
        pytest.param(
            (np.float64(2**130),),
            id="float64_scalar",
        ),
        pytest.param(
            ("52.8523",),
            id="str_float_scalar",
        ),
        pytest.param(
            ("-inf",),
            id="str_inf_scalar",
        ),
        pytest.param(
            ("-234",),
            id="str_int_scalar",
        ),
        pytest.param(
            (True,),
            id="bool_scalar",
        ),
    ]
)
def numeric_arrays(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    pd.date_range("01-01-2020", "01-01-2022", 10, None, unit="ns"),
                    dtype="datetime64[ns]",
                ),
            ),
            id="dt_array",
        ),
        pytest.param(
            (pd.Series(pd.timedelta_range("1 day", "10 day", 10, None, unit="ns")),),
            id="td_array",
        ),
        pytest.param(
            (pd.to_datetime(2**60),),
            id="dt_scalar",
        ),
        pytest.param(
            (pd.to_timedelta(2**50),),
            id="td_scalar",
        ),
        pytest.param(
            (pd.to_timedelta(-(2**40)),),
            id="neg_td_scalar",
        ),
    ]
)
def time_arrays(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [
                        "2020-01-01",
                        None,
                        "02-05-1980 12:20:20",
                        "08-20-1985",
                        "08-20-1985 02:02:02.0202",
                        "20-08-1985",
                    ]
                ),
            ),
            id="str_date_array",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        7223372036854.775,
                        722337203.6854775,
                        922337203654775,
                        0,
                        -231902383.23,
                        None,
                    ]
                ),
            ),
            id="num_date_array",
        ),
    ]
)
def dt_arrays(request):
    return request.param


def test_cast_float64(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_float64(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_float64(arr)

    # Simulates casting to float64 on a single row
    def float64_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return np.float64(elem)

    answer = vectorized_sol(args, float64_scalar_fn, "Float64")
    arg = numeric_arrays[0]
    if isinstance(arg, pd.Series) and isinstance(arg.iloc[0], str):
        # The str example tests casting "NaN" -> np.nan. If we set
        # dtype="Float64" then Pandas will cast this to pd.NA.
        # As a result we just compare with float64
        answer = answer.astype(np.float64)

    check_func(
        impl,
        args,
        py_output=answer,
        reset_index=True,
        convert_to_nullable_float=False,
    )


def test_cast_float32(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_float32(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_float32(arr)

    # Simulates casting to float32 on a single row
    def float32_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return np.float32(elem)

    answer = vectorized_sol(args, float32_scalar_fn, "Float32")
    arg = numeric_arrays[0]
    if isinstance(arg, pd.Series) and isinstance(arg.iloc[0], str):
        # The str example tests casting "NaN" -> np.nan. If we set
        # dtype="Float32" then Pandas will cast this to pd.NA.
        # As a result we just compare with float64
        answer = answer.astype(np.float32)
    check_func(
        impl,
        args,
        py_output=answer,
        reset_index=True,
        convert_to_nullable_float=False,
    )


def _round_float(f):
    return np.int64(np.fix(f + 0.5 * np.sign(f)))


def test_cast_int64(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_int64(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_int64(arr)

    # Simulates casting to float64 on a single row
    def int64_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elem = np.float64(elem)
        if (
            pd.isna(elem)
            or np.isinf(elem)
            or elem < np.iinfo(np.int64).min
            or elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            # additionally, we must round half away from zero rounding rather than
            # normal banker's rounding in Python / Numpy
            return _round_float(elem)

    answer = vectorized_sol(args, int64_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int64")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int32(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_int32(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_int32(arr)

    # Simulates casting to float64 on a single row
    def int32_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elem = np.float64(elem)
        if (
            pd.isna(elem)
            or np.isinf(elem)
            or elem < np.iinfo(np.int64).min
            or elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            return np.int32(_round_float(elem))

    answer = vectorized_sol(args, int32_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int32")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int16(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_int16(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_int16(arr)

    # Simulates casting to float64 on a single row
    def int16_scalar_fn(elem):
        if pd.isna(elem):
            return None
        f_elem = np.float64(elem)
        if (
            pd.isna(f_elem)
            or np.isinf(f_elem)
            or f_elem < np.iinfo(np.int64).min
            or f_elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            if f_elem.is_integer():
                return np.int64(elem).astype(np.int16)
            else:
                return np.int16(_round_float(f_elem))

    answer = vectorized_sol(args, int16_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int16")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_int8(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_int8(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_int8(arr)

    # Simulates casting to float64 on a single row
    def int8_scalar_fn(elem):
        if pd.isna(elem):
            return None
        f_elem = np.float64(elem)
        if (
            pd.isna(f_elem)
            or np.isinf(f_elem)
            or f_elem < np.iinfo(np.int64).min
            or f_elem > np.iinfo(np.int64).max
        ):
            return None
        else:
            # we normally must calculate the underflow/overflow behavior for ints
            # however casting to float64 first will handle this for us
            if f_elem.is_integer():
                return np.int64(elem).astype(np.int8)
            else:
                return np.int8(_round_float(f_elem))

    answer = vectorized_sol(args, int8_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.astype("Int8")
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=True,
        reset_index=True,
    )


def test_cast_boolean(numeric_arrays, memory_leak_check):
    args = numeric_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_boolean(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_boolean(arr)

    # Simulates casting to float64 on a single row
    if is_string_dtype(args[0]) or isinstance(args[0], str):
        to_bool_scalar_fn = (
            lambda x: None if pd.isna(str_to_bool(x)) else str_to_bool(x)
        )
    elif is_float_dtype(args[0]) or isinstance(args[0], float):
        to_bool_scalar_fn = lambda x: None if np.isnan(x) or np.isinf(x) else bool(x)
    else:
        to_bool_scalar_fn = lambda x: None if pd.isna(x) else bool(x)

    answer = vectorized_sol(args, to_bool_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        bodo.types.Time(1, 2, 3, 4, 5, 6),
                        bodo.types.Time(1, 52, 33, 443, 534, 632),
                        None,
                    ]
                    * 4
                ),
            ),
            id="Time",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        bytes.fromhex("a3b2"),
                        bytes.fromhex("deadbeef"),
                        bytes(0),
                        None,
                    ]
                    * 4
                ),
            ),
            id="Bytes",
        ),
    ],
)
def test_cast_char_other(args, memory_leak_check):
    def impl(arr):
        return pd.Series(bodosql.kernels.cast_char(arr))

    def char_scalar_fn(elem):
        if pd.isna(elem):
            return None
        elif isinstance(elem, (bodo.types.Time, bodo.types.TimeType)):
            # Using Snowflake's default TIME format: HH:MM:SS
            return f"{elem.hour:02}:{elem.minute:02}:{elem.second:02}"
        elif isinstance(elem, bytes):
            return elem.hex()
        else:
            return str(elem)

    answer = vectorized_sol(args, char_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


def test_cast_char_nums(numeric_arrays, memory_leak_check):
    def impl(arr):
        return pd.Series(bodosql.kernels.cast_char(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(numeric_arrays[0], (int, float, str)):
        impl = lambda arr: bodosql.kernels.cast_char(arr)

    # Simulates casting to string on a single row
    def to_char_scalar_fn(x):
        if pd.isna(x):
            return None
        elif isinstance(x, float):
            return f"{x:f}"
        elif isinstance(x, (bool, np.bool_)):
            return str(x).lower()
        else:
            return str(x)

    answer = vectorized_sol(numeric_arrays, to_char_scalar_fn, None)
    check_func(
        impl,
        numeric_arrays,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


def test_cast_char_times(time_arrays, memory_leak_check):
    args = time_arrays

    def impl(arr):
        return pd.Series(bodosql.kernels.cast_char(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(args[0], (pd.Timestamp, pd.Timedelta)):
        impl = lambda arr: bodosql.kernels.cast_char(arr)

    # Simulates casting to float64 on a single row
    def to_char_scalar_fn(x):
        if pd.isna(x):
            return None
        elif isinstance(x, pd.Timedelta):
            return str(np.timedelta64(x.value, "ns"))
        else:
            return str(x)

    answer = vectorized_sol(args, to_char_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (pd.Series(["1 d", "2 m", "4 ns", "-1 h", None, "0"]),),
            id="string",
        ),
        pytest.param(
            (pd.Series([1, 102039209310213, 0, -21231123, None]),),
            id="int",
        ),
    ],
)
def test_cast_interval(args, memory_leak_check):
    def impl(arr):
        return pd.Series(bodosql.kernels.cast_interval(arr))

    def to_interval_scalar_fn(x):
        if pd.isna(x):
            return None
        else:
            return pd.Timedelta(x)

    answer = vectorized_sol(args, to_interval_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "ts_val",
    [
        pytest.param(
            pd.Timestamp("2022-11-05 23:13:12.242"),
            id="timestamp-scalar",
        ),
        pytest.param(
            pd.Series(
                [None] * 3
                + list(
                    pd.date_range("2022-1-1", periods=21, freq="40D5h4s", unit="ns")
                ),
                dtype="datetime64[ns]",
            ).values,
            id="timestamp-vector",
        ),
    ],
)
def test_cast_tz_naive_to_tz_aware(ts_val, representative_tz, memory_leak_check):
    tz_literal = representative_tz

    def impl(ts_val):
        return pd.Series(bodosql.kernels.cast_tz_naive_to_tz_aware(ts_val, tz_literal))

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_val, pd.Timestamp):
        impl = lambda ts_val: bodosql.kernels.cast_tz_naive_to_tz_aware(
            ts_val, tz_literal
        )

    def localize_scalar_fn(ts_val):
        if pd.isna(ts_val):
            return None
        else:
            return pd.Timestamp(ts_val).tz_localize(tz_literal)

    answer = vectorized_sol((ts_val,), localize_scalar_fn, None)
    check_func(
        impl,
        (ts_val,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "date_val",
    [
        pytest.param(
            datetime.date(2023, 4, 10),
            id="date-scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.date(1989, 10, 4),
                    datetime.date(2012, 12, 31),
                    None,
                    datetime.date(2023, 1, 1),
                    datetime.date(2002, 5, 13),
                ]
                * 4
            ),
            id="date-vector",
        ),
    ],
)
@pytest.mark.slow
def test_cast_date_to_tz_aware(date_val, representative_tz, memory_leak_check):
    """
    Tests cast_date_to_tz_aware kernel
    """
    tz_literal = representative_tz

    def impl(date_val):
        return pd.Series(bodosql.kernels.cast_date_to_tz_aware(date_val, tz_literal))

    # avoid pd.Series() conversion for scalar output
    if isinstance(date_val, datetime.date):
        impl = lambda date_val: bodosql.kernels.cast_date_to_tz_aware(
            date_val, tz_literal
        )

    def localize_scalar_fn(date_val):
        if pd.isna(date_val):
            return None
        else:
            return pd.Timestamp(date_val).tz_localize(tz_literal)

    answer = vectorized_sol((date_val,), localize_scalar_fn, None)
    check_func(
        impl,
        (date_val,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_cast_date_to_tz_aware_non_literal_tz_error_handling(memory_leak_check):
    """
    Tests cast_date_to_tz_aware kernel can throw error
    correctly when tz input is not a string literal
    """
    date_val = pd.Series(
        [
            datetime.date(1989, 10, 4),
            datetime.date(2012, 12, 31),
            None,
            datetime.date(2023, 1, 1),
            datetime.date(2002, 5, 13),
        ]
        * 4
    )

    tz = pd.Series(
        [
            "UTC",
            "US/Pacific",
            None,
            "Europe/Berlin",
            "Africa/Casablanca",
        ]
        * 4
    )

    def impl(date_val):
        return pd.Series(bodosql.kernels.cast_date_to_tz_aware(date_val, tz))

    with pytest.raises(
        Exception,
        match="'tz' must be a literal value",
    ):
        check_func(
            impl,
            (date_val,),
            py_output=pd.Series(),
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "ts_val",
    [
        pytest.param(
            pd.Timestamp("2022-11-05 23:13:12.242", tz="Australia/Lord_Howe"),
            id="scalar",
        ),
        pytest.param(
            pd.Series(
                [None] * 3
                + list(
                    pd.date_range(
                        "2022-1-1",
                        periods=21,
                        freq="40D5h4s",
                        tz="US/Pacific",
                        unit="ns",
                    )
                ),
                dtype="datetime64[ns, US/Pacific]",
            ).array,
            id="vector",
        ),
    ],
)
def test_cast_tz_aware_to_tz_naive(ts_val, memory_leak_check):
    def impl(ts_val):
        return pd.Series(bodosql.kernels.cast_tz_aware_to_tz_naive(ts_val))

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_val, pd.Timestamp):
        impl = lambda ts_val: bodosql.kernels.cast_tz_aware_to_tz_naive(ts_val)

    def generate_localize_scalar_fn():
        def localize_scalar_fn(ts_val):
            if pd.isna(ts_val):
                return None
            else:
                ts = ts_val.tz_localize(None)
                return ts

        return localize_scalar_fn

    answer = vectorized_sol((ts_val,), generate_localize_scalar_fn(), None)
    check_func(
        impl,
        (ts_val,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "ts_val",
    [
        pytest.param(
            "2022-11-05 23:13:12.242",
            id="scalar",
        ),
        pytest.param(
            pd.Series(
                [None] * 3
                + list(
                    pd.date_range("2022-1-1", periods=21, freq="40D5h4s", unit="ns")
                    .to_series()
                    .astype("str")
                )
            ).values,
            id="vector",
        ),
    ],
)
def test_cast_str_to_tz_aware(ts_val, representative_tz, memory_leak_check):
    tz_literal = representative_tz

    def impl(ts_val):
        return pd.Series(bodosql.kernels.cast_str_to_tz_aware(ts_val, tz_literal))

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_val, str):
        impl = lambda ts_val: bodosql.kernels.cast_str_to_tz_aware(ts_val, tz_literal)

    def localize_scalar_fn(ts_val):
        if pd.isna(ts_val):
            return None
        else:
            return pd.Timestamp(ts_val).tz_localize(tz_literal)

    answer = vectorized_sol((ts_val,), localize_scalar_fn, None)
    check_func(
        impl,
        (ts_val,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_cast_float_opt(memory_leak_check):
    def impl(a, b, flag0, flag1):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        return (
            bodosql.kernels.cast_float32(arg0),
            bodosql.kernels.cast_float64(arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                np.float32(12) if flag0 else None,
                np.float64(12) if flag1 else None,
            )
            check_func(
                impl,
                (12, 12, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_cast_int_opt(memory_leak_check):
    def impl(a, b, c, d, flag0, flag1, flag2, flag3):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        arg2 = c if flag2 else None
        arg3 = d if flag3 else None
        return (
            bodosql.kernels.cast_int8(arg0),
            bodosql.kernels.cast_int16(arg1),
            bodosql.kernels.cast_int32(arg2),
            bodosql.kernels.cast_int64(arg3),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    answer = (
                        np.int8(-120) if flag0 else None,
                        np.int16(12) if flag1 else None,
                        np.int32(_round_float(140.5)) if flag2 else None,
                        np.int64(12) if flag3 else None,
                    )
                    check_func(
                        impl,
                        (-120, 12, 141, 12, flag0, flag1, flag2, flag3),
                        py_output=answer,
                    )


@pytest.mark.slow
def test_cast_boolean_opt(memory_leak_check):
    def impl(a, b, flag0, flag1):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        return (
            bodosql.kernels.cast_boolean(arg0),
            bodosql.kernels.cast_boolean(arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (True if flag0 else None, False if flag1 else None)
            check_func(
                impl,
                ("t", 0, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_cast_char_opt(memory_leak_check):
    def impl(a, b, c, d, flag0, flag1, flag2, flag3):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        arg2 = c if flag2 else None
        arg3 = d if flag3 else None

        return (
            bodosql.kernels.cast_char(arg0),
            bodosql.kernels.cast_char(arg1),
            bodosql.kernels.cast_char(arg2),
            bodosql.kernels.cast_char(arg3),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    answer = (
                        "12" if flag0 else None,
                        "false" if flag1 else None,
                        "deadbeef" if flag2 else None,
                        "22.500000" if flag3 else None,
                    )
                    check_func(
                        impl,
                        (
                            12,
                            False,
                            bytes.fromhex("deadbeef"),
                            22.5,
                            flag0,
                            flag1,
                            flag2,
                            flag3,
                        ),
                        py_output=answer,
                    )


def test_cast_date_opt(memory_leak_check):
    def impl(a, flag):
        arg0 = a if flag else None
        return bodosql.kernels.cast_date(arg0)

    for flag in [True, False]:
        answer = pd.Timestamp("2020-01-01").normalize().date() if flag else None
        check_func(
            impl,
            ("2020-01-01", flag),
            py_output=answer,
        )


def test_cast_tz_naive_to_tz_aware_opt(memory_leak_check):
    tz_literal = "US/Pacific"

    def impl(a, flag):
        arg0 = a if flag else None
        return bodosql.kernels.cast_tz_naive_to_tz_aware(arg0, tz_literal)

    ts = pd.Timestamp("2022-11-06 00:52:31")
    for flag in [True, False]:
        answer = ts.tz_localize(tz_literal) if flag else None
        check_func(
            impl,
            (ts, flag),
            py_output=answer,
        )


def test_cast_tz_aware_to_tz_naive_opt(memory_leak_check):
    def impl(a, flag):
        arg0 = a if flag else None
        return bodosql.kernels.cast_tz_aware_to_tz_naive(arg0)

    ts = pd.Timestamp("2022-11-06 00:52:31", tz="US/Pacific")
    localized_ts = ts.tz_localize(None)
    for flag in [True, False]:
        answer = localized_ts if flag else None
        check_func(
            impl,
            (ts, flag),
            py_output=answer,
        )


def test_cast_str_to_tz_aware_opt(memory_leak_check):
    tz_literal = "US/Pacific"

    def impl(a, flag):
        arg0 = a if flag else None
        return bodosql.kernels.cast_str_to_tz_aware(arg0, tz_literal)

    ts_str = "2022-11-06 00:52:31"
    for flag in [True, False]:
        answer = pd.to_datetime(ts_str).tz_localize(tz_literal) if flag else None
        check_func(
            impl,
            (ts_str, flag),
            py_output=answer,
        )
