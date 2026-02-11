"""Test array kernels for LEAD/LAG"""

import datetime
import string
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
    gen_nonascii_list,
    pytest_slow_unless_codegen,
)
from bodosql.kernels.lead_lag import lead_lag_seq

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize(
    "shift_amt",
    [
        pytest.param(1, id="positive_shift"),
        pytest.param(-3, id="negative_shift"),
        pytest.param(50, id="big_shift", marks=pytest.mark.slow),
        pytest.param(-20, id="big_negative_shift", marks=pytest.mark.slow),
        pytest.param(0, id="no_shift", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "use_default",
    [
        pytest.param(False, id="no_default", marks=pytest.mark.slow),
        pytest.param(True, id="with_default"),
    ],
)
@pytest.mark.parametrize(
    "in_col, default_fill_val, use_dict",
    [
        pytest.param(
            pd.Series(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]), dtype=np.int32),
            21,
            False,
            id="int32_numpy",
        ),
        pytest.param(
            pd.Series([0, 1, 2, 3, None, 4, 5, 6, 7, 8, 9], dtype=pd.Int32Dtype()),
            21,
            False,
            id="int32",
        ),
        pytest.param(
            pd.Series(
                [
                    0,
                    1223372036854775806,
                    -2223372036854775806,
                    3223372036854775806,
                    None,
                    -4223372036854775806,
                    5223372036854775806,
                    -6223372036854775806,
                    7223372036854775808,
                    8223372036854775806,
                    9223372036854775806,
                ],
                dtype=pd.Int64Dtype(),
            ),
            42,
            False,
            id="int64",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [0.0, 1.0, 2.0, 3.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                dtype=pd.Float64Dtype(),
            ),
            42.0,
            False,
            id="float64",
        ),
        pytest.param(
            pd.Series(
                [
                    Decimal("1.0"),
                    Decimal("-2.2"),
                    None,
                    Decimal("4.0"),
                    Decimal("3.14"),
                    Decimal("1024"),
                ]
            ),
            Decimal("999999.89"),
            False,
            id="decimal",
        ),
        pytest.param(
            pd.Series(["", "cool", "test", "string", None, "right", "here", "yeaaaah"]),
            string.ascii_lowercase,
            False,
            id="str",
        ),
        pytest.param(
            pd.Series(["", "cool", "test", "string", None, "right", "here", "yeaaaah"]),
            string.ascii_lowercase,
            True,
            id="str_dict",
        ),
        pytest.param(
            pd.Series(["", "cool", "test", "string", None, "right", "here", "yeaaaah"]),
            "cool",
            True,
            id="str_dict_default_present",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(gen_nonascii_list(7) + [None] + gen_nonascii_list(6)),
            "ボード🐍",
            False,
            id="str_non_ascii",
        ),
        pytest.param(
            pd.Series(gen_nonascii_list(7) + [None] + gen_nonascii_list(6)),
            "ボード🐍",
            True,
            id="str_non_ascii_dict",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    False,
                    True,
                    None,
                    True,
                    False,
                    True,
                ]
                * 5,
                dtype=pd.BooleanDtype(),
            ),
            True,
            False,
            id="boolean",
        ),
        pytest.param(
            np.array(
                [
                    False,
                    True,
                    True,
                    False,
                    True,
                ]
                * 5
            ),
            True,
            False,
            id="boolean_non_nullable",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.date(2020, 1, 1),
                    datetime.date(1999, 12, 31),
                    datetime.date(2000, 3, 4),
                    None,
                    datetime.date(2004, 5, 13),
                    datetime.date(1979, 1, 14),
                    datetime.date(2130, 12, 31),
                ],
            ),
            datetime.date(2000, 9, 9),
            False,
            id="date",
        ),
        pytest.param(
            pd.Series(
                [
                    bodo.types.Time(12, 30, 0),
                    bodo.types.Time(10, 59, 59, millisecond=250),
                    None,
                    bodo.types.Time(nanosecond=1234567890),
                    bodo.types.Time(23, 11),
                ],
            ),
            bodo.types.Time(16, 21),
            False,
            id="time",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("1999-12-31"),
                    pd.Timestamp("2000-03-04"),
                    None,
                    pd.Timestamp("2004-05-13"),
                    pd.Timestamp("1979-01-14"),
                    pd.Timestamp("2130-12-31"),
                ],
                dtype="datetime64[ns]",
            ),
            pd.Timestamp("2000-10-29"),
            False,
            id="timestamp",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp("2020-01-01", tz="Asia/Tokyo"),
                    pd.Timestamp("1999-12-31", tz="Asia/Tokyo"),
                    pd.Timestamp("2000-03-04", tz="Asia/Tokyo"),
                    None,
                    pd.Timestamp("2004-05-13", tz="Asia/Tokyo"),
                    pd.Timestamp("1979-01-14", tz="Asia/Tokyo"),
                    pd.Timestamp("2130-12-31", tz="Asia/Tokyo"),
                ],
                dtype="datetime64[ns, Asia/Tokyo]",
            ),
            pd.Timestamp("2000-10-29 20:00:00.00", tz="Asia/Tokyo"),
            False,
            id="timestamp_ltz",
        ),
        pytest.param(
            pd.Series(
                [
                    pd.Timedelta(days=128),
                    None,
                    pd.Timedelta(days=400, hours=12),
                    pd.Timedelta(days=-1, seconds=1),
                    None,
                    pd.Timedelta(minutes=30),
                ],
                dtype="timedelta64[ns]",
            ),
            pd.Timedelta(days=3),
            False,
            id="timedelta",
        ),
        pytest.param(
            pd.Series(
                [b"cool", b"test", b"", b"string", None, b"right", b"here", b"yeaaaah"]
            ),
            b"\0very",
            False,
            id="binary",
        ),
        pytest.param(
            pd.Series(
                [b"cool", b"test", b"", b"string", None, b"right", b"here", b"yeaaaah"]
            ),
            b"\0very",
            True,
            id="binary_dict",
        ),
    ],
)
def test_lead_lag_seq(
    in_col, shift_amt, default_fill_val, use_default, use_dict, memory_leak_check
):
    def impl(in_col, shift_amt, default_fill_val):
        return pd.Series(lead_lag_seq(in_col, shift_amt, default_fill_val))

    if use_default:
        default = default_fill_val
    else:
        default = None

    # workaround for testing non-nullable types
    if in_col.dtype == np.int32:
        out = in_col.astype("Int32").shift(shift_amt, fill_value=default)
    elif in_col.dtype == np.bool_:
        out = pd.Series(in_col).shift(shift_amt, fill_value=default)
    else:
        out = in_col.shift(shift_amt, fill_value=default)

    check_func(
        impl,
        (in_col, shift_amt, default),
        py_output=out,
        only_seq=True,
        use_dict_encoded_strings=use_dict,  # Passing None doesn't test with dictionary since we're not passing any string arguments
    )


@pytest.mark.parametrize(
    "use_default",
    [pytest.param(True, id="with_default"), pytest.param(False, id="no_default")],
)
@pytest.mark.parametrize(
    "shift, answer, pattern_list",
    [
        pytest.param(
            -1, [0, 1, 1, 2, 2, -1], [None, 0, None, 1, None, 2], id="negative-pat1"
        ),
        pytest.param(
            -1,
            [1, 1, 1, 2, 3, 4, 4, -1, -1],
            [0, None, None, 1, 2, 3, None, 4, None],
            id="negative-pat2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            1,
            [-1, -1, 0, 0, 1, 1],
            [None, 0, None, 1, None, 2],
            id="positive-pat1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            2,
            [-1, -1, -1, -1, 0, 1, 2, 2, 3],
            [0, None, None, 1, 2, 3, None, 4, None],
            id="positive-pat2",
        ),
        pytest.param(
            0,
            [None, 0, None, 1, None, 2],
            [None, 0, None, 1, None, 2],
            id="zero-pat1",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "values, dtype, default",
    [
        pytest.param([0, 1, 2, 3, 4], pd.Int32Dtype(), -1, id="int32"),
        pytest.param(
            [10.0, 3.1415926, -64.0, 125.0, 2.718281828], np.float64, 0.0, id="float64"
        ),
        pytest.param(["", "beta", "gamma", "delta", "epsilon"], None, "", id="string"),
        pytest.param(
            [b"romeo", b"juliet", b"othello", b"hamlet", b"viola"],
            None,
            b"",
            id="binary",
        ),
        pytest.param(
            [True, False, True, False, True], pd.BooleanDtype(), True, id="boolean"
        ),
        pytest.param(
            [
                datetime.date.fromordinal(i)
                for i in [736879, 737729, 733133, 680082, 688783]
            ],
            None,
            datetime.date(1999, 12, 31),
            id="date",
        ),
        pytest.param(
            [bodo.types.Time(second=i) for i in [45000, 86399, 21600, 1, 1020]],
            None,
            bodo.types.Time(0, 0, 0),
            id="time",
        ),
        pytest.param(
            [
                pd.Timestamp("2023-1-1") + pd.Timedelta(hours=i)
                for i in [0, -10000, 20000, -30000, 40000]
            ],
            None,
            pd.Timestamp("1999-1-1"),
            id="naive_timestamp",
        ),
        pytest.param(
            [
                pd.Timestamp("2023-1-1", tz="US/Pacific") + pd.Timedelta(hours=i)
                for i in [0, 10000, -20000, 30000, -40000]
            ],
            None,
            pd.Timestamp("1999-1-1", tz="US/Pacific"),
            id="tz_timestamp",
        ),
    ],
)
def test_lead_lag_seq_ignore_nulls(
    values, dtype, shift, pattern_list, default, use_default, answer, memory_leak_check
):
    """Tests null_ignoring_shift on multiple types with various shifts
    and default values via the following pattern:

     - Values: a list of five distinct values of the type being tested
     - Dtype: the datatype to be used when converting to a Series
     - Default: the default value to provide to the kernel
     - Shift: how much to shift by
     - Answer: a list of length 9 that provides an output pattern corresponding
       to the input pattern shifted by the amount

     The pattern (defined below) is [0, None, None, 1, 2, 3, None, 4, None].
     Suppose our values were ["a", "b", "c", "d", "e"]. Then this would
     construct the following list:

         ["a", None, None, "b", "c", "d", None, "e", None]

     If we shifted by -1 with a default of "" we would get the following:

         ["b", "b", "b", "c", "d", "e", "e", "", ""]

     So the output pattern to replicate this would be as follows:

         [1, 1, 1, 2, 3, 4, 4, -1, -1]

     Where the numbers 0 to 4 represent which value from the original list
     of 5 values is used, and -1 indicates using the default.

     Note: if use_default is False, then None is used instead of the default
     value provided
    """

    if not use_default:
        default = None

    input_list = []
    output_list = []
    for i in range(len(pattern_list)):
        if pattern_list[i] is None:
            input_list.append(None)
        else:
            input_list.append(values[pattern_list[i]])
        if answer[i] is None:
            output_list.append(None)
        elif answer[i] == -1:
            output_list.append(default)
        else:
            output_list.append(values[answer[i]])

    def impl(in_col, shift_amt, default_fill_val):
        return pd.Series(
            lead_lag_seq(in_col, shift_amt, default_fill_val, ignore_nulls=True)
        )

    check_func(
        impl,
        (pd.Series(input_list, dtype=dtype), shift, default),
        py_output=pd.Series(output_list, dtype=dtype),
        check_dtype=False,
        reset_index=True,
        # For now, only works sequentially because it can only be used inside
        # of a Window function with a partition
        only_seq=True,
    )
