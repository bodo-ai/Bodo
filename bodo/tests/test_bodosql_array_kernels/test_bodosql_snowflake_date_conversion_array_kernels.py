# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        ("2022-02-18",),
        pytest.param(("14-DEC-2017",), id="month_name", marks=pytest.mark.slow),
        pytest.param(("1/15/2021",), id="/ date", marks=pytest.mark.slow),
        pytest.param(
            (
                pd.Series(
                    [
                        "17-May-2029",
                        "14-mar-2029",
                        "3/13/2021",
                        "03/17/2025",
                        "2022-02-18",
                    ]
                    * 2
                ),
            ),
            id="mixed-format-series",
            marks=pytest.mark.slow,
        ),
        ("2020-12-01T13:56:03.172",),
        ("2007-01-01T03:30",),
        ("1701-12-01T12:12:02.21",),
        ("2201-12-01T12:12:02.21",),
        (
            pd.Series(
                pd.date_range(start="1/2/2013", end="3/13/2021", periods=12)
            ).astype(str),
        ),
        (
            pd.Series(
                pd.date_range(start="1/2/2013", end="1/3/2013", periods=113)
            ).astype(str),
        ),
    ]
)
def valid_to_date_strings(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((0,), id="scalar_0"),
        pytest.param((pd.Series([1, -1, 92036, -1232, 3153600000]),), id="vector_data"),
        pytest.param((pd.Series(np.arange(100)),), id="nonnull_int_array"),
        pytest.param(
            (pd.Series([-23, 3, 94421, 0, None] * 4, dtype=pd.Int64Dtype()),),
            id="null_int_array",
        ),
    ]
)
def valid_to_date_ints(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((0,), id="scalar_0"),
        pytest.param(
            (
                pd.Series(
                    [
                        1,
                        -1,
                        92036,
                        -1232,
                        31536000000,
                        31536000000000,
                        31536000000000000,
                    ]
                ),
            ),
            id="vector_data",
        ),
        pytest.param((pd.Series(np.arange(100)),), id="nonnull_int_array"),
        pytest.param(
            (pd.Series([-23, 3, 94421, 0, None] * 4, dtype=pd.Int64Dtype()),),
            id="null_int_array",
        ),
    ]
)
def valid_to_date_integers_for_strings(request):
    return request.param


@pytest.fixture(
    params=[
        ("2020-12-01T13:56:03.172:00",),
        ("2342-312",),
        ("2020-13-01",),
        ("-20200-15-15",),
        ("2100-12-01-01-01-01-01-01-01-01-01-01-01-01-100",),
        (pd.Series(["2022-02-18", "2022-14-18"] * 10),),
        ("2022___02____18___",),
        ("20.234",),
        ("2ABD-2X-0Z",),
    ]
)
def invalid_to_date_args(request):
    """set of arguments which cause NA in try_to_date, and throw an error for date/to_date"""
    return request.param


@pytest.fixture(
    params=[
        ("2022-02-18", "yyyy-mm-dd"),
        ("03/23/2023", "mm/dd/yyyy"),
        ("//23/03/2023//", "//dd/mm/yyyy//"),
        (
            pd.Series(
                [
                    "Mon-15-May-2000",
                    "Tue-06-Mar-1990",
                    None,
                    "Wed-04-Jun-2019",
                    "Thu-30-Oct-2021",
                ]
                * 4
            ),
            "dy-dd-mon-yyyy",
        ),
        (
            pd.Series(
                [
                    "00*15*May",
                    "90*06*March",
                    None,
                    "19*04*June",
                    "21*30*October",
                ]
                * 4
            ),
            "yy*dd*mmmm",
        ),
    ]
)
def valid_to_date_strings_with_format_str(request):
    """
    See https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    """
    return request.param


@pytest.fixture(
    params=[
        ("12:23:00 PM", "HH12:MI:SS PM"),
        ("12:23:00 AM", "HH12:MI:SS AM"),
        ("23:23:52 PM", "HH24:MI:SS"),
        (
            pd.Series(
                [
                    "05-05-06 PM",
                    None,
                    "12-00-00 PM",
                    "09-59-59 PM",
                    "01-15-26 PM",
                ]
                * 4
            ),
            "HH12-MI-SS PM",
        ),
        (
            pd.Series(
                [
                    "00*15*20",
                    "23*06*13",
                    None,
                    "15*04*44",
                    "12*30*00",
                ]
                * 4
            ),
            "HH24*MI*SS",
        ),
    ]
)
def valid_to_time_strings_with_format_str(request):
    """
    See https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    """
    return request.param


@pytest.fixture(
    params=[
        ("Apr 15, 2000 | 12:23:00 PM", "MON DD, YYYY | HH12:MI:SS PM"),
        ("2000:08:17 06:45:00", "YYYY:MM:DD HH24:MI:SS"),
        ("20/12/25 06:45:00 AM", "YY/MM/DD HH12:MI:SS AM"),
        (
            pd.Series(
                [
                    "2023-09-15 06-42-37 PM",
                    "2022-12-03 11-11-11 PM",
                    "2023-07-28 03-20-50 PM",
                    "2022-11-10 09-30-25 PM",
                    "2023-05-02 12-55-05 PM",
                ]
                * 4
            ),
            "YYYY-MM-DD HH12-MI-SS PM",
        ),
        (
            pd.Series(
                [
                    "07/15/23 16*42*37",
                    "12/03/22 11*11*11",
                    "08/28/23 15*20*50",
                    "11/10/22 21*30*25",
                    "05/02/23 00*55*05",
                ]
                * 4
            ),
            "MM/DD/YY HH24*MI*SS",
        ),
    ]
)
def valid_to_timestamp_strings_with_format_str(request):
    """
    See https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    """
    return request.param


@pytest.fixture(
    params=[
        ("02-18-2022", "yyyy-mm-dd"),
        ("03/23/2023", "mmmm/dd/yyyy"),
        ("//23/03/2023//", "yy*dd*mon"),
        (
            pd.Series(
                [
                    "2000-May-2000",
                    "1990-Mar-1990",
                    None,
                    "2019-Jun-2019",
                    "2021-Oct-2021",
                ]
                * 4
            ),
            "dd-mon-yyyy",
        ),
        (
            pd.Series(
                [
                    "00*15*May",
                    "90*06*Mar",
                    None,
                    "19*04*Jun",
                    "21*30*Oct",
                ]
                * 4
            ),
            "//dd/mm/yyyy//",
        ),
    ]
)
def invalid_to_date_strings_with_format_str(request):
    """
    See https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    pd.date_range(start="1/2/2013", end="1/3/2013", periods=113)
                ).astype("datetime64[ns]"),
            ),
            id="non_null_dt_series",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.date_range(start="1/2/2023", end="1/3/2025", freq="5D")
                ).dt.date,
            ),
            id="date",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("4/2/2003"),
                        None,
                        pd.Timestamp("4/2/2007"),
                        pd.Timestamp("4/2/2003"),
                        None,
                    ]
                    * 4
                ).astype("datetime64[ns]"),
            ),
            id="nullable_dt_series",
        ),
        pytest.param(
            (pd.Series(pd.date_range("2018", "2025", periods=13, tz="US/Pacific")),),
            id="tz_series",
        ),
    ]
)
def to_date_td_vals(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("try_to_date", id="try_to_date"),
        pytest.param("to_date", id="to_date"),
    ]
)
def to_date_kernel(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("to_timestamp", id="to_timestamp"),
        pytest.param("try_to_timestamp", id="try_to_timestamp"),
    ]
)
def to_timestamp_kernel(request):
    return request.param


def scalar_to_date_equiv_fn(val, formatstr=None, tz=None, scale=0):
    """wrapper fn that handles timezones conversions"""
    result = scalar_to_date_equiv_fn_inner(val, formatstr, scale)
    if result is None:
        return None
    return result.tz_localize(None).tz_localize(tz).date()


def scalar_to_date_equiv_fn_inner(val, formatstr=None, scale=0):
    """equivalent to to_date for scalar value and formatstring"""
    if pd.isna(val):
        return None
    elif not pd.isna(formatstr):
        tmp_val = pd.to_datetime(
            val,
            format=convert_snowflake_date_format_str_to_py_format(formatstr),
            errors="coerce",
        ).floor(freq="D")
        if not pd.isna(tmp_val):
            return tmp_val
        else:
            return None
    elif isinstance(val, str):
        if val.isnumeric() or (len(val) > 1 and val[0] == "-" and val[1:].isnumeric()):
            return number_to_datetime(np.int64(val)).floor(freq="D")
        else:
            if "-" in val:
                parts = val.split("-")
                if len(parts) != 3:
                    return None
                # Standardize to YYYY-MM-DD
                months_map = {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "sep": 9,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }
                if parts[1] in months_map:
                    new_parts = [parts[2], str(months_map[parts[1]]), parts[0]]
                    new_str = "-".join(new_parts)
                else:
                    new_str = val
            else:
                parts = val.split("/")
                if len(parts) != 3:
                    return None
                # Standardize to YYYY-MM-DD
                new_parts = [parts[2], parts[0], parts[1]]
                new_str = "-".join(new_parts)
            tmp_val = pd.to_datetime(new_str, errors="coerce").floor(freq="D")
            if not pd.isna(tmp_val):
                return tmp_val
            else:
                return None
    elif isinstance(val, (int, np.integer, float)):
        return pd.Timestamp(val * (10 ** (9 - scale))).floor(freq="D")
    else:
        tmp_val = pd.to_datetime(val).floor(freq="D")
        if not pd.isna(tmp_val):
            return tmp_val
        else:
            return None


@pytest.mark.parametrize(
    "use_dict_enc",
    [
        pytest.param(True, id="use_dict_enc"),
        pytest.param(False, id="don't_use_dict_enc"),
    ],
)
def test_to_date_valid_strings(
    valid_to_date_strings, to_date_kernel, use_dict_enc, memory_leak_check
):
    if not use_dict_enc:
        return

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_to_date_strings, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            valid_to_date_strings,
            py_output=to_date_sol,
            sort_output=False,
            use_dict_encoded_strings=use_dict_enc,
        )
    else:
        check_func(
            to_date_impl,
            valid_to_date_strings,
            py_output=to_date_sol,
            sort_output=False,
            use_dict_encoded_strings=use_dict_enc,
        )


def test_to_date_valid_digit_strings(
    valid_to_date_integers_for_strings, to_date_kernel, memory_leak_check
):
    if isinstance(valid_to_date_integers_for_strings[0], int):
        valid_digit_strs = (str(valid_to_date_integers_for_strings[0]),)
    else:
        tmp_val = valid_to_date_integers_for_strings[0].astype(str)
        # Cast to str doesn't preserve null values, so need to re-add them
        tmp_val[pd.isna(valid_to_date_integers_for_strings[0])] = None
        valid_digit_strs = (tmp_val,)

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_digit_strs, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            valid_digit_strs,
            py_output=to_date_sol,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            valid_digit_strs,
            py_output=to_date_sol,
            sort_output=False,
        )


def test_to_date_valid_datetime_types(
    to_date_td_vals, to_date_kernel, memory_leak_check
):
    def to_date_impl(val):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_date(val, None))

    def try_to_date_impl(val):
        return pd.Series(bodo.libs.bodosql_array_kernels.try_to_date(val, None))

    to_date_sol = vectorized_sol(to_date_td_vals, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            to_date_td_vals,
            py_output=to_date_sol,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            to_date_td_vals,
            py_output=to_date_sol,
            sort_output=False,
        )


def test_to_date_valid_strings_with_format(
    valid_to_date_strings_with_format_str, to_date_kernel, memory_leak_check
):
    """
    Tests date/to_date/try_to_date kernels with valid format strings
    """

    def to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_date(val, format)

    def try_to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, format)

    to_date_sol = vectorized_sol(
        valid_to_date_strings_with_format_str, scalar_to_date_equiv_fn, None
    )

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            valid_to_date_strings_with_format_str,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            valid_to_date_strings_with_format_str,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )


def test_to_date_invalid_strings_with_format(
    invalid_to_date_strings_with_format_str, to_date_kernel
):
    """
    Tests date/to_date/try_to_date kernels with invalid format strings
    """

    def to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_date(val, format)

    def try_to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, format)

    to_date_sol = vectorized_sol(
        invalid_to_date_strings_with_format_str, scalar_to_date_equiv_fn, None
    )

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            invalid_to_date_strings_with_format_str,
            py_output=to_date_sol,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(ValueError, match=msg):
            check_func(
                to_date_impl,
                invalid_to_date_strings_with_format_str,
                py_output=to_date_sol,
            )


def test_invalid_to_date_args(invalid_to_date_args, to_date_kernel):
    """set of arguments which cause NA in try_to_date, and throw an error for date/to_date"""

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(invalid_to_date_args, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if to_date_kernel == "try_to_date":
        check_func(
            try_to_date_impl,
            invalid_to_date_args,
            py_output=to_date_sol,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(ValueError, match=msg):
            check_func(
                to_date_impl,
                invalid_to_date_args,
                py_output=to_date_sol,
            )


@pytest.mark.parametrize(
    "timestamp_str, format_str, answer",
    [
        pytest.param(
            "2000:08:17 06:45:00",
            "YYYY:MM:DD HH24:MI:SS",
            pd.Timestamp(2000, 8, 17, 6, 45, 0),
            id="scalar-1",
        ),
        pytest.param(
            "20/12/25 06:45:00 AM",
            "YY/MM/DD HH12:MI:SS AM",
            pd.Timestamp(2020, 12, 25, 6, 45, 0),
            id="scalar-2",
        ),
        pytest.param(
            "01---01---2000",
            "DD---MM---YYYY",
            pd.Timestamp(2000, 1, 1, 0, 0, 0),
            id="scalar-3",
        ),
        pytest.param(
            pd.Series(
                [
                    "2023-09-15 06-42-37 PM",
                    "2022-12-03 11-11-11 PM",
                    "2023-07-28 03-20-50 PM",
                    "2022-11-10 09-30-25 PM",
                    "2023-05-02 12-55-05 PM",
                ]
                * 4
            ),
            "YYYY-MM-DD HH12-MI-SS PM",
            pd.Series(
                [
                    pd.Timestamp(2023, 9, 15, 18, 42, 37),
                    pd.Timestamp(2022, 12, 3, 23, 11, 11),
                    pd.Timestamp(2023, 7, 28, 15, 20, 50),
                    pd.Timestamp(2022, 11, 10, 21, 30, 25),
                    pd.Timestamp(2023, 5, 2, 12, 55, 5),
                ]
                * 4
            ),
            id="series-1",
        ),
        pytest.param(
            pd.Series(
                [
                    "07/15/23 16*42*37",
                    "12/03/22 11*11*11",
                    "08/28/23 15*20*50",
                    "11/10/22 21*30*25",
                    "05/02/23 00*55*05",
                ]
                * 4
            ),
            "MM/DD/YY HH24*MI*SS",
            pd.Series(
                [
                    pd.Timestamp(2023, 7, 15, 16, 42, 37),
                    pd.Timestamp(2022, 12, 3, 11, 11, 11),
                    pd.Timestamp(2023, 8, 28, 15, 20, 50),
                    pd.Timestamp(2022, 11, 10, 21, 30, 25),
                    pd.Timestamp(2023, 5, 2, 0, 55, 5),
                ]
                * 4
            ),
            id="series-2",
        ),
    ],
)
def test_to_timestamp_valid_strings_with_format(
    timestamp_str, format_str, answer, to_timestamp_kernel, memory_leak_check
):
    """
    Tests to_timestamp/try_to_timestamp kernel with valid format strings
    """

    def to_timestamp_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_timestamp(val, format, None, 0)

    def try_to_timestamp_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_timestamp(val, format, None, 0)

    if isinstance(answer, pd.Series):
        answer = answer.to_numpy()

    if to_timestamp_kernel == "try_to_timestamp":
        check_func(
            try_to_timestamp_impl,
            (timestamp_str, format_str),
            py_output=answer,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_func(
            to_timestamp_impl,
            (timestamp_str, format_str),
            py_output=answer,
            check_dtype=False,
            sort_output=False,
        )


@pytest.mark.parametrize(
    "time_str, format_str, answer",
    [
        pytest.param(
            "06:45:00",
            "HH24:MI:SS",
            bodo.Time(6, 45, 0),
            id="scalar-24",
        ),
        pytest.param(
            "10:45:59 PM",
            "HH12:MI:SS PM",
            bodo.Time(22, 45, 59),
            id="scalar-PM",
        ),
        pytest.param(
            "12:00:00 AM",
            "HH12:MI:SS AM",
            bodo.Time(0, 0, 0),
            id="scalar-AM",
        ),
        pytest.param(
            pd.Series(
                [
                    "12:34:56 PM",
                    "05:45:23 PM",
                    "10:15:30 AM",
                    "07:55:10 PM",
                ]
                * 4
            ),
            "HH12:MI:SS PM",
            pd.Series(
                [
                    bodo.Time(12, 34, 56),
                    bodo.Time(17, 45, 23),
                    bodo.Time(10, 15, 30),
                    bodo.Time(19, 55, 10),
                ]
                * 4
            ),
            id="series-12",
        ),
        pytest.param(
            pd.Series(["08:12:34", "14:23:45", "18:36:59", "22:48:15"] * 4),
            "HH24:MI:SS",
            pd.Series(
                [
                    bodo.Time(8, 12, 34),
                    bodo.Time(14, 23, 45),
                    bodo.Time(18, 36, 59),
                    bodo.Time(22, 48, 15),
                ]
                * 4
            ),
            id="series-24",
        ),
    ],
)
@pytest.mark.parametrize("_try", [False, True])
def test_to_time_valid_strings_with_format(
    time_str, format_str, _try, answer, memory_leak_check
):
    """
    Tests to_time kernel with valid format strings
    """

    if isinstance(answer, pd.Series):
        answer = answer.to_numpy()

    def to_time_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_time(val, format, _try=_try)

    check_func(
        to_time_impl,
        (time_str, format_str),
        py_output=answer,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.slow
def test_to_dates_option(memory_leak_check):
    def impl(A, flag):
        arg0 = A if flag else None
        return (
            bodo.libs.bodosql_array_kernels.to_date(arg0, None),
            bodo.libs.bodosql_array_kernels.try_to_date(arg0, None),
        )

    # TODO: change this to test the format str for non-None values once
    # it's supported. (https://bodo.atlassian.net/browse/BE-3614)
    for flag in [True, False]:
        fn_output = pd.Timestamp("2022-02-18").date() if flag else None

        answer = (fn_output, fn_output)
        check_func(impl, ("2022-02-18", flag), py_output=answer)


@pytest.mark.parametrize(
    "source_tz, target_tz, dt, return_tz, answer",
    [
        pytest.param(
            "America/New_York",
            "America/Los_Angeles",
            pd.Timestamp("2022-08-17T12"),
            False,
            pd.Timestamp("2022-08-17T09"),
            id="three-arg",
        ),
        pytest.param(
            "Poland",
            "America/Chicago",
            pd.Timestamp("2022-08-17T12", tz="America/New_York"),
            True,
            pd.Timestamp("2022-08-17T11", tz="America/Chicago"),
            id="two-arg-data-tz",
        ),
        pytest.param(
            "Poland",
            "America/Chicago",
            pd.Timestamp("2022-08-17T12"),
            True,
            pd.Timestamp("2022-08-17T05", tz="America/Chicago"),
            id="two-arg-source-tz",
        ),
    ],
)
def test_convert_timezone_scalars(source_tz, target_tz, dt, return_tz, answer):
    def impl(data):
        return bodo.libs.bodosql_array_kernels.convert_timezone(
            source_tz,
            target_tz,
            data,
            return_tz,
        )

    check_func(
        impl,
        (dt,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.parametrize(
    "source_tz, target_tz, data, return_tz, answer",
    [
        pytest.param(
            "America/New_York",
            "America/Los_Angeles",
            pd.Series(
                [
                    pd.Timestamp("2023-01-01T12"),
                    pd.Timestamp("2022-09-01T23"),
                    pd.Timestamp("2021-08-01T14"),
                    pd.Timestamp("2020-03-01T10"),
                ]
                * 3
            ),
            True,
            pd.Series(
                [
                    pd.Timestamp("2023-01-01T09", tz="America/Los_Angeles"),
                    pd.Timestamp("2022-09-01T20", tz="America/Los_Angeles"),
                    pd.Timestamp("2021-08-01T11", tz="America/Los_Angeles"),
                    pd.Timestamp("2020-03-01T07", tz="America/Los_Angeles"),
                ]
                * 3
            ),
            id="two-arg",
        ),
        pytest.param(
            "America/New_York",
            "America/Chicago",
            pd.Series(
                [
                    pd.Timestamp("2023-01-01T12", tz="Europe/Berlin"),
                    pd.Timestamp("2022-09-01T23", tz="Europe/Berlin"),
                    pd.Timestamp("2021-08-01T14", tz="Europe/Berlin"),
                    pd.Timestamp("2020-03-01T10", tz="Europe/Berlin"),
                ]
                * 3
            ),
            True,
            pd.Series(
                [
                    pd.Timestamp("2023-01-01T05", tz="America/Chicago"),
                    pd.Timestamp("2022-09-01T16", tz="America/Chicago"),
                    pd.Timestamp("2021-08-01T07", tz="America/Chicago"),
                    pd.Timestamp("2020-03-01T03", tz="America/Chicago"),
                ]
                * 3
            ),
            id="three-arg-data-tz",
        ),
    ],
)
def test_convert_timezone(source_tz, target_tz, data, return_tz, answer):
    def impl(data):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.convert_timezone(
                source_tz, target_tz, data, return_tz
            )
        )

    check_func(
        impl,
        (data,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        check_names=False,
    )
