# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import bodosql_use_date_type, check_func


@pytest.fixture(
    params=[
        ("2022-02-18",),
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
        pytest.param("DATE", id="date"),
        pytest.param("TRY_TO_DATE", id="try_to_date"),
        pytest.param("TO_DATE", id="to_date"),
    ]
)
def test_fn(request):
    return request.param


def scalar_to_date_equiv_fn(val, formatstr=None, tz=None, scale=0):
    """wrapper fn that handles timezones conversions"""
    result = scalar_to_date_equiv_fn_inner(val, formatstr, scale)
    if result is None:
        return None
    return result.tz_localize(None).tz_localize(tz).date()


def scalar_to_date_equiv_fn_inner(val, formatstr=None, scale=0):
    """equivalent to TO_DATE for scalar value and formatstring"""
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
            tmp_val = pd.to_datetime(val, errors="coerce").floor(freq="D")
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
    valid_to_date_strings, test_fn, use_dict_enc, memory_leak_check
):

    if not use_dict_enc:
        return

    def date_impl(val):
        return bodo.libs.bodosql_array_kernels.date(val, None)

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_to_date_strings, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                valid_to_date_strings,
                py_output=to_date_sol,
                sort_output=False,
                use_dict_encoded_strings=use_dict_enc,
            )
        elif test_fn == "TO_DATE":
            check_func(
                to_date_impl,
                valid_to_date_strings,
                py_output=to_date_sol,
                sort_output=False,
                use_dict_encoded_strings=use_dict_enc,
            )
        else:
            check_func(
                date_impl,
                valid_to_date_strings,
                py_output=to_date_sol,
                sort_output=False,
                use_dict_encoded_strings=use_dict_enc,
            )


def test_to_date_valid_digit_strings(
    valid_to_date_integers_for_strings, test_fn, memory_leak_check
):

    if isinstance(valid_to_date_integers_for_strings[0], int):
        valid_digit_strs = (str(valid_to_date_integers_for_strings[0]),)
    else:
        tmp_val = valid_to_date_integers_for_strings[0].astype(str)
        # Cast to str doesn't preserve null values, so need to re-add them
        tmp_val[pd.isna(valid_to_date_integers_for_strings[0])] = None
        valid_digit_strs = (tmp_val,)

    def date_impl(val):
        return bodo.libs.bodosql_array_kernels.date(val, None)

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_digit_strs, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                valid_digit_strs,
                py_output=to_date_sol,
                sort_output=False,
            )
        elif test_fn == "TO_DATE":
            check_func(
                to_date_impl,
                valid_digit_strs,
                py_output=to_date_sol,
                sort_output=False,
            )
        else:
            check_func(
                date_impl,
                valid_digit_strs,
                py_output=to_date_sol,
                sort_output=False,
            )


def test_to_date_valid_datetime_types(to_date_td_vals, test_fn, memory_leak_check):
    def date_impl(val):
        return bodo.libs.bodosql_array_kernels.date(val, None)

    def to_date_impl(val):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_date(val, None))

    def try_to_date_impl(val):
        return pd.Series(bodo.libs.bodosql_array_kernels.try_to_date(val, None))

    to_date_sol = vectorized_sol(to_date_td_vals, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                to_date_td_vals,
                py_output=to_date_sol,
                sort_output=False,
            )
        elif test_fn == "TO_DATE":
            check_func(
                to_date_impl,
                to_date_td_vals,
                py_output=to_date_sol,
                sort_output=False,
            )
        else:
            check_func(
                date_impl,
                to_date_td_vals,
                py_output=to_date_sol,
                sort_output=False,
            )


def test_to_date_valid_strings_with_format(
    valid_to_date_strings_with_format_str, test_fn, memory_leak_check
):
    """
    Tests date/to_date/try_to_date kernels with valid format strings
    """

    def date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.date(val, format)

    def to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_date(val, format)

    def try_to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, format)

    to_date_sol = vectorized_sol(
        valid_to_date_strings_with_format_str, scalar_to_date_equiv_fn, None
    )

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                valid_to_date_strings_with_format_str,
                py_output=to_date_sol,
                check_dtype=False,
                sort_output=False,
            )
        elif test_fn == "TO_DATE":
            check_func(
                to_date_impl,
                valid_to_date_strings_with_format_str,
                py_output=to_date_sol,
                check_dtype=False,
                sort_output=False,
            )
        else:
            check_func(
                date_impl,
                valid_to_date_strings_with_format_str,
                py_output=to_date_sol,
                check_dtype=False,
                sort_output=False,
            )


def test_to_date_invalid_strings_with_format(
    invalid_to_date_strings_with_format_str, test_fn
):
    """
    Tests date/to_date/try_to_date kernels with invalid format strings
    """

    def date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.date(val, format)

    def to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_date(val, format)

    def try_to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, format)

    to_date_sol = vectorized_sol(
        invalid_to_date_strings_with_format_str, scalar_to_date_equiv_fn, None
    )

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                invalid_to_date_strings_with_format_str,
                py_output=to_date_sol,
            )
        elif test_fn == "TO_DATE":
            msg = "Invalid input while converting to date value"
            with pytest.raises(ValueError, match=msg):
                check_func(
                    to_date_impl,
                    invalid_to_date_strings_with_format_str,
                    py_output=to_date_sol,
                )
        else:
            msg = "Invalid input while converting to date value"
            with pytest.raises(ValueError, match=msg):
                check_func(
                    date_impl,
                    invalid_to_date_strings_with_format_str,
                    py_output=to_date_sol,
                )


def test_invalid_to_date_args(invalid_to_date_args, test_fn):
    """set of arguments which cause NA in try_to_date, and throw an error for date/to_date"""

    def date_impl(val):
        return bodo.libs.bodosql_array_kernels.date(val, None)

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(invalid_to_date_args, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    with bodosql_use_date_type():
        if test_fn == "TRY_TO_DATE":
            check_func(
                try_to_date_impl,
                invalid_to_date_args,
                py_output=to_date_sol,
            )
        elif test_fn == "TO_DATE":
            msg = "Invalid input while converting to date value"
            with pytest.raises(ValueError, match=msg):
                check_func(
                    to_date_impl,
                    invalid_to_date_args,
                    py_output=to_date_sol,
                )
        else:
            msg = "Invalid input while converting to date value"
            with pytest.raises(ValueError, match=msg):
                check_func(
                    date_impl,
                    invalid_to_date_args,
                    py_output=to_date_sol,
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
        fn_output = pd.Timestamp("2022-02-18") if flag else None

        answer = (fn_output, fn_output)
        check_func(impl, ("2022-02-18", flag), py_output=answer)
