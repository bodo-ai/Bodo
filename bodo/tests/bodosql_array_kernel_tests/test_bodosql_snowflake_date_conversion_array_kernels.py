# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


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
        pytest.param((0,), id="int_0"),
        pytest.param((1,), id="int_1"),
        pytest.param((-1,), id="int_neg_1"),
        pytest.param((92036,), id="int_92036"),
        pytest.param((-1232,), id="int_neg_1232"),
        pytest.param((31536000000,), id="int_31536000000"),
        pytest.param(
            (31535999999,),
            id="int_31535999999",
            marks=pytest.mark.skip(
                "Timestamp doesn't have sufficient range (max is ~9223372036 seconds)"
            ),
        ),
        pytest.param((31536000000000,), id="int_31536000000000"),
        pytest.param(
            (31535999999999,),
            id="int_31535999999999",
            marks=pytest.mark.skip(
                "Timestamp doesn't have sufficient range (max is ~9223372036 * 10^3 ms)"
            ),
        ),
        pytest.param((31536000000000000,), id="int_31536000000000000"),
        pytest.param(
            (31535999999999999,),
            id="int_31535999999999999",
            marks=pytest.mark.skip(
                "Timestamp doesn't have sufficient range (max is ~9223372036 * 10^6 us)"
            ),
        ),
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
    """set of arguments which cause NA in try_to_date, and throw an error for to_date"""
    return request.param


# TODO: support format string, https://bodo.atlassian.net/browse/BE-3614
@pytest.fixture(params=[("2022___02____18___", "yyyy___mm____dd___")])
def valid_to_date_strings_with_format_str(request):
    """
    See https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    """
    return request.param


@pytest.fixture(
    params=[
        (pd.Timestamp(0),),
        (pd.Timestamp("2019-02-14 04:00:01"),),
        pytest.param(
            (
                pd.Series(
                    pd.date_range(start="1/2/2013", end="1/3/2013", periods=113)
                ).astype("datetime64[ns]"),
            ),
            id="non_null_dt_series_1",
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
            id="nullable_dt_series_1",
        ),
        pytest.param(
            (pd.Series(pd.date_range(start="1/2/2013", end="3/13/2021", periods=12)),),
            id="non_null_dt_series_2",
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
                ),
            ),
            id="nullable_dt_series_2",
        ),
    ]
)
def to_date_td_vals(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(True, id="try_to_date"), pytest.param(False, id="to_date")]
)
def test_try(request):
    return request.param


def scalar_to_date_equiv_fn(val, formatstr=None):
    """wrapper fn that converts timestamp to np dt64 if needed"""
    return scalar_to_date_equiv_fn_inner(val, formatstr=None)


def scalar_to_date_equiv_fn_inner(val, formatstr=None):
    """equivalent to TO_DATE for scalar value and formatstring"""
    if pd.isna(val):
        return None
    elif not pd.isna(formatstr):
        tmp_val = pd.to_datetime(
            val,
            format=convert_sql_date_format_str_to_py_format(formatstr),
            errors="coerce",
        ).floor(freq="D")
        if not pd.isna(tmp_val):
            return tmp_val
        else:
            return None
    elif isinstance(val, str):
        if val.isnumeric() or (len(val) > 1 and val[0] == "-" and val[1:].isnumeric()):
            return int_to_datetime(np.int64(val)).floor(freq="D")
        else:
            tmp_val = pd.to_datetime(val, errors="coerce").floor(freq="D")
            if not pd.isna(tmp_val):
                return tmp_val
            else:
                return None
    elif isinstance(val, (int, np.integer)):
        return int_to_datetime(val).floor(freq="D")
    else:
        tmp_val = pd.to_datetime(val).floor(freq="D")
        if not pd.isna(tmp_val):
            return tmp_val
        else:
            return None


def test_to_date_valid_ints(valid_to_date_ints, test_try, memory_leak_check):
    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_to_date_ints, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if test_try:
        check_func(
            try_to_date_impl,
            valid_to_date_ints,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            valid_to_date_ints,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )


@pytest.mark.parametrize(
    "use_dict_enc",
    [
        pytest.param(True, id="use_dict_enc"),
        pytest.param(False, id="don't_use_dict_enc"),
    ],
)
def test_to_date_valid_strings(
    valid_to_date_strings, test_try, use_dict_enc, memory_leak_check
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

    if test_try:
        check_func(
            try_to_date_impl,
            valid_to_date_strings,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
            use_dict_encoded_strings=use_dict_enc,
        )
    else:
        check_func(
            to_date_impl,
            valid_to_date_strings,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
            use_dict_encoded_strings=use_dict_enc,
        )


def test_to_date_valid_digit_strings(valid_to_date_ints, test_try, memory_leak_check):

    if isinstance(valid_to_date_ints[0], int):
        valid_digit_strs = (str(valid_to_date_ints[0]),)
    else:
        tmp_val = valid_to_date_ints[0].astype(str)
        # Cast to str doesn't preserve null values, so need to re-add them
        tmp_val[pd.isna(valid_to_date_ints[0])] = None
        valid_digit_strs = (tmp_val,)

    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(valid_digit_strs, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if test_try:
        check_func(
            try_to_date_impl,
            valid_digit_strs,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            valid_digit_strs,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )


def test_to_date_valid_datetime_types(to_date_td_vals, test_try, memory_leak_check):
    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(to_date_td_vals, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if test_try:
        check_func(
            try_to_date_impl,
            to_date_td_vals,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_func(
            to_date_impl,
            to_date_td_vals,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )


@pytest.mark.skip(
    "TODO: support format string, https://bodo.atlassian.net/browse/BE-3614"
)
def test_to_date_valid_strings_with_format(
    valid_to_date_strings_with_format_str, test_try, memory_leak_check
):
    def to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val, format):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(
        valid_to_date_strings_with_format_str, scalar_to_date_equiv_fn, None
    )

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if test_try:
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


def test_invalid_to_date_args(invalid_to_date_args, test_try):
    """set of arguments which cause NA in try_to_date, and throw an error for to_date"""

    @bodo.jit
    def to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.to_date(val, None)

    def try_to_date_impl(val):
        return bodo.libs.bodosql_array_kernels.try_to_date(val, None)

    to_date_sol = vectorized_sol(invalid_to_date_args, scalar_to_date_equiv_fn, None)

    if isinstance(to_date_sol, pd.Series):
        to_date_sol = to_date_sol.to_numpy()

    if test_try:
        check_func(
            try_to_date_impl,
            invalid_to_date_args,
            py_output=to_date_sol,
            check_dtype=False,
            sort_output=False,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(ValueError, match=msg):
            to_date_impl(invalid_to_date_args[0])


@pytest.mark.slow
def test_to_dates_option(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.to_date(arg0, arg1),
            bodo.libs.bodosql_array_kernels.try_to_date(arg0, arg1),
        )

    # TODO: change this to test the format str for non-None values once
    # it's supported. (https://bodo.atlassian.net/browse/BE-3614)
    for flag0 in [True, False]:
        for flag1 in [False]:
            fn_output = pd.Timestamp("2022-02-18") if flag0 else None

            answer = (fn_output, fn_output)
            check_func(impl, ("2022-02-18", "TODO", flag0, flag1), py_output=answer)
