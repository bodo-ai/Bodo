# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function.

The functions tested in this file are deprecated and will be phased out for bodosql array kernels
eventually.
"""
import pandas as pd

import bodo
import bodosql
from bodo.tests.utils import pytest_slow_unless_codegen

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


def test_strftime_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strftime(
            pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d"
        )
        == "2021-07-14"
    )


def test_strftime_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strftime(
            pd.Timestamp("2021-07-14 15:44:04.498582"), "%d, %b, %y. Time: %H:%M"
        )
        == "14, Jul, 21. Time: 15:44"
    )


def test_strftime_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strftime(None, "%Y-%m-%d") is None
    )


def test_strftime_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strftime(
            pd.Timestamp("2021-07-14 15:44:04.498582"), None
        )
        is None
    )


def test_strftime_optional_num_0():
    @bodo.jit
    def strftime_run_with_optional_args(flag, arg0, arg1, optional_num):
        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

    assert (
        strftime_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 0
        )
        == "2021-07-14"
    )
    assert (
        strftime_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 0
        )
        is None
    )


def test_strftime_optional_num_1():
    @bodo.jit
    def strftime_run_with_optional_args(flag, arg0, arg1, optional_num):
        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

    assert (
        strftime_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 1
        )
        == "2021-07-14"
    )
    assert (
        strftime_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 1
        )
        is None
    )


def test_strftime_optional_num_2():
    @bodo.jit
    def strftime_run_with_optional_args(flag, arg0, arg1, optional_num):
        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strftime(arg0, arg1)

    assert (
        strftime_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 2
        )
        == "2021-07-14"
    )
    assert (
        strftime_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), "%Y-%m-%d", 2
        )
        is None
    )


def test_pd_to_datetime_with_format_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
        "2021-07-14", "%Y-%m-%d"
    ) == pd.Timestamp("2021-07-14")


def test_pd_to_datetime_with_format_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
        "14, Jul, 21. Time: 15:44", "%d, %b, %y. Time: %H:%M"
    ) == pd.Timestamp("2021-07-14 15:44:00")


def test_pd_to_datetime_with_format_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
            None, "%Y-%m-%d"
        )
        is None
    )


def test_pd_to_datetime_with_format_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
            "2021-07-14", None
        )
        is None
    )


def test_pd_to_datetime_with_format_optional_num_0():
    @bodo.jit
    def pd_to_datetime_with_format_run_with_optional_args(
        flag, arg0, arg1, optional_num
    ):
        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

    assert pd_to_datetime_with_format_run_with_optional_args(
        False, "2021-07-14", "%Y-%m-%d", 0
    ) == pd.Timestamp("2021-07-14")
    assert (
        pd_to_datetime_with_format_run_with_optional_args(
            True, "2021-07-14", "%Y-%m-%d", 0
        )
        is None
    )


def test_pd_to_datetime_with_format_optional_num_1():
    @bodo.jit
    def pd_to_datetime_with_format_run_with_optional_args(
        flag, arg0, arg1, optional_num
    ):
        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

    assert pd_to_datetime_with_format_run_with_optional_args(
        False, "2021-07-14", "%Y-%m-%d", 1
    ) == pd.Timestamp("2021-07-14")
    assert (
        pd_to_datetime_with_format_run_with_optional_args(
            True, "2021-07-14", "%Y-%m-%d", 1
        )
        is None
    )


def test_pd_to_datetime_with_format_optional_num_2():
    @bodo.jit
    def pd_to_datetime_with_format_run_with_optional_args(
        flag, arg0, arg1, optional_num
    ):
        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_to_datetime_with_format(
                    arg0, arg1
                )
            )

    assert pd_to_datetime_with_format_run_with_optional_args(
        False, "2021-07-14", "%Y-%m-%d", 2
    ) == pd.Timestamp("2021-07-14")
    assert (
        pd_to_datetime_with_format_run_with_optional_args(
            True, "2021-07-14", "%Y-%m-%d", 2
        )
        is None
    )