# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function.

The functions tested in this file are deprecated and will be phased out for bodosql array kernels
eventually.
"""
import bodosql
import pandas as pd

import bodo


def test_equal_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_equal(1, 2) == False


def test_equal_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_equal(True, True) == True


def test_equal_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_equal("hello world", "hello")
        == False
    )


def test_equal_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_equal(None, 2) is None


def test_equal_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_equal(1, None) is None


def test_equal_optional_num_0():
    @bodo.jit
    def equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

    assert equal_run_with_optional_args(False, 1, 2, 0) == False
    assert equal_run_with_optional_args(True, 1, 2, 0) is None


def test_equal_optional_num_1():
    @bodo.jit
    def equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

    assert equal_run_with_optional_args(False, 1, 2, 1) == False
    assert equal_run_with_optional_args(True, 1, 2, 1) is None


def test_equal_optional_num_2():
    @bodo.jit
    def equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_equal(arg0, arg1)

    assert equal_run_with_optional_args(False, 1, 2, 2) == False
    assert equal_run_with_optional_args(True, 1, 2, 2) is None


def test_not_equal_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_not_equal(1, 2) == True


def test_not_equal_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_not_equal(True, True) == False


def test_not_equal_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_not_equal("hello world", "hello")
        == True
    )


def test_not_equal_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_not_equal(None, "hello") is None


def test_not_equal_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_not_equal("hello world", None)
        is None
    )


def test_not_equal_optional_num_0():
    @bodo.jit
    def not_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

    assert not_equal_run_with_optional_args(False, "hello world", "hello", 0) == True
    assert not_equal_run_with_optional_args(True, "hello world", "hello", 0) is None


def test_not_equal_optional_num_1():
    @bodo.jit
    def not_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

    assert not_equal_run_with_optional_args(False, "hello world", "hello", 1) == True
    assert not_equal_run_with_optional_args(True, "hello world", "hello", 1) is None


def test_not_equal_optional_num_2():
    @bodo.jit
    def not_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_not_equal(arg0, arg1)

    assert not_equal_run_with_optional_args(False, "hello world", "hello", 2) == True
    assert not_equal_run_with_optional_args(True, "hello world", "hello", 2) is None


def test_less_than_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than(1, 2) == True


def test_less_than_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than(True, False) == False


def test_less_than_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than("hi", "hi") == False


def test_less_than_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than(None, 2) is None


def test_less_than_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than(1, None) is None


def test_less_than_optional_num_0():
    @bodo.jit
    def less_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

    assert less_than_run_with_optional_args(False, 1, 2, 0) == True
    assert less_than_run_with_optional_args(True, 1, 2, 0) is None


def test_less_than_optional_num_1():
    @bodo.jit
    def less_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

    assert less_than_run_with_optional_args(False, 1, 2, 1) == True
    assert less_than_run_with_optional_args(True, 1, 2, 1) is None


def test_less_than_optional_num_2():
    @bodo.jit
    def less_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than(arg0, arg1)

    assert less_than_run_with_optional_args(False, 1, 2, 2) == True
    assert less_than_run_with_optional_args(True, 1, 2, 2) is None


def test_less_than_or_equal_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(1, 2) == True


def test_less_than_or_equal_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(True, False)
        == False
    )


def test_less_than_or_equal_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal("hi", "hi")
        == True
    )


def test_less_than_or_equal_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(None, False)
        is None
    )


def test_less_than_or_equal_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(True, None)
        is None
    )


def test_less_than_or_equal_optional_num_0():
    @bodo.jit
    def less_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

    assert less_than_or_equal_run_with_optional_args(False, True, False, 0) == False
    assert less_than_or_equal_run_with_optional_args(True, True, False, 0) is None


def test_less_than_or_equal_optional_num_1():
    @bodo.jit
    def less_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

    assert less_than_or_equal_run_with_optional_args(False, True, False, 1) == False
    assert less_than_or_equal_run_with_optional_args(True, True, False, 1) is None


def test_less_than_or_equal_optional_num_2():
    @bodo.jit
    def less_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal(
                arg0, arg1
            )

    assert less_than_or_equal_run_with_optional_args(False, True, False, 2) == False
    assert less_than_or_equal_run_with_optional_args(True, True, False, 2) is None


def test_greater_than_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_greater_than(1, 2) == False


def test_greater_than_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than(True, False) == True
    )


def test_greater_than_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than("hi", "hi") == False
    )


def test_greater_than_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than(None, False) is None
    )


def test_greater_than_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_greater_than(True, None) is None


def test_greater_than_optional_num_0():
    @bodo.jit
    def greater_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

    assert greater_than_run_with_optional_args(False, True, False, 0) == True
    assert greater_than_run_with_optional_args(True, True, False, 0) is None


def test_greater_than_optional_num_1():
    @bodo.jit
    def greater_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

    assert greater_than_run_with_optional_args(False, True, False, 1) == True
    assert greater_than_run_with_optional_args(True, True, False, 1) is None


def test_greater_than_optional_num_2():
    @bodo.jit
    def greater_than_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than(arg0, arg1)

    assert greater_than_run_with_optional_args(False, True, False, 2) == True
    assert greater_than_run_with_optional_args(True, True, False, 2) is None


def test_greater_than_or_equal_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(1, 2)
        == False
    )


def test_greater_than_or_equal_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(True, False)
        == True
    )


def test_greater_than_or_equal_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal("hi", "hi")
        == True
    )


def test_greater_than_or_equal_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(None, False)
        is None
    )


def test_greater_than_or_equal_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(True, None)
        is None
    )


def test_greater_than_or_equal_optional_num_0():
    @bodo.jit
    def greater_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

    assert greater_than_or_equal_run_with_optional_args(False, True, False, 0) == True
    assert greater_than_or_equal_run_with_optional_args(True, True, False, 0) is None


def test_greater_than_or_equal_optional_num_1():
    @bodo.jit
    def greater_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

    assert greater_than_or_equal_run_with_optional_args(False, True, False, 1) == True
    assert greater_than_or_equal_run_with_optional_args(True, True, False, 1) is None


def test_greater_than_or_equal_optional_num_2():
    @bodo.jit
    def greater_than_or_equal_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal(
                arg0, arg1
            )

    assert greater_than_or_equal_run_with_optional_args(False, True, False, 2) == True
    assert greater_than_or_equal_run_with_optional_args(True, True, False, 2) is None


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
