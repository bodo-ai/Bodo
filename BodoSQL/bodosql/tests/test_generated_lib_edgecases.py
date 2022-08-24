# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""contains the tests for the generated library fn that need special handling, for one reason or another"""
import bodosql
import numpy as np
import pytest

import bodo


def test_re_match_default_input_0():
    assert (
        bool(bodosql.libs.generated_lib.sql_null_checking_re_match("^h", "hello"))
        == True
    )


def test_re_match_default_input_1():
    assert (
        bool(bodosql.libs.generated_lib.sql_null_checking_re_match("$x", "hello"))
        == False
    )


def test_re_match_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_re_match(None, "hello") is None


def test_re_match_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_re_match("$x", None) is None

    @bodo.jit
    def re_match_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)


def test_re_match_optional_num_0():
    @bodo.jit
    def re_match_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

    assert bool(re_match_run_with_optional_args(False, "$x", "hello", 0)) == False
    assert re_match_run_with_optional_args(True, "$x", "hello", 0) is None


def test_re_match_optional_num_1():
    @bodo.jit
    def re_match_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

    assert bool(re_match_run_with_optional_args(False, "$x", "hello", 1)) == False
    assert re_match_run_with_optional_args(True, "$x", "hello", 1) is None


def test_re_match_optional_num_2():
    @bodo.jit
    def re_match_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match(arg0, arg1)

    assert bool(re_match_run_with_optional_args(False, "$x", "hello", 2)) == False
    assert re_match_run_with_optional_args(True, "$x", "hello", 2) is None


@pytest.mark.slow
def test_arccos():
    @bodo.jit
    def arccos_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodo.libs.bodosql_array_kernels.acos(arg0)

        else:
            if flag:
                arg0 = None
            return bodo.libs.bodosql_array_kernels.acos(arg0)

    for x in [-np.pi, np.pi, 2, -2, 131.20, -12.312]:
        normal_out = arccos_run_with_optional_args(False, x, 0)
        none_out = arccos_run_with_optional_args(True, -1, 0)
        assert np.isnan(normal_out)
        assert none_out is None


@pytest.mark.slow
def test_arcsin():
    @bodo.jit
    def arcsin_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodo.libs.bodosql_array_kernels.asin(arg0)

        else:
            if flag:
                arg0 = None
            return bodo.libs.bodosql_array_kernels.asin(arg0)

    for x in [-np.pi, np.pi, 2, -2, 131.20, -12.312]:
        normal_out = arcsin_run_with_optional_args(False, x, 0)
        none_out = arcsin_run_with_optional_args(True, -1, 0)
        assert np.isnan(normal_out)
        assert none_out is None
