# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function.

The functions tested in this file are deprecated and will be phased out for bodosql array kernels
eventually.
"""
import bodosql
import numpy as np
import pandas as pd

import bodo


def test_not_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_not(True) == False


def test_not_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_not(False) == True


def test_not_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_not(None) is None


def test_not_optional_num_0():
    @bodo.jit
    def not_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not(arg0)

    assert not_run_with_optional_args(False, False, 0) == True
    assert not_run_with_optional_args(True, False, 0) is None


def test_not_optional_num_1():
    @bodo.jit
    def not_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_not(arg0)

    assert not_run_with_optional_args(False, False, 1) == True
    assert not_run_with_optional_args(True, False, 1) is None


def test_addition_default_input_0():
    assert np.isclose(bodosql.libs.generated_lib.sql_null_checking_addition(1, 2), 3)


def test_addition_default_input_1():
    assert np.isclose(bodosql.libs.generated_lib.sql_null_checking_addition(3, -7), -4)


def test_addition_default_input_2():
    assert np.isclose(bodosql.libs.generated_lib.sql_null_checking_addition(0, 0.0), 0)


def test_addition_default_input_3():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_addition(-9.23, 12.898), 3.668
    )


def test_addition_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_addition(None, 2) is None


def test_addition_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_addition(1, None) is None


def test_addition_optional_num_0():
    @bodo.jit
    def addition_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert np.isclose(addition_run_with_optional_args(False, 1, 2, 0), 3)
    assert addition_run_with_optional_args(True, 1, 2, 0) is None


def test_addition_optional_num_1():
    @bodo.jit
    def addition_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert np.isclose(addition_run_with_optional_args(False, 1, 2, 1), 3)
    assert addition_run_with_optional_args(True, 1, 2, 1) is None


def test_addition_optional_num_2():
    @bodo.jit
    def addition_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert np.isclose(addition_run_with_optional_args(False, 1, 2, 2), 3)
    assert addition_run_with_optional_args(True, 1, 2, 2) is None


def test_concatination_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_addition("hello", " world")
        == "hello world"
    )


def test_concatination_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_addition([1, 2], [3, 4]) == [
        1,
        2,
        3,
        4,
    ]


def test_concatination_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_addition(None, [3, 4]) is None


def test_concatination_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_addition([1, 2], None) is None


def test_concatination_optional_num_0():
    @bodo.jit
    def concatination_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert concatination_run_with_optional_args(False, [1, 2], [3, 4], 0) == [
        1,
        2,
        3,
        4,
    ]
    assert concatination_run_with_optional_args(True, [1, 2], [3, 4], 0) is None


def test_concatination_optional_num_1():
    @bodo.jit
    def concatination_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert concatination_run_with_optional_args(False, [1, 2], [3, 4], 1) == [
        1,
        2,
        3,
        4,
    ]
    assert concatination_run_with_optional_args(True, [1, 2], [3, 4], 1) is None


def test_concatination_optional_num_2():
    @bodo.jit
    def concatination_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_addition(arg0, arg1)

    assert concatination_run_with_optional_args(False, [1, 2], [3, 4], 2) == [
        1,
        2,
        3,
        4,
    ]
    assert concatination_run_with_optional_args(True, [1, 2], [3, 4], 2) is None


def test_subtraction_default_input_0():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_subtraction(1, 2), -1
    )


def test_subtraction_default_input_1():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_subtraction(3, -7), 10
    )


def test_subtraction_default_input_2():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_subtraction(0, 0.0), 0
    )


def test_subtraction_default_input_3():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_subtraction(-9.23, 12.898), -22.128
    )


def test_subtraction_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_subtraction(None, 12.898) is None
    )


def test_subtraction_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_subtraction(-9.23, None) is None


def test_subtraction_optional_num_0():
    @bodo.jit
    def subtraction_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

    assert np.isclose(
        subtraction_run_with_optional_args(False, -9.23, 12.898, 0), -22.128
    )
    assert subtraction_run_with_optional_args(True, -9.23, 12.898, 0) is None


def test_subtraction_optional_num_1():
    @bodo.jit
    def subtraction_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

    assert np.isclose(
        subtraction_run_with_optional_args(False, -9.23, 12.898, 1), -22.128
    )
    assert subtraction_run_with_optional_args(True, -9.23, 12.898, 1) is None


def test_subtraction_optional_num_2():
    @bodo.jit
    def subtraction_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_subtraction(arg0, arg1)

    assert np.isclose(
        subtraction_run_with_optional_args(False, -9.23, 12.898, 2), -22.128
    )
    assert subtraction_run_with_optional_args(True, -9.23, 12.898, 2) is None


def test_multiplication_default_input_0():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_multiplication(1, 2), 2
    )


def test_multiplication_default_input_1():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_multiplication(3, -12), -36
    )


def test_multiplication_default_input_2():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_multiplication(0.0, 100.0), 0
    )


def test_multiplication_default_input_3():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_multiplication(-1.123, 2.765),
        -3.105095,
    )


def test_multiplication_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_multiplication(None, 2.765) is None
    )


def test_multiplication_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_multiplication(-1.123, None)
        is None
    )


def test_multiplication_optional_num_0():
    @bodo.jit
    def multiplication_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

    assert np.isclose(
        multiplication_run_with_optional_args(False, -1.123, 2.765, 0), -3.105095
    )
    assert multiplication_run_with_optional_args(True, -1.123, 2.765, 0) is None


def test_multiplication_optional_num_1():
    @bodo.jit
    def multiplication_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

    assert np.isclose(
        multiplication_run_with_optional_args(False, -1.123, 2.765, 1), -3.105095
    )
    assert multiplication_run_with_optional_args(True, -1.123, 2.765, 1) is None


def test_multiplication_optional_num_2():
    @bodo.jit
    def multiplication_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_multiplication(
                arg0, arg1
            )

    assert np.isclose(
        multiplication_run_with_optional_args(False, -1.123, 2.765, 2), -3.105095
    )
    assert multiplication_run_with_optional_args(True, -1.123, 2.765, 2) is None


def test_true_division_default_input_0():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_true_division(1, 0.0), np.inf
    )


def test_true_division_default_input_1():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_true_division(0, 4), 0
    )


def test_true_division_default_input_2():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_true_division(0.123, 0.7),
        0.17571428571,
    )


def test_true_division_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_true_division(None, 4) is None


def test_true_division_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_true_division(0, None) is None


def test_true_division_optional_num_0():
    @bodo.jit
    def true_division_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

    assert np.isclose(true_division_run_with_optional_args(False, 0, 4, 0), 0)
    assert true_division_run_with_optional_args(True, 0, 4, 0) is None


def test_true_division_optional_num_1():
    @bodo.jit
    def true_division_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

    assert np.isclose(true_division_run_with_optional_args(False, 0, 4, 1), 0)
    assert true_division_run_with_optional_args(True, 0, 4, 1) is None


def test_true_division_optional_num_2():
    @bodo.jit
    def true_division_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_true_division(
                arg0, arg1
            )

    assert np.isclose(true_division_run_with_optional_args(False, 0, 4, 2), 0)
    assert true_division_run_with_optional_args(True, 0, 4, 2) is None


def test_modulo_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_modulo(0, 0) == 0


def test_modulo_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_modulo(1, 4) == 1


def test_modulo_default_input_2():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_modulo(13.23, 2.6), 0.23
    )


def test_modulo_default_input_3():
    assert bodosql.libs.generated_lib.sql_null_checking_modulo(5, 3) == 2


def test_modulo_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_modulo(None, 2.6) is None


def test_modulo_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_modulo(13.23, None) is None


def test_modulo_optional_num_0():
    @bodo.jit
    def modulo_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

    assert np.isclose(modulo_run_with_optional_args(False, 13.23, 2.6, 0), 0.23)
    assert modulo_run_with_optional_args(True, 13.23, 2.6, 0) is None


def test_modulo_optional_num_1():
    @bodo.jit
    def modulo_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

    assert np.isclose(modulo_run_with_optional_args(False, 13.23, 2.6, 1), 0.23)
    assert modulo_run_with_optional_args(True, 13.23, 2.6, 1) is None


def test_modulo_optional_num_2():
    @bodo.jit
    def modulo_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_modulo(arg0, arg1)

    assert np.isclose(modulo_run_with_optional_args(False, 13.23, 2.6, 2), 0.23)
    assert modulo_run_with_optional_args(True, 13.23, 2.6, 2) is None


def test_power_default_input_0():
    assert np.isclose(bodosql.libs.generated_lib.sql_null_checking_power(1, 2), 1)


def test_power_default_input_1():
    assert np.isclose(bodosql.libs.generated_lib.sql_null_checking_power(2, 4), 16)


def test_power_default_input_2():
    assert np.isclose(
        bodosql.libs.generated_lib.sql_null_checking_power(0.876, 1.8),
        0.7879658414462107,
    )


def test_power_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_power(None, 4) is None


def test_power_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_power(2, None) is None


def test_power_optional_num_0():
    @bodo.jit
    def power_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

    assert np.isclose(power_run_with_optional_args(False, 2, 4, 0), 16)
    assert power_run_with_optional_args(True, 2, 4, 0) is None


def test_power_optional_num_1():
    @bodo.jit
    def power_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

    assert np.isclose(power_run_with_optional_args(False, 2, 4, 1), 16)
    assert power_run_with_optional_args(True, 2, 4, 1) is None


def test_power_optional_num_2():
    @bodo.jit
    def power_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_power(arg0, arg1)

    assert np.isclose(power_run_with_optional_args(False, 2, 4, 2), 16)
    assert power_run_with_optional_args(True, 2, 4, 2) is None


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


def test_in_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_in(1, [1, 2, 3, 4]) == True


def test_in_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_in("e", "hello") == True


def test_in_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_in(None, "hello") is None


def test_in_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_in("e", None) is None


def test_in_optional_num_0():
    @bodo.jit
    def in_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

    assert in_run_with_optional_args(False, "e", "hello", 0) == True
    assert in_run_with_optional_args(True, "e", "hello", 0) is None


def test_in_optional_num_1():
    @bodo.jit
    def in_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

    assert in_run_with_optional_args(False, "e", "hello", 1) == True
    assert in_run_with_optional_args(True, "e", "hello", 1) is None


def test_in_optional_num_2():
    @bodo.jit
    def in_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_in(arg0, arg1)

    assert in_run_with_optional_args(False, "e", "hello", 2) == True
    assert in_run_with_optional_args(True, "e", "hello", 2) is None


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


def test_pd_timedelta_days_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(
            pd.Timedelta(100)
        )
        == 0
    )


def test_pd_timedelta_days_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(
            pd.Timedelta(100, unit="D")
        )
        == 100
    )


def test_pd_timedelta_days_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(
            pd.Timedelta(3, unit="W")
        )
        == 21
    )


def test_pd_timedelta_days_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(None) is None


def test_pd_timedelta_days_optional_num_0():
    @bodo.jit
    def pd_timedelta_days_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(arg0)

    assert (
        pd_timedelta_days_run_with_optional_args(False, pd.Timedelta(3, unit="W"), 0)
        == 21
    )
    assert (
        pd_timedelta_days_run_with_optional_args(True, pd.Timedelta(3, unit="W"), 0)
        is None
    )


def test_pd_timedelta_days_optional_num_1():
    @bodo.jit
    def pd_timedelta_days_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(arg0)

    assert (
        pd_timedelta_days_run_with_optional_args(False, pd.Timedelta(3, unit="W"), 1)
        == 21
    )
    assert (
        pd_timedelta_days_run_with_optional_args(True, pd.Timedelta(3, unit="W"), 1)
        is None
    )


def test_sql_to_python_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_sql_to_python("%%dfwfwe")
        == "dfwfwe$"
    )


def test_sql_to_python_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_sql_to_python("dfwfwe%%")
        == "^dfwfwe"
    )


def test_sql_to_python_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_sql_to_python("%%dfwfwe%%")
        == "dfwfwe"
    )


def test_sql_to_python_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_sql_to_python(None) is None


def test_sql_to_python_optional_num_0():
    @bodo.jit
    def sql_to_python_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_sql_to_python(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_sql_to_python(arg0)

    assert sql_to_python_run_with_optional_args(False, "dfwfwe%%", 0) == "^dfwfwe"
    assert sql_to_python_run_with_optional_args(True, "dfwfwe%%", 0) is None


def test_sql_to_python_optional_num_1():
    @bodo.jit
    def sql_to_python_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_sql_to_python(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_sql_to_python(arg0)

    assert sql_to_python_run_with_optional_args(False, "dfwfwe%%", 1) == "^dfwfwe"
    assert sql_to_python_run_with_optional_args(True, "dfwfwe%%", 1) is None


def test_re_match_nocase_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(
            ".*name.*", "myName"
        )
        == True
    )


def test_re_match_nocase_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(
            ".*NAME.*",
            "mynames",
        )
        == True
    )


def test_re_match_nocase_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(
            ".*name.*",
            "naMe",
        )
        == True
    )


def test_re_match_nocase_default_input_3():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(".*name.*", "nam")
        == False
    )


def test_re_match_nocase_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(".*NAME.*", None)
        is None
    )


def test_re_match_nocase_optional_num_1():
    @bodo.jit
    def re_match_nocase_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(
                arg0, arg1
            )
        else:
            return bodosql.libs.generated_lib.sql_null_checking_re_match_nocase(
                arg0, arg1
            )

    assert (
        re_match_nocase_run_with_optional_args(False, ".*name.*", "mynames", 1) == True
    )
    assert (
        re_match_nocase_run_with_optional_args(True, ".*name.*", "mynames", 1) is None
    )
