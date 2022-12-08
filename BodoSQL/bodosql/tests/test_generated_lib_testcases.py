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


def test_strip_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strip("   hello", " ") == "hello"
    )


def test_strip_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strip("hello   ", " ") == "hello"
    )


def test_strip_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_strip("   hello   ", " ")
        == "hello"
    )


def test_strip_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_strip(None, " ") is None


def test_strip_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_strip("   hello", None) is None


def test_strip_optional_num_0():
    @bodo.jit
    def strip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

    assert strip_run_with_optional_args(False, "   hello", " ", 0) == "hello"
    assert strip_run_with_optional_args(True, "   hello", " ", 0) is None


def test_strip_optional_num_1():
    @bodo.jit
    def strip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

    assert strip_run_with_optional_args(False, "   hello", " ", 1) == "hello"
    assert strip_run_with_optional_args(True, "   hello", " ", 1) is None


def test_strip_optional_num_2():
    @bodo.jit
    def strip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_strip(arg0, arg1)

    assert strip_run_with_optional_args(False, "   hello", " ", 2) == "hello"
    assert strip_run_with_optional_args(True, "   hello", " ", 2) is None


def test_lstrip_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_lstrip("   hello", " ") == "hello"
    )


def test_lstrip_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_lstrip("hello   ", " ")
        == "hello   "
    )


def test_lstrip_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_lstrip("   hello   ", " ")
        == "hello   "
    )


def test_lstrip_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_lstrip(None, " ") is None


def test_lstrip_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_lstrip("   hello   ", None) is None
    )


def test_lstrip_optional_num_0():
    @bodo.jit
    def lstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

    assert lstrip_run_with_optional_args(False, "   hello   ", " ", 0) == "hello   "
    assert lstrip_run_with_optional_args(True, "   hello   ", " ", 0) is None


def test_lstrip_optional_num_1():
    @bodo.jit
    def lstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

    assert lstrip_run_with_optional_args(False, "   hello   ", " ", 1) == "hello   "
    assert lstrip_run_with_optional_args(True, "   hello   ", " ", 1) is None


def test_lstrip_optional_num_2():
    @bodo.jit
    def lstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_lstrip(arg0, arg1)

    assert lstrip_run_with_optional_args(False, "   hello   ", " ", 2) == "hello   "
    assert lstrip_run_with_optional_args(True, "   hello   ", " ", 2) is None


def test_rstrip_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_rstrip("   hello", " ")
        == "   hello"
    )


def test_rstrip_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_rstrip("hello   ", " ") == "hello"
    )


def test_rstrip_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_rstrip("   hello   ", " ")
        == "   hello"
    )


def test_rstrip_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_rstrip(None, " ") is None


def test_rstrip_None_Arg_1():
    assert bodosql.libs.generated_lib.sql_null_checking_rstrip("hello   ", None) is None


def test_rstrip_optional_num_0():
    @bodo.jit
    def rstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

    assert rstrip_run_with_optional_args(False, "hello   ", " ", 0) == "hello"
    assert rstrip_run_with_optional_args(True, "hello   ", " ", 0) is None


def test_rstrip_optional_num_1():
    @bodo.jit
    def rstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

    assert rstrip_run_with_optional_args(False, "hello   ", " ", 1) == "hello"
    assert rstrip_run_with_optional_args(True, "hello   ", " ", 1) is None


def test_rstrip_optional_num_2():
    @bodo.jit
    def rstrip_run_with_optional_args(flag, arg0, arg1, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_rstrip(arg0, arg1)

    assert rstrip_run_with_optional_args(False, "hello   ", " ", 2) == "hello"
    assert rstrip_run_with_optional_args(True, "hello   ", " ", 2) is None


def test_len_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_len("hello") == 5


def test_len_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_len([1, 2, 3]) == 3


def test_len_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_len(None) is None


def test_len_optional_num_0():
    @bodo.jit
    def len_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_len(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_len(arg0)

    assert len_run_with_optional_args(False, [1, 2, 3], 0) == 3
    assert len_run_with_optional_args(True, [1, 2, 3], 0) is None


def test_len_optional_num_1():
    @bodo.jit
    def len_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_len(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_len(arg0)

    assert len_run_with_optional_args(False, [1, 2, 3], 1) == 3
    assert len_run_with_optional_args(True, [1, 2, 3], 1) is None


def test_upper_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_upper("HELLO") == "HELLO"


def test_upper_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_upper("HeLlO") == "HELLO"


def test_upper_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_upper("hello") == "HELLO"


def test_upper_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_upper(None) is None


def test_upper_optional_num_0():
    @bodo.jit
    def upper_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_upper(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_upper(arg0)

    assert upper_run_with_optional_args(False, "hello", 0) == "HELLO"
    assert upper_run_with_optional_args(True, "hello", 0) is None


def test_upper_optional_num_1():
    @bodo.jit
    def upper_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_upper(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_upper(arg0)

    assert upper_run_with_optional_args(False, "hello", 1) == "HELLO"
    assert upper_run_with_optional_args(True, "hello", 1) is None


def test_lower_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_lower("HELLO") == "hello"


def test_lower_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_lower("HeLlO") == "hello"


def test_lower_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_lower("hello") == "hello"


def test_lower_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_lower(None) is None


def test_lower_optional_num_0():
    @bodo.jit
    def lower_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lower(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lower(arg0)

    assert lower_run_with_optional_args(False, "HeLlO", 0) == "hello"
    assert lower_run_with_optional_args(True, "HeLlO", 0) is None


def test_lower_optional_num_1():
    @bodo.jit
    def lower_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lower(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_lower(arg0)

    assert lower_run_with_optional_args(False, "HeLlO", 1) == "hello"
    assert lower_run_with_optional_args(True, "HeLlO", 1) is None


def test_replace_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_replace("hello world", " ", "_")
        == "hello_world"
    )


def test_replace_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_replace("hello world", "o", "0")
        == "hell0 w0rld"
    )


def test_replace_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_replace(None, "o", "0") is None


def test_replace_None_Arg_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_replace("hello world", None, "0")
        is None
    )


def test_replace_None_Arg_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_replace("hello world", "o", None)
        is None
    )


def test_replace_optional_num_0():
    @bodo.jit
    def replace_run_with_optional_args(flag, arg0, arg1, arg2, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 2:
            if flag:
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

    assert (
        replace_run_with_optional_args(False, "hello world", "o", "0", 0)
        == "hell0 w0rld"
    )
    assert replace_run_with_optional_args(True, "hello world", "o", "0", 0) is None


def test_replace_optional_num_1():
    @bodo.jit
    def replace_run_with_optional_args(flag, arg0, arg1, arg2, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 2:
            if flag:
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

    assert (
        replace_run_with_optional_args(False, "hello world", "o", "0", 1)
        == "hell0 w0rld"
    )
    assert replace_run_with_optional_args(True, "hello world", "o", "0", 1) is None


def test_replace_optional_num_2():
    @bodo.jit
    def replace_run_with_optional_args(flag, arg0, arg1, arg2, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 2:
            if flag:
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

    assert (
        replace_run_with_optional_args(False, "hello world", "o", "0", 2)
        == "hell0 w0rld"
    )
    assert replace_run_with_optional_args(True, "hello world", "o", "0", 2) is None


def test_replace_optional_num_3():
    @bodo.jit
    def replace_run_with_optional_args(flag, arg0, arg1, arg2, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        elif optional_num == 2:
            if flag:
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

        else:
            if flag:
                arg0 = None
                arg1 = None
                arg2 = None
            return bodosql.libs.generated_lib.sql_null_checking_replace(
                arg0, arg1, arg2
            )

    assert (
        replace_run_with_optional_args(False, "hello world", "o", "0", 3)
        == "hell0 w0rld"
    )
    assert replace_run_with_optional_args(True, "hello world", "o", "0", 3) is None


# def test_spaces_default_input_0():
#     assert bodosql.libs.generated_lib.sql_null_checking_spaces(12) == "            "


# def test_spaces_default_input_1():
#     assert bodosql.libs.generated_lib.sql_null_checking_spaces(0) == ""


# def test_spaces_default_input_2():
#     assert bodosql.libs.generated_lib.sql_null_checking_spaces(-10) == ""


# def test_spaces_default_input_3():
#     assert bodosql.libs.generated_lib.sql_null_checking_spaces(2) == "  "


# def test_spaces_None_Arg_0():
#     assert bodosql.libs.generated_lib.sql_null_checking_spaces(None) is None


# def test_spaces_optional_num_0():
#     @bodo.jit
#     def spaces_run_with_optional_args(flag, arg0, optional_num):

#         if optional_num == 0:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_spaces(arg0)

#         else:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_spaces(arg0)

#     assert spaces_run_with_optional_args(False, 2, 0) == "  "
#     assert spaces_run_with_optional_args(True, 2, 0) is None


# def test_spaces_optional_num_1():
#     @bodo.jit
#     def spaces_run_with_optional_args(flag, arg0, optional_num):

#         if optional_num == 0:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_spaces(arg0)

#         else:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_spaces(arg0)

#     assert spaces_run_with_optional_args(False, 2, 1) == "  "
#     assert spaces_run_with_optional_args(True, 2, 1) is None


# def test_reverse_default_input_0():
#     assert bodosql.libs.generated_lib.sql_null_checking_reverse("hello") == "olleh"


# def test_reverse_default_input_1():
#     assert bodosql.libs.generated_lib.sql_null_checking_reverse([1, 2, 3]) == [3, 2, 1]


# def test_reverse_default_input_2():
#     assert bodosql.libs.generated_lib.sql_null_checking_reverse("") == ""


# def test_reverse_None_Arg_0():
#     assert bodosql.libs.generated_lib.sql_null_checking_reverse(None) is None


# def test_reverse_optional_num_0():
#     @bodo.jit
#     def reverse_run_with_optional_args(flag, arg0, optional_num):

#         if optional_num == 0:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_reverse(arg0)

#         else:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_reverse(arg0)

#     assert reverse_run_with_optional_args(False, [1, 2, 3], 0) == [3, 2, 1]
#     assert reverse_run_with_optional_args(True, [1, 2, 3], 0) is None


# def test_reverse_optional_num_1():
#     @bodo.jit
#     def reverse_run_with_optional_args(flag, arg0, optional_num):

#         if optional_num == 0:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_reverse(arg0)

#         else:
#             if flag:
#                 arg0 = None
#             return bodosql.libs.generated_lib.sql_null_checking_reverse(arg0)

#     assert reverse_run_with_optional_args(False, [1, 2, 3], 1) == [3, 2, 1]
#     assert reverse_run_with_optional_args(True, [1, 2, 3], 1) is None


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


def test_pd_Timestamp_single_value_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
        1000
    ) == pd.Timestamp(1000)


def test_pd_Timestamp_single_value_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
        "2021-07-14 15:44:04.498582"
    ) == pd.Timestamp("2021-07-14 15:44:04.498582")


def test_pd_Timestamp_single_value_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(None)
        is None
    )


def test_pd_Timestamp_single_value_optional_num_0():
    @bodo.jit
    def pd_Timestamp_single_value_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
                    arg0
                )
            )

        else:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
                    arg0
                )
            )

    assert pd_Timestamp_single_value_run_with_optional_args(
        False, 1000, 0
    ) == pd.Timestamp(1000)
    assert pd_Timestamp_single_value_run_with_optional_args(True, 1000, 0) is None


def test_pd_Timestamp_single_value_optional_num_1():
    @bodo.jit
    def pd_Timestamp_single_value_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
                    arg0
                )
            )

        else:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value(
                    arg0
                )
            )

    assert pd_Timestamp_single_value_run_with_optional_args(
        False, 1000, 1
    ) == pd.Timestamp(1000)
    assert pd_Timestamp_single_value_run_with_optional_args(True, 1000, 1) is None


def test_pd_Timestamp_single_value_with_second_unit_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
        100
    ) == pd.Timestamp(
        100, unit="s"
    )


def test_pd_Timestamp_single_value_with_second_unit_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
        12
    ) == pd.Timestamp(
        12, unit="s"
    )


def test_pd_Timestamp_single_value_with_second_unit_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
        14
    ) == pd.Timestamp(
        14, unit="s"
    )


def test_pd_Timestamp_single_value_with_second_unit_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
            None
        )
        is None
    )


def test_pd_Timestamp_single_value_with_second_unit_optional_num_0():
    @bodo.jit
    def pd_Timestamp_single_value_with_second_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_second_unit_run_with_optional_args(
        False, 12, 0
    ) == pd.Timestamp(12, unit="s")
    assert (
        pd_Timestamp_single_value_with_second_unit_run_with_optional_args(True, 12, 0)
        is None
    )


def test_pd_Timestamp_single_value_with_second_unit_optional_num_1():
    @bodo.jit
    def pd_Timestamp_single_value_with_second_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_second_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_second_unit_run_with_optional_args(
        False, 12, 1
    ) == pd.Timestamp(12, unit="s")
    assert (
        pd_Timestamp_single_value_with_second_unit_run_with_optional_args(True, 12, 1)
        is None
    )


def test_pd_Timestamp_single_value_with_day_unit_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
        100
    ) == pd.Timestamp(
        100, unit="D"
    )


def test_pd_Timestamp_single_value_with_day_unit_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
        12
    ) == pd.Timestamp(
        12, unit="D"
    )


def test_pd_Timestamp_single_value_with_day_unit_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
        14
    ) == pd.Timestamp(
        14, unit="D"
    )


def test_pd_Timestamp_single_value_with_day_unit_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
            None
        )
        is None
    )


def test_pd_Timestamp_single_value_with_day_unit_optional_num_0():
    @bodo.jit
    def pd_Timestamp_single_value_with_day_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_day_unit_run_with_optional_args(
        False, 12, 0
    ) == pd.Timestamp(12, unit="D")
    assert (
        pd_Timestamp_single_value_with_day_unit_run_with_optional_args(True, 12, 0)
        is None
    )


def test_pd_Timestamp_single_value_with_day_unit_optional_num_1():
    @bodo.jit
    def pd_Timestamp_single_value_with_day_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_day_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_day_unit_run_with_optional_args(
        False, 12, 1
    ) == pd.Timestamp(12, unit="D")
    assert (
        pd_Timestamp_single_value_with_day_unit_run_with_optional_args(True, 12, 1)
        is None
    )


def test_pd_Timestamp_single_value_with_year_unit_default_input_0():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
        100
    ) == pd.Timestamp(
        100, unit="Y"
    )


def test_pd_Timestamp_single_value_with_year_unit_default_input_1():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
        12
    ) == pd.Timestamp(
        12, unit="Y"
    )


def test_pd_Timestamp_single_value_with_year_unit_default_input_2():
    assert bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
        14
    ) == pd.Timestamp(
        14, unit="Y"
    )


def test_pd_Timestamp_single_value_with_year_unit_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
            None
        )
        is None
    )


def test_pd_Timestamp_single_value_with_year_unit_optional_num_0():
    @bodo.jit
    def pd_Timestamp_single_value_with_year_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_year_unit_run_with_optional_args(
        False, 14, 0
    ) == pd.Timestamp(14, unit="Y")
    assert (
        pd_Timestamp_single_value_with_year_unit_run_with_optional_args(True, 14, 0)
        is None
    )


def test_pd_Timestamp_single_value_with_year_unit_optional_num_1():
    @bodo.jit
    def pd_Timestamp_single_value_with_year_unit_run_with_optional_args(
        flag, arg0, optional_num
    ):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
                arg0
            )

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_pd_Timestamp_single_value_with_year_unit(
                arg0
            )

    assert pd_Timestamp_single_value_with_year_unit_run_with_optional_args(
        False, 14, 1
    ) == pd.Timestamp(14, unit="Y")
    assert (
        pd_Timestamp_single_value_with_year_unit_run_with_optional_args(True, 14, 1)
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


def test_pd_timedelta_total_seconds_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
            pd.Timedelta(100)
        )
        == 0
    )


def test_pd_timedelta_total_seconds_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
            pd.Timedelta(100, unit="s")
        )
        == 100
    )


def test_pd_timedelta_total_seconds_default_input_2():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
            pd.Timedelta(1, unit="D")
        )
        == 86400
    )


def test_pd_timedelta_total_seconds_None_Arg_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(None)
        is None
    )


def test_pd_timedelta_total_seconds_optional_num_0():
    @bodo.jit
    def pd_timedelta_total_seconds_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
                    arg0
                )
            )

        else:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
                    arg0
                )
            )

    assert (
        pd_timedelta_total_seconds_run_with_optional_args(False, pd.Timedelta(100), 0)
        == 0
    )
    assert (
        pd_timedelta_total_seconds_run_with_optional_args(True, pd.Timedelta(100), 0)
        is None
    )


def test_pd_timedelta_total_seconds_optional_num_1():
    @bodo.jit
    def pd_timedelta_total_seconds_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
                    arg0
                )
            )

        else:
            if flag:
                arg0 = None
            return (
                bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_total_seconds(
                    arg0
                )
            )

    assert (
        pd_timedelta_total_seconds_run_with_optional_args(False, pd.Timedelta(100), 1)
        == 0
    )
    assert (
        pd_timedelta_total_seconds_run_with_optional_args(True, pd.Timedelta(100), 1)
        is None
    )


def test_dayofyear_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_dayofyear(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 195
    )


def test_dayofyear_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_dayofyear(
            pd.Timestamp("2010-01-01")
        )
        == 1
    )


def test_dayofyear_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_dayofyear(None) is None


def test_dayofyear_optional_num_0():
    @bodo.jit
    def dayofyear_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_dayofyear(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_dayofyear(arg0)

    assert (
        dayofyear_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        == 195
    )
    assert (
        dayofyear_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        is None
    )


def test_dayofyear_optional_num_1():
    @bodo.jit
    def dayofyear_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_dayofyear(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_dayofyear(arg0)

    assert (
        dayofyear_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        == 195
    )
    assert (
        dayofyear_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        is None
    )


def test_microsecond_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_microsecond(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 498582
    )


def test_microsecond_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_microsecond(
            pd.Timestamp("2010-01-01")
        )
        == 0
    )


def test_microsecond_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_microsecond(None) is None


def test_microsecond_optional_num_0():
    @bodo.jit
    def microsecond_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_microsecond(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_microsecond(arg0)

    assert microsecond_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 0) == 0
    assert (
        microsecond_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 0) is None
    )


def test_microsecond_optional_num_1():
    @bodo.jit
    def microsecond_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_microsecond(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_microsecond(arg0)

    assert microsecond_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 1) == 0
    assert (
        microsecond_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 1) is None
    )


def test_second_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_second(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 4
    )


def test_second_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_second(pd.Timestamp("2010-01-01"))
        == 0
    )


def test_second_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_second(None) is None


def test_second_optional_num_0():
    @bodo.jit
    def second_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_second(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_second(arg0)

    assert (
        second_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        == 4
    )
    assert (
        second_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        is None
    )


def test_second_optional_num_1():
    @bodo.jit
    def second_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_second(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_second(arg0)

    assert (
        second_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        == 4
    )
    assert (
        second_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        is None
    )


def test_minute_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_minute(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 44
    )


def test_minute_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_minute(pd.Timestamp("2010-01-01"))
        == 0
    )


def test_minute_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_minute(None) is None


def test_minute_optional_num_0():
    @bodo.jit
    def minute_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_minute(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_minute(arg0)

    assert (
        minute_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        == 44
    )
    assert (
        minute_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        is None
    )


def test_minute_optional_num_1():
    @bodo.jit
    def minute_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_minute(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_minute(arg0)

    assert (
        minute_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        == 44
    )
    assert (
        minute_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        is None
    )


def test_hour_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_hour(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 15
    )


def test_hour_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_hour(pd.Timestamp("2010-01-01"))
        == 0
    )


def test_hour_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_hour(None) is None


def test_hour_optional_num_0():
    @bodo.jit
    def hour_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_hour(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_hour(arg0)

    assert hour_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 0) == 0
    assert hour_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 0) is None


def test_hour_optional_num_1():
    @bodo.jit
    def hour_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_hour(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_hour(arg0)

    assert hour_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 1) == 0
    assert hour_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 1) is None


def test_day_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_day(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 14
    )


def test_day_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_day(pd.Timestamp("2010-01-01"))
        == 1
    )


def test_day_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_day(None) is None


def test_day_optional_num_0():
    @bodo.jit
    def day_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_day(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_day(arg0)

    assert day_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 0) == 1
    assert day_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 0) is None


def test_day_optional_num_1():
    @bodo.jit
    def day_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_day(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_day(arg0)

    assert day_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 1) == 1
    assert day_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 1) is None


def test_month_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_month(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 7
    )


def test_month_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_month(pd.Timestamp("2010-01-01"))
        == 1
    )


def test_month_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_month(None) is None


def test_month_optional_num_0():
    @bodo.jit
    def month_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_month(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_month(arg0)

    assert month_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 0) == 1
    assert month_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 0) is None


def test_month_optional_num_1():
    @bodo.jit
    def month_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_month(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_month(arg0)

    assert month_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 1) == 1
    assert month_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 1) is None


def test_quarter_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_quarter(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 3
    )


def test_quarter_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_quarter(pd.Timestamp("2010-01-01"))
        == 1
    )


def test_quarter_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_quarter(None) is None


def test_quarter_optional_num_0():
    @bodo.jit
    def quarter_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_quarter(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_quarter(arg0)

    assert (
        quarter_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        == 3
    )
    assert (
        quarter_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 0
        )
        is None
    )


def test_quarter_optional_num_1():
    @bodo.jit
    def quarter_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_quarter(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_quarter(arg0)

    assert (
        quarter_run_with_optional_args(
            False, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        == 3
    )
    assert (
        quarter_run_with_optional_args(
            True, pd.Timestamp("2021-07-14 15:44:04.498582"), 1
        )
        is None
    )


def test_year_default_input_0():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_year(
            pd.Timestamp("2021-07-14 15:44:04.498582")
        )
        == 2021
    )


def test_year_default_input_1():
    assert (
        bodosql.libs.generated_lib.sql_null_checking_year(pd.Timestamp("2010-01-01"))
        == 2010
    )


def test_year_None_Arg_0():
    assert bodosql.libs.generated_lib.sql_null_checking_year(None) is None


def test_year_optional_num_0():
    @bodo.jit
    def year_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_year(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_year(arg0)

    assert year_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 0) == 2010
    assert year_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 0) is None


def test_year_optional_num_1():
    @bodo.jit
    def year_run_with_optional_args(flag, arg0, optional_num):

        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_year(arg0)

        else:
            if flag:
                arg0 = None
            return bodosql.libs.generated_lib.sql_null_checking_year(arg0)

    assert year_run_with_optional_args(False, pd.Timestamp("2010-01-01"), 1) == 2010
    assert year_run_with_optional_args(True, pd.Timestamp("2010-01-01"), 1) is None


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
