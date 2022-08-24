# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Checks the Library of BodoSQL functions that are used for performing "AND" and "OR"
    operations involving potentially null values
"""
import bodosql
import pytest

import bodo


@pytest.mark.slow
def test_and_non_null_inputs():
    """tests the and implementation on non-null values"""
    for a in [True, False]:
        for b in [True, False]:
            assert (
                a and b
            ) == bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
                a, b
            )


@pytest.mark.slow
def test_or_non_null_inputs():
    """tests the or implementation on non-null values"""
    for a in [True, False]:
        for b in [True, False]:
            assert (
                a or b
            ) == bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
                a, b
            )


@pytest.mark.slow
def test_and_null_inputs():
    """tests the and implementation on null values"""
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
            None, False
        )
        is None
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
            None, True
        )
        is None
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
            False, None
        )
        is False
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
            True, None
        )
        is None
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
            None, None
        )
        is None
    )


@pytest.mark.slow
def test_or_null_inputs():
    """tests the or implementation on null values"""
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
            None, False
        )
        is None
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
            None, True
        )
        is True
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
            False, None
        )
        is None
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
            True, None
        )
        is True
    )
    assert (
        bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
            None, None
        )
        is None
    )


@pytest.mark.slow
def test_and_optional_inputs():
    """tests the and implementation on optional values"""

    @bodo.jit
    def and_run_with_optional_args(flag, arg0, arg1, optional_num):
        if optional_num == 0:
            if flag:
                arg0 = None
            return bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
                arg0, arg1
            )
        elif optional_num == 1:
            if flag:
                arg1 = None
            return bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
                arg0, arg1
            )
        else:
            if flag:
                arg0 = None
                arg1 = None
            return bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_and(
                arg0, arg1
            )

    for flag in [True, False]:
        for a in [None, True, False]:
            for b in [None, True, False]:
                for i in [0, 1, 2]:
                    fn_result = and_run_with_optional_args(flag, a, b, i)
                    a_is_none = a is None or (flag and (i == 0 or i == 2))
                    b_is_none = b is None or (flag and (i == 1 or i == 2))
                    if a_is_none:
                        expected_output = None
                    elif b_is_none:
                        # a is not none
                        if a is False:
                            expected_output = False
                        else:
                            expected_output = None
                    else:
                        expected_output = a and b

                    assert fn_result == expected_output


@pytest.mark.slow
def test_or_optional_inputs():
    """tests the or implementation on optional values"""

    @bodo.jit
    def or_run_with_optional_args(flag, arg0, arg1, optional_num):
        if optional_num == 0:
            if flag:
                arg0 = None
            return (
                bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
                    arg0, arg1
                )
            )
        elif optional_num == 1:
            if flag:
                arg1 = None
            return (
                bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
                    arg0, arg1
                )
            )
        else:
            if flag:
                arg0 = None
                arg1 = None
            return (
                bodosql.libs.nullchecked_logical_operators.mysql_nullchecking_scalar_or(
                    arg0, arg1
                )
            )

    for a in [True, False]:
        for b in [True, False]:
            for i in [0, 1, 2]:
                assert or_run_with_optional_args(False, a, b, i) == (a or b)

    for flag in [True, False]:
        for a in [True, False]:
            for b in [True, False]:
                for i in [0, 1, 2]:
                    fn_result = or_run_with_optional_args(flag, a, b, i)
                    a_is_none = a is None or (flag and (i == 0 or i == 2))
                    b_is_none = b is None or (flag and (i == 1 or i == 2))
                    a_is_true = (not a_is_none) and a
                    b_is_true = (not b_is_none) and b

                    if a_is_true or b_is_true:
                        expected_result = True
                    elif a_is_none or b_is_none:
                        # a and b are both non-true
                        expected_result = None
                    else:
                        # a and b are both False
                        expected_result = False

                    assert fn_result == expected_result
