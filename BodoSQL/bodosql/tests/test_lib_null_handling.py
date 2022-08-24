# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of bodosql.libs.null_handling functions
"""

import bodosql
import numpy as np
import pytest

import bodo


@pytest.mark.slow
def test_scalar_not_bool():
    """
    Checks that scalar_nullable_logical_not works as expected on regular
    boolean values.
    """
    assert bodosql.libs.null_handling.scalar_nullable_logical_not(True) is False
    assert bodosql.libs.null_handling.scalar_nullable_logical_not(False) is True


@pytest.mark.slow
def test_scalar_not_none():
    """
    Checks that scalar_nullable_logical_not works as expected on None.
    """
    assert bodosql.libs.null_handling.scalar_nullable_logical_not(None) is None


@pytest.mark.slow
def test_scalar_not_optional():
    """
    Checks that scalar_nullable_logical_not works as expected on optional types.
    """

    @bodo.jit
    def test_impl(flag, value):
        # Function that generates an optional type and calls scalar_nullable_logical_not
        if flag:
            value = None
        return bodosql.libs.null_handling.scalar_nullable_logical_not(value)

    assert test_impl(True, True) is None
    assert test_impl(True, False) is None
    # This isn't an optional value, but its an edge case to check
    assert test_impl(True, None) is None
    assert test_impl(False, True) is False
    assert test_impl(False, False) is True
    # This isn't an optional value, but its an edge case to check
    assert test_impl(False, None) is None


@pytest.mark.slow
def test_scalar_add_default():
    """
    Checks that scalar_nullable_add works as expected on regular
    values.
    """
    assert bodosql.libs.null_handling.scalar_nullable_add(1, 4) == 5
    assert np.isclose(bodosql.libs.null_handling.scalar_nullable_add(1.4, 4), 5.4)
    assert np.isclose(bodosql.libs.null_handling.scalar_nullable_add(1.1, 4.6), 5.7)


@pytest.mark.slow
def test_scalar_add_none():
    """
    Checks that scalar_nullable_add works as expected on None.
    """
    assert bodosql.libs.null_handling.scalar_nullable_add(1, None) is None
    assert bodosql.libs.null_handling.scalar_nullable_add(None, 4) is None
    assert bodosql.libs.null_handling.scalar_nullable_add(None, None) is None


@pytest.mark.slow
def test_scalar_add_optional():
    """
    Checks that scalar_nullable_add works as expected on optional typee.
    """

    @bodo.jit
    def test_impl(flag, a, b, optional_num):
        # Function that generates an optional type and calls scalar_nullable_add
        # optional_num = 0 if a should be optional, 1 if b should be optional, 2 if both
        if optional_num == 0:
            if flag:
                a = None
            return bodosql.libs.null_handling.scalar_nullable_add(a, b)
        elif optional_num == 1:
            if flag:
                b = None
            return bodosql.libs.null_handling.scalar_nullable_add(a, b)
        else:
            if flag:
                a = None
                b = None
            return bodosql.libs.null_handling.scalar_nullable_add(a, b)

    assert test_impl(True, 1, 4, 0) is None
    assert test_impl(True, 1.4, 4, 1) is None
    assert test_impl(True, 1.1, 4.7, 2) is None
    assert test_impl(False, 1, 4, 0) == 5
    assert np.isclose(test_impl(False, 1.4, 4, 1), 5.4)
    assert np.isclose(test_impl(False, 1.1, 4.6, 2), 5.7)
    # This isn't an optional value, but its an edge case to check
    assert test_impl(False, 1.4, None, 2) is None
