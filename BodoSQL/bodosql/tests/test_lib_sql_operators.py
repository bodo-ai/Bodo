import bodo
import pandas as pd
import pytest

import bodosql.libs.sql_operators


@bodo.jit
def sql_null_equal_column_impl(val1, val2):
    # This function is used to test sql_null_equal_column because
    # an overload cannot be called from regular Python
    return bodosql.libs.sql_operators.sql_null_equal_column(val1, val2)


@pytest.mark.slow
def test_null_equals_columns():
    """Tests for <=> library function on Series"""
    S1 = pd.Series([True, False, None] * 3, dtype="boolean")
    S2 = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    expected_output = pd.Series(
        [True, False, False, False, True, False, False, False, True], dtype="boolean"
    )
    pd.testing.assert_series_equal(sql_null_equal_column_impl(S1, S2), expected_output)
    S1 = pd.Series([-12, 0, None] * 3, dtype="Int32")
    S2 = pd.Series([-12] * 3 + [0] * 3 + [None] * 3, dtype="Int64")
    pd.testing.assert_series_equal(sql_null_equal_column_impl(S1, S2), expected_output)


@pytest.mark.slow
def test_null_equals_column_null_scalar():
    """Tests for <=> library function on a column and NULL scalar"""
    val = None
    S1 = pd.Series([True, False, None] * 3, dtype="boolean")
    expected_output = pd.Series([False, False, True] * 3, dtype="boolean")
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(S1, val), expected_output, check_dtype=False
    )
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(val, S1), expected_output, check_dtype=False
    )
    S1 = pd.Series([-12, 0, None] * 3, dtype="Int32")
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(S1, val), expected_output, check_dtype=False
    )
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(val, S1), expected_output, check_dtype=False
    )


@pytest.mark.slow
def test_null_equals_column_scalar():
    """Tests for <=> library function on a column and non-null scalar"""
    val = True
    S1 = pd.Series([True, False, None] * 3, dtype="boolean")
    expected_output = pd.Series([True, False, False] * 3, dtype="boolean")
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(S1, val), expected_output, check_dtype=False
    )
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(val, S1), expected_output, check_dtype=False
    )
    val = -12
    S1 = pd.Series([-12, 0, None] * 3, dtype="Int32")
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(S1, val), expected_output, check_dtype=False
    )
    pd.testing.assert_series_equal(
        sql_null_equal_column_impl(val, S1), expected_output, check_dtype=False
    )


@pytest.mark.slow
def test_null_equals_column_optional_scalar():
    """Tests for <=> library function on a column and optional scalar"""

    @bodo.jit
    def test_impl_first(flag, val1, val2):
        # Function that generates an optional type and calls sql_null_equal_column
        if flag == 0:
            val1 = None
        return bodosql.libs.sql_operators.sql_null_equal_column(val1, val2)

    @bodo.jit
    def test_impl_second(flag, val1, val2):
        # Function that generates an optional type and calls sql_null_equal_column
        if flag == 0:
            val2 = None
        return bodosql.libs.sql_operators.sql_null_equal_column(val1, val2)

    val = True
    S1 = pd.Series([True, False, None] * 3, dtype="boolean")
    expected_output_nonnull = pd.Series([True, False, False] * 3, dtype="boolean")
    expected_output_null = pd.Series([False, False, True] * 3, dtype="boolean")
    pd.testing.assert_series_equal(
        test_impl_second(0, S1, val), expected_output_null, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_second(-1, S1, val), expected_output_nonnull, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_first(0, val, S1), expected_output_null, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_first(-1, val, S1), expected_output_nonnull, check_dtype=False
    )
    val = -12
    S1 = pd.Series([-12, 0, None] * 3, dtype="Int32")
    pd.testing.assert_series_equal(
        test_impl_second(0, S1, val), expected_output_null, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_second(-1, S1, val), expected_output_nonnull, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_first(0, val, S1), expected_output_null, check_dtype=False
    )
    pd.testing.assert_series_equal(
        test_impl_first(-1, val, S1), expected_output_nonnull, check_dtype=False
    )


@pytest.mark.slow
def test_null_equals_scalars_none():
    """
    Checks that sql_null_equal_scalar works as expected with NULL scalars.
    """
    val1 = None
    val2 = None
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val1, val2) is True
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val2, val1) is True
    val2 = False
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val1, val2) is False
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val2, val1) is False


@pytest.mark.slow
def test_null_equals_scalars_val():
    """
    Checks that sql_null_equal_scalar works as expected with non-NULL scalars.
    """
    val1 = -1
    val2 = -2
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val1, val2) is False
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val2, val1) is False
    val2 = -1
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val1, val2) is True
    assert bodosql.libs.sql_operators.sql_null_equal_scalar(val2, val1) is True


@pytest.mark.slow
def test_null_equals_scalars_optional():
    """
    Checks that sql_null_equal_scalar works as expected on optional types.
    """

    @bodo.jit
    def test_impl(flag, val1, val2):
        # Function that generates an optional type and calls sql_null_equal_column
        if flag == 0:
            val1 = None
        elif flag == 1:
            val2 = None
        elif flag == 2:
            val1 = None
            val2 = None
        return bodosql.libs.sql_operators.sql_null_equal_scalar(val1, val2)

    val1 = -1
    val2 = -2
    assert test_impl(-1, val1, val2) is False
    assert test_impl(0, val1, val2) is False
    assert test_impl(1, val1, val2) is False
    assert test_impl(2, val1, val2) is True
    assert test_impl(-1, val1, val1) is True
    assert test_impl(0, val1, val1) is False
    assert test_impl(1, val1, val1) is False
    assert test_impl(2, val1, val1) is True
    assert test_impl(-1, val1, None) is False
    assert test_impl(0, val1, None) is True
