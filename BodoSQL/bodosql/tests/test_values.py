# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests for the SQL Values syntax that require end to end testing.
"""

import pandas as pd

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


def test_values_multiple_rows(memory_leak_check):
    """
    Tests an implementation of values with multiple rows
    and columns.
    """
    query = """
        select
        *
        from (
        values
            ('pasta', 1, 'pasta'),
            ('pf candle', 2, 'pf candle'),
            ('Fête des mères', 3, 'mothers day'),
            ('Fête des Mères', 3, 'mothers day'),
            ('fête des mères', 3, 'mothers day'),
            ('Fete des meres', 3, 'mothers day'),
            ('Fete des Meres', 3, NULL),
            ('fete des meres', 3, 'mothers day'),
            ('Fête des pères', 3, 'fathers day'),
            ('Fête des Pères', 3, 'fathers day'),
            ('fête des pères', 3, 'fathers day'),
            ('Fete des peres', 3, 'fathers day'),
            ('Fete des Peres', 3, NULL),
            ('fete des peres', 3, 'fathers day')
        ) tbl (col1, col2, col3)
    """
    expected_output = pd.DataFrame(
        {
            "COL1": pd.array(
                [
                    "pasta",
                    "pf candle",
                    "Fête des mères",
                    "Fête des Mères",
                    "fête des mères",
                    "Fete des meres",
                    "Fete des Meres",
                    "fete des meres",
                    "Fête des pères",
                    "Fête des Pères",
                    "fête des pères",
                    "Fete des peres",
                    "Fete des Peres",
                    "fete des peres",
                ]
            ),
            "COL2": pd.array([1, 2] + [3] * 12),
            "COL3": pd.array(
                [
                    "pasta",
                    "pf candle",
                    "mothers day",
                    "mothers day",
                    "mothers day",
                    "mothers day",
                    None,
                    "mothers day",
                    "fathers day",
                    "fathers day",
                    "fathers day",
                    "fathers day",
                    None,
                    "fathers day",
                ]
            ),
        }
    )
    check_query(
        query,
        {},
        None,
        expected_output=expected_output,
        check_dtype=False,
    )
