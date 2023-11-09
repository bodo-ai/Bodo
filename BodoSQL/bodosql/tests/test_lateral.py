# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test correctness of SQL the flatten operation in BodoSQL
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT int_col as R, lat.index as I, lat.value as V FROM table1, lateral flatten(arr_col) lat",
            pd.DataFrame(
                {
                    "R": [0, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 7, 9],
                    "I": [0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 0],
                    "V": [1, 2, 3, 4, 5, None, 7, 8, 9, 10, 11, 12, 13, 14],
                }
            ),
            id="flatten_array-output_index_value-replicate_int",
        ),
        pytest.param(
            "SELECT str_col as S, lat.this as T FROM table1, lateral flatten(arr_col) lat",
            pd.DataFrame(
                {
                    "S": [
                        "a b c",
                        "GHI",
                        "GHI",
                        "we attack at dawn",
                        "we attack at dawn",
                        "we attack at dawn",
                        "",
                        "a b c",
                        "a b c",
                        "d e f",
                        "d e f",
                        "d e f",
                        "GHI",
                        "",
                    ],
                    "T": [
                        [1],
                        [2, 3],
                        [2, 3],
                        [4, 5, None],
                        [4, 5, None],
                        [4, 5, None],
                        [7],
                        [8, 9],
                        [8, 9],
                        [10, 11, 12],
                        [10, 11, 12],
                        [10, 11, 12],
                        [13],
                        [14],
                    ],
                }
            ),
            id="flatten_array-output_this-replicate_string",
        ),
        pytest.param(
            "SELECT int_col as R, lat.value as V FROM table1, lateral flatten(INPUT=>split(str_col, ' ')) lat",
            pd.DataFrame(
                {
                    "R": [
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        2,
                        3,
                        3,
                        3,
                        3,
                        4,
                        5,
                        5,
                        5,
                        6,
                        6,
                        6,
                        7,
                        8,
                        8,
                        8,
                        8,
                        9,
                    ],
                    "V": [
                        "a",
                        "b",
                        "c",
                        "d",
                        "e",
                        "f",
                        "GHI",
                        "we",
                        "attack",
                        "at",
                        "dawn",
                        "",
                    ]
                    * 2,
                }
            ),
            id="split_string-output_value-replicate_int",
            marks=pytest.mark.skip(reason="Skip until [BSE-1746] is merged"),
        ),
    ],
)
def test_lateral_flatten_arrays(query, answer, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "arr_col": [
                    [1],
                    [],
                    [2, 3],
                    [4, 5, None],
                    [7],
                    [8, 9],
                    [10, 11, 12],
                    [13],
                    None,
                    [14],
                ],
                "str_col": ["a b c", "d e f", "GHI", "we attack at dawn", ""] * 2,
                "int_col": list(range(10)),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        sort_output=False,  # Sorting semi-structured data unsupported in Python
    )
