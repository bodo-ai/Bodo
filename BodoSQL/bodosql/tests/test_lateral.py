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


@pytest.mark.parametrize(
    "query, df, use_map_arrays, answer",
    [
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(J) lat",
            pd.DataFrame(
                {
                    "I": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "J": [
                        {"A": 0},
                        None,
                        {"B": 1, "I": 8, "L": 11},
                        {"C": 2},
                        {"D": 3, "J": None, "M": 12},
                        {"E": 4},
                        {"F": 5, "K": 10},
                        {"G": 6},
                        {},
                        {"H": 7},
                    ],
                }
            ),
            True,
            pd.DataFrame(
                {
                    "I": [0, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9],
                    "K": list("ABILCDJMEFKGH"),
                    "V": [0, 1, 8, 11, 2, 3, None, 12, 4, 5, 10, 6, 7],
                }
            ),
            id="flatten_map-output_key_value-replicate_int",
        ),
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(J) lat",
            pd.DataFrame(
                {
                    "I": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "J": [
                        {"A": 0, "B": 9},
                        {"A": 1, "B": 10},
                        {"A": 2, "B": 11},
                        {"A": 3, "B": 12},
                        {"A": 4, "B": 13},
                        {"A": 5, "B": 14},
                        {"A": 6, "B": 15},
                        {"A": 7, "B": 16},
                        None,
                        {"A": 8, "B": None},
                    ],
                }
            ),
            False,
            pd.DataFrame(
                {
                    "I": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 9, 9],
                    "K": ["A", "B"] * 9,
                    "V": [
                        0,
                        9,
                        1,
                        10,
                        2,
                        11,
                        3,
                        12,
                        4,
                        13,
                        5,
                        14,
                        6,
                        15,
                        7,
                        16,
                        8,
                        None,
                    ],
                }
            ),
            id="flatten_struct-output_key_value-replicate_int",
            marks=pytest.mark.skip(
                reason="[BSE-2001] Support flatten kernel on JSON data with struct arrays"
            ),
        ),
    ],
)
def test_lateral_flatten_json(query, df, use_map_arrays, answer, memory_leak_check):
    ctx = {"table1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        use_map_arrays=use_map_arrays,
        # Can't use check_python because of intricacies of unboxing map arrays
        only_jit_1DVar=True,
        sort_output=False,  # Sorting semi-structured data unsupported in Python
    )
