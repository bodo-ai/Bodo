"""
Test correctness of SQL the flatten operation in BodoSQL
"""

import datetime

import pandas as pd
import pyarrow as pa
import pytest

from bodosql.tests.utils import check_query


def test_lateral_split_to_table(memory_leak_check):
    query = "SELECT lat.value, COUNT(*) FROM table1, LATERAL SPLIT_TO_TABLE(str_col, ';') lat GROUP BY 1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "STR_COL": pd.array(
                    [
                        "red;orange;yellow",
                        "red",
                        None,
                        "blue;green",
                        None,
                        None,
                        "",
                        "red;yellow;blue",
                        "red;green;blue",
                        None,
                        "green",
                        ";;;;",
                        ";red;red;red;;red;",
                    ],
                    dtype=pd.ArrowDtype(pa.string()),
                ),
            }
        )
    }
    answer = pd.DataFrame(
        {
            0: pd.array(
                ["red", "orange", "yellow", "green", "blue", ""],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            1: pd.array([8, 1, 2, 3, 3, 9], dtype=pd.ArrowDtype(pa.int64())),
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT int_col as R, lat.index as I, lat.value as V FROM table1, lateral flatten(arr_col) lat",
            pd.DataFrame(
                {
                    "R": pd.array(
                        [0, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 7, 9],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "I": pd.array(
                        [0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 0],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "V": pd.array(
                        [1, 2, 3, 4, 5, None, 7, 8, 9, 10, 11, 12, 13, 14],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="flatten_array-output_index_value-replicate_int",
        ),
        pytest.param(
            "SELECT int_col as R, lat.index as I, lat.value as V, lat.this as T FROM table1, lateral flatten(input=>arr_col, outer=>true) lat",
            pd.DataFrame(
                {
                    "R": pd.array(
                        [0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 7, 8, 9],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "I": pd.array(
                        [0, None, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, None, 0],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "V": pd.array(
                        [1, None, 2, 3, 4, 5, None, 7, 8, 9, 10, 11, 12, 13, None, 14],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "T": pd.array(
                        [
                            [1],
                            [],
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
                            None,
                            [14],
                        ],
                        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                    ),
                }
            ),
            id="flatten_array-output_index_value_this-replicate_int-outer",
        ),
        pytest.param(
            "SELECT str_col as S, lat.this as T FROM table1, lateral flatten(arr_col) lat",
            pd.DataFrame(
                {
                    "S": pd.array(
                        [
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
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                    "T": pd.array(
                        [
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
                        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                    ),
                }
            ),
            id="flatten_array-output_this-replicate_string",
        ),
        pytest.param(
            "SELECT int_col as R, lat.value as V FROM table1, lateral flatten(INPUT=>split(str_col, ' ')) lat",
            pd.DataFrame(
                {
                    "R": pd.array(
                        [
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
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "V": pd.array(
                        [
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
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                }
            ),
            id="split_string-output_value-replicate_int",
        ),
    ],
)
def test_lateral_flatten_arrays(query, answer, memory_leak_check):
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "ARR_COL": pd.array(
                    [
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
                    dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                ),
                "STR_COL": pd.array(
                    ["a b c", "d e f", "GHI", "we attack at dawn", ""] * 2,
                    dtype=pd.ArrowDtype(pa.string()),
                ),
                "INT_COL": pd.array(list(range(10)), dtype=pd.ArrowDtype(pa.int64())),
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
    "query, answer, is_out_distributed",
    [
        pytest.param(
            "SELECT 'A' as K, COUNT(*) as C FROM TABLE(GENERATOR(ROWCOUNT=>1776)) GROUP BY 1",
            pd.DataFrame(
                {
                    "K": pd.array(["A"], dtype=pd.ArrowDtype(pa.string())),
                    "C": pd.array([1776], dtype=pd.ArrowDtype(pa.int64())),
                }
            ),
            None,
            id="groupby_aggregate-without_lateral",
        ),
        pytest.param(
            "SELECT COUNT(*) as C FROM TABLE(GENERATOR(ROWCOUNT=>1776))",
            pd.DataFrame({"C": pd.array([1776], dtype=pd.ArrowDtype(pa.int64()))}),
            False,
            id="nogroupby_aggregate-without_lateral",
        ),
        pytest.param(
            "SELECT I, COUNT(*) as C FROM table1, LATERAL TABLE(GENERATOR(ROWCOUNT=>10)) GROUP BY I",
            pd.DataFrame(
                {
                    "I": pd.array([0, 1, 2], dtype=pd.ArrowDtype(pa.int64())),
                    "C": pd.array([50, 30, 10], dtype=pd.ArrowDtype(pa.int64())),
                }
            ),
            None,
            id="groupby_aggregate-with_lateral",
            marks=pytest.mark.skip(
                reason="[BSE-2309] Support zero-column table in streaming join"
            ),
        ),
        pytest.param(
            "SELECT DATEADD(week, 3 * ROW_NUMBER() OVER (ORDER BY NULL), DATE '2023-01-01') as dates FROM TABLE(GENERATOR(ROWCOUNT=>5))",
            pd.DataFrame(
                {
                    "dates": [
                        datetime.date(2023, 1, 22),
                        datetime.date(2023, 2, 12),
                        datetime.date(2023, 3, 5),
                        datetime.date(2023, 3, 26),
                        datetime.date(2023, 4, 16),
                    ]
                },
            ),
            None,
            id="row_number-without_lateral",
        ),
    ],
)
def test_generator(query, answer, is_out_distributed, memory_leak_check):
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I": pd.array(
                    [0, 0, 1, 0, 0, 1, 2, 1, 0], dtype=pd.ArrowDtype(pa.int64())
                )
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
        is_out_distributed=is_out_distributed,
    )


@pytest.mark.parametrize(
    "query, df, answer",
    [
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(J) lat",
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pd.ArrowDtype(pa.int64())
                    ),
                    "J": pd.Series(
                        [
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
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "K": pd.array(
                        list("ABILCDJMEFKGH"), dtype=pd.ArrowDtype(pa.string())
                    ),
                    "V": pd.array(
                        [0, 1, 8, 11, 2, 3, None, 12, 4, 5, 10, 6, 7],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="flatten_map-output_key_value-replicate_int",
        ),
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(J) lat",
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pd.ArrowDtype(pa.int64())
                    ),
                    "J": pd.Series(
                        [
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
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [pa.field("A", pa.int8()), pa.field("B", pa.int32())]
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 9, 9],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "K": pd.array(["A", "B"] * 9, dtype=pd.ArrowDtype(pa.string())),
                    "V": pd.array(
                        [
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
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="flatten_struct_nullable_int-output_key_value-replicate_int",
        ),
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(OBJECT_CONSTRUCT_KEEP_NULL('x-coordinate', I1, 'y-coordinate', I2)) lat",
            pd.DataFrame(
                {
                    "I": pd.array([0, 1, 2, 3, 4], dtype=pd.ArrowDtype(pa.int64())),
                    "I1": pd.array([1, 2, -4, 8, 16], dtype=pd.ArrowDtype(pa.int8())),
                    "I2": pd.array(
                        [9, 27, 81, None, 729], dtype=pd.ArrowDtype(pa.uint16())
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=pd.ArrowDtype(pa.int64())
                    ),
                    "K": pd.array(
                        ["x-coordinate", "y-coordinate"] * 5,
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                    "V": pd.array(
                        [
                            1,
                            9,
                            2,
                            27,
                            -4,
                            81,
                            8,
                            None,
                            16,
                            729,
                        ],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                }
            ),
            id="flatten_struct_numeric_mix-output_key_value-replicate_int",
        ),
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(OBJECT_CONSTRUCT_KEEP_NULL('S1', S1, 'S2', S2)) lat",
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pd.ArrowDtype(pa.int64())
                    ),
                    "S1": pd.array(
                        ["A", "B", "AB", "A", "B", None, "C", "AB", "A", "B"],
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                    "S2": pd.array(
                        ["A", "AB", "ABC", "A", "AB", "ABC", "A", "AB", "ABC", None],
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "I": pd.array(
                        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "K": pd.array(["S1", "S2"] * 10, dtype=pd.ArrowDtype(pa.string())),
                    "V": pd.array(
                        [
                            "A",
                            "A",
                            "B",
                            "AB",
                            "AB",
                            "ABC",
                            "A",
                            "A",
                            "B",
                            "AB",
                            None,
                            "ABC",
                            "C",
                            "A",
                            "AB",
                            "AB",
                            "A",
                            "ABC",
                            "B",
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                }
            ),
            id="flatten_struct_string-output_key_value-replicate_int",
        ),
        pytest.param(
            "SELECT F, lat.key as K, lat.value as V FROM table1, lateral flatten(OBJECT_CONSTRUCT_KEEP_NULL('regular', OBJECT_CONSTRUCT_KEEP_NULL('F', f, 'Fp', f+1), 'squared', OBJECT_CONSTRUCT_KEEP_NULL('F', f*f, 'Fp', f*f+1), 'cubed', OBJECT_CONSTRUCT_KEEP_NULL('F', POW(f, 0.5), 'Fp', POW(f, 0.5)+1))) lat",
            pd.DataFrame(
                {
                    "F": pd.array(
                        [0.0, 1.0, 4.0, 0.25, 25.0], dtype=pd.ArrowDtype(pa.float64())
                    )
                }
            ),
            pd.DataFrame(
                {
                    "F": pd.array(
                        list(pd.Series([0.0, 1.0, 4.0, 0.25, 25.0]).repeat(3).values),
                        dtype=pd.ArrowDtype(pa.float64()),
                    ),
                    "K": pd.array(
                        ["regular", "squared", "root"] * 5,
                        dtype=pd.ArrowDtype(pa.string()),
                    ),
                    "V": [
                        {"F": 0.0, "Fp": 1.0},
                        {"F": 0.0, "Fp": 1.0},
                        {"F": 0.0, "Fp": 1.0},
                        {"F": 1.0, "Fp": 2.0},
                        {"F": 1.0, "Fp": 2.0},
                        {"F": 1.0, "Fp": 2.0},
                        {"F": 4.0, "Fp": 5.0},
                        {"F": 16.0, "Fp": 17.0},
                        {"F": 2.0, "Fp": 3.0},
                        {"F": 0.25, "Fp": 1.25},
                        {"F": 0.0625, "Fp": 1.0625},
                        {"F": 0.5, "Fp": 1.5},
                        {"F": 25.0, "Fp": 26.0},
                        {"F": 625.0, "Fp": 626.0},
                        {"F": 5.0, "Fp": 6.0},
                    ],
                }
            ),
            id="flatten_struct_struct_float-output_key_value-replicate_float",
            marks=pytest.mark.skip(
                reason="[BSE-2102] TODO: support flatten on structs containing structs"
            ),
        ),
        pytest.param(
            "SELECT I, lat.key as K, lat.value as V FROM table1, lateral flatten(OBJECT_CONSTRUCT_KEEP_NULL('ordmap', M, 'wo_vowel', OBJECT_DELETE(M, 'A', 'E', 'I', 'O', 'U'))) lat",
            pd.DataFrame(
                {
                    "I": pd.array(
                        [10, 20, 30, 40, 50], dtype=pd.ArrowDtype(pa.int64())
                    ),
                    "M": pd.Series(
                        [
                            {char: ord(char) for char in word}
                            for word in "ABC A  AEIOU RSTLNE".split(" ")
                        ],
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "F": pd.array(
                        list(pd.Series([10, 20, 30, 40, 50]).repeat(2).values),
                        dtype=pd.ArrowDtype(pa.int64()),
                    ),
                    "K": pd.array(
                        ["ordmap", "wo_vowel"] * 5, dtype=pd.ArrowDtype(pa.string())
                    ),
                    "V": pd.Series(
                        [
                            {"A": 65, "B": 66, "C": 67},
                            {"B": 66, "C": 67},
                            {"A": 65},
                            {},
                            {},
                            {},
                            {"A": 65, "E": 69, "I": 73, "O": 79, "U": 85},
                            {},
                            {"R": 82, "S": 83, "T": 84, "L": 76, "N": 78, "E": 69},
                            {"R": 82, "S": 83, "T": 84, "L": 76, "N": 78},
                        ],
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                    ),
                }
            ),
            id="flatten_struct_map_int-output_key_value-replicate_int",
            marks=pytest.mark.skip(
                reason="[BSE-2102] TODO: support flatten on structs containing maps"
            ),
        ),
    ],
)
def test_lateral_flatten_json(query, df, answer, memory_leak_check):
    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        sort_output=False,  # Sorting semi-structured data unsupported in Python
    )
