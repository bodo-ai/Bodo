import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(
            (
                [
                    0,
                    -1,
                    2,
                    -4,
                    8,
                    -16,
                    32,
                    -64,
                    128,
                ],
                pd.Int32Dtype(),
                pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="int64",
        ),
        pytest.param(
            (
                [
                    "",
                    "A",
                    "BC",
                    "DEF",
                    "AB",
                    "CD",
                    "EF",
                    "ABC",
                    "E",
                ],
                None,
                pd.ArrowDtype(pa.map_(pa.string(), pa.large_string())),
            ),
            id="string",
        ),
        pytest.param(
            (
                [
                    [1],
                    [],
                    [2],
                    [3, 4],
                    [None],
                    [5, None, 6],
                    [7, 8, 1, 9, 8],
                    [3],
                    [1, 2],
                ],
                pd.ArrowDtype(pa.list_(pa.int64())),
                pd.ArrowDtype(pa.map_(pa.string(), pa.list_(pa.int64()))),
            ),
            id="array_int",
        ),
        pytest.param(
            (
                [
                    ["A"],
                    [],
                    [""],
                    ["A", ""],
                    [None],
                    ["A", None, ""],
                    ["B", "CD", "", "EFGH", "CD"],
                    ["CD"],
                    ["", "A"],
                ],
                pd.ArrowDtype(pa.list_(pa.string())),
                pd.ArrowDtype(pa.map_(pa.string(), pa.list_(pa.string()))),
            ),
            id="array_string",
        ),
        pytest.param(
            (
                [
                    {"lat": 40.6892, "lon": 74.0445, "name": "Statue of Liberty"},
                    {"lat": 27.1715, "lon": 78.0421, "name": "Taj Mahal"},
                    {"lat": 37.8199, "lon": -122.4783, "name": "Golden Gate Bridge"},
                    {"lat": -22.9519, "lon": -43.2105, "name": "Christ the Redeemer"},
                    {"lat": 51.5007, "lon": -0.1246, "name": "Big Ben"},
                    {"lat": -33.8568, "lon": 151.2153, "name": "Sydney Opera House"},
                    {"lat": 29.9792, "lon": 31.1342, "name": "Great Pyramid of Giza"},
                    {"lat": 41.4036, "lon": 2.1744, "name": "Sagrada Familia"},
                    {"lat": 13.4125, "lon": 103.8670, "name": "Angkor Wat"},
                ],
                pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("lat", pa.float64()),
                            pa.field("lon", pa.float64()),
                            pa.field("name", pa.string()),
                        ]
                    )
                ),
                pd.ArrowDtype(
                    pa.map_(
                        pa.string(),
                        pa.struct(
                            [
                                pa.field("lat", pa.float64()),
                                pa.field("lon", pa.float64()),
                                pa.field("name", pa.string()),
                            ]
                        ),
                    )
                ),
            ),
            id="struct_simple",
        ),
        pytest.param(
            (
                [
                    {
                        "Exams": [
                            {"Topic": "History", "Score": 93.4},
                            {"Topic": "English", "Score": 85.1},
                        ],
                        "Grade": "B+",
                    },
                    {"Exams": [{"Topic": "Physics", "Score": 94.5}], "Grade": "A"},
                    {"Exams": [], "Grade": "C+"},
                    {
                        "Exams": [
                            {"Topic": "Astronomy", "Score": 99.4},
                            {"Topic": "Physics", "Score": 85.0},
                        ],
                        "Grade": "A-",
                    },
                    {
                        "Exams": [
                            {"Topic": "History", "Score": 93.4},
                            {"Topic": "Physics", "Score": 68.5},
                        ],
                        "Grade": "B-",
                    },
                    {"Exams": [{"Topic": "English", "Score": 85.4}], "Grade": "B"},
                    {"Exams": [{"Topic": "English", "Score": 98.9}], "Grade": "A+"},
                    {"Exams": [{"Topic": "Physics", "Score": 88.7}], "Grade": "B+"},
                    {
                        "Exams": [
                            {"Topic": "History", "Score": 98.4},
                            {"Topic": "Physics", "Score": 90.0},
                        ],
                        "Grade": "A",
                    },
                ],
                pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "Exams",
                                pa.large_list(
                                    pa.struct(
                                        [
                                            pa.field("Topic", pa.string()),
                                            pa.field("Score", pa.float64()),
                                        ]
                                    )
                                ),
                            ),
                            pa.field("Grade", pa.string()),
                        ]
                    )
                ),
                pd.ArrowDtype(
                    pa.map_(
                        pa.string(),
                        pa.struct(
                            [
                                pa.field(
                                    "Exams",
                                    pa.list_(
                                        pa.struct(
                                            [
                                                pa.field("Topic", pa.string()),
                                                pa.field("Score", pa.float64()),
                                            ]
                                        )
                                    ),
                                ),
                                pa.field("Grade", pa.string()),
                            ]
                        ),
                    )
                ),
            ),
            id="struct_nested",
        ),
        pytest.param(
            (
                [
                    {"A": 65},
                    {"B": 66, "I": 73},
                    {},
                    {"C": 67, "J": 74, "N": 78, "Q": 81},
                    {"D": 68, "K": 75, "O": 79},
                    {"F": 69},
                    {"G": 70},
                    {"A": 71, "L": 76, "P": 80, "R": 82, "S": 83},
                    {"H": 72, "M": 77},
                ],
                pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                pd.ArrowDtype(pa.map_(pa.string(), pa.map_(pa.string(), pa.int64()))),
            ),
            id="map_simple",
        ),
    ]
)
def object_agg_data(request):
    """
    Produces the data used for OBJECT_DATA tests with various dtypes.
    Parametrizes on the following:

    value_pool: a list of 9 values of a specific dtype that are used
    to construct the input column.
    """
    vals, val_dtype, out_dtype = request.param
    value_pool = vals[:5] + [None] + vals[5:]
    group_keys_unique = list("ABCDEFGHIJ")
    counts = [0] * len(group_keys_unique)
    group_keys = []
    json_keys = []
    json_values = []
    for i in range(100):
        prng_a = ((i**9) % 29) % len(group_keys_unique)
        prng_b = (i**15) % len(group_keys_unique)
        group_key_idx = min(prng_a, prng_b)
        group_key = group_keys_unique[group_key_idx]
        counts[group_key_idx] += 1
        json_key = (
            None if counts[group_key_idx] == 2 else f"k{counts[group_key_idx] ** 2}"
        )
        json_value = value_pool[max(prng_a, prng_b)]
        group_keys.append(group_key)
        json_keys.append(json_key)
        json_values.append(json_value)

    group_keys_unique.append("K")
    for idx in [80, 42]:
        group_keys.insert(idx, "K")
        json_keys.insert(idx, None)
        json_values.insert(idx, None)

    in_data = pd.DataFrame(
        {
            "group_key": group_keys,
            "json_key": json_keys,
            "json_value": pd.Series(json_values, dtype=val_dtype),
        }
    )

    json_out = []
    for group_key in group_keys_unique:
        json_obj = {}
        for i in range(len(group_keys)):
            if group_keys[i] == group_key:
                json_key = json_keys[i]
                json_value = json_values[i]
                if json_key is None or json_value is None:
                    continue
                json_obj[json_key] = json_value
        json_out.append(json_obj)

    # Use Arrow MapArray to match Bodo output
    json_out = pd.Series(json_out, dtype=out_dtype)

    expected_answer = pd.DataFrame(
        {
            "group_key": group_keys_unique,
            "res": json_out,
        }
    )
    return in_data, expected_answer


def test_object_agg(object_agg_data, memory_leak_check):
    """
    Tests that calling OBJECT_AGG on various input data types
    """
    in_data, expected_answer = object_agg_data

    def impl(df):
        return df.groupby(
            ["group_key"], as_index=False, dropna=False, _is_bodosql=True
        ).agg(
            res=bodo.utils.utils.ExtendedNamedAgg(
                column="json_value", aggfunc="object_agg", additional_args=("json_key",)
            ),
        )

    check_func(
        impl,
        (in_data,),
        py_output=expected_answer,
        sort_output=True,
        reset_index=True,
        convert_columns_to_pandas=True,
    )
