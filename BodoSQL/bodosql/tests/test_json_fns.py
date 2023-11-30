import pandas as pd
import pyarrow as pa
import pytest

from bodosql.tests.utils import check_query


@pytest.fixture
def json_nested_array_data():
    """JSON data with arrays nested at various levels"""
    return '[["alpha",[["beta","gamma"],["delta"]],[["epsilon","zeta","eta"],[]],"theta","iota"],"kappa",[[["lambdba"],["mu","nu"]],[[],["xi","omicron","pi"]]],["rho",[["sigma"],["tau","upsilon"],"phi"],"chi",[["psi"],[]]],"omega"]'


@pytest.fixture
def json_nested_map_data():
    """JSON data with objects nested at various levels"""
    return '{"California": {\'Capital\': "Sacramento", "Senators": [{"First": "Dianne", \'Last\': "Feinstein"}, {"First": "Alex", \'Last\': "Padilla"}], "Bird": "quail", "Nickname": "golden"},"Georgia": {\'Capital\': "Atlanta", "Senators": [{"First": "Raphael", \'Last\': "Warnock"}, {"First": "Jon", \'Last\': "Ossof"}], "Bird": "thrasher", "Nickname": "peach"},"Texas": {\'Capital\': "Austin", "Senators": [{"First": "Ted", \'Last\': "Cruz"}, {"First": "John", \'Last\': "Cornyn"}], "Bird": "mockingbird", "Nickname": "lone star"}}'


@pytest.fixture
def json_senator_data(datapath):
    """Large sample JSON dataset with an array of objects"""
    with open(datapath("json_data/senators.txt"), "r") as f:
        return f.read()


@pytest.fixture
def json_extract_path_args(
    json_nested_array_data, json_nested_map_data, json_senator_data
):
    data = (
        [json_nested_array_data] * 12
        + [json_nested_map_data] * 12
        + [json_senator_data] * 4
    )
    path = [
        # Nested array tests
        "[0][0]",
        "[0][1][0][0]",
        "[0][1][1][0]",
        "[0][2][0][2]",
        "[0][4]",
        "[1]",
        "[2][1][1][2]",
        "[3][0]",
        "[3][1][0][0]",
        "[3][1][2]",
        "[3][3][0][0]",
        "[4]",
        # Nested map tests
        "California.Capital",
        '["California"]["Bird"]',
        'California["Nickname"]',
        '["Georgia"].Senators[1].First',
        '["Georgia"][\'Senators\'][0]["Last"]',
        "Georgia.Nickname",
        'Texas["Capital"]',
        "Texas.Senators[0].Last",
        "Texas.Bird",
        # Invalid Tests
        'Texas["Tree"]',
        "Alaska.Nickname",
        "Texas.Bird.Genus",
        # Senator data tests
        "objects[12].person.name",
        "objects[99].extra.office",
        "['meta'][\"limit\"]",
        '["objects"][67].person["twitterid"]',
    ]
    answer = [
        # Nested array tests
        "alpha",
        "beta",
        "delta",
        "eta",
        "iota",
        "kappa",
        "pi",
        "rho",
        "sigma",
        "phi",
        "psi",
        "omega",
        # Nested map tests
        "Sacramento",
        "quail",
        "golden",
        "Jon",
        "Warnock",
        "peach",
        "Austin",
        "Cruz",
        "mockingbird",
        # Invalid Tests
        None,
        None,
        None,
        # Senator data tests
        "Sen. Mazie Hirono [D-HI]",
        "B40b Dirksen Senate Office Building",
        "100",
        "JohnBoozman",
    ]
    return (data, path, answer)


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
def test_json_extract_path_text(json_extract_path_args, use_case, memory_leak_check):
    data, path, answer = json_extract_path_args
    if use_case:
        query = "SELECT I, CASE WHEN B THEN NULL ELSE JSON_EXTRACT_PATH_TEXT(D, P) END FROM TABLE1"
    else:
        query = "SELECT I, JSON_EXTRACT_PATH_TEXT(D, P) FROM TABLE1"
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": range(len(data)),
                "D": data,
                "P": path,
                "B": [i % 4 == 3 for i in range(len(data))],
            }
        )
    }
    expected_output = pd.DataFrame({0: range(len(data)), 1: answer})
    if use_case:
        expected_output[1][ctx["table1"].B] = None
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case"),
    ],
)
@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(
            pd.Series(
                [
                    {"Q1": 100, "Q2": -13, "Q3": 40, "Q4": 500},
                    {"Q1": 0, "Q2": 42, "Q3": 90, "Q4": 300},
                    None,
                    {"Q1": 50, "Q2": 256, "Q3": -10, "Q4": 64},
                    {"Q1": 26, "Q2": 128, "Q3": 72, "Q4": 512},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("Q1", pa.int32()),
                            pa.field("Q2", pa.int32()),
                            pa.field("Q3", pa.int32()),
                            pa.field("Q4", pa.int32()),
                        ]
                    )
                ),
            ),
            pd.Series(
                [
                    ["Q1", "Q2", "Q3", "Q4"],
                    ["Q1", "Q2", "Q3", "Q4"],
                    None,
                    ["Q1", "Q2", "Q3", "Q4"],
                    ["Q1", "Q2", "Q3", "Q4"],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="struct_array",
        ),
        pytest.param(
            pd.Series(
                [
                    {"Jan": [0, 10, 10, 15], "Feb": [10, None, 5, 13], "Mar": []},
                    {"Apr": [20]},
                    None,
                    {"Sep": [1], "Oct": [2, 3], "Nov": [], "Dec": [4, 5, 6]},
                    {},
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.large_list(pa.int32()))),
            ),
            pd.Series(
                [
                    ["Jan", "Feb", "Mar"],
                    ["Apr"],
                    None,
                    ["Sep", "Oct", "Nov", "Dec"],
                    [],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="map_array",
        ),
    ],
)
def test_object_keys(data, use_case, answer, memory_leak_check):
    if use_case:
        query = "SELECT CASE WHEN B THEN OBJECT_KEYS(J) ELSE OBJECT_KEYS(J_COPY) END FROM table1"
    else:
        query = "SELECT OBJECT_KEYS(J) FROM table1"
    check_query(
        query,
        {"table1": pd.DataFrame({"J": data, "J_COPY": data, "B": [True] * len(data)})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_dtype=False,
        check_names=False,
        # Can't sort output due to gaps when sorting columns of arrays
        sort_output=False,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('id', I, 'tag', S) FROM table1",
            pd.Series(
                [
                    {"id": 1, "tag": "A"},
                    {"id": 2, "tag": "BC"},
                    {"id": 4, "tag": "DEF"},
                    {"id": 8, "tag": "GHIJ"},
                    {"id": 16, "tag": "KLMNO"},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("id", pa.int32()),
                            pa.field("tag", pa.string()),
                        ]
                    )
                ),
            ),
            id="no_nested-no_null-no_case",
        ),
        pytest.param(
            "SELECT CASE WHEN DUMMY THEN OBJECT_CONSTRUCT_KEEP_NULL('id', I, 'tag', S) ELSE OBJECT_CONSTRUCT_KEEP_NULL('id', 0, 'tag', 'foo') END FROM table1",
            pd.Series(
                [
                    {"id": 1, "tag": "A"},
                    {"id": 2, "tag": "BC"},
                    {"id": 4, "tag": "DEF"},
                    {"id": 8, "tag": "GHIJ"},
                    {"id": 16, "tag": "KLMNO"},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("id", pa.int32()),
                            pa.field("tag", pa.string()),
                        ]
                    )
                ),
            ),
            id="no_nested-no_null-with_case",
            marks=pytest.mark.skip(reason="[BSE-1889] Support JSON in CASE statements"),
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('arr', A, 'map', J) FROM table1",
            pd.Series(
                [
                    {"arr": [0], "map": {"A": 1, "B": "1", "C": [-1, 0, None, 1]}},
                    {"arr": [1, 2], "map": {"A": 2, "B": "3", "C": [0, 1, None, 2]}},
                    {"arr": [], "map": {"A": 4, "B": "9", "C": [1, 2, None, 3]}},
                    {
                        "arr": [3, None, 5],
                        "map": {"A": 8, "B": "27", "C": [2, 3, None, 4]},
                    },
                    {
                        "arr": [6, 7, 8, None],
                        "map": {"A": 16, "B": "81", "C": [3, 4, None, 5]},
                    },
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("arr", pa.large_list(pa.int32())),
                            pa.field(
                                "map",
                                pa.struct(
                                    [
                                        pa.field("A", pa.int32()),
                                        pa.field("B", pa.string()),
                                        pa.field("C", pa.large_list(pa.int32())),
                                    ]
                                ),
                            ),
                        ]
                    )
                ),
            ),
            id="with_nested-no_map-no_null-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('arr', A, 'map', M) FROM table1",
            pd.Series(
                [
                    {"arr": [0], "map": {"B": 66}},
                    {"arr": [1, 2], "map": {"C": 67, "D": 68}},
                    {"arr": [], "map": {"D": 68, "E": 69, "F": 70}},
                    {"arr": [3, None, 5], "map": {"E": 69, "F": 70, "G": 71, "H": 72}},
                    {
                        "arr": [6, 7, 8, None],
                        "map": {"F": 70, "G": 71, "H": 72, "I": 73, "J": 74},
                    },
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("arr", pa.large_list(pa.int32())),
                            pa.field("map", pa.map_(pa.string(), pa.int32())),
                        ]
                    )
                ),
            ),
            id="with_nested-with_map-no_null-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('id', I_N, 'tag', S_N) FROM table1",
            pd.Series(
                [
                    {"id": 1, "tag": "Alpha"},
                    {"id": None, "tag": None},
                    {"id": 16, "tag": "Beta"},
                    {"id": None, "tag": None},
                    {"id": 256, "tag": "Gamma"},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("id", pa.int32()),
                            pa.field("tag", pa.string()),
                        ]
                    )
                ),
            ),
            id="no_nested-with_null-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('arr', A_N, 'map', J_N) FROM table1",
            pd.Series(
                [
                    {"arr": [0], "map": {"A": 0, "B": 1}},
                    {"arr": [1, 2], "map": {"A": 1, "B": 0}},
                    {"arr": [6, 7, 8, None], "map": {"A": 2, "B": 3}},
                    {"arr": None, "map": None},
                    {"arr": [3, None, 5], "map": {"A": 4, "B": 5}},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("arr", pa.large_list(pa.int32())),
                            pa.field(
                                "map",
                                pa.struct(
                                    [
                                        pa.field("A", pa.int32()),
                                        pa.field("B", pa.int32()),
                                    ]
                                ),
                            ),
                        ]
                    )
                ),
            ),
            id="with_nested-no_map-with_null-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL('arr', A_N, 'map', M_N) FROM table1",
            pd.Series(
                [
                    {"arr": [0], "map": {"A": 0, "B": 1}},
                    {"arr": [1, 2], "map": None},
                    {"arr": [6, 7, 8, None], "map": {"C": 2}},
                    {"arr": None, "map": None},
                    {"arr": [3, None, 5], "map": {"D": 3, "E": 4, "F": 5}},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("arr", pa.large_list(pa.int32())),
                            pa.field("map", pa.map_(pa.string(), pa.int32())),
                        ]
                    )
                ),
            ),
            id="with_nested-with_map-with_null-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT_KEEP_NULL(*) FROM (SELECT I as col_1, I_N as COL_2, S as cOl_3, S_N as Col_4 FROM table1)",
            pd.Series(
                [
                    {"col_1": 1, "COL_2": 1, "cOl_3": "A", "Col_4": "Alpha"},
                    {"col_1": 2, "COL_2": None, "cOl_3": "BC", "Col_4": None},
                    {"col_1": 4, "COL_2": 16, "cOl_3": "DEF", "Col_4": "Beta"},
                    {"col_1": 8, "COL_2": None, "cOl_3": "GHIJ", "Col_4": None},
                    {"col_1": 16, "COL_2": 256, "cOl_3": "KLMNO", "Col_4": "Gamma"},
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("col_1", pa.int32()),
                            pa.field("COL_2", pa.int32()),
                            pa.field("cOl_3", pa.string()),
                            pa.field("Col_4", pa.string()),
                        ]
                    )
                ),
            ),
            id="star_syntax-no_nested-no_case",
        ),
    ],
)
def test_object_construct_keep_null(query, answer, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "DUMMY": [True] * 15,
                "I": [1, 2, 4, 8, 16] * 3,
                "I_N": pd.Series([1, None, 16, None, 256] * 3, dtype=pd.Int32Dtype()),
                "S": ["A", "BC", "DEF", "GHIJ", "KLMNO"] * 3,
                "S_N": ["Alpha", None, "Beta", None, "Gamma"] * 3,
                "A": [[0], [1, 2], [], [3, None, 5], [6, 7, 8, None]] * 3,
                "A_N": [[0], [1, 2], [6, 7, 8, None], None, [3, None, 5]] * 3,
                "J": pd.Series(
                    [
                        {"A": 2**i, "B": str(3**i), "C": [i - 1, i, None, i + 1]}
                        for i in range(5)
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                                pa.field("C", pa.large_list(pa.int32())),
                            ]
                        )
                    ),
                ),
                "J_N": pd.Series(
                    [
                        {"A": 0, "B": 1},
                        {"A": 1, "B": 0},
                        {"A": 2, "B": 3},
                        None,
                        {"A": 4, "B": 5},
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.int32()),
                            ]
                        )
                    ),
                ),
                "M": pd.Series(
                    [
                        {chr(i): i for i in range(65 + j, 65 + j * 2)}
                        for j in range(1, 6)
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                ),
                "M_N": pd.Series(
                    [
                        {"A": 0, "B": 1},
                        None,
                        {"C": 2},
                        None,
                        {"D": 3, "E": 4, "F": 5},
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_dtype=False,
        check_names=False,
        # Can't sort semi-structured data outputs in Python
        sort_output=False,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT OBJECT_CONSTRUCT('linear', I, 'polynomial', ISQ, 'exponential', I2EXP) FROM table1",
            pd.array(
                [
                    {"linear": 0, "polynomial": 0, "exponential": 1},
                    {},
                    {"polynomial": 4, "exponential": 4},
                    {"linear": 3},
                    {"polynomial": 16},
                    {"exponential": 32},
                    {"linear": 6, "polynomial": 36},
                    {"linear": 7, "exponential": 128},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            id="integers-no_case",
        ),
        pytest.param(
            "SELECT CASE WHEN DUMMY THEN ARRAY_SIZE(OBJECT_KEYS(OBJECT_CONSTRUCT('linear', I, 'polynomial', ISQ, 'exponential', I2EXP))) ELSE -1 END FROM table1",
            pd.array([3, 0, 2, 1, 1, 1, 2, 2]),
            id="integers-with_case",
        ),
        pytest.param(
            "SELECT OBJECT_CONSTRUCT('full', names[idx], 'abbr', upper(left(names[idx], 3))) FROM table1",
            pd.array(
                [
                    {"full": "Golden Gate Bridge", "abbr": "GOL"},
                    {"full": "Taj Mahal", "abbr": "TAJ"},
                    {"full": "Eiffel Tower", "abbr": "EIF"},
                    {},
                    {"full": "Golden Gate Bridge", "abbr": "GOL"},
                    {"full": "Taj Mahal", "abbr": "TAJ"},
                    {"full": "Eiffel Tower", "abbr": "EIF"},
                    {},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ),
            id="strings-no_case",
        ),
    ],
)
def test_object_construct(query, answer, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "DUMMY": [True] * 8,
                "names": [["Golden Gate Bridge", "Taj Mahal", "Eiffel Tower"]] * 8,
                "idx": [0, 1, 2, 3] * 2,
                "I": pd.Series(
                    [0, None, None, 3, None, None, 6, 7], dtype=pd.Int8Dtype()
                ),
                "ISQ": pd.Series(
                    [0, None, 4, None, 16, None, 36, None], dtype=pd.Int16Dtype()
                ),
                "I2EXP": pd.Series(
                    [1, None, 4, None, None, 32, None, 128], dtype=pd.Int32Dtype()
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_dtype=False,
        check_names=False,
        # Can't sort semi-structured data outputs in Python
        sort_output=False,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize(
    "query, df, answer",
    [
        pytest.param(
            "SELECT OBJECT_DELETE(J, 'id') FROM table1",
            pd.DataFrame(
                {
                    "J": [
                        {
                            "id": i,
                            "tags": [
                                "abcdefghij"[(i**2) % 10 : (i**3) % 10 + j]
                                for j in range(1 + (i**2) % 7)
                            ],
                            "attrs": {"A": i, "B": str(i), "C": [i]},
                        }
                        for i in range(50)
                    ]
                    + [
                        None,
                        {
                            "id": None,
                            "tags": [],
                            "attrs": {"A": None, "B": None, "C": []},
                        },
                        {
                            "id": -1,
                            "tags": None,
                            "attrs": {"A": None, "B": None, "C": None},
                        },
                    ]
                }
            ),
            [
                {
                    "tags": [
                        "abcdefghij"[(i**2) % 10 : (i**3) % 10 + j]
                        for j in range(1 + (i**2) % 7)
                    ],
                    "attrs": {"A": i, "B": str(i), "C": [i]},
                }
                for i in range(50)
            ]
            + [
                None,
                {"tags": [], "attrs": {"A": None, "B": None, "C": []}},
                {"tags": None, "attrs": {"A": None, "B": None, "C": None}},
            ],
            id="struct-drop_literal-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_DELETE(J, K) FROM table1",
            pd.DataFrame(
                {
                    "J": pd.Series(
                        (
                            [{"A": 0, "B": 1, "C": 2, "D": 3}] * 4
                            + [{"A": 4, "C": 5}] * 4
                        )
                        * 5
                        + [
                            None,
                            {"A": 6, "B": 7, "C": None, "D": None},
                            {},
                            {"A": 8, "B": 9, "C": None, "D": None},
                        ],
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                    ),
                    "K": ["A", "B", "C", "D"] * 11,
                }
            ),
            pd.Series(
                [
                    {"B": 1, "C": 2, "D": 3},
                    {"A": 0, "C": 2, "D": 3},
                    {"A": 0, "B": 1, "D": 3},
                    {"A": 0, "B": 1, "C": 2},
                    {"C": 5},
                    {"A": 4, "C": 5},
                    {"A": 4},
                    {"A": 4, "C": 5},
                ]
                * 5
                + [
                    None,
                    {"A": 6, "C": None, "D": None},
                    {},
                    {"A": 8, "B": 9, "C": None},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="map-drop_column-no_case",
        ),
        pytest.param(
            "SELECT OBJECT_DELETE(J, 'D', K, 'E', 'A') FROM table1",
            pd.DataFrame(
                {
                    "J": pd.Series(
                        (
                            [{"A": 0, "B": 1, "C": 2, "D": 3}] * 4
                            + [{"A": 4, "C": 5}] * 4
                        )
                        * 5
                        + [
                            None,
                            {"A": 6, "B": 7, "C": None, "D": None},
                            {},
                            {"A": 8, "B": 9, "C": None, "D": None},
                        ],
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                    ),
                    "K": ["A", "B", "C", "D"] * 11,
                }
            ),
            pd.Series(
                [
                    {"B": 1, "C": 2},
                    {"C": 2},
                    {"B": 1},
                    {"B": 1, "C": 2},
                    {"C": 5},
                    {"C": 5},
                    {},
                    {"C": 5},
                ]
                * 5
                + [None, {"C": None}, {}, {"B": 9, "C": None}],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="map-drop_mixed-no_case",
        ),
    ],
)
def test_object_delete(query, df, answer, memory_leak_check):
    check_query(
        query,
        {"table1": df},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_dtype=False,
        check_names=False,
        # Can't sort semi-structured data outputs in Python
        sort_output=False,
    )
