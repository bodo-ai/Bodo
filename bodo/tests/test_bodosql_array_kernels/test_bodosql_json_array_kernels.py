# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL JSON utilities
"""


import json

import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(
            None,
            id="scalar_null",
        ),
        pytest.param(
            "",
            id="scalar_empty_str",
        ),
        pytest.param(
            "\n  \t",
            id="scalar_whitespace",
        ),
        pytest.param(
            "{}",
            id="scalar_empty_map",
        ),
        pytest.param(
            '{"First": "Rhaenyra", "Last": "Targaryen"}',
            id="scalar_simple_map",
        ),
        pytest.param(
            '{"A": "He said \\"Yay!\\" last night", "Quote: \\"": "Fudge\\\\", "Hello": "\\\\\\"\\\\\\""}',
            id="scalar_escape",
        ),
        pytest.param(
            pd.array(["{}", "{}", "{}", "{}", "{}"]),
            id="array_empty_maps",
        ),
        pytest.param(
            pd.array(
                [
                    '{"City": "SF", "State": "CA"}',
                    '{"City": "PIT", "State": "PA"}',
                    '{"City": "LA", "State": "CA"}',
                    '{"City": "NOLA", "State": "LA"}',
                    '{"City": "NYC", "State": "NY"}',
                ]
            ),
            id="array_simple_maps",
        ),
        pytest.param(
            pd.array(
                [
                    '{"A": 1}',
                    " {   }   ",
                    '{"B": 2, "C": 3, "D": 4, "E": 5}',
                    '{"F": 6, "G": 7, "H": 8}',
                    '{"I": 9, "J": 10}',
                ]
            ),
            id="array_variable_maps_no_nulls",
        ),
        pytest.param(
            pd.array(
                [
                    '{"A": 1}',
                    "",
                    " {   }   ",
                    '{"B": 2, "C": 3, "D": 4, "E": 5}',
                    '{"F": 6, "G": 7, "H": 8}',
                    "{}",
                    "",
                    '{"I": 9, "J": 10}',
                ]
            ),
            id="array_variable_maps_empty_nulls",
        ),
        pytest.param(
            pd.array(
                [
                    None,
                    '{"A": 1}',
                    '{"Q": "R", "S": "T"}',
                    None,
                    " {   }   ",
                    '{"F": 6, "G": 7, "H": 8}',
                    "{}",
                    '{"B": 2, "C": 3, "D": 4, "E": 5}',
                    None,
                    None,
                    '{"I": 9, "J": 10}',
                ]
            ),
            id="array_variable_maps_true_nulls",
        ),
        pytest.param(
            pd.array(
                [
                    None,
                    '{"Lat": "w:1º,s:-3º", "Lon": "∞"}',
                    " ",
                    '{"f(x, y)": "∫∫√x^2+y^2dydx", "x": 0.5, "y": -1.3}',
                    "",
                    " { } ",
                    None,
                    '{"o4": "¢", "o8": "•", "og": "©", "ocv": "◊", "5±2": "3≤7"}',
                    "\n",
                    '{"cºntr0l": "~\xaf"}',
                    " \t ",
                    '{"œ∑´®†¥": "qwerty", "uiop[]": "¨ˆøπ“‘"}',
                    None,
                ]
            ),
            id="array_variable_maps_nulls_empty_nonascii",
        ),
        pytest.param(
            pd.array(
                [
                    ' {"C[[ity": "SF",  " S ta[}te" :  "C{]A" }  ',
                    '{  "City": "PI}}T", "Sta }{ te": "PA"}',
                    '{ " C i t y " :"LA"  ,"   St ate": "CA" }   ',
                    '{"Ci]ty]": "     NOLA"  , "State   "   : "LA"   }    ',
                    '    {"City": "{NYC"   ,   "State": "{NY}"}',
                    '   {  "City" : "[CHI]"   ,   "State": "IL"  } ',
                ]
            ),
            id="array_spacing_symbols",
        ),
        pytest.param(
            pd.array(
                [
                    '{"A": {"B": 1, "C": 2, "D": {}}, "E": {"F": {}}}',
                    '{"A": {"B": {"C": {"D": {"E{]": "F}["}}}}}',
                    '{"]A[": {"B": {" C ": {"  D  ": {"   E   ": "F"}}}}}',
                    '{"A": {"B": "{}", "C": "}{", "[D]": {}}, "E": {"F": {}}}',
                    '{"A": {"B": "C"}, "D": {"E": "F", "G": "HIJ"}}',
                ]
            ),
            id="array_nesting_symbols",
        ),
        pytest.param(
            pd.array(
                [
                    '{"A": [1, 2.0, -3.1415, 4, 5], "B": ["A", "E", "I"]}',
                    "",
                    '{"A": [1, {}, ["B"], {"C": "D"}], "B": [[], [[[]], "[]", []], "{}", []]}',
                    "",
                    '{"A": [[[[[[[[[]],[]]]]]]], ["10"]]}',
                    "",
                    '{"Q": [], "R": [], "S": [], "T": []}',
                ]
            ),
            id="array_arrays",
        ),
        pytest.param(
            pd.array(
                [
                    '{"Error": "Wrong symbols"}',
                    ".}",
                    '{"Error": "Unclosed outer {"}',
                    "  {  ",
                    '{"Error": "No :"}',
                    '{"A" 1 "B": 2}',
                    '{"Error": "No ,"}',
                    '{"A": {"B": 1} "C": 2}',
                    '{"Error": "; instead of ,"}',
                    '{"A": {"B": 1} ; "C": 2}',
                    '{"Error": "single quoted key"}',
                    "{'A': \"B\"}",
                    '{"Error": "non quoted key"}',
                    '{A: "B"}',
                    '{"Error": "Incomplete key"}',
                    '{"ABC',
                    '{"Error": "Incomplete colon"}',
                    '{"A" : ',
                    '{"Error": "Missing value"}',
                    '{"A" : 1, "B" : }',
                    '{"Error": "Unclosed ["}',
                    '{"A": [[1, 2], "B": 3}',
                    '{"Error": "Unmatched ]"}',
                    '{"A": 4, "B": 3] }',
                    '{"Error": "Unclosed inner {"}',
                    '{"A": {"A": {"B": }, "B": 3] }',
                    '{"Error": "Unmatched inner }"}',
                    '{"A": "B": 1}}',
                    '{"Error": "Characters after map"}',
                    '{"A": 1}    [',
                    '{"Error": "Inner stack mess"}',
                    '{"A": [[], [{"B"}{], [}]]}',
                    '{"Error": "Under-escaped key quote"}',
                    '{"A\\": 1}',
                    '{"Error": "Under-escaped value quote"}',
                    '{"A": "B\\"}',
                ]
            ),
            id="array_malformed",
        ),
    ],
)
def test_parse_json(arg):
    def impl(arg):
        return bodo.libs.bodosql_json_array_kernels.parse_json(arg)

    # Recursively parses values in a value outputted by JSON.loads to remove
    # all whitespace that are not part of a string
    def remove_whitespace(elem):
        if isinstance(elem, list):
            elems = [remove_whitespace(sub) for sub in elem]
            return "[" + ",".join(elems) + "]"
        if isinstance(elem, dict):
            elems = [f"{repr(key)}:{remove_whitespace(elem[key])}" for key in elem]
            return "{" + ",".join(elems) + "}"
        return repr(elem)

    # receives a string, uses Python's JSON module to parse it, then converts
    # it to use the same format as Bodo's parse_json util
    def parse_json_scalar_fn(arg):
        if pd.isna(arg) or len(arg.strip()) == 0:
            return None
        try:
            result = json.loads(arg)
            for key in result:
                result[key] = remove_whitespace(result[key]).replace("'", '"')
            return result
        except:
            return None

    answer = vectorized_sol((arg,), parse_json_scalar_fn, None)
    if isinstance(answer, pd.Series):
        answer = answer.values

    # [BE-3772] distributed testing currently banned because of segfaulting on
    # gatherv with the array type
    check_func(impl, (arg,), py_output=answer, check_dtype=False, only_seq=True)


@pytest.mark.parametrize(
    "data, path, answer",
    [
        pytest.param(
            '{"key": "value"}',
            "key",
            "value",
            id="scalar_test",
        ),
        pytest.param(
            None,
            "key",
            None,
            id="null_test",
        ),
        pytest.param(
            '{"first": "Daemon", "last": "Targaryen", "hair": "platinum"}',
            pd.Series(["first", None, '["last"]', "age", "hair"]),
            pd.Series(["Daemon", None, "Targaryen", None, "platinum"]),
            id="simple_map",
        ),
        pytest.param(
            '[ "Alpha" , "Beta" , "Gamma" ]',
            pd.Series(["[0]", None, "[1]", "[3]", "[2]"]),
            pd.Series(["Alpha", None, "Beta", None, "Gamma"]),
            id="simple_array",
        ),
        pytest.param(
            '{"[0]": "A", "[\'Z\']": "B", "]": \'C\', "Y\\"":"D", \'Q\': "R\\""}',
            pd.Series(['["[0]"]', "[\"['Z']\"]", '["]"]', "['Y\"']", "Q"]),
            pd.Series(["A", "B", "C", "D", 'R"']),
            id="escape_test",
        ),
        pytest.param(
            '\n{\n    "Name" :"Daemon Targaryen","Location":{\n\n "Continent" : "Westeros","Castle" :"Dragonstone"\n\n},\n\n            "Spouses": [\n\n{"First":"Rhea","Last":"Royce"},\n\n{ "First":"Laena", "Last" :"Velaryon" },{"First": "Rhaenyra", "Last": "Targaryen"}],\n"Age": 40}',
            pd.Series(
                [
                    "Name",
                    "Location.Continent",
                    "Location.Castle",
                    "Spouses[0].First",
                    "Spouses[0].Last",
                    "Spouses[1]['First']",
                    'Spouses[1]["Last"]',
                    "Spouses[2].First",
                    "Spouses[2].Last",
                ]
            ),
            pd.Series(
                [
                    "Daemon Targaryen",
                    "Westeros",
                    "Dragonstone",
                    "Rhea",
                    "Royce",
                    "Laena",
                    "Velaryon",
                    "Rhaenyra",
                    "Targaryen",
                ]
            ),
            id="house_of_dragon_map",
        ),
        pytest.param(
            '[[["A",\'B\'],["C","D"]],[[\'E\',\'F\'],["G","H"]]]',
            pd.Series(
                [
                    "[0][0][0]",
                    "[0][0][3]",
                    "[0][0][1]",
                    "[5][0][0]",
                    "[0][1][0]",
                    "[0][1][1]",
                    "[1][0][0]",
                    "[1][0][1]",
                    "[1][1][0]",
                    "[1][1][1]",
                    "[1][2][1]",
                ]
            ),
            pd.Series(["A", None, "B", None, "C", "D", "E", "F", "G", "H", None]),
            id="nested_grid_array",
        ),
        pytest.param(
            '[{"Name":\'Ringo\',\'Courses\':[{"Name":"CS","Grade":"A"},{"Name":"Science","Grade":"C"}]},{"Name":"Paul","Courses":[{"Name":"Math","Grade":"B"},{"Name":"CS","Grade":"A"},{"Name":"English","Grade":"A"}]},{"Name":"John","Courses":[{"Name":"English","Grade":"A"},{"Name":"Art","Grade":"A"},{"Name":"Math","Grade":"D"}]}]',
            pd.Series(
                [
                    '[0]["Name"]',
                    '[0]["Courses"][0]["Name"]',
                    '[0]["Courses"][0]["Grade"]',
                    '[0]["Courses"][1].Name',
                    '[0]["Courses"][1].Grade',
                    '[0]["Courses"][2].Name',
                    '[0]["Courses"][1].Foo',
                    "[1].Name",
                    '[1].Courses[0]["Name"]',
                    '[1].Courses[0]["Grade"]',
                    '[1].Courses[1]["Name"]',
                    '[1].Courses[1]["Grade"]',
                    '[1].Courses[2]["Name"]',
                    '[1].Courses[2]["Grade"]',
                    '[1].course[2]["Name"]',
                    '["2"].Courses[2]["Grade"]',
                    "[2].Name",
                    '[2].Courses[0]["Name"]',
                    '[2].Courses[0]["Grade"]',
                    '[2].Courses[1]["Name"]',
                    "[2].Courses[1].Grade",
                    "[2].Courses[2].Name",
                    "[2].Courses[2].Grade",
                ]
            ),
            pd.Series(
                [
                    "Ringo",
                    "CS",
                    "A",
                    "Science",
                    "C",
                    None,
                    None,
                    "Paul",
                    "Math",
                    "B",
                    "CS",
                    "A",
                    "English",
                    "A",
                    None,
                    None,
                    "John",
                    "English",
                    "A",
                    "Art",
                    "A",
                    "Math",
                    "D",
                ]
            ),
            id="roster_map_array",
        ),
        pytest.param(
            " { \"Snake\" : '🐍' , '😊' : \"Smile\" , \"Sigma\" : 'Σ' } ",
            pd.Series(
                [
                    "Snake",
                    None,
                    '["😊"]',
                    "Scissor",
                    "Sigma",
                ]
            ),
            pd.Series(
                [
                    "🐍",
                    None,
                    "Smile",
                    None,
                    "Σ",
                ]
            ),
            id="non_ascii",
        ),
    ],
)
def test_json_extract_path_text(data, path, answer):
    def impl(data, path):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.json_extract_path_text(data, path)
        )

    if not (isinstance(data, pd.Series) or isinstance(path, pd.Series)):
        impl = (
            lambda data, path: bodo.libs.bodosql_array_kernels.json_extract_path_text(
                data, path
            )
        )

    check_func(impl, (data, path), py_output=answer, check_dtype=False)


@pytest.mark.parametrize(
    "data, path",
    [
        pytest.param('{"A": "B"}', "", id="empty_path"),
        pytest.param('{"A": "B"}', ".A", id="dot_prefix_path"),
        pytest.param('{"A": "B"}', "A..B", id="double_dot_path"),
        pytest.param('{"A": "B"}', "A.", id="dot_suffix_path"),
        pytest.param('{"A": "B"}', 'A.["B"]', id="dot_bracket_path"),
        pytest.param('["A", "B", "C", "D", "E"]', "[-3]", id="negative_index_path"),
        pytest.param('["A", "B", "C", "D", "E"]', "[0 1]", id="malformed_index_path"),
        pytest.param('{"A": "B"}', "[\" ']", id="malformed_bracket_path"),
        pytest.param('{"A" "B"}', "A", id="missing_colon_data"),
        pytest.param('{"A": [{"B"]}}', "A", id="improper_internal_data"),
        pytest.param('{"A\': "B"}', "A", id="inconsistent_quote_data"),
    ],
)
def test_json_extract_path_text_invalid(data, path):
    """Check cases where JSON_EXTRACT_PATH_TEXT raises an exception, such as
    malformed JSON data or invalid path strings.

    Note: some malformed cases that should raise an exception instead output
    None because they are not detected by this function. This is because
    the function hunts for a specific entry in the JSON data and parses
    along the way, rather than scanning the entire string and making sure
    it is well formed."""

    def impl(data, path):
        return bodo.libs.bodosql_array_kernels.json_extract_path_text(data, path)

    func = bodo.jit(impl)

    with pytest.raises(ValueError):
        func(data, path)


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(True, id="vector"),
        pytest.param(False, id="scalar"),
    ],
)
@pytest.mark.parametrize(
    "data, use_map, answer",
    [
        pytest.param(
            pd.Series(
                [
                    {"first": "Rand", "last": "al'Thor", "nation": "Andor"},
                    {"first": "Lan", "last": "Mandragoran", "nation": "Malkier"},
                    None,
                    {"first": "Rodel", "last": "Ituralde", "nation": "Arad Doman"},
                    {"first": "Faile", "last": "Aybara", "nation": "Saldea"},
                ]
                * 3
            ),
            False,
            pd.Series(
                [
                    ["first", "last", "nation"],
                    ["first", "last", "nation"],
                    None,
                    ["first", "last", "nation"],
                    ["first", "last", "nation"],
                ]
                * 3
            ),
            id="struct_array",
        ),
        pytest.param(
            pd.Series(
                [
                    {
                        "A": ["Computer Science", "Physics"],
                        "B+": ["Math"],
                        "B-": ["Spanish"],
                    },
                    {"B": ["Math", "Physics", "English"]},
                    None,
                    {"A": ["German"], "A-": ["English", "Spanish"], "B+": ["Math"]},
                    {},
                    {"A-": ["Computer Science", "Art"], "B-": ["Spanish", "German"]},
                ]
                * 3
            ),
            True,
            pd.Series(
                [["A", "B+", "B-"], ["B"], None, ["A", "A-", "B+"], [], ["A-", "B-"]]
                * 3
            ),
            id="map_array",
        ),
    ],
)
def test_object_keys(data, use_map, answer, vector, memory_leak_check):
    def impl_vector(data):
        return pd.Series(bodo.libs.bodosql_array_kernels.object_keys(data))

    def impl_scalar(data):
        return bodo.libs.bodosql_array_kernels.object_keys(data[0])

    if vector:
        check_func(
            impl_vector,
            (data,),
            py_output=answer,
            check_dtype=False,
            use_map_arrays=use_map,
        )
    else:
        check_func(
            impl_scalar,
            (data,),
            py_output=answer[0],
            check_dtype=False,
            use_map_arrays=use_map,
            distributed=False,
            is_out_distributed=False,
        )
