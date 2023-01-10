# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL JSON utilities
"""


import json

import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


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

    # Recieves a string, uses Python's JSON module to parse it, then converts
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
