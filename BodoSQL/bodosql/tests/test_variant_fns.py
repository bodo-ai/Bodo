# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test correctness of nested data functions with BodoSQL
"""
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize("use_case", [True, False])
@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(
            pd.Series([0, 1, 2, None]), pd.Series([False] * 3 + [None]), id="int"
        ),
        pytest.param(
            pd.Series(["a", "b", "c", None]),
            pd.Series([False] * 3 + [None]),
            id="string",
        ),
        pytest.param(
            pd.Series([["a"], ["b"], ["c"], None]),
            pd.Series([True] * 3 + [None]),
            id="array",
        ),
    ],
)
def test_is_array(use_case, data, answer):
    query = "SELECT IS_ARRAY(A) FROM table1"
    if use_case:
        query = "SELECT CASE WHEN FALSE THEN NULL ELSE IS_ARRAY(A) END FROM table1"

    check_query(
        query,
        {"TABLE1": pd.DataFrame({"A": data})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.parametrize("use_case", [True, False])
@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(
            pd.Series([0, 1, 2, None]), pd.Series([False] * 3 + [None]), id="int"
        ),
        pytest.param(
            pd.Series(["a", "b", "c", None]),
            pd.Series([False] * 3 + [None]),
            id="string",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3 + [None],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            pd.Series([True] * 3 + [None]),
            id="map",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3 + [None],
                dtype=pd.ArrowDtype(pa.struct([pa.field("a", pa.int32())])),
            ),
            pd.Series([True] * 3 + [None]),
            id="struct",
        ),
    ],
)
def test_is_object(use_case, data, answer):
    query = "SELECT IS_OBJECT(A) FROM table1"
    if use_case:
        query = "SELECT CASE WHEN FALSE THEN NULL ELSE IS_OBJECT(A) END FROM table1"

    check_query(
        query,
        {"TABLE1": pd.DataFrame({"A": data})},
        None,
        expected_output=pd.DataFrame({0: answer}),
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )
