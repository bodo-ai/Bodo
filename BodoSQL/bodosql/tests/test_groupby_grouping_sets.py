"""
Tests correctness of various grouping sets operations, either
by directly generating queries using grouping sets or via query
that after 1 or more transformations results in grouping sets.
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param(
            (
                np.array([1, 1, 2, 2, 4], dtype=np.int64),
                pd.array([3, 3, 3, None, 5], dtype="Int64"),
            ),
            id="numpy-nullable",
        ),
        pytest.param(
            (pd.array(["1", "1", "2", "2", "4"]), pd.array(["3", "3", "3", None, "5"])),
            id="string",
        ),
        pytest.param(
            (
                np.array([1, 1, 2, 2, 4], dtype="datetime64[ns]"),
                pd.array(
                    [
                        pd.Timestamp("2001-01-01"),
                        pd.Timestamp("2001-01-01"),
                        pd.Timestamp("2001-01-01"),
                        pd.NaT,
                        pd.Timestamp("2002-01-01"),
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            id="timestamp",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        bodo.types.TimestampTZ.fromLocal("2021-01-02 14:00:00", 300),
                        bodo.types.TimestampTZ.fromLocal("2021-01-02 14:00:00", 300),
                        bodo.types.TimestampTZ.fromLocal("2021-01-02 12:30:00", 0),
                        bodo.types.TimestampTZ.fromLocal("2021-01-02 12:30:00", 0),
                        bodo.types.TimestampTZ.fromLocal("2024-03-19 12:00:00", 120),
                    ]
                ),
                np.array(
                    [
                        bodo.types.TimestampTZ.fromLocal("2021-03-14 00:00:00", -1),
                        bodo.types.TimestampTZ.fromLocal("2021-03-14 00:00:00", -1),
                        bodo.types.TimestampTZ.fromLocal("2021-03-14 00:00:00", -1),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-18 12:00:00", 120),
                    ]
                ),
            ),
            id="timestamp-tz",
            marks=pytest.mark.slow,
        ),
    ]
)
def grouping_sets_inputs(request):
    """
    Fixture for grouping inputs to grouping sets with the
    same table structure but various key types. The data pattern is also
    the same, including the data column(s) so we can generate a constant
    output that depends only on the column locations.

    Note we generate a different DataFrame for each key array type to test nullability.
    """
    data_column = pd.array([1, 12, None, 12, None], dtype="Int64")
    return pd.DataFrame(
        {
            "A": request.param[0],
            "B": request.param[1],
            "C": data_column,
        }
    )


def test_grouping_sets_basic(grouping_sets_inputs, memory_leak_check):
    """Test grouping sets for basic single key group by."""
    query = "SELECT A, B, SUM(C) as OUTPUT FROM TABLE1 GROUP BY GROUPING SETS (A, B)"
    ctx = {"TABLE1": grouping_sets_inputs}
    output_column = pd.array([13, 12, None, 13, 12, None], dtype="Int64")
    a_data_list = [
        grouping_sets_inputs["A"][0],
        grouping_sets_inputs["A"][2],
        grouping_sets_inputs["A"][4],
        None,
        None,
        None,
    ]
    b_data_list = [
        None,
        None,
        None,
        grouping_sets_inputs["B"][0],
        grouping_sets_inputs["B"][3],
        grouping_sets_inputs["B"][4],
    ]
    expected_output = pd.DataFrame(
        {
            "A": a_data_list,
            "B": b_data_list,
            "OUTPUT": output_column,
        }
    )
    check_query(query, ctx, None, check_dtype=False, expected_output=expected_output)


def test_grouping_sets_subset(grouping_sets_inputs, memory_leak_check):
    """Test grouping sets for (A, B) and (A)"""
    query = (
        "SELECT A, B, SUM(C) as OUTPUT FROM TABLE1 GROUP BY GROUPING SETS ((A, B), A)"
    )
    ctx = {"TABLE1": grouping_sets_inputs}
    output_column = pd.array([13, None, 12, None, 13, 12, None], dtype="Int64")
    a_data_list = [
        grouping_sets_inputs["A"][0],
        grouping_sets_inputs["A"][2],
        grouping_sets_inputs["A"][3],
        grouping_sets_inputs["A"][4],
        grouping_sets_inputs["A"][0],
        grouping_sets_inputs["A"][2],
        grouping_sets_inputs["A"][4],
    ]
    b_data_list = [
        grouping_sets_inputs["B"][0],
        grouping_sets_inputs["B"][2],
        grouping_sets_inputs["B"][3],
        grouping_sets_inputs["B"][4],
        None,
        None,
        None,
    ]
    expected_output = pd.DataFrame(
        {
            "A": a_data_list,
            "B": b_data_list,
            "OUTPUT": output_column,
        }
    )
    check_query(query, ctx, None, check_dtype=False, expected_output=expected_output)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT A, SUM(B1) as SUM_B1, SUM(B2) as SUM_B2, COUNT(DISTINCT(C)) as CD_C, COUNT(DISTINCT(D)) as CD_D, COUNT(DISTINCT(E)) as CD_E, COUNT(*) as COUNT_STAR FROM TABLE1 GROUP BY A",
            id="no-overlap",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT A, SUM(B1) as SUM_B1, COUNT(DISTINCT(B1)) as CD_B1, SUM(B2) as SUM_B2, COUNT(DISTINCT(C)) as CD_C, COUNT(DISTINCT(D)) as CD_D FROM TABLE1 GROUP BY A",
            id="overlap-sum-sum",
        ),
        pytest.param(
            "SELECT A, SUM(B1) as SUM_B1, COUNT(DISTINCT(B1)) as CD_B1, COUNT(*) as COUNT_STAR, COUNT(C) as COUNT_C, COUNT(DISTINCT(C)) as CD_C, COUNT(DISTINCT(D)) as CD_D FROM TABLE1 GROUP BY A",
            id="overlap-sum-size",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_implicit_grouping_sets(query, spark_info, memory_leak_check):
    """
    Test use cases where a grouping set is implicitly generated because of
    multiple COUNT(DISTINCT)s.
    """

    input_df = pd.DataFrame(
        {
            "A": np.array([1, 1, 2, 2, 4] * 10, dtype=np.int64),
            "B1": pd.array(
                [90, 283, 119, 34] * 11 + [39, 1124, 432, 534, 561, 523], dtype="Int64"
            ),
            "B2": pd.array(
                [90, 283, 119, 214] * 11 + [39, 1124, 432, 534, 561, 523], dtype="Int64"
            ),
            "C": pd.array(
                ["pol", "ert", "iop", "lkj", "sdf", "ewq"] * 8 + ["fgj", "mnb"]
            ),
            "D": pd.array(
                [90, 283, 119, 34] * 11 + [39, 1124, 432, 534, 561, 523], dtype="Int64"
            ),
            "E": pd.array(
                ["pol", "ert", "iop", "lkj", "sdf", "ewq"] * 8 + ["fgj", "mnb"]
            ),
        }
    )

    ctx = {"TABLE1": input_df}
    check_query(query, ctx, spark_info, check_dtype=False)


def test_grouping_non_streaming(grouping_sets_inputs, memory_leak_check):
    """
    Tests the GROUPING function when taking the non-streaming code path.
    """
    query = "SELECT A, B, GROUPING(B, A) as OUTPUT1, GROUPING(A) as OUTPUT2, GROUPING(A, B) as OUTPUT3, SUM(C) as OUTPUT4 FROM TABLE1 GROUP BY GROUPING SETS (A, B, ())"
    ctx = {"TABLE1": grouping_sets_inputs}
    output_column1 = pd.array([2, 2, 2, 1, 1, 1, 3], dtype="Int64")
    output_column2 = pd.array([0, 0, 0, 1, 1, 1, 1], dtype="Int64")
    output_column3 = pd.array([1, 1, 1, 2, 2, 2, 3], dtype="Int64")
    output_column4 = pd.array([13, 12, None, 13, 12, None, 25], dtype="Int64")
    a_data_list = [
        grouping_sets_inputs["A"][0],
        grouping_sets_inputs["A"][2],
        grouping_sets_inputs["A"][4],
        None,
        None,
        None,
        None,
    ]
    b_data_list = [
        None,
        None,
        None,
        grouping_sets_inputs["B"][0],
        grouping_sets_inputs["B"][3],
        grouping_sets_inputs["B"][4],
        None,
    ]
    expected_output = pd.DataFrame(
        {
            "A": a_data_list,
            "B": b_data_list,
            "OUTPUT1": output_column1,
            "OUTPUT2": output_column2,
            "OUTPUT3": output_column3,
            "OUTPUT4": output_column4,
        }
    )
    check_query(query, ctx, None, check_dtype=False, expected_output=expected_output)


def test_grouping_streaming(grouping_sets_inputs, memory_leak_check):
    """
    Tests the GROUPING function when taking the streaming code path.
    """
    query = "SELECT A, B, GROUPING(B, A) as OUTPUT1, GROUPING(A) as OUTPUT2, GROUPING(A, B) as OUTPUT3, SUM(C) as OUTPUT4 FROM TABLE1 GROUP BY GROUPING SETS (A, B)"
    ctx = {"TABLE1": grouping_sets_inputs}
    output_column1 = pd.array([2, 2, 2, 1, 1, 1], dtype="Int64")
    output_column2 = pd.array([0, 0, 0, 1, 1, 1], dtype="Int64")
    output_column3 = pd.array([1, 1, 1, 2, 2, 2], dtype="Int64")
    output_column4 = pd.array([13, 12, None, 13, 12, None], dtype="Int64")
    a_data_list = [
        grouping_sets_inputs["A"][0],
        grouping_sets_inputs["A"][2],
        grouping_sets_inputs["A"][4],
        None,
        None,
        None,
    ]
    b_data_list = [
        None,
        None,
        None,
        grouping_sets_inputs["B"][0],
        grouping_sets_inputs["B"][3],
        grouping_sets_inputs["B"][4],
    ]
    expected_output = pd.DataFrame(
        {
            "A": a_data_list,
            "B": b_data_list,
            "OUTPUT1": output_column1,
            "OUTPUT2": output_column2,
            "OUTPUT3": output_column3,
            "OUTPUT4": output_column4,
        }
    )
    check_query(query, ctx, None, check_dtype=False, expected_output=expected_output)
