# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Test correctness of the PERCENTILE aggregation operations for BodoSQL
"""

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_groupby
from bodosql.tests.utils import check_query

# Skip unless any groupby-related files were changed
pytestmark = pytest_slow_unless_groupby


@pytest.fixture
def percentile_df_data():
    return pd.DataFrame(
        {
            "KEYS": ["singleton"] * 3
            + ["all_null"] * 3
            + ["four"] * 4
            + ["hundred"] * 100
            + ["five"] * 6,
            "I": pd.Series(
                [None, 42, None]
                + [None, None, None]
                + [10, 80, 20, 40]
                + [i**2 for i in range(100)]
                + [1, 10, 100, None, 1000, 10000],
                dtype=pd.Int32Dtype(),
            ),
            "F": pd.Series(
                [None, None, 3.141592]
                + [None, None, None]
                + [100.0, -10.0, 0.0, -5.5]
                + list(range(100))
                + [-100.0, -10.0, 0.0, None, 10.0, 100.0],
                dtype=np.float64,
            ),
        }
    )


@pytest.fixture(
    params=[
        pytest.param(
            (
                "I",
                pd.DataFrame(
                    {
                        "KEYS": ["singleton", "all_null", "four", "hundred", "five"],
                        "Q0": [42.0, None, 10.0, 0.0, 1.0],
                        "Q10": [42.0, None, 13.0, 98.1, 4.6],
                        "Q25": [42.0, None, 17.5, 612.75, 10.0],
                        "Q50": [42.0, None, 30.0, 2450.5, 100.0],
                        "Q75": [42.0, None, 50.0, 5513.25, 1000.0],
                        "Q90": [42.0, None, 68.0, 7938.9, 6399.999],
                        "Q100": [42.0, None, 80.0, 9801.0, 10000.0],
                    }
                ),
            ),
            id="int32",
        ),
        pytest.param(
            (
                "F",
                pd.DataFrame(
                    {
                        "KEYS": ["singleton", "all_null", "four", "hundred", "five"],
                        "Q0": [3.141592, None, -10.0, 0.0, -100.0],
                        "Q10": [3.141592, None, -8.65, 9.9, -64.0],
                        "Q25": [3.141592, None, -6.625, 24.75, -10.0],
                        "Q50": [3.141592, None, -2.75, 49.5, 0.0],
                        "Q75": [3.141592, None, 25, 74.25, 10.0],
                        "Q90": [3.141592, None, 70, 89.1, 64.0],
                        "Q100": [3.141592, None, 100, 99.0, 100.0],
                    }
                ),
            ),
            id="float64",
        ),
    ]
)
def percentile_cont_args(request):
    """
    The results of calling PERCENTILE_CONT on each of the 3 data columns
    of percentile_df_data using the column "keys" as the grouping key with
    percentiles 0.0, 0.1, 0.25, 0.5, 0.75, 0.9 and 1.0. Answers computed
    using Snowflake.
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                "I",
                pd.DataFrame(
                    {
                        "KEYS": ["singleton", "all_null", "four", "hundred", "five"],
                        "0.00": [42.0, None, 10.0, 0.0, 1.0],
                        "0.10": [42.0, None, 10.0, 81.0, 1.0],
                        "0.25": [42.0, None, 10.0, 576.0, 10.0],
                        "0.50": [42.0, None, 20.0, 2401.0, 100.0],
                        "0.75": [42.0, None, 40.0, 5476.0, 1000.0],
                        "0.90": [42.0, None, 80.0, 7921.0, 10000.0],
                        "1.00": [42.0, None, 80.0, 9801.0, 10000.0],
                    }
                ),
            ),
            id="int32",
        ),
        pytest.param(
            (
                "F",
                pd.DataFrame(
                    {
                        "KEYS": ["singleton", "all_null", "four", "hundred", "five"],
                        "0.00": [3.141592, None, -10.0, 0.0, -100.0],
                        "0.10": [3.141592, None, -10.0, 9.0, -100.0],
                        "0.25": [3.141592, None, -10.0, 24.0, -10.0],
                        "0.50": [3.141592, None, -5.5, 49.0, 0.0],
                        "0.75": [3.141592, None, 0.0, 74.0, 10.0],
                        "0.90": [3.141592, None, 100.0, 89.0, 100.0],
                        "1.00": [3.141592, None, 100.0, 99.0, 100.0],
                    }
                ),
            ),
            id="float64",
        ),
    ]
)
def percentile_disc_args(request):
    """
    The results of calling PERCENTILE_DISC on each of the 3 data columns
    of percentile_df_data using the column "keys" as the grouping key with
    percentiles 0.0, 0.1, 0.25, 0.5, 0.75, 0.9 and 1.0.  Answers computed
    using Snowflake.
    """
    return request.param


def test_percentile_cont_groupby(
    percentile_df_data, percentile_cont_args, memory_leak_check
):
    """Full E2E test for listagg with groupby and with different sorting options"""
    col, answer = percentile_cont_args
    query = """
SELECT
    keys,
    PERCENTILE_CONT(0.0) WITHIN GROUP (ORDER BY {0}) AS q0,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY {0}) AS q10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {0}) AS q25,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {0}) AS q50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {0}) AS q75,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY {0}) AS q90,
    PERCENTILE_CONT(1.0) WITHIN GROUP (ORDER BY {0}) AS q100
FROM table1
GROUP BY keys""".format(col)

    check_query(
        query,
        {"TABLE1": percentile_df_data},
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
    )


def test_percentile_disc_groupby(
    percentile_df_data, percentile_disc_args, memory_leak_check
):
    """Full E2E test for listagg with groupby and with different sorting options"""
    col, answer = percentile_disc_args
    query = """
SELECT
    keys,
    PERCENTILE_DISC(0.0) WITHIN GROUP (ORDER BY {0}) AS q0,
    PERCENTILE_DISC(0.1) WITHIN GROUP (ORDER BY {0}) AS q10,
    PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY {0}) AS q25,
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY {0}) AS q50,
    PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY {0}) AS q75,
    PERCENTILE_DISC(0.9) WITHIN GROUP (ORDER BY {0}) AS q90,
    PERCENTILE_DISC(1.0) WITHIN GROUP (ORDER BY {0}) AS q100
FROM table1
GROUP BY keys""".format(col)

    check_query(
        query,
        {"TABLE1": percentile_df_data},
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
    )


def test_percentile_cont_no_groupby(percentile_df_data, memory_leak_check):
    """Full E2E test for PERCENTILE_CONT without a groupby clause."""
    selects = []
    combinations = [
        ("I", 0.0),
        ("I", 0.25),
        ("I", 0.5),
        ("F", 0.1),
        ("F", 0.75),
        ("F", 1.0),
    ]
    for col, q in combinations:
        selects.append(f"PERCENTILE_CONT({q}) WITHIN GROUP (ORDER BY {col})")
    query = f"SELECT {', '.join(selects)} FROM table1"
    answer = pd.DataFrame(
        {
            0: 0.0,
            1: 370.75,
            2: 2070.50,
            3: 3.9141592,
            4: 73.75,
            5: 100.0,
        },
        index=np.arange(1),
    )
    check_query(
        query,
        {"TABLE1": percentile_df_data},
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_percentile_disc_no_groupby(percentile_df_data, memory_leak_check):
    """Full E2E test for PERCENTILE_DISC without a groupby clause."""
    selects = []
    combinations = [
        ("I", 0.0),
        ("I", 0.25),
        ("I", 0.5),
        ("F", 0.1),
        ("F", 0.75),
        ("F", 1.0),
    ]
    for col, q in combinations:
        selects.append(f"PERCENTILE_DISC({q}) WITHIN GROUP (ORDER BY {col})")
    query = f"SELECT {', '.join(selects)} FROM table1"
    answer = pd.DataFrame(
        {
            0: 0.0,
            1: 361.0,
            2: 2025.0,
            3: 3.141592,
            4: 74.0,
            5: 100.0,
        },
        index=np.arange(1),
    )
    check_query(
        query,
        {"TABLE1": percentile_df_data},
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )
