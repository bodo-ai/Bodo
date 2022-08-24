# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests BodoSQL queries with types that cannot be directly
supported but has implicit conversions that can be automated.
"""
import datetime

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_categorical(memory_leak_check):
    """
    Tests that various categorical types can be converted
    to their element type.
    """
    query = "select * from categorical_table"
    df = pd.DataFrame(
        {
            # String category
            "A": pd.Categorical(["anve", "Er2"] * 5),
            # int64
            "B": pd.Categorical([5, -32] * 5),
            # uint64
            "C": pd.Categorical(pd.array([5, 2] * 5, "uint64")),
            # float64
            "D": pd.Categorical([1.1, 2.7] * 5),
            # dt64
            "E": pd.Categorical(
                [pd.Timestamp(2021, 4, 5), pd.Timestamp(2021, 4, 4)] * 5
            ),
            # td64
            "F": pd.Categorical([pd.Timedelta(2), pd.Timedelta(seconds=-4)] * 5),
            # boolean
            "G": pd.Categorical([True, False] * 5),
            # Unconverted string type
            "H": ["Don't change", "me"] * 5,
        }
    )
    # Since this is all about typing, we generate an expected output.
    expected_output = pd.DataFrame(
        {
            "A": df["A"].astype(str),
            "B": df["B"].astype("Int64"),
            "C": df["C"].astype("UInt64"),
            "D": df["D"].astype(np.float64),
            "E": df["E"].astype("datetime64[ns]"),
            "F": df["F"].astype("timedelta64[ns]"),
            "G": df["G"].astype("boolean"),
            "H": df["H"],
        }
    )
    check_query(
        query,
        {"categorical_table": df},
        None,
        expected_output=expected_output,
        convert_nullable_bodosql=False,
    )


@pytest.mark.slow
def test_date(memory_leak_check):
    """
    Tests that date can be converted to their element type.
    """
    query = "select * from categorical_table"
    df = pd.DataFrame(
        {
            # Date
            "A": [datetime.date(2021, 4, 1), datetime.date(2019, 2, 3)] * 5,
            # Unconverted string type
            "B": ["Don't change", "me"] * 5,
        }
    )
    # Since this is all about typing, we generate an expected output.
    expected_output = pd.DataFrame(
        {
            "A": df["A"].astype("datetime64[ns]"),
            "B": df["B"],
        }
    )
    check_query(query, {"categorical_table": df}, None, expected_output=expected_output)
