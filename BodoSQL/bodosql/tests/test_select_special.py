# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of special SELECT operators
"""
import pandas as pd

from bodosql.tests.utils import check_query


def test_exclude(memory_leak_check):
    """
    Test selecting all columns except some excluded names
    """
    query = "SELECT * EXCLUDE (A, E, I, O, U) FROM T"
    df = pd.DataFrame(
        {chr(i): list(range(i**2, i**2 + 5)) for i in range(ord("A"), ord("Z") + 1)}
    )
    answer = df.drop(list("AEIOU"), axis=1)
    check_query(
        query,
        {"T": df},
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
    )
