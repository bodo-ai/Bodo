# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries that generate bodo code that utilizes metatypes
"""

from bodosql.tests.utils import check_query


def test_init_dataframe(basic_df, spark_info, memory_leak_check):
    """
    Test that the init_dataframe function works correctly when passed in metaTypes.
    init_dataframe should be called ever time we have a non trivial (can't be handled by
    loc) column projections.
    """
    query = "select SUM(A) over (PARTITION BY B ORDER BY C ASC ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) from table1"
    ("1 PRECEDING", "1 FOLLOWING")
    codegen = check_query(
        query,
        basic_df,
        spark_info,
        return_codegen=True,
        check_dtype=False,
        check_names=False,
    )

    # Ensure that we generated a call to init dataframe, and ensure that we generate a ColNamesMetaType
    # in the output functext.
    # Note that the ColNamesMetaType will be lowered as a global when calling .sql, but we keep
    # the ColNamesMetaType as local constant when calling conver_to_pd, for easier readability
    assert (
        "init_dataframe" in codegen["pandas_code"]
        and "ColNamesMetaType" in codegen["pandas_code"]
    )
