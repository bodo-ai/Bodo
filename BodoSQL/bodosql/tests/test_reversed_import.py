# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test file that checks if a user imports bodosql before bodo
    then queries still execute as expected.
"""
import bodosql  # noqa
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

import bodo  # noqa


@pytest.mark.slow
def test_simple(spark_info, memory_leak_check):
    dataframe_dict = {"table1": pd.DataFrame({"A": [1, 2, 3]})}
    check_query("select * from table1", dataframe_dict, spark_info)
