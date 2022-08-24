# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Measure performance of sql operations.
"""

import pandas as pd

from bodo.tests.utils import check_timing_func


def test_sql_hardcoded_aws():
    """This test measures the speed of downloading the sql data"""

    def impl():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_timing_func(impl, ())
