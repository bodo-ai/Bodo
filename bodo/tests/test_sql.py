# Copyright (C) 2019 Bodo Inc. All rights reserved.
from sqlalchemy import create_engine
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
import pymysql
import pandas as pd

import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


def test_sql_hardcoded_aws():
    """This test for an hardcoded request and connection"""

    def test_impl():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, ())


def test_sql_hardcoded_time_offset_aws():
    """This test does not pass because the type of dates is not supported"""

    def test_impl():
        sql_request = "select * from employees limit 1000 offset 4000"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, ())


def test_sql_hardcoded_twocol_aws():
    """Selecting two columns without dates"""

    def test_impl():
        sql_request = "select first_name,last_name from employees"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    bodo_impl = bodo.jit(test_impl)
    frame = bodo_impl()


def test_sql_argument_passing():
    """Test passing SQL query and connection as arguments
    """

    def test_impl(sql_request, conn):
        df = pd.read_sql(sql_request, conn)
        return df

    sql_request = "select * from employees"
    conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
    check_func(test_impl, (sql_request, conn))
