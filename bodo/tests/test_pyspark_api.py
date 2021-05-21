# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Tests for pyspark APIs supported by Bodo
"""
from datetime import date, datetime

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import Row

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_session_box():
    """test boxing/unboxing for SparkSession object"""
    # just unbox
    def impl(arg):
        return True

    # unbox and box
    def impl2(arg):
        return arg

    spark = SparkSession.builder.appName("TestSpark").getOrCreate()
    check_func(impl, (spark,))
    check_func(impl2, (spark,))


def test_session_create():
    """test creating SparkSession object"""

    def impl():
        return SparkSession.builder.appName("TestSpark").getOrCreate()

    check_func(impl, ())


def test_session_const_lowering():
    """test constant lowering for SparkSession object"""
    spark = SparkSession.builder.appName("TestSpark").getOrCreate()

    def impl():
        return spark

    check_func(impl, ())


def test_row_box():
    """test boxing/unboxing for Row object"""
    # just unbox
    def impl(arg):
        return True

    # unbox and box
    def impl2(arg):
        return arg

    r = Row(A=3, B="AB")
    check_func(impl, (r,))
    check_func(impl2, (r,))


def test_row_constructor():
    """test Row constructor calls"""
    # kws
    def impl():
        return Row(A=3, B="ABC")

    # anonymous field names
    def impl2():
        return Row(3, "ABC")

    check_func(impl, ())
    check_func(impl2, ())


def test_row_get_field():
    """test Row constructor calls"""
    # getattr
    def impl1(r):
        return r.A

    # getitem with name
    def impl2(r):
        return r["A"]

    # getitem with int
    def impl3(r):
        return r[0]

    # getitem with slice
    def impl4(r):
        return r[:2]

    r = Row(A=3, B="AB", C=1.3)
    check_func(impl1, (r,))
    check_func(impl2, (r,))
    check_func(impl3, (r,))
    check_func(impl4, (r,))


def test_create_dataframe():
    """test spark.createDataFrame() calls"""
    # pandas input
    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.toPandas()

    # list of Rows input
    def impl2():
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(
            [
                Row(
                    a=1,
                    b=2.0,
                    c="string1",
                    d=date(2000, 1, 1),
                    e=datetime(2000, 1, 1, 12, 0),
                ),
                Row(
                    a=2,
                    b=3.0,
                    c="string2",
                    d=date(2000, 2, 1),
                    e=datetime(2000, 1, 2, 12, 0),
                ),
                Row(
                    a=4,
                    b=5.0,
                    c="string3",
                    d=date(2000, 3, 1),
                    e=datetime(2000, 1, 3, 12, 0),
                ),
            ]
        )
        return sdf.toPandas()

    df = pd.DataFrame({"A": [1, 2, 3], "B": ["A", "B", "C"]})
    check_func(impl, (df,), only_seq=True)
    check_func(impl2, (), only_seq=True)
    with pytest.raises(
        BodoError,
        match="createDataFrame\(\): 'data' should be a Pandas dataframe or list of Rows",
    ):
        bodo.jit(impl)(3)
