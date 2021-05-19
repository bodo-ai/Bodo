# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Tests for scipy.sparse.csr_matrix data structure
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import Row

from bodo.tests.utils import check_func


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
