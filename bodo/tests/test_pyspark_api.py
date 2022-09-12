# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for pyspark APIs supported by Bodo
"""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import Row

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_session_box(memory_leak_check):
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


@pytest.mark.slow
def test_session_create(memory_leak_check):
    """test creating SparkSession object"""

    def impl():
        return SparkSession.builder.appName("TestSpark").getOrCreate()

    check_func(impl, ())


@pytest.mark.slow
def test_session_const_lowering(memory_leak_check):
    """test constant lowering for SparkSession object"""
    spark = SparkSession.builder.appName("TestSpark").getOrCreate()

    def impl():
        return spark

    check_func(impl, ())


@pytest.mark.slow
def test_row_box(memory_leak_check):
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


@pytest.mark.slow
def test_row_constructor(memory_leak_check):
    """test Row constructor calls"""
    # kws
    def impl():
        return Row(A=3, B="ABC")

    # anonymous field names
    def impl2():
        return Row(3, "ABC")

    check_func(impl, ())
    check_func(impl2, ())


@pytest.mark.slow
def test_row_get_field(memory_leak_check):
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


@pytest.mark.slow
def test_create_dataframe(memory_leak_check):
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
    # NOTE: using custom testing code since output is gathered on rank 0
    df_out = bodo.jit(impl)(df)
    if bodo.get_rank() == 0:
        pd.testing.assert_frame_equal(
            df_out, df, check_column_type=False, check_dtype=False
        )
    df_out = bodo.jit(impl2)()
    if bodo.get_rank() == 0:
        pd.testing.assert_frame_equal(
            df_out, impl2(), check_column_type=False, check_dtype=False
        )
    with pytest.raises(
        BodoError,
        match="createDataFrame\(\): 'data' should be a Pandas dataframe or list of Rows",
    ):
        bodo.jit(impl)(3)


@pytest.mark.slow
def test_dataframe_distribution(memory_leak_check):
    """make sure output of toPandas() is distributed if dist flag is set"""

    @bodo.jit(distributed={"df", "df2"})
    def f(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.toPandas()
        return df2.A.sum()

    # NOTE: not using check_func because of extra distributed flag, TODO: support
    df = pd.DataFrame({"A": np.arange(11)})
    df = _get_dist_arg(df)
    assert f(df) == 55


@pytest.mark.slow
def test_dataframe_select(memory_leak_check):
    """test basic column selection in SparkDataFrame.select()"""

    # single column name
    def impl1(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select("A").toPandas()
        return df2

    # multiple column names
    def impl2(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select("A", "B").toPandas()
        return df2

    # list of column names
    def impl3(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select(["A", "B"]).toPandas()
        return df2

    # df column selection
    def impl4(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select(sdf.A).toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    py_out = df[["A"]]
    check_func(
        impl1,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )
    py_out = df[["A", "B"]]
    check_func(
        impl2,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1DVar=True,
        py_output=py_out,
    )
    check_func(
        impl3,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )
    py_out = df[["A"]]
    check_func(
        impl4,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1DVar=True,
        py_output=py_out,
    )


@pytest.mark.slow
def test_dataframe_with_columns(memory_leak_check):
    """test SparkDataFrame.withColumns()"""

    # existing column
    def impl1(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.withColumn("A", sdf.B).toPandas()
        return df2

    # new column
    def impl2(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.withColumn("C", sdf.B).toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    py_out = pd.DataFrame({"A": df.B.values, "B": df.B.values})
    check_func(
        impl1,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )
    py_out = pd.DataFrame({"A": df.A.values, "B": df.B.values, "C": df.B.values})
    check_func(
        impl2,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1DVar=True,
        py_output=py_out,
    )


@pytest.mark.slow
def test_dataframe_with_columns_renamed(memory_leak_check):
    """test SparkDataFrame.withColumnsRenamed()"""

    # column name exists
    def impl1(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.withColumnRenamed("A", "A123").toPandas()
        return df2

    # column name doesn't exist, no-op case
    def impl2(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.withColumnRenamed("C", "A1").toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    py_out = df.rename(columns={"A": "A123"})
    check_func(
        impl1,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )
    check_func(
        impl2,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1DVar=True,
        py_output=df,
    )


@pytest.mark.slow
def test_dataframe_columns(memory_leak_check):
    """test 'columns' attribute of SparkDataFrame"""

    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.columns

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    assert bodo.jit(distributed=["df"])(impl)(df) == ["A", "B"]


@pytest.mark.slow
def test_dataframe_show(memory_leak_check, capsys):
    """test Spark dataframe.show()"""

    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        sdf.show()

    df = pd.DataFrame(
        {"A": np.arange(7), "B": ["S1", "S2", "C", "D", "AB", "AC", "AD"]}
    )
    bodo.jit(distributed=["df"])(impl)(df)
    captured = capsys.readouterr()
    if bodo.get_rank() == 0:
        assert "A" in captured.out
        assert "B" in captured.out
        assert "S1" in captured.out
        assert "0" in captured.out


@pytest.mark.slow
def test_dataframe_print_schema(memory_leak_check, capsys):
    """test Spark dataframe.printSchema()"""

    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        sdf.printSchema()

    df = pd.DataFrame(
        {"A": np.arange(7), "B": ["S1", "S2", "C", "D", "AB", "AC", "AD"]}
    )
    bodo.jit(distributed=["df"])(impl)(df)
    captured = capsys.readouterr()
    if bodo.get_rank() == 0:
        assert "A" in captured.out
        assert "int64" in captured.out
        assert "B" in captured.out


@pytest.mark.slow
def test_dataframe_limit(memory_leak_check):
    """test SparkDataFrame.limit()"""

    def impl1(df, n):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.limit(n).toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    n = 5
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    py_out = df.iloc[:n]
    check_func(
        impl1,
        (df, n),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )


@pytest.mark.slow
def test_dataframe_collect(memory_leak_check):
    """test SparkDataFrame.collect()"""

    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.collect()

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    out_list = bodo.jit(impl)(df)
    if bodo.get_rank() == 0:
        A = df.A.values
        B = df.B.values
        assert out_list == [Row(A=A[i], B=B[i]) for i in range(len(df))]


@pytest.mark.slow
def test_dataframe_take(memory_leak_check):
    """test SparkDataFrame.take()"""

    def impl(df, n):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.take(n)

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    n = 5
    out_list = bodo.jit(impl)(df, n)
    if bodo.get_rank() == 0:
        A = df.A.values[:n]
        B = df.B.values[:n]
        assert out_list == [Row(A=A[i], B=B[i]) for i in range(len(A))]


@pytest.mark.slow
def test_functions_col(memory_leak_check):
    """test creating Column object using F.col()"""

    def impl(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select(F.col("A")).toPandas()
        return df2

    def impl_err(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select(F.col(3)).toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    py_out = df[["A"]]
    check_func(
        impl,
        (df,),
        additional_compiler_arguments={"distributed": ["df2"]},
        only_1D=True,
        py_output=py_out,
    )
    with pytest.raises(
        BodoError,
        match="functions\.col\(\): column name should be a constant string",
    ):
        bodo.jit(distributed=["df"])(impl_err)(df)


@pytest.mark.slow
def test_functions_sum(memory_leak_check):
    """test F.sum()"""

    def impl1(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.select(F.sum(F.col("A"))).toPandas()

    def impl2(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        return sdf.select(F.sum("A")).toPandas()

    def impl_err(df):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        df2 = sdf.select(F.sum(3)).toPandas()
        return df2

    df = pd.DataFrame({"A": np.arange(7), "B": ["A", "B", "C", "D", "AB", "AC", "AD"]})
    # NOTE: using pyout since Spark is slow and fails in multi-process case
    check_func(
        impl1,
        (df,),
        only_1D=True,
        py_output=pd.DataFrame({"sum(A)": [df.A.sum()]}),
        is_out_distributed=False,
    )
    check_func(
        impl2,
        (df,),
        only_1D=True,
        py_output=pd.DataFrame({"sum(A)": [df.A.sum()]}),
        is_out_distributed=False,
    )
    with pytest.raises(
        BodoError,
        match="functions\.sum\(\): input should be a Column object or a constant string",
    ):
        bodo.jit(distributed=["df"])(impl_err)(df)
