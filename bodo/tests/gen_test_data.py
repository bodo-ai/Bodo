# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime
import random

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def gen_lr(file_name, N, D):
    points = np.random.random((N, D))
    responses = np.random.random(N)
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N, D), dtype="f8")
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N,), dtype="f8")
    dset2[:] = responses
    f.close()


def gen_kde_pq(file_name, N):
    df = pd.DataFrame({"points": np.random.random(N)})
    table = pa.Table.from_pandas(df)
    row_group_size = 128
    pq.write_table(table, file_name, row_group_size)


def gen_pq_test(file_name):
    df = pd.DataFrame(
        {
            "one": [-1, np.nan, 2.5, 3.0, 4.0, 6.0, 10.0],
            "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
            "three": [True, False, True, True, True, False, False],
            "four": [-1, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],  # float without NA
            "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],  # str with NA
        }
    )
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "example.parquet")
    pq.write_table(table, "example2.parquet", row_group_size=2)


random.seed(5)
np.random.seed(5)
N = 101
D = 10
gen_lr("lr.hdf5", N, D)

arr = np.arange(N)
f = h5py.File("test_group_read.hdf5", "w")
g1 = f.create_group("G")
dset1 = g1.create_dataset("data", (N,), dtype="i8")
dset1[:] = arr
f.close()

# h5 filter test
n = 11
size = (n, 13, 21, 3)
A = np.random.randint(0, 120, size, np.uint8)
f = h5py.File("h5_test_filter.h5", "w")
f.create_dataset("test", data=A)
f.close()

# test_np_io1
n = 111
A = np.random.ranf(n)
A.tofile("np_file1.dat")

gen_kde_pq("kde.parquet", N)
gen_pq_test("example.parquet")

df = pd.DataFrame(
    {"A": ["bc"] + ["a"] * 3 + ["bc"] * 3 + ["a"], "B": [-8, 1, 2, 3, 1, 5, 6, 7]}
)
df.to_parquet("groupby3.pq")

df = pd.DataFrame(
    {
        "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
        "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
        "C": [
            "small",
            "large",
            "large",
            "small",
            "small",
            "large",
            "small",
            "small",
            "large",
        ],
        "D": [1, 2, 2, 6, 3, 4, 5, 6, 9],
    }
)
df.to_parquet("pivot2.pq")


df = pd.DataFrame(
    {"A": ["bc"] + ["a"] * 3 + ["bc"] * 3 + ["a"]}, index=[-8, 1, 2, 3, 1, 5, 6, 7]
)
df.to_parquet("index_test1.pq")
df = pd.DataFrame(
    index=["bc"] + ["a"] * 3 + ["bc"] * 3 + ["a"], data={"B": [-8, 1, 2, 3, 1, 5, 6, 7]}
)
df.to_parquet("index_test2.pq")


# test datetime64, spark dates
dt1 = pd.DatetimeIndex(["2017-03-03 03:23", "1990-10-23", "1993-07-02 10:33:01"])
df = pd.DataFrame({"DT64": dt1, "DATE": dt1.copy()})
df.to_parquet("pandas_dt.pq")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DateType,
    FloatType,
    LongType,
    MapType,
    Row,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

spark = SparkSession.builder.appName("GenSparkData").getOrCreate()
schema = StructType(
    [StructField("DT64", DateType(), True), StructField("DATE", TimestampType(), True)]
)
sdf = spark.createDataFrame(df, schema)
sdf.write.parquet("sdf_dt.pq", "overwrite")

schema = StructType([StructField("A", LongType(), True)])
A = np.random.randint(0, 100, 1211)
A = [int(a) if random.random() < 0.8 else None for a in A]
data = [Row(a) for a in A]
df = pd.DataFrame({"A": A})
df.to_csv("int_nulls.csv", header=False)
df.to_json("int_nulls.json", orient="records", lines=True)
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("int_nulls_multi.pq", "overwrite")

sdf.write.mode("overwrite").csv("int_nulls_multi.csv")
sdf.write.mode("overwrite").option("header", "true").csv("int_nulls_header_multi.csv")
sdf.write.mode("overwrite").json("int_nulls_multi.json")
sdf = sdf.repartition(1)
sdf.write.parquet("int_nulls_single.pq", "overwrite")
sdf.write.mode("overwrite").csv("int_nulls_single.csv")
sdf.write.mode("overwrite").json("int_nulls_single.json")

# copy data file from int_nulls_single.pq directory to make single file

df = pd.DataFrame({"A": [True, False, False, np.nan, True]})
df.to_parquet("bool_nulls.pq")


# list(int) data generation
schema = StructType([StructField("A", ArrayType(LongType()), True)])
data = [
    Row([1, 2, 3]),
    Row([1, 2]),
    Row(None),
    Row([1, 11, 123, 1, 2]),
    Row([]),
    Row([3, 1]),
]
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("list_int.pq", "overwrite")


# list(float32) data generation
schema = StructType([StructField("B", ArrayType(FloatType()), True)])
data = [
    Row([1.3, -2.4, 3.2]),
    Row([-0.3, 3.3]),
    Row(None),
    Row([1.234, -11.11, 123.0, 1.0, 2.0]),
    Row([]),
    Row([-3.0, 1.2]),
]
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("list_float32.pq", "overwrite")

# CSV reader test
data = "0,2.3,4.6,47736\n" "1,2.3,4.6,47736\n" "2,2.3,4.6,47736\n" "4,2.3,4.6,47736\n"

with open("csv_data1.csv", "w") as f:
    f.write(data)


with open("csv_data_infer1.csv", "w") as f:
    f.write("A,B,C,D\n" + data)

data = (
    "0,2.3,2015-01-03,47736\n"
    "1,2.3,1966-11-13,47736\n"
    "2,2.3,1998-05-21,47736\n"
    "4,2.3,2018-07-11,47736\n"
)

with open("csv_data_date1.csv", "w") as f:
    f.write(data)

data = "2015-01-03\n" "1966-11-13\n" "1998-05-21\n" "2018-07-11\n"

with open("csv_data_only_date1.csv", "w") as f:
    f.write(data)

data = (
    "2015-01-03,1966-11-13\n"
    "1966-11-13,1998-05-21\n"
    "1998-05-21,2018-07-11\n"
    "2018-07-11,2015-01-03\n"
)

with open("csv_data_only_date2.csv", "w") as f:
    f.write(data)

# test_csv_cat1
data = "2,B,SA\n" "3,A,SBC\n" "4,C,S123\n" "5,B,BCD\n"

with open("csv_data_cat1.csv", "w") as f:
    f.write(data)

# test_csv_single_dtype1
data = "2,4.1\n" "3,3.4\n" "4,1.3\n" "5,1.1\n"

with open("csv_data_dtype1.csv", "w") as f:
    f.write(data)


# generated data for parallel merge_asof testing
df1 = pd.DataFrame(
    {
        "time": pd.DatetimeIndex(
            ["2017-01-03", "2017-01-06", "2017-02-15", "2017-02-21"]
        ),
        "B": [4, 5, 9, 6],
    }
)
df2 = pd.DataFrame(
    {
        "time": pd.DatetimeIndex(
            [
                "2017-01-01",
                "2017-01-14",
                "2017-01-16",
                "2017-02-23",
                "2017-02-23",
                "2017-02-25",
            ]
        ),
        "A": [2, 3, 7, 8, 9, 10],
    }
)
df1.to_parquet("asof1.pq")
df2.to_parquet("asof2.pq")


# data for list(str) array read from Parquet
df = pd.DataFrame(
    {
        "A": [
            None,
            ["холодн", "¿abc¡Y "],
            ["¡úú,úũ¿ééé"],
            [],
            None,
            ["ABC", "C", "", "A"],
            ["늘 저녁", ",고싶다ㅠ"],
            [""],
        ]
        * 3
    }
)
df.to_parquet("list_str_arr.pq")
sdf = spark.createDataFrame(df)
sdf.write.parquet("list_str_parts.pq", "overwrite")


dtype = StructType(
    [StructField("X", LongType(), True), StructField("Y", FloatType(), True)]
)
schema = StructType([StructField("A", dtype, True)])
data = [Row(Row(1, 1.1)), Row(Row(4, 2.2)), None, Row(Row(-1, 3.0))] * 2
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("struct.pq", "overwrite")


dtype = MapType(LongType(), StringType())
schema = StructType([StructField("A", dtype, True)])
data = [
    Row({1: "AA", 2: "BB", 3: "C"}),
    Row({6: "H"}),
    Row({1: "", 4: None, 0: "J"}),
] * 3
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("map.pq", "overwrite")

# Arrow cannot read/write map type or even convert to Pandas yet (as of 0.17.1)
# pa.array([[{"key": 1, "value": "A"}, {"key": 2, "value": ""}],
#     [{"key": 6, "value": "BCD"}]]*3, pa.map_(pa.int64(), pa.string()))

spark.stop()

# data for str array read from csv
df = pd.DataFrame(
    {
        "A": [
            None,
            "холодн",
            "¿abc¡Y",
            "¡úú,úũ¿ééé",
            None,
            "ABC",
            "C",
            "",
            "A",
            "늘 저녁",
            ",고싶다ㅠ",
            "",
        ]
        * 3,
        "B": [
            None,
            "холодн",
            "¿abc¡Y",
            "¡úú,úũ¿ééé",
            None,
            "ABC",
            "C",
            "",
            "A",
            "늘 저녁",
            ",고싶다ㅠ",
            "",
        ]
        * 3,
    }
)
df.to_csv("str_arr.csv", header=False, index=False)
df.to_json("str_arr.json", orient="records", lines=True)
sdf = spark.createDataFrame(df)
sdf.write.mode("overwrite").csv("str_arr_parts.csv")
sdf.write.mode("overwrite").json("str_arr_parts.json")
sdf.repartition(1).write.mode("overwrite").csv("str_arr_single.csv")
sdf.repartition(1).write.mode("overwrite").json("str_arr_single.json")


df = pd.DataFrame(
    {
        "one": [
            -1,
            np.nan,
            2.5,
            3.0,
            4.0,
            6.0,
            10.0,
            -1,
            np.nan,
            2.5,
            3.0,
            4.0,
            6.0,
            10.0,
        ],
        "two": [
            "foo",
            "bar",
            "baz",
            "foo",
            "bar",
            "baz",
            "foo",
            "foo",
            "bar",
            "baz",
            "foo",
            "bar",
            "baz",
            "foo",
        ],
        "three": [
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
        ],
        "four": [
            -1,
            5.1,
            2.5,
            3.0,
            4.0,
            6.0,
            11.0,
            -1,
            5.1,
            2.5,
            3.0,
            4.0,
            6.0,
            11.0,
        ],  # float without NA
        "five": [
            "foo",
            "bar",
            "baz",
            "",
            "bar",
            "baz",
            "foo",
            "foo",
            "bar",
            "baz",
            "",
            "bar",
            "baz",
            "foo",
        ],  # str without NA
    }
)
sdf = spark.createDataFrame(df)
df.to_json("example.json", orient="records", lines=True)
df.to_csv("example.csv", index=False)
sdf.write.mode("overwrite").json("example_multi.json")
sdf.write.mode("overwrite").option("header", "true").csv("example_multi.csv")
sdf = sdf.repartition(1)
sdf.write.mode("overwrite").json("example_single.json")
sdf.write.mode("overwrite").option("header", "true").csv("example_single.csv")
spark.stop()

# data for testing read of parquet files with unsupported column types in unselected
# columns.
# using list(list(int)) type that we are not likely to support soon
t = pa.Table.from_pandas(pd.DataFrame({"A": [[[1], [3]], [[3, 5]]], "B": [3.5, 1.2]}))
pq.write_table(t, "nested_struct_example.pq")


# date32 values
df = pd.DataFrame(
    {
        "A": [
            datetime.date(2012, 1, 2),
            datetime.date(1944, 12, 21),
            None,
            datetime.date(1999, 6, 11),
        ]
        * 3
    }
)
df.to_parquet("date32_1.pq")


# excel file test
df = pd.DataFrame(
    {
        "A": [3, 4, 1, 5, 7, -3],
        "B": [1.1, 3.2, -1.1, 3.555, 1.31, 111.2],
        "C": [
            "2012-01-03",
            "1988-03-11",
            "2015-11-11",
            "2020-08-01",
            "1998-09-09",
            "2001-03-19",
        ],
        "D": ["AAA", None, "C", "", "AB", "12"],
        "E": [True, True, False, False, True, False],
    }
)
df.to_excel("../data.xlsx", index=False)
