# Copyright (C) 2019 Bodo Inc. All rights reserved.
import random
import h5py
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


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
    StructType,
    StructField,
    DateType,
    TimestampType,
    LongType,
    Row,
)

spark = SparkSession.builder.appName("GenSparkData").getOrCreate()
schema = StructType(
    [StructField("DT64", DateType(), True), StructField("DATE", TimestampType(), True)]
)
sdf = spark.createDataFrame(df, schema)
sdf.write.parquet("sdf_dt.pq", "overwrite")


schema = StructType([StructField("A", LongType(), True)])
A = np.random.randint(0, 100, 1211)
data = [Row(int(a)) if random.random() < 0.8 else Row(None) for a in A]
sdf = spark.createDataFrame(data, schema)
sdf.write.parquet("int_nulls_multi.pq", "overwrite")
sdf = sdf.repartition(1)
sdf.write.parquet("int_nulls_single.pq", "overwrite")
# copy data file from int_nulls_single.pq directory to make single file

spark.stop()

df = pd.DataFrame({"A": [True, False, False, np.nan, True]})
df.to_parquet("bool_nulls.pq")


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
