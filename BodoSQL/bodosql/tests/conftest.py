import gc
import os
import sys

import bodo
import bodo.utils.allocation_tracking
import numpy as np
import pandas as pd
import pytest
from bodo.tests.utils import gen_nonascii_list
from numba.core.runtime import rtsys
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
import json, hashlib

# Fix Issue on Azure CI where the driver defaults to a different Python version
# See: https://stackoverflow.com/a/65010346/14810655
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession


def pytest_addoption(parser):
    """Used with caching tests, stores the --is_cached flag into the pytestconfig"""
    parser.addoption("--is_cached", type=str, action="store", default="n")


@pytest.fixture()
def is_cached(pytestconfig):
    """Fixture used with caching tests, returns the value passed to pytest
    with --is_cached, or 'n' if not passed."""
    return pytestconfig.getoption("is_cached")


# similar to Pandas
@pytest.fixture(scope="session")
def datapath():
    """Get the path to a test data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``bodosql/tests/data``

    Returns
    -------
    path : path including ``bodosql/tests/data``.

    Raises
    ------
    ValueError
        If the path doesn't exist.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

    def deco(*args, check_exists=True):
        path = os.path.join(BASE_PATH, *args)
        if check_exists and not os.path.exists(path):
            msg = "Could not find file {}."
            raise ValueError(msg.format(path))
        return path

    return deco


@pytest.fixture(scope="module")
def spark_info():
    spark = SparkSession.builder.appName("TestSQL").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def basic_df():
    df = pd.DataFrame(
        {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
    )
    return {"table1": df}


@pytest.fixture
def zeros_df():
    """
    DataFrame containing zero entries. This is used
    to check issues with dividing by zero
    """
    df = pd.DataFrame({"A": np.arange(12), "B": [0, 1, -2, 1] * 3})
    return {"table1": df}


@pytest.fixture(
    params=[
        pytest.param(np.int8, marks=pytest.mark.slow),
        pytest.param(np.uint8, marks=pytest.mark.slow),
        pytest.param(np.int16, marks=pytest.mark.slow),
        pytest.param(np.uint16, marks=pytest.mark.slow),
        pytest.param(np.int32, marks=pytest.mark.slow),
        pytest.param(np.uint32, marks=pytest.mark.slow),
        np.int64,
        pytest.param(np.uint64, marks=pytest.mark.slow),
        pytest.param(np.float32, marks=pytest.mark.slow),
        np.float64,
    ]
)
def bodosql_numeric_types(request):
    """
    Fixture for dataframes with numeric BodoSQL types:
        - int8
        - uint8
        - int16
        - uint16
        - int32
        - uint32
        - int64
        - uint64
        - float32
        - float64

    For each datatable, it provides a dictionary mapping table1 -> DataFrame.
    All dataframes have the same column names so the queries can be applied to
    each table.
    """
    dtype = request.param
    int_data = {"A": [1, 2, 3] * 4, "B": [4, 5, 6] * 4, "C": [7, 8, 9] * 4}
    return {"table1": pd.DataFrame(data=int_data, dtype=dtype)}


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [4, 1, 2, 3] * 4,
                    "B": [1.0, 2.0, 3.0, 4.0] * 4,
                    "C": ["bird", "dog", "flamingo", "cat"] * 4,
                }
            ),
            "table2": pd.DataFrame(
                {
                    "A": [3, 1, 2, 4] * 4,
                    "B": [1.0, 2.0, 4.0, 3.0] * 4,
                    "D": [
                        pd.Timestamp(2021, 5, 19),
                        pd.Timestamp(1999, 12, 31),
                        pd.Timestamp(2020, 10, 11),
                        pd.Timestamp(2025, 1, 1),
                    ]
                    * 4,
                }
            ),
            "table3": pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6] * 2}),
        },
        {
            "table1": pd.DataFrame(
                {
                    "A": [1, 1, 3, 3, 5, 7] * 2,
                    "B": [1.0, 2.0, 4.0, 3.0] * 3,
                    "C": ["T1_1", "T1_2", "T1_3", "T1_4", "T1_5", "T1_6"] * 2,
                }
            ),
            "table2": pd.DataFrame(
                {
                    "A": [2, 4, 6, 6, 1, 1] * 2,
                    "B": [1.0, 2.0, 4.0, 3.0] * 3,
                    "D": ["T2_1", "T2_2", "T2_3", "T2_4", "T2_5", "T2_6"] * 2,
                }
            ),
            "table3": pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6] * 2}),
        },
        pytest.param(
            {
                "table1": pd.DataFrame(
                    {
                        "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, object),
                        "B": np.array(
                            [bytes(32), b"abcde", b"ihohi04324", None] * 3, object
                        ),
                        "C": np.array(
                            [None, b"poiu", b"fewfqqqqq", b"3f3"] * 3, object
                        ),
                    }
                ),
                "table2": pd.DataFrame(
                    {
                        "A": np.array([b"cew", b"abc", b"r2r", None] * 3, object),
                        "B": np.array(
                            [bytes(12), b"abcde", b"ihohi04324", None] * 3, object
                        ),
                        "D": np.array([b"r32r2", b"poiu", b"3r32", b"3f3"] * 3, object),
                    }
                ),
                "table3": pd.DataFrame(
                    {"Y": [b"abc", b"c", b"cew", b"ce2r", b"r2r", None] * 2}
                ),
            },
            id="join_binary_keys",
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    {
                        "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
                        "B": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                        "C": ["T1_1", "T1_2", "T1_3", "T1_4", "T1_5", "T1_6"] * 2,
                    }
                ),
                "table2": pd.DataFrame(
                    {
                        "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
                        "B": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                        "D": ["T2_1", "T2_2", "T2_3", "T2_4", "T2_5", "T2_6"] * 2,
                    }
                ),
                "table3": pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6] * 2}),
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    {
                        "A": (gen_nonascii_list(2) + ["A", "B"]) * 3,
                        "B": gen_nonascii_list(3) * 4,
                        "C": gen_nonascii_list(6)[3:] * 4,
                    }
                ),
                "table2": pd.DataFrame(
                    {
                        "A": ["a", "b", "c", None, "e", "aa"] * 2,
                        "B": gen_nonascii_list(6)[3:] * 4,
                        "D": gen_nonascii_list(9)[6:] * 4,
                    }
                ),
                "table3": pd.DataFrame({"Y": ["Z", "Y", "X", "W", "V", "U"] * 2}),
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def join_dataframes(request):
    """
    Fixture with similar dataframes to use for join queries.

    table1 has columns A, B, C
    table2 has columns A, B, D
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame({"A": [1, 2, 3], "D": [4, 5, 6]}),
            "table2": pd.DataFrame({"B": [1, 2, 3], "C": [4, 5, 6]}),
        },
    ]
)
def simple_join_fixture(request):
    """a very simple context with no overlapping column names
    mostly used for testing join optimizations
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            {
                "table1": pd.DataFrame(
                    data={
                        "E": [2**63, 2**62, 2**61] * 4,
                        "F": [2**60, 2**59, 2**58] * 4,
                    },
                    dtype=np.uint64,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    data={
                        "E": [(-2) ** 61, (-2) ** 60, (-2) ** 59] * 4,
                        "F": [(-2) ** 58, (-2) ** 57, (-2) ** 56] * 4,
                    },
                    dtype=np.int64,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    data={
                        "E": [2**31, 2**30, 2**29] * 4,
                        "F": [2**28, 2**27, 2**26] * 4,
                    },
                    dtype=np.uint32,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    data={
                        "E": [(-2) ** 30, (-2) ** 29, (-2) ** 28] * 4,
                        "F": [(-2) ** 27, (-2) ** 26, (-2) ** 25] * 4,
                    },
                    dtype=np.int32,
                )
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def bodosql_large_numeric_types(request):
    """
    Fixture used to check very large values for numeric types.
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [
                        np.datetime64("2011-01-01"),
                        np.datetime64("1971-02-02"),
                        np.datetime64("2021-03-03"),
                        np.datetime64("2004-12-07"),
                    ]
                    * 3,
                    "B": [
                        np.datetime64("2007-01-01T03:30"),
                        np.datetime64("nat"),
                        np.datetime64("2020-12-01T13:56:03.172"),
                    ]
                    * 4,
                    "C": [
                        pd.Timestamp(2021, 11, 21),
                        pd.Timestamp(2022, 1, 12),
                        pd.Timestamp(2021, 3, 3),
                    ]
                    * 4,
                }
            ),
        },
    ]
)
def bodosql_datetime_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [
                        np.timedelta64(10, "Y"),
                        np.timedelta64(9, "M"),
                        np.timedelta64(8, "W"),
                    ]
                    * 4,
                    "B": [
                        np.timedelta64("nat"),
                        np.timedelta64(6, "h"),
                        np.timedelta64(5, "m"),
                    ]
                    * 4,
                    "C": [
                        np.timedelta64(4, "s"),
                        np.timedelta64(3, "ms"),
                        np.timedelta64(2000000, "us"),
                    ]
                    * 4,
                }
            )
        },
    ]
)
def bodosql_interval_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [True, False, False, None] * 3,
                    "B": [False, None, True, False] * 3,
                    "C": [False, False, False, False] * 3,
                }
            )
        }
    ]
)
def bodosql_boolean_types(request):
    return request.param


@pytest.fixture(
    params=[
        "2011-01-01",
        "1971-02-02",
        "2021-03-03",
        "2004-12-07",
        "2007-01-01 03:30:00",
    ]
)
def timestamp_literal_strings(request):
    """
    Fixture containing timestamp literal strings for use in testing
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": ["hElLo", "GoOdByE", "SpEaK", "WeIrD"] * 3,
                    "B": ["hello", "world", "how"] * 4,
                    "C": ["ARE", "YOU", "TODAY"] * 4,
                }
            )
        },
        pytest.param(
            {
                "table1": pd.DataFrame(
                    {
                        "A": ["HELLO", "HeLlLllo", "heyO", "HI"] * 3,
                        "B": ["hi", "HaPpY", "how"] * 4,
                        "C": ["Ello", "O", "HowEver"] * 4,
                    }
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "table1": pd.DataFrame(
                    {
                        "A": (gen_nonascii_list(2) + ["A", "B"]) * 3,
                        "B": gen_nonascii_list(3) * 4,
                        "C": gen_nonascii_list(3) * 4,
                    }
                )
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def bodosql_string_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": ["0", "1", "24", "27"] * 3,
                    "B": ["43", "121", "43"] * 4,
                    "C": ["44", "14", "42"] * 4,
                }
            )
        },
    ]
)
def bodosql_integers_string_types(request):
    """Fixture for strings where all values can be cast to integers"""
    return request.param


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": ["2011-01-01", "1971-02-02", "2021-03-03", "2004-12-07"] * 3,
                    "B": ["2021-11-09", "2019-08-25", "2017-05-04"] * 4,
                    "C": ["2010-09-14", "2021-01-12", "2022-07-13"] * 4,
                }
            )
        },
    ]
)
def bodosql_date_string_types(request):
    """Fixture for strings where all values can be cast to dates"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("Int8", marks=pytest.mark.slow),
        pytest.param("UInt8", marks=pytest.mark.slow),
        pytest.param("Int16", marks=pytest.mark.slow),
        pytest.param("UInt16", marks=pytest.mark.slow),
        pytest.param("Int32", marks=pytest.mark.slow),
        pytest.param("UInt32", marks=pytest.mark.slow),
        "Int64",
        pytest.param("UInt64", marks=pytest.mark.slow),
    ]
)
def bodosql_nullable_numeric_types(request):
    """
    Fixture for dataframes nullable numeric BodoSQL types:
        - Int8
        - UInt8
        - Int16
        - UInt16
        - Int32
        - UInt32
        - Int64
        - UInt64

    For each datatable, it provides a dictionary mapping table1 -> DataFrame.
    All dataframes have the same column names so the queries can be applied to
    each table.
    """
    dtype = request.param
    int_data = {
        "A": [1, 2, 3, None] * 3,
        "B": [4, None, 5, 6] * 3,
        "C": [7, 8, None, 9] * 3,
    }
    return {"table1": pd.DataFrame(data=int_data, dtype=dtype)}


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, object),
                    "B": np.array(
                        [bytes(32), b"abcde", b"ihohi04324", None] * 3, object
                    ),
                    "C": np.array([None, b"poiu", b"fewfqqqqq", b"3f3"] * 3, object),
                }
            ),
        ),
    ]
)
def bodosql_binary_types(request):
    """
    Fixture for dataframe of binary array types in BodoSQL.
    Binary arrays are nullable.
    """
    return {"table1": request.param}


@pytest.fixture(
    params=[
        "=",
        "<>",
        pytest.param("!=", marks=pytest.mark.slow),
        pytest.param("<=", marks=pytest.mark.slow),
        pytest.param("<", marks=pytest.mark.slow),
        pytest.param(">=", marks=pytest.mark.slow),
        pytest.param(">", marks=pytest.mark.slow),
        pytest.param("<=>", marks=pytest.mark.slow),
    ]
)
def comparison_ops(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            "BIT_AND",
            marks=pytest.mark.skip(
                "Bitwise agregations not currently supported, see [BE-919]"
            ),
        ),
        pytest.param(
            "BIT_OR",
            marks=pytest.mark.skip(
                "Bitwise agregations not currently supported, see [BE-919]"
            ),
        ),
        pytest.param(
            "BIT_XOR",
            marks=pytest.mark.skip(
                "Bitwise agregations not currently supported, see [BE-919]"
            ),
        ),
        "AVG",
        "COUNT",
        "MAX",
        "MIN",
        "STDDEV",
        pytest.param("STDDEV_SAMP", marks=pytest.mark.slow),
        "SUM",
        "VARIANCE",
        pytest.param("VAR_SAMP", marks=pytest.mark.slow),
        "STDDEV_POP",
        "VAR_POP",
    ]
)
def numeric_agg_builtin_funcs(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("TINYINT", marks=pytest.mark.slow),
        pytest.param("SMALLINT", marks=pytest.mark.slow),
        "INTEGER",
        pytest.param("BIGINT", marks=pytest.mark.slow),
        "FLOAT",
        pytest.param("DOUBLE", marks=pytest.mark.slow),
        pytest.param("DECIMAL", marks=pytest.mark.slow),
    ]
)
def sql_numeric_typestrings(request):
    """
    Collection of numeric sql types
    """
    return request.param


@pytest.fixture(params=["-", "+", "*", "/"])
def arith_ops(request):
    """fixture that returns arithmatic operators"""
    return request.param


@pytest.fixture(params=["-", "+"])
def datetime_arith_ops(request):
    """fixture that returns datetime valid arithmetic operators"""
    return request.param


@pytest.fixture(params=["DATE", "TIMESTAMP"])
def sql_datetime_typestrings(request):
    """
    Collection of numeric sql types
    """
    return request.param


@pytest.fixture(
    params=[
        0,
        pytest.param(1, marks=pytest.mark.slow),
        pytest.param(-1, marks=pytest.mark.slow),
        pytest.param(0.0, marks=pytest.mark.slow),
        pytest.param(1.0, marks=pytest.mark.slow),
        -1.0,
        pytest.param(2, marks=pytest.mark.slow),
        0.001,
        pytest.param(4, marks=pytest.mark.slow),
        -7,
        pytest.param(11, marks=pytest.mark.slow),
    ]
)
def numeric_values(request):
    """
    Collection of numeric values used for testing, all the integer values should fit within a bytee
    """
    return request.param


@pytest.fixture(scope="module")
def tpch_data(datapath):
    """
    Fixture with TPCH data for BodoSQL Contexts.
    """
    (
        customer_df,
        orders_df,
        lineitem_df,
        nation_df,
        region_df,
        supplier_df,
        part_df,
        partsupp_df,
    ) = load_tpch_data(datapath("tpch-test-data/parquet"))
    # BodoSQL doesn't support Date datatypes yet. Convert the input
    # data to datetime64 for BodoSQL. Spark has correctness issues
    # if this is converted though, so keep Spark data types separate.
    spark_lineitem_df = lineitem_df.copy(deep=True)
    spark_orders_df = orders_df.copy(deep=True)
    lineitem_df["L_SHIPDATE"] = lineitem_df["L_SHIPDATE"].astype("datetime64[ns]")
    lineitem_df["L_COMMITDATE"] = lineitem_df["L_COMMITDATE"].astype("datetime64[ns]")
    lineitem_df["L_RECEIPTDATE"] = lineitem_df["L_RECEIPTDATE"].astype("datetime64[ns]")
    orders_df["O_ORDERDATE"] = orders_df["O_ORDERDATE"].astype("datetime64[ns]")
    bodo_dataframe_dict = {
        "customer": customer_df,
        "orders": orders_df,
        "lineitem": lineitem_df,
        "nation": nation_df,
        "region": region_df,
        "supplier": supplier_df,
        "part": part_df,
        "partsupp": partsupp_df,
    }
    spark_dataframe_dict = {
        "customer": customer_df,
        "orders": spark_orders_df,
        "lineitem": spark_lineitem_df,
        "nation": nation_df,
        "region": region_df,
        "supplier": supplier_df,
        "part": part_df,
        "partsupp": partsupp_df,
    }
    return bodo_dataframe_dict, spark_dataframe_dict


@bodo.jit(returns_maybe_distributed=False, cache=True)
def load_tpch_data(dir_name):
    """Load the necessary TPCH dataframes given a root directory.
    We use bodo.jit so we can read easily from a directory."""
    customer_df = pd.read_parquet(dir_name + "/customer.parquet/")
    orders_df = pd.read_parquet(dir_name + "/orders.parquet/")
    lineitem_df = pd.read_parquet(dir_name + "/lineitem.parquet/")
    nation_df = pd.read_parquet(dir_name + "/nation.parquet/")
    region_df = pd.read_parquet(dir_name + "/region.parquet/")
    supplier_df = pd.read_parquet(dir_name + "/supplier.parquet/")
    part_df = pd.read_parquet(dir_name + "/part.parquet/")
    partsupp_df = pd.read_parquet(dir_name + "/partsupp.parquet/")
    return (
        customer_df,
        orders_df,
        lineitem_df,
        nation_df,
        region_df,
        supplier_df,
        part_df,
        partsupp_df,
    )


@pytest.fixture(scope="module")
def tpcxbb_data(datapath):
    """
    Fixture with TPCxBB data for BodoSQL Contexts.
    """
    # TODO: Update when we know how to generate the full
    # TPCxBB dataset.
    (
        store_sales_df,
        item_df,
        customer_df,
        customer_address_df,
        customer_demographics_df,
        date_dim_df,
        product_reviews_df,
        store_df,
        web_clickstreams_df,
        web_sales_df,
        household_demographics_df,
        inventory_df,
        item_marketprices_df,
        promotion_df,
        store_returns_df,
        time_dim_df,
        warehouse_df,
        web_page_df,
        web_returns_df,
    ) = load_tpcxbb_data(datapath("tpcxbb-test-data"))
    # Some end dates are all null values, so its unclear its type.
    # Convert to string.
    item_df["i_rec_end_date"] = item_df["i_rec_end_date"].astype(pd.StringDtype())
    store_df["s_rec_end_date"] = store_df["s_rec_end_date"].astype(pd.StringDtype())
    web_page_df["wp_rec_end_date"] = web_page_df["wp_rec_end_date"].astype(
        pd.StringDtype()
    )
    time_dim_df["t_meal_time"] = time_dim_df["t_meal_time"].astype(pd.StringDtype())
    dataframe_dict = {
        "store_sales": store_sales_df,
        "item": item_df,
        "customer": customer_df,
        "customer_address": customer_address_df,
        "customer_demographics": customer_demographics_df,
        "date_dim": date_dim_df,
        "product_reviews": product_reviews_df,
        "store": store_df,
        "web_clickstreams": web_clickstreams_df,
        "web_sales": web_sales_df,
        "household_demographics": household_demographics_df,
        "inventory": inventory_df,
        "item_marketprices": item_marketprices_df,
        "promotion": promotion_df,
        "store_returns": store_returns_df,
        "time_dim": time_dim_df,
        "warehouse": warehouse_df,
        "web_page": web_page_df,
        "web_returns": web_returns_df,
    }
    pyspark_schemas = {
        "store_sales": StructType(
            [
                StructField("ss_sold_date_sk", IntegerType(), True),
                StructField("ss_sold_time_sk", IntegerType(), True),
                StructField("ss_item_sk", IntegerType(), True),
                StructField("ss_customer_sk", IntegerType(), True),
                StructField("ss_cdemo_sk", IntegerType(), True),
                StructField("ss_hdemo_sk", IntegerType(), True),
                StructField("ss_addr_sk", IntegerType(), True),
                StructField("ss_store_sk", IntegerType(), True),
                StructField("ss_promo_sk", IntegerType(), True),
                StructField("ss_ticket_number", IntegerType(), True),
                StructField("ss_quantity", IntegerType(), True),
                StructField("ss_wholesale_cost", DoubleType(), False),
                StructField("ss_list_price", DoubleType(), False),
                StructField("ss_sales_price", DoubleType(), False),
                StructField("ss_ext_discount_amt", DoubleType(), False),
                StructField("ss_ext_sales_price", DoubleType(), False),
                StructField("ss_ext_wholesale_cost", DoubleType(), False),
                StructField("ss_ext_list_price", DoubleType(), False),
                StructField("ss_ext_tax", DoubleType(), False),
                StructField("ss_coupon_amt", DoubleType(), False),
                StructField("ss_net_paid", DoubleType(), False),
                StructField("ss_net_paid_inc_tax", DoubleType(), False),
                StructField("ss_net_profit", DoubleType(), False),
            ]
        ),
        "item": StructType(
            [
                StructField("i_item_sk", IntegerType(), True),
                StructField("i_item_id", StringType(), True),
                StructField("i_rec_start_date", StringType(), True),
                StructField("i_rec_end_date", StringType(), True),
                StructField("i_item_desc", StringType(), True),
                StructField("i_current_price", DoubleType(), False),
                StructField("i_wholesale_cost", DoubleType(), False),
                StructField("i_brand_id", IntegerType(), True),
                StructField("i_brand", StringType(), True),
                StructField("i_class_id", IntegerType(), True),
                StructField("i_class", StringType(), True),
                StructField("i_category_id", IntegerType(), True),
                StructField("i_category", StringType(), True),
                StructField("i_manufact_id", IntegerType(), True),
                StructField("i_manufact", StringType(), True),
                StructField("i_size", StringType(), True),
                StructField("i_formulation", StringType(), True),
                StructField("i_color", StringType(), True),
                StructField("i_units", StringType(), True),
                StructField("i_container", StringType(), True),
                StructField("i_manager_id", IntegerType(), True),
                StructField("i_product_name", StringType(), True),
            ]
        ),
        # Taken from https://github.com/rapidsai/gpu-bdb/tree/main/gpu_bdb/spark_table_schemas
        "customer": StructType(
            [
                StructField("c_customer_sk", IntegerType(), True),
                StructField("c_customer_id", StringType(), True),
                StructField("c_current_cdemo_sk", IntegerType(), True),
                StructField("c_current_hdemo_sk", IntegerType(), True),
                StructField("c_current_addr_sk", IntegerType(), True),
                StructField("c_first_shipto_date_sk", IntegerType(), True),
                StructField("c_first_sales_date_sk", IntegerType(), True),
                StructField("c_salutation", StringType(), True),
                StructField("c_first_name", StringType(), True),
                StructField("c_last_name", StringType(), True),
                StructField("c_preferred_cust_flag", StringType(), True),
                StructField("c_birth_day", IntegerType(), True),
                StructField("c_birth_month", IntegerType(), True),
                StructField("c_birth_year", IntegerType(), True),
                StructField("c_birth_country", StringType(), True),
                StructField("c_login", StringType(), True),
                StructField("c_email_address", StringType(), True),
                StructField("c_last_review_date", StringType(), True),
            ]
        ),
        "customer_address": StructType(
            [
                StructField("ca_address_sk", IntegerType(), True),
                StructField("ca_address_id", StringType(), True),
                StructField("ca_street_number", StringType(), True),
                StructField("ca_street_name", StringType(), True),
                StructField("ca_street_type", StringType(), True),
                StructField("ca_suite_number", StringType(), True),
                StructField("ca_city", StringType(), True),
                StructField("ca_county", StringType(), True),
                StructField("ca_state", StringType(), True),
                StructField("ca_zip", StringType(), True),
                StructField("ca_country", StringType(), True),
                StructField("ca_gmt_offset", DoubleType(), False),
                StructField("ca_location_type", StringType(), True),
            ]
        ),
        "customer_demographics": StructType(
            [
                StructField("cd_demo_sk", IntegerType(), True),
                StructField("cd_gender", StringType(), True),
                StructField("cd_marital_status", StringType(), True),
                StructField("cd_education_status", StringType(), True),
                StructField("cd_purchase_estimate", IntegerType(), True),
                StructField("cd_credit_rating", StringType(), True),
                StructField("cd_dep_count", IntegerType(), True),
                StructField("cd_dep_employed_count", IntegerType(), True),
                StructField("cd_dep_college_count", IntegerType(), True),
            ]
        ),
        "date_dim": StructType(
            [
                StructField("d_date_sk", IntegerType(), True),
                StructField("d_date_id", StringType(), True),
                StructField("d_date", StringType(), True),
                StructField("d_month_seq", IntegerType(), True),
                StructField("d_week_seq", IntegerType(), True),
                StructField("d_quarter_seq", IntegerType(), True),
                StructField("d_year", IntegerType(), True),
                StructField("d_dow", IntegerType(), True),
                StructField("d_moy", IntegerType(), True),
                StructField("d_dom", IntegerType(), True),
                StructField("d_qoy", IntegerType(), True),
                StructField("d_fy_year", IntegerType(), True),
                StructField("d_fy_quarter_seq", IntegerType(), True),
                StructField("d_fy_week_seq", IntegerType(), True),
                StructField("d_day_name", StringType(), True),
                StructField("d_quarter_name", StringType(), True),
                StructField("d_holiday", StringType(), True),
                StructField("d_weekend", StringType(), True),
                StructField("d_following_holiday", StringType(), True),
                StructField("d_first_dom", IntegerType(), True),
                StructField("d_last_dom", IntegerType(), True),
                StructField("d_same_day_ly", IntegerType(), True),
                StructField("d_same_day_lq", IntegerType(), True),
                StructField("d_current_day", StringType(), True),
                StructField("d_current_week", StringType(), True),
                StructField("d_current_month", StringType(), True),
                StructField("d_current_quarter", StringType(), True),
                StructField("d_current_year", StringType(), True),
            ]
        ),
        "product_reviews": StructType(
            [
                StructField("pr_review_sk", IntegerType(), True),
                StructField("pr_review_date", StringType(), True),
                StructField("pr_review_time", StringType(), True),
                StructField("pr_review_rating", IntegerType(), True),
                StructField("pr_item_sk", IntegerType(), True),
                StructField("pr_user_sk", IntegerType(), True),
                StructField("pr_order_sk", IntegerType(), True),
                StructField("pr_review_content", StringType(), True),
            ]
        ),
        "store": StructType(
            [
                StructField("s_store_sk", IntegerType(), True),
                StructField("s_store_id", StringType(), True),
                StructField("s_rec_start_date", StringType(), True),
                StructField("s_rec_end_date", StringType(), True),
                StructField("s_closed_date_sk", IntegerType(), True),
                StructField("s_store_name", StringType(), True),
                StructField("s_number_employees", IntegerType(), True),
                StructField("s_floor_space", IntegerType(), True),
                StructField("s_hours", StringType(), True),
                StructField("s_manager", StringType(), True),
                StructField("s_market_id", IntegerType(), True),
                StructField("s_geography_class", StringType(), True),
                StructField("s_market_desc", StringType(), True),
                StructField("s_market_manager", StringType(), True),
                StructField("s_division_id", IntegerType(), True),
                StructField("s_division_name", StringType(), True),
                StructField("s_company_id", IntegerType(), True),
                StructField("s_company_name", StringType(), True),
                StructField("s_street_number", StringType(), True),
                StructField("s_street_name", StringType(), True),
                StructField("s_street_type", StringType(), True),
                StructField("s_suite_number", StringType(), True),
                StructField("s_city", StringType(), True),
                StructField("s_county", StringType(), True),
                StructField("s_state", StringType(), True),
                StructField("s_zip", StringType(), True),
                StructField("s_country", StringType(), True),
                StructField("s_gmt_offset", DoubleType(), False),
                StructField("s_tax_precentage", DoubleType(), False),
            ]
        ),
        "web_clickstreams": StructType(
            [
                StructField("wcs_click_date_sk", IntegerType(), True),
                StructField("wcs_click_time_sk", IntegerType(), True),
                StructField("wcs_sales_sk", IntegerType(), True),
                StructField("wcs_item_sk", IntegerType(), True),
                StructField("wcs_web_page_sk", IntegerType(), True),
                StructField("wcs_user_sk", IntegerType(), True),
            ]
        ),
        "web_sales": StructType(
            [
                StructField("ws_sold_date_sk", IntegerType(), True),
                StructField("ws_sold_time_sk", IntegerType(), True),
                StructField("ws_ship_date_sk", IntegerType(), True),
                StructField("ws_item_sk", IntegerType(), True),
                StructField("ws_bill_customer_sk", IntegerType(), True),
                StructField("ws_bill_cdemo_sk", IntegerType(), True),
                StructField("ws_bill_hdemo_sk", IntegerType(), True),
                StructField("ws_bill_addr_sk", IntegerType(), True),
                StructField("ws_ship_customer_sk", IntegerType(), True),
                StructField("ws_ship_cdemo_sk", IntegerType(), True),
                StructField("ws_ship_hdemo_sk", IntegerType(), True),
                StructField("ws_ship_addr_sk", IntegerType(), True),
                StructField("ws_web_page_sk", IntegerType(), True),
                StructField("ws_web_site_sk", IntegerType(), True),
                StructField("ws_ship_mode_sk", IntegerType(), True),
                StructField("ws_warehouse_sk", IntegerType(), True),
                StructField("ws_promo_sk", IntegerType(), True),
                StructField("ws_order_number", IntegerType(), True),
                StructField("ws_quantity", IntegerType(), True),
                StructField("ws_wholesale_cost", DoubleType(), False),
                StructField("ws_list_price", DoubleType(), False),
                StructField("ws_sales_price", DoubleType(), False),
                StructField("ws_ext_discount_amt", DoubleType(), False),
                StructField("ws_ext_sales_price", DoubleType(), False),
                StructField("ws_ext_wholesale_cost", DoubleType(), False),
                StructField("ws_ext_list_price", DoubleType(), False),
                StructField("ws_ext_tax", DoubleType(), False),
                StructField("ws_coupon_amt", DoubleType(), False),
                StructField("ws_ext_ship_cost", DoubleType(), False),
                StructField("ws_net_paid", DoubleType(), False),
                StructField("ws_net_paid_inc_tax", DoubleType(), False),
                StructField("ws_net_paid_inc_ship", DoubleType(), False),
                StructField("ws_net_paid_inc_ship_tax", DoubleType(), False),
                StructField("ws_net_profit", DoubleType(), False),
            ]
        ),
        "household_demographics": StructType(
            [
                StructField("hd_demo_sk", IntegerType(), True),
                StructField("hd_income_band_sk", IntegerType(), True),
                StructField("hd_buy_potential", StringType(), True),
                StructField("hd_dep_count", IntegerType(), True),
                StructField("hd_vehicle_count", IntegerType(), True),
            ]
        ),
        "inventory": StructType(
            [
                StructField("inv_date_sk", IntegerType(), True),
                StructField("inv_item_sk", IntegerType(), True),
                StructField("inv_warehouse_sk", IntegerType(), True),
                StructField("inv_quantity_on_hand", IntegerType(), True),
            ]
        ),
        "item_marketprices": StructType(
            [
                StructField("imp_sk", IntegerType(), True),
                StructField("imp_item_sk", IntegerType(), True),
                StructField("imp_competitor", StringType(), True),
                StructField("imp_competitor_price", DoubleType(), True),
                StructField("imp_start_date", IntegerType(), True),
                StructField("imp_end_date", IntegerType(), True),
            ]
        ),
        "promotion": StructType(
            [
                StructField("p_promo_sk", IntegerType(), True),
                StructField("p_promo_id", StringType(), True),
                StructField("p_start_date_sk", IntegerType(), True),
                StructField("p_end_date_sk", IntegerType(), True),
                StructField("p_item_sk", IntegerType(), True),
                StructField("p_cost", DoubleType(), False),
                StructField("p_response_target", StringType(), True),
                StructField("p_promo_name", StringType(), True),
                StructField("p_channel_dmail", StringType(), True),
                StructField("p_channel_email", StringType(), True),
                StructField("p_channel_catalog", StringType(), True),
                StructField("p_channel_tv", StringType(), True),
                StructField("p_channel_radio", StringType(), True),
                StructField("p_channel_press", StringType(), True),
                StructField("p_channel_event", StringType(), True),
                StructField("p_channel_demo", StringType(), True),
                StructField("p_channel_details", StringType(), True),
                StructField("p_purpose", StringType(), True),
                StructField("p_discount_active", StringType(), True),
            ]
        ),
        "store_returns": StructType(
            [
                StructField("sr_returned_date_sk", IntegerType(), True),
                StructField("sr_return_time_sk", IntegerType(), True),
                StructField("sr_item_sk", IntegerType(), True),
                StructField("sr_customer_sk", IntegerType(), True),
                StructField("sr_cdemo_sk", IntegerType(), True),
                StructField("sr_hdemo_sk", IntegerType(), True),
                StructField("sr_addr_sk", IntegerType(), True),
                StructField("sr_store_sk", IntegerType(), True),
                StructField("sr_reason_sk", IntegerType(), True),
                StructField("sr_ticket_number", IntegerType(), True),
                StructField("sr_return_quantity", IntegerType(), True),
                StructField("sr_return_amt", DoubleType(), False),
                StructField("sr_return_tax", DoubleType(), False),
                StructField("sr_return_amt_inc_tax", DoubleType(), False),
                StructField("sr_fee", DoubleType(), False),
                StructField("sr_return_ship_cost", DoubleType(), False),
                StructField("sr_refunded_cash", DoubleType(), False),
                StructField("sr_reversed_charge", DoubleType(), False),
                StructField("sr_store_credit", DoubleType(), False),
                StructField("sr_net_loss", DoubleType(), False),
            ]
        ),
        "time_dim": StructType(
            [
                StructField("t_time_sk", IntegerType(), True),
                StructField("t_time_id", StringType(), True),
                StructField("t_time", IntegerType(), True),
                StructField("t_hour", IntegerType(), True),
                StructField("t_minute", IntegerType(), True),
                StructField("t_second", IntegerType(), True),
                StructField("t_am_pm", StringType(), True),
                StructField("t_shift", StringType(), True),
                StructField("t_sub_shift", StringType(), True),
                StructField("t_meal_time", StringType(), True),
            ]
        ),
        "warehouse": StructType(
            [
                StructField("w_warehouse_sk", IntegerType(), True),
                StructField("w_warehouse_id", StringType(), True),
                StructField("w_warehouse_name", StringType(), True),
                StructField("w_warehouse_sq_ft", IntegerType(), True),
                StructField("w_street_number", StringType(), True),
                StructField("w_street_name", StringType(), True),
                StructField("w_street_type", StringType(), True),
                StructField("w_suite_number", StringType(), True),
                StructField("w_city", StringType(), True),
                StructField("w_county", StringType(), True),
                StructField("w_state", StringType(), True),
                StructField("w_zip", StringType(), True),
                StructField("w_country", StringType(), True),
                StructField("w_gmt_offset", DoubleType(), False),
            ]
        ),
        "web_page": StructType(
            [
                StructField("wp_web_page_sk", IntegerType(), True),
                StructField("wp_web_page_id", StringType(), True),
                StructField("wp_rec_start_date", StringType(), True),
                StructField("wp_rec_end_date", StringType(), True),
                StructField("wp_creation_date_sk", IntegerType(), True),
                StructField("wp_access_date_sk", IntegerType(), True),
                StructField("wp_autogen_flag", StringType(), True),
                StructField("wp_customer_sk", IntegerType(), True),
                StructField("wp_url", StringType(), True),
                StructField("wp_type", StringType(), True),
                StructField("wp_char_count", IntegerType(), True),
                StructField("wp_link_count", IntegerType(), True),
                StructField("wp_image_count", IntegerType(), True),
                StructField("wp_max_ad_count", IntegerType(), True),
            ]
        ),
        "web_returns": StructType(
            [
                StructField("wr_returned_date_sk", IntegerType(), True),
                StructField("wr_returned_time_sk", IntegerType(), True),
                StructField("wr_item_sk", IntegerType(), True),
                StructField("wr_refunded_customer_sk", IntegerType(), True),
                StructField("wr_refunded_cdemo_sk", IntegerType(), True),
                StructField("wr_refunded_hdemo_sk", IntegerType(), True),
                StructField("wr_refunded_addr_sk", IntegerType(), True),
                StructField("wr_returning_customer_sk", IntegerType(), True),
                StructField("wr_returning_cdemo_sk", IntegerType(), True),
                StructField("wr_returning_hdemo_sk", IntegerType(), True),
                StructField("wr_returning_addr_sk", IntegerType(), True),
                StructField("wr_web_page_sk", IntegerType(), True),
                StructField("wr_reason_sk", IntegerType(), True),
                StructField("wr_order_number", IntegerType(), True),
                StructField("wr_return_quantity", IntegerType(), True),
                StructField("wr_return_amt", DoubleType(), False),
                StructField("wr_return_tax", DoubleType(), False),
                StructField("wr_return_amt_inc_tax", DoubleType(), False),
                StructField("wr_fee", DoubleType(), False),
                StructField("wr_return_ship_cost", DoubleType(), False),
                StructField("wr_refunded_cash", DoubleType(), False),
                StructField("wr_reversed_charge", DoubleType(), False),
                StructField("wr_account_credit", DoubleType(), False),
                StructField("wr_net_loss", DoubleType(), False),
            ]
        ),
    }
    return (dataframe_dict, pyspark_schemas)


@bodo.jit(returns_maybe_distributed=False, cache=True)
def load_tpcxbb_data(dir_name):
    """Load the necessary TPCxBB dataframes given a root directory.
    We use bodo.jit so we can read easily from a directory."""
    # TODO: Update all the data frames selected so every query returns
    # a non-empty DataFrame
    store_sales_df = pd.read_parquet(dir_name + "/store_sales").head(1000)
    # we need the entire item df, so we don't get empty queries
    item_df = pd.read_parquet(dir_name + "/item")
    customer_df = pd.read_parquet(dir_name + "/customer").head(1000)
    customer_address_df = pd.read_parquet(dir_name + "/customer_address").head(1000)
    customer_demographics_df = pd.read_parquet(
        dir_name + "/customer_demographics"
    ).head(1000)
    date_dim_df = pd.read_parquet(dir_name + "/date_dim").head(1000)
    product_reviews_df = pd.read_parquet(dir_name + "/product_reviews").head(1000)
    store_df = pd.read_parquet(dir_name + "/store").head(1000)
    web_clickstreams_df = pd.read_parquet(dir_name + "/web_clickstreams").head(1000)
    web_sales_df = pd.read_parquet(dir_name + "/web_sales").head(1000)
    household_demographics_df = pd.read_parquet(
        dir_name + "/household_demographics"
    ).head(1000)
    inventory_df = pd.read_parquet(dir_name + "/inventory").head(1000)
    item_marketprices_df = pd.read_parquet(dir_name + "/item_marketprices").head(1000)
    promotion_df = pd.read_parquet(dir_name + "/promotion").head(1000)
    store_returns_df = pd.read_parquet(dir_name + "/store_returns").head(1000)
    time_dim_df = pd.read_parquet(dir_name + "/time_dim").head(1000)
    # the warehouse df and web_page_df currently only contains 3 and 4 values respectivley,
    # which causes issues with distributed tests.
    # For right now, since neither of these dataframes are actually being used in any of the
    # queries, I'm just concatinating it to itself 3 times to make it large enough so
    # we don't get distribution errors.
    warehouse_df = pd.read_parquet(dir_name + "/warehouse")
    warehouse_df = pd.concat(
        [warehouse_df, warehouse_df, warehouse_df], ignore_index=True
    )
    web_page_df = pd.read_parquet(dir_name + "/web_page")
    web_page_df = pd.concat([web_page_df, web_page_df, web_page_df], ignore_index=True)
    web_returns_df = pd.read_parquet(dir_name + "/web_returns").head(1000)
    return (
        store_sales_df,
        item_df,
        customer_df,
        customer_address_df,
        customer_demographics_df,
        date_dim_df,
        product_reviews_df,
        store_df,
        web_clickstreams_df,
        web_sales_df,
        household_demographics_df,
        inventory_df,
        item_marketprices_df,
        promotion_df,
        store_returns_df,
        time_dim_df,
        warehouse_df,
        web_page_df,
        web_returns_df,
    )


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    Marks the test to run as single_mode.
    Also Marks the tests with marker "part1", "part2", or "part3".
    """
    # BODO_TEST_PYTEST_MOD environment variable indicates that we only want
    # to run the tests from the given test file. In this case, we add the
    # "single_mod" mark to the tests belonging to that module. This envvar is
    # set in runtests.py, which also adds the "-m single_mod" to the pytest
    # command (thus ensuring that only those tests run)
    module_to_run = os.environ.get("BODO_TEST_PYTEST_MOD", None)
    if module_to_run is not None:
        for item in items:
            if module_to_run == item.module.__name__.split(".")[-1] + ".py":
                item.add_marker(pytest.mark.single_mod)
    for i, item in enumerate(items):
        # Divide the tests evenly so larger tests like TPCH
        # don't end up entirely in 1 group
        if i % 3 == 0:
            item.add_marker(pytest.mark.part1)
        if i % 3 == 1:
            item.add_marker(pytest.mark.part2)
        if i % 3 == 2:
            item.add_marker(pytest.mark.part3)

    # Check if we should try and mark groups for AWS Codebuild
    if "NUMBER_GROUPS_SPLIT" in os.environ:
        num_groups = int(os.environ["NUMBER_GROUPS_SPLIT"])

        # TODO: make a testTiming json file similar to how the bodo Repo does it
        # For now, we just use hash

        for item in items:
            # Gives filename + function name
            testname = item.module.__name__.split(".")[-1] + ".py" + "::" + item.name
            group_marker = group_from_hash(testname, num_groups)
            item.add_marker(getattr(pytest.mark, group_marker))


def group_from_hash(testname, num_groups):
    """
    Hash function to randomly distribute tests not found in the log.
    Keeps all s3 tests together in group 0.
    """
    if "test_s3.py" in testname:
        return "0"
    # TODO(Nick): Replace with a cheaper function.
    # Python's builtin hash fails on mpiexec -n 2 because
    # it has randomness. Instead we use a cryptographic hash,
    # but we don't need that level of support.
    hash_val = hashlib.sha1(testname.encode("utf-8")).hexdigest()
    # Hash val is a hex-string
    int_hash = int(hash_val, base=16) % num_groups
    return str(int_hash)


@pytest.fixture(scope="function")
def memory_leak_check():
    """
    A context manager fixture that makes sure there is no memory leak in the test.
    Equivalent to Numba's MemoryLeakMixin:
    https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/tests/support.py#L688
    """
    gc.collect()
    old = rtsys.get_allocation_stats()
    old_bodo = bodo.utils.allocation_tracking.get_allocation_stats()
    yield
    gc.collect()
    new = rtsys.get_allocation_stats()
    new_bodo = bodo.utils.allocation_tracking.get_allocation_stats()
    old_stats = [old, old_bodo]
    new_stats = [new, new_bodo]
    total_alloc = sum([m[0] for m in new_stats]) - sum([m[0] for m in old_stats])
    total_free = sum([m[1] for m in new_stats]) - sum([m[1] for m in old_stats])
    total_mi_alloc = sum([m[2] for m in new_stats]) - sum([m[2] for m in old_stats])
    total_mi_free = sum([m[3] for m in new_stats]) - sum([m[3] for m in old_stats])
    assert total_alloc == total_free
    assert total_mi_alloc == total_mi_free
