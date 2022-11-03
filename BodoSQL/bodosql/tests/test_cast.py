# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL cast queries on BodoSQL
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

import bodo


@pytest.fixture(
    params=[
        pytest.param("TINYINT", marks=pytest.mark.slow),
        pytest.param("SMALLINT", marks=pytest.mark.slow),
        pytest.param("INTEGER"),
        pytest.param("BIGINT", marks=pytest.mark.slow),
        pytest.param("FLOAT", marks=pytest.mark.slow),
        pytest.param("DOUBLE", marks=pytest.mark.slow),
        pytest.param("DECIMAL", marks=pytest.mark.slow),
    ]
)
def numeric_type_names(request):
    return request.param


@pytest.mark.slow
def test_cast_str_to_numeric(basic_df, spark_info, memory_leak_check):
    """Tests casting str literals to numeric datatypes"""
    query1 = "SELECT CAST('5' AS INT)"
    query2 = "SELECT CAST('-3' AS INT)"
    query3 = "SELECT CAST('-8454757700450211157' AS INT)"
    query4 = "SELECT CAST('5.2' AS FLOAT)"
    check_query(query1, basic_df, spark_info, check_names=False)
    check_query(query2, basic_df, spark_info, check_names=False)
    # TODO[BE-3834]: determine why dtype check is failing, query when run by hand returns correct result and dtype.
    check_query(
        query3,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({"EXPR$0": pd.Series([-1431655765])}),
    )
    check_query(query4, basic_df, spark_info, check_names=False)


@pytest.mark.skip("[BS-416] Calcite produces incorrect results")
def test_numeric_to_str(basic_df, spark_info, memory_leak_check):
    """test that you can cast numeric literals to strings"""
    query1 = "SELECT CAST(13 AS CHAR)"
    spark_query1 = "SELECT CAST(13 AS STRING)"
    query2 = "SELECT CAST(-103 AS CHAR)"
    spark_query2 = "SELECT CAST(-103 AS STRING)"
    query3 = "SELECT CAST(5.012 AS CHAR)"
    spark_query3 = "SELECT CAST(5.012 AS STRING)"
    check_query(
        query1,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query1,
        check_names=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query2,
        check_names=False,
    )
    check_query(
        query3,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query3,
        check_names=False,
    )


@pytest.mark.slow
def test_numeric_to_str_varchar(basic_df, spark_info, memory_leak_check):
    """test that you can cast numeric literals to strings"""
    query1 = "SELECT CAST(13 AS VARCHAR)"
    spark_query1 = "SELECT CAST(13 AS STRING)"
    query2 = "SELECT CAST(-103 AS VARCHAR)"
    spark_query2 = "SELECT CAST(-103 AS STRING)"
    query3 = "SELECT CAST(5.012 AS VARCHAR)"
    spark_query3 = "SELECT CAST(5.012 AS STRING)"
    check_query(
        query1,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query1,
        check_names=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query2,
        check_names=False,
    )
    check_query(
        query3,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query3,
        check_names=False,
    )


@pytest.mark.slow
def test_str_to_date(basic_df, spark_info, memory_leak_check):
    """Tests casting str literals to date types"""
    query1 = "SELECT CAST('2017-08-29' AS DATE)"
    query2 = "SELECT CAST('2019-02-13' AS DATE)"
    # Check dtype=False because spark outputs object type
    check_query(query1, basic_df, spark_info, check_names=False, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_like_to_like(basic_df, spark_info, memory_leak_check):
    """tests that you casting to the same type doesn't cause any weird issues"""
    query1 = "SELECT CAST(5 AS Int)"
    query2 = "SELECT CAST(-45 AS Int)"
    query3 = "SELECT CAST(3.123 AS Float)"
    query4 = f"SELECT CAST(X'{b'HELLO'.hex()}' AS VARBINARY)"
    check_query(query1, basic_df, spark_info, check_names=False)
    check_query(query2, basic_df, spark_info, check_names=False)
    check_query(query3, basic_df, spark_info, check_names=False)
    # TODO: [BE-957] Support Bytes.fromhex]
    # check_query(query4, basic_df, spark_info, check_names=False)


@pytest.mark.skip("[BS-414] casting strings/string literals to Binary not supported")
def test_str_to_binary(basic_df, spark_info, memory_leak_check):
    """Tests casting str literals to binary types"""
    query1 = "SELECT CAST('HELLO' AS BINARY)"
    query2 = "SELECT CAST('WORLD' AS VARBINARY)"
    # Check dtype=False because spark outputs object type
    check_query(query1, basic_df, spark_info, check_names=False, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_names=False, check_dtype=False)


# missing gaps are string and binary
@pytest.mark.skip("[BS-414] casting strings/string literals to Binary not supported")
def test_str_to_binary_cols(bodosql_string_types, spark_info, memory_leak_check):
    """Tests casting str columns to binary types"""
    query = "SELECT CAST(A AS BINARY), CAST(B as VARBINARY) from table1"
    # Check dtype=False because spark outputs object type
    check_query(
        query, bodosql_string_types, spark_info, check_names=False, check_dtype=False
    )


@pytest.mark.skip(
    "[BS-415] Calcite converts binary string to string version of binary value, not the string it encodes."
)
def test_binary_to_str(basic_df, spark_info, memory_leak_check):
    """Tests casting str literals to date types"""
    query1 = f"SELECT CAST(X'{b'HELLO'.hex()}' AS CHAR)"
    spark_query1 = f"SELECT CAST(X'{b'HELLO'.hex()}' AS STRING)"
    query2 = f"SELECT CAST(X'{b'WORLD'.hex()}' AS VARCHAR)"
    spark_query2 = f"SELECT CAST(X'{b'WORLD'.hex()}' AS STRING)"
    # Check dtype=False because spark outputs object type
    check_query(
        query1,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query1,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query2,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_scalar_to_numeric(
    bodosql_numeric_types, spark_info, numeric_type_names
):
    """Tests casting int scalars (from columns) to other numeric types"""
    query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS {numeric_type_names}) ELSE CAST (1 AS {numeric_type_names}) END FROM TABLE1"
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_nullable_scalar_to_numeric(
    bodosql_nullable_numeric_types, spark_info, numeric_type_names
):
    """Tests casting nullable int scalars (from columns) to numeric types"""
    query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS {numeric_type_names}) ELSE CAST (1 AS {numeric_type_names}) END FROM TABLE1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_string_scalar_to_numeric(
    bodosql_integers_string_types, spark_info, numeric_type_names
):
    """Tests casting string scalars (from columns) to numeric types"""
    query = f"SELECT CASE WHEN B = '43' THEN CAST(A AS {numeric_type_names}) ELSE CAST (1 AS {numeric_type_names}) END FROM TABLE1"
    check_query(
        query,
        bodosql_integers_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_scalar_to_str(bodosql_numeric_types, spark_info):
    """Tests casting int scalars (from columns) to str types"""
    # Use substring to avoid difference in Number of decimal places for
    query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS VARCHAR), 1, 3) ELSE 'OTHER' END FROM TABLE1"
    spark_query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS STRING), 1, 3) ELSE 'OTHER' END FROM TABLE1"
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_nullable_scalar_to_str(bodosql_nullable_numeric_types, spark_info):
    """Tests casting nullable int scalars (from columns) to str types"""
    query = (
        "SELECT CASE WHEN B > 5 THEN CAST(A AS VARCHAR) ELSE 'OTHER' END FROM TABLE1"
    )
    spark_query = (
        "SELECT CASE WHEN B > 5 THEN CAST(A AS STRING) ELSE 'OTHER' END FROM TABLE1"
    )
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_string_scalar_to_str(bodosql_string_types, spark_info):
    """Tests casting string scalars (from columns) to str types"""
    query = "SELECT CASE WHEN B <> 'how' THEN CAST(A AS VARCHAR) ELSE 'OTHER' END FROM TABLE1"
    spark_query = "SELECT CASE WHEN B <> 'how' THEN CAST(A AS STRING) ELSE 'OTHER' END FROM TABLE1"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_timestamp_scalar_to_str(bodosql_datetime_types, spark_info):
    """Tests casting datetime scalars (from columns) to string types"""
    query = "SELECT CASE WHEN B > TIMESTAMP '2010-01-01' THEN CAST(A AS VARCHAR) ELSE 'OTHER' END FROM TABLE1"
    spark_query = "SELECT CASE WHEN B > TIMESTAMP '2010-01-01' THEN CAST(A AS STRING) ELSE 'OTHER' END FROM TABLE1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_nullable_scalar_to_datetime(
    bodosql_nullable_numeric_types, spark_info
):
    """Tests casting numeric scalars (from columns) to str types"""
    query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS TIMESTAMP) ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_datetime_scalar_to_datetime(
    bodosql_datetime_types, spark_info, sql_datetime_typestrings
):
    """Tests casting datetime scalars (from columns) to datetime types"""
    query = f"SELECT CASE WHEN A > TIMESTAMP '1970-01-01' THEN CAST(B AS {sql_datetime_typestrings}) ELSE CAST (TIMESTAMP '2010-01-01' AS {sql_datetime_typestrings}) END FROM TABLE1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_timestamp_col_to_str(bodosql_datetime_types, spark_info):
    """Tests casting datetime columns to string types"""
    query = "SELECT CAST(A AS VARCHAR) FROM TABLE1"
    spark_query = "SELECT CAST(A AS STRING) FROM TABLE1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series([0, None, 42, 127, 255], dtype=pd.UInt8Dtype()), id="uint8"
        ),
        pytest.param(
            pd.Series([-1, None, 64, 127, -128], dtype=pd.Int8Dtype()), id="int8"
        ),
        pytest.param(
            pd.Series([0, None, 64, 4096, 65535], dtype=pd.UInt16Dtype()), id="uint16"
        ),
        pytest.param(
            pd.Series([100, None, -12345, -32768, 32767], dtype=pd.Int16Dtype()),
            id="int16",
        ),
        pytest.param(
            pd.Series([25, None, 625, 15625, 4294967295], dtype=pd.UInt32Dtype()),
            id="uint32",
        ),
        pytest.param(
            pd.Series(
                [1234567890, None, -7, -2147483648, 2147483647], dtype=pd.Int32Dtype()
            ),
            id="int32",
        ),
        pytest.param(
            pd.Series(
                [149162536496481100, None, 0, 2048, 8446744073709551615],
                dtype=pd.UInt64Dtype(),
            ),
            id="uint64",
        ),
        pytest.param(
            pd.Series(
                [
                    -2344264343534706688,
                    None,
                    1154048505100107776,
                    -9223372036854775808,
                    9223372036854775807,
                ],
                dtype=pd.Int64Dtype(),
            ),
            id="int64",
        ),
        pytest.param(
            pd.Series(
                [
                    0,
                    None,
                    4611686018427387903,
                    -102030405060.708090102030,
                    3.14159265358979323,
                ],
                dtype=np.float32,
            ),
            id="float32",
            marks=pytest.mark.skip(
                "we do not support displaying varying precision floats as strings"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    0,
                    None,
                    4611686018427387903,
                    -102030405060.708090102030,
                    3.14159265358979323,
                ],
                dtype=np.float64,
            ),
            id="float64",
            marks=pytest.mark.skip(
                "we do not support displaying varying precision floats as strings"
            ),
        ),
        pytest.param(pd.Series([True, None, True, False, True]), id="bool"),
        pytest.param(
            pd.Series(
                [
                    bytes.fromhex("a2b3"),
                    None,
                    bytes(0),
                    bytes.fromhex("deadbeef"),
                    bytes.fromhex("cafe"),
                ]
            ),
            id="binary",
        ),
        pytest.param(
            pd.Series(
                [pd.Timestamp(2018, 1, 1), None]
                + list(pd.date_range("2015", "2018", 3))
            ),
            id="datetime",
        ),
        pytest.param(
            pd.Series(
                [
                    pd.Timedelta(400, "days"),
                    None,
                    pd.Timedelta(12345, "minutes"),
                    pd.Timedelta(42, "hours"),
                    pd.Timedelta(14916253649, "nanoseconds"),
                ]
            ),
            id="timedelta",
            marks=pytest.mark.skip(
                "skip until we have parity with Snowflake intervals"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    bodo.Time(10, 0, 0),
                    None,
                    bodo.Time(13, 45, 0, 500),
                    bodo.Time(20, 0, 30, 100, 625),
                    bodo.Time(23, 59, 59, 999, 999, 999),
                ]
            ),
            id="time",
            marks=pytest.mark.skip(reason="Time type producing typing issues"),
        ),
    ]
)
def type_to_string(request):
    return request.param


def test_casting_to_string_cols(type_to_string, spark_info):
    """Tests multiple vector type cases for casting to a string"""
    query = "SELECT CAST(A AS VARCHAR) FROM table1"
    spark_query = "SELECT CAST(A AS STRING) FROM table1"
    ctx = {"table1": pd.DataFrame({"A": type_to_string})}
    if isinstance(type_to_string[0], bytes):
        check_query(
            query,
            ctx,
            spark_info,
            expected_output=pd.DataFrame(
                {"A": [a.hex() if not pd.isna(a) else None for a in type_to_string]}
            ),
            check_names=False,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_query(
            query,
            ctx,
            spark_info,
            equivalent_spark_query=spark_query,
            check_names=False,
            check_dtype=False,
        )


def test_casting_to_string_scalar(type_to_string, spark_info):
    """Tests multiple scalar (non literal) type cases for casting to a string"""
    query = (
        "SELECT CASE WHEN A IS NULL THEN NULL ELSE CAST(A AS VARCHAR) END  FROM table1"
    )
    spark_query = (
        "SELECT CASE WHEN A IS NULL THEN NULL ELSE CAST(A AS STRING) END FROM table1"
    )
    ctx = {"table1": pd.DataFrame({"A": type_to_string})}
    if isinstance(type_to_string[0], bytes):
        check_query(
            query,
            ctx,
            spark_info,
            expected_output=pd.DataFrame(
                {"A": [a.hex() if not pd.isna(a) else None for a in type_to_string]}
            ),
            check_names=False,
            check_dtype=False,
            sort_output=False,
        )
    else:
        check_query(
            query,
            ctx,
            spark_info,
            equivalent_spark_query=spark_query,
            check_names=False,
            check_dtype=False,
        )


# Using a fixture with hardcoded answers to prevent Gregorian / Julian error.
@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [12312435, 0, -2390482093, None, 31537000000000],
                    dtype=pd.Int64Dtype(),
                ),
                pd.Series(
                    [
                        12312435000000000,
                        0,
                        -2390482093000000000,
                        -9223372036854775808,
                        31537000000000000,
                    ]
                ).astype(np.dtype("datetime64[ns]")),
            ),
            id="int64",
        ),
        pytest.param(
            (
                pd.Series([-123234234.234, 0.0, 31537000000000.20, None, np.nan]),
                pd.Series(
                    [
                        -123234234233999997,
                        0,
                        31537000000000199,
                        -9223372036854775808,
                        -9223372036854775808,
                    ]
                ).astype(np.dtype("datetime64[ns]")),
            ),
            id="float64",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        "2020-01-01",
                        None,
                        "02-05-1980 12:20:20",
                        "08-20-1985",
                        "08-20-1985 02:02:02.0202",
                        "20-08-1985",
                    ]
                ),
                pd.Series(
                    [
                        1577836800000000000,
                        -9223372036854775808,
                        318601220000000000,
                        493344000000000000,
                        493351322020200000,
                        493344000000000000,
                    ]
                ).astype(np.dtype("datetime64[ns]")),
            ),
            id="strings",
        ),
    ],
)
def type_to_date_args(request):
    return request.param


def test_casting_to_timestamp_cols(type_to_date_args, spark_info):
    type_to_date, answer = type_to_date_args
    """Tests multiple vector type cases for casting to a timestamp"""
    query = "SELECT CAST(A AS TIMESTAMP) FROM table1"
    ctx = {"table1": pd.DataFrame({"A": type_to_date})}
    check_query(
        query,
        ctx,
        spark_info,
        expected_output=pd.DataFrame({"A": answer}),
        check_names=False,
        check_dtype=False,
    )


def test_casting_to_timestamp_scalar(type_to_date_args, spark_info):
    type_to_date, answer = type_to_date_args
    """Tests multiple scalar (non literal) type cases for casting to a timestamp"""
    query = (
        "SELECT CASE WHEN B = 0 THEN NULL ELSE CAST(A AS TIMESTAMP) END  FROM table1"
    )
    ctx = {"table1": pd.DataFrame({"A": type_to_date, "B": [1] * len(type_to_date)})}
    check_query(
        query,
        ctx,
        spark_info,
        expected_output=pd.DataFrame({"A": answer}),
        check_names=False,
        check_dtype=False,
    )
