# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL cast queries on BodoSQL
"""
import datetime

import pandas as pd
import pytest
from bodosql.tests.utils import check_query


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


@pytest.fixture(
    params=[
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ]
)
def use_sf_cast_syntax(request):
    return request.param


@pytest.mark.slow
def test_cast_str_to_numeric(
    basic_df, spark_info, use_sf_cast_syntax, memory_leak_check
):
    """Tests casting str literals to numeric datatypes"""

    spark_query1 = "SELECT CAST('5' AS INT)"
    spark_query2 = "SELECT CAST('-3' AS INT)"
    spark_query3 = "SELECT CAST('5.2' AS FLOAT)"
    if use_sf_cast_syntax:
        query1 = "SELECT '5'::INT"
        query2 = "SELECT '-3'::INT"
        query3 = "SELECT '5.2'::FLOAT"
    else:
        (query1, query2, query3) = (spark_query1, spark_query2, spark_query3)
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
    check_query(
        query3,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query3,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-416] Calcite produces incorrect results")
def test_numeric_to_str(basic_df, spark_info, use_sf_cast_syntax, memory_leak_check):
    """test that you can cast numeric literals to strings"""

    if use_sf_cast_syntax:
        query1 = "SELECT 13::CHAR"
        query2 = "SELECT -103::CHAR"
        query3 = "SELECT 5.012::CHAR"
    else:
        query1 = "SELECT CAST(13 AS CHAR)"
        query2 = "SELECT CAST(-103 AS CHAR)"
        query3 = "SELECT CAST(5.012 AS CHAR)"

    spark_query1 = "SELECT CAST(13 AS STRING)"
    spark_query2 = "SELECT CAST(-103 AS STRING)"
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
def test_numeric_to_str_varchar(
    basic_df, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """test that you can cast numeric literals to strings"""

    if use_sf_cast_syntax:
        query1 = "SELECT 13::VARCHAR"
        query2 = "SELECT (-103)::VARCHAR"
        query3 = "SELECT 5.012::VARCHAR"
    else:
        query1 = "SELECT CAST(13 AS VARCHAR)"
        query2 = "SELECT CAST(-103 AS VARCHAR)"
        query3 = "SELECT CAST(5.012 AS VARCHAR)"

    spark_query1 = "SELECT CAST(13 AS STRING)"
    spark_query2 = "SELECT CAST(-103 AS STRING)"
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
def test_str_to_date(basic_df, use_sf_cast_syntax, spark_info, memory_leak_check):
    """Tests casting str literals to date types"""
    spark_query1 = "SELECT CAST('2017-08-29' AS DATE)"
    spark_query2 = "SELECT CAST('2019-02-13' AS DATE)"

    if use_sf_cast_syntax:
        query1 = "SELECT '2017-08-29'::DATE"
        query2 = "SELECT '2019-02-13'::DATE"
    else:
        query1 = spark_query1
        query2 = spark_query2

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
def test_like_to_like(basic_df, use_sf_cast_syntax, spark_info, memory_leak_check):
    """tests that you casting to the same type doesn't cause any weird issues"""
    spark_query1 = "SELECT CAST(5 AS Int)"
    spark_query2 = "SELECT CAST(-45 AS Int)"
    spark_query3 = "SELECT CAST(3.123 AS Float)"
    spark_query4 = f"SELECT CAST(X'{b'HELLO'.hex()}' AS VARBINARY)"

    if use_sf_cast_syntax:
        query1 = "SELECT 5::Int"
        query2 = "SELECT (-45)::Int"
        query3 = "SELECT 3.123::Float"
        query4 = f"SELECT X'{b'HELLO'.hex()}'::VARBINARY"
    else:
        query1, query2, query3, query4 = (
            spark_query1,
            spark_query2,
            spark_query3,
            spark_query4,
        )

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
    check_query(
        query3,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query3,
        check_names=False,
        check_dtype=False,
    )
    # TODO: [BE-957] Support Bytes.fromhex]
    # check_query(query4, basic_df, spark_info, equivalent_spark_query=spark_query4, check_names=False)


@pytest.mark.skip("[BS-414] casting strings/string literals to Binary not supported")
def test_str_to_binary(basic_df, use_sf_cast_syntax, spark_info, memory_leak_check):
    """Tests casting str literals to binary types"""
    spark_query1 = "SELECT CAST('HELLO' AS BINARY)"
    spark_query2 = "SELECT CAST('WORLD' AS VARBINARY)"
    if use_sf_cast_syntax:
        query1 = "SELECT 'HELLO'::BINARY"
        query2 = "SELECT 'WORLD'::VARBINARY"
    else:
        query1, query2 = spark_query1, spark_query2

    # Check dtype=False because spark outputs object type
    check_query(
        query1,
        basic_df,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query1,
        check_dtype=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query2,
        check_dtype=False,
    )


# missing gaps are string and binary
@pytest.mark.skip("[BS-414] casting strings/string literals to Binary not supported")
def test_str_to_binary_cols(
    bodosql_string_types, spark_info, use_sf_cast_syntax, memory_leak_check
):
    """Tests casting str columns to binary types"""
    spark_query = "SELECT CAST(A AS BINARY), CAST(B as VARBINARY) from table1"
    if use_sf_cast_syntax:
        query = "SELECT A::BINARY, B::VARBINARY from table1"
    else:
        query = spark_query
    # Check dtype=False because spark outputs object type
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip(
    "[BS-415] Calcite converts binary string to string version of binary value, not the string it encodes."
)
def test_binary_to_str(basic_df, use_sf_cast_syntax, spark_info, memory_leak_check):
    """Tests casting str literals to date types"""
    spark_query1 = f"SELECT CAST(X'{b'HELLO'.hex()}' AS STRING)"
    spark_query2 = f"SELECT CAST(X'{b'WORLD'.hex()}' AS STRING)"

    if use_sf_cast_syntax:
        query1 = f"SELECT X'{b'HELLO'.hex()}'::CHAR"
        query2 = f"SELECT X'{b'WORLD'.hex()}'::VARCHAR"
    else:
        query1 = f"SELECT (X'{b'HELLO'.hex()}' AS CHAR)"
        query2 = f"SELECT CAST(X'{b'WORLD'.hex()}' AS VARCHAR)"

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
    bodosql_numeric_types,
    use_sf_cast_syntax,
    spark_info,
    numeric_type_names,
    memory_leak_check,
):
    """Tests casting int scalars (from columns) to other numeric types"""
    spark_query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS {numeric_type_names}) ELSE CAST(1 AS {numeric_type_names}) END FROM TABLE1"

    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > 5 THEN A::{numeric_type_names} ELSE 1::{numeric_type_names} END FROM TABLE1"
    else:
        query = spark_query

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_numeric_nullable_scalar_to_numeric(
    bodosql_nullable_numeric_types,
    use_sf_cast_syntax,
    spark_info,
    numeric_type_names,
    memory_leak_check,
):
    """Tests casting nullable int scalars (from columns) to numeric types"""
    spark_query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS {numeric_type_names}) ELSE CAST (1 AS {numeric_type_names}) END FROM TABLE1"

    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > 5 THEN A::{numeric_type_names} ELSE 1::{numeric_type_names} END FROM TABLE1"
    else:
        query = spark_query

    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_string_scalar_to_numeric(
    bodosql_integers_string_types,
    use_sf_cast_syntax,
    spark_info,
    numeric_type_names,
    memory_leak_check,
):
    """Tests casting string scalars (from columns) to numeric types"""
    spark_query = f"SELECT CASE WHEN B = '43' THEN CAST(A AS {numeric_type_names}) ELSE CAST (1 AS {numeric_type_names}) END FROM TABLE1"

    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B = '43' THEN A::{numeric_type_names} ELSE 1::{numeric_type_names} END FROM TABLE1"
    else:
        query = spark_query

    check_query(
        query,
        bodosql_integers_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_numeric_scalar_to_str(
    bodosql_numeric_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting int scalars (from columns) to str types"""
    # Use substring to avoid difference in Number of decimal places for

    spark_query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS STRING), 1, 3) ELSE 'OTHER' END FROM TABLE1"

    if use_sf_cast_syntax:
        query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(A::VARCHAR, 1, 3) ELSE 'OTHER' END FROM TABLE1"
    else:
        query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS VARCHAR), 1, 3) ELSE 'OTHER' END FROM TABLE1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_numeric_nullable_scalar_to_str(
    bodosql_nullable_numeric_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting nullable int scalars (from columns) to str types"""

    if use_sf_cast_syntax:
        query = "SELECT CASE WHEN B > 5 THEN A::VARCHAR ELSE 'OTHER' END FROM TABLE1"
    else:
        query = "SELECT CASE WHEN B > 5 THEN CAST(A AS VARCHAR) ELSE 'OTHER' END FROM TABLE1"
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
def test_string_scalar_to_str(
    bodosql_string_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting string scalars (from columns) to str types"""
    if use_sf_cast_syntax:
        query = (
            "SELECT CASE WHEN B <> 'how' THEN A::VARCHAR ELSE 'OTHER' END FROM TABLE1"
        )
    else:
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
def test_timestamp_scalar_to_str(
    bodosql_datetime_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting datetime scalars (from columns) to string types"""
    if use_sf_cast_syntax:
        query = "SELECT CASE WHEN B > TIMESTAMP '2010-01-01' THEN A::VARCHAR ELSE 'OTHER' END FROM TABLE1"
    else:
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
    bodosql_nullable_numeric_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting numeric scalars (from columns) to str types"""
    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > 5 THEN A::TIMESTAMP ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS TIMESTAMP) ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"

    spark_query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS TIMESTAMP) ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_datetime_scalar_to_datetime(
    bodosql_datetime_types,
    spark_info,
    sql_datetime_typestrings,
    use_sf_cast_syntax,
    memory_leak_check,
):
    """Tests casting datetime scalars (from columns) to datetime types"""
    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN A > TIMESTAMP '1970-01-01' THEN B::{sql_datetime_typestrings} ELSE (TIMESTAMP '2010-01-01')::{sql_datetime_typestrings} END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN A > TIMESTAMP '1970-01-01' THEN CAST(B AS {sql_datetime_typestrings}) ELSE CAST (TIMESTAMP '2010-01-01' AS {sql_datetime_typestrings}) END FROM TABLE1"
    spark_query = f"SELECT CASE WHEN A > TIMESTAMP '1970-01-01' THEN CAST(B AS {sql_datetime_typestrings}) ELSE CAST (TIMESTAMP '2010-01-01' AS {sql_datetime_typestrings}) END FROM TABLE1"
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_timestamp_col_to_str(
    bodosql_datetime_types, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """Tests casting datetime columns to string types"""
    if use_sf_cast_syntax:
        query = "SELECT A::VARCHAR FROM TABLE1"
    else:
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


@pytest.mark.tz_aware
def test_tz_aware_datetime_to_char_cast(
    tz_aware_df, use_sf_cast_syntax, memory_leak_check
):
    """simplest test for TO_CHAR on timezone aware data"""

    if use_sf_cast_syntax:
        query = "SELECT A::VARCHAR as A from table1"
    else:
        query = "SELECT CAST(A as VARCHAR) as A from table1"

    spark_query = "SELECT CAST(A as VARCHAR) as A from table1"

    expected_output = pd.DataFrame({"A": tz_aware_df["table1"]["A"].astype(str)})
    check_query(
        query,
        tz_aware_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.tz_aware
def test_tz_aware_datetime_to_timestamp_cast(
    tz_aware_df, use_sf_cast_syntax, memory_leak_check
):
    """Test Casting TZ-Aware data to Timestamp and dates"""
    if use_sf_cast_syntax:
        query1 = "SELECT A::Timestamp as A from table1"
    else:
        query1 = "SELECT CAST(A as Timestamp) as A from table1"
    spark_query1 = "SELECT CAST(A as Timestamp) as A from table1"
    expected_output1 = pd.DataFrame(
        {"A": tz_aware_df["table1"]["A"].dt.tz_localize(None)}
    )
    check_query(
        query1,
        tz_aware_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output1,
        equivalent_spark_query=spark_query1,
    )

    if use_sf_cast_syntax:
        query2 = "SELECT A::Date as A from table1"
    else:
        query2 = "SELECT CAST(A as Date) as A from table1"
    spark_query2 = "SELECT CAST(A as Date) as A from table1"

    expected_output2 = pd.DataFrame(
        {"A": tz_aware_df["table1"]["A"].dt.tz_localize(None).dt.normalize()}
    )
    check_query(
        query2,
        tz_aware_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output2,
        equivalent_spark_query=spark_query2,
    )


def test_implicit_cast_date_to_tz_aware(tz_aware_df, memory_leak_check):

    query = "SELECT * FROM table1 WHERE table1.A BETWEEN DATE '2020-1-1' AND DATE '2021-12-31'"
    expected_filter = (
        pd.Timestamp("2020-1-1", tz="US/Pacific") <= tz_aware_df["table1"]["A"]
    ) & (tz_aware_df["table1"]["A"] <= pd.Timestamp("2021-12-31", tz="US/Pacific"))
    expected_output = tz_aware_df["table1"][expected_filter]

    check_query(
        query,
        tz_aware_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_cast_date_scalar_to_timestamp(basic_df, use_sf_cast_syntax, memory_leak_check):
    """tests casting date scalar to timestamp"""

    if use_sf_cast_syntax:
        query = "SELECT DATE('2013-05-06')::TIMESTAMP"
    else:
        query = "SELECT CAST(DATE('2013-05-06') as TIMESTAMP)"

    expected_output = pd.DataFrame({"A": [pd.Timestamp(2013, 5, 6)]})
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        expected_output=expected_output,
    )


def test_cast_scalars_to_timestamp_ntz(basic_df, use_sf_cast_syntax, memory_leak_check):
    """tests casting date and string scalars to timestamp_ntz"""

    if use_sf_cast_syntax:
        query = "SELECT DATE('2013-05-06')::TIMESTAMP_NTZ, '2013-05-06 12:34:56'::TIMESTAMP_NTZ"
    else:
        query = "SELECT CAST(DATE('2013-05-06') as TIMESTAMP_NTZ), CAST('2013-05-06 12:34:56' as TIMESTAMP_NTZ)"

    expected_output = pd.DataFrame(
        {"A": [pd.Timestamp(2013, 5, 6)], "B": [pd.Timestamp(2013, 5, 6, 12, 34, 56)]}
    )
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        expected_output=expected_output,
    )


def test_cast_columns_to_timestamp_ntz(basic_df, use_sf_cast_syntax, memory_leak_check):
    """tests casting date and string columns to timestamp_ntz"""
    ctx = {
        "table1": pd.DataFrame(
            {
                "DATES": pd.Series(
                    [
                        datetime.date(2022, 1, 1),
                        datetime.date(2022, 3, 15),
                        None,
                        datetime.date(2019, 3, 15),
                        datetime.date(2010, 1, 11),
                    ]
                    * 3
                ),
                "STRINGS": pd.Series(
                    [
                        "2011-01-01",
                        "1971-02-02 16:43:25",
                        "2021-03-03",
                        None,
                        "2007-01-01 03:30:00",
                    ]
                    * 3
                ),
            }
        )
    }
    if use_sf_cast_syntax:
        query = "SELECT DATES::TIMESTAMP_NTZ, STRINGS::TIMESTAMP_NTZ from table1"
    else:
        query = "SELECT CAST(DATES as TIMESTAMP_NTZ), CAST(STRINGS as TIMESTAMP_NTZ) from table1"

    expected_output = pd.DataFrame(
        {
            "DATES": [pd.Timestamp(date) for date in ctx["table1"]["DATES"]],
            "STRINGS": [pd.Timestamp(string) for string in ctx["table1"]["STRINGS"]],
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=expected_output,
    )
