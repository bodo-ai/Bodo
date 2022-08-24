# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL cast queries on BodoSQL
"""
import pytest
from bodosql.tests.utils import bodo_version_older, check_query


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
    query3 = "SELECT CAST('5.2' AS FLOAT)"
    check_query(query1, basic_df, spark_info, check_names=False)
    check_query(query2, basic_df, spark_info, check_names=False)
    check_query(query3, basic_df, spark_info, check_names=False)


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
@pytest.mark.skipif(
    bodo_version_older(2021, 7, 3),
    reason="Requires next mini-release for engine changes to support casting strings to integers",
)
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


@pytest.mark.skipif(
    bodo_version_older(2021, 9, 2),
    reason="requires next mini release for support for columnwise str(np.dt64)",
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
