"""
Test correctness of SQL cast queries on BodoSQL
"""

import datetime

import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


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
    spark_query4 = "SELECT CAST('23.1204' AS DOUBLE)"
    if use_sf_cast_syntax:
        query1 = "SELECT '5'::INT"
        query2 = "SELECT '-3'::INT"
        query3 = "SELECT '5.2'::FLOAT"
        query4 = "SELECT '23.1204'::NUMBER(38, 4)"
    else:
        (query1, query2, query3, query4) = (
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
    check_query(
        query4,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query4,
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


@pytest.fixture(
    params=[
        pytest.param("VARCHAR", marks=pytest.mark.slow),
        "TEXT",
        "STRING",
        pytest.param("NVARCHAR", marks=pytest.mark.slow),
        pytest.param("NVARCHAR2", marks=pytest.mark.slow),
        pytest.param("CHAR VARYING", marks=pytest.mark.slow),
        "NCHAR VARYING",
    ]
)
def cast_str_typename(request):
    """
    The type name used for casting to string
    """
    return request.param


@pytest.mark.slow
def test_numeric_to_str(
    basic_df, use_sf_cast_syntax, cast_str_typename, spark_info, memory_leak_check
):
    """test that you can cast numeric literals to strings"""

    if use_sf_cast_syntax:
        query1 = f"SELECT 13::{cast_str_typename}"
        query2 = f"SELECT (-103)::{cast_str_typename}"
        query3 = f"SELECT 5.012::{cast_str_typename}"
    else:
        query1 = f"SELECT CAST(13 AS {cast_str_typename})"
        query2 = f"SELECT CAST(-103 AS {cast_str_typename})"
        query3 = f"SELECT CAST(5.012 AS {cast_str_typename})"

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
@pytest.mark.parametrize(
    "spark_query,sf_query",
    [
        ("SELECT CAST(5 AS Int)", "SELECT 5::Int"),
        ("SELECT CAST(-45 AS Int)", "SELECT (-45)::Int"),
        ("SELECT CAST(3.123 AS Float)", "SELECT 3.123::Float"),
        pytest.param(
            f"SELECT CAST(X'{b'HELLO'.hex()}' AS VARBINARY)",
            f"SELECT X'{b'HELLO'.hex()}'::VARBINARY",
            marks=pytest.mark.skip("[BE-957] Support Bytes.fromhex"),
        ),
    ],
)
def test_like_to_like(
    spark_query, sf_query, basic_df, use_sf_cast_syntax, spark_info, memory_leak_check
):
    """tests that you casting to the same type doesn't cause any weird issues"""

    query = sf_query if use_sf_cast_syntax else spark_query
    check_query(
        query,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


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
    bodosql_numeric_types,
    use_sf_cast_syntax,
    cast_str_typename,
    spark_info,
    memory_leak_check,
):
    """Tests casting int scalars (from columns) to str types"""
    # Use substring to avoid difference in Number of decimal places for

    spark_query = "SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS STRING), 1, 3) ELSE 'OTHER' END FROM TABLE1"

    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > 5 THEN SUBSTRING(A::{cast_str_typename}, 1, 3) ELSE 'OTHER' END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN B > 5 THEN SUBSTRING(CAST(A AS {cast_str_typename}), 1, 3) ELSE 'OTHER' END FROM TABLE1"

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
    bodosql_nullable_numeric_types,
    use_sf_cast_syntax,
    cast_str_typename,
    spark_info,
    memory_leak_check,
):
    """Tests casting nullable int scalars (from columns) to str types"""

    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > 5 THEN A::{cast_str_typename} ELSE 'OTHER' END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN B > 5 THEN CAST(A AS {cast_str_typename}) ELSE 'OTHER' END FROM TABLE1"
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
    bodosql_string_types,
    use_sf_cast_syntax,
    cast_str_typename,
    spark_info,
    memory_leak_check,
):
    """Tests casting string scalars (from columns) to str types"""
    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B <> 'how' THEN A::{cast_str_typename} ELSE 'OTHER' END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN B <> 'how' THEN CAST(A AS {cast_str_typename}) ELSE 'OTHER' END FROM TABLE1"
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
    bodosql_datetime_types,
    use_sf_cast_syntax,
    cast_str_typename,
    spark_info,
    memory_leak_check,
):
    """Tests casting datetime scalars (from columns) to string types"""
    if use_sf_cast_syntax:
        query = f"SELECT CASE WHEN B > TIMESTAMP '2010-01-01' THEN A::{cast_str_typename} ELSE 'OTHER' END FROM TABLE1"
    else:
        query = f"SELECT CASE WHEN B > TIMESTAMP '2010-01-01' THEN CAST(A AS {cast_str_typename}) ELSE 'OTHER' END FROM TABLE1"
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
        query = "SELECT CASE WHEN B > 5 THEN A::TIMESTAMP ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"
    else:
        query = "SELECT CASE WHEN B > 5 THEN CAST(A AS TIMESTAMP) ELSE TIMESTAMP '2010-01-01' END FROM TABLE1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=query.replace("TIMESTAMP", "TIMESTAMP_NS"),
        use_duckdb=True,
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
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=query.replace("TIMESTAMP", "TIMESTAMP_NS"),
        use_duckdb=True,
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

    expected_output = pd.DataFrame({"A": tz_aware_df["TABLE1"]["A"].astype(str)})
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
        {"A": tz_aware_df["TABLE1"]["A"].dt.tz_localize(None)}
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
        {"A": tz_aware_df["TABLE1"]["A"].dt.tz_localize(None).dt.normalize().dt.date}
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
    # BodoSQL converts this to a TZ-aware timestamp and so the timezones don't match. This can't
    # occur with SF, so we just update the test for now.
    df = tz_aware_df["TABLE1"]
    new_df = pd.DataFrame({"A": df["A"].dt.tz_convert("UTC")})
    ctx = {"TABLE1": new_df}
    expected_filter = (pd.Timestamp("2020-1-1", tz="UTC") <= new_df["A"]) & (
        new_df["A"] <= pd.Timestamp("2021-12-31", tz="UTC")
    )
    expected_output = tz_aware_df["TABLE1"][expected_filter]

    check_query(
        query,
        ctx,
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

    expected_output = pd.DataFrame(
        {"A": pd.Series([pd.Timestamp(2013, 5, 6)], dtype="datetime64[ns]")}
    )
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
        {
            "A": pd.Series([pd.Timestamp(2013, 5, 6)], dtype="datetime64[ns]"),
            "B": pd.Series(
                [pd.Timestamp(2013, 5, 6, 12, 34, 56)], dtype="datetime64[ns]"
            ),
        }
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
        "TABLE1": pd.DataFrame(
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
            "DATES": pd.Series(
                [pd.Timestamp(date) for date in ctx["TABLE1"]["DATES"]],
                dtype="datetime64[ns]",
            ),
            "STRINGS": pd.Series(
                [pd.Timestamp(string) for string in ctx["TABLE1"]["STRINGS"]],
                dtype="datetime64[ns]",
            ),
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.fixture(
    params=[
        pytest.param(
            (
                "VARCHAR",
                pd.Series(["", "kafae", None, "!@$$#", "1999-12-31"] * 4),
                pd.Series(["", "kafae", None, "!@$$#", "1999-12-31"] * 4),
            ),
            id="VARCHAR",
        ),
        pytest.param(
            (
                "DOUBLE",
                pd.Series(["634.234", "425", "asda", None, "-0.1251"] * 4),
                pd.Series([634.234, 425.0, None, None, -0.1251] * 4),
            ),
            id="DOUBLE",
        ),
        pytest.param(
            (
                "FLOAT",
                pd.Series(["-435.392", None, "-999", "1rfw43te", "0.0001"] * 4),
                pd.Series([-435.392, None, -999.0, None, 0.0001] * 4),
            ),
            id="FLOAT",
        ),
        pytest.param(
            (
                "NUMBER",
                pd.Series(["734", "-103", "105+106", "58.47", None] * 4),
                pd.Series([734, -103, None, 58, None] * 4),
            ),
            id="NUMBER",
        ),
        pytest.param(
            (
                "INTEGER",
                pd.Series(["0", "-49.36", "1999-12-31", None, "482"] * 4),
                pd.Series([0, -49, None, None, 482] * 4),
            ),
            id="INTEGER",
        ),
        pytest.param(
            (
                "DATE",
                pd.Series(
                    ["2014-02-25", "97/52/63", None, "1942-04-30", "2019-10-03"] * 4
                ),
                pd.Series(
                    [
                        datetime.date(2014, 2, 25),
                        None,
                        None,
                        datetime.date(1942, 4, 30),
                        datetime.date(2019, 10, 3),
                    ]
                    * 4
                ),
            ),
            id="DATE",
        ),
        pytest.param(
            (
                "TIME",
                pd.Series(
                    [
                        "03:24:55",
                        None,
                        "1942-04-30",
                        "20:39:47.876",
                        "19:57:28.082374912",
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        bodo.types.Time(3, 24, 55),
                        None,
                        None,
                        bodo.types.Time(20, 39, 47, 876),
                        bodo.types.Time(19, 57, 28, 82, 374, 912),
                    ]
                    * 4
                ),
            ),
            id="TIME",
        ),
        pytest.param(
            (
                "TIMESTAMP",
                pd.Series(
                    [
                        None,
                        "2014-02-25",
                        "20:39:47.876",
                        "1942-04-30 03:24:55",
                        "2019-10-03 19:57:28.082374912",
                    ]
                    * 4
                ),
                pd.Series(
                    [
                        None,
                        pd.Timestamp("2014-02-25"),
                        None,
                        pd.Timestamp("1942-04-30 03:24:55"),
                        pd.Timestamp("2019-10-03 19:57:28.082374912"),
                    ]
                    * 4,
                    dtype="datetime64[ns]",
                ),
            ),
            id="TIMESTAMP",
        ),
    ],
)
def try_cast_argument(request):
    """Inputs for test_try_cast"""
    return request.param


def test_try_cast(try_cast_argument, memory_leak_check):
    """Tests TRY_CAST behaves as expected"""
    type, data, answer = try_cast_argument
    query = f"SELECT TRY_CAST(A AS {type}) from table1"
    ctx = {"TABLE1": pd.DataFrame({"A": data})}
    expected_output = pd.DataFrame({"A": answer})
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "query",
    [
        "SELECT TO_VARIANT(X) from table1",
        "SELECT CAST(X as VARIANT) from table1",
        "SELECT X::VARIANT from table1",
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.Series(["a"] * 10 + [None]), id="string"),
        pytest.param(pd.Series([7] * 10 + [None]), id="int"),
        pytest.param(pd.Series([1.1] * 10 + [None]), id="float"),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 10, dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64()))
            ),
            id="map",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 10,
                dtype=pd.ArrowDtype(pa.struct([pa.field("A", pa.int32())])),
            ),
            id="struct",
        ),
    ],
)
def test_cast_to_variant(query, data):
    df = pd.DataFrame({"X": data})
    ctx = {"TABLE1": df}
    expected_output = pd.DataFrame({0: data})
    check_query(
        query,
        ctx,
        None,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )
