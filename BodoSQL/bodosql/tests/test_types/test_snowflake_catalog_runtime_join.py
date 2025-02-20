"""
Tests reading data from a SnowflakeCatalog in a manner that will cause a runtime join filter
to be pushed down to I/O.

Note, the following sequence of Snowflake commands were used to set up several of the tables
referenced in this file:

----------------------------------
--- Creating RTJF_TEST_TABLE_A ---
----------------------------------

CREATE OR REPLACE ICEBERG TABLE rtjf_test_table_a(
    id int,
    value decimal(10,5),
    hsh binary,
    time_of_day time(6),
    neutral_time timestamp_ntz(6),
    utc_time timestamp_ltz(6)
)
  CATALOG = 'SNOWFLAKE'
  EXTERNAL_VOLUME = 'exvol'
  BASE_LOCATION = 'rtjf_test_table_a'
;
GRANT OWNERSHIP ON rtjf_test_table_a TO SYSADMIN;

INSERT INTO rtjf_test_table_a (
SELECT
    SEQ8(),
    SEQ8() / 2500,
    TO_BINARY(BASE64_ENCODE(SEQ8()::VARCHAR), 'base64'),
    TIME_FROM_PARTS(SEQ1(), SEQ1(), SEQ1(), SEQ8()*1000),
    TIMESTAMP_FROM_PARTS(2024, 1, 1, SEQ8(), 0, 0),
    TIMESTAMP_LTZ_FROM_PARTS(2024, 1, SEQ8(), SEQ1(), 0, 0),
FROM TABLE(GENERATOR(ROWCOUNT=>10000))
);

----------------------------------
--- Creating RTJF_TEST_TABLE_B ---
----------------------------------

CREATE OR REPLACE ICEBERG TABLE rtjf_test_table_b(
    id int,
    value decimal(10,5),
    hsh binary,
    time_of_day time(6),
    neutral_time timestamp_ntz(6),
    utc_time timestamp_ltz(6)
)
  CATALOG = 'SNOWFLAKE'
  EXTERNAL_VOLUME = 'exvol'
  BASE_LOCATION = 'rtjf_test_table_b'
;
GRANT OWNERSHIP ON rtjf_test_table_b TO SYSADMIN;

INSERT INTO rtjf_test_table_b (
SELECT
    SEQ8()+4000,
    (SEQ8() * SEQ8()) / 2500,
    TO_BINARY(BASE64_ENCODE((SEQ8()-123)::VARCHAR), 'base64'),
    TIME_FROM_PARTS(SEQ1(), 0, SEQ1(), SEQ8()*1000),
    TIMESTAMP_FROM_PARTS(2024, 1, 1, SEQ8()+1234, 0, 0),
    TIMESTAMP_LTZ_FROM_PARTS(2024, 1, SEQ1(), SEQ8(), 0, 0),
FROM TABLE(GENERATOR(ROWCOUNT=>5000))
);

----------------------------------
--- Creating RTJF_TEST_TABLE_C ---
----------------------------------

CREATE OR REPLACE TABLE rtjf_test_table_c(
    id int,
    value decimal(10,5),
    hsh binary,
    time_of_day time(6),
    neutral_time timestamp_ntz(6),
    utc_time timestamp_ltz(6)
) AS SELECT
    SEQ8(),
    SEQ8() / 2500,
    TO_BINARY(BASE64_ENCODE(SEQ8()::VARCHAR), 'base64'),
    TIME_FROM_PARTS(SEQ1(), SEQ1(), SEQ1(), SEQ8()*1000),
    TIMESTAMP_FROM_PARTS(2024, 1, 1, SEQ8(), 0, 0),
    TIMESTAMP_LTZ_FROM_PARTS(2024, 1, SEQ8(), SEQ1(), 0, 0),
FROM TABLE(GENERATOR(ROWCOUNT=>10000))
;
GRANT OWNERSHIP ON rtjf_test_table_c TO SYSADMIN;



----------------------------------
--- Creating RTJF_TEST_TABLE_D ---
----------------------------------

CREATE OR REPLACE TABLE rtjf_test_table_d(
    id int,
    value decimal(10,5),
    hsh binary,
    time_of_day time(6),
    neutral_time timestamp_ntz(6),
    utc_time timestamp_ltz(6)
) AS SELECT
    SEQ8()+4000,
    (SEQ8() * SEQ8()) / 2500,
    TO_BINARY(BASE64_ENCODE((SEQ8()-123)::VARCHAR), 'base64'),
    TIME_FROM_PARTS(SEQ1(), 0, SEQ1(), SEQ8()*1000),
    TIMESTAMP_FROM_PARTS(2024, 1, 1, SEQ8()+1234, 0, 0),
    TIMESTAMP_LTZ_FROM_PARTS(2024, 1, SEQ1(), SEQ8(), 0, 0),
FROM TABLE(GENERATOR(ROWCOUNT=>5000))
;
GRANT OWNERSHIP ON rtjf_test_table_d TO SYSADMIN;
"""

import io

import pandas as pd
import pytest

import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    pytest_snowflake,
    temp_env_override,
)
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_snowflake_catalog,
    test_db_snowflake_catalog,
    test_db_snowflake_iceberg_catalog,
)

pytestmark = pytest_snowflake


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_simple_join(snowflake_sample_data_snowflake_catalog, memory_leak_check):
    """
    Tests the presence of runtime join filters in a Snowflake read on numeric keys.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    # Joins part and partsupp on the partkey with several filters
    # on the build side.
    # This should result in the nation table producing a runtime
    # join filter on partsupp to only read the rows with ps_partkey
    # between 5544 and 194830
    query = """
    SELECT SPLIT_PART(p.p_type, ' ', 3) AS METAL, COUNT(*) as NMATCH
    FROM tpch_sf1.part p, tpch_sf1.partsupp ps
    WHERE STARTSWITH(p_comment, 'alo')
    AND STARTSWITH(p_type, 'ECO')
    AND p.p_partkey = ps.ps_partkey
    GROUP BY 1
    """

    py_output = pd.DataFrame(
        {
            "METAL": ["COPPER", "TIN", "STEEL", "BRASS", "NICKEL"],
            "NMATCH": [56, 48, 48, 20, 24],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "PS_PARTKEY" FROM (SELECT "PS_PARTKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP") as TEMP) WHERE TRUE AND ($1 >= 5544) AND ($1 <= 194830)',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_larger_join(snowflake_sample_data_snowflake_catalog, memory_leak_check):
    """
    Variant of test_simple_join where the nature of the join will create a more complicated
    runtime join filter with a larger set of values removed by the min/max bounding.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    # Joins partsupp and lineitem to count the number of
    # customers for each combination of status/returnflag
    # This should result in 2 runtime join filters on
    # lineitem: l_partkey should be between 36 and 199963,
    # and l_suppkey should be between 1 and 9998.
    query = """
    SELECT l_linestatus, l_returnflag, COUNT(*) as total
    FROM tpch_sf1.lineitem, tpch_sf1.partsupp
    WHERE l_suppkey = ps_suppkey AND l_partkey = ps_partkey
    AND ps_availqty < 250
    AND ps_supplycost < 500.0
    GROUP BY 1, 2
    """

    py_output = pd.DataFrame(
        {
            "L_LINESTATUS": ["F", "F", "F", "O"],
            "L_RETURNFLAG": ["A", "N", "R", "N"],
            "TOTAL": [18572, 513, 18522, 37294],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "L_PARTKEY", "L_SUPPKEY", "L_RETURNFLAG", "L_LINESTATUS" FROM (SELECT "L_PARTKEY", "L_SUPPKEY", "L_RETURNFLAG", "L_LINESTATUS" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM") as TEMP) WHERE TRUE AND ($1 >= 36) AND ($1 <= 199963) AND ($2 >= 1) AND ($2 <= 9998)',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
@pytest.mark.skip("Disabled RTJF on non-dictionary-encoded string columns")
def test_string_key_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Variant of test_simple_join where the join is being done on string keys.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Joins the raw_dictionary and shakespeare tables to get the definitions
    # of all 6+ letter words spoken by Viola at least once in Act 1
    # of Twelfth Night. For tiebreakers, the first definition is chosen lexigraphically.
    # This should impose the following filters on raw_dictionary:
    # - WORD >= "Adorations"
    # - WORD <= "Rudeness"
    query = """
    WITH words AS
    (
        SELECT SPLIT_PART(playerline, ' ', 2) as word
        FROM shakespeare
        WHERE PLAY='Twelfth Night'
            AND PLAYER='VIOLA'
            AND ACTSCENELINE LIKE '1.%'
            AND LENGTH(word) > 6
    )
    SELECT words.word, raw_dictionary.definition
    FROM words, raw_dictionary
    WHERE words.word = raw_dictionary.word
    QUALIFY ROW_NUMBER() OVER (PARTITION BY words.word ORDER BY raw_dictionary.definition) = 1
    """

    py_output = pd.DataFrame(
        {
            "WORD": [
                "Assurance",
                "Brother",
                "Certain",
                "Country",
                "Nonpareil",
                "Prithee",
                "Radiant",
                "Reserve",
            ],
            "DEFINITION": [
                '"Any written or other legal evidence of the conveyance of property; a conveyance; a deed."',
                '"A male person who has the same father and mother with another person  or who has one of them only. In the latter case he is more definitely called a half brother or brother of the half blood."',
                '"A certain number or quantity."',
                '"A jury  as representing the citizens of a country."',
                '"A beautifully colored finch (Passerina ciris)  native of the Southern United States. The male has the head and neck deep blue rump and under parts bright red back and wings golden green and the tail bluish purple. Called also painted finch."',
                '"A corruption of pray thee; as  I prithee; generally used without I."',
                '"A straight line proceeding from a given point  or fixed pole about which it is conceived to revolve."',
                '"A body of troops in the rear of an army drawn up for battle  reserved to support the other lines as occasion may require; a force or body of troops kept for an exigency."',
            ],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "WORD", "DEFINITION" FROM (SELECT "WORD", "DEFINITION" FROM "TEST_DB"."PUBLIC"."RAW_DICTIONARY" WHERE "WORD" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= \'Adorations\') AND ($1 <= \'Rudeness\')',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_dict_key_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Variant of test_simple_join where the join is being done on string keys
    where the build and probe sides are dictionary encoded.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Joins the raw_dictionary and on itself in a ways that
    # will impose the following filters on the POS column
    # of raw_dictionary the second time it is read:
    # - pos >= '""'
    # - pos <= '"v."'
    query = """
    WITH lhs AS (
        SELECT word, pos
        FROM raw_dictionary
        WHERE STARTSWITH(word, 'Q')
        AND LENGTH(word) > 4
    ),
    rhs AS (
        SELECT word, pos
        FROM raw_dictionary
        WHERE STARTSWITH(word, 'R')
    )
    SELECT lhs.word as q_word, rhs.word as r_word, lhs.pos
    FROM lhs, rhs
    WHERE lhs.pos = rhs.pos
    QUALIFY ROW_NUMBER() OVER (PARTITION BY lhs.pos ORDER BY lhs.word, rhs.word) = 1
    """
    py_output = pd.DataFrame(
        {
            "Q_WORD": [
                "Quirl",
                "Qraspine",
                "Quack",
                "Quadragesimals",
                "Quackle",
                "Quack",
                "Quick",
                "Qua-bird",
                "Quadrennially",
                "Quadrate",
                "Quackled",
                "Quack grass",
                "Quackeries",
                "Quiesce",
                "Quits",
                "Querulous",
                "Queme",
            ],
            "R_WORD": [
                "Rain",
                "Rabbeting",
                "Rabble",
                "Radiata",
                "Reapproach",
                "Rabbinic",
                "Racy",
                "Ra",
                "Rabbinically",
                "Rabbate",
                "Rabbeted",
                "R",
                "Rabbies",
                "Reciproque",
                "Roint",
                "Rabate",
                "Rapturize",
            ],
            "POS": [
                '"n. & v."',
                '"p. pr. & vb. n."',
                '"v. i."',
                '"n. pl."',
                '"v. i. & t."',
                '"a."',
                '"superl."',
                '"n."',
                '"adv."',
                '"v. t."',
                '"imp. & p. p."',
                '""',
                '"pl. "',
                '"a. & n."',
                '"interj."',
                '"v."',
                '"v. t. & i."',
            ],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "WORD", "POS" FROM (SELECT "WORD", "POS" FROM "TEST_DB"."PUBLIC"."RAW_DICTIONARY" WHERE "POS" IS NOT NULL AND STARTSWITH("WORD", $$R$$)) as TEMP) WHERE TRUE AND ($2 >= \'""\') AND ($2 <= \'"v."\')',
        )
        # Also verify that the POS column was loaded in a dictionary encoded form
        check_logger_msg(
            stream,
            "pos: DictionaryArrayType(StringArrayType())",
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_date_key_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Variant of test_simple_join where the join is being done on date keys.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Joins the LINEITEM and ORDERS tables in a way that will
    # cause the following runtime join filters on lineitem:
    # - l_shipdate >= 1995-02-21
    # - l_shipdate <= 1998-08-02
    query = """
    SELECT YEAR(L_SHIPDATE) as Y, COUNT(*) as C
    FROM tpch_sf1.orders, tpch_sf1.lineitem
    WHERE O_ORDERPRIORITY='1-URGENT' AND O_ORDERSTATUS='O' AND L_SHIPDATE = O_ORDERDATE
    GROUP BY 1
    """
    py_output = pd.DataFrame(
        {
            "Y": [1995, 1996, 1997, 1998],
            "C": [70830467, 114831846, 113988017, 66500233],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "L_SHIPDATE" FROM (SELECT "L_SHIPDATE" FROM "TEST_DB"."TPCH_SF1"."LINEITEM" WHERE "L_SHIPDATE" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= DATE \'1995-02-21\') AND ($1 <= DATE \'1998-08-02\')',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_float_key_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Variant of test_simple_join where the join is being done on float keys.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Joins the SUPPLIER and CUSTOMER tables in a way that will
    # cause the following runtime join filters on customer:
    # - c_acctbal >= -935.13
    # - c_acctbal <= 9931.82
    query = """
    SELECT s_nationkey, COUNT(*) as n_match
    FROM tpch_sf1.supplier, tpch_sf1.customer
    WHERE supplier.s_acctbal = customer.c_acctbal
    AND s_nationkey BETWEEN 10 AND 15
    AND STARTSWITH(s_phone, '21')
    GROUP BY 1
    """
    py_output = pd.DataFrame(
        {
            "S_NATIONKEY": [11],
            "N_MATCH": [64],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "C_ACCTBAL" FROM (SELECT "C_ACCTBAL" FROM "TEST_DB"."TPCH_SF1"."CUSTOMER" WHERE "C_ACCTBAL" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= -935.13) AND ($1 <= 9931.82)',
        )


@temp_env_override({"AWS_REGION": "us-east-1", "BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
@pytest.mark.parametrize(
    "use_iceberg, table_1, table_2, log_message",
    [
        pytest.param(
            True,
            "rtjf_test_table_a",
            "rtjf_test_table_b",
            "Runtime join filter expression: ((ds.field('{NEUTRAL_TIME}') >= pa.scalar(1708513200000000000, pa.timestamp('ns'))) & (ds.field('{NEUTRAL_TIME}') <= pa.scalar(1726491600000000000, pa.timestamp('ns'))))",
            id="snowflake_iceberg",
            marks=pytest.mark.iceberg,
        ),
        pytest.param(
            False,
            "rtjf_test_table_c",
            "rtjf_test_table_d",
            'Runtime join filter query: SELECT * FROM (SELECT "NEUTRAL_TIME" FROM (SELECT "NEUTRAL_TIME" FROM "TEST_DB"."PUBLIC"."RTJF_TEST_TABLE_C" WHERE "NEUTRAL_TIME" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= TIMESTAMP_NTZ_FROM_PARTS(2024, 2, 21, 11, 0, 0, 0)) AND ($1 <= TIMESTAMP_NTZ_FROM_PARTS(2024, 9, 16, 13, 0, 0, 0))',
            id="snowflake_native",
        ),
    ],
)
def test_timestamp_ntz_key_join(
    use_iceberg,
    table_1,
    table_2,
    log_message,
    test_db_snowflake_catalog,
    test_db_snowflake_iceberg_catalog,
    memory_leak_check,
):
    """
    Variant of test_simple_join where the join is being done on timestamp_ntz keys.
    Tested on a Snowflake native table, and a Snowflake Iceberg table.
    """

    def impl(bc, query):
        return bc.sql(query)

    catalog = (
        test_db_snowflake_iceberg_catalog if use_iceberg else test_db_snowflake_catalog
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    # Joins the Snowflake Iceberg RTJF tables 1 & 2 in a way that will
    # cause the following runtime join filters on table 2:
    # - neutral_time >= 2024-02-21 11:00:00.000
    # - neutral_time <= 2024-09-16 13:00:00.000
    query = ""
    query += f"SELECT HOUR({table_1}.neutral_time) as hod, COUNT(*) as n_match\n"
    query += f"FROM {table_1}, {table_2}\n"
    query += f"WHERE {table_1}.neutral_time = {table_2}.neutral_time\n"
    query += f"AND HOUR({table_2}.time_of_day) BETWEEN 1 AND 3\n"
    query += "GROUP BY 1\n"
    py_output = pd.DataFrame(
        {
            "HOD": [3, 4, 5, 11, 12, 13, 19, 20, 21],
            "N_MATCH": [78] * 3 + [79] * 3 + [78] * 3,
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(stream, log_message)


@temp_env_override({"AWS_REGION": "us-east-1", "BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
@pytest.mark.parametrize(
    "use_iceberg, table_1, table_2, log_message",
    [
        pytest.param(
            True,
            "rtjf_test_table_a",
            "rtjf_test_table_b",
            "Runtime join filter expression: ((ds.field('{UTC_TIME}') >= pa.scalar(1704070800000000000, pa.timestamp('ns'))) & (ds.field('{UTC_TIME}') <= pa.scalar(1732561200000000000, pa.timestamp('ns'))))",
            id="snowflake_iceberg",
            marks=pytest.mark.skip(
                reason="[BSE-3493] Support min/max I/O runtime join filters on TIMESTAMP_LTZ for Iceberg"
            ),
        ),
        pytest.param(
            False,
            "rtjf_test_table_c",
            "rtjf_test_table_d",
            'Runtime join filter query: SELECT * FROM (SELECT "UTC_TIME" FROM (SELECT "UTC_TIME" FROM "TEST_DB"."PUBLIC"."RTJF_TEST_TABLE_C" WHERE "UTC_TIME" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= TIMESTAMP_LTZ_FROM_PARTS(2024, 1, 1, 1, 0, 0, 0)) AND ($1 <= TIMESTAMP_LTZ_FROM_PARTS(2024, 11, 25, 19, 0, 0, 0))',
            id="snowflake_native",
        ),
    ],
)
def test_timestamp_ltz_key_join(
    use_iceberg,
    table_1,
    table_2,
    log_message,
    test_db_snowflake_catalog,
    test_db_snowflake_iceberg_catalog,
    memory_leak_check,
):
    """
    Variant of test_simple_join where the join is being done on timestamp_ntz keys.
    Tested on a Snowflake native table, and a Snowflake Iceberg table.
    """

    def impl(bc, query):
        return bc.sql(query)

    catalog = (
        test_db_snowflake_iceberg_catalog if use_iceberg else test_db_snowflake_catalog
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    # Joins the Snowflake Iceberg RTJF tables 1 & 2 in a way that will
    # cause the following runtime join filters on table 2:
    # - utc_time >= 2024-01-01 01:00:00.000000
    # - utc_time <= 2024-11-25 19:00:00.000000
    query = ""
    query += f"SELECT HOUR({table_1}.utc_time) as hod, COUNT(*) as n_match\n"
    query += f"FROM {table_1}, {table_2}\n"
    query += f"WHERE {table_1}.utc_time = {table_2}.utc_time\n"
    query += f"AND HOUR({table_2}.time_of_day) BETWEEN 1 AND 3\n"
    query += "GROUP BY 1\n"
    py_output = pd.DataFrame(
        {
            "HOD": [1, 2, 3],
            "N_MATCH": [12] * 3,
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(stream, log_message)


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
@pytest.mark.skip("Disabled RTJF on non-dictionary-encoded string columns")
def test_larger_string_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Variant of test_string_key with a larger build table. This test can be
    used to verify that the parallel merging is working for strings, and
    also to check the relative performance impact of the min/max step on
    the join build.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # Joins the raw_dictionary and app_store_reviews tables to get the count &
    # average review score, along with the most frequent review definition, for each of the
    # top 20 apps, only including reviews where the first word of the content
    # is also located in the dictionary, and that are strictly only letters/spaces (without
    # any xs, ys, or zs). Uses a lexigraphic tiebreaker on the definitions.
    # This results in a build table with 74421 rows of 9106 distinct strings
    # This should impose the following filters on raw_dictionary:
    # - WORD >= "A"
    # - WORD <= "Wwwwwwwe"
    query = """
    SELECT app as app, count(*) as n_reviews, avg(score) as avg_review, mode(definition) as best_defn
    FROM (
        SELECT reviewId, app, score, word, definition
        FROM app_store_reviews reviews, raw_dictionary dict
        WHERE INITCAP(SPLIT_PART(reviews.content, ' ', 1)) = dict.word
        AND reviews.content RLIKE '[A-Za-z ]*'
        AND NOT content RLIKE '.*[XYZxyz].*'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY reviewId, word ORDER BY definition) = 1
    )
    GROUP BY app
    """

    py_output = pd.DataFrame(
        {
            "APP": [
                "Subway Surfers",
                "Candy Crush Saga",
                "LINE",
                "Dropbox",
                "Microsoft PowerPoint",
                "Facebook",
                "WhatsApp",
                "Flipboard",
                "Snapchat",
                "Instagram",
                "SHAREit",
                "Skype",
                "Netflix",
                "Facebook Lite",
                "Twitter",
                "Microsoft Word",
                "TikTok",
                "Facebook Messenger",
                "Viber",
                "Spotify",
            ],
            "N_REVIEWS": [
                2862,
                4030,
                2007,
                2361,
                3931,
                3357,
                3228,
                1711,
                2714,
                2926,
                4016,
                2285,
                2033,
                3460,
                2586,
                4004,
                2666,
                3187,
                2919,
                1867,
            ],
            "AVG_REVIEW": [
                4.554507,
                4.635484,
                3.924763,
                4.305379,
                4.289748,
                4.470360,
                4.201363,
                3.863238,
                4.058954,
                4.384484,
                4.587151,
                3.826258,
                4.039351,
                4.292486,
                4.225831,
                4.370879,
                4.249812,
                4.261374,
                4.294964,
                3.100161,
            ],
            "BEST_DEFN": [
                '"Adequate; sufficient; competent; sound; not fallacious; valid; in a commercial sense  to be depended on for the discharge of obligations incurred; having pecuniary ability; of unimpaired credit."'
            ]
            * 19
            + [
                '"Bad  evil or pernicious in the highest degree whether in a physical or moral sense. See Worse."'
            ],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'SELECT * FROM (SELECT "WORD", "DEFINITION" FROM (SELECT "WORD", "DEFINITION" FROM "TEST_DB"."PUBLIC"."RAW_DICTIONARY" WHERE "WORD" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= \'A\') AND ($1 <= \'Wwwwwwwe\')',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_multiple_filter_join(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """
    Variant of test_simple_join where the nature of the join will force multiple
    runtime join filters with different probe sides to filter the same I/O call.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    # The query counts the number of unique suppliers that are in
    # a specific country and sell at least one of the products
    # that meet certain size criteria.
    # This should result in 2 runtime join filters on
    # partsupp: ps_suppkey should be between 33 and 9990,
    # and ps_partkey should be between 449 and 199589.
    query = """
    SELECT COUNT(distinct s_suppkey) as n_suppliers
    FROM tpch_sf1.part, tpch_sf1.supplier, tpch_sf1.partsupp
    WHERE p_partkey = ps_partkey
    AND s_suppkey = ps_suppkey
    AND s_nationkey = 7
    AND p_size > 40
    AND p_container = 'JUMBO BAG'
    """

    py_output = pd.DataFrame({"N_SUPPLIERS": [120]})

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
            is_out_distributed=False,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" GROUP BY "PS_PARTKEY", "PS_SUPPKEY") as TEMP) WHERE TRUE AND ($1 >= 449) AND ($1 <= 199589) AND ($2 >= 33) AND ($2 <= 9990)',
        )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_numeric_low_ndv_join(
    snowflake_sample_data_snowflake_catalog, memory_leak_check
):
    """
    Tests a join with numeric keys on Snowflake tables that should generate a
    runtime join filter with IN instead of min/max.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    # Joins supplier and partsupp after filtering supplier so it only has
    # 2 s_suppkey values (1 and 4260). This causes a runtime join filter on
    # partsupp that checks when `ps_suppkey IN (1, 4260)`.
    query = """
    SELECT s.s_suppkey as SK, COUNT(*) as NM
    FROM tpch_sf1.supplier s, tpch_sf1.partsupp ps
    WHERE STARTSWITH(s.s_comment, 'each')
    AND s.s_suppkey = ps.ps_suppkey
    GROUP BY 1
    """

    py_output = pd.DataFrame(
        {
            "SK": [1, 4260],
            "NM": [80, 80],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "PS_SUPPKEY" FROM (SELECT "PS_SUPPKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP") as TEMP) WHERE TRUE AND ($1 IN (1, 4260))',
        )


@pytest.mark.parametrize(
    "in_df, answer, unique_limit, filter_text",
    [
        pytest.param(
            pd.DataFrame({"KEY": [120500 + i // 10 for i in range(300)]}),
            pd.DataFrame(
                {
                    "METAL": ["BRASS", "COPPER", "NICKEL", "STEEL", "TIN"],
                    "NMATCH": [40, 60, 70, 70, 60],
                }
            ),
            "20",
            'Runtime join filter query: SELECT * FROM (SELECT "P_PARTKEY", "P_TYPE" FROM (SELECT "P_PARTKEY", "P_TYPE" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART") as TEMP) WHERE TRUE AND ($1 >= 120500) AND ($1 <= 120529)',
            id="total_over_unique_threshold",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "KEY": pd.array(
                        [121000 + i // 100 + i % 10 for i in range(500)]
                        + [-1, -987654321]
                    )
                }
            ),
            pd.DataFrame(
                {
                    "METAL": ["BRASS", "COPPER", "NICKEL", "STEEL", "TIN"],
                    "NMATCH": [130, 120, 100, 100, 50],
                }
            ),
            "20",
            'Runtime join filter query: SELECT * FROM (SELECT "P_PARTKEY", "P_TYPE" FROM (SELECT "P_PARTKEY", "P_TYPE" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART") as TEMP) WHERE TRUE AND ($1 IN (-1, -987654321, 121000, 121001, 121002, 121003, 121004, 121005, 121006, 121007, 121008, 121009, 121010, 121011, 121012, 121013))',
            id="total_under_unique_threshold",
        ),
        pytest.param(
            pd.DataFrame({"KEY": [12700 + i for i in range(300)]}),
            pd.DataFrame(
                {
                    "METAL": ["BRASS", "COPPER", "NICKEL", "STEEL", "TIN"],
                    "NMATCH": [54, 76, 53, 49, 68],
                }
            ),
            "20",
            'Runtime join filter query: SELECT * FROM (SELECT "P_PARTKEY", "P_TYPE" FROM (SELECT "P_PARTKEY", "P_TYPE" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART") as TEMP) WHERE TRUE AND ($1 >= 12700) AND ($1 <= 12999)',
            id="all_unique",
        ),
        pytest.param(
            pd.DataFrame({"KEY": [121000 + i // 100 + i % 10 for i in range(500)]}),
            pd.DataFrame(
                {
                    "METAL": ["BRASS", "COPPER", "NICKEL", "STEEL", "TIN"],
                    "NMATCH": [130, 120, 100, 100, 50],
                }
            ),
            "10",
            'Runtime join filter query: SELECT * FROM (SELECT "P_PARTKEY", "P_TYPE" FROM (SELECT "P_PARTKEY", "P_TYPE" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART") as TEMP) WHERE TRUE AND ($1 >= 121000) AND ($1 <= 121013)',
            id="reduced_threshold",
        ),
    ],
)
def test_sf_unique_values_edge(
    snowflake_sample_data_snowflake_catalog,
    in_df,
    answer,
    unique_limit,
    filter_text,
    memory_leak_check,
):
    """
    Tests edge cases that may cause a runtime join filter to be a min/max filter versus
    a unique values filter, especially when run with multiple ranks.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)
    bc = bc.add_or_replace_view("TABLE1", in_df)

    # Joins in_df and part. This should result in a runtime join filter on
    # part that matches the filter_text argument.
    query = """
    SELECT SPLIT_PART(p_type, ' ', 3) AS METAL, COUNT(*) as NMATCH
    FROM tpch_sf1.part p, TABLE1 t
    WHERE p.p_partkey = t.key
    GROUP BY 1
    """

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": unique_limit}):
            check_func(
                impl,
                (bc, query),
                py_output=answer,
                only_1DVar=True,
                sort_output=True,
                reset_index=True,
            )
            # Verify that the correct bounds were added to the data requested
            # from Snowflake.
            check_logger_msg(
                stream,
                filter_text,
            )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "20"})
def test_sf_small_subset(
    snowflake_sample_data_snowflake_catalog,
    memory_leak_check,
):
    """
    Tests a subset of small query, with additional filters
    to make it practical to test w/o the large warehouse.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    query = """
    WITH
        golden_rules_exclusions_31701_SMALL as (
            select ss_sold_date_sk, ss_customer_sk, ss_ticket_number
            from SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL.STORE_SALES
            -- EXTRA FILTER TO REDUCE TOTAL ROWS READ FOR TESTING PURPOSES
            WHERE SS_CUSTOMER_SK BETWEEN 31618850 AND 31628850
        ),
        cte_date_tbl as (
            select
                d_year,
                d_date_sk,
                d_date,
                timeadd(YEAR, -1, d_date)       as year_ago_d_date
            from SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL.date_dim
            where d_week_seq = 5218    --first week of year 2000
        ),
        stage_time_period_selections_31701_SMALL as (
            select
                d_year,
                d_date_sk,
                'CURRENT YEAR' AS time_period_comparison_name,
                'CY' AS time_period_comparison_code,
                d_date
            from cte_date_tbl
            union all
            select
                d_tbl.d_year,
                d_dim.d_date_sk,
                'YEAR AGO' as time_period_comparison_name,
                'YA' as time_period_comparison_code,
                d_dim.d_date
            from cte_date_tbl d_tbl
            inner join SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL.date_dim d_dim
                on d_tbl.year_ago_d_date = d_dim.d_date
        )
    select
        time_period_selections.d_year,
        time_period_selections.TIME_PERIOD_COMPARISON_CODE,
        COUNT(DISTINCT(facts.SS_CUSTOMER_SK)) AS households,
        COUNT(DISTINCT(facts.SS_TICKET_NUMBER)) AS trips
    from golden_rules_exclusions_31701_SMALL facts
    inner join stage_time_period_selections_31701_SMALL time_period_selections
        on facts.ss_sold_date_sk = time_period_selections.d_date_sk
    GROUP BY 1, 2
    """

    answer = pd.DataFrame(
        {
            "D_YEAR": [1999, 1999, 2000, 2000],
            "TIME_PERIOD_COMPARISON_CODE": ["CY", "YA", "YA", "CY"],
            "HOUSEHOLDS": [1346, 1299, 918, 872],
            "TRIPS": [1566, 1587, 1030, 1021],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=answer,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "SS_SOLD_DATE_SK", "SS_CUSTOMER_SK", "SS_TICKET_NUMBER" FROM (SELECT "SS_SOLD_DATE_SK", "SS_CUSTOMER_SK", "SS_TICKET_NUMBER" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCDS_SF10TCL"."STORE_SALES" WHERE "SS_SOLD_DATE_SK" IS NOT NULL AND "SS_CUSTOMER_SK" >= 31618850 AND "SS_CUSTOMER_SK" <= 31628850) as TEMP) WHERE TRUE AND ($1 IN (2451176, 2451177, 2451178, 2451179, 2451180, 2451181, 2451182, 2451541, 2451542, 2451543, 2451544, 2451545, 2451546, 2451547))',
        )


def test_interval_join_rtjf(
    snowflake_sample_data_snowflake_catalog,
    memory_leak_check,
):
    """
    Adds a test for Runtime Join Filter support for joins created with
    interval syntax.
    """
    df = pd.DataFrame(
        {
            "KEY1": [5, 35, 6, 7, 15],
            "KEY2": [3, 4, 7, 9, 1],
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {"SMALL_TABLE": df}, catalog=snowflake_sample_data_snowflake_catalog
    )

    query = """SELECT L.L_ORDERKEY FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF100.LINEITEM L inner join SMALL_TABLE S on L.L_ORDERKEY > S.KEY1 AND L.L_ORDERKEY <= S.KEY2"""
    answer = pd.DataFrame(
        {
            "L_ORDERKEY": [7] * 7,
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=answer,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Snowflake.
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "L_ORDERKEY" FROM (SELECT "L_ORDERKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF100"."LINEITEM") as TEMP) WHERE TRUE AND ($1 <= 9) AND ($1 > 5)',
        )
