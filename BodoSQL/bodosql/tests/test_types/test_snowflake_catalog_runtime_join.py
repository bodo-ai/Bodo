# Copyright (C) 2024 Bodo Inc. All rights reserved.
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


def test_simple_join(snowflake_sample_data_snowflake_catalog, memory_leak_check):
    """
    Tests the presence of runtime join filters in a Snowflake read.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=snowflake_sample_data_snowflake_catalog)

    # Joins region, nation, and customer to count the number of
    # customers per region, but first filters the regions to only
    # include Europe or Asia.
    # This should result in the nation table producing a runtime
    # join filter on the customer key to only read the rows with
    # the relevant 10 nation keys, with nationkey values between
    # 6 and 23.
    query = """
    SELECT r_name, COUNT(*) as n_cust
    FROM tpch_sf1.region, tpch_sf1.nation, tpch_sf1.customer
    WHERE region.r_regionkey = nation.n_regionkey
    AND nation.n_nationkey = customer.c_nationkey
    AND region.r_name IN ('EUROPE', 'ASIA')
    GROUP BY r_name
    """

    py_output = pd.DataFrame(
        {
            "r_name": ["ASIA", "EUROPE"],
            "n_cust": [30183, 30197],
        }
    )
    py_output.columns = py_output.columns.str.upper()

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
            'Runtime join filter query: SELECT * FROM (SELECT "N_NATIONKEY", "N_REGIONKEY" FROM (SELECT "N_NATIONKEY", "N_REGIONKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION") as TEMP) WHERE TRUE AND ($2 >= 2) AND ($2 <= 3)',
        )
        check_logger_msg(
            stream,
            'Runtime join filter query: SELECT * FROM (SELECT "C_NATIONKEY" FROM (SELECT "C_NATIONKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER") as TEMP) WHERE TRUE AND ($1 >= 6) AND ($1 <= 23)',
        )


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


@temp_env_override({"AWS_REGION": "us-east-1"})
@pytest.mark.parametrize(
    "use_iceberg, table_1, table_2, log_message",
    [
        pytest.param(
            True,
            "rtjf_test_table_a",
            "rtjf_test_table_b",
            "Runtime join filter expression: ((ds.field('{NEUTRAL_TIME}') >= pa.scalar(1708513200000000000, pa.timestamp('ns'))) & (ds.field('{NEUTRAL_TIME}') <= pa.scalar(1726491600000000000, pa.timestamp('ns'))))",
            id="snowflake_iceberg",
        ),
        pytest.param(
            False,
            "rtjf_test_table_c",
            "rtjf_test_table_d",
            'Runtime join filter query: SELECT * FROM (SELECT "NEUTRAL_TIME" FROM (SELECT "NEUTRAL_TIME" FROM "TEST_DB"."PUBLIC"."RTJF_TEST_TABLE_C" WHERE "NEUTRAL_TIME" IS NOT NULL) as TEMP) WHERE TRUE AND ($1 >= TIMESTAMP_FROM_PARTS(2024, 2, 21, 11, 0, 0, 0)) AND ($1 <= TIMESTAMP_FROM_PARTS(2024, 9, 16, 13, 0, 0, 0))',
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


@temp_env_override({"AWS_REGION": "us-east-1"})
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
            'SELECT * FROM (SELECT * FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM (SELECT "PS_PARTKEY", "PS_SUPPKEY" FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP") as TEMP) WHERE TRUE AND ($1 >= 449) AND ($1 <= 199589)) WHERE TRUE AND ($2 >= 33) AND ($2 <= 9990)',
        )
