# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests reading data from a SnowflakeCatalog in a manner that will cause a runtime join filter 
to be pushed down to I/O.
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
)
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_snowflake_catalog,
    test_db_snowflake_catalog,
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
