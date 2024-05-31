# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests reading data from a SnowflakeCatalog in a manner that will cause a runtime join filter 
to be pushed down to I/O.
"""


import io

import pandas as pd

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
