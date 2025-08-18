"""
Tests UDTF operations with a Snowflake catalog.

This files uses several UDTFs that are defined inside our partner Snowflake account.
In case any of these get deleted, here are commands that can be used to regenerate them,
including any dependent tables or views.

REGION_WRAPPER_FUNCTION:

create or replace function region_wrapper_function() RETURNS
TABLE (R_REGIONKEY NUMBER, R_NAME VARCHAR, R_COMMENT VARCHAR) as
$$
select * from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

ALIAS_TABLE_FUNCTION:

create or replace function alias_table_function() RETURNS TABLE("BAD NAME1" Varchar, "BAD,NAME,2" Number) as
$$
    select 'string', 1
$$

TIMES_TWO_TABLE:

create or replace function TIMES_TWO_TABLE(N NUMBER) RETURNS
TABLE (Value NUMBER, R_NAME VARCHAR) as
$$
select N * 2, REPEAT(R_NAME, 2) from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

TIMES_TWO_TABLE:

create or replace function TIMES_TWO_TABLE(S VARCHAR) RETURNS
TABLE (Value VARCHAR, R_NAME VARCHAR) as
$$
select REPEAT(S, 2), REPEAT(R_NAME, 2) from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

ADD_N_DEFAULT_ONE:

create or replace function ADD_N_DEFAULT_ONE(N NUMBER DEFAULT 1) RETURNS
TABLE (R_REGIONKEY NUMBER, R_NAME VARCHAR) as
$$
select R_REGIONKEY + N, R_NAME from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

SECURE_TABLE_FUNCTION:

create or replace secure function secure_table_function() RETURNS
TABLE (R_REGIONKEY NUMBER, R_NAME VARCHAR, R_COMMENT VARCHAR) as
$$
select * from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

NESTED_REGION_WRAPPER_FUNCTION:

create or replace function nested_region_wrapper_function() RETURNS
TABLE (R_REGIONKEY NUMBER, R_NAME VARCHAR, R_COMMENT VARCHAR) as
$$
select * from TABLE(region_wrapper_function())
$$

TIMES_N_TABLE_FUNCTION:

create or replace function TIMES_N_TABLE_FUNCTION(N NUMBER) RETURNS TABLE(RESULT NUMBER) as
$$
select A * N from COUNT_TABLE
$$

NESTED_LATERAL_FUNCTION:

create or replace function NESTED_LATERAL_FUNCTION(N NUMBER) RETURNS TABLE (A NUMBER, B NUMBER, C VARCHAR) as
$$
select * from count_table, lateral(TABLE(ADD_N_DEFAULT_ONE(N => (A + N))))
$$

ROW_EXPLODING_FUNCTION:

create or replace function ROW_EXPLODING_FUNCTION(V VARCHAR)
RETURNS TABLE (W VARCHAR) AS
$$
SELECT S.value as W FROM TABLE(SPLIT_TO_TABLE(V, ' ')) S
$$


In our azure account we have also defined the following UDTFs:

PYTHON_TABLE_FUNCTION:

create or replace function PYTHON_TABLE_FUNCTION(symbol varchar, quantity number, price number)
returns table (symbol varchar, total number(10,2))
language python
runtime_version=3.10
handler='StockSaleSum'
as $$
class StockSaleSum:
    def __init__(self):
        self._cost_total = 0
        self._symbol = ""

    def process(self, symbol, quantity, price):
      self._symbol = symbol
      cost = quantity * price
      self._cost_total += cost
      yield (symbol, cost)

    def end_partition(self):
      yield (self._symbol, self._cost_total)
$$;

"""

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import check_func, pytest_mark_one_rank, pytest_snowflake
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    azure_snowflake_catalog,
    test_db_snowflake_catalog,
)

pytestmark = pytest_snowflake


def test_udtf_inlining(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs can be inlined.

    REGION_WRAPPER_FUNCTION is manually defined inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(REGION_WRAPPER_FUNCTION())"
    py_output = pd.DataFrame(
        {
            "R_REGIONKEY": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
            "R_COMMENT": [
                "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to ",
                "hs use ironic, even requests. s",
                "ges. thinly even pinto beans ca",
                "ly final courts cajole furiously final excuse",
                "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl",
            ],
        }
    )
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=py_output)


@pytest_mark_one_rank
def test_unsupported_metadata_error_udtf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs whose metadata cannot be properly parsed give
    a specific error message that they cannot be supported.

    ALIAS_TABLE_FUNCTION is manually defined inside TEST_DB.PUBLIC to contain
    both ',' and ' ' inside the column names, which is not properly preserved
    in the metadata.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(alias_table_function())"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Error encountered while resolving TEST_DB.PUBLIC\\.ALIAS_TABLE_FUNCTION\\. Describe Function\'s "returns" metadata is improperly formatted or contains a type Bodo does not yet support\\. Snowflake does not properly escape columns containing special characters like space or comma in the metadata, which could be the source of failure\\.',
    ):
        impl(bc, query)


# TODO: fix memory leak and add back memory_leak_check
@pytest_mark_one_rank
def test_udtf_multiple_definitions(test_db_snowflake_catalog):
    """
    Test that Snowflake UDTFs with multiple definitions can be properly
    inlined and select the correct implementation.

    TIMES_TWO_TABLE is manually defined inside TEST_DB.PUBLIC twice, once on strings and
    once on numbers.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    def impl(bc, query):
        return bc.sql(query)

    local_table = pd.DataFrame(
        {
            # Number
            "A": np.arange(10),
            # String
            "B": [str(i) for i in range(10)],
            # Double which casts to string. Note all types prioritize
            # string over number except for number.
            "C": [np.float64(i) for i in np.arange(1, 11)],
            # Binary which cannot be supported
            "D": [str(i).encode("utf-8") for i in range(10)],
        }
    )
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": local_table},
        catalog=test_db_snowflake_catalog,
    )
    r_name_column = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
    repeated_r_name_column = [x + x for x in r_name_column]
    query1 = "select t1.A, t2.value, t2.r_name from local_table t1, lateral(table(times_two_table(A))) t2"
    left_table = pd.DataFrame({"A": local_table["A"], "VALUE": local_table["A"] * 2})
    right_table = pd.DataFrame({"R_NAME": repeated_r_name_column})
    py_output1 = pd.merge(left_table, right_table, how="cross")
    check_func(
        impl,
        (bc, query1),
        py_output=py_output1,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query2 = "select t1.B, t2.value, t2.r_name from local_table t1, lateral(table(times_two_table(B))) t2"
    left_table = pd.DataFrame(
        {"B": local_table["B"], "VALUE": local_table["B"] + local_table["B"]}
    )
    right_table = pd.DataFrame({"R_NAME": repeated_r_name_column})
    py_output2 = pd.merge(left_table, right_table, how="cross")
    check_func(
        impl,
        (bc, query2),
        py_output=py_output2,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query3 = "select t1.C, t2.value, t2.r_name from local_table t1, lateral(table(times_two_table(C))) t2"
    # Bodo always uses 6 decimal places right now.
    float_to_str = local_table["C"].astype(str) + "00000"
    left_table = pd.DataFrame(
        {"C": local_table["C"], "VALUE": float_to_str + float_to_str}
    )
    right_table = pd.DataFrame({"R_NAME": repeated_r_name_column})
    py_output3 = pd.merge(left_table, right_table, how="cross")
    check_func(
        impl,
        (bc, query3),
        py_output=py_output3,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query4 = "select t1.D, t2.value, t2.r_name from local_table t1, lateral(table(times_two_table(D))) t2"
    with pytest.raises(
        BodoError,
        match="No match found for function signature TIMES_TWO_TABLE\\(<BINARY>\\)",
    ):
        bodo.jit(impl)(bc, query4)


@pytest_mark_one_rank
def test_unsupported_udtf_defaults(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs with defaults gives a message they aren't supported.
    This is because we can't find the default values yet.

    ADD_N_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(ADD_N_DEFAULT_ONE())"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."ADD_N_DEFAULT_ONE" uses default arguments, which are not supported on Snowflake UDFs because the default values cannot be found in Snowflake metadata. Missing argument\\(s\\): N',
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_secure_udtf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake secure UDTFs, which we can't support, throws an appropriate
    error message.

    SECURE_TABLE_FUNCTION is manually defined inside TEST_DB.PUBLIC as secure.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(SECURE_TABLE_FUNCTION())"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."SECURE_TABLE_FUNCTION" contains an unsupported feature. Error message: "Bodo does not support secure UDTFs',
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_python_udtf(azure_snowflake_catalog, memory_leak_check):
    """
    Test that a Python UTDF, which we can't support, throws an appropriate
    error message. We must test with Azure because our partner account doesn't
    have permission to use Python.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(PYTHON_TABLE_FUNCTION('a', 1, 2))"
    bc = bodosql.BodoSQLContext(catalog=azure_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."PYTHON_TABLE_FUNCTION" contains an unsupported feature. Error message: "Unsupported source language. Bodo supports SQL UDTFs\\. Source language found was PYTHON\\."',
    ):
        impl(bc, query)


def test_udtf_with_provided_defaults(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs that allow defaults can be inlined if values are provided
    for the defaults.

    ADD_N_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(ADD_N_DEFAULT_ONE(4))"
    py_output = pd.DataFrame(
        {
            "R_REGIONKEY": np.array([4, 5, 6, 7, 8], dtype=np.int64),
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
        }
    )
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=py_output)


def test_udtf_with_named_args(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs with named args can be inlined.

    ADD_N_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(ADD_N_DEFAULT_ONE(N => 5))"
    py_output = pd.DataFrame(
        {
            "R_REGIONKEY": np.array([5, 6, 7, 8, 9], dtype=np.int64),
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
        }
    )
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=py_output)


def test_nested_udtf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDTFs that call another table function can be fully inlined.

    NESTED_REGION_WRAPPER_FUNCTION is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from TABLE(NESTED_REGION_WRAPPER_FUNCTION())"
    py_output = pd.DataFrame(
        {
            "R_REGIONKEY": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
            "R_COMMENT": [
                "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to ",
                "hs use ironic, even requests. s",
                "ges. thinly even pinto beans ca",
                "ly final courts cajole furiously final excuse",
                "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl",
            ],
        }
    )
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=py_output)


def test_udtf_comma_join(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that a UDTF can be inlined when used in a comma join.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = """
        select
            t1.r_regionkey as A,
            t1.r_name as B,
            t1.r_comment as C,
            t2.r_regionkey as D,
            t2.r_name as E,
            t2.r_comment as F
        from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION t1, TABLE(region_wrapper_function()) t2
    """
    left = pd.DataFrame(
        {
            "A": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "B": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
            "C": [
                "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to ",
                "hs use ironic, even requests. s",
                "ges. thinly even pinto beans ca",
                "ly final courts cajole furiously final excuse",
                "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl",
            ],
        }
    )
    right = left.rename(columns={"A": "D", "B": "E", "C": "F"})
    py_output = pd.merge(left, right, how="cross")
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_udtf_lateral_join(test_db_snowflake_catalog):
    """
    Tests that a UDTF can be inlined when used in a lateral join.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = """
        select
            t1.r_regionkey as A,
            t1.r_name as B,
            t1.r_comment as C,
            t2.r_regionkey as D,
            t2.r_name as E,
            t2.r_comment as F
        from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION t1, lateral TABLE(region_wrapper_function()) t2
    """
    left = pd.DataFrame(
        {
            "A": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "B": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
            "C": [
                "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to ",
                "hs use ironic, even requests. s",
                "ges. thinly even pinto beans ca",
                "ly final courts cajole furiously final excuse",
                "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl",
            ],
        }
    )
    right = left.rename(columns={"A": "D", "B": "E", "C": "F"})
    py_output = pd.merge(left, right, how="cross")
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_lateral_join_correlated(test_db_snowflake_catalog):
    """
    Tests that a UDTF used in a lateral join with correlated arguments
    can be fully inlined.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from LOCAL_TABLE, LATERAL TABLE(ADD_N_DEFAULT_ONE(N => A))"
    local_table = pd.DataFrame({"A": np.arange(10)})
    py_output = pd.DataFrame(
        {
            "A": [0] * 5
            + [1] * 5
            + [2] * 5
            + [3] * 5
            + [4] * 5
            + [5] * 5
            + [6] * 5
            + [7] * 5
            + [8] * 5
            + [9] * 5,
            "R_REGIONKEY": [0, 1, 2, 3, 4]
            + [1, 2, 3, 4, 5]
            + [2, 3, 4, 5, 6]
            + [3, 4, 5, 6, 7]
            + [4, 5, 6, 7, 8]
            + [5, 6, 7, 8, 9]
            + [6, 7, 8, 9, 10]
            + [7, 8, 9, 10, 11]
            + [8, 9, 10, 11, 12]
            + [9, 10, 11, 12, 13],
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"] * 10,
        }
    )
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": local_table,
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_nested_lateral_join_correlated(test_db_snowflake_catalog):
    """
    Tests that a UDTF used in a nested lateral join with correlated arguments
    can be fully inlined.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from LOCAL_TABLE, LATERAL TABLE(ADD_N_DEFAULT_ONE(N => A)), LATERAL TABLE(TIMES_N_TABLE_FUNCTION(N => B))"
    local_table = pd.DataFrame({"A": np.arange(10), "B": np.arange(9, -1, -1)})
    default_one_table = pd.DataFrame(
        {
            "R_REGIONKEY": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "R_NAME": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
        }
    )
    count_table = pd.DataFrame(
        {
            "RESULT": [1, 2, 3] * 5 + [4, 5, 6] * 3 + [7, 8, 9],
        }
    )
    join_output = pd.merge(local_table, default_one_table, how="cross").merge(
        count_table, how="cross"
    )
    py_output = pd.DataFrame(
        {
            "A": join_output["A"],
            "B": join_output["B"],
            "R_REGIONKEY": join_output["R_REGIONKEY"] + join_output["A"],
            "R_NAME": join_output["R_NAME"],
            "RESULT": join_output["RESULT"] * join_output["B"],
        }
    )
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": local_table,
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_nested_lateral_function(test_db_snowflake_catalog):
    """
    Tests that a UDTF that contains a lateral can be fully inlined.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from table(NESTED_LATERAL_FUNCTION(5))"

    count_table = pd.DataFrame(
        {
            "A": [1, 2, 3] * 5 + [4, 5, 6] * 3 + [7, 8, 9],
        }
    )
    default_one_table = pd.DataFrame(
        {
            "B": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "C": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
        }
    )
    join_output = pd.merge(count_table, default_one_table, how="cross")
    py_output = pd.DataFrame(
        {
            "A": join_output["A"],
            "B": join_output["B"] + join_output["A"] + 5,
            "C": join_output["C"],
        }
    )
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_lateral_with_lateral_udtf(test_db_snowflake_catalog):
    """
    Tests that a UDTF used in a lateral that also contains a lateral can be successfully
    inlined.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select t1.A as A, t2.A as B, t2.B as C, t2.C as D from LOCAL_TABLE t1, lateral (table(NESTED_LATERAL_FUNCTION(t1.A))) t2"
    local_table = pd.DataFrame({"A": np.arange(10)})
    default_one_table = pd.DataFrame(
        {
            "C": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "D": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
        }
    )
    count_table = pd.DataFrame(
        {
            "B": [1, 2, 3] * 5 + [4, 5, 6] * 3 + [7, 8, 9],
        }
    )
    join_output = pd.merge(local_table, default_one_table, how="cross").merge(
        count_table, how="cross"
    )
    py_output = pd.DataFrame(
        {
            "A": join_output["A"],
            "B": join_output["B"],
            "C": join_output["C"] + join_output["A"] + join_output["B"],
            "D": join_output["D"],
        }
    )
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": local_table,
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl, (bc, query), py_output=py_output, reset_index=True, sort_output=True
    )


# TODO: fix memory leak and add back memory_leak_check
def test_exploding_function(test_db_snowflake_catalog):
    """
    Tests that a UDTF containing a row exploding function can be inlined.

    Note: Snowflake produces a message "Unsupported subquery type cannot be evaluated"
    so we have more support than Snowflake.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = """
    SELECT W as word, COUNT(DISTINCT(T1.R_NAME)) as regions_using_word
    FROM TABLE(REGION_WRAPPER_FUNCTION()) T1,
    LATERAL (TABLE(ROW_EXPLODING_FUNCTION(T1.R_COMMENT))) T2
    WHERE NOT ENDSWITH(W, '.')
    GROUP BY word
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "WORD": [
                "",
                "according",
                "accounts",
                "are",
                "asymptotes",
                "beans",
                "blithely",
                "ca",
                "cajole",
                "carefully",
                "close",
                "courts",
                "even",
                "excuse",
                "final",
                "furiousl",
                "furiously",
                "haggle",
                "hs",
                "ironic,",
                "lar",
                "ly",
                "packages",
                "pinto",
                "regular",
                "s",
                "special",
                "thinly",
                "to",
                "uickly",
                "use",
                "waters",
            ],
            "REGIONS_USING_WORD": [
                1,
                1,
                2,
                1,
                1,
                1,
                2,
                1,
                2,
                1,
                1,
                1,
                2,
                1,
                3,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )
    check_func(
        impl,
        (bc, query),
        py_output=py_output,
        reset_index=True,
        sort_output=True,
        # Note: Since this test doesn't have any actual table inputs our infrastructure
        # doesn't successfully set the result to replicated. As a result, we only run this
        # test as distributed data.
        only_1DVar=True,
    )


def test_exploding_function_param(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that a UDTF containing a row exploding function and a named param
    can be inlined.
    """

    def impl(bc, query):
        return bc.sql(
            query, {"secret_phrase": "WE ATTACK AT DAWN WHILE THEY ATTACK AT DUSK"}
        )

    query = """
    SELECT T.W as word, COUNT(*) as n_uses
    FROM TABLE(ROW_EXPLODING_FUNCTION(@secret_phrase)) T
    WHERE LENGTH(T.w) > 2
    GROUP BY word
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "WORD": ["ATTACK", "DAWN", "WHILE", "THEY", "DUSK"],
            "N_USES": [2, 1, 1, 1, 1],
        }
    )
    check_func(
        impl,
        (bc, query),
        py_output=py_output,
        reset_index=True,
        sort_output=True,
        # Note: Since this test doesn't have any actual table inputs our infrastructure
        # doesn't successfully set the result to replicated. As a result, we only run this
        # test as distributed data.
        only_1DVar=True,
    )
