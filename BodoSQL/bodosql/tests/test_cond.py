"""
Test correctness of SQL conditional functions on BodoSQL
"""

import copy
import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from numba.core.utils import PYVERSION

import bodo
from bodosql.tests.string_ops_common import bodosql_string_fn_testing_df  # noqa
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        "COALESCE",
        pytest.param("NVL", marks=pytest.mark.slow),
        pytest.param("IFNULL", marks=pytest.mark.slow),
    ]
)
def ifnull_equivalent_fn(request):
    return request.param


def test_coalesce_cols_basic(spark_info, basic_df, memory_leak_check):
    """tests the coalesce function on column values"""
    query = "select COALESCE(A, B, C) from table1"

    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skip(
    "General support for DECIMAL numbers in BodoSQL, and in particular for COALESCE."
)
def test_coalesce_128_int(spark_info, memory_leak_check):
    """tests the coalesce function with a NUMBER(38, 18) column and integer
    scalars."""
    query = "select COALESCE(A, 1) from table1"

    @bodo.jit
    def make_d128_df():
        A = pd.Series(
            [
                Decimal(0),
                None,
                Decimal(2),
                None,
                Decimal((2**33 - 1) / 13),
                None,
                Decimal((2**65 - 1) / 4),
                None,
                Decimal((2**97 - 1) / 100),
                None,
            ]
        )
        return pd.DataFrame({"A": A})

    ctx = {"TABLE1": make_d128_df()}

    check_query(query, ctx, spark_info, check_dtype=False, check_names=False)


def test_coalesce_128_float(memory_leak_check):
    """tests the coalesce function with a NUMBER(38, 18) column and float
    scalars."""
    query = "select COALESCE(A, -0.1) from table1"

    @bodo.jit
    def make_d128_df():
        A = pd.Series(
            [
                Decimal(0),
                None,
                Decimal(2),
                None,
                Decimal((2**33 - 1) / 13),
                None,
                Decimal((2**65 - 1) / 4),
                None,
                Decimal((2**97 - 1) / 100),
                None,
            ]
        )
        return pd.DataFrame({"A": A})

    ctx = {"TABLE1": make_d128_df()}
    expected_out = pd.DataFrame(
        {"A": [0.0, -0.1, 2.0, -0.1, 660764199.307692, -0.1, -0.25, -0.1, -0.01, -0.1]}
    )

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_out,
    )


def test_coalesce_timestamp_date(memory_leak_check):
    """Tests the coalesce function on a timestamp column and the current date"""
    query = "select COALESCE(A, current_date()) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        pd.Timestamp("2018-7-4"),
                        None,
                        pd.Timestamp("2022-12-31"),
                        None,
                        pd.Timestamp("2000-1-1"),
                    ],
                    dtype="datetime64[ns]",
                )
            }
        )
    }
    current_date = datetime.date.today()
    answer = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    pd.Timestamp("2018-7-4"),
                    pd.Timestamp(current_date),
                    pd.Timestamp("2022-12-31"),
                    pd.Timestamp(current_date),
                    pd.Timestamp("2000-1-1"),
                ],
                dtype="datetime64[ns]",
            )
        }
    )

    check_query(query, ctx, None, expected_output=answer, check_names=False)


@pytest.mark.parametrize("use_case", [True, False])
def test_coalesce_time(use_case, memory_leak_check):
    """Tests the coalesce function on time columns"""
    if use_case:
        query = "select CASE when A is NULL THEN B ELSE COALESCE(A, B) END from table1"
    else:
        query = "select COALESCE(A, B) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        bodo.types.Time(12, 0),
                        bodo.types.Time(1, 1, 3),
                        None,
                        None,
                        bodo.types.Time(12, 0, 31, 5, 92),
                    ]
                ),
                "B": pd.Series(
                    [
                        None,
                        bodo.types.Time(17, 12, 13, 92, 234, 193),
                        bodo.types.Time(2, 18, 37),
                        None,
                        bodo.types.Time(15, 26, 3, 44),
                    ]
                ),
            }
        )
    }
    answer = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    bodo.types.Time(12, 0),
                    bodo.types.Time(1, 1, 3),
                    bodo.types.Time(2, 18, 37),
                    None,
                    bodo.types.Time(12, 0, 31, 5, 92),
                ]
            )
        }
    )

    check_query(query, ctx, None, expected_output=answer, check_names=False)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT COALESCE(strings_null_1, strings_null_2, strings) from table1",
            id="nStrCol_nStrCol_StrCol",
        ),
        pytest.param(
            "SELECT COALESCE(strings_null_1, strings_null_2) from table1",
            id="nStrCol_nStrCol",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT COALESCE(strings_null_1, strings_null_2, '') from table1",
            id="nStrCol_nStrCol_Str",
        ),
        pytest.param(
            "SELECT COALESCE(strings_null_1, 'X') from table1",
            id="nStrCol_Str",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT COALESCE('A', 'B', 'C') from table1",
            id="Str_Str_Str",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT COALESCE(mixed_ints_null, mixed_ints_null, mixed_ints_null, mixed_ints_null) from table1",
            id="nIntCol_nIntCol_nIntCol_nIntCol",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT COALESCE(mixed_ints_null, 42) from table1",
            id="nIntCol_Int",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "SELECT COALESCE(0, 1, 2, 3) from table1",
            id="Int_Int_Int_Int",
            marks=(pytest.mark.slow,),
        ),
    ],
)
def test_coalesce_cols_adv(
    query, spark_info, bodosql_string_fn_testing_df, memory_leak_check
):
    """tests the coalesce function on more complex cases"""

    check_query(
        query,
        bodosql_string_fn_testing_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_coalesce_scalars(spark_info, memory_leak_check):
    """tests the coalesce function on scalar values"""
    query = "select CASE WHEN ColD = 1 THEN COALESCE(ColA, ColB, ColC) ELSE ColA * 10 END from table1"
    df = pd.DataFrame(
        {
            "COLA": pd.Series(pd.array([None, None, None, None, 1, 2, 3, 4])),
            "COLB": pd.Series(pd.array([None, None, 5, 6, None, None, 7, 8])),
            "COLC": pd.Series(pd.array([None, 9, None, 10, None, 11, None, 12])),
            "COLD": pd.Series(pd.array([1, 1, 1, 1, 1, 1, 1, 2])),
        }
    )
    check_query(query, {"TABLE1": df}, spark_info, check_dtype=False, check_names=False)


def test_coalesce_nested_expressions(spark_info, memory_leak_check):
    df = pd.DataFrame(
        {
            "COLA": pd.Series(pd.array([None, None, None, None, 1, 2, 3, 4])),
            "COLB": pd.Series(pd.array([None, None, 5, 6, None, None, 7, 8])),
            "COLC": pd.Series(pd.array([None, 9, None, 10, None, 11, None, 12])),
            "COLD": pd.Series(pd.array([1] * 8)),
        }
    )
    ctx = {"TABLE1": df}

    query = "Select CASE WHEN ColD = 1 THEN COALESCE(ColA + ColB, ColB + ColC, ColC * 2) ELSE -1 END from table1"

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip(
    "We're currently treating the behavior of coalesce on variable types as undefined behavior, see BS-435"
)
def test_coalesce_variable_type_cols(
    spark_info,
    bodosql_datetime_types,
    bodosql_string_types,
    basic_df,
    memory_leak_check,
):
    """tests the coalesce function on column values which have varying types
    Currently, Calcite allows for variable types in coalesce, so long as they converge to
    some common type. For our purposes this behavior is undefined.
    """
    new_ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": bodosql_datetime_types["table1"]["A"],
                "B": bodosql_string_types["table1"]["B"],
                "C": basic_df["table1"]["C"],
            }
        )
    }
    query = "select COALESCE(A, B, C) from table1"

    check_query(query, new_ctx, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skip(
    "We're currently treating the behavior of coalesce on variable types as undefined behavior, see BS-435"
)
def test_coalesce_variable_type_scalars(
    spark_info,
    bodosql_datetime_types,
    bodosql_string_types,
    basic_df,
    memory_leak_check,
):
    """tests the coalesce function on scalar values which have varying types
    Currently, Calcite allows for variable types in coalesce, so long as they converge to
    some common type. For our purposes this behavior is undefined.
    """
    new_ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": bodosql_datetime_types["table1"]["A"],
                "B": bodosql_string_types["table1"]["B"],
                "C": basic_df["table1"]["C"],
            }
        )
    }
    query = "select CASE WHEN COALESCE(A, B, C) = B THEN C ELSE COALESCE(A, B, C) END from table1"

    check_query(query, new_ctx, spark_info, check_dtype=False, check_names=False)


def test_nvl2(spark_info, memory_leak_check):
    """Tests NVL2 (equivalent to IF(A IS NOT NULL, B, C)"""
    query = "SELECT NVL2(A+B, B+C, C+A) from table1"
    spark_query = "SELECT IF((A+B) IS NOT NULL, B+C, C+A) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.array([None, 2345, 3456, 4567, None, 6789]),
                "B": pd.array([7891, None, 9123, 1234, 2345, None]),
                "C": pd.array([3456, 4567, None, 6789, None, 2345]),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_zeroifnull(spark_info, memory_leak_check):
    """Tests ZEROIFNULL (same as COALESCE(X, 0))"""
    query = "SELECT ZEROIFNULL(A) from table1"
    spark_query = "SELECT COALESCE(A, 0) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.array([None, 2, 3, 4, None, 6], dtype=pd.Int32Dtype()),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT REGR_VALX(Y, X) from table1",
                pd.DataFrame(
                    {0: pd.Series([1.0, None, 3.0, 4.0, None, None, None, 8.0])}
                ),
            ),
            id="regr_valx_all_vector",
        ),
        pytest.param(
            (
                "SELECT REGR_VALY(Y, X) from table1",
                pd.DataFrame(
                    {0: pd.Series([1.0, None, 9.0, 16.0, None, None, None, 64.0])}
                ),
            ),
            id="regr_valy_all_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGR_VALX(Y, 0.0) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0.0, 0.0, 0.0, 0.0, None, None, None, 0.0])}
                ),
            ),
            id="regr_valx_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT REGR_VALY(Y, 0.0) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([1.0, 4.0, 9.0, 16.0, None, None, None, 64.0])}
                ),
            ),
            id="regr_valy_vector_scalar",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN X IS NULL OR Y IS NULL OR X <> 4 THEN REGR_VALX(Y, X) ELSE -1.0 END FROM table1",
                pd.DataFrame(
                    {0: pd.Series([1.0, None, 3.0, -1.0, None, None, None, 8.0])}
                ),
            ),
            id="regr_valx_case",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN X IS NULL OR Y IS NULL OR X <> 4 THEN REGR_VALY(Y, X) ELSE -1.0 END FROM table1",
                pd.DataFrame(
                    {0: pd.Series([1.0, None, 9.0, -1.0, None, None, None, 64.0])}
                ),
            ),
            id="regr_valy_case",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_regr_valx_regr_valy(args, spark_info, memory_leak_check):
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "X": pd.array(
                    [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0], dtype="Float64"
                ),
                "Y": pd.array(
                    [1.0, 4.0, 9.0, 16.0, None, None, None, 64.0], dtype="Float64"
                ),
            }
        )
    }
    query, answer = args
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


def test_if_columns(basic_df, spark_info, memory_leak_check):
    """Checks if function with all column values"""
    query = "Select IF(B > C, A, C) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_if_scalar(basic_df, spark_info, memory_leak_check):
    """Checks if function with all scalar values"""
    query = "Select IFF(1 < 2, 7, 31)"
    spark_query = "Select IF(1 < 2, 7, 31)"
    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_if_dt(spark_info, memory_leak_check):
    """Checks if function with datetime values"""
    query = "Select IF(YEAR(A) < 2010, makedate(2010, 1), A) FROM table1"
    equivalent_spark_query = (
        "Select IF(YEAR(A) < 2010, make_date(2010, 1, 1), A) FROM table1"
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        datetime.date(2017, 12, 25),
                        datetime.date(2005, 6, 13),
                        datetime.date(1998, 2, 20),
                        datetime.date(2010, 3, 14),
                        datetime.date(2020, 5, 5),
                    ]
                )
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        equivalent_spark_query=equivalent_spark_query,
    )


@pytest.mark.slow
def test_if_mixed(basic_df, spark_info, memory_leak_check):
    """Checks if function with a mix of scalar and column values"""
    query = "Select IFF(B > C, A, -45) from table1"
    spark_query = "Select IF(B > C, A, -45) from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_if_case(basic_df, spark_info, memory_leak_check):
    """Checks if function inside a case statement"""
    query = "Select CASE WHEN A > B THEN IF(B > C, A, C) ELSE B END from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


def test_if_null_column(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks if function with all nullable columns"""
    query = "Select IF(B < C, A, C) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "func_name",
    [
        "IF",
        pytest.param("IFF", marks=pytest.mark.slow),
    ],
)
def test_if_time_column(bodosql_time_types, func_name, memory_leak_check):
    """Checks IF/IFF function with time columns"""
    query = f"Select {func_name}(B < C, B, C) from table1"
    expected_output = pd.DataFrame(
        {
            "OUTPUT": pd.Series(
                [
                    None,
                    bodo.types.Time(13, 37, 45),
                    bodo.types.Time(1, 47, 59, 290, 574),
                ]
                * 4
            )
        }
    )
    check_query(
        query,
        bodosql_time_types,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_if_multi_table(join_dataframes, spark_info, memory_leak_check):
    """Checks if function with columns from multiple tables"""
    query = "Select IF(table2.B > table1.B, table1.A, table2.A) from table1, table2"
    check_query(
        query, join_dataframes, spark_info, check_names=False, check_dtype=False
    )


def test_ifnull_columns(
    bodosql_nullable_numeric_types, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    """Checks ifnull function with all column values"""

    query = f"Select {ifnull_equivalent_fn}(A, B) from table1"
    spark_query = "Select IFNULL(A, B) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_ifnull_scalar(basic_df, spark_info, ifnull_equivalent_fn, memory_leak_check):
    """Checks ifnull function with all scalar values"""

    query = f"Select {ifnull_equivalent_fn}(-1, 45)"
    spark_query = "Select IFNULL(-1, 45)"
    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_ifnull_mixed(
    bodosql_nullable_numeric_types, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    if bodosql_nullable_numeric_types["TABLE1"].A.dtype.name == "UInt64":
        pytest.skip("Currently a bug in fillna for Uint64, see BE-1380")

    """Checks ifnull function with a mix of scalar and column values"""
    query = f"Select {ifnull_equivalent_fn}(A, 0) from table1"
    spark_query = "Select IFNULL(A, 0) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_ifnull_case(
    bodosql_nullable_numeric_types, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    """Checks ifnull function inside a case statement"""
    query = f"Select CASE WHEN A > B THEN {ifnull_equivalent_fn}(A, C) ELSE B END from table1"
    spark_query = "Select CASE WHEN A > B THEN IFNULL(A, C) ELSE B END from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_ifnull_multitable(
    join_dataframes, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    """Checks ifnull function with columns from multiple tables"""
    query = "Select IFNULL(table2.B, table1.B) from table1, table2"
    spark_query = (
        f"Select {ifnull_equivalent_fn}(table2.B, table1.B) from table1, table2"
    )
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_nullif_columns(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks nullif function with all column values"""

    # making a minor change, to ensure that we have an index where A == B to check correctness
    bodosql_nullable_numeric_types = copy.deepcopy(bodosql_nullable_numeric_types)
    bodosql_nullable_numeric_types["TABLE1"].loc[0, "A"] = (
        bodosql_nullable_numeric_types["TABLE1"]["B"][0]
    )

    query = "Select NULLIF(A, B) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_nullif_scalar(basic_df, spark_info, memory_leak_check):
    """Checks nullif function with all scalar values"""
    query = "Select NULLIF(0, 0) from table1"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=False)


@pytest.mark.slow
def test_nullif_mixed(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks nullif function with a mix of scalar and column values"""
    query = "Select NULLIF(A, 1) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )

    query = "Select NULLIF(1, A) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_nullif_time(memory_leak_check):
    """Checks nullif function with two time columns"""
    query = "Select NULLIF(A, B) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        bodo.types.Time(12, 0),
                        bodo.types.Time(8, 17, 43),
                        bodo.types.Time(2, 18, 37),
                        None,
                        bodo.types.Time(12, 0, 31, 5, 92),
                    ]
                ),
                "B": pd.Series(
                    [
                        None,
                        bodo.types.Time(17, 12, 13, 92, 234, 193),
                        bodo.types.Time(2, 18, 37),
                        bodo.types.Time(22, 56, 41),
                        bodo.types.Time(15, 26, 3, 44),
                    ]
                ),
            }
        )
    }
    answer = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    bodo.types.Time(12, 0),
                    bodo.types.Time(8, 17, 43),
                    None,
                    None,
                    bodo.types.Time(12, 0, 31, 5, 92),
                ]
            )
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=answer,
    )

    query = "Select case when A is NULL then NULL else NULLIF(A, B) end from table1"
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.slow
def test_nullif_case(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks nullif function inside a case statement"""
    import copy

    # making a minor change, to ensure that we have an index where A == C to check correctness
    bodosql_nullable_numeric_types = copy.deepcopy(bodosql_nullable_numeric_types)
    bodosql_nullable_numeric_types["TABLE1"].loc[0, "A"] = (
        bodosql_nullable_numeric_types["TABLE1"]["C"][0]
    )

    query = "Select CASE WHEN A > B THEN NULLIF(A, C) ELSE B END from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_nullif_multi_table(join_dataframes, spark_info, memory_leak_check):
    """Checks nullif function with columns from multiple tables"""
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["X"]
    else:
        convert_columns_bytearray = None
    query = "Select NULLIF(table2.B, table1.B) as X from table1, table2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nullifzero_cols(spark_info, memory_leak_check):
    """Tests NULLIFZERO (same as NULLIF(X, 0))"""
    query = "SELECT NULLIFZERO(A) from table1"
    spark_query = "SELECT NULLIF(A, 0) from table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.array(
                    [None, 2, 0, 3, 4, None, 6, 0, 0, 1], dtype=pd.Int32Dtype()
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT DECODE(A, B, C, D, E) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [1] + [None] * 14 + [6] + [None] * 8 + [5],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_no_default",
        ),
        pytest.param(
            (
                "SELECT DECODE(A, B, C, D, E, F) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [1]
                            + ([9, 10, 11, 12, None] * 3)[1:]
                            + [6]
                            + [10, 11, 12, None, 9, 10, 11, 12]
                            + [5],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_with_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT DECODE(A, 'A', 'a', 'a', 'a', 'E', 'e', 'e', 'e', 'I', 'i', 'i', 'i', 'O', 'o', 'o', 'o') FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            list("aaeaaeieaaeioieaaeio") + [None, "i", "e", "a", None]
                        )
                    }
                ),
            ),
            id="vector_scalar_no_default",
        ),
        pytest.param(
            (
                "SELECT DECODE(A, 'A', 'a', 'a', 'a', 'E', 'e', 'e', 'e', 'I', 'i', 'i', 'i', 'O', 'o', 'o', 'o', '_') FROM table1",
                pd.DataFrame({0: pd.Series(list("aaeaaeieaaeioieaaeio_iea_"))}),
            ),
            id="vector_scalar_with_default",
        ),
        pytest.param(
            (
                "SELECT DECODE(C, 1, 1, 2, 2, NULLIF(C, C), 3, 4, 4, 0) FROM table1",
                pd.DataFrame({0: pd.Series([1, 2, 3, 4, 0] * 5)}),
            ),
            id="vector_scalar_with_null_and_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT DECODE(10, 0, 'A') FROM table1",
                pd.DataFrame({0: pd.Series([None] * 25, dtype=pd.StringDtype())}),
            ),
            id="all_scalar_no_case_no_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT DECODE(10, 0, 'A', 'B') FROM table1",
                pd.DataFrame({0: ["B"] * 25}),
            ),
            id="all_scalar_no_case_with_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN B IS NULL THEN -1 ELSE DECODE(B, 'A', 1, 'E', 2, 3) END FROM table1",
                pd.DataFrame({0: [1, 2, 3, 3, -1] * 5}),
            ),
            id="all_scalar_with_case",
        ),
        pytest.param(
            (
                "SELECT DECODE(G, H, I, J, K) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [255, 255, 2**63 - 1, None, 255] * 5,
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_multiple_types",
        ),
    ],
)
def test_decode(args, spark_info, memory_leak_check):
    """Checks if function with all column values"""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": list("AaEaAeIeAaEiOiEaAeIoUiEa") + [None],
                "B": (list("AEIO") + [None]) * 5,
                "C": pd.Series([1, 2, None, 4, 5] * 5, dtype=pd.Int32Dtype()),
                "D": list("aeiou") * 5,
                "E": pd.Series([6, None, 7, None, 8] * 5, dtype=pd.Int32Dtype()),
                "F": pd.Series([9, 10, 11, 12, None] * 5, dtype=pd.Int32Dtype()),
                "G": pd.Series([0, 127, 128, 255, None] * 5, dtype=pd.UInt8Dtype()),
                "H": pd.Series([0, 127, -128, -1, None] * 5, dtype=pd.Int8Dtype()),
                "I": pd.Series([255] * 25, dtype=pd.UInt8Dtype()),
                "J": pd.Series(
                    [-128, -1, 128, -1, -(2**34)] * 5, dtype=pd.Int64Dtype()
                ),
                "K": pd.Series([2**63 - 1] * 25, dtype=pd.Int64Dtype()),
            }
        )
    }
    query, answer = args
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


def test_decode_time(bodosql_time_types, memory_leak_check):
    """Test DECODE with time columns"""
    query = "SELECT DECODE(A, TO_TIME('14:28:57'), 1, NULL, 2, 0) from table1"
    answer = pd.DataFrame({"output": pd.Series([0, 1, 0, 2] * 3)})
    check_query(
        query,
        bodosql_time_types,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


def test_nvl_ifnull_time_column_with_case(bodosql_time_types, memory_leak_check):
    """Test NVL and IFNULL with time columns and CASE statement"""
    query = (
        "SELECT CASE WHEN HOUR(A) < 12 THEN NVL(A, B) ELSE IFNULL(B, C) END FROM table1"
    )
    answer = pd.DataFrame(
        {
            "OUTPUT": pd.Series(
                [
                    bodo.types.Time(5, 13, 29),
                    bodo.types.Time(13, 37, 45),
                    bodo.types.Time(8, 2, 5, 0, 1),
                    bodo.types.Time(5, 13, 29),
                    bodo.types.Time(13, 37, 45),
                    bodo.types.Time(22, 7, 16),
                    bodo.types.Time(8, 2, 5, 0, 1),
                    bodo.types.Time(13, 37, 45),
                    bodo.types.Time(22, 7, 16),
                    bodo.types.Time(5, 13, 29),
                    bodo.types.Time(8, 2, 5, 0, 1),
                    bodo.types.Time(22, 7, 16),
                ]
            )
        }
    )
    check_query(
        query,
        bodosql_time_types,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "raw_query, expected_hashes",
    [
        pytest.param(
            "SELECT HASH(A) AS H FROM T",
            (10 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 14),
            id="one_col_A",
        ),
        pytest.param(
            "SELECT HASH(X) AS H FROM S", 5, id="one_col_B", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT HASH(*) AS H FROM T",
            (17 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 21),
            id="star",
        ),
        pytest.param(
            "SELECT HASH(S.*) AS H FROM S",
            (25 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 37),
            id="dot_star",
        ),
        pytest.param(
            "SELECT HASH(*) AS H FROM T INNER JOIN S ON T.A=S.A",
            (34 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 44),
            id="join_star",
        ),
        pytest.param(
            "SELECT HASH(T.*) AS H FROM T INNER JOIN S ON T.A=S.A",
            (15 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 19),
            id="join_dot_star_A",
        ),
        pytest.param(
            "SELECT HASH(S.*) AS H FROM T INNER JOIN S ON T.A=S.A",
            (18 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 29),
            id="join_dot_star_B",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT HASH(T.*, 16, *, S.*) AS H FROM T INNER JOIN S ON T.A=S.A",
            (34 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 44),
            id="join_star_multiple",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT HASH(ARRAY_CONSTRUCT_COMPACT(NULLIF(NULLIF(NULLIF(A, 'A'), 'T'), 'E'), NULLIF(NULLIF(NULLIF(A, 'A'), 'C'), 'E'))) AS H FROM S",
            # There are 16 distinct values but for some reason only 14 distinct hashes are produced
            (14 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 16),
            id="array",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_hash(raw_query, expected_hashes, memory_leak_check):
    """
    Tests HASH, HASH(*) and HASH(T.*) syntaxes to ensure that the correct
    number of distinct hash values are produced. Takes in a query that
    hashes the input tables into a column named H and creates a query
    that will count how many distinct hashes were produced.
    """
    query = f"SELECT COUNT(DISTINCT(H)) FROM ({raw_query})"
    ctx = {
        "S": pd.DataFrame(
            {
                "A": list("ALPHABETAGAMMADELTAEPSILONZETAETATHETAIOTAKAPPALAMBDAMU"),
                "X": pd.Series([0, 1, None, 3, 4] * 11, dtype=pd.Int32Dtype()),
            }
        ),
        "T": pd.DataFrame({"A": list("ALPHABETSOUPISDELICIOUS!"), "B": [0, 1] * 12}),
    }
    expected_output = pd.DataFrame({0: expected_hashes}, index=np.arange(1))
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        is_out_distributed=False,
    )
