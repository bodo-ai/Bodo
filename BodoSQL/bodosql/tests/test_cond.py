# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL conditional functions on BodoSQL
"""
import copy

import numpy as np
import pandas as pd
import pytest
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
            "ColA": pd.Series(pd.array([None, None, None, None, 1, 2, 3, 4])),
            "ColB": pd.Series(pd.array([None, None, 5, 6, None, None, 7, 8])),
            "ColC": pd.Series(pd.array([None, 9, None, 10, None, 11, None, 12])),
            "ColD": pd.Series(pd.array([1, 1, 1, 1, 1, 1, 1, 2])),
        }
    )
    check_query(query, {"table1": df}, spark_info, check_dtype=False, check_names=False)


def test_coalesce_nested_expresions(spark_info):
    df = pd.DataFrame(
        {
            "ColA": pd.Series(pd.array([None, None, None, None, 1, 2, 3, 4])),
            "ColB": pd.Series(pd.array([None, None, 5, 6, None, None, 7, 8])),
            "ColC": pd.Series(pd.array([None, 9, None, 10, None, 11, None, 12])),
            "ColD": pd.Series(pd.array([1] * 8)),
        }
    )
    ctx = {"table1": df}

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
        "table1": pd.DataFrame(
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
        "table1": pd.DataFrame(
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
    """Tests NVL2 (coalesce/NVL but with 3 args)"""
    query = "SELECT NVL2(A+B, B+C, C+A) from table1"
    spark_query = "SELECT COALESCE(A+B, B+C, C+A) from table1"
    ctx = {
        "table1": pd.DataFrame(
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
        "table1": pd.DataFrame(
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
        ),
        pytest.param(
            (
                "SELECT REGR_VALX(Y, 0.0) FROM table1",
                pd.DataFrame(
                    {0: pd.Series([0.0, 0.0, 0.0, 0.0, None, None, None, 0.0])}
                ),
            ),
            id="regr_valx_vector_scalar",
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
        ),
    ],
)
def test_regr_valx_regr_valy(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "X": [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0],
                "Y": [1.0, 4.0, 9.0, 16.0, None, None, None, 64.0],
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
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        pd.Timestamp("2017-12-25"),
                        pd.Timestamp("2005-06-13"),
                        pd.Timestamp("1998-02-20"),
                        pd.Timestamp("2010-03-14"),
                        pd.Timestamp("2020-05-05"),
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


@pytest.mark.slow
def test_if_multitable(join_dataframes, spark_info, memory_leak_check):
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
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_ifnull_mixed(
    bodosql_nullable_numeric_types, spark_info, ifnull_equivalent_fn, memory_leak_check
):

    if bodosql_nullable_numeric_types["table1"].A.dtype.name == "UInt64":
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
def test_ifnull_null_float(
    zeros_df, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    """Checks ifnull function with values that generate np.nan"""
    # Note: 1 / 0 returns np.inf in BodoSQL but NULL in Spark, so
    # we use expected Output
    expected_output = pd.DataFrame(
        {"val": (zeros_df["table1"]["A"] / zeros_df["table1"]["B"]).replace(np.nan, -1)}
    )
    query = f"Select {ifnull_equivalent_fn}(A / B, -1) as val from table1"
    check_query(query, zeros_df, spark_info, expected_output=expected_output)


@pytest.mark.slow
def test_ifnull_multitable(
    join_dataframes, spark_info, ifnull_equivalent_fn, memory_leak_check
):
    """Checks ifnull function with columns from multiple tables"""
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
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
        check_dtype=check_dtype,
    )


def test_nullif_columns(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks nullif function with all column values"""

    # making a minor change, to ensure that we have an index where A == B to check correctness
    bodosql_nullable_numeric_types = copy.deepcopy(bodosql_nullable_numeric_types)
    bodosql_nullable_numeric_types["table1"]["A"][0] = bodosql_nullable_numeric_types[
        "table1"
    ]["B"][0]

    query = "Select NULLIF(A, B) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_float_nan=True,
    )


@pytest.mark.skip("Support setting a NULL scalar")
def test_nullif_scalar(basic_df, spark_info, memory_leak_check):
    """Checks nullif function with all scalar values"""
    query = "Select NULLIF(0, 0) from table1"
    check_query(query, basic_df, spark_info, check_names=False)


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
        convert_float_nan=True,
    )

    query = "Select NULLIF(1, A) from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_float_nan=True,
    )


@pytest.mark.slow
def test_nullif_case(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """Checks nullif function inside a case statement"""
    import copy

    # making a minor change, to ensure that we have an index where A == C to check correctness
    bodosql_nullable_numeric_types = copy.deepcopy(bodosql_nullable_numeric_types)
    bodosql_nullable_numeric_types["table1"]["A"][0] = bodosql_nullable_numeric_types[
        "table1"
    ]["C"][0]

    query = "Select CASE WHEN A > B THEN NULLIF(A, C) ELSE B END from table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_nullif_multitable(join_dataframes, spark_info, memory_leak_check):
    """Checks nullif function with columns from multiple tables"""
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
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
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nullifzero_cols(spark_info, memory_leak_check):
    """Tests NULLIFZERO (same as NULLIF(X, 0))"""
    query = "SELECT NULLIFZERO(A) from table1"
    spark_query = "SELECT NULLIF(A, 0) from table1"
    ctx = {
        "table1": pd.DataFrame(
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
        "table1": pd.DataFrame(
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
