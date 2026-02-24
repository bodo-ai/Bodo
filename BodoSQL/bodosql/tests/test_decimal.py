"""
Tests for BodoSQL involving the decimal type
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import pytest_mark_one_rank, temp_config_override
from bodosql.tests.utils import check_query


@pytest.fixture
def decimal_data():
    """
    Returns a context with 3 tables, each containing
    a column "group_ids" column that can be used as join keys.
    """
    group_ids = [1, 2, 3, 4, 4, 3, 2, 2, 1, 1]
    benchmark_cashflow_group_ids = [1, 2, 3, 4, 4, 3, 2, 2, 1, 1]
    df_1 = pd.DataFrame(
        {
            "GROUP_IDS": group_ids,
            "BENCHMARK_CASHFLOW_GROUP_IDS": benchmark_cashflow_group_ids,
        }
    )
    group_ids_2 = [1, 2, 5, 6, 6, 5, 2, 2, 1, 1]
    amounts = [1, 2, 3, 4, 4, 3, 2, 2, 1, 1]
    amounts = pa.array(amounts, type=pa.decimal128(28, 4))
    cashflow_type_ids = [1, 2, 3, 4, 4, 3, 2, 2, 1, 1]
    df_2 = pd.DataFrame(
        {
            "GROUP_IDS": group_ids_2,
            "AMOUNTS": amounts,
            "CASHFLOW_TYPE_IDS": cashflow_type_ids,
        }
    )
    cashflow_group_ids = [1, 2, 6, 7, 7, 6, 2, 2, 1, 1]
    some_col = [1, 2, 3, 4, 4, 3, 2, 2, 1, 1]
    df_3 = pd.DataFrame(
        {
            "CASHFLOW_GROUP_IDS": cashflow_group_ids,
            "SOME_COL": some_col,
        }
    )
    return {"TABLE_1": df_1, "TABLE_2": df_2, "TABLE_3": df_3}


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "expr",
    [
        pytest.param("sum(D)", id="sum"),
        pytest.param("avg(D)", id="avg"),
        pytest.param("var_pop(D)", id="var_pop"),
        pytest.param("var_samp(D)", id="var_samp"),
        pytest.param("stddev_pop(D)", id="stddev_pop"),
        pytest.param("stddev_samp(D)", id="stddev_samp"),
        pytest.param("covar_samp(D, D-1)", id="covar_samp"),
        pytest.param("covar_pop(D, D-1)", id="covar_pop"),
        pytest.param("corr(D, D-1)", id="corr"),
    ],
)
def test_decimal_moment_functions_overflow(expr):
    """
    Tests that decimal moment-based aggregation functions (with groupby)
    correctly raise an error when their underlying data causes them to go
    out of bounds as a result of their subcomputations.
    """
    query = f"SELECT K, {expr} FROM TABLE1 GROUP BY K"
    keys = ["A"] * 25
    decimals = pd.array(
        [Decimal("123.45")] * 25, dtype=pd.ArrowDtype(pa.decimal128(38, 35))
    )
    ctx = {"TABLE1": pd.DataFrame({"K": keys, "D": decimals})}

    with temp_config_override("bodo_use_decimal", True):
        with pytest.raises(
            Exception,
            match="(Overflow detected in groupby sum of Decimal data)|(Number out of representable range)",
        ):
            check_query(
                query,
                ctx,
                None,
                expected_output=pd.DataFrame({"K": [0], "RES": [0]}),
                check_dtype=False,
            )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT K, SUM(D) :: VARCHAR as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        "493.80",
                        "1111111111111.10",
                        None,
                        "-3.14",
                        "1989999999999.97",
                        "2767777777777.09",
                        "3434444444443.75",
                        "479074.00",
                        "-645.12",
                        "0.00",
                    ],
                }
            ),
            id="sum",
        ),
        pytest.param(
            "SELECT K, AVG(D) :: VARCHAR as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        "123.45000000",
                        "277777777777.77500000",
                        None,
                        "-3.14000000",
                        "994999999999.98500000",
                        "922592592592.36333333",
                        "858611111110.93750000",
                        "108.51053228",
                        "-645.12000000",
                        "0.00000000",
                    ],
                }
            ),
            id="avg",
        ),
        pytest.param(
            "SELECT K, VAR_POP(D) :: VARCHAR as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        "0.0000000000",
                        "15432098765431790123456.7901250000",
                        None,
                        "0.0000000000",
                        "24999999999950000000.0000250000",
                        "10502331961653243347050.8462888889",
                        "20157638888914043055555.6337187500",
                        "48214.5026637031",
                        "0.0000000000",
                        "320.4062500000",
                    ],
                }
            ),
            id="var_pop",
        ),
        pytest.param(
            # NOTE: trimming the last decimal place due to instability in the
            # last digit when division is capped by scale-level truncation.
            "SELECT K, REGEXP_REPLACE(VAR_SAMP(D) :: VARCHAR, '.$', '') as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        "0.000000000",
                        "20576131687242386831275.720166666",
                        None,
                        None,
                        "49999999999900000000.000050000",
                        "15753497942479865020576.269433333",
                        "26876851851885390740740.844958333",
                        "48225.425749943",
                        None,
                        "427.208333333",
                    ],
                }
            ),
            id="var_samp",
        ),
        pytest.param(
            "SELECT K, STDDEV_POP(D) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        0.0,
                        124225998749.987,
                        None,
                        0.0,
                        4999999999.995,
                        102480885835.619,
                        141977599954.761,
                        219.578010428,
                        0.0,
                        17.899895251,
                    ],
                }
            ),
            id="stddev_pop",
        ),
        pytest.param(
            "SELECT K, STDDEV_SAMP(D) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "RES": [
                        0.0,
                        143443827637.31,
                        None,
                        None,
                        7071067811.8584,
                        125512939342.842,
                        163941611105.556,
                        219.602881925,
                        None,
                        20.669018683,
                    ],
                }
            ),
            id="stddev_samp",
        ),
    ],
)
def test_decimal_moment_functions_groupby(query, answer, memory_leak_check):
    """
    Tests the correctness of decimal moment-based aggregation
    functions (with group by) on decimal data. Converts to a varchar
    at the end to verify the correct scales are being used. Reference
    answers calculated from Snowflake. Uses strings for many of the answers
    to check the scale, at least when the output of the aggfunc is a decimal.
    """
    keys = []
    decimals = []

    # First group: all the same value (4x)
    keys.extend(["A"] * 4)
    decimals.extend([Decimal("123.45")] * 4)

    # Second group: evenly spaced values (4x)
    keys.extend(["B"] * 4)
    decimals.extend(
        [
            Decimal("1" * 12 + ".11"),
            Decimal("2" * 12 + ".22"),
            Decimal("3" * 12 + ".33"),
            Decimal("4" * 12 + ".44"),
        ]
    )

    # Third group: all null (10x)
    keys.extend(["C"] * 10)
    decimals.extend([None] * 10)

    # Fourth group: all null except 1 value (5x)
    keys.extend(["D"] * 5)
    decimals.extend([None, None, None, None, Decimal("-3.14")])

    # Fifth group: all null except 2 values (5x)
    keys.extend(["E"] * 5)
    decimals.extend(
        [None, None, None, Decimal("98" + "9" * 10 + ".99"), Decimal("9" * 12 + ".98")]
    )

    # Sixth group: all null except 3 values (5x)
    keys.extend(["F"] * 5)
    decimals.extend(
        [
            None,
            None,
            Decimal("7" * 12 + ".12"),
            Decimal("98" + "9" * 10 + ".99"),
            Decimal("9" * 12 + ".98"),
        ]
    )

    # Seventh group: all null except 4 values (5x)
    keys.extend(["G"] * 5)
    decimals.extend(
        [
            None,
            Decimal("6" * 12 + ".66"),
            Decimal("7" * 12 + ".12"),
            Decimal("98" + "9" * 10 + ".99"),
            Decimal("9" * 12 + ".98"),
        ]
    )

    # Eighth group: much larger group with smaller individual values
    keys.extend(["H"] * 10000)
    for i in range(10000):
        vals = [i % 3, i % 5, i % 7, i % 29]
        if 1 in vals:
            decimals.append(None)
        else:
            decimals.append(Decimal(vals[0] * vals[1] * vals[2] * vals[3]))

    # Ninth group: single row (non-null)
    keys.extend(["I"])
    decimals.extend([Decimal("-645.12")])

    # Ninth group: four rows that sum to zero
    keys.extend(["J"] * 4)
    decimals.extend(
        [Decimal("-10.75"), Decimal("-10"), Decimal("-10.25"), Decimal("31")]
    )

    # Combine with the correct dtype, and deterministically shuffle the order of the data
    df = pd.DataFrame(
        {"K": keys, "D": pd.array(decimals, dtype=pd.ArrowDtype(pa.decimal128(14, 2)))}
    )
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    df = df.iloc[perm, :]
    ctx = {"TABLE1": df}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=answer,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT K, COVAR_POP(D1, D2) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F"],
                    "RES": [
                        36.666666667,
                        None,
                        None,
                        0,
                        0,
                        -47.0391,
                    ],
                }
            ),
            id="covar_pop",
        ),
        pytest.param(
            "SELECT K, COVAR_SAMP(D1, D2) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F"],
                    "RES": [
                        55.0,
                        None,
                        None,
                        None,
                        0,
                        -70.55865,
                    ],
                }
            ),
            id="covar_samp",
        ),
        pytest.param(
            "SELECT K, CORR(D1, D2) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F"],
                    "RES": [
                        0.9986254289,
                        None,
                        None,
                        None,
                        None,
                        -0.2949576235,
                    ],
                }
            ),
            id="corr",
        ),
        pytest.param(
            "SELECT K, COVAR_SAMP(D1, D2) + COVAR_POP(D1, D2) + CORR(D1, D2) as RES FROM TABLE1 GROUP BY K",
            pd.DataFrame(
                {
                    "K": ["A", "B", "C", "D", "E", "F"],
                    "RES": [
                        92.665292096,
                        None,
                        None,
                        None,
                        None,
                        -117.892707624,
                    ],
                }
            ),
            id="multiple",
        ),
    ],
)
def test_decimal_two_arg_moment_functions_groupby(query, answer, memory_leak_check):
    """
    Tests the correctness of decimal covariance/correlation aggregation
    functions (with group by) on decimal data. Reference
    answers calculated from Snowflake.
    """
    keys = []
    decimals_1 = []
    decimals_2 = []

    # First group: 6 values, 3 have at least 1 null.
    keys.extend(["A"] * 6)
    decimals_1.extend(
        [
            Decimal("1.0"),
            Decimal("3.0"),
            Decimal("5.0"),
            Decimal("7.0"),
            Decimal("8.0"),
            None,
        ]
    )
    decimals_2.extend(
        [Decimal("20.0"), Decimal("50.0"), Decimal("75.0"), None, None, Decimal("-1.0")]
    )

    # Second group: 3 values, all 3 have at least 1 null.
    keys.extend(["B"] * 3)
    decimals_1.extend([None, Decimal("1.0"), None])
    decimals_2.extend([Decimal("7.5"), None, None])

    # Third group: 1 value, everything is null
    keys.extend(["C"] * 3)
    decimals_1.extend([None] * 3)
    decimals_2.extend([None] * 3)

    # Fourth group: 3 values, 2 have at least 1 null.
    keys.extend(["D"] * 3)
    decimals_1.extend([Decimal("-5.63"), Decimal("7.0"), None])
    decimals_2.extend([Decimal("20.14"), None, Decimal("19.0")])

    # Fifth group: 3 values, no nulls, no covariance.
    keys.extend(["E"] * 3)
    decimals_1.extend([Decimal("100.0"), Decimal("125.3"), Decimal("128.61")])
    decimals_2.extend([Decimal("20.25")] * 3)

    # Sixth group: 3 values, no nulls, with covariance.
    keys.extend(["F"] * 3)
    decimals_1.extend([Decimal("100.0"), Decimal("125.3"), Decimal("128.61")])
    decimals_2.extend([Decimal("20.25"), Decimal("30.16"), Decimal("0.16")])

    # Combine with the correct dtype, and deterministically shuffle the order of the data
    df = pd.DataFrame(
        {
            "K": keys,
            "D1": pd.array(decimals_1, dtype=pd.ArrowDtype(pa.decimal128(10, 2))),
            "D2": pd.array(decimals_2, dtype=pd.ArrowDtype(pa.decimal128(10, 2))),
        }
    )
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    df = df.iloc[perm, :]
    ctx = {"TABLE1": df}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=answer,
            check_dtype=False,
        )


def test_decimal_int_multiply_vector(decimal_data, memory_leak_check):
    """
    Tests multiplication of integers and decimals that is forced
    via a transposition of a SUM aggregation below a join.
    """
    query = """
    WITH CTE_TABLE_1 AS (
        SELECT DISTINCT group_ids
        FROM TABLE_1
        UNION
        SELECT DISTINCT benchmark_cashflow_group_ids
        FROM TABLE_1
    )
    SELECT
        TABLE2.group_ids,
        SUM(TABLE2.amounts) AS AMOUNTS,
        TABLE2.cashflow_type_ids,
        TABLE3.cashflow_group_ids,
        TABLE3.some_col
    FROM TABLE_2 TABLE2
    INNER JOIN TABLE_3 TABLE3
        ON TABLE2.group_ids = TABLE3.cashflow_group_ids
    INNER JOIN CTE_TABLE_1 M
        ON M.group_ids = TABLE2.group_ids
    GROUP BY
        TABLE2.group_ids,
        TABLE2.amounts,
        TABLE2.cashflow_type_ids,
        TABLE3.cashflow_group_ids,
        TABLE3.some_col
    """
    answer = pd.DataFrame(
        {
            "group_ids": [1, 2],
            "amounts": [9.0, 18.0],
            "cashflow_type_ids": [1, 2],
            "cashflow_group_ids": [1, 2],
            "some_col": [1, 2],
        }
    )
    check_query(
        query,
        decimal_data,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "int_data, result",
    [
        pytest.param(
            pd.array([1, 2, 3, 0, 5], dtype=pd.Int8Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("7.5"), None, Decimal("51.25")],
                type=pa.decimal128(38, 2),
            ),
            id="int8",
        ),
        pytest.param(
            pd.array([1, 20, 30, 0, 50], dtype=pd.Int16Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("75"), None, Decimal("512.5")],
                type=pa.decimal128(38, 2),
            ),
            id="int16",
        ),
        pytest.param(
            pd.array([1, -125, 300, 0, 131072], dtype=pd.Int32Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("750"), None, Decimal("1343488")],
                type=pa.decimal128(38, 2),
            ),
            id="int32",
        ),
        pytest.param(
            pd.array([1, 2, 3, 0, 987654321], dtype=pd.Int64Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("7.5"), None, Decimal("10123456790.25")],
                type=pa.decimal128(38, 2),
            ),
            id="int64",
        ),
        pytest.param(
            pd.array([1, 20, 40, 0, 80], dtype=pd.UInt8Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("100"), None, Decimal("820")],
                type=pa.decimal128(38, 2),
            ),
            id="uint8",
        ),
        pytest.param(
            pd.array([1, 20, 300, 0, 5000], dtype=pd.UInt16Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("750"), None, Decimal("51250")],
                type=pa.decimal128(38, 2),
            ),
            id="uint16",
        ),
        pytest.param(
            pd.array([1, 21, 321, 0, 54321], dtype=pd.UInt32Dtype()),
            pa.array(
                [Decimal("0"), None, Decimal("802.5"), None, Decimal("556790.25")],
                type=pa.decimal128(38, 2),
            ),
            id="uint32",
        ),
        pytest.param(
            pd.array([1, 54321, 654321, 0, 7654321], dtype=pd.UInt64Dtype()),
            pa.array(
                [
                    Decimal("0"),
                    None,
                    Decimal("1635802.5"),
                    None,
                    Decimal("78456790.25"),
                ],
                type=pa.decimal128(38, 2),
            ),
            id="uint64",
        ),
    ],
)
def test_decimal_int_multiply_scalar(int_data, result, memory_leak_check):
    """
    Tests multiplication of integers and decimals in a case statement
    """
    query = """
    SELECT CASE WHEN I > 0 THEN NULLIF(D, 1) * I ELSE NULL END as RES
    FROM TABLE1
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I": int_data,
                "D": pd.array(
                    pa.array(
                        [
                            Decimal("0"),
                            Decimal("1"),
                            Decimal("2.5"),
                            Decimal("3"),
                            Decimal("10.25"),
                        ],
                        type=pa.decimal128(4, 2),
                    ),
                    dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                ),
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame(
            {"RES": pd.array(result, dtype=pd.ArrowDtype(result.type))}
        ),
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="without_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "int_data, prec, scale, result",
    [
        pytest.param(
            pd.array([1, -4, 8, -32, None, -128, 127], dtype=pd.Int8Dtype()),
            4,
            1,
            pa.array(
                [
                    Decimal("1.0"),
                    Decimal("-4.0"),
                    Decimal("8.0"),
                    Decimal("-32.0"),
                    None,
                    Decimal("-128.0"),
                    Decimal("127.0"),
                ],
                type=pa.decimal128(4, 1),
            ),
            id="int8-prec_4-scale_1",
        ),
        pytest.param(
            pd.array([i**3 for i in range(3, 8)], dtype=pd.Int16Dtype()),
            10,
            2,
            pa.array(
                [
                    Decimal("27.00"),
                    Decimal("64.00"),
                    Decimal("125.00"),
                    Decimal("216.00"),
                    Decimal("343.00"),
                ],
                type=pa.decimal128(10, 2),
            ),
            id="int16-prec_10-scale_2",
        ),
        pytest.param(
            pd.array(
                [None if i % 2 == 1 else 4**i - 1 for i in range(2, 9)],
                dtype=pd.Int64Dtype(),
            ),
            24,
            0,
            pa.array(
                [
                    Decimal("15"),
                    None,
                    Decimal("255"),
                    None,
                    Decimal("4095"),
                    None,
                    Decimal("65535"),
                ],
                type=pa.decimal128(24, 0),
            ),
            id="int64-prec_24-scale_0",
        ),
    ],
)
def test_int_to_decimal(int_data, prec, scale, result, use_case, memory_leak_check):
    """
    Tests direct conversion of integers to decimals with various precisions/scales
    and at both with or without case statements. The case statement is a no-op
    so long as the input is not 0.
    """
    if use_case:
        query = f"SELECT CASE WHEN I = 0 THEN NULL ELSE I :: NUMBER({prec}, {scale}) END FROM TABLE1"
    else:
        query = f"SELECT I :: NUMBER({prec}, {scale}) FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"I": int_data})}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=pd.DataFrame(
                {"RES": pd.array(result, dtype=pd.ArrowDtype(result.type))}
            ),
            sort_output=False,
        )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="without_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "float_data, prec, scale, result",
    [
        pytest.param(
            pd.array(
                [1.15, 2.93, None, 1234.5, -12.5, 143.5, -98.76],
                dtype=pd.Float32Dtype(),
            ),
            4,
            0,
            pa.array(
                [
                    Decimal("1"),
                    Decimal("3"),
                    None,
                    Decimal("1234"),
                    Decimal("-12"),
                    Decimal("144"),
                    Decimal("-99"),
                ],
                type=pa.decimal128(4, 0),
            ),
            id="float32-prec_4-scale_0",
        ),
        pytest.param(
            pd.array(
                [12.3456, -9876.54321, None, 123456.789, -0.00000000123],
                dtype=pd.Float64Dtype(),
            ),
            8,
            2,
            pa.array(
                [
                    Decimal("12.35"),
                    Decimal("-9876.54"),
                    None,
                    Decimal("123456.79"),
                    Decimal("0.00"),
                ],
                type=pa.decimal128(8, 2),
            ),
            id="float64-prec_8-scale_2",
        ),
    ],
)
def test_float_to_decimal(float_data, prec, scale, result, use_case, memory_leak_check):
    """
    Tests direct conversion of floats to decimals with various precisions/scales
    and at both with or without case statements. The case statement is a no-op
    so long as the input is not 0.
    """
    if use_case:
        query = f"SELECT CASE WHEN F = 0 THEN NULL ELSE F :: NUMBER({prec}, {scale}) END as RES FROM TABLE1"
    else:
        query = f"SELECT F :: NUMBER({prec}, {scale}) FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"F": float_data})}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=pd.DataFrame(
                {"RES": pd.array(result, dtype=pd.ArrowDtype(result.type))}
            ),
            sort_output=False,
            convert_columns_decimal=["RES"] if use_case else None,
        )


@pytest.mark.skipif(bodo.get_size() != 1, reason="skip on multiple ranks")
@pytest.mark.parametrize(
    "expr, error_message",
    [
        pytest.param(
            "CASE WHEN F = 0 THEN NULL ELSE F :: NUMBER(5, 2) END",
            "Number out of representable range",
            id="scalar",
        ),
        pytest.param("F :: NUMBER(5, 2)", "Invalid float to decimal cast", id="vector"),
    ],
)
def test_float_to_decimal_error(expr, error_message):
    """
    Variant of test_float_to_decimal where the floats won't fit in the decimal range.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": pd.DataFrame({"F": [12345.6789] * 5})}

    with temp_config_override("bodo_use_decimal", True):
        with pytest.raises(Exception, match=error_message):
            check_query(
                query,
                ctx,
                None,
                check_dtype=False,
                check_names=False,
                expected_output=pd.DataFrame({"RES": [-1.0] * 5}),
                sort_output=False,
            )


def test_decimal_to_float_cast(memory_leak_check):
    query = "Select A::DOUBLE as OUTPUT from TABLE1"
    df = pd.DataFrame(
        {
            "A": pa.array(
                [Decimal("0"), None, Decimal("7.5"), None, Decimal("51.25")],
                type=pa.decimal128(38, 2),
            )
        }
    )
    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"OUTPUT": [0.0, None, 7.5, None, 51.25]}),
        sort_output=False,
    )


@pytest.mark.parametrize(
    "df, expr, expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("0"),
                            Decimal("123456789012345678901234567890.12345678"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
                    )
                }
            ),
            "SIGN(D1)",
            pd.array([1, None, -1, 0, 1]),
            id="array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("0"),
                            Decimal("123456789012345678901234567890.12345678"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE SIGN(D1) :: VARCHAR END",
            pd.array(["1", "", "-1", "0", "1"]),
            id="array-case",
        ),
    ],
)
def test_decimal_sign(df, expr, expected, memory_leak_check):
    """
    Tests the correctness of the SIGN function for decimals.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"RES": expected}),
        sort_output=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("13.14"),
                            None,
                            Decimal("15.16"),
                            None,
                            Decimal("-17.18"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "D1 + D2",
            pd.array(
                [Decimal("14.37"), None, Decimal("-30.51"), None, Decimal("-28.30")],
                dtype=pd.ArrowDtype(pa.decimal128(11, 2)),
            ),
            id="array-array-same_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("10"),
                            None,
                            Decimal("20"),
                            Decimal("30"),
                            Decimal("-40"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(2, 0)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("3.14"),
                            None,
                            Decimal("95.12"),
                            None,
                            Decimal("-65.5"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            "D1 + D2",
            pd.array(
                [Decimal("13.14"), None, Decimal("115.12"), None, Decimal("-105.5")],
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-array-smaller_scale",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1234.5"),
                            None,
                            Decimal("9876.5"),
                            Decimal("9090.9"),
                            Decimal("-9999.9"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(5, 1)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("54321"),
                            None,
                            Decimal("11110"),
                            None,
                            Decimal("-99999"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(5, 0)),
                    ),
                }
            ),
            "D1 + D2",
            pd.array(
                [
                    Decimal("55555.5"),
                    None,
                    Decimal("20986.5"),
                    None,
                    Decimal("-109998.9"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(7, 1)),
            ),
            id="array-array-larger_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("-11.12"),
                            Decimal("89.10"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            # Using an expression that does not currently simplify to a float literal in BodoSQL
            "D1 + (LEFT('15.3', 4) :: NUMBER(4, 2))",
            pd.array(
                [
                    Decimal("16.53"),
                    None,
                    Decimal("-30.37"),
                    Decimal("4.18"),
                    Decimal("104.4"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-same_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D2": pd.array(
                        [
                            Decimal("12.34"),
                            None,
                            Decimal("56.78"),
                            Decimal("-12.34"),
                            Decimal("-56.78"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            # Using an expression that does not currently simplify to a float literal in BodoSQL
            "(LEFT('9999', 4) :: NUMBER(4, 0)) + D2",
            pd.array(
                [
                    Decimal("10011.34"),
                    None,
                    Decimal("10055.78"),
                    Decimal("9986.66"),
                    Decimal("9942.22"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(7, 2)),
            ),
            id="scalar-array-smaller_scale",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("13.14"),
                            None,
                            Decimal("15.16"),
                            None,
                            Decimal("-17.18"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "(CASE WHEN D1 IS NULL THEN '' ELSE (D1 + D2)::VARCHAR END)",
            pd.array(["14.37", "", "-30.51", None, "-28.30"]),
            id="scalar-scalar-same_scale",
        ),
    ],
)
def test_decimal_addition(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal addition against other decimals.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("-13.14"),
                            None,
                            Decimal("-15.16"),
                            None,
                            Decimal("17.18"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "D1 - D2",
            pd.array(
                [Decimal("14.37"), None, Decimal("-30.51"), None, Decimal("-28.30")],
                dtype=pd.ArrowDtype(pa.decimal128(11, 2)),
            ),
            id="array-array-same_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("10"),
                            None,
                            Decimal("20"),
                            Decimal("30"),
                            Decimal("-40"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(2, 0)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("-3.14"),
                            None,
                            Decimal("-95.12"),
                            None,
                            Decimal("65.5"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            "D1 - D2",
            pd.array(
                [Decimal("13.14"), None, Decimal("115.12"), None, Decimal("-105.5")],
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-array-smaller_scale",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1234.5"),
                            None,
                            Decimal("9876.5"),
                            Decimal("9090.9"),
                            Decimal("-9999.9"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(5, 1)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("-54321"),
                            None,
                            Decimal("-11110"),
                            None,
                            Decimal("99999"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(5, 0)),
                    ),
                }
            ),
            "D1 - D2",
            pd.array(
                [
                    Decimal("55555.5"),
                    None,
                    Decimal("20986.5"),
                    None,
                    Decimal("-109998.9"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(7, 1)),
            ),
            id="array-array-larger_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("-11.12"),
                            Decimal("89.10"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            # Using an expression that does not currently simplify to a float literal in BodoSQL
            "D1 - (LEFT('15.3', 4) :: NUMBER(4, 2))",
            pd.array(
                [
                    Decimal("-14.07"),
                    None,
                    Decimal("-60.97"),
                    Decimal("-26.42"),
                    Decimal("73.80"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-same_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D2": pd.array(
                        [
                            Decimal("-12.34"),
                            None,
                            Decimal("-56.78"),
                            Decimal("12.34"),
                            Decimal("56.78"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
                    ),
                }
            ),
            # Using an expression that does not currently simplify to a float literal in BodoSQL
            "(LEFT('9999', 4) :: NUMBER(4, 0)) - D2",
            pd.array(
                [
                    Decimal("10011.34"),
                    None,
                    Decimal("10055.78"),
                    Decimal("9986.66"),
                    Decimal("9942.22"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(7, 2)),
            ),
            id="scalar-array-smaller_scale",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("-13.14"),
                            None,
                            Decimal("-15.16"),
                            None,
                            Decimal("17.18"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "(CASE WHEN D1 IS NULL THEN '' ELSE (D1 - D2)::VARCHAR END)",
            pd.array(["14.37", "", "-30.51", None, "-28.30"]),
            id="scalar-scalar-same_scale",
        ),
    ],
)
def test_decimal_subtraction(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal subtraction against other decimals.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.2345"),
                            Decimal("5.6789"),
                            Decimal("2.9999"),
                            Decimal("313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "ROUND(D1, 3)",
            pd.array(
                [
                    Decimal("1.235"),
                    Decimal("5.679"),
                    Decimal("3"),
                    Decimal("313.212"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(14, 3)),
            ),
            id="array-round",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1.2345"),
                            Decimal("-5.6789"),
                            Decimal("-2.9999"),
                            Decimal("-313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "ROUND(D1, 3)",
            pd.array(
                [
                    Decimal("-1.235"),
                    Decimal("-5.679"),
                    Decimal("-3"),
                    Decimal("-313.212"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(14, 3)),
            ),
            id="array-negative-round",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1521.2345"),
                            Decimal("63455.6789"),
                            Decimal("17542.9999"),
                            Decimal("99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "ROUND(D1, -1)",
            pd.array(
                [
                    Decimal("1520"),
                    Decimal("63460"),
                    Decimal("17540"),
                    Decimal("99310"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(14, 0)),
            ),
            id="array-round-negative_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1521.2345"),
                            Decimal("-63455.6789"),
                            Decimal("-17542.9999"),
                            Decimal("-99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "ROUND(D1, -1)",
            pd.array(
                [
                    Decimal("-1520"),
                    Decimal("-63460"),
                    Decimal("-17540"),
                    Decimal("-99310"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(14, 0)),
            ),
            id="array-negative-round-negative_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1521.2345"),
                            Decimal("-63455.6789"),
                            Decimal("-17542.9999"),
                            Decimal("-99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "ROUND(D1, 7)",
            pd.array(
                [
                    Decimal("-1521.2345"),
                    Decimal("-63455.6789"),
                    Decimal("-17542.9999"),
                    Decimal("-99313.2121561"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
            ),
            id="array-no_change",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("999.9999"),
                            Decimal("-999.9999"),
                            Decimal("99999.9999"),
                            Decimal("-99999.999"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 4)),
                    )
                }
            ),
            "ROUND(D1, 2)",
            pd.array(
                [
                    Decimal("1000"),
                    Decimal("-1000"),
                    Decimal("100000"),
                    Decimal("-100000"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(14, 2)),
            ),
            id="array-round-propagate",
        ),
        # Case statements
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.2345"),
                            Decimal("5.6789"),
                            Decimal("2.9999"),
                            Decimal("313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, 3)::VARCHAR  END",
            pd.array(
                [
                    "1.235",
                    "5.679",
                    "3.000",
                    "313.212",
                    "",
                ],
            ),
            id="array-round-case",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1.2345"),
                            Decimal("-5.6789"),
                            Decimal("-2.9999"),
                            Decimal("-313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, 3)::VARCHAR END",
            pd.array(
                [
                    "-1.235",
                    "-5.679",
                    "-3.000",
                    "-313.212",
                    "",
                ],
            ),
            id="array-negative-round-case",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1521.2345"),
                            Decimal("63455.6789"),
                            Decimal("17542.9999"),
                            Decimal("99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, -1)::VARCHAR END",
            pd.array(
                [
                    "1520",
                    "63460",
                    "17540",
                    "99310",
                    "",
                ],
            ),
            id="array-round-negative_scale-case",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1521.2345"),
                            Decimal("-63455.6789"),
                            Decimal("-17542.9999"),
                            Decimal("-99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, -1)::VARCHAR END",
            pd.array(
                [
                    "-1520",
                    "-63460",
                    "-17540",
                    "-99310",
                    "",
                ],
            ),
            id="array-negative-round-negative_scale-case",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("-1521.2345"),
                            Decimal("-63455.6789"),
                            Decimal("-17542.9999"),
                            Decimal("-99313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, 7)::VARCHAR END",
            pd.array(
                [
                    "-1521.2345000",
                    "-63455.6789000",
                    "-17542.9999000",
                    "-99313.2121561",
                    "",
                ],
            ),
            id="array-no_change-case",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("999.9999"),
                            Decimal("-999.9999"),
                            Decimal("99999.9999"),
                            Decimal("-99999.999"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 4)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE ROUND(D1, 2)::VARCHAR END",
            pd.array(
                [
                    "1000.00",
                    "-1000.00",
                    "100000.00",
                    "-100000.00",
                    "",
                ],
            ),
            id="array-round-propagate-case",
        ),
    ],
)
def test_decimal_rounding(df, expr, answer, spark_info, memory_leak_check):
    """
    Tests the correctness of decimal rounding with different scales.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": df}

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            spark_info,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "CEIL(D1, 2)",
            pd.array(
                [
                    Decimal("648.30"),
                    Decimal("-152.58"),
                    Decimal("-0.15"),
                    Decimal("0.53"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="positive_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "CEIL(D1)",
            pd.array(
                [
                    Decimal("649"),
                    Decimal("-152"),
                    Decimal("0"),
                    Decimal("1"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="no_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "CEIL(D1, -2)",
            pd.array(
                [
                    Decimal("700"),
                    Decimal("-100"),
                    Decimal("0"),
                    Decimal("100"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 0)),
            ),
            id="negative_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("999.9999"),
                            Decimal("-999.9999"),
                            Decimal("99999.9999"),
                            Decimal("-99999.999"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 4)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE CEIL(D1, 2)::VARCHAR END",
            pd.array(
                [
                    "1000.00",
                    "-999.99",
                    "100000.00",
                    "-99999.99",
                    "",
                ],
            ),
            id="case",
        ),
    ],
)
def test_decimal_ceil(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal CEIL with different scales.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "FLOOR(D1, 2)",
            pd.array(
                [
                    Decimal("648.29"),
                    Decimal("-152.59"),
                    Decimal("-0.16"),
                    Decimal("0.52"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="positive_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "FLOOR(D1)",
            pd.array(
                [
                    Decimal("648"),
                    Decimal("-153"),
                    Decimal("-1"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="no_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "FLOOR(D1, -2)",
            pd.array(
                [
                    Decimal("600"),
                    Decimal("-200"),
                    Decimal("-100"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 0)),
            ),
            id="negative_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("999.9999"),
                            Decimal("-999.9999"),
                            Decimal("99999.9999"),
                            Decimal("-99999.999"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 4)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE FLOOR(D1, 2)::VARCHAR END",
            pd.array(
                [
                    "999.99",
                    "-1000.00",
                    "99999.99",
                    "-100000.00",
                    "",
                ],
            ),
            id="case",
        ),
    ],
)
def test_decimal_floor(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal FLOOR with different scales.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "TRUNC(D1, 2) :: VARCHAR",
            pd.array(
                [
                    "648.29",
                    "-152.58",
                    "-0.15",
                    "0.52",
                    "0.00",
                    None,
                ],
            ),
            id="positive_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "TRUNC(D1) :: VARCHAR",
            pd.array(
                [
                    "648",
                    "-152",
                    "0",
                    "0",
                    "0",
                    None,
                ],
            ),
            id="no_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("648.2935"),
                            Decimal("-152.5826"),
                            Decimal("-0.15122"),
                            Decimal("0.5233"),
                            Decimal("0"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    )
                }
            ),
            "TRUNC(D1, -2) :: VARCHAR",
            pd.array(
                [
                    "600",
                    "-100",
                    "0",
                    "0",
                    "0",
                    None,
                ],
            ),
            id="negative_scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("999.9999"),
                            Decimal("-999.9999"),
                            Decimal("99999.9999"),
                            Decimal("-99999.999"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 4)),
                    )
                }
            ),
            "CASE WHEN D1 IS NULL THEN '' ELSE TRUNC(D1, 2)::VARCHAR END",
            pd.array(
                [
                    "999.99",
                    "-999.99",
                    "99999.99",
                    "-99999.99",
                    "",
                ],
            ),
            id="case",
        ),
    ],
)
def test_decimal_trunc(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal TRUNC with different scales.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("-8.0"),
                            Decimal("75.12"),
                            Decimal("-16777216"),
                            Decimal("16777.216"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(11, 3)),
                    )
                }
            ),
            "CBRT(D)",
            pd.array(
                [
                    -2.0,
                    4.2194112818,
                    -256,
                    25.6,
                    None,
                ],
            ),
            id="cbrt-vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "ATAN(D)",
            pd.array(
                [
                    0.8881737743776796,
                    None,
                    -1.2124361655324638,
                    0.48211338922722113,
                    -0.11942892601833846,
                ],
            ),
            id="atan_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("0.23"),
                            None,
                            Decimal("-1.67"),
                            Decimal("0.1233"),
                            Decimal("-2.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "ATAN2(D, D2)",
            pd.array(
                [
                    1.385939294776456,
                    None,
                    -2.1297322291334897,
                    1.3393967973933203,
                    -3.085049216644976,
                ],
            ),
            id="atan2_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("0.23"),
                            None,
                            Decimal("-0.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "ATANH(D)",
            pd.array(
                [
                    0.2341894667593668,
                    None,
                    -0.8107431254751375,
                    0.5808734754659377,
                    -0.12058102840844405,
                ],
            ),
            id="atanh_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "COS(D)",
            pd.array(
                [
                    0.3342377271245024,
                    None,
                    -0.8908458667805766,
                    0.8661747529276824,
                    0.9928086358538663,
                ],
            ),
            id="cos_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "COSH(D)",
            pd.array(
                [
                    1.8567610569852668,
                    None,
                    7.2546107090561165,
                    1.140074686717299,
                    1.0072086441482666,
                ],
            ),
            id="cosh_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "COT(D)",
            pd.array(
                [
                    0.35463310167660184,
                    None,
                    1.9608953309196617,
                    1.7332465287768206,
                    -8.29329488059453,
                ],
            ),
            id="cot_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "DEGREES(D)",
            pd.array(
                [
                    70.47380880109127,
                    None,
                    -152.97973129992982,
                    29.98288141919598,
                    -6.875493541569879,
                ],
            ),
            id="degrees_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "RADIANS(D)",
            pd.array(
                [
                    0.021467549799530257,
                    None,
                    -0.04660029102824861,
                    0.009133307975686327,
                    -0.0020943951023931957,
                ],
            ),
            id="radians_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "SIN(D)",
            pd.array(
                [
                    0.9424888019316976,
                    None,
                    -0.45430566983030607,
                    0.4997412304289775,
                    -0.11971220728891938,
                ],
            ),
            id="sin_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "SINH(D)",
            pd.array(
                [
                    1.5644684793044075,
                    None,
                    -7.1853584837467706,
                    0.5475128229489677,
                    -0.12028820743110909,
                ],
            ),
            id="sinh_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "TAN(D)",
            pd.array(
                [
                    2.819815734268154,
                    None,
                    0.5099711260626028,
                    0.5769519704191854,
                    -0.12057933721130533,
                ],
            ),
            id="tan_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-2.67"),
                            Decimal("0.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 5)),
                    ),
                }
            ),
            "TANH(D)",
            pd.array(
                [
                    0.8425793256589296,
                    None,
                    -0.9904540397704736,
                    0.4802429431403847,
                    -0.1194272985343859,
                ],
            ),
            id="tanh_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "EXP(D)",
            pd.array(
                [
                    507.7554834957939,
                    None,
                    4.464286286584116e222,
                    2676536492286818.5,
                    0.8869204367171575,
                ],
            ),
            id="exp_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "LN(D)",
            pd.array(
                [
                    1.8293763327993617,
                    None,
                    6.239632363326927,
                    3.570188819213935,
                    -2.120263536200091,
                ],
            ),
            id="ln_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "SQRT(D)",
            pd.array(
                [
                    2.495996794869737,
                    None,
                    22.642217205918683,
                    5.960142615743352,
                    0.34641016151377546,
                ],
            ),
            id="sqrt_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("-0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "SQUARE(D)",
            pd.array(
                [
                    38.812900000000006,
                    None,
                    262830.5289000001,
                    1261.9048428899998,
                    0.014400000000000003,
                ],
            ),
            id="square_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("1.2"),
                            None,
                            Decimal("2.34"),
                            Decimal("0"),
                            Decimal("-0.9"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "POWER(D, D2)",
            pd.array(
                [8.982260703582106, None, 2192909.437077487, 1.0, 6.741194822493127],
            ),
            id="power_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "D2": pd.array(
                        [
                            Decimal("1.2"),
                            None,
                            Decimal("2.34"),
                            Decimal("4.1"),
                            Decimal("0.9"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "LOG(D, D2)",
            pd.array(
                [
                    10.033790655192673,
                    None,
                    7.339440736662648,
                    2.5302776607681507,
                    20.123891032253084,
                ],
            ),
            id="log_vector",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D": pd.array(
                        [
                            Decimal("6.23"),
                            None,
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("0.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "LOG(D)",
            pd.array(
                [
                    0.7944880466591696,
                    None,
                    2.7098379044978307,
                    1.5505133035372982,
                    -0.9208187539523751,
                ],
            ),
            id="log_onearg_vector",
        ),
    ],
)
def test_decimal_to_float_functions(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of functions that when called on decimals
    should cast to float.
    """
    query = f"SELECT {expr} AS RES FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                ["1.23", None, "-45.67", "89.10", "-11.12"],
            ),
            id="same-scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.673"),
                            Decimal("89.1056"),
                            Decimal("-11.123456"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 6)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                ["1.230000", None, "-45.673000", "89.105600", "-11.123456"],
            ),
            id="varying-scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("123.4567"),
                            None,
                            Decimal("-0.0001"),
                            Decimal("789.0123"),
                            Decimal("-456.7890"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 4)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                ["123.4567", None, "-0.0001", "789.0123", "-456.7890"],
            ),
            id="different-precision-scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            None,
                            None,
                            None,
                            None,
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array([None, None, None, None, None]),
            id="all-none-values",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("100.01"),
                            Decimal("-200.02"),
                            Decimal("300.03"),
                            Decimal("-400.04"),
                            Decimal("500.05"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                ["100.01", "-200.02", "300.03", "-400.04", "500.05"],
            ),
            id="mix-positive-negative",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("0.12345678901234567890"),
                            None,
                            Decimal("-0.12345678901234567890123456789012345678"),
                            Decimal("0.00000000000000000000000000000000000001"),
                            Decimal("-0.00000000000000000000000000000000000001"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 38)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                [
                    "0.12345678901234567890000000000000000000",
                    None,
                    "-0.12345678901234567890123456789012345678",
                    "0.00000000000000000000000000000000000001",
                    "-0.00000000000000000000000000000000000001",
                ],
            ),
            id="large-scale-38-38",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("12345678901234567890.123456789012345678"),
                            None,
                            Decimal("-12345678901234567890.123456789012345678"),
                            Decimal("12345678901234567890.123456789012345678"),
                            Decimal("-12345678901234567890.123456789012345678"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
                    ),
                }
            ),
            "D1 :: VARCHAR",
            pd.array(
                [
                    "12345678901234567890.123456789012345678",
                    None,
                    "-12345678901234567890.123456789012345678",
                    "12345678901234567890.123456789012345678",
                    "-12345678901234567890.123456789012345678",
                ],
            ),
            id="large-precision-38-18",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "(CASE WHEN D1 IS NULL THEN '' ELSE D1::VARCHAR END)",
            pd.array(["1.23", "", "-45.67", "89.10", "-11.12"]),
            id="case-same-scale",
        ),
    ],
)
def test_decimal_to_string(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal conversion to string.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame(
                {"RES": pd.array(answer, dtype="string[pyarrow]")}
            ),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 3, 3, 4, 4, 4, 4, 4],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "5.12309891",
                            "6.123125236",
                            "1.6325",
                            "2.5123",
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    0: [1, 2, 3, 4],
                    1: [
                        Decimal("1.5"),
                        Decimal("3"),
                        Decimal("4.5"),
                        Decimal("4.10906127"),
                    ],
                }
            ),
            id="basic",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, 4, 5],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    0: [1, 2, 3, 4, 5],
                    1: [
                        Decimal("1"),
                        Decimal("2"),
                        Decimal("3"),
                        Decimal("4"),
                        Decimal("5"),
                    ],
                }
            ),
            id="all-separate",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                    "B": pd.array(
                        [
                            "0",
                            "6.123125236",
                            "-1.6325",
                            "2.5123",
                            "4.10906127",
                            "0",
                            None,
                            "-1.6325",
                            None,
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            pd.DataFrame({0: [1, 2], 1: [Decimal("2.5123"), Decimal("0")]}),
            id="negatives-zeroes-nulls",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 2],
                    "B": pd.array(
                        [
                            None,
                            None,
                            "-1.6325",
                            "0.45",
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    0: [1, 2],
                    1: pd.array(
                        [None, Decimal("0.45")],
                        dtype=pd.ArrowDtype(pa.decimal128(23, 13)),
                    ),
                }
            ),
            id="group-of-nulls",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1.1",
                            "1.1",
                            "1.1234567890123456789012345678901234",
                            "1.1234567890123456789012345678901234",
                            "1.0000000000000000000000000000000001",
                            "1.0000000000000000000000000000000003",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(37, 34)),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    0: [1, 2, 3],
                    1: [
                        Decimal("1.1"),
                        Decimal("1.1234567890123456789012345678901234"),
                        Decimal("1.0000000000000000000000000000000002"),
                    ],
                }
            ),
            id="large_scale_precision",
        ),
    ],
)
def test_decimal_median(df, expected, spark_info, memory_leak_check):
    query = "SELECT A, median(B) FROM TABLE1 GROUP BY A"

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            expected_output=expected,
        )


@pytest.mark.parametrize(
    "df, percentile, expected",
    [
        # 10-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 10,
                    "B": pd.array(
                        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["1.90000000"],
                }
            ),
            id="ten-row-group",
        ),
        # 20-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 20,
                    "B": pd.array(
                        [str(i) for i in range(1, 21)],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["2.90000000"],
                }
            ),
            id="twenty-row-group",
        ),
        # 15-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 15,
                    "B": pd.array(
                        [str(i) for i in range(1, 16)],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["2.40000000"],
                }
            ),
            id="fifteen-row-group",
        ),
        # 17-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 17,
                    "B": pd.array(
                        [str(i) for i in range(1, 18)],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["2.60000000"],
                }
            ),
            id="seventeen-row-group",
        ),
        # Large numbers
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1],
                    "B": pd.array(
                        ["-1234123432.25242", "93573485693832.9573493"],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 10)),
                    ),
                }
            ),
            0.385,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["36025033006214.8533411805000"],
                }
            ),
            id="large-numbers",
        ),
        # exact selections
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["0.00000000"],
                }
            ),
            id="exact-first",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.4,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["4.00000000"],
                }
            ),
            id="exact-middle",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["10.00000000"],
                }
            ),
            id="exact-last",
        ),
        # Multiple groups
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        1,
                        1,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                        3,
                    ],
                    "B": pd.array(
                        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            0.37,
            pd.DataFrame(
                {
                    0: [1, 2, 3],
                    1: ["1.37000", "3.74000", "7.48000"],
                }
            ),
            id="multiple-groups",
        ),
    ],
)
def test_decimal_percentile_cont(
    df, percentile, expected, spark_info, memory_leak_check
):
    query = f"SELECT A, (PERCENTILE_CONT({percentile}) WITHIN GROUP (ORDER BY B))::VARCHAR AS C FROM TABLE1 GROUP BY A"

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            expected_output=expected,
        )


@pytest.mark.parametrize(
    "df, percentile, expected",
    [
        # exact selections
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["0.00000"],
                }
            ),
            id="exact-first",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.4,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["4.00000"],
                }
            ),
            id="exact-middle",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 11,
                    "B": pd.array(
                        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            1,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["10.00000"],
                }
            ),
            id="exact-last",
        ),
        # 10-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 10,
                    "B": pd.array(
                        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.18,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["2.00000"],
                }
            ),
            id="ten-row-group",
        ),
        # 15-row group
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1] * 15,
                    "B": pd.array(
                        [str(i) for i in range(1, 16)],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
                    ),
                }
            ),
            0.24,
            pd.DataFrame(
                {
                    0: [1],
                    1: ["4.00000"],
                }
            ),
            id="fifteen-row-group",
        ),
        # Multiple groups
        pytest.param(
            pd.DataFrame(
                {
                    "A": [
                        1,
                        1,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                        3,
                    ],
                    "B": pd.array(
                        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            0.62,
            pd.DataFrame(
                {
                    0: [1, 2, 3],
                    1: ["2.00", "4.00", "9.00"],
                }
            ),
            id="multiple-groups",
        ),
    ],
)
def test_decimal_percentile_disc(
    df, percentile, expected, spark_info, memory_leak_check
):
    query = f"SELECT A, (PERCENTILE_DISC({percentile}) WITHIN GROUP (ORDER BY B))::VARCHAR AS C FROM TABLE1 GROUP BY A"

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            expected_output=expected,
        )


@pytest.mark.parametrize(
    "arr, error_msg",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 35)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="overflow_1",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(37, 35)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="overflow_2",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="scale_too_large",
        ),
    ],
)
def test_decimal_median_error(arr, error_msg, spark_info):
    query = "SELECT A, median(B) FROM TABLE1 GROUP BY A"

    with temp_config_override("bodo_use_decimal", True):
        with pytest.raises(Exception, match=error_msg):
            check_query(
                query,
                {"TABLE1": arr},
                spark_info,
                check_names=False,
                check_dtype=False,
            )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "ABS(D1)",
            pd.array(
                [
                    Decimal("1.23"),
                    None,
                    Decimal("45.67"),
                    Decimal("89.10"),
                    Decimal("11.12"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            id="same-scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.673"),
                            Decimal("89.1056"),
                            Decimal("-11.123456"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 6)),
                    ),
                }
            ),
            "ABS(D1)",
            pd.array(
                [
                    Decimal("1.23"),
                    None,
                    Decimal("45.673"),
                    Decimal("89.1056"),
                    Decimal("11.123456"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(20, 6)),
            ),
            id="varying-scale",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            None,
                            None,
                            None,
                            None,
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "ABS(D1)",
            pd.array(
                [None, None, None, None, None],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            id="all-none-values",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("0.12345678901234567890"),
                            None,
                            Decimal("-0.1234567890123456789012345678901234567"),
                            Decimal("0.0000000000000000000000000000000000001"),
                            Decimal("-0.0000000000000000000000000000000000001"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 38)),
                    ),
                }
            ),
            "ABS(D1)",
            pd.array(
                [
                    Decimal("0.12345678901234567890"),
                    None,
                    Decimal("0.1234567890123456789012345678901234567"),
                    Decimal("0.0000000000000000000000000000000000001"),
                    Decimal("0.0000000000000000000000000000000000001"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 37)),
            ),
            id="large-scale-38-37",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("1.23"),
                            None,
                            Decimal("-45.67"),
                            Decimal("89.10"),
                            Decimal("-11.12"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            "(CASE WHEN D1 IS NULL THEN '' ELSE ABS(D1)::VARCHAR END)",
            pd.array(["1.23", "", "45.67", "89.10", "11.12"]),
            id="case-same-scale",
        ),
    ],
)
def test_decimal_abs(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal conversion to string.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, expr, answer",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("2.4"),
                            None,
                            Decimal("-0.12"),
                            Decimal("5"),
                            Decimal("6.7"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "FACTORIAL(D1)",
            pd.array(
                [
                    Decimal("2"),
                    None,
                    Decimal("1"),
                    Decimal("120"),
                    Decimal("5040"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(37, 0)),
            ),
            id="array",
        ),
        # Float test here
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            2.4,
                            None,
                            -0.12,
                            5,
                            6.7,
                        ],
                        dtype=np.float64,
                    ),
                }
            ),
            "FACTORIAL(D1)",
            pd.array(
                [
                    2,
                    None,
                    1,
                    120,
                    5040,
                ],
            ),
            id="array-float",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "D1": pd.array(
                        [
                            Decimal("2.4"),
                            None,
                            Decimal("-0.12"),
                            Decimal("5"),
                            Decimal("6.7"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            "(CASE WHEN D1 IS NULL THEN '' ELSE FACTORIAL(D1)::VARCHAR END)",
            pd.array(["2", "", "1", "120", "5040"]),
            id="case",
        ),
    ],
)
def test_decimal_factorial(df, expr, answer, memory_leak_check):
    """
    Tests the correctness of decimal conversion to string.
    """
    query = f"SELECT {expr} AS res FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "df, ans",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            Decimal("40.7127"),
                            Decimal("99.88"),
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("-97.45"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "B": pd.array(
                        [
                            Decimal("-74.0059"),
                            Decimal("123.45"),
                            Decimal("44.984"),
                            Decimal("65.321"),
                            Decimal("45.67"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "C": pd.array(
                        [
                            Decimal("34.0500"),
                            Decimal("-32.56"),
                            None,
                            Decimal("-135.5243"),
                            Decimal("200.45"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "D": pd.array(
                        [
                            Decimal("-118.2500"),
                            Decimal("190"),
                            Decimal("2.34"),
                            Decimal("-180.66"),
                            Decimal("350.5"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            # Numbers from Snowflake
            # SELECT HAVERSINE(40.712,-74.005, 34.0500, -118.250); ...
            pd.array(
                [
                    3936.465802926,
                    14010.290778799,
                    None,
                    11100.369881961,
                    7275.318467481,
                ],
            ),
            id="all-decimal",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            Decimal("40.7127"),
                            Decimal("99.88"),
                            Decimal("512.67"),
                            Decimal("35.5233"),
                            Decimal("-97.45"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                    "B": pd.array(
                        [1.15, 342.93, 1234.5, -12.5, 143.5],
                        dtype=pd.Float64Dtype(),
                    ),
                    "C": pd.array(
                        [
                            -118.2500,
                            190,
                            2.34,
                            -180.66,
                            350.5,
                        ],
                        dtype=pd.Float64Dtype(),
                    ),
                    "D": pd.array(
                        [
                            Decimal("34.0500"),
                            Decimal("-32.56"),
                            None,
                            Decimal("-135.5243"),
                            Decimal("200.45"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            pd.array(
                [
                    16806.072534259,
                    10059.991237631,
                    None,
                    7032.187957063,
                    9408.354442329,
                ],
            ),
            id="mix-float-decimal",
        ),
    ],
)
@pytest.mark.slow
def test_haversine_decimal(df, ans, memory_leak_check):
    """
    Test correctness of haverine with decimal input
    """
    query = "SELECT HAVERSINE(A, B, C, D) AS res FROM TABLE1"
    ctx = {"TABLE1": df}
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": ans}),
            rtol=1e-04,
            sort_output=False,
            check_dtype=False,
        )
