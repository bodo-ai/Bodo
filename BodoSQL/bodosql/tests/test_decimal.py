# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests for BodoSQL involving the decimal type
"""

from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

import bodo
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
    amounts = pa.array([x for x in amounts], type=pa.decimal128(28, 4))
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
                "D": pa.array(
                    [
                        Decimal("0"),
                        Decimal("1"),
                        Decimal("2.5"),
                        Decimal("3"),
                        Decimal("10.25"),
                    ],
                    type=pa.decimal128(4, 2),
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
        expected_output=pd.DataFrame({"RES": result}),
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

    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=pd.DataFrame({"RES": result}),
            sort_output=False,
        )
    finally:
        bodo.bodo_use_decimal = old_use_decimal


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

    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=pd.DataFrame({"RES": result}),
            sort_output=False,
            convert_columns_decimal=["RES"] if use_case else None,
        )
    finally:
        bodo.bodo_use_decimal = old_use_decimal


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

    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
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
    finally:
        bodo.bodo_use_decimal = old_use_decimal


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
    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )
    finally:
        bodo.bodo_use_decimal = False


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
    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )
    finally:
        bodo.bodo_use_decimal = False


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
    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            spark_info,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )
    finally:
        bodo.bodo_use_decimal = False


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
    old_use_decimal = bodo.bodo_use_decimal
    try:
        bodo.bodo_use_decimal = True
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame({"RES": answer}),
            sort_output=False,
            check_dtype=False,
        )
    finally:
        bodo.bodo_use_decimal = False
