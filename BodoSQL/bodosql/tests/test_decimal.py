# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests for BodoSQL involving the decimal type
"""

from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

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
