import pandas as pd
import pytest
from bodosql.tests.test_window.window_common import (  # noqa
    all_types_window_df,
    count_window_applies,
)
from bodosql.tests.utils import check_query


def test_row_number_orderby(datapath, memory_leak_check):
    """Test that row_number properly handles the orderby."""
    query = "select uuid, ROW_NUMBER() OVER(PARTITION BY store_id, ret_product_id ORDER BY last_seen DESC) as row_num from table1"

    parquet_path = datapath("sample-parquet-data/rphd_sample.pq")

    ctx = {
        "table1": pd.read_parquet(parquet_path)[
            ["uuid", "store_id", "ret_product_id", "last_seen"]
        ]
    }
    py_output = pd.DataFrame(
        {
            "uuid": [
                "67cd102b-e12f-49cb-88f5-c71d6be6642f",
                "ce4d3aa7-476b-4772-94b4-18224490c7a1",
                "bb9fb6cd-477d-4923-be3b-95615bbec5a5",
                "2adcb7de-464f-4c60-81a3-58dff3f8b1c9",
                "465cbfbb-c4c9-4837-83c7-c6be96597ca4",
                "fd5db816-902e-485b-b52d-a094709439a4",
            ],
            "row_num": [3, 5, 1, 6, 4, 2],
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "order_clause",
    [
        pytest.param("A ASC NULLS FIRST", id="asc_nf"),
        pytest.param("A DESC NULLS FIRST", id="desc_nf", marks=pytest.mark.slow),
        pytest.param("A ASC NULLS LAST", id="asc_nl", marks=pytest.mark.slow),
        pytest.param("A DESC NULLS LAST", id="desc_nl", marks=pytest.mark.slow),
        pytest.param("W3 % 3 DESC, A ASC NULLS FIRST", id="combo"),
    ],
)
def test_rank_fns(all_types_window_df, spark_info, order_clause, memory_leak_check):
    """Tests rank, dense_rank, percent_rank, ntile and row_number at the same
    where the input dtype and different combinatons of asc/desc & nulls
    first/last are parametrized so that each test can have total
    fusion into a single closure"""
    is_binary = type(all_types_window_df["table1"]["A"].iloc[0]) == bytes
    is_tz_aware = (
        getattr(all_types_window_df["table1"]["A"].dtype, "tz", None) is not None
    )
    selects = []
    funcs = [
        "RANK()",
        "DENSE_RANK()",
        "PERCENT_RANK()",
        "CUME_DIST()",
        "NTILE(4)",
        "NTILE(27)",
        "ROW_NUMBER()",
    ]
    convert_columns_bytearray = ["A"] if is_binary else None
    # Convert the spark input to tz-naive bc it can't handle timezones
    convert_columns_tz_naive = ["A"] if is_tz_aware else None
    for i, func in enumerate(funcs):
        selects.append(f"{func} OVER (PARTITION BY W2 ORDER BY {order_clause}) AS C{i}")
    query = f"SELECT A, W4, {', '.join(selects)} FROM table1"
    pandas_code = check_query(
        query,
        all_types_window_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
        convert_columns_tz_naive=convert_columns_tz_naive,
    )["pandas_code"]
    count_window_applies(pandas_code, 1, ["RANK"])
