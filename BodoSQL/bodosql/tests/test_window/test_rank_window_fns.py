import datetime

import numpy as np
import pandas as pd
import pytest

import bodo

# Skip unless any window-related files were changed
from bodo.tests.utils import pytest_slow_unless_window
from bodosql.tests.test_window.window_common import (  # noqa
    all_types_window_df,
    all_window_df,
    count_window_applies,
)
from bodosql.tests.utils import check_query

pytestmark = pytest_slow_unless_window


@pytest.mark.parametrize(
    "orderby_multiple_columns",
    [
        pytest.param(False, id="single_column"),
        pytest.param(True, id="multi_column", marks=pytest.mark.slow),
    ],
)
def test_row_number_orderby(datapath, memory_leak_check, orderby_multiple_columns):
    """Test that row_number properly handles the orderby."""
    # Note we test multiple columns to confirm we still use the C++
    # infrastructure, not for checking direct correctness. The order
    # is entirely determined by last_seen
    # TODO: Verify the C++ path is used (this is tested as a Codegen unit test)
    if orderby_multiple_columns:
        orderby_cols = '"in_stock" ASC, "last_seen" DESC'
    else:
        orderby_cols = '"last_seen" DESC'

    query = f'select "uuid", ROW_NUMBER() OVER(PARTITION BY "store_id", "ret_product_id" ORDER BY {orderby_cols}) as row_num from table1'

    parquet_path = datapath("sample-parquet-data/rphd_sample.pq")

    ctx = {
        "TABLE1": pd.read_parquet(parquet_path)[
            ["uuid", "store_id", "ret_product_id", "in_stock", "last_seen"]
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
            "ROW_NUM": [3, 5, 1, 6, 4, 2],
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
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
    is_binary = type(all_types_window_df["TABLE1"]["A"].iloc[0]) == bytes
    is_tz_aware = (
        getattr(all_types_window_df["TABLE1"]["A"].dtype, "tz", None) is not None
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
    count_window_applies(pandas_code, 0, ["RANK"])


@pytest.mark.parametrize(
    "input_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(8),
                    "B": ["A", "B", "C", "B", "C", "B", "B", "C"],
                    "C": pd.Series(
                        [1.1, -1.2, 0.9, -1000.0, 1.1, None, 0.0, 1.4], dtype="Float64"
                    ),
                }
            ),
            id="float64_orderby",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.arange(8),
                    "B": ["A", "B", "C", "B", "C", "B", "B", "C"],
                    "C": ["k", "b", "gdge", "a", "k", None, "e", "zed"],
                }
            ),
            id="string_orderby",
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize(
    "ascending",
    [
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "nulls_last",
    [
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ],
)
def test_min_row_number_filter_simple(
    memory_leak_check, input_df, ascending, nulls_last
):
    """
    Tests queries involving `where row_number() = 1`. This query
    will generate a special filter function that ensures the ROW_NUMBER()
    can compute just a max/min and skip sorting each group.

    This function tests for various input combinations, including
    testing ascending/descending and nulls first/last.
    """
    ascending_str = "ASC" if ascending else "DESC"
    nulls_last_str = "NULLS LAST" if nulls_last else "NULLS FIRST"
    query = f"""
    SELECT
        A
    FROM
        (
            SELECT
                A,
                ROW_NUMBER() OVER(PARTITION BY B ORDER BY C {ascending_str} {nulls_last_str}) as rn
            FROM table1
        )
    WHERE rn = 1
    """
    ctx = {"TABLE1": input_df}
    if ascending:
        if nulls_last:
            py_output = pd.DataFrame({"A": [0, 3, 2]})
        else:
            py_output = pd.DataFrame({"A": [0, 5, 2]})
    else:
        if nulls_last:
            py_output = pd.DataFrame({"A": [0, 6, 7]})
        else:
            py_output = pd.DataFrame({"A": [0, 5, 7]})
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


def test_min_row_number_filter_complex(memory_leak_check, spark_info):
    """
    A variant of test_min_row_number_filter_simple with a more complex
    min row number filter with more involved column pruning behavior,
    more pass-through columns, and more tie-checking between various
    orderby columns.
    """
    query = """
    SELECT A, B, D, E, G, H
    FROM TABLE1
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY A, I, G
        ORDER BY D ASC NULLS FIRST, B ASC NULLS LAST, F DESC NULLS FIRST, H DESC NULLS LAST) = 1
    """
    spark_query = """
    SELECT A, B, D, E, G, H
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY A, I, G
            ORDER BY D ASC NULLS FIRST, B ASC NULLS LAST, F DESC NULLS FIRST, H DESC NULLS LAST) as rn
        FROM table1
    )
    WHERE rn = 1
    """
    df = pd.DataFrame(
        {
            "A": pd.array([str(i)[0] for i in range(2000)], dtype=pd.Int64Dtype()),
            "B": pd.array(
                [["Alpha", "Alpha", "Beta", "", None][i % 5] for i in range(2000)]
            ),
            "C": pd.array([i for i in range(2000)], dtype=pd.Int64Dtype()),
            "D": pd.array(
                [[1, -1, None][i % 3] for i in range(2000)], dtype=pd.Int64Dtype()
            ),
            "E": pd.array([str(i) for i in range(2000)], dtype=pd.Int64Dtype()),
            "F": pd.array(
                [None if i % 10 < 9 else i**2 for i in range(2000)],
                dtype=pd.Int64Dtype(),
            ),
            "G": pd.array(
                [round(i**0.5) ** 2 == i for i in range(2000)],
                dtype=pd.BooleanDtype(),
            ),
            "H": pd.array([-i for i in range(2000)], dtype=pd.Int64Dtype()),
            "I": pd.array(
                [min(i % 6, i % 7, i % 11) for i in range(2000)], dtype=pd.Int64Dtype()
            ),
            "J": pd.array([i for i in range(2000)], dtype=pd.Int64Dtype()),
        }
    )
    ctx = {"TABLE1": df}
    check_query(
        query, ctx, spark_info, equivalent_spark_query=spark_query, check_dtype=False
    )


@pytest.mark.parametrize(
    "partition_cols, order_cols",
    [
        pytest.param(["A", "F"], ["NULL"], id="no_order"),
        pytest.param(
            ["G", "A", "E"],
            ["C ASC", "J DESC", "E ASC", "A DESC", "F ASC"],
            id="order_partial_overlap",
        ),
        pytest.param(["B", "D", "H"], ["H DESC", "D ASC"], id="order_total_overlap"),
    ],
)
def test_mrnf_order_edgecases(
    partition_cols, order_cols, spark_info, memory_leak_check
):
    """
    Test that the MRNF output is correct when some or all of the orderby columns
    are also partition columns, and when there are no orderby columns.
    """
    df = pd.DataFrame(
        {
            "A": np.arange(50) % 2,
            "B": np.arange(50) // 5,
            "C": np.arange(50) % 3,
            "D": (np.arange(50) ** 2) % 10,
            "E": [str(i)[0] for i in range(50)],
            "F": np.sin(np.arange(50)),
            "G": np.arange(50) % 4,
            "H": np.arange(50) % 5,
            "I": np.arange(50),
            "J": np.arange(50) % 7,
        }
    )
    # Shuffle the rows to ensure there are no coincidentally-correct answers
    perm = np.random.default_rng(42).permutation(50)
    df = df.iloc[perm]
    query = f"""
    SELECT I
    FROM
        (   
            SELECT
                I,
                ROW_NUMBER() OVER(
                    PARTITION BY {', '.join(partition_cols)} 
                    ORDER BY {', '.join(order_cols)}) as rn
            FROM table1
        )
    WHERE rn = 1
    """

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by=["C"], ascending=True, na_position="first").iloc[0]

    expected_df = df.groupby(["B"], as_index=False, dropna=False).apply(py_mrnf)
    ctx = {"TABLE1": df}
    check_query(query, ctx, spark_info, check_dtype=False, check_names=False)


def test_mrnf_all_ties(memory_leak_check):
    """
    Test that the MRNF output is correct when all the values in the order-by
    column are the same.
    """
    df = pd.DataFrame(
        {
            "B": pd.array(list(np.arange(50)) * 2, dtype="Int32"),
            "C": pd.Series([True] * 100, dtype="bool"),
        }
    )
    query = f"""
    SELECT
        B, C
    FROM
        (   
            SELECT
                B, C,
                ROW_NUMBER() OVER(PARTITION BY B ORDER BY C ASC NULLS FIRST) as rn
            FROM table1
        )
    WHERE rn = 1
    """

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by=["C"], ascending=True, na_position="first").iloc[0]

    expected_df = df.groupby(["B"], as_index=False, dropna=False).apply(py_mrnf)
    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_df,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "input_arrs",
    [
        pytest.param(
            (
                pd.array(
                    # Resolved by A
                    [2, 1, -14, 2, 3, None, None, 4]
                    # Resolved by B
                    + [5, 5, -42, -42, 6, 6, None, None]
                    # Resolved by C
                    + [5, 5, -42, -42, 6, 6, None, None]
                    # Resolved by D
                    + [5, 5, -42, -42, 6, 6, None, None],
                    dtype="Int64",
                ),
                (
                    # Resolved by A
                    ["", "", "c", "a", "e", "e", "f", "b"]
                    # Resolved by B
                    + [
                        "a",
                        "b",
                        "ewq1rw1q",
                        "aiohwoihefhiowiohfweiofo",
                        "c",
                        None,
                        None,
                        "Qre2r",
                    ]
                    # Resolved by C
                    + [None, None, "a", "a", "3424", "3424", "df2", "df2"]
                    # Resolved by D
                    + [None, None, "a", "a", "3424", "3424", "df2", "df2"]
                ),
                pd.array(
                    # Resolved by A or B
                    [1.0] * 16
                    # Resolved by C
                    + [2.2, 1.3, -1.4, 213.0, None, 42.1, 32131.243214, None]
                    # Resolved by D
                    + [31.2, 31.2, None, None, -1, -1, 0, 0],
                    dtype="Float64",
                ),
                pd.array(
                    # Resolved by A or B or C
                    [False] * 24
                    # Resolved by D
                    + [True, False, False, True, None, True, False, None],
                    dtype="boolean",
                ),
            ),
            id="int64_string_float64_boolean",
        ),
        pytest.param(
            (
                (
                    # Resolved by A
                    [
                        datetime.date(2023, 1, 2),
                        datetime.date(2023, 1, 1),
                        datetime.date(2022, 12, 25),
                        datetime.date(2023, 1, 2),
                        datetime.date(2023, 1, 3),
                        None,
                        None,
                        datetime.date(2023, 1, 4),
                    ]
                    # Resolved by B
                    + [
                        datetime.date(2024, 1, 1),
                        datetime.date(2024, 1, 1),
                        datetime.date(2019, 11, 15),
                        datetime.date(2019, 11, 15),
                        datetime.date(2023, 8, 25),
                        datetime.date(2023, 8, 25),
                        None,
                        None,
                    ]
                    # Resolved by C
                    + [
                        datetime.date(2024, 1, 1),
                        datetime.date(2024, 1, 1),
                        datetime.date(2019, 11, 15),
                        datetime.date(2019, 11, 15),
                        datetime.date(2023, 8, 25),
                        datetime.date(2023, 8, 25),
                        None,
                        None,
                    ]
                    # Resolved by D
                    + [
                        datetime.date(2024, 1, 1),
                        datetime.date(2024, 1, 1),
                        datetime.date(2019, 11, 15),
                        datetime.date(2019, 11, 15),
                        datetime.date(2023, 8, 25),
                        datetime.date(2023, 8, 25),
                        None,
                        None,
                    ]
                ),
                (
                    # Resolved by A
                    [
                        bodo.Time(1, 1, 1, 1),
                        bodo.Time(1, 1, 1, 1),
                        bodo.Time(23),
                        bodo.Time(1),
                        bodo.Time(2),
                        bodo.Time(2),
                        bodo.Time(11, 10, 5),
                        bodo.Time(11),
                    ]
                    # Resolved by B
                    + [
                        bodo.Time(11),
                        bodo.Time(11, 10, 5),
                        bodo.Time(1, 1, 1, 1, 1, 1),
                        bodo.Time(1, 1, 1, 1, 1),
                        bodo.Time(11, 21),
                        None,
                        None,
                        bodo.Time(14),
                    ]
                    # Resolved by C
                    + [
                        None,
                        None,
                        bodo.Time(),
                        bodo.Time(),
                        bodo.Time(1),
                        bodo.Time(1),
                        bodo.Time(10, 5),
                        bodo.Time(10, 5),
                    ]
                    # Resolved by D
                    + [
                        None,
                        None,
                        bodo.Time(),
                        bodo.Time(),
                        bodo.Time(10, 5, 11),
                        bodo.Time(10, 5, 11),
                        bodo.Time(10, 5),
                        bodo.Time(10, 5),
                    ]
                ),
                pd.Series(
                    # Resolved by A or B
                    [pd.Timestamp(year=2024, month=1, day=1)] * 16
                    # Resolved by C
                    + [
                        pd.Timestamp(year=2024, month=1, day=19, minute=1),
                        pd.Timestamp(year=2024, month=1, day=19, second=1),
                        pd.Timestamp(year=2024, month=1, day=10),
                        pd.Timestamp(year=2024, month=1, day=19),
                        None,
                        pd.Timestamp(year=2023, month=9, day=13),
                        pd.Timestamp(year=2023, month=12, day=12),
                        None,
                    ]
                    # Resolved by D
                    + [
                        pd.Timestamp(year=2024, month=1, day=1, second=5),
                        pd.Timestamp(year=2024, month=1, day=1, second=5),
                        None,
                        None,
                        pd.Timestamp(year=2024, month=1, day=1, microsecond=5),
                        pd.Timestamp(year=2024, month=1, day=1, microsecond=5),
                        None,
                        None,
                    ],
                ).values,
                pd.array(
                    # Resolved by A or B or C
                    [0.0] * 24
                    # Resolved by D
                    + [43243, 43242, -1000.1, -1000, None, 111.42, -12312.431, None],
                    dtype="Float32",
                ),
            ),
            id="date_time_datetime64_float32",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_row_number_filter_multicolumn(input_arrs, memory_leak_check):
    """
    Tests a query with multiple orderby columns involving `where row_number() = 1`. This query
    will generate a special filter function that ensures the ROW_NUMBER()
    can compute just a max/min and skip sorting each group.

    This function tests for various input combinations, including
    testing ascending/descending and nulls first/last.
    """
    query = f"""
    SELECT
        F
    FROM
        (
            SELECT
                F,
                ROW_NUMBER() OVER(PARTITION BY E ORDER BY A ASC NULLS FIRST, B ASC NULLS LAST, C DESC NULLS FIRST, D DESC NULLS LAST) as rn
            FROM table1
        )
    WHERE rn = 1
    """
    a_arr, b_arr, c_arr, d_arr = input_arrs
    # Note we test all nullable arrays since we want to test the nulls first/last.
    # To do this we need 1 group for each decision being made.
    input_df = pd.DataFrame(
        {
            "A": a_arr,
            "B": b_arr,
            "C": c_arr,
            "D": d_arr,
            "E": [str(i // 2) for i in range(32)],
            "F": np.arange(32),
        }
    )

    ctx = {"TABLE1": input_df}
    py_output = pd.DataFrame(
        {"F": [1, 2, 5, 6, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 29, 30]}
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )