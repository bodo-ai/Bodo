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
        "TABLE1": pd.read_parquet(parquet_path, dtype_backend="pyarrow")[
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
    is_binary = isinstance(all_types_window_df["TABLE1"]["A"].iloc[0], bytes)
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
    convert_columns_tz_naive = None
    for i, func in enumerate(funcs):
        selects.append(f"{func} OVER (PARTITION BY W2 ORDER BY {order_clause}) AS C{i}")
    query = f"SELECT W2, A, {', '.join(selects)} FROM table1"
    check_query(
        query,
        all_types_window_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
        convert_columns_tz_naive=convert_columns_tz_naive,
        use_duckdb=True,
    )


def test_rank_ntile_mix(spark_info, memory_leak_check):
    """
    Tests a mix of window functions that use the hash-based streaming window
    impl together.
    """

    window = " OVER (PARTITION BY P ORDER BY O)"
    window_terms = [
        "RANK()",
        "NTILE(4)",
        "DENSE_RANK()",
        "NTILE(75)",
        "NTILE(126)",
        "NTILE(7)",
        "NTILE(500)",
        "PERCENT_RANK()",
        "NTILE(13)",
        "NTILE(2)",
    ]
    query = f"SELECT IDX, O, {', '.join([term + window for term in window_terms])} FROM TABLE1"
    n_rows = 10000
    df = pd.DataFrame(
        {
            "P": [int(i**0.25 + np.tan(i)) for i in range(n_rows)],
            "O": [np.tan(i) for i in range(n_rows)],
            "IDX": range(n_rows),
        }
    )

    check_query(
        query,
        {"TABLE1": df},
        spark_info,
        check_dtype=False,
        check_names=False,
    )


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
            "C": pd.array(list(range(2000)), dtype=pd.Int64Dtype()),
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
            "J": pd.array(list(range(2000)), dtype=pd.Int64Dtype()),
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
                    PARTITION BY {", ".join(partition_cols)}
                    ORDER BY {", ".join(order_cols)}) as rn
            FROM table1
        )
    WHERE rn = 1
    """

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by=["C"], ascending=True, na_position="first").iloc[0]

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
    query = """
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
                        bodo.types.Time(1, 1, 1, 1),
                        bodo.types.Time(1, 1, 1, 1),
                        bodo.types.Time(23),
                        bodo.types.Time(1),
                        bodo.types.Time(2),
                        bodo.types.Time(2),
                        bodo.types.Time(11, 10, 5),
                        bodo.types.Time(11),
                    ]
                    # Resolved by B
                    + [
                        bodo.types.Time(11),
                        bodo.types.Time(11, 10, 5),
                        bodo.types.Time(1, 1, 1, 1, 1, 1),
                        bodo.types.Time(1, 1, 1, 1, 1),
                        bodo.types.Time(11, 21),
                        None,
                        None,
                        bodo.types.Time(14),
                    ]
                    # Resolved by C
                    + [
                        None,
                        None,
                        bodo.types.Time(),
                        bodo.types.Time(),
                        bodo.types.Time(1),
                        bodo.types.Time(1),
                        bodo.types.Time(10, 5),
                        bodo.types.Time(10, 5),
                    ]
                    # Resolved by D
                    + [
                        None,
                        None,
                        bodo.types.Time(),
                        bodo.types.Time(),
                        bodo.types.Time(10, 5, 11),
                        bodo.types.Time(10, 5, 11),
                        bodo.types.Time(10, 5),
                        bodo.types.Time(10, 5),
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
                    dtype="datetime64[ns]",
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
    query = """
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


def test_dense_rank_stress_test(datapath, memory_leak_check):
    """
    A more complex test for streaming window functions focused on the correctness
    of dense rank in a larger data context. The specifics of the test:

    Takes in a database of words, transformed into a table where each row is
    one of the words with one of the letters in it (1 row per letter in the word) and
    another table mapping each letter of the alphabet to a scrabble score. Calculates
    the scrabble scores for every word and for each word length finds all words that
    have the highest score with that length.
    """

    query = """
    SELECT 
        LENGTH(words.word) as n_chars,
        words.word,
    FROM scrabble, words
    WHERE scrabble.letter = words.letter
    GROUP BY words.word
    QUALIFY dense_rank() OVER (PARTITION BY n_chars ORDER BY sum(scrabble.score) DESC) = 1
    """

    words = []
    letters = []
    jw_data = pd.read_csv(datapath("jaro_winkler_data.csv"))
    seen = set()
    for word in jw_data["A"]:
        word = word.upper().replace(" ", "").replace(".", "")
        if word in seen:
            continue
        seen.add(word)
        for letter in word:
            words.append(word)
            letters.append(letter)
    ctx = {
        "SCRABBLE": pd.DataFrame(
            {
                "LETTER": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                "SCORE": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] * 2,
            }
        ),
        "WORDS": pd.DataFrame(
            {
                "WORD": words,
                "LETTER": letters,
            }
        ),
    }
    expected_df = pd.DataFrame(
        {
            "word_length": [
                10,
                11,
                11,
                12,
                12,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                18,
                19,
                20,
                20,
                21,
                22,
                23,
                23,
                24,
                25,
                25,
                26,
                27,
                28,
                29,
            ],
            "word": [
                "LCHTRDBLDM",
                "DMCRMLVZRPL",
                "DMZRPLMLVTN",
                "SMKLVBLLMNVJ",
                "LCZRKHKNVYLM",
                "SKYLMLCDMGRY",
                "YLLWLMPLLVVRY",
                "MDMLVLMVRYFLRL",
                "WHTSKYYLLWLMWHT",
                "RYLKHKVLTYLLWVRY",
                "YLLWPLKHKMTLLCRYL",
                "THSTLSMKPLMYLLWSKY",
                "LMSSHLLMGNTYLLWSMK",
                "SSHLLLMMDMYLLWMTLLC",
                "NVJTHSTLSSHLLYLLWKHK",
                "VLTPLMMTLLCYLLWLVNDR",
                "FLRLLMBRLYWDYLLWTHSTL",
                "SSHLLFLRLSMKMDNGHTYLLW",
                "YLLWBRNSHDMTLLCSSHLLKHK",
                "BRLYWDMTLLCSLMNQMRNYLLW",
                "LMNDTMTCRNFLWRMTLLCSSHLL",
                "YLLWFRSTDPLMBRLYWDCRNFLWR",
                "SSHLLTHSTLCHCLTKHKCRNFLWR",
                "SLMNGLDNRDCRNFLWRYLLWSSHLL",
                "CRNFLWRYLLWGLDNRDBRLYWDBLCK",
                "BRLYWDGLDNRDPWDRMTLLCCRNFLWR",
                "FRSTDCHRTRSBLNCHDCRNSLKBRLYWD",
            ],
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_df,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "window_func, expected_df",
    [
        pytest.param(
            "ROW_NUMBER()",
            pd.DataFrame(
                {
                    "IDX": list(range(25)),
                    "VAL": list(range(1, 26)),
                }
            ),
            id="row_number",
        ),
        pytest.param(
            "RANK()",
            pd.DataFrame(
                {"IDX": list(range(25)), "VAL": [1 + 5 * (i // 5) for i in range(25)]}
            ),
            id="rank",
        ),
        pytest.param(
            "DENSE_RANK()",
            pd.DataFrame(
                {"IDX": list(range(25)), "VAL": [1 + (i // 5) for i in range(25)]}
            ),
            id="dense_rank",
        ),
        pytest.param(
            "PERCENT_RANK()",
            pd.DataFrame(
                {
                    "IDX": list(range(25)),
                    "VAL": [(5 * (i // 5)) / 24 for i in range(25)],
                }
            ),
            id="percent_rank",
            marks=pytest.mark.skip(
                reason="Blocked until multiple window functions are allowed in streaming together"
            ),
        ),
        pytest.param(
            "CUME_DIST()",
            pd.DataFrame(
                {
                    "IDX": list(range(25)),
                    "VAL": [5 * (1 + (i // 5)) / 25 for i in range(25)],
                }
            ),
            id="cume_dist",
        ),
    ],
)
def test_partitionless_rank_fns(capfd, window_func, expected_df, memory_leak_check):
    """
    Tests executing the RANK family of functions with the streaming
    window code when there are no partition columns.
    """
    query = f"SELECT IDX, {window_func} OVER (ORDER BY O1) AS VAL FROM TABLE1"

    in_df = pd.DataFrame(
        {
            "IDX": range(25),
            "O1": [i // 5 for i in range(25)],
        }
    )
    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    comm = MPI.COMM_WORLD

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": in_df},
            None,
            expected_output=expected_df,
            check_dtype=False,
            check_names=False,
        )
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "window_func, expected_df, expected_log_message",
    [
        pytest.param(
            "ROW_NUMBER()",
            pd.DataFrame(
                {
                    "window_out": np.concatenate(
                        (
                            np.arange(1, 5),
                            np.arange(1, 101),
                            np.arange(1, 101),
                            np.array([1]),
                        )
                    )
                }
            ),
            "[DEBUG] WindowState::FinalizeBuild: Finished",
            id="row_number",
        ),
        pytest.param(
            "RANK()",
            pd.DataFrame(
                {
                    "window_out": np.concatenate(
                        (
                            np.array([1, 2, 2, 4]),
                            np.ones(100),
                            np.arange(1, 101),
                            np.array([1]),
                        )
                    )
                }
            ),
            "[DEBUG] WindowState::FinalizeBuild: Finished",
            id="rank",
        ),
        pytest.param(
            "DENSE_RANK()",
            pd.DataFrame(
                {
                    "window_out": np.concatenate(
                        (
                            np.array([1, 2, 2, 3]),
                            np.ones(100),
                            np.arange(1, 101),
                            np.array([1]),
                        )
                    )
                }
            ),
            "[DEBUG] WindowState::FinalizeBuild: Finished",
            id="dense_rank",
        ),
        pytest.param(
            "PERCENT_RANK()",
            pd.DataFrame(
                {
                    "window_out": np.concatenate(
                        (
                            np.array([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]),
                            np.zeros(100, dtype="float64"),
                            (np.arange(1, 101) - 1.0) / 99.0,
                            np.array([0.0]),
                        )
                    )
                }
            ),
            "[DEBUG] WindowState::FinalizeBuild: Finished",
            id="percent_rank",
            marks=pytest.mark.skip(
                reason="Blocked until multiple window functions are allowed in streaming together"
            ),
        ),
        pytest.param(
            "CUME_DIST()",
            pd.DataFrame(
                {
                    "window_out": np.concatenate(
                        (
                            np.array([0.25, 0.75, 0.75, 1.0]),
                            np.ones(100, dtype="float64"),
                            (np.arange(1, 101)) / 100.0,
                            np.array([1.0]),
                        )
                    )
                }
            ),
            "[DEBUG] WindowState::FinalizeBuild: Finished",
            id="cume_dist",
        ),
    ],
)
def test_rank_fns_sort_path_taken(
    capfd, window_func, expected_df, expected_log_message, memory_leak_check
):
    """verifies sort path is taken in row_number()"""

    from mpi4py import MPI

    import bodosql
    from bodo.tests.utils import temp_env_override

    comm = MPI.COMM_WORLD

    test_df = pd.DataFrame(
        {
            "ID": np.arange(205),
            "A": np.concatenate(
                (
                    np.array([1, 2, 2, 3]),
                    np.ones(100),
                    np.arange(100),
                    np.array(
                        [
                            1,
                        ]
                    ),
                )
            ),
            "B": [0] * 4 + [1] * 100 + [2] * 100 + [3],
        }
    )

    seed = 42
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.arange(len(test_df)))
    test_df = test_df.iloc[perm]

    ctx = {"TABLE1": test_df}
    bc = bodosql.BodoSQLContext(ctx)
    query = f"""
SELECT {window_func} OVER (PARTITION BY B ORDER BY A) FROM TABLE1
"""

    @bodo.jit()
    def impl(bc, query):
        bc.sql(query)

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        impl(bc, query)

    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_df,
        check_dtype=False,
        check_names=False,
    )

    assert assert_success


@pytest.mark.parametrize(
    "func",
    [
        pytest.param("ROW_NUMBER", id="row_number"),
        pytest.param("RANK", id="rank", marks=pytest.mark.slow),
        pytest.param("DENSE_RANK", id="dense_rank", marks=pytest.mark.slow),
    ],
)
def test_row_number_intense(func, spark_info, memory_leak_check):
    """
    Tests ROW_NUMBER on a larger set of data with a lot of skew between the partition sizes.
    """
    query = f"SELECT P1, P2, O1, O2, {func}() OVER (PARTITION BY P1, P2 ORDER BY O1 ASC NULLS LAST, O2 ASC NULLS LAST) FROM TABLE1"
    n_rows = 1_000_000
    p1 = []
    p2 = []
    o1 = []
    o2 = []
    strings = ["A", "L", "P", "A", "B", "E", "T", None]
    for i in range(n_rows):
        p1.append(i % 2)
        p2.append(np.int64(np.log10(i + 1)))
        o1.append(strings[i % 8])
        o2.append(int(np.tan(i) // 3))

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P1": pd.array(p1, dtype=pd.Int32Dtype()),
                "P2": pd.array(p2, dtype=pd.Int32Dtype()),
                "O1": o1,
                "O2": o2,
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.fixture(scope="session")
def ntile_df():
    df = pd.DataFrame(
        {
            "ID": list(range(10010)),
            "A": [1] * 5005 + [2] * 5005,
            "B": list(range(5004)) + [None] + list(range(5004)) + [None],
            "C": ["socks", "shoes", None, "shirt", None] * 2002,
        }
    )

    # Randomize the order of the input data
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    df = df.iloc[perm, :]
    return df


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "n_bins",
    [
        pytest.param(
            1,
            id="nbins_1",
        ),
        pytest.param(
            3,
            id="nbins_3",
        ),
        pytest.param(
            17,
            id="nbins_17",
        ),
        pytest.param(
            255,
            id="nbins_255",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            1026,
            id="nbins_1026",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            5005,
            id="nbins_5005",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            10010,
            id="nbins_10010",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_ntile(ntile_df, capfd, spark_info, n_bins):
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    comm = MPI.COMM_WORLD

    query = f"SELECT ID, NTILE({n_bins}) OVER (PARTITION BY A ORDER BY B ASC NULLS LAST) FROM TABLE1"
    expected_log_message = "[DEBUG] GroupbyState::FinalizeBuild:"

    # checks that we are using stream groupby
    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": ntile_df},
            spark_info,
            check_names=False,
            check_dtype=False,
        )
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
def test_multiple_rank_fns(spark_info, capfd, memory_leak_check):
    """
    Tests that multiple rank functions can be computed together in the sort based impl
    """
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    comm = MPI.COMM_WORLD

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    window = " OVER (PARTITION BY P ORDER BY O)"
    window_terms = ["RANK()", "DENSE_RANK()"]
    query = f"SELECT IDX, O, {', '.join([term + window for term in window_terms])} FROM TABLE1"
    n_rows = 10000
    df = pd.DataFrame(
        {
            "P": [int(i**0.25 + np.tan(i)) for i in range(n_rows)],
            "O": [int(np.tan(i)) for i in range(n_rows)],
            "IDX": range(n_rows),
        }
    )

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_dtype=False,
            check_names=False,
        )
        _, err = capfd.readouterr()
        assert_success = expected_log_message in err
        assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success
