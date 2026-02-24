from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, nullable_float_arr_maker


@pytest.fixture
def test_window_df():
    """
    Creates a DataFrame used for testing groupby.window with various
    partitions, orders and data types.

    Columns:
    A: 3 distinct strings repeated 5 times, useful for partitioning.
    B: Random integers with several nulls, useful for ordering.
    C: Random strings with some repeats, useful for ordering with a tiebreaker.
    D: 12 of one integer and 3 of another, useful for partitioning where
       one of the partitions has a somewhat larger amount of elements.
    E: Nullable boolean data.
    F: 2 distinct strings mixed with some nulls. Useful for testing
       functions where duplicates or nulls have interesting behavior.
    G: Nullable integers with some nulls and some duplicates. Useful for
       testing functions where duplicates or nulls have interesting behavior.
    H: 15 distinct integers in ascending order, useful for a deterministic
       ordering tiebreaker.
    I: Mix of unique timestamps (no timezone) and nulls.
    J: Mix of floats and nulls.
    K: Non-nullable booleans.
    L: Random strings with multiple characters and some nulls.
    M: Decimal array.
    N: Date array.
    O: Time array.
    P: Nullable uint8 array which when grouped by D will have an all-null partition.
    Q: Nullable float array which when grouped by D will have a partition containing NaNs.
    R: Nullable float array with all NA values to test all NA corner cases.
    """
    import bodo.decorators  # noqa

    return pd.DataFrame(
        {
            "A": pd.array(["A", "B", "C"] * 5, dtype=pd.ArrowDtype(pa.string())),
            "B": pd.array(
                [None if i % 5 == 0 else int(10 * np.tan(i)) for i in range(15)],
                dtype=pd.ArrowDtype(pa.int32()),
            ),
            "C": pd.array(list("ALPHATHETAGAMMA"), dtype=pd.ArrowDtype(pa.string())),
            "D": pd.array([0] * 12 + [1] * 3, dtype=pd.ArrowDtype(pa.int64())),
            "E": pd.array(
                [[True, False, None, True, False][i % 5] for i in range(15)],
                dtype=pd.ArrowDtype(pa.bool_()),
            ),
            "F": pd.array(
                [["A", "B", "A", None, "A", "B", None][i % 7] for i in range(15)],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "G": pd.array(
                [None if i % 6 == 0 else i // 4 for i in range(15)],
                dtype=pd.ArrowDtype(pa.int32()),
            ),
            "H": pd.array(list(range(15)), dtype=pd.ArrowDtype(pa.int32())),
            "I": pd.Series(
                [
                    None
                    if i % 4 == 0
                    else pd.Timestamp("1999-12-31") + pd.Timedelta(weeks=100 * i)
                    for i in range(15)
                ]
            ),
            "J": pd.array(
                [None if i % 2 == 0 else i / 2 for i in range(15)],
                dtype=pd.ArrowDtype(pa.float32()),
            ),
            "K": pd.array([True, False, False] * 5, dtype=pd.ArrowDtype(pa.bool_())),
            "L": pd.array(
                [
                    None if i % 5 == 2 else "ABCDEFGHIJKLMNOP"[i : i + i % 3 + 2]
                    for i in range(15)
                ],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "M": pd.array(
                [None if i % 7 == 5 else Decimal(2 ** (4 - i % 8)) for i in range(15)],
                dtype=pd.ArrowDtype(pa.decimal128(38, 10)),
            ),
            "N": pd.array(
                [
                    None
                    if i % 5 == 4
                    else datetime.date(1999, 12, 30)
                    + datetime.timedelta(days=5 ** (5 - i % 6))
                    for i in range(15)
                ],
                dtype=pd.ArrowDtype(pa.date32()),
            ),
            "O": pd.array(
                [
                    None if i % 8 == 4 else bodo.types.Time(microsecond=10**i)
                    for i in range(15)
                ],
                dtype=pd.ArrowDtype(pa.time64("ns")),
            ),
            "P": pd.array(
                [None if i > 10 else (i + 3) ** 2 for i in range(15)],
                dtype=pd.ArrowDtype(pa.uint8()),
            ),
            "Q": nullable_float_arr_maker(list(range(15)), [12, 14], [5, 7]),
            "R": nullable_float_arr_maker(
                list(range(15)), list(range(15)), list(range(15))
            ),
        }
    )


def permute_df_and_answer(df, answer):
    """
    Takes in an input DataFrame and an answer DataFrame and returns both with their rows
    arbitrarily permuted so that the window function tests can make sure that sorting
    & un-sorting are working correctly. Uses the length of the string of the answer
    DataFrame to seed the randomness so that it is deterministic, yet there will be
    some variation to the orderings across all of the tests.
    """
    assert len(df) == len(answer)
    rng = np.random.default_rng(len(str(answer)))
    perm = rng.permutation(np.arange(len(df)))
    return df.iloc[perm], answer.iloc[perm]


@pytest.fixture(
    params=[
        pytest.param(
            (
                ["A"],
                (("row_number",),),
                ("B",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                5,
                                4,
                                3,
                                4,
                                3,
                                5,
                                3,
                                2,
                                2,
                                2,
                                5,
                                1,
                                1,
                                1,
                                4,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="row_number-simple_ordering",
        ),
        pytest.param(
            (
                ["A"],
                (("row_number",),),
                ("C", "B"),
                (False, True),
                ("last", "first"),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                4,
                                2,
                                3,
                                3,
                                5,
                                1,
                                2,
                                4,
                                2,
                                5,
                                3,
                                4,
                                1,
                                1,
                                5,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="row_number-compound_ordering",
        ),
        pytest.param(
            (
                ["A"],
                (("min_row_number_filter",),),
                ("B",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [False] * 11 + [True] * 3 + [False],
                            dtype=pd.ArrowDtype(pa.bool_()),
                        ),
                    }
                ),
            ),
            id="min_row_number_filter",
        ),
        pytest.param(
            (
                ["A"],
                (("min_row_number_filter",),),
                ("R",),
                (False,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [True] * 3 + [False] * 12,
                            dtype=pd.ArrowDtype(pa.bool_()),
                        ),
                    }
                ),
            ),
            id="min_row_number_filter_all_NA",
        ),
        pytest.param(
            (
                ["D"],
                (("row_number",), ("min_row_number_filter",), ("row_number",)),
                ("H",),
                (True,),
                ("first",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            list(range(1, 13)) + [1, 2, 3],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [True] + [False] * 11 + [True, False, False],
                            dtype=pd.ArrowDtype(pa.bool_()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            list(range(1, 13)) + [1, 2, 3],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="multiple_row_number_calls",
        ),
        pytest.param(
            (
                ["A"],
                (("rank",), ("dense_rank",), ("percent_rank",), ("cume_dist",)),
                ("C",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                1,
                                4,
                                3,
                                3,
                                1,
                                4,
                                3,
                                2,
                                4,
                                1,
                                3,
                                1,
                                5,
                                5,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                1,
                                4,
                                2,
                                2,
                                1,
                                3,
                                2,
                                2,
                                3,
                                1,
                                3,
                                1,
                                3,
                                5,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                0.0,
                                0.75,
                                0.5,
                                0.5,
                                0.0,
                                0.75,
                                0.5,
                                0.25,
                                0.75,
                                0.0,
                                0.5,
                                0.0,
                                1.0,
                                1.0,
                                0.0,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [
                                0.4,
                                0.8,
                                0.6,
                                0.8,
                                0.2,
                                1.0,
                                0.8,
                                0.4,
                                1.0,
                                0.4,
                                0.6,
                                0.4,
                                1.0,
                                1.0,
                                0.4,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="rank_fns",
        ),
        pytest.param(
            (
                # Testing rank with a partition key so that some of the partitions
                # have only a single value (thus causing some of the rank functions
                # to use special edge cases).
                ["C"],
                (("rank",), ("dense_rank",), ("percent_rank",), ("cume_dist",)),
                ("A",),
                (True,),
                ("first",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                1,
                                1,
                                1,
                                1,
                                3,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                4,
                                1,
                                2,
                                4,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                1,
                                1,
                                1,
                                1,
                                2,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                3,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.5,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.75,
                                0.0,
                                1.0,
                                0.75,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [
                                0.4,
                                1.0,
                                1.0,
                                1.0,
                                0.6,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                0.4,
                                1.0,
                                1.0,
                                0.5,
                                1.0,
                                1.0,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="rank_fns-mixed_sized_groups",
        ),
        pytest.param(
            (
                # Testing rank with a partition key so that some of the partitions
                # are all-null.
                ["D"],
                (("rank",), ("dense_rank",), ("percent_rank",), ("cume_dist",)),
                ("P",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            list(range(1, 13)) + [1] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            list(range(1, 13)) + [1] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [i / 11 for i in range(12)] + [0] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [(i + 1) / 12 for i in range(12)] + [1] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="rank_fns-all_null_group",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # Note: row_number is included so that there is a mix of window functions that do & don't
                    # take in arguments, to verify that the offsets are used correctly.
                    ("ntile", 2),
                    ("ntile", 3),
                    ("row_number",),
                    ("ntile", 5),
                    ("ntile", 6),
                    ("ntile", 10),
                ),
                ("H",),
                (True,),
                ("first",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [1] * 6 + [2] * 6 + [1, 1, 2],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [1] * 4 + [2] * 4 + [3] * 4 + [1, 2, 3],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            list(range(1, 13)) + [1, 2, 3],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [
                                1,
                                1,
                                1,
                                2,
                                2,
                                2,
                                3,
                                3,
                                4,
                                4,
                                5,
                                5,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [
                                1,
                                1,
                                2,
                                2,
                                3,
                                3,
                                4,
                                4,
                                5,
                                5,
                                6,
                                6,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_5": pd.array(
                            [
                                1,
                                1,
                                2,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="ntile",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # Note: row_number & ntile are included so that there is a mix of window functions that
                    # take in scalar arguments, vector arguments, and no arguments.
                    ("conditional_true_event", "E"),
                    ("row_number",),
                    ("conditional_change_event", "E"),
                    ("conditional_change_event", "F"),
                    ("ntile", 6),
                    ("conditional_change_event", "G"),
                ),
                ("B", "H"),
                (True, True),
                ("first", "first"),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                1,
                                5,
                                4,
                                5,
                                5,
                                2,
                                4,
                                5,
                                4,
                                4,
                                3,
                                3,
                                0,
                                1,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                1,
                                12,
                                6,
                                9,
                                11,
                                2,
                                8,
                                10,
                                5,
                                7,
                                3,
                                4,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                0,
                                5,
                                2,
                                4,
                                5,
                                0,
                                3,
                                4,
                                2,
                                3,
                                0,
                                1,
                                0,
                                0,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [
                                0,
                                5,
                                4,
                                4,
                                4,
                                1,
                                4,
                                4,
                                3,
                                4,
                                1,
                                2,
                                0,
                                0,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [
                                1,
                                6,
                                3,
                                5,
                                6,
                                1,
                                4,
                                5,
                                3,
                                4,
                                2,
                                2,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_5": pd.array(
                            [
                                0,
                                6,
                                2,
                                4,
                                5,
                                0,
                                3,
                                5,
                                1,
                                3,
                                1,
                                1,
                                0,
                                0,
                                0,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="conditional_events",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # COUNT(*) with no frame
                    ("size", "None", "None"),
                    # COUNT on an array of nullable integers with no frame
                    ("count", "B", "None", "None"),
                    # COUNT on an array of nullable booleans with no frame
                    ("count", "E", "None", "None"),
                    # COUNT_IF on an array of nullable booleans with no frame
                    ("count_if", "E", "None", "None"),
                ),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [12] * 12 + [3] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [9] * 12 + [3] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [10] * 12 + [2] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [5] * 12 + [1] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="count_fns-no_frame-no_order",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # COUNT(*) with a sliding frame
                    ("size", -3, 3),
                    # COUNT on an array of nullable integers with a sliding frame
                    ("count", "B", -4, -2),
                    # COUNT_IF on an array of nullable booleans with a sliding frame
                    ("count_if", "E", 1, 6),
                ),
                ("B", "H"),
                (True, True),
                ("first", "first"),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                4,
                                4,
                                7,
                                7,
                                5,
                                5,
                                7,
                                6,
                                7,
                                7,
                                6,
                                7,
                                3,
                                3,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                0,
                                3,
                                1,
                                3,
                                3,
                                0,
                                3,
                                3,
                                0,
                                2,
                                0,
                                0,
                                0,
                                0,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                3,
                                0,
                                1,
                                0,
                                0,
                                2,
                                1,
                                0,
                                1,
                                1,
                                2,
                                2,
                                1,
                                0,
                                0,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="count_fns-sliding_frames",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # COUNT(*) with a prefix frame
                    ("size", "None", -1),
                    # COUNT on an array of nullable integers with a prefix frame
                    ("count", "B", "None", 0),
                    # COUNT_IF on an array of nullable booleans with a prefix frame
                    ("count_if", "E", "None", 2),
                ),
                ("B", "H"),
                (True, True),
                ("first", "first"),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                0,
                                11,
                                5,
                                8,
                                10,
                                1,
                                7,
                                9,
                                4,
                                6,
                                2,
                                3,
                                0,
                                1,
                                2,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                0,
                                9,
                                3,
                                6,
                                8,
                                0,
                                5,
                                7,
                                2,
                                4,
                                0,
                                1,
                                1,
                                2,
                                3,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                3,
                                5,
                                4,
                                5,
                                5,
                                3,
                                5,
                                5,
                                4,
                                5,
                                4,
                                4,
                                1,
                                1,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="count_fns-prefix_frames",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # COUNT(*) with a suffix frame
                    ("size", 0, "None"),
                    # COUNT on an array of nullable integers with a suffix frame
                    ("count", "B", 1, "None"),
                    # COUNT_IF on an array of nullable booleans with a suffix frame
                    ("count_if", "E", -2, "None"),
                ),
                ("B", "H"),
                (True, True),
                ("first", "first"),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                12,
                                1,
                                7,
                                4,
                                2,
                                11,
                                5,
                                3,
                                8,
                                6,
                                10,
                                9,
                                3,
                                2,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                9,
                                0,
                                6,
                                3,
                                1,
                                9,
                                4,
                                2,
                                7,
                                5,
                                9,
                                8,
                                2,
                                1,
                                0,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                5,
                                0,
                                2,
                                1,
                                1,
                                5,
                                1,
                                1,
                                3,
                                2,
                                5,
                                4,
                                1,
                                1,
                                1,
                            ],
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="count_fns-suffix_frames",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # COUNT on string array with no frame
                    ("count", "F", "None", "None"),
                    # COUNT on non-nullable integers with no frame
                    ("count", "H", "None", "None"),
                    # COUNT on timestamp (uses sentinal values as nuls) with no frame
                    ("count", "I", "None", "None"),
                    # COUNT on numpy floats (uses NaN as nuls) with no frame
                    ("count", "J", "None", "None"),
                    # COUNT on numpy boolean array with no frame
                    ("count", "K", "None", "None"),
                    # COUNT_IF on numpy boolean array with no frame
                    ("count_if", "K", "None", "None"),
                ),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [9] * 12 + [2] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [12] * 12 + [3] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [9] * 12 + [2] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [6] * 12 + [1] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [12] * 12 + [3] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_5": pd.array(
                            [4] * 12 + [1] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                    }
                ),
            ),
            id="count_fns-other_arrays",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # Sample variance on a nullable integer array, no frame
                    ("var", "B", "None", "None"),
                    # Population variance on a nullable integer array, no frame
                    ("var_pop", "B", "None", "None"),
                    # Sample standard deviation on a nullable integer array, no frame
                    ("std", "B", "None", "None"),
                    # Population standard deviation on a nullable integer array, no frame
                    ("std_pop", "B", "None", "None"),
                    # Sample variance on a nullable float array, no frame
                    ("var", "Q", "None", "None"),
                    # Population variance on a nullable float array, no frame
                    ("var_pop", "Q", "None", "None"),
                    # Sample standard deviation on a nullable float array, no frame
                    ("std", "Q", "None", "None"),
                    # Population standard deviation on a nullable float array, no frame
                    ("std_pop", "Q", "None", "None"),
                ),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [563799.694444] * 12 + [1801.333333] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [501155.283951] * 12 + [1200.888889] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [750.865963] * 12 + [42.442117] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [707.923219] * 12 + [34.653844] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_4": nullable_float_arr_maker(
                            [0.0] * 15, [12, 13, 14], list(range(12))
                        ),
                        "AGG_OUTPUT_5": nullable_float_arr_maker(
                            [0.0] * 15, [-1], list(range(12))
                        ),
                        "AGG_OUTPUT_6": nullable_float_arr_maker(
                            [0.0] * 15, [12, 13, 14], list(range(12))
                        ),
                        "AGG_OUTPUT_7": nullable_float_arr_maker(
                            [0.0] * 15, [-1], list(range(12))
                        ),
                    }
                ),
            ),
            id="var_std-no_frame",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # Sample variance on a nullable integer array, prefix frame
                    ("var", "B", "None", 0),
                    # Population variance on a nullable float array, suffix frame
                    ("var_pop", "Q", 0, "None"),
                    # Sample standard deviation on a numpy float array, sliding frame
                    ("std", "J", -1, 1),
                    # Population standard deviation on a numpy integer array, prefix frame
                    ("std_pop", "H", "None", -1),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": nullable_float_arr_maker(
                            [
                                0.0,
                                0.0,
                                648.0,
                                325.33333333333337,
                                261.3333333333333,
                                261.3333333333333,
                                197.79999999999998,
                                167.8666666666667,
                                813.4761904761905,
                                699.4107142857143,
                                699.4107142857143,
                                563799.6944444444,
                                0.0,
                                50.0,
                                1801.3333333333335,
                            ],
                            [0, 1, 12],
                            [-1],
                        ),
                        "AGG_OUTPUT_1": nullable_float_arr_maker(
                            [0.0] * 8 + [1.25, 0.6666666666666666, 0.25, 0.0, 0.0, 0.0],
                            [-1],
                            list(range(8)),
                        ),
                        "AGG_OUTPUT_2": nullable_float_arr_maker(
                            [0.70710677] * 15,
                            [0, 1, 3, 5, 7, 9, 11, 12, 13, 14],
                            [-1],
                        ),
                        "AGG_OUTPUT_3": nullable_float_arr_maker(
                            [
                                0.0,
                                0.0,
                                0.5,
                                0.816496580927726,
                                1.118033988749895,
                                1.4142135623730951,
                                1.707825127659933,
                                2.0,
                                2.29128784747792,
                                2.581988897471611,
                                2.8722813232690143,
                                3.1622776601683795,
                                0.0,
                                0.0,
                                0.5,
                            ],
                            [0, 12],
                            [-1],
                        ),
                    }
                ),
            ),
            id="var_std-with_frame",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # AVG on a nullable integer array, no frame
                    ("mean", "B", "None", "None"),
                    # AVG on a numpy integer array, no frame
                    ("mean", "H", "None", "None"),
                    # AVG on a nullable float array, no frame
                    ("mean", "Q", "None", "None"),
                    # AVG on a numpy float array, no frame
                    ("mean", "J", "None", "None"),
                    # AVG on a nullable integer array with an all-null partition, no frame
                    ("mean", "P", "None", "None"),
                ),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [-257.777778] * 12 + [23.333333] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [5.5] * 12 + [13.0] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_2": nullable_float_arr_maker(
                            [13.0] * 15, [-1], list(range(12))
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [3.0] * 12 + [6.5] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [74.0] * 12 + [None] * 3, dtype=pd.ArrowDtype(pa.float64())
                        ),
                    }
                ),
            ),
            id="avg-no_frame",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # AVG on a nullable float array, prefix frame
                    ("mean", "Q", "None", 0),
                    # AVG on a nullable float array, sliding frame
                    ("mean", "Q", 1, 4),
                    # AVG on a nullable integer array, suffix frame
                    ("mean", "B", 0, "None"),
                    # AVG on a numpy integer array, prefix frame
                    ("mean", "H", "None", -1),
                    # AVG on a numpy float array, suffix frame
                    ("mean", "J", -1, "None"),
                    # AVG on a nullable integer array with an all-null partition, sliding frame
                    ("mean", "P", -2, 1),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": nullable_float_arr_maker(
                            [0.0, 0.5, 1.0, 1.5, 2.0] + [-1.0] * 8 + [13.0, 13.0],
                            [12],
                            list(range(5, 12)),
                        ),
                        "AGG_OUTPUT_1": nullable_float_arr_maker(
                            [2.5]
                            + [-1.0] * 6
                            + [9.5, 10.0, 10.5, 11.0, -1.0, 13.0, -1.0, -1.0],
                            [11, 13, 14],
                            list(range(1, 7)),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [
                                -257.77777777777777,
                                -257.77777777777777,
                                -291.875,
                                -330.57142857142856,
                                -385.5,
                                -464.8,
                                -464.8,
                                -580.5,
                                -776.6666666666666,
                                -1131.5,
                                -2259.0,
                                -2259.0,
                                23.333333333333332,
                                38.0,
                                72.0,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [
                                None,
                                0.0,
                                0.5,
                                1.0,
                                1.5,
                                2.0,
                                2.5,
                                3.0,
                                3.5,
                                4.0,
                                4.5,
                                5.0,
                                None,
                                12.0,
                                12.5,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [
                                3.0,
                                3.0,
                                3.0,
                                3.5,
                                3.5,
                                4.0,
                                4.0,
                                4.5,
                                4.5,
                                5.0,
                                5.0,
                                5.5,
                                6.5,
                                6.5,
                                6.5,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_5": pd.array(
                            [
                                12.5,
                                16.666666666666668,
                                21.5,
                                31.5,
                                43.5,
                                57.5,
                                73.5,
                                91.5,
                                111.5,
                                133.5,
                                144.66666666666666,
                                156.5,
                                None,
                                None,
                                None,
                            ],
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="avg-with_frame",
        ),
        pytest.param(
            (
                ["D"],
                (
                    ("mean", "P", "None", "None"),
                    ("var", "P", "None", "None"),
                    ("var_pop", "P", "None", "None"),
                    ("std", "P", "None", "None"),
                    ("std_pop", "P", "None", "None"),
                ),
                ("P",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [74.000] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [2901.800000] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [2638.000000] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [53.868358059] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [51.361464154] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="moment_family-all_null",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # ANY_VALUE on a nullable integer array
                    ("any_value", "B"),
                    # ANY_VALUE on a non-nullable array of booleans
                    ("any_value", "K"),
                    # ANY_VALUE on a decimal array
                    ("any_value", "M"),
                    # ANY_VALUE on a non-nullable array of integers
                    ("any_value", "H"),
                    # ANY_VALUE on a date array
                    ("any_value", "N"),
                    # ANY_VALUE on a time array
                    ("any_value", "O"),
                    # ANY_VALUE on a string array
                    ("any_value", "L"),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [None] * 12 + [-6] * 3, dtype=pd.ArrowDtype(pa.int32())
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [True] * 15, dtype=pd.ArrowDtype(pa.bool_())
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [Decimal("16")] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.decimal128(38, 10)),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [0] * 12 + [12] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_4": pd.array(
                            [datetime.date(2008, 7, 20)] * 15,
                            dtype=pd.ArrowDtype(pa.date32()),
                        ),
                        "AGG_OUTPUT_5": pd.array(
                            [bodo.types.Time(microsecond=1)] * 12 + [None] * 3,
                            pd.ArrowDtype(pa.time64("ns")),
                        ),
                        "AGG_OUTPUT_6": pd.array(
                            ["AB"] * 12 + [None] * 3, dtype=pd.ArrowDtype(pa.string())
                        ),
                    }
                ),
            ),
            id="any_value",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # FIRST_VALUE on a non-nullable integer array with no order / frame
                    ("first", "H", "None", "None"),
                    # FIRST_VALUE on a nullable integer array with no order / frame
                    ("first", "G", "None", "None"),
                    # FIRST_VALUE on a nullable boolean with no order / frame
                    ("first", "E", "None", "None"),
                    # FIRST_VALUE on a string array with no order / frame
                    ("first", "C", "None", "None"),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [0] * 12 + [12] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [None] * 15, dtype=pd.ArrowDtype(pa.int32())
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [True] * 12 + [None] * 3, dtype=pd.ArrowDtype(pa.bool_())
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            ["A"] * 12 + ["M"] * 3,
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                    }
                ),
            ),
            id="first_value-no_frame",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # FIRST_VALUE on a non-nullable integer array with a prefix frame
                    ("first", "H", "None", -1),
                    # FIRST_VALUE on a float array with a sliding frame
                    ("first", "J", -1, 1),
                    # FIRST_VALUE on a decimal array with a suffix frame
                    ("first", "M", 0, "None"),
                    # FIRST_VALUE on a string array with a sliding frame
                    ("first", "C", -5, -2),
                ),
                ("H",),
                (True,),
                ("first",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [None] + [0] * 11 + [None] + [12, 12],
                            dtype=pd.ArrowDtype(pa.int32()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [
                                None,
                                None,
                                0.5,
                                None,
                                1.5,
                                None,
                                2.5,
                                None,
                                3.5,
                                None,
                                4.5,
                                None,
                                None,
                                None,
                                6.5,
                            ],
                            dtype=pd.ArrowDtype(pa.float32()),
                        ),
                        # This answer is identical to the input column M
                        "AGG_OUTPUT_2": pd.Series(
                            [
                                None if i % 7 == 5 else Decimal(2 ** (4 - i % 8))
                                for i in range(15)
                            ],
                            dtype=pd.ArrowDtype(pa.decimal128(38, 10)),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [None] * 2 + list("AAAALPHATH") + [None, None, "M"],
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                    }
                ),
            ),
            id="first_value-with_frame",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # LAST_VALUE on a non-nullable integer array with no order / frame
                    ("last", "H", "None", "None"),
                    # LAST_VALUE on a float array  with no order / frame
                    ("last", "J", "None", "None"),
                    # LAST_VALUE on a date array  with no order / frame
                    ("last", "N", "None", "None"),
                    # LAST_VALUE on a string array with no order / frame
                    ("last", "L", "None", "None"),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [11] * 12 + [14] * 3,
                            dtype=pd.ArrowDtype(pa.int64()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [5.5] * 12 + [None] * 3, dtype=pd.ArrowDtype(pa.float32())
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [datetime.date(1999, 12, 31)] * 12 + [None] * 3,
                            dtype=pd.ArrowDtype(pa.date32()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            ["LMNO"] * 12 + ["OP"] * 3,
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                    }
                ),
            ),
            id="last_value-no_frame",
        ),
        pytest.param(
            lambda: (
                ["D"],
                (
                    # LAST_VALUE on a nullable boolean array with a prefix frame
                    ("last", "E", "None", 0),
                    # LAST_VALUE on a decimal array with a suffix frame
                    ("last", "M", 0, "None"),
                    # LAST_VALUE on a time array  with with a sliding frame
                    ("last", "O", -5, 0),
                    # LAST_VALUE on a string array with a sliding frame
                    ("last", "L", 3, 5),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        # This answer is identical to the input column
                        "AGG_OUTPUT_0": pd.array(
                            [
                                [True, False, None, True, False][i % 5]
                                for i in range(15)
                            ],
                            dtype=pd.ArrowDtype(pa.bool_()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [Decimal("2")] * 12 + [Decimal("0.25")] * 3,
                            dtype=pd.ArrowDtype(pa.decimal128(38, 10)),
                        ),
                        # This answer is identical to the input column
                        "AGG_OUTPUT_2": pd.array(
                            [
                                None
                                if i % 8 == 4
                                else bodo.types.Time(microsecond=10**i)
                                for i in range(15)
                            ],
                            dtype=pd.ArrowDtype(pa.time64("ns")),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            ["FGHI", "GH", None, "IJKL", "JK", "KLM"]
                            + ["LMNO"] * 3
                            + [None] * 6,
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                    }
                ),
            ),
            id="last_value-with_frame",
        ),
        pytest.param(
            (
                ["D"],
                (
                    # NTH_VALUE on a string array with no order / frame
                    ("nth_value", "L", 3, "None", "None"),
                    # NTH_VALUE on a non-nullable integer array with no order / frame
                    ("nth_value", "H", 7, "None", "None"),
                    # NTH_VALUE on a nullable boolean array with no order / frame
                    ("nth_value", "E", 2, "None", "None"),
                    # NTH_VALUE on a date array with no order / frame
                    ("nth_value", "N", 5, "None", "None"),
                ),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [None] * 12 + ["OP"] * 3,
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [6] * 12 + [None] * 3, dtype=pd.ArrowDtype(pa.int32())
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [False] * 12 + [True] * 3, dtype=pd.ArrowDtype(pa.bool_())
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [None] * 15,
                            dtype=pd.ArrowDtype(pa.date32()),
                        ),
                    }
                ),
            ),
            id="nth_value-no_frame",
            marks=pytest.mark.skip(
                reason="[BSE-903] TODO: support nth_value in groupby.window"
            ),
        ),
        pytest.param(
            (
                ["D"],
                (
                    # NTH_VALUE on a string array with a prefix frame
                    ("nth_value", "L", 2, "None", 0),
                    # NTH_VALUE on a nullable integer array with a suffix frame
                    ("nth_value", "B", 5, 0, "None"),
                    # NTH_VALUE on a string array with a sliding frame
                    ("nth_value", "C", 5, -4, 0),
                    # NTH_VALUE on a float array with a sliding frame
                    ("nth_value", "N", 25, 1, 50),
                ),
                ("H",),
                (True,),
                ("last",),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [None] + ["BCD"] * 11 + [None, "NOP", "NOP"],
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                        "AGG_OUTPUT_1": pd.array(
                            [11, None, -2, 8, -67, -4, None, -2259] + [None] * 7,
                            dtype=pd.ArrowDtype(pa.int32()),
                        ),
                        "AGG_OUTPUT_2": pd.array(
                            [None] * 4
                            + ["A", "T", "H", "E", "T", "A", "G", "A"]
                            + [None] * 3,
                            dtype=pd.ArrowDtype(pa.string()),
                        ),
                        "AGG_OUTPUT_3": pd.array(
                            [None] * 15,
                            dtype=pd.ArrowDtype(pa.date32()),
                        ),
                    }
                ),
            ),
            id="nth_value-with_frame",
            marks=pytest.mark.skip(
                reason="[BSE-903] TODO: support nth_value in groupby.window"
            ),
        ),
        pytest.param(
            (
                ["D"],
                (("ratio_to_report", "P"),),
                (),
                (),
                (),
                pd.DataFrame(
                    {
                        "AGG_OUTPUT_0": pd.array(
                            [
                                0.011057,
                                0.019656,
                                0.030713,
                                0.044226,
                                0.060197,
                                0.078624,
                                0.099509,
                                0.122850,
                                0.148649,
                                0.176904,
                                0.207617,
                            ]
                            + [None] * 4,
                            dtype=pd.ArrowDtype(pa.float64()),
                        ),
                    }
                ),
            ),
            id="ratio_to_report",
        ),
    ],
)
def window_args(request):
    """
    Returns the arguments for a test of groupby.window as a tuple of the following:
    - keys: the tuple of column names from test_window_df to group by.
    - funcs: a tuple of tuples where each inner tuple contains a window funciton name
            followed by any scalar/column arguments.
    - orderby: a tuple of column names from test_window_df to order by within each group.
    - ascending: a tuple of booleans indicating which columns from orderby to sort in ascending
                vs descending order.
    - napos: a tuple of strings indicating which columns from orderby to place in nulls at the
            begining vs the end when sorting.
    - answer: the expected result of the call to groupby.window.
    """
    import bodo.decorators  # noqa

    # evaluates arguments potentially lazily to avoid importing the compiler at test collection time.
    val = request.param

    return val() if callable(val) else val


def test_window(test_window_df, window_args, memory_leak_check):
    """
    Tests using groupby.window on various aggregation types, sometimes
    multiple at once
    """
    keys, funcs, orderby, ascending, napos, answer = window_args

    func_text = "def impl(df):\n"
    func_text += f"   return df.groupby({keys}, as_index = False, dropna = False, _is_bodosql = True)"
    func_text += f".window({funcs}, {orderby}, {ascending}, {napos})"
    local_vars = {}
    exec(func_text, local_vars)
    impl = local_vars["impl"]

    # Shuffle the order of the rows and the answer (can disable during local testing
    # for convenience of debugging)
    # Avoid shuffling for min_row_number_filter_all_NA test since answer won't be
    # consistent
    if not (funcs == (("min_row_number_filter",),) and orderby == ("R",)):
        test_window_df, answer = permute_df_and_answer(test_window_df, answer)

    check_func(
        impl,
        (test_window_df,),
        py_output=answer,
        sort_output=False,
        reset_index=True,
        check_names=False,
        check_dtype=False,
        atol=1e-6,
    )
