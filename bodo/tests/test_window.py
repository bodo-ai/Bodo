# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture
def test_window_df():
    """
    Creates a DataFrame used for testing groupby.window with various
    partitions, orders and data types.

    Columns:
    A: 3 distinct strings repeated 5 times, useful for partitioning.
    B: Random integers with several nulls, useful for ordering.
    C: Randm strings with some repeats, useful for ordering with a tiebreaker.
    D: 12 of one integer and 3 of another, useful for partitioning where
       one of the partitions has a somewhat larger amount of elements.
    E: Nullable boolean data.
    F: 2 distinct strings mixed with some nulls. Useful for testing
       functions where duplicates or nulls have interesting behavior.
    G: Nullable integers with some nulls and some duplicates. Useful for
       testing functions where duplicates or nulls have interesting behavior.
    H: 15 distinct integers in ascending order.
    I: Mix of identical timestamps (no timezone) and nulls.
    J: Mix of floats and nulls.
    K: Non-nullable booleans.
    """
    return pd.DataFrame(
        {
            "A": pd.Series(["A", "B", "C"] * 5),
            "B": pd.Series(
                [None if i % 5 == 0 else int(10 * np.tan(i)) for i in range(15)],
                dtype=pd.Int32Dtype(),
            ),
            "C": pd.Series(list("ALPHATHETAGAMMA")),
            "D": pd.Series([0] * 12 + [1] * 3),
            "E": pd.Series(
                [[True, False, None, True, False][i % 5] for i in range(15)],
                dtype=pd.BooleanDtype(),
            ),
            "F": pd.Series(
                [["A", "B", "A", None, "A", "B", None][i % 7] for i in range(15)]
            ),
            "G": pd.Series(
                [None if i % 6 == 0 else i // 4 for i in range(15)],
                dtype=pd.Int32Dtype(),
            ),
            "H": pd.Series(list(range(15)), dtype=np.int32),
            "I": pd.Series(
                [None if i % 4 == 0 else pd.Timestamp("1999-12-31") for i in range(15)]
            ),
            "J": pd.Series(
                [None if i % 2 == 0 else i / 2 for i in range(15)], dtype=np.float32
            ),
            "K": pd.Series([True, False, False] * 5, dtype=np.bool_),
        }
    )


@pytest.mark.parametrize(
    "keys, funcs, orderby, ascending, napos, answer",
    [
        pytest.param(
            ["A"],
            (("row_number",),),
            ("B",),
            (True,),
            ("last",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [5, 4, 3, 4, 3, 5, 3, 2, 2, 2, 5, 1, 1, 1, 4],
                }
            ),
            id="row_number-simple_ordering",
        ),
        pytest.param(
            ["A"],
            (("row_number",),),
            ("C", "B"),
            (False, True),
            ("last", "first"),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [4, 2, 3, 3, 5, 1, 2, 4, 2, 5, 3, 4, 1, 1, 5],
                }
            ),
            id="row_number-compound_ordering",
        ),
        pytest.param(
            ["A"],
            (("min_row_number_filter",),),
            ("B",),
            (True,),
            ("last",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [False] * 11 + [True] * 3 + [False],
                }
            ),
            id="min_row_number_filter-simple_ordering",
        ),
        pytest.param(
            ["A"],
            (("row_number",), ("min_row_number_filter",), ("row_number",)),
            ("B",),
            (True,),
            ("last",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [5, 4, 3, 4, 3, 5, 3, 2, 2, 2, 5, 1, 1, 1, 4],
                    "AGG_OUTPUT_1": [False] * 11 + [True] * 3 + [False],
                    "AGG_OUTPUT_2": [5, 4, 3, 4, 3, 5, 3, 2, 2, 2, 5, 1, 1, 1, 4],
                }
            ),
            id="multiple_calls-simple_ordering",
        ),
        pytest.param(
            ["A"],
            (("rank",), ("dense_rank",), ("percent_rank",), ("cume_dist",)),
            ("C",),
            (True,),
            ("last",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [1, 4, 3, 3, 1, 4, 3, 2, 4, 1, 3, 1, 5, 5, 1],
                    "AGG_OUTPUT_1": [1, 4, 2, 2, 1, 3, 2, 2, 3, 1, 3, 1, 3, 5, 1],
                    "AGG_OUTPUT_2": [
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
                    "AGG_OUTPUT_3": [
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
                }
            ),
            id="rank_fns-simple_ordering",
        ),
        pytest.param(
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
                    "AGG_OUTPUT_0": [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4, 1, 2, 4],
                    "AGG_OUTPUT_1": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2, 3],
                    "AGG_OUTPUT_2": [
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
                    "AGG_OUTPUT_3": [
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
                }
            ),
            id="rank_fns-mixed_sized_groups",
        ),
        pytest.param(
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
            ("B",),
            (False,),
            ("last",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1],
                    "AGG_OUTPUT_1": [3, 1, 2, 1, 1, 3, 2, 1, 2, 2, 3, 3, 3, 2, 1],
                    "AGG_OUTPUT_2": [10, 1, 7, 4, 2, 11, 5, 3, 8, 6, 12, 9, 3, 2, 1],
                    "AGG_OUTPUT_3": [4, 1, 3, 2, 1, 5, 2, 1, 3, 2, 5, 4, 3, 2, 1],
                    "AGG_OUTPUT_4": [5, 1, 4, 2, 1, 6, 3, 2, 4, 3, 6, 5, 3, 2, 1],
                    "AGG_OUTPUT_5": [8, 1, 5, 2, 1, 9, 3, 2, 6, 4, 10, 7, 3, 2, 1],
                }
            ),
            id="ntile",
        ),
        pytest.param(
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
            ("B",),
            (True,),
            ("first",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [1, 5, 4, 5, 5, 2, 4, 5, 4, 4, 3, 3, 0, 1, 1],
                    "AGG_OUTPUT_1": [1, 12, 6, 9, 11, 2, 8, 10, 5, 7, 3, 4, 1, 2, 3],
                    "AGG_OUTPUT_2": [0, 5, 2, 4, 5, 0, 3, 4, 2, 3, 0, 1, 0, 0, 1],
                    "AGG_OUTPUT_3": [0, 5, 4, 4, 4, 1, 4, 4, 3, 4, 1, 2, 0, 0, 1],
                    "AGG_OUTPUT_4": [1, 6, 3, 5, 6, 1, 4, 5, 3, 4, 2, 2, 1, 2, 3],
                    "AGG_OUTPUT_5": [0, 6, 2, 4, 5, 0, 3, 5, 1, 3, 1, 1, 0, 0, 0],
                }
            ),
            id="conditional_events",
        ),
        pytest.param(
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
                    "AGG_OUTPUT_0": [12] * 12 + [3] * 3,
                    "AGG_OUTPUT_1": [9] * 12 + [3] * 3,
                    "AGG_OUTPUT_2": [10] * 12 + [2] * 3,
                    "AGG_OUTPUT_3": [5] * 12 + [1] * 3,
                }
            ),
            id="count_fns-no_frame-no_order",
        ),
        pytest.param(
            ["D"],
            (
                # COUNT(*) with a sliding frame
                ("size", -3, 3),
                # COUNT on an array of nullable integers with a sliding frame
                ("count", "B", -4, -2),
                # COUNT_IF on an array of nullable booleans with a sliding frame
                ("count_if", "E", 1, 6),
            ),
            ("B",),
            (True,),
            ("first",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [4, 4, 7, 7, 5, 5, 7, 6, 7, 7, 6, 7, 3, 3, 3],
                    "AGG_OUTPUT_1": [0, 3, 1, 3, 3, 0, 3, 3, 0, 2, 0, 0, 0, 0, 1],
                    "AGG_OUTPUT_2": [3, 0, 1, 0, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 0],
                }
            ),
            id="count_fns-sliding_frames",
        ),
        pytest.param(
            ["D"],
            (
                # COUNT(*) with a prefix frame
                ("size", "None", -1),
                # COUNT on an array of nullable integers with a prefix frame
                ("count", "B", "None", 0),
                # COUNT_IF on an array of nullable booleans with a prefix frame
                ("count_if", "E", "None", 2),
            ),
            ("B",),
            (True,),
            ("first",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [0, 11, 5, 8, 10, 1, 7, 9, 4, 6, 2, 3, 0, 1, 2],
                    "AGG_OUTPUT_1": [0, 9, 3, 6, 8, 0, 5, 7, 2, 4, 0, 1, 1, 2, 3],
                    "AGG_OUTPUT_2": [3, 5, 4, 5, 5, 3, 5, 5, 4, 5, 4, 4, 1, 1, 1],
                }
            ),
            id="count_fns-prefix_frames",
        ),
        pytest.param(
            ["D"],
            (
                # COUNT(*) with a suffix frame
                ("size", 0, "None"),
                # COUNT on an array of nullable integers with a suffix frame
                ("count", "B", 1, "None"),
                # COUNT_IF on an array of nullable booleans with a suffix frame
                ("count_if", "E", -2, "None"),
            ),
            ("B",),
            (True,),
            ("first",),
            pd.DataFrame(
                {
                    "AGG_OUTPUT_0": [12, 1, 7, 4, 2, 11, 5, 3, 8, 6, 10, 9, 3, 2, 1],
                    "AGG_OUTPUT_1": [9, 0, 6, 3, 1, 9, 4, 2, 7, 5, 9, 8, 2, 1, 0],
                    "AGG_OUTPUT_2": [5, 0, 2, 1, 1, 5, 1, 1, 3, 2, 5, 4, 1, 1, 1],
                }
            ),
            id="count_fns-suffix_frames",
        ),
        pytest.param(
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
                    "AGG_OUTPUT_0": [9] * 12 + [2] * 3,
                    "AGG_OUTPUT_1": [12] * 12 + [3] * 3,
                    "AGG_OUTPUT_2": [9] * 12 + [2] * 3,
                    "AGG_OUTPUT_3": [6] * 12 + [1] * 3,
                    "AGG_OUTPUT_4": [12] * 12 + [3] * 3,
                    "AGG_OUTPUT_5": [4] * 12 + [1] * 3,
                }
            ),
            id="count_fns-other_arrays",
        ),
    ],
)
def test_window(
    test_window_df, keys, funcs, orderby, ascending, napos, answer, memory_leak_check
):
    """
    Tests using groupby.window on various aggregation types, sometimes
    multiple at once
    """

    func_text = "def impl(df):\n"
    func_text += f"   return df.groupby({keys}, as_index = False, dropna = False, _is_bodosql = True)"
    func_text += f".window({funcs}, {orderby}, {ascending}, {napos})"
    local_vars = {}
    exec(func_text, local_vars)
    impl = local_vars["impl"]

    check_func(
        impl,
        (test_window_df,),
        py_output=answer,
        sort_output=False,
        reset_index=True,
        check_names=False,
        check_dtype=False,
    )
