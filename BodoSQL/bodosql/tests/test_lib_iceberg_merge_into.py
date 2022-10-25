"""
Tests helper functions used in our MERGE_INTO implementation, specifically those defined
at BodoSQL/bodosql/libs/merge_into.py
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import random

import pytest
from bodosql.libs.iceberg_merge_into import *  # noqa
from bodosql.tests.named_params_common import *  # noqa

from bodo.tests.utils import (
    check_func,
    gen_nonascii_list,
    gen_random_string_binary_array,
)
from bodo.utils.typing import BodoError

small_df_len = 12

base_df_int = pd.DataFrame(
    {
        "A": pd.Series(np.arange(small_df_len)),
        "B": pd.Series(np.arange(small_df_len) * 2),
        "C": pd.Series(np.arange(small_df_len) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
    }
)
# Make sure non range-index doesn't cause any problems.
base_df_int.index = (-1, -1, 0, 1, 2, 3, 12, 11, 5, 6, 20, 19)


delta_df_int = pd.DataFrame(
    {
        "A": [-1, -2, -3],
        "B": pd.Series([-1, -2, -3]),
        "C": [-1, -2, -3],
        ROW_ID_COL_NAME: [0, 11, 2],
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM, INSERT_ENUM, UPDATE_ENUM],
    }
)

delete_everything_df_int = pd.DataFrame(
    {
        "A": pd.Series(np.arange(small_df_len)),
        "B": pd.Series(np.arange(small_df_len) * 2),
        "C": pd.Series(np.arange(small_df_len) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
    }
)

delete_everything_and_insert_some_stuff_df_int = pd.DataFrame(
    {
        "A": pd.Series(np.arange(14)),
        "B": pd.Series(np.arange(14) * 2),
        "C": pd.Series(np.arange(14) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(14) - 1),
        MERGE_ACTION_ENUM_COL_NAME: [INSERT_ENUM] + [DELETE_ENUM] * 12 + [INSERT_ENUM],
    }
)

# Make sure everything works with string, binary, and timestamp types
base_df_string = pd.DataFrame(
    {
        "A": pd.Series(np.arange(small_df_len).astype(str)),
        "B": pd.Series(np.arange(small_df_len) * 2).astype(bytes),
        "C": pd.date_range(start="2018-07-24", end="2018-11-29", periods=small_df_len),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
    }
)

delta_df_string = pd.DataFrame(
    {
        "A": gen_nonascii_list(6),
        "B": [b"how", b"are", b"you"] * 2,
        "C": [
            pd.Timestamp("2013-07-22"),
            pd.Timestamp("2021-01-18"),
            pd.Timestamp("2001-11-01"),
        ]
        * 2,
        ROW_ID_COL_NAME: [2, 3, 0, 1, 5, 11],
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM, INSERT_ENUM, UPDATE_ENUM] * 2,
    }
)

delete_everything_df_string = pd.DataFrame(
    {
        "A": gen_nonascii_list(small_df_len),
        "B": gen_random_string_binary_array(small_df_len, is_binary=True),
        "C": pd.date_range(start="2018-07-24", end="2018-07-25", periods=small_df_len),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
    }
)

delete_everything_and_insert_some_stuff_df_string = pd.DataFrame(
    {
        "A": gen_random_string_binary_array(14, is_binary=False),
        "B": gen_random_string_binary_array(14, is_binary=True),
        "C": pd.date_range(start="2018-07-24", end="2018-07-25", periods=14),
        ROW_ID_COL_NAME: pd.Series(np.arange(14) - 1),
        MERGE_ACTION_ENUM_COL_NAME: [INSERT_ENUM] + [DELETE_ENUM] * 12 + [INSERT_ENUM],
    }
)


# Large df, to stress test the implementation with large number of rows
np.random.seed(42)
stress_test_base_df = pd.DataFrame(
    {
        "int_col": np.random.randint(-1000000, 1000000, size=100),
        "str_col": gen_random_string_binary_array(100, is_binary=False),
        "non_ascii_str_col": gen_nonascii_list(100),
        "bytes_col": gen_random_string_binary_array(100, is_binary=True),
        "ts_col": pd.date_range(start="2011-02-24", end="2013-01-1", periods=100),
        "td_col": pd.Series(np.random.randint(-1000000, 1000000, size=100)).astype(
            "timedelta64[s]"
        ),
        "bool_col": np.random.randint(0, 2, size=100).astype(bool),
        "float_col": np.random.uniform(-1000000, 1000000, size=100),
    }
)

stress_test_delta_df = pd.DataFrame(
    {
        "int_col": np.random.randint(-1000000, 1000000, size=75),
        # Flip the values for str and non-ascii, so we can be sure that we can
        # insert/update non-ascii string into normal str array, and visa versa
        "str_col": gen_nonascii_list(75),
        "non_ascii_str_col": gen_random_string_binary_array(75, is_binary=False),
        "bytes_col": gen_random_string_binary_array(75, is_binary=True),
        "ts_col": pd.date_range(start="2011-02-24", end="2013-01-1", periods=75),
        "td_col": pd.Series(np.random.randint(-1000000, 1000000, size=75)).astype(
            "timedelta64[s]"
        ),
        "bool_col": np.random.randint(0, 2, size=75).astype(bool),
        "float_col": np.random.uniform(-1000000, 1000000, size=75),
    }
)

# Randomly insert nulls
cond = np.random.ranf(stress_test_base_df.shape) < 0.5
stress_test_base_df = stress_test_base_df.mask(cond, stress_test_base_df, axis=0)
cond = np.random.ranf(stress_test_delta_df.shape) < 0.5
stress_test_delta_df = stress_test_delta_df.mask(cond, stress_test_delta_df, axis=0)

base_row_ids = random.sample(list(np.arange(1000)), 100)
delta_row_ids = random.sample(base_row_ids, 75)
# The row id column must be sorted for the base df
stress_test_base_df[ROW_ID_COL_NAME] = sorted(base_row_ids)
# The delta df doesn't need to be
stress_test_delta_df[ROW_ID_COL_NAME] = delta_row_ids
stress_test_delta_df[MERGE_ACTION_ENUM_COL_NAME] = np.random.choice(
    [DELETE_ENUM, INSERT_ENUM, UPDATE_ENUM], 75
)


def delta_merge_equiv(orig_base_df, delta_df):
    """helper fn used to generate expected output for
    test_do_delta_merge_with_target

    Args:
        base_df (dataframe): The original/target dataframe, to apply the changes to.
                             must have a row id column with name ROW_ID_COL_NAME.
        delta_df (dataframe): The delta dataframe, containing the changes to apply to
                              the target dataframe. Must have a row id column with
                              name ROW_ID_COL_NAME, and a action enum colum with name
                              MERGE_ACTION_ENUM_COL_NAME.
    """
    base_df = orig_base_df.copy()
    base_df = base_df.reset_index(drop=True)

    delta_df_insert = delta_df[(delta_df[MERGE_ACTION_ENUM_COL_NAME] == INSERT_ENUM)]
    delta_df_update = delta_df[(delta_df[MERGE_ACTION_ENUM_COL_NAME] == UPDATE_ENUM)]
    delta_df_delete = delta_df[delta_df[MERGE_ACTION_ENUM_COL_NAME] == DELETE_ENUM]

    base_df = base_df.set_index(ROW_ID_COL_NAME, drop=True)
    for _idx, row in delta_df_update.iterrows():
        cur_update_id = row[ROW_ID_COL_NAME]
        base_df.loc[cur_update_id] = row

    to_delete = base_df.index.isin(delta_df_delete[ROW_ID_COL_NAME])
    base_df = base_df[~to_delete]
    base_df = pd.concat([base_df, delta_df_insert])

    return base_df.loc[:, list(orig_base_df.columns.drop(ROW_ID_COL_NAME))]


def do_delta_merge_with_target_py_wrapper(target_df, delta_df):
    """Pure python wrapper around do_delta_merge_with_target, so that check_func can properly compile it
    with the correct distribution arguments
    """
    return do_delta_merge_with_target(target_df, delta_df)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                base_df_int,
                delta_df_int,
            ),
            id="base_df_int_with_delta_df_int",
        ),
        pytest.param(
            (
                base_df_int,
                delete_everything_and_insert_some_stuff_df_int,
            ),
            id="base_df_int_with_delete_all_and_insert_df_int",
        ),
        pytest.param(
            (
                base_df_int,
                delete_everything_df_int,
            ),
            id="base_df_int_with_delete_everything_df_int",
        ),
        pytest.param(
            (
                base_df_string,
                delta_df_string,
            ),
            id="base_df_string_with_delta_df_string",
        ),
        pytest.param(
            (
                base_df_string,
                delete_everything_and_insert_some_stuff_df_string,
            ),
            id="base_df_string_with_delete_all_and_insert_df_string",
        ),
        pytest.param(
            (
                base_df_string,
                delete_everything_df_string,
            ),
            id="base_df_string_with_delete_everything_df_string",
        ),
        pytest.param(
            (
                stress_test_base_df,
                stress_test_delta_df,
            ),
            id="stress_test_df",
        ),
    ],
)
@pytest.mark.parametrize(
    "use_table_format",
    [
        True,
        False,
    ],
)
def test_do_delta_merge_with_target_dist(args, use_table_format, memory_leak_check):
    """
    Tests our helper functions used in the distributed MERGE INTO case.
    The distributed tests are separated from the replicated tests due to a memory
    leak in sort with _bodo_chunk_bounds: https://bodo.atlassian.net/browse/BE-3775
    """
    target_df, delta_df = args
    expected_output = delta_merge_equiv(target_df, delta_df)

    check_func(
        do_delta_merge_with_target_py_wrapper,
        (target_df, delta_df),
        py_output=expected_output,
        reset_index=True,
        sort_output=True,
        check_dtype=False,
        only_1DVar=True,
        use_table_format=use_table_format,
    )
    check_func(
        do_delta_merge_with_target_py_wrapper,
        (target_df, delta_df),
        py_output=expected_output,
        reset_index=True,
        sort_output=True,
        check_dtype=False,
        only_1D=True,
        use_table_format=use_table_format,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                base_df_int,
                delta_df_int,
            ),
            id="base_df_int_with_delta_df_int",
        ),
        pytest.param(
            (
                base_df_int,
                delete_everything_and_insert_some_stuff_df_int,
            ),
            id="base_df_int_with_delete_all_and_insert_df_int",
        ),
        pytest.param(
            (
                base_df_int,
                delete_everything_df_int,
            ),
            id="base_df_int_with_delete_everything_df_int",
        ),
        pytest.param(
            (
                base_df_string,
                delta_df_string,
            ),
            id="base_df_string_with_delta_df_string",
        ),
        pytest.param(
            (
                base_df_string,
                delete_everything_and_insert_some_stuff_df_string,
            ),
            id="base_df_string_with_delete_all_and_insert_df_string",
        ),
        pytest.param(
            (
                base_df_string,
                delete_everything_df_string,
            ),
            id="base_df_string_with_delete_everything_df_string",
        ),
        pytest.param(
            (
                stress_test_base_df,
                stress_test_delta_df,
            ),
            id="stress_test_df",
        ),
    ],
)
@pytest.mark.parametrize(
    "use_table_format",
    [
        True,
        False,
    ],
)
def test_do_delta_merge_with_target_seq(args, use_table_format):
    """
    Tests our helper functions used in the replicated MERGE INTO case.
    The distributed tests are separated from the replicated tests due to a memory
    leak in sort with _bodo_chunk_bounds: https://bodo.atlassian.net/browse/BE-3775
    """
    target_df, delta_df = args
    expected_output = delta_merge_equiv(target_df, delta_df)

    check_func(
        do_delta_merge_with_target_py_wrapper,
        (target_df, delta_df),
        py_output=expected_output,
        reset_index=True,
        sort_output=True,
        check_dtype=False,
        only_seq=True,
        use_table_format=use_table_format,
    )


def test_do_delta_merge_failure():
    """
    Tests our helper functions used in MERGE INTO throw a reasonable error
    when encountering an invalid delta table.
    """
    target_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len)),
            ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
        }
    )
    delta_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len)),
            ROW_ID_COL_NAME: [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM, UPDATE_ENUM] * 6,
        }
    )

    with pytest.raises(
        BodoError,
        match="Error in MERGE INTO: Found multiple actions to apply to the same row in the target table",
    ):
        out_val = do_delta_merge_with_target(target_df, delta_df)


def test_do_delta_merge_disallow_multiple_delete():
    """
    Tests our helper functions in MERGE INTO does not allow multiple delete actions for the same row.
    This can potentially be relaxed in future versions of BodoSQL, but for now, it simplifies codegen.
    """
    target_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len)),
            ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len)),
        }
    )
    delta_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len)),
            ROW_ID_COL_NAME: [1, 2, 3, 4, 5, 6] * 2,
            MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
        }
    )

    with pytest.raises(
        BodoError,
        match="Error in MERGE INTO: Found multiple actions to apply to the same row in the target table",
    ):
        do_delta_merge_with_target(target_df, delta_df)
