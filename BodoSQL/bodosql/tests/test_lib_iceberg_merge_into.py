"""
Tests helper functions used in our MERGE_INTO implementation, specifically those defined
at BodoSQL/bodosql/libs/merge_into.py
"""

import io
import random

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    gen_nonascii_list,
    gen_random_string_binary_array,
)
from bodo.utils.typing import BodoError
from bodosql.libs.iceberg_merge_into import (
    DELETE_ENUM,
    INSERT_ENUM,
    MERGE_ACTION_ENUM_COL_NAME,
    ROW_ID_COL_NAME,
    UPDATE_ENUM,
    do_delta_merge_with_target,
)

pytestmark = [pytest.mark.iceberg, pytest.mark.skip]

small_df_len = 12

base_df_int = pd.DataFrame(
    {
        "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
        "B": pd.Series(np.arange(small_df_len, dtype=np.int64) * 2),
        "C": pd.Series(np.arange(small_df_len, dtype=np.int64) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
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
        "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
        "B": pd.Series(np.arange(small_df_len, dtype=np.int64) * 2),
        "C": pd.Series(np.arange(small_df_len, dtype=np.int64) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
    }
)

delete_everything_and_insert_some_stuff_df_int = pd.DataFrame(
    {
        "A": pd.Series(np.arange(14, dtype=np.int64)),
        "B": pd.Series(np.arange(14, dtype=np.int64) * 2),
        "C": pd.Series(np.arange(14, dtype=np.int64) * 3),
        ROW_ID_COL_NAME: pd.Series(np.arange(14, dtype=np.int64) - 1),
        MERGE_ACTION_ENUM_COL_NAME: [INSERT_ENUM] + [DELETE_ENUM] * 12 + [INSERT_ENUM],
    }
)

# Make sure everything works with string, binary, and timestamp types
base_df_string = pd.DataFrame(
    {
        "A": pd.Series(np.arange(small_df_len, dtype=np.int64).astype(str)),
        "B": pd.Series(np.arange(small_df_len, dtype=np.int64) * 2).astype(bytes),
        "C": pd.date_range(
            start="2018-07-24", end="2018-11-29", periods=small_df_len, unit="ns"
        ),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
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
        "C": pd.date_range(
            start="2018-07-24", end="2018-07-25", periods=small_df_len, unit="ns"
        ),
        ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
        MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
    }
)

delete_everything_and_insert_some_stuff_df_string = pd.DataFrame(
    {
        "A": gen_random_string_binary_array(14, is_binary=False),
        "B": gen_random_string_binary_array(14, is_binary=True),
        "C": pd.date_range(start="2018-07-24", end="2018-07-25", periods=14, unit="ns"),
        ROW_ID_COL_NAME: pd.Series(np.arange(14, dtype=np.int64) - 1),
        MERGE_ACTION_ENUM_COL_NAME: [INSERT_ENUM] + [DELETE_ENUM] * 12 + [INSERT_ENUM],
    }
)


# Large df, to stress test the implementation with large number of rows
np.random.seed(42)
stress_test_base_df = pd.DataFrame(
    {
        "INT_COL": np.random.randint(-1000000, 1000000, size=100),
        "STR_COL": gen_random_string_binary_array(100, is_binary=False),
        "NON_ASCII_STR_COL": gen_nonascii_list(100),
        "BYTES_COL": gen_random_string_binary_array(100, is_binary=True),
        "TS_COL": pd.date_range(
            start="2011-02-24", end="2013-01-1", periods=100, unit="ns"
        ),
        "TD_COL": pd.Series(np.random.randint(-1000000, 1000000, size=100)).astype(
            "timedelta64[ns]"
        ),
        "BOOL_COL": np.random.randint(0, 2, size=100).astype(bool),
        "FLOAT_COL": np.random.uniform(-1000000, 1000000, size=100),
    }
)

stress_test_delta_df = pd.DataFrame(
    {
        "INT_COL": np.random.randint(-1000000, 1000000, size=75),
        # Flip the values for str and non-ascii, so we can be sure that we can
        # insert/update non-ascii string into normal str array, and visa versa
        "STR_COL": gen_nonascii_list(75),
        "NON_ASCII_STR_COL": gen_random_string_binary_array(75, is_binary=False),
        "BYTES_COL": gen_random_string_binary_array(75, is_binary=True),
        "TS_COL": pd.date_range(
            start="2011-02-24", end="2013-01-1", periods=75, unit="ns"
        ),
        "TD_COL": pd.Series(np.random.randint(-1000000, 1000000, size=75)).astype(
            "timedelta64[ns]"
        ),
        "BOOL_COL": np.random.randint(0, 2, size=75).astype(bool),
        "FLOAT_COL": np.random.uniform(-1000000, 1000000, size=75),
    }
)

# Randomly insert nulls
cond = np.random.ranf(stress_test_base_df.shape) < 0.5
stress_test_base_df = stress_test_base_df.mask(cond, stress_test_base_df, axis=0)
cond = np.random.ranf(stress_test_delta_df.shape) < 0.5
stress_test_delta_df = stress_test_delta_df.mask(cond, stress_test_delta_df, axis=0)

base_row_ids = random.sample(list(np.arange(1000, dtype=np.int64)), 100)
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
        base_df (DataFrame): The original/target DataFrame, to apply the changes to.
                             must have a row id column with name ROW_ID_COL_NAME.
        delta_df (DataFrame): The delta DataFrame, containing the changes to apply to
                              the target DataFrame. Must have a row id column with
                              name ROW_ID_COL_NAME, and a action enum colum with name
                              MERGE_ACTION_ENUM_COL_NAME.
    """
    base_df = orig_base_df.copy()
    base_df = base_df.reset_index(drop=True)

    delta_df_insert = delta_df[(delta_df[MERGE_ACTION_ENUM_COL_NAME] == INSERT_ENUM)]
    delta_df_update = delta_df[(delta_df[MERGE_ACTION_ENUM_COL_NAME] == UPDATE_ENUM)]
    delta_df_delete = delta_df[delta_df[MERGE_ACTION_ENUM_COL_NAME] == DELETE_ENUM]

    base_df = base_df.set_index(ROW_ID_COL_NAME, drop=True)
    for _, row in delta_df_update.iterrows():
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


@pytest.mark.slow
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
def test_do_delta_merge_with_target(args, use_table_format, memory_leak_check):
    """
    Tests our helper functions used in the distributed MERGE INTO case.
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
        use_table_format=use_table_format,
    )


@pytest.mark.slow
def test_do_delta_merge_failure():
    """
    Tests our helper functions used in MERGE INTO throw a reasonable error
    when encountering an invalid delta table.
    """
    target_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
            ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
        }
    )
    delta_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
            ROW_ID_COL_NAME: [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM, UPDATE_ENUM] * 6,
        }
    )

    with pytest.raises(
        BodoError,
        match="Error in MERGE INTO: Found multiple actions to apply to the same row in the target table",
    ):
        do_delta_merge_with_target(target_df, delta_df)


@pytest.mark.slow
def test_do_delta_merge_disallow_multiple_delete():
    """
    Tests our helper functions in MERGE INTO does not allow multiple delete actions for the same row.
    This can potentially be relaxed in future versions of BodoSQL, but for now, it simplifies codegen.
    """
    target_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
            ROW_ID_COL_NAME: pd.Series(np.arange(small_df_len, dtype=np.int64)),
        }
    )
    delta_df = pd.DataFrame(
        {
            "A": pd.Series(np.arange(small_df_len, dtype=np.int64)),
            ROW_ID_COL_NAME: [1, 2, 3, 4, 5, 6] * 2,
            MERGE_ACTION_ENUM_COL_NAME: [DELETE_ENUM] * small_df_len,
        }
    )

    with pytest.raises(
        BodoError,
        match="Error in MERGE INTO: Found multiple actions to apply to the same row in the target table",
    ):
        do_delta_merge_with_target(target_df, delta_df)


@pytest.mark.slow
def test_do_delta_merge_with_target_filter_pushdown_simple(
    iceberg_database, iceberg_table_conn
):
    """
    Tests that do_delta_merge_with_target doesn't 'use' target_df, for purposes of filter
    pushdown.
    #TODO: Add a more extensive E2E test consisting of actual BodoSQL IR once we have codegen working
    """

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        orig_df, _, _ = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        # Normally, this section would be a join on some secondary source table to
        # produce a delta table
        # For now, we're just deleting columns where B = 2 using do_delta_merge_with_target
        filtered_orig_df = orig_df[orig_df.B == 2]
        filtered_orig_df[ROW_ID_COL_NAME] = np.arange(
            len(filtered_orig_df), dtype=np.int64
        )

        delta_table = filtered_orig_df
        delta_table[MERGE_ACTION_ENUM_COL_NAME] = 0
        output_df = do_delta_merge_with_target(filtered_orig_df, delta_table)
        return output_df

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
    check_logger_msg(
        stream, "Columns loaded ['A', 'B', 'C', 'D', 'E', 'F', '_BODO_ROW_ID']"
    )
    check_logger_msg(stream, "Filter pushdown successfully performed")
