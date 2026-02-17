"""
Test correctness of MERGE INTO plan generation within BodoSQL.
Specifically, tests the plan generation for the "Delta table",
the input to the LogicalTableModify node.

This file should not be extended, as it contains a significant amount of code that is
dedicated to testing MERGE INTO plan generation within BodoSQL.
"""

import itertools
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

"""In this file, to create test cases, we manually generate the rows that are
matched/not matched depending on the join condition and the source/dest table, since there isn't
A way of handling it in pandas (for non equality joins specifically)

Once that's done, we have a variety of helper functions
that can automatically compute the expected delta table, given the matched/not matched conditions/actions,
and their ordering in the query.

"""

# Skip this file until we merge the Iceberg branch
pytest.skip(
    allow_module_level=True,
    reason="Waiting for MERGE INTO support to fix the Calcite generated issue",
)

from bodosql.libs.iceberg_merge_into import (
    DELETE_ENUM,
    INSERT_ENUM,
    ROW_ID_COL_NAME,
    UPDATE_ENUM,
)

# This is the section where we define the source/dest tables, and the join conditions/using conditions
target_df_one = pd.DataFrame(
    {"A": np.arange(12), "B": np.arange(12) * 2, "C": np.arange(12) * 3}
)
source_df_one = pd.DataFrame(
    {"X": [1, 3, 10, 20], "Y": [-1, -2, -3, -4], "Z": [-5, -6, -7, -8]}
)
join_condition_one = "DEST_TABLE.A = SOURCE_TABLE.X"
using_cond_one = "SOURCE_TABLE as SOURCE_TABLE"

target_one_source_one_condition_one_matched_rows = source_df_one.merge(
    target_df_one, left_on="X", right_on="A"
)
target_one_source_one_condition_one_matched_rows[ROW_ID_COL_NAME] = (
    target_one_source_one_condition_one_matched_rows["A"]
)
target_one_source_one_condition_one_not_matched_rows = pd.DataFrame(
    {
        "X": [20],
        "Y": [-4],
        "Z": [-8],
        ROW_ID_COL_NAME: pd.array([None], dtype=pd.ArrowDtype(pa.int64())),
    }
)

join_condition_two = "(SOURCE_TABLE.X - 9) = DEST_TABLE.A AND SOURCE_TABLE.Y in (-5, -4) AND DEST_TABLE.A IN (11, 20, 100)"

target_one_source_one_condition_two_matched_rows = pd.DataFrame(
    {
        "A": [11],
        "B": [11 * 2],
        "C": [11 * 3],
        "X": [20],
        "Y": [-4],
        "Z": [-8],
        ROW_ID_COL_NAME: [11],
    }
)
target_one_source_one_condition_two_not_matched_rows = pd.DataFrame(
    {
        "X": [1, 3, 10],
        "Y": [-1, -2, -3],
        "Z": [-5, -6, -7],
        ROW_ID_COL_NAME: pd.array([None] * 3, dtype=pd.ArrowDtype(pa.int64())),
    }
)

not_matched_condition0 = "X = 1"
not_matched_condition1 = "X <= 3"
not_matched_condition2 = "X > 10"
not_matched_condition3 = "X + X > 2 AND (Select MIN(inner_tbl.A) > 0 from __bodolocal__.DEST_TABLE as inner_tbl)"

matched_condition0 = "A = X"
matched_condition1 = "A <= X"
matched_condition2 = "A >= X"
matched_condition3 = "A = 3"
matched_condition4 = "(A + X) > 0 AND (Select MAX(inner_tbl.A) > 0 from __bodolocal__.DEST_TABLE as inner_tbl)"
matched_condition5 = "(A -  2 * X) < 0 AND (Select MAX(inner_tbl.X) - MIN(inner_tbl.Y) > 1 from __bodolocal__.SOURCE_TABLE as inner_tbl)"

valid_not_matched_conditions = [
    None,  # None is equivalent to no condition, or always true
    not_matched_condition0,
    not_matched_condition1,
    not_matched_condition2,
    # These conditions generate an unsupported outer join in BodoSQL: https://bodo.atlassian.net/browse/BE-3716
    # not_matched_condition3,
]
valid_matched_conditions = valid_not_matched_conditions + [
    matched_condition0,
    matched_condition1,
    matched_condition2,
    matched_condition3,
    # These conditions generate an unsupported outer join in BodoSQL: https://bodo.atlassian.net/browse/BE-3716
    # matched_condition4,
    # matched_condition5,
]


def find_rows_matching_condition(df, cond):
    """returns a dataframe containing only the filtered rows, and a dataframe with everything but the filtered rows"""
    assert cond in valid_matched_conditions, f"Found impossible condition: {cond}"

    if cond == not_matched_condition0:
        return (df[df.X == 1], df[df.X != 1])
    elif cond == not_matched_condition1:
        return (df[df.X <= 3], df[df.X > 3])
    elif cond == not_matched_condition2:
        return (df[df.X > 10], df[df.X <= 10])
    elif cond == not_matched_condition3:
        return (df[df.X * 2 > 2], df[df.X * 2 <= 2])
    elif cond == matched_condition0:
        return (df[df.A == df.X], df[df.A != df.X])
    elif cond == matched_condition1:
        return (df[df.A <= df.X], df[df.A > df.X])
    elif cond == matched_condition2:
        return (df[df.A >= df.X], df[df.A < df.X])
    elif cond == matched_condition3:
        return (df[df.A == 3], df[df.A != 3])
    elif cond == matched_condition4:
        # Aggregation portion should always be true
        return (df[(df.A + df.X) > 2], df[(df.A + df.X) <= 2])
    elif cond == matched_condition5:
        # Aggregation portion should always be true
        return (df[(df.A - 2 * df.X) < 0], df[(df.A - 2 * df.X) >= 0])
    elif cond == None:
        return df.copy(), pd.DataFrame({})
    else:
        raise Exception(
            f"Error, unhandled condition in find_rows_matching_condition: {cond}"
        )


matched_action_0 = "THEN UPDATE SET B = Y"
matched_action_1 = "THEN DELETE"

valid_matched_actions = [matched_action_0, matched_action_1]


def apply_matched_action(df, action):
    assert action in valid_matched_actions, f"Found impossible matched action: {action}"
    if action == matched_action_0:
        df["B"] = df["Y"]
        df["_merge_into_change"] = UPDATE_ENUM
    elif action == matched_action_1:
        # No changes needed for delete
        df["_merge_into_change"] = DELETE_ENUM
    else:
        raise Exception(f"Error, unhandled action in apply_matched_action: {action}")

    return df.loc[:, ["A", "B", "C", ROW_ID_COL_NAME, "_merge_into_change"]]


not_matched_action_0 = "THEN INSERT (A, B, C) VALUES (-101, -102, -103)"
not_matched_action_1 = "THEN INSERT (A, B, C) VALUES (X, Y, Z)"
not_matched_action_2 = "THEN INSERT (A, C) VALUES (-10, -11)"

valid_not_matched_actions = [
    not_matched_action_0,
    not_matched_action_1,
    not_matched_action_2,
]


def apply_not_matched_action(df, action):
    assert action in valid_not_matched_actions, (
        f"Found impossible not matched action: {action}"
    )

    # Insert sets to NA by default
    df["A"] = pd.NA
    df["B"] = pd.NA
    df["C"] = pd.NA
    if action == not_matched_action_0:
        df["A"] = -101
        df["B"] = -102
        df["C"] = -103
    elif action == not_matched_action_1:
        df["A"] = df["X"]
        df["B"] = df["Y"]
        df["C"] = df["Z"]
    elif action == not_matched_action_2:
        df["A"] = -10
        df["C"] = -11
    else:
        raise Exception(
            f"Error, unhandled action in def apply_not_matched_action: {action}"
        )

    df = df.loc[:, ["A", "B", "C"]]
    df["_merge_into_change"] = INSERT_ENUM
    return df


def gen_clauses(matched_conditions_and_actions, not_matched_conditions_and_actions):
    cond_str = ""
    for matched_condition, matched_action in matched_conditions_and_actions:
        if matched_condition != None:
            cond_str += f"WHEN MATCHED AND {matched_condition} "
        else:
            cond_str += "WHEN MATCHED "
        cond_str += matched_action + "\n"

    for not_matched_condition, not_matched_action in not_matched_conditions_and_actions:
        if not_matched_condition != None:
            cond_str += f"WHEN NOT MATCHED AND {not_matched_condition} "
        else:
            cond_str += "WHEN NOT MATCHED "
        cond_str += not_matched_action + "\n"
    return cond_str


def gen_expected_query_and_expected_df(
    using_cond,
    join_cond,
    matched_rows,
    not_matched_rows,
    matched_conditions_and_actions,
    not_matched_conditions_and_actions,
):
    matched_rows = matched_rows.copy()
    not_matched_rows = not_matched_rows.copy()

    # First construct the query
    query = f"""
        MERGE INTO DEST_TABLE as DEST_TABLE
        USING {using_cond} ON ({join_cond})\n
    """
    output_df = pd.DataFrame({}, columns=["A", "B", "C", ROW_ID_COL_NAME])

    query += gen_clauses(
        matched_conditions_and_actions, not_matched_conditions_and_actions
    )

    # Next, iterate over each of the not/matched conditions to construct the output dataframe
    # In MERGE INTO queries, each row is updated/inserted/deleted depending on the first matching condition
    # Note that we omit no-ops, both here and in the codegen
    cur_not_matched_rows = not_matched_rows

    for not_matched_condition, not_matched_action in not_matched_conditions_and_actions:
        rows_with_condition, remaining_matched_rows = find_rows_matching_condition(
            cur_not_matched_rows, not_matched_condition
        )
        cur_not_matched_rows = remaining_matched_rows
        rows_with_applied_insert = apply_not_matched_action(
            rows_with_condition, not_matched_action
        )
        output_df = pd.concat([output_df, rows_with_applied_insert])

    cur_matched_rows = matched_rows
    for matched_condition, matched_action in matched_conditions_and_actions:
        rows_with_condition, remaining_matched_rows = find_rows_matching_condition(
            cur_matched_rows, matched_condition
        )
        cur_matched_rows = remaining_matched_rows
        rows_with_applied_update_or_delete = apply_matched_action(
            rows_with_condition, matched_action
        )
        output_df = pd.concat([output_df, rows_with_applied_update_or_delete])

    return query, output_df


# Some constants used with get_matched_actions and get_not_matched_actions to control the number of
# generated conditions these can be adjusted as needed
num_conditional_permutations = 5
num_non_conditional_permutations = 1


def get_actions(gen_matched_actions=True):
    """
    Generates a list of either matched conditions + actions (DELETE or UPDATE) or
    not matched conditions + actions (INSERT). Returns a list of lists of tuples.
    Each tuple is a combination of
    condition plus action. IE [(if cond1, then update...), (if cond2 then delete...)] translates to

    WHEN MATCHED AND cond1 then update...
    WHEN MATCHED AND cond2 then delete...

    Generates at least one check for each condition, a number of checks that contain every
    condition in a random ordering (equal to the global val num_conditional_permutations)
    and a number of checks that contain every
    condition including always true (equal to num_non_conditional_permutations).

    Args:
        gen_matched_actions (bool, optional): Controls weather we generate matched actions
                (UPDATE/DELETE) or not matched actions (INSERT). Defaults to True.

    Returns:
        list[list(tuple(str))]: A list of combinations of conditions to apply.
    """
    random.seed(42)
    if gen_matched_actions:
        # Intentionally omitting None here
        valid_conditions = valid_matched_conditions[1:]
        valid_actions = valid_matched_actions
    else:
        # Intentionally omitting None here
        valid_conditions = valid_not_matched_conditions[1:]
        valid_actions = valid_not_matched_actions

    all_conditional_clauses = list(
        # Intentionally omitting None here
        itertools.product(valid_conditions, valid_actions)
    )
    all_non_conditional_clauses = list(itertools.product([None], valid_actions))

    chosen_clauses = []

    # Make sure that we test each possible condition/action pair at least once individually
    for conditional_clause in all_conditional_clauses:
        chosen_clauses.append((conditional_clause,))

    # Add the test cases that check multiple clauses concurrently
    seen_permutations = set()
    for i in range(num_conditional_permutations + num_non_conditional_permutations):
        saw_new_permutation = False
        # Forcibly calculating every possible permutation and then doing random.choices causes
        # my computer to go OOM locally. Since the number of permutations is so large.
        # Therefore, we just randomly shuffle the valid condition/action combinations,
        # and append it if we haven't seen that permutation before

        # The for loop is to
        # make sure we don't hang, if num_conditional_permutations + num_non_conditional_permutations
        # is very large relative to the number of possible permutations
        # (this is a safety precaution, it shouldn't ever be anywhere close)
        for j in range(3):
            random.shuffle(all_conditional_clauses)
            new_tuple = tuple(all_conditional_clauses)
            if new_tuple not in seen_permutations:
                seen_permutations.add(new_tuple)
                saw_new_permutation = True
                break
        if not saw_new_permutation:
            break

        # Add a few cases where we have a non-conditional matched action
        if i >= num_conditional_permutations:
            new_tuple += (random.choice(all_non_conditional_clauses),)

        chosen_clauses.append(new_tuple)

    # Always include a query with no matched/not matched clauses, as it's an important edge case to check
    chosen_clauses += [[]]

    random.shuffle(chosen_clauses)
    return chosen_clauses


def get_matched_actions():
    return get_actions(True)


def get_not_matched_actions():
    return get_actions(False)


def make_parameterized(params_list, matched=True):
    if matched:
        match_str = "WHEN MATCHED"
    else:
        match_str = "WHEN NOT MATCHED"
    out_params_list = []

    for condition_action_list in params_list:
        id = ""
        for condition, action in condition_action_list:
            id += f"{match_str} AND {condition} THEN {action}\n"

        out_params_list.append(pytest.param(condition_action_list, id=id))

    return out_params_list
