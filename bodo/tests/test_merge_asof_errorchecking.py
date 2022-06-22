import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError

# ------------------------------ merge_asof() ------------------------------ #


df1 = pd.DataFrame({"A": [1, 2, 3], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
df2 = pd.DataFrame({"A": [1, 2, 5], "B": ["aa", "b", "c"], "C": ["aa", "bb", "cc"]})

# tests left is of type dataframe
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_left_dataframe(memory_leak_check):
    def impl(df1):
        return pd.merge_asof("abc", df1)

    with pytest.raises(BodoError, match="requires dataframe inputs"):
        bodo.jit(impl)(df1)


# tests right is of type dataframe
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_right_dataframe(memory_leak_check):
    def impl(df1):
        return pd.merge_asof(df1, "abc")

    with pytest.raises(BodoError, match="requires dataframe inputs"):
        bodo.jit(impl)(df1)


# tests invalid on key in left dataframe
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_invalid_index_left(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid on key in right dataframe
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_invalid_index_right(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A", "E"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid on key in both dataframes
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_invalid_index_both(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A", "break"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests on without common cols
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_no_comm_cols(memory_leak_check):
    df3 = pd.DataFrame(
        {"AA": [1, 2, 3], "CC": ["aa", "b", "c"], "EE": ["aa", "bb", "cc"]}
    )

    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A"])

    with pytest.raises(BodoError, match="No common columns to perform merge on"):
        bodo.jit(impl)(df1, df3)


# tests lefton type
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_str_strlist2(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=(1, "A"))

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests both on and left_on specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_lefton(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A"], left_on=["C"])

    with pytest.raises(
        BodoError,
        match='Can only pass argument "on" OR "left_on" '
        'and "right_on", not a combination of both',
    ):
        bodo.jit(impl)(df1, df2)


# tests both on and lefton specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_righton(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, on=["A"], right_on=["C"])

    with pytest.raises(
        BodoError,
        match='Can only pass argument "on" OR "left_on" '
        'and "right_on", not a combination of both',
    ):
        bodo.jit(impl)(df1, df2)


# tests merging on columns with incompatible types
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_on_incompatible_dtype(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on="C", right_on="A")

    with pytest.raises(
        BodoError,
        match="You are trying to merge on column .* of .*" "and column .* of .*",
    ):
        bodo.jit(impl)(df1, df2)


# tests only left_on specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_lefton_only(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=["C"])

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests only right_on specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_righton_only(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_on=["C"])

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests invalid left_on key
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_lefton_invalid(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=["A", "B"], right_on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid right_on key
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_righton_invalid(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=["A", "E"], right_on=["A", "E"])

    with pytest.raises(BodoError, match="invalid key .* on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests lefton type
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_lefton_str_strlist1(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=3, right_on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests lefton type
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_lefton_str_strlist2(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=(1, "A"), right_on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests righton type
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_righton_str_strlist1(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_on=3, left_on=["A", "C"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests righton type
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_righton_str_strlist2(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_on=(1, "A"), left_on=["A", "C"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests unequal lengths of left_on and right_on
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_lefton_righton_len_unequal(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_on=["A"], right_on=["A", "B"])

    with pytest.raises(BodoError, match="len\(right_on\) must equal len\(left_on\)"):
        bodo.jit(impl)(df1, df2)


# tests left_index is of type bool
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_leftindex_bool(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_index="A", right_index=True)

    with pytest.raises(
        BodoError, match="argument 'left_index' should be a constant boolean"
    ):
        bodo.jit(impl)(df1, df2)


# tests right_index is of type bool
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_rightindex_bool(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_index=True, right_index="B")

    with pytest.raises(
        BodoError, match="argument 'right_index' should be a constant boolean"
    ):
        bodo.jit(impl)(df1, df2)


# tests only left_on specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_leftindex_only(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_index=True)

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests only right_on specified
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_rightindex_only(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_index=True)

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests right_index=True and len(left_on)!=1
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_rightindex_lefton(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_index=True, left_on=["A"])

    with pytest.raises(
        BodoError,
        match="right_index = True and specifying left_on is not suppported yet",
    ):
        bodo.jit(impl)(df1, df2)


# tests left_index=True and len(right_on)!=1
@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_leftindex_righton(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_index=True, right_on=["A"])

    with pytest.raises(
        BodoError,
        match="left_index = True and specifying right_on is not suppported yet",
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_suffixes_number(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, suffixes=["_x", "_y", "_z"])

    with pytest.raises(BodoError, match="number of suffixes should be exactly 2"):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_direction(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, direction="break")

    with pytest.raises(
        BodoError, match="direction parameter only supports default value"
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_by(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, by="break")

    with pytest.raises(
        BodoError, match="by parameter only supports default value None"
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_left_by(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, left_by="break")

    with pytest.raises(
        BodoError, match="left_by parameter only supports default value None"
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_right_by(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, right_by="break")

    with pytest.raises(
        BodoError, match="right_by parameter only supports default value None"
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_tolerance(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, tolerance="break")

    with pytest.raises(
        BodoError, match="tolerance parameter only supports default value None"
    ):
        bodo.jit(impl)(df1, df2)


@pytest.mark.skip("[BE-3083] asof needs to be supported with table format")
def test_merge_asof_allow_exact_matches(memory_leak_check):
    def impl(df1, df2):
        return pd.merge_asof(df1, df2, allow_exact_matches=False)

    with pytest.raises(
        BodoError,
        match="allow_exact_matches parameter only supports default value True",
    ):
        bodo.jit(impl)(df1, df2)
