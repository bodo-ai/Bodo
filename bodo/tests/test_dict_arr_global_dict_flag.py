import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.libs.array import drop_duplicates_local_dictionary
from bodo.tests.utils import check_func

data = []
for str in ["HeLlo Wor1d!", "aBcDEfg", "muHAhaHAha"]:
    data.append(str)
    data.append(str.title())
    data.append(str.upper())
    data.append(str.lower())
    data.append(None)

capitalization_array = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))

data = []
for i in range(10):
    data.append("0" * i)
pad_array = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))

data = []
for i in range(10):
    padstr = "0" * i
    data.append(f"{padstr}hello world{padstr}")
    data.append(f"{padstr}hello world")
    data.append(f"hello world{padstr}")

strip_array = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))


@pytest.mark.parametrize(
    "args",
    [
        # Capitalization functions
        pytest.param(("capitalize", (), (), capitalization_array), id="capitalize"),
        pytest.param(("upper", (), (), capitalization_array), id="upper"),
        pytest.param(("lower", (), (), capitalization_array), id="lower"),
        pytest.param(("title", (), (), capitalization_array), id="title"),
        # Pad functions
        pytest.param(("center", (10, "0"), (), capitalization_array), id="center"),
        pytest.param(("rjust", (10, "0"), (), capitalization_array), id="rjust"),
        pytest.param(("ljust", (10, "0"), (), capitalization_array), id="ljust"),
        pytest.param(("zfill", (10,), (), capitalization_array), id="zfill"),
        pytest.param(("pad", (10, "left", "0"), (), capitalization_array), id="pad"),
        # Strip functions
        pytest.param(("lstrip", ("0",), (), strip_array), id="lstrip"),
        pytest.param(("rstrip", ("0",), (), strip_array), id="rstrip"),
        pytest.param(("strip", ("0",), (), strip_array), id="strip"),
        # Other functions
        pytest.param(
            (
                "slice",
                (
                    2,
                    1,
                    3,
                ),
                (),
                pad_array,
            ),
            id="slice",
        ),
        pytest.param(
            (
                "replace",
                (
                    "AB*",
                    "EE",
                ),
                ("regex=True",),
                strip_array,
            ),
            id="replace",
        ),
        pytest.param(("repeat", (0,), (), pad_array), id="repeat"),
        pytest.param(
            ("extract", (r"(hello world)",), ("expand=False",), pad_array), id="extract"
        ),
    ],
)
@pytest.mark.slow
def test_dict_enc_ops_set_global_dict_flag(args, memory_leak_check):
    """tests Series.str.x functions that can result in output dictionaries with duplicate
    values in the data array correctly set _has_deduped_local_dictionary=False
    and keeps _has_global_dictionary=True"""
    fn_name, fn_args, named_params, input_arr = args
    args_str = ", ".join([repr(x) for x in fn_args] + [x for x in named_params])
    func_text = (
        "def bodo_impl(A):\n"
        "  global_A = drop_duplicates_local_dictionary(A, False)\n"
        f"  output_series = pd.Series(global_A).str.{fn_name}({args_str})\n"
        "  return output_series.values._has_deduped_local_dictionary, output_series\n"
    )
    func_text += (
        "def py_impl(A):\n" f"  return pd.Series(A).str.{fn_name}({args_str})\n"
    )
    loc_vars = {}
    global_vars = {
        "pd": pd,
        "drop_duplicates_local_dictionary": drop_duplicates_local_dictionary,
    }
    exec(func_text, global_vars, loc_vars)
    py_impl = loc_vars["py_impl"]
    bodo_impl = loc_vars["bodo_impl"]

    check_func(
        bodo_impl,
        (input_arr,),
        py_output=(False, py_impl(pd.Series(input_arr))),
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )


@pytest.mark.slow
def test_str_extractall_sets_global_dict_flag(memory_leak_check):
    """tests Series.str.extractall() correctly sets _has_global_dictionary for dict array
    inputs/outputs.
    """

    # non-string index, single group
    def impl1(A):
        global_A = drop_duplicates_local_dictionary(A, False)
        output_df = pd.Series(global_A, name="AA").str.extractall(r"(?P<BBB>[abd]+)\d+")
        all_cols_not_unique = True
        for col in output_df.columns:
            all_cols_not_unique = (
                all_cols_not_unique
                and not output_df[col].values._has_deduped_local_dictionary
            )
        return all_cols_not_unique, output_df

    # non-string index, multiple groups
    def impl2(A):
        global_A = drop_duplicates_local_dictionary(A, False)
        output_df = pd.Series(global_A).str.extractall(r"([чен]+)\d+([ст]+)\d+")
        all_cols_not_unique = True
        for col in output_df.columns:
            all_cols_not_unique = (
                all_cols_not_unique
                and not output_df[col].values._has_deduped_local_dictionary
            )
        return all_cols_not_unique, output_df

    # string index, single group
    def impl3(A, I):
        global_A = drop_duplicates_local_dictionary(A, False)
        output_df = pd.Series(data=global_A, index=I).str.extractall(
            r"(?P<BBB>[abd]+)\d+"
        )
        all_cols_not_unique = True
        for col in output_df.columns:
            all_cols_not_unique = (
                all_cols_not_unique
                and not output_df[col].values._has_deduped_local_dictionary
            )
        return all_cols_not_unique, output_df

    # string index, multiple groups
    def impl4(A, I):
        global_A = drop_duplicates_local_dictionary(A, False)
        output_df = pd.Series(data=global_A, index=I).str.extractall(
            r"([чен]+)\d+([ст]+)\d+"
        )
        all_cols_not_unique = True
        for col in output_df.columns:
            all_cols_not_unique = (
                all_cols_not_unique
                and not output_df[col].values._has_deduped_local_dictionary
            )
        return all_cols_not_unique, output_df

    S1 = pd.Series(
        ["a1b1", "b1", np.nan, "a2", "c2", "ddd", "dd4d1", "d22c2"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    S2 = pd.Series(
        ["чьь1т33", "ьнн2с222", "странаст2", np.nan, "ьнне33ст3"] * 2,
        ["е3", "не3", "н2с2", "AA", "C"] * 2,
    )
    A1 = pa.array(S1, type=pa.dictionary(pa.int32(), pa.string()))
    A2 = pa.array(S2, type=pa.dictionary(pa.int32(), pa.string()))

    I1 = pd.Index(["a", "b", "e", "好", "e2", "yun", "c", "dd"])
    I2 = pd.Index(["е3", "не3", "н2с2", "AA", "C"] * 2)

    check_func(
        impl1,
        (A1,),
        py_output=(
            True,
            pd.Series(A1, name="AA").str.extractall(r"(?P<BBB>[abd]+)\d+"),
        ),
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )
    check_func(
        impl2,
        (A2,),
        py_output=(True, pd.Series(A2).str.extractall(r"([чен]+)\d+([ст]+)\d+")),
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )

    check_func(
        impl3,
        (A1, I1),
        py_output=(
            True,
            pd.Series(
                data=A1, index=["a", "b", "e", "好", "e2", "yun", "c", "dd"]
            ).str.extractall(r"(?P<BBB>[abd]+)\d+"),
        ),
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )
    check_func(
        impl4,
        (A2, I2),
        py_output=(
            True,
            pd.Series(
                data=A2, index=["е3", "не3", "н2с2", "AA", "C"] * 2
            ).str.extractall(r"([чен]+)\d+([ст]+)\d+"),
        ),
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )
    # make sure IR has the optimized function


# Data for bodosql array kernel tests
string_vals = []
test_delim = "."
for word1 in ["hello", "Hello"]:
    for word2 in ["world", "World"]:
        string_vals.append(word1 + test_delim + word2)
initcap_data = pd.Series(string_vals)


string_vals = []
rlpad_fill_val = "a"
rlpad_filler_len = 24
for i in range(rlpad_filler_len):
    string_vals.append(rlpad_fill_val * i)
lpad_rpad_data = pd.Series(string_vals)

repeat_data = pd.Series(["a", "b", "c", "d"] * 4)

string_vals = []
to_replace = "hello"
replace_with = "world"
for substr1 in ["hello", "world"]:
    for substr2 in ["hello", "world"]:
        for substr3 in ["hello", "world"]:
            string_vals.append(substr1 + substr2 + substr3)
replace_data = pd.Series(string_vals)

string_vals = []
first_group = "hello"
delimiter = "."
for fill_val in [
    "(*YUHDSKJFDS)",
    "NBGYU&*(OIK",
    "][098uygb1",
    "=-09iuhnm",
    "098uyhgbnmk",
]:
    string_vals.append(first_group + delimiter + fill_val)
split_strtok_data = pd.Series(string_vals * 4)


string_vals = []
common_substring = "hello_world"
filler_len = 10
for word in ["sdfadsfasd", "asdkjhg323", "0987uytgfg"]:
    assert len(word) == filler_len  # need to be same len for substr impl
    string_vals.append(common_substring + "." + word + "." + common_substring)
left_right_substring_data = pd.Series(string_vals * 4)


string_vals = []
for letter1 in ["a", "x"]:
    for letter2 in ["b", "y"]:
        for letter3 in ["c", "z"]:
            string_vals.append(letter1 + letter2 + letter3)
translate_data = pd.Series(string_vals * 4)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("initcap", (".",), (), initcap_data), id="capitalize"),
        pytest.param(("lpad", (rlpad_filler_len, "a"), (), lpad_rpad_data), id="lpad"),
        pytest.param(("rpad", (rlpad_filler_len, "a"), (), lpad_rpad_data), id="rpad"),
        pytest.param(("repeat", (0,), (), repeat_data), id="repeat"),
        pytest.param(
            ("replace", (to_replace, replace_with), (), replace_data), id="replace"
        ),
        pytest.param(
            ("split_part", (delimiter, 1), (), split_strtok_data), id="split_part"
        ),
        pytest.param(("strtok", (delimiter, 1), (), split_strtok_data), id="strtok"),
        pytest.param(
            ("left", (len(common_substring),), (), left_right_substring_data), id="left"
        ),
        pytest.param(
            ("right", (len(common_substring),), (), left_right_substring_data),
            id="right",
        ),
        pytest.param(
            (
                "substring",
                (
                    1,
                    len(common_substring),
                ),
                (),
                left_right_substring_data,
            ),
            id="substring",
        ),
        pytest.param(
            (
                "substring_index",
                (
                    ".",
                    1,
                ),
                (),
                left_right_substring_data,
            ),
            id="substring_index",
        ),
        pytest.param(
            (
                "translate",
                (
                    "abc",
                    "xyz",
                ),
                (),
                translate_data,
            ),
            id="translate",
        ),
    ],
)
@pytest.mark.slow
def test_bodosql_array_kernels(args, memory_leak_check):
    """tests that the bodosql array kernels correctly set _has_deduped_local_dictionary for dict array
    inputs/outputs.
    """
    fn_name, fn_args, named_params, input_arr = args
    args_str = ", ".join([repr(x) for x in fn_args] + [x for x in named_params])
    func_text = (
        "def bodo_impl(A):\n"
        "  global_A = drop_duplicates_local_dictionary(A.values, False)\n"
        f"  out_array = bodo.libs.bodosql_array_kernels.{fn_name}(global_A, {args_str})\n"
        "  return out_array._has_deduped_local_dictionary, pd.Series(out_array)\n"
    )
    func_text += (
        "def py_impl(A):\n"
        f"  return pd.Series(bodo.libs.bodosql_array_kernels.{fn_name}(A, {args_str}))\n"
    )
    loc_vars = {}
    global_vars = {
        "pd": pd,
        "bodo": bodo,
        "drop_duplicates_local_dictionary": drop_duplicates_local_dictionary,
    }
    exec(func_text, global_vars, loc_vars)
    py_impl = bodo.jit(loc_vars["py_impl"])
    bodo_impl = loc_vars["bodo_impl"]

    check_func(
        bodo_impl,
        (input_arr,),
        py_output=(False, py_impl(pd.Series(input_arr))),
        use_dict_encoded_strings=True,
        # drop_duplicates_local_dictionary doesn't have dist support in the main IR
        only_seq=True,
    )
