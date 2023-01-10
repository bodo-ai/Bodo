# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL string functions
"""


import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    _gather_output,
    _test_equal_guard,
    check_func,
    find_nested_dispatcher_and_args,
    reduce_sum,
)
from bodo.utils.typing import BodoError


def verify_dict_encoded_in_impl(impl, args):
    # Verify get_str_arr_item_copy is in the IR and there is no intermediate
    # allocation. This function is not inlined so we must traverse several steps to get to
    # the actual IR in question.
    bodo_func = bodo.jit(parallel=True)(impl)

    # Find the is_in dispatcher in the IR
    arg_typs = tuple([bodo.typeof(arg) for arg in args])

    dispatcher, used_sig = find_nested_dispatcher_and_args(
        bodo_func, arg_typs, ("is_in", "bodo.libs.bodosql_array_kernels")
    )

    # Note: infrastructure doesn't handel defaults, so we need to manually add this to
    # the signature
    used_sig = used_sig + (bodo.bool_,)
    # Find the is_in_util dispatcher in the IR
    dispatcher, used_sig = find_nested_dispatcher_and_args(
        dispatcher,
        used_sig,
        ("is_in_util", "bodo.libs.bodosql_special_handling_array_kernels"),
    )

    # Verify we have str_arr_to_dict_str_arr in the IR. find_nested_dispatcher_and_args
    # will throw an assertion error if it doesn't exist.
    find_nested_dispatcher_and_args(
        dispatcher,
        used_sig,
        ("str_arr_to_dict_str_arr", "bodo.libs.str_arr_ext"),
        return_dispatcher=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.array(["alpha", "beta", "zeta", "pi", "epsilon", ""]),
                pd.array(["e", ""]),
            ),
            id="non_null_array_string_case",
        ),
        pytest.param(
            (
                pd.array(["hello", "world", None, "are", "you"]),
                pd.array(["hello", "world", "e"]),
            ),
            id="is_null_array_string",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        "⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹",
                        "⁽ ⁺ ⁼ ⁾",
                        "₍ ₊ ₌ ₎",
                        "₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉",
                        "½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞",
                        None,
                        None,
                    ]
                ),
                pd.array(["hello", "½⅓¼⅕", "⁽ ⁺ ⁼ ⁾", None, "⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹"]),
            ),
            id="null_array_non_ascii_string",
        ),
        pytest.param(
            (
                "x",
                pd.array(["x"]),
            ),
            id="non_null_scalar_string_case",
        ),
        pytest.param(
            (
                None,
                pd.array([None], dtype="string"),
            ),
            id="null_scalar_string",
        ),
        pytest.param(
            (
                None,
                pd.array(["hello world", "hello", "world"]),
            ),
            id="null_scalar_string_2",
        ),
        pytest.param(
            (
                pd.array([1, -10, 10]),
                pd.array([10, 0, None]),
            ),
            id="non_null_array_int_case",
        ),
        pytest.param(
            (
                pd.array([1, 3, None, -1, -3]),
                pd.array([1, 3, None]),
            ),
            id="is_null_array_int",
        ),
        pytest.param(
            (
                10,
                pd.array([10]),
            ),
            id="non_null_scalar_int_case",
        ),
        pytest.param(
            (
                None,
                pd.array([10]),
            ),
            id="null_scalar_int",
        ),
        pytest.param(
            (
                None,
                pd.array([10, 1, None]),
            ),
            id="null_scalar_int",
        ),
    ],
)
def test_is_in_string_int(memory_leak_check, args):

    arr, search_vals = args

    def impl(arr, search_vals):
        return bodo.libs.bodosql_array_kernels.is_in(arr, search_vals)

    if isinstance(
        arr,
        (
            pd.Series,
            np.ndarray,
            pd.core.arrays.base.ExtensionArray,
            pd.core.arrays.PandasArray,
        ),
    ):

        def is_in_equiv_fn(arr, search_vals):
            out = pd.array(arr.isin(search_vals))
            out._mask = pd.isna(arr)
            return out

        expected_answer = is_in_equiv_fn(arr, search_vals)
    elif arr is None:
        expected_answer = None
    else:

        def is_in_equiv_fn(arr, search_vals):
            return pd.array([arr], dtype=search_vals.dtype).isin(search_vals)[0]

        expected_answer = is_in_equiv_fn(arr, search_vals)

    check_func(
        impl,
        (arr, search_vals),
        py_output=expected_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pa.array(
                    ["alpha", "beta", "zeta", "pi", "epsilon", ""] * 10,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                pd.array(["e", "", "beta"]),
            ),
            id="non_null_array_string_case",
        ),
        pytest.param(
            (
                pa.array(
                    ["hello", "world", None, "are", "you"] * 10,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                pd.array(["hello", "world", "e"]),
            ),
            id="is_null_array_string",
        ),
        pytest.param(
            (
                pa.array(
                    [
                        "⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹",
                        "⁽ ⁺ ⁼ ⁾",
                        "₍ ₊ ₌ ₎",
                        "₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉",
                        "½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞",
                        None,
                        None,
                    ]
                    * 10,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                pd.array(["hello", "½⅓¼⅕", "⁽ ⁺ ⁼ ⁾", None, "⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹"]),
            ),
            id="null_array_non_ascii_string",
        ),
    ],
)
def test_is_in_dict_enc_string(memory_leak_check, args):

    arr, search_vals = args

    def impl(arr, search_vals):
        return bodo.libs.bodosql_array_kernels.is_in(arr, search_vals)

    def is_in_equiv_fn(arr, search_vals):
        tmp = pd.Series(arr).isin(search_vals)
        tmp[pd.Series(arr).isna()] = None
        out = pd.array(tmp).astype("boolean")
        return out

    expected_answer = is_in_equiv_fn(arr, search_vals)

    check_func(
        impl,
        (arr, search_vals),
        py_output=expected_answer,
        check_dtype=False,
        reset_index=True,
    )

    verify_dict_encoded_in_impl(
        impl,
        (arr, search_vals),
    )


@pytest.mark.slow
def test_is_in_option(memory_leak_check):

    search_vals = pd.array([1, 2, 3])
    input_arr = 1

    def impl(A, flag0):
        arg0 = A if flag0 else None
        return bodo.libs.bodosql_array_kernels.is_in(arg0, search_vals)

    for flag0 in [True, False]:
        fn_output = True if flag0 else None

        check_func(impl, (input_arr, flag0), py_output=fn_output)


def test_is_in_distribution_handling(memory_leak_check):

    # Case 1: DIST DIST -> DIST, is_parallel=True
    # Case 2: REP  REP  -> REP, is_parallel=False
    # Case 3: DIST REP  -> DIST, is_parallel=False
    # Case 4: REP  DIST:   Banned by construction

    vals_to_search = pd.array(np.arange(15))
    vals_to_search_for = pd.array([2, 3, 5, 7, 11, 13, 17])
    expected_output = pd.array(
        [
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
        ]
    )

    @bodo.jit(distributed=["retval"])
    def case_1(vals_to_search, vals_to_search_for):
        vals_to_search = bodo.scatterv(vals_to_search)
        vals_to_search_for = bodo.scatterv(vals_to_search_for)
        retval = bodo.libs.bodosql_array_kernels.is_in(
            vals_to_search, vals_to_search_for
        )
        return retval

    @bodo.jit(replicated=["retval"])
    def case_2(vals_to_search, vals_to_search_for):
        vals_to_search = bodo.allgatherv(vals_to_search)
        vals_to_search_for = bodo.allgatherv(vals_to_search_for)
        retval = bodo.libs.bodosql_array_kernels.is_in(
            vals_to_search, vals_to_search_for
        )
        return retval

    @bodo.jit(distributed=["retval"])
    def case_3(vals_to_search, vals_to_search_for):
        vals_to_search = bodo.scatterv(vals_to_search)
        vals_to_search_for = bodo.allgatherv(vals_to_search_for)
        retval = bodo.libs.bodosql_array_kernels.is_in(
            vals_to_search, vals_to_search_for
        )
        return retval

    @bodo.jit()
    def case_4(vals_to_search, vals_to_search_for):
        vals_to_search = bodo.allgatherv(vals_to_search)
        vals_to_search_for = bodo.scatterv(vals_to_search_for)
        retval = bodo.libs.bodosql_array_kernels.is_in(
            vals_to_search, vals_to_search_for
        )
        return retval

    for i, fn in enumerate([case_1, case_2, case_3]):
        orig_bodo_output = fn(vals_to_search, vals_to_search_for)

        if i in (0, 2):
            # Gather output if distributed
            bodo_output = _gather_output(orig_bodo_output)

        passed = 1
        if bodo.get_rank() == 0:
            passed = _test_equal_guard(
                bodo_output,
                expected_output,
                True,
                False,
                False,
                True,
                False,
            )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()

        bodo.barrier()

    with pytest.raises(
        BodoError, match="Output of scatterv should be a distributed array"
    ):
        orig_bodo_output = case_4(vals_to_search, vals_to_search_for)
