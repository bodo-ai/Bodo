"""Test file for DictEncodingState used by streaming operations.
These tests are focused on the core correctness of the state operations
and any SQL APIs requiring additional testing.
"""
import textwrap
from typing import Tuple

import pandas as pd

import bodo
from bodo.tests.utils import check_func


def test_concat_allocation(memory_leak_check):
    """
    Test that concat won't allocate a new id if all dictionaries
    have the same id.
    """
    slice_size = 3

    def impl(S):
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        # Allocate ids at the start and end to determine how many ids
        # we used in this function. Concat should use 0 since its the
        # same dictionary.
        start_id = bodo.libs.dict_arr_ext.generate_dict_id(len(arr))
        finished = False
        batch_num = 0
        arr_size = len(arr)
        batches = []
        while not finished:
            section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            batches.append(section)
            batch_num += 1
        out_arr = bodo.libs.array_kernels.concat(batches)
        end_id = bodo.libs.dict_arr_ext.generate_dict_id(len(out_arr))
        num_ids = (end_id - start_id) - 1
        return (out_arr, num_ids)

    arr = pd.array(["Hierq", "owerew", None, "Help", "help", "HELP", "cons"] * 2)
    # Verify the input is unchanged and no new ids were allocated.
    py_output = (arr, 0)
    check_func(
        impl,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_basic_caching(memory_leak_check):
    """
    Test that a basic implementation with the dictionary returns
    the correct output and utilizes the cache.
    """
    func_id = 1
    slice_size = 3

    def impl(S):
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        finished = False
        batch_num = 0
        arr_size = len(arr)
        batches = []
        while not finished:
            section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            if bodo.libs.stream_dict_encoding.state_contains_dict_array(
                dict_encoding_state, func_id, section._dict_id
            ):
                new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                    dict_encoding_state,
                    func_id,
                    section._dict_id,
                    bodo.string_array_type,
                )
            else:
                new_dict = bodo.libs.bodosql_array_kernels.lower(section._data)
                new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(len(new_dict))
                # Update the cache
                bodo.libs.stream_dict_encoding.set_array(
                    dict_encoding_state,
                    func_id,
                    section._dict_id,
                    new_dict,
                    new_dict_id,
                )
            out_arr = bodo.libs.dict_arr_ext.init_dict_arr(
                new_dict,
                section._indices,
                section._has_global_dictionary,
                False,
                new_dict_id,
            )
            batches.append(out_arr)
            batch_num += 1
        num_sets = bodo.libs.stream_dict_encoding.get_state_num_set_calls(
            dict_encoding_state
        )
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, num_sets)

    arr = pd.array(["Hierq", "owerew", None, "Help", "help", "HELP", "cons"] * 2)
    py_output = (pd.Series(arr).str.lower().array, 1)
    check_func(
        impl,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_multi_dictionary(memory_leak_check):
    """
    Test that when using processing different dictionaries in batches,
    each dictionary only results in a single cache miss.
    """
    func_id = 1
    slice_size = 3

    def impl(S1, S2):
        def process_arr(dict_encoding_state, arr, batches):
            arr_size = len(arr)
            batch_num = 0
            finished = False
            while not finished:
                section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
                finished = ((batch_num + 1) * slice_size) >= arr_size
                if bodo.libs.stream_dict_encoding.state_contains_dict_array(
                    dict_encoding_state, func_id, section._dict_id
                ):
                    # Cache hit
                    new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                        dict_encoding_state,
                        func_id,
                        section._dict_id,
                        bodo.string_array_type,
                    )
                else:
                    new_dict = bodo.libs.bodosql_array_kernels.lower(section._data)
                    new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(len(new_dict))
                    # Update the cache
                    bodo.libs.stream_dict_encoding.set_array(
                        dict_encoding_state,
                        func_id,
                        section._dict_id,
                        new_dict,
                        new_dict_id,
                    )
                out_arr = bodo.libs.dict_arr_ext.init_dict_arr(
                    new_dict,
                    section._indices,
                    section._has_global_dictionary,
                    False,
                    new_dict_id,
                )
                batches.append(out_arr)
                batch_num += 1

        arr1 = bodo.hiframes.pd_series_ext.get_series_data(S1)
        arr2 = bodo.hiframes.pd_series_ext.get_series_data(S2)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        batches = []
        process_arr(dict_encoding_state, arr1, batches)
        process_arr(dict_encoding_state, arr2, batches)
        num_sets = bodo.libs.stream_dict_encoding.get_state_num_set_calls(
            dict_encoding_state
        )
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, num_sets)

    arr1 = pd.array(["Hierq", "owerew", None, "Help", "help", "HELP", "cons"] * 2)
    arr2 = pd.array(["qwew", "fe", None, "Help"] * 2)
    py_output = (
        pd.concat((pd.Series(arr1).str.lower(), pd.Series(arr2).str.lower())).array,
        2,
    )
    # TODO(njriasan): Only sequential tests work by the slices in this function aren't "local"
    # slices. As a result the code isn't logically consistent in parallel.
    check_func(
        impl,
        (pd.Series(arr1), pd.Series(arr2)),
        only_seq=True,
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_multi_function(memory_leak_check):
    """
    Tests that using multiple functions back to back do not result in cache
    conflicts and that the cached output can be used for caching in a subsequent
    function call.
    """
    func_id1 = 1
    func_id2 = 2
    slice_size = 3

    def impl(S):
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        finished = False
        batch_num = 0
        arr_size = len(arr)
        batches = []
        while not finished:
            section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            if bodo.libs.stream_dict_encoding.state_contains_dict_array(
                dict_encoding_state, func_id1, section._dict_id
            ):
                new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                    dict_encoding_state,
                    func_id1,
                    section._dict_id,
                    bodo.string_array_type,
                )
            else:
                new_dict = bodo.libs.bodosql_array_kernels.lower(section._data)
                new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(len(new_dict))
                # Update the cache
                bodo.libs.stream_dict_encoding.set_array(
                    dict_encoding_state,
                    func_id1,
                    section._dict_id,
                    new_dict,
                    new_dict_id,
                )
            out_arr1 = bodo.libs.dict_arr_ext.init_dict_arr(
                new_dict,
                section._indices,
                section._has_global_dictionary,
                False,
                new_dict_id,
            )
            if bodo.libs.stream_dict_encoding.state_contains_dict_array(
                dict_encoding_state, func_id2, out_arr1._dict_id
            ):
                new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                    dict_encoding_state,
                    func_id2,
                    out_arr1._dict_id,
                    bodo.string_array_type,
                )
            else:
                new_dict = bodo.libs.bodosql_array_kernels.ltrim(out_arr1._data, " ")
                new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(len(new_dict))
                # Update the cache
                bodo.libs.stream_dict_encoding.set_array(
                    dict_encoding_state,
                    func_id2,
                    out_arr1._dict_id,
                    new_dict,
                    new_dict_id,
                )
            out_arr2 = bodo.libs.dict_arr_ext.init_dict_arr(
                new_dict,
                out_arr1._indices,
                out_arr1._has_global_dictionary,
                False,
                new_dict_id,
            )
            batches.append(out_arr2)
            batch_num += 1
        num_sets = bodo.libs.stream_dict_encoding.get_state_num_set_calls(
            dict_encoding_state
        )
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, num_sets)

    arr = pd.array(
        [" Hierq", "owerew ", None, " Help", "help  ", "HELP", "   cons   "] * 2
    )
    py_output = (pd.Series(arr).str.lower().str.lstrip().array, 2)
    check_func(
        impl,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_coalesce(memory_leak_check):
    """
    Tests that coalesce works when supplying dictionary inputs and caching state.
    """
    func_id = 1
    slice_size = 3

    def impl(S1, S2):
        arr1 = bodo.hiframes.pd_series_ext.get_series_data(S1)
        arr2 = bodo.hiframes.pd_series_ext.get_series_data(S2)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        finished = False
        batch_num = 0
        arr_size = len(arr1)
        batches = []
        while not finished:
            section1 = arr1[batch_num * slice_size : (batch_num + 1) * slice_size]
            section2 = arr2[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            out_batch = bodo.libs.bodosql_array_kernels.coalesce(
                (section1, section2), dict_encoding_state, func_id
            )
            batches.append(out_batch)
            batch_num += 1
        num_sets = bodo.libs.stream_dict_encoding.get_state_num_set_calls(
            dict_encoding_state
        )
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return out_arr, num_sets

    arr1 = pd.array(["Hierq", "owerew", None, "Help", None, "HELP", "cons"] * 2)
    arr2 = pd.array(["Hieq", None, None, "HelP", "heLp", "HELP", None] * 2)
    py_output = (
        pd.array([arr2[i] if pd.isna(arr1[i]) else arr1[i] for i in range(len(arr1))]),
        1,
    )
    check_func(
        impl,
        (pd.Series(arr1), pd.Series(arr2)),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def _build_1_arg_streaming_function(
    output_generation_text: str, additional_arg_names: Tuple[str] = ()
):
    """Generate the implementation of a 1 argument streaming function for testing where
    the only difference is the "function call that generates the output.

    This function assumes a single Series input (that will be converted to an array)
    and can have additional scalar arguments.

    Args:
        output_generation_text (str): Block of code inserted to update the output array.
        This should not include an output variable name.
        additional_args (Tuple[str]): Names of any additional scalars.

    Return:
        A function that can be executed in a Bodo streaming fashion. This function always returns
        2 arguments
            - An output array.
            - The number of set calls to the dictionary state (e.g. cache misses).
    """
    additional_args_str = ", ".join(additional_arg_names)

    func_id = 1
    slice_size = 3

    func_text = textwrap.dedent(
        f"""
    def impl(S, {additional_args_str}):
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        finished = False
        batch_num = 0
        arr_size = len(arr)
        batches = []
        while not finished:
            section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            out_batch = {output_generation_text}
            batches.append(out_batch)
            batch_num += 1
        num_sets = bodo.libs.stream_dict_encoding.get_state_num_set_calls(
            dict_encoding_state
        )
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return out_arr, num_sets
    """
    )
    glbls = {"bodo": bodo, "func_id": func_id, "slice_size": slice_size}
    local_vars = {}
    exec(func_text, glbls, local_vars)
    return local_vars["impl"]


def test_like(memory_leak_check):
    """
    Tests that like works when supplying a dictionary input and caching state.
    """

    impl1 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.like_kernel(section, "h%", "", False, dict_encoding_state, func_id)'
    )
    impl2 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.like_kernel(section, pattern, "", True, dict_encoding_state, func_id)',
        ("pattern",),
    )

    arr = pd.array(["hierq", "owerew", None, "Help", None, "HELP", "cons"] * 2)
    py_output = (
        pd.array([True, False, None, False, None, False, False] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl1,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )

    py_output = (
        pd.array([True, False, None, True, None, True, False] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl2,
        (pd.Series(arr), "h%"),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_decode(memory_leak_check):
    impl1 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.decode((section, "a", "one", "b", "two"), dict_encoding_state, func_id)'
    )
    impl2 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.decode((section, "a", 1, "b", 2), dict_encoding_state, func_id)'
    )
    arr = pd.array(["a", "b", None, "c", None, "d", "v"] * 2)
    py_output = (
        pd.array(["one", "two", None, None, None, None, None] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl1,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )
    py_output = (
        pd.array([1, 2, None, None, None, None, None] * 2, dtype="Int64"),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl2,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_concat_ws(memory_leak_check):
    impl = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.concat_ws(("a", section, "c"), ",", dict_encoding_state, func_id)'
    )
    arr = pd.array(["a", "b", None, "c", None, "def", "v"] * 2)
    py_output = (
        pd.array(["a,a,c", "a,b,c", None, "a,c,c", None, "a,def,c", "a,v,c"] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_least_greatest(memory_leak_check):
    impl1 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.least(("cat", section), dict_encoding_state, func_id)'
    )
    impl2 = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.greatest((section, "cat"), dict_encoding_state, func_id)'
    )
    arr = pd.array(["a", "b", None, "c", None, "def", "v"] * 2)
    py_output = (
        pd.array(["a", "b", None, "c", None, "cat", "cat"] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl1,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )
    py_output = (
        pd.array(["cat", "cat", None, "cat", None, "def", "v"] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl2,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )


def test_regexp_replace_cplusplus(memory_leak_check):
    """
    Test the regexp_replace code that goes to C++ properly does dictionary caching.
    """
    impl = _build_1_arg_streaming_function(
        'bodo.libs.bodosql_array_kernels.regexp_replace(section, "ad?a", "done", 1, 0, "", dict_encoding_state, func_id)'
    )
    arr = pd.array(["adabc", "fabbbea", None, "aafewfew", None, "fewf", "qeqewr"] * 2)
    py_output = (
        pd.array(["donebc", "fabbbea", None, "donefewfew", None, "fewf", "qeqewr"] * 2),
        # Number of cache misses. Should be 1 for the first iteration.
        1,
    )
    check_func(
        impl,
        (pd.Series(arr),),
        py_output=py_output,
        use_dict_encoded_strings=True,
    )
