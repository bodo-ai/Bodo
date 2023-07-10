"""Test file for DictEncodingState used by streaming operations.
These tests are focused on the core correctness of the state operations
and any SQL APIs requiring additional testing.
"""
import pandas as pd

import bodo
from bodo.tests.utils import check_func


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
        cache_misses = 0
        arr_size = len(arr)
        batches = []
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
                )
            else:
                # Cache miss
                cache_misses += 1
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
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, cache_misses)

    arr = pd.array(["Hierq", "owerew", None, "Help", "help", "HELP", "cons"] * 2)
    py_output = (pd.Series(arr).str.lower().array, 1)
    # Only test sequential because it simplifies the test logic with pd.concat.
    check_func(
        impl,
        (pd.Series(arr),),
        only_seq=True,
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
        def process_arr(dict_encoding_state, arr, batches, cache_misses):
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
                    )
                else:
                    # Cache miss
                    cache_misses += 1
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
            return cache_misses

        arr1 = bodo.hiframes.pd_series_ext.get_series_data(S1)
        arr2 = bodo.hiframes.pd_series_ext.get_series_data(S2)
        dict_encoding_state = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
        cache_misses = 0
        batches = []
        cache_misses = process_arr(dict_encoding_state, arr1, batches, cache_misses)
        cache_misses = process_arr(dict_encoding_state, arr2, batches, cache_misses)
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, cache_misses)

    arr1 = pd.array(["Hierq", "owerew", None, "Help", "help", "HELP", "cons"] * 2)
    arr2 = pd.array(["qwew", "fe", None, "Help"] * 2)
    py_output = (
        pd.concat((pd.Series(arr1).str.lower(), pd.Series(arr2).str.lower())).array,
        2,
    )
    # Only test sequential because it simplifies the test logic with pd.concat.
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
        cache_misses = 0
        arr_size = len(arr)
        batches = []
        while not finished:
            section = arr[batch_num * slice_size : (batch_num + 1) * slice_size]
            finished = ((batch_num + 1) * slice_size) >= arr_size
            if bodo.libs.stream_dict_encoding.state_contains_dict_array(
                dict_encoding_state, func_id1, section._dict_id
            ):
                # Cache hit
                new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                    dict_encoding_state,
                    func_id1,
                    section._dict_id,
                )
            else:
                # Cache miss
                cache_misses += 1
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
                # Cache hit
                new_dict, new_dict_id = bodo.libs.stream_dict_encoding.get_array(
                    dict_encoding_state,
                    func_id2,
                    out_arr1._dict_id,
                )
            else:
                # Cache miss
                cache_misses += 1
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
        bodo.libs.stream_dict_encoding.delete_dict_encoding_state(dict_encoding_state)
        out_arr = bodo.libs.array_kernels.concat(batches)
        return (out_arr, cache_misses)

    arr = pd.array(
        [" Hierq", "owerew ", None, " Help", "help  ", "HELP", "   cons   "] * 2
    )
    py_output = (pd.Series(arr).str.lower().str.lstrip().array, 2)
    # Only test sequential because it simplifies the test logic with pd.concat.
    check_func(
        impl,
        (pd.Series(arr),),
        only_seq=True,
        py_output=py_output,
        use_dict_encoded_strings=True,
    )
