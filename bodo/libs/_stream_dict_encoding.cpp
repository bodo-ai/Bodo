#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * @brief Implementation of the C++ side for Dictionary Encoding streaming state
 * used by SQL Projection and Filter operators. This state object is responsible
 * for "caching" the output of calling a dictionary processing operation for a
 * given dictionary id. This implementation is basically a dictionary mapping
 * each unique function call in the code to 1 single array output. We assume
 * that dictionaries have very high temporal locality, so each function has a
 * cache of length 1, but if necessary this can be changed by updating
 * DictEncodingState and the function without any Python API changes.
 *
 * For more information check the confluence design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1402175534/Dictionary+Encoding+Parfors
 *
 */

struct DictEncodingState {
    // Hashmap for functions to operate on a single array. This to avoid paying
    // the vector penalty in the common case.
    bodo::unord_map_container<
        int64_t, std::tuple<int64_t, std::shared_ptr<array_info>, int64_t>>
        one_input_map;
    // Uncommon functions that can work with multiple dictionaries (e.g. concat,
    // coalesce)
    bodo::unord_map_container<
        int64_t,
        std::tuple<std::vector<int64_t>, std::shared_ptr<array_info>, int64_t>>
        multi_input_map;
};

/**
 * @brief Create a new DictEncodingState to return to Python.
 *
 * @return DictEncodingState*
 */
DictEncodingState* dict_encoding_state_init_py_entry() {
    return new DictEncodingState();
}

/**
 * @brief Validate that the one_input_map contains the requested dictionary
 * output and return the output. If the dictionary is not contained the output
 * is garbage.
 *
 * @param state The state to check.
 * @param func_id The id of the function to find.
 * @param dict_id The dictionary id to check. This function will also validate
 * the id.
 * @return std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
 * Is the dictionary found in the given cache. If so returns the other contents
 * (aside from the cached id). If not found this is garbage and should be
 * ignored.
 */
std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
_state_contains_and_value(DictEncodingState* state, int64_t func_id,
                          int64_t dict_id) {
    if (dict_id < 0) {
        return std::tuple(false, std::tuple(nullptr, -1));
    }
    auto result = state->one_input_map.find(func_id);
    if (result != state->one_input_map.end()) {
        return std::tuple(std::get<0>(result->second) == dict_id,
                          std::tuple(std::get<1>(result->second),
                                     std::get<2>(result->second)));
    }
    return std::tuple(false, std::tuple(nullptr, -1));
}

/**
 * @brief Determine if the state contains the output of the dictionary array for
 * the given function.
 *
 * @param state The state to check.
 * @param func_id The id of the function to find.
 * @param dict_id The dictionary id to check.
 * @return Has the state definitely cached this dictionary array.
 */
bool state_contains_dict_array(DictEncodingState* state, int64_t func_id,
                               int64_t dict_id) {
    return std::get<0>(_state_contains_and_value(state, func_id, dict_id));
}

/**
 * @brief Get the cached dictionary array. This code assumes that
 * state_contains_dict_array has already returned true.
 *
 * @param state The state from which to load the dictionary.
 * @param func_id The function id for finding the array.
 * @param cache_dict_id The id that should be cached.
 * @param[out] new_dict_id The pointer in which to store the new dict id. This
 * is necessary because we cannot return tuples to Python.
 * @return array_info* The array to return.
 */
array_info* get_array_py_entry(DictEncodingState* state, int64_t func_id,
                               int64_t cache_dict_id, int64_t* new_dict_id) {
    try {
        auto [is_valid, payload] =
            _state_contains_and_value(state, func_id, cache_dict_id);
        if (!is_valid) {
            throw std::runtime_error(
                "get_array_py_entry:: Missing cache entry");
        }
        *new_dict_id = std::get<1>(payload);
        return new array_info(*std::get<0>(payload));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Update state with a new cached value for the given function.
 *
 * @param state State to update.
 * @param func_id id of the function.
 * @param cache_dict_id id of the dictionary that produced this array.
 * @param arr Output array to cache.
 * @param new_dict_id id of the new array. -1 if this is not a string array.
 */
void set_array_py_entry(DictEncodingState* state, int64_t func_id,
                        int64_t cache_dict_id, array_info* arr,
                        int64_t new_dict_id) {
    state->one_input_map.insert_or_assign(
        func_id, std::tuple(cache_dict_id, std::shared_ptr<array_info>(arr),
                            new_dict_id));
}

/**
 * @brief Validate that the multi_input_map contains the requested dictionary
 * output and return the output. If the dictionary is not contained the output
 * is garbage.
 *
 * @param state The state to check.
 * @param func_id The id of the function to find.
 * @param dict_ids The dictionary ids to check. This function will also validate
 * the ids.
 * @param num_dict_ids How many ids to check.
 * @return std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
 * Is the dictionary found in the given cache. If so returns the other contents
 * (aside from the cached id). If not found this is garbage and should be
 * ignored.
 */
std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
_state_contains_and_value_multi_input(DictEncodingState* state, int64_t func_id,
                                      int64_t* dict_ids,
                                      uint64_t num_dict_ids) {
    auto result = state->multi_input_map.find(func_id);
    if (result != state->multi_input_map.end()) {
        std::vector<int64_t>& cached_ids = std::get<0>(result->second);
        if (num_dict_ids == cached_ids.size()) {
            for (size_t i = 0; i < num_dict_ids; i++) {
                if (dict_ids[i] < 0 || (dict_ids[i] != cached_ids[i])) {
                    return std::tuple(false, std::tuple(nullptr, -1));
                }
            }
            return std::tuple(true, std::tuple(std::get<1>(result->second),
                                               std::get<2>(result->second)));
        }
    }
    return std::tuple(false, std::tuple(nullptr, -1));
}

/**
 * @brief Determine if the state contains the output of the dictionary array for
 * the given function with multiple inputs dictionaries.
 *
 * @param state The state to check.
 * @param func_id The id of the function to find.
 * @param dict_ids The dictionary ids to check.
 * @param num_dict_ids How many ids to check.
 * @return Has the state definitely cached this dictionary array.
 */
bool state_contains_multi_input_dict_array(DictEncodingState* state,
                                           int64_t func_id, int64_t* dict_ids,
                                           uint64_t num_dict_ids) {
    return std::get<0>(_state_contains_and_value_multi_input(
        state, func_id, dict_ids, num_dict_ids));
}

/**
 * @brief Get the cached dictionary array for a kernels that depends
 * on multiple input dictionaries. This code assumes that
 * state_contains_multi_input_dict_array has already returned true.
 *
 * @param state The state from which to load the dictionary.
 * @param func_id The function id for finding the array.
 * @param cache_dict_ids The ids that should be cached.
 * @param num_cached_dict_ids How many ids should be cached.
 * @param[out] new_dict_id The pointer in which to store the new dict id. This
 * is necessary because we cannot return tuples to Python.
 * @return array_info* The array to return.
 */
array_info* get_array_multi_input_py_entry(DictEncodingState* state,
                                           int64_t func_id,
                                           int64_t* cache_dict_ids,
                                           uint64_t num_cached_dict_ids,
                                           int64_t* new_dict_id) {
    try {
        auto [is_valid, payload] = _state_contains_and_value_multi_input(
            state, func_id, cache_dict_ids, num_cached_dict_ids);
        if (!is_valid) {
            throw std::runtime_error(
                "get_array_multi_input_py_entry:: Missing cache entry");
        }
        *new_dict_id = std::get<1>(payload);
        return new array_info(*std::get<0>(payload));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Update state with a new cached value for the given function.
 *
 * @param state State to update.
 * @param func_id id of the function.
 * @param cache_dict_ids The ids of the dictionaries that produced this array.
 * @param num_cached_dict_ids How many ids where used.
 * @param arr Output array to cache.
 * @param new_dict_id id of the new array. -1 if this is not a string array.
 */
void set_array_multi_input_py_entry(DictEncodingState* state, int64_t func_id,
                                    int64_t* cache_dict_ids,
                                    uint64_t num_cached_dict_ids,
                                    array_info* arr, int64_t new_dict_id) {
    std::vector<int64_t> cache_dict_ids_copy(
        cache_dict_ids, cache_dict_ids + num_cached_dict_ids);
    state->multi_input_map.insert_or_assign(
        func_id, std::tuple(cache_dict_ids_copy,
                            std::shared_ptr<array_info>(arr), new_dict_id));
}

/**
 * @brief Delete the given state.
 *
 * @param state Object to delete.
 */
void delete_dict_encoding_state(DictEncodingState* state) { delete state; }

PyMODINIT_FUNC PyInit_stream_dict_encoding_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_dict_encoding_cpp", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, dict_encoding_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, state_contains_dict_array);
    SetAttrStringFromVoidPtr(m, get_array_py_entry);
    SetAttrStringFromVoidPtr(m, set_array_py_entry);
    SetAttrStringFromVoidPtr(m, state_contains_multi_input_dict_array);
    SetAttrStringFromVoidPtr(m, get_array_multi_input_py_entry);
    SetAttrStringFromVoidPtr(m, set_array_multi_input_py_entry);
    SetAttrStringFromVoidPtr(m, delete_dict_encoding_state);
    return m;
}
