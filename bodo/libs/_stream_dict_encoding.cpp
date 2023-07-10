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

typedef bodo::unord_map_container<
    int64_t, std::tuple<int64_t, std::shared_ptr<array_info>, int64_t>>
    DictEncodingState;

/**
 * @brief Create a new DictEncodingState to return to Python.
 *
 * @return DictEncodingState*
 */
DictEncodingState* dict_encoding_state_init_py_entry() {
    return new DictEncodingState();
}

/**
 * @brief Determine if the state contains the output of the dictionary array for
 * the given function.
 *
 * @param state The state to check.
 * @param func_id The id of the function to find.
 * @param dict_id The dictionary id to check. This function will also validate
 * the id.
 * @return Has the state definitely cached this dictionary array.
 */
bool state_contains_dict_array(DictEncodingState* state, int64_t func_id,
                               int64_t dict_id) {
    if (dict_id < 0) {
        return false;
    }
    auto result = state->find(func_id);
    if (result != state->end()) {
        return std::get<0>(result->second) == dict_id;
    }
    return false;
}

/**
 * @brief Get the cached dictionary array. This code assumes that
 * state_contains_dict_array has already returned true.
 *
 * @param state The state from which to load the dictionary.
 * @param func_id The function id for finding the array.
 * @param cache_dict_id The id that should be cached. This is used to prevent
 * errors in development.
 * @param[out] new_dict_id The pointer in which to store the new dict id. This
 * is necessary because we cannot return tuples to Python.
 * @return array_info* The array to return.
 */
array_info* get_array_py_entry(DictEncodingState* state, int64_t func_id,
                               int64_t cache_dict_id, int64_t* new_dict_id) {
    try {
        if (cache_dict_id < 0) {
            throw std::runtime_error(
                "get_array_py_entry:: Invalid dictionary id");
        }
        auto result = state->find(func_id);
        if (result == state->end() ||
            std::get<0>(result->second) != cache_dict_id) {
            throw std::runtime_error(
                "get_array_py_entry:: Dictionary expected in the cache but not "
                "found");
        }
        *new_dict_id = std::get<2>(result->second);
        return new array_info(*std::get<1>(result->second));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Update state with a new cached value for the given function.
 *
 * @param state State to update.
 * @param func_id ID of the function.
 * @param cache_dict_id ID of the dictionary that produced this array.
 * @param arr Output array to cache.
 * @param new_dict_id ID of the new array. -1 if this is not a string array.
 */
void set_array_py_entry(DictEncodingState* state, int64_t func_id,
                        int64_t cache_dict_id, array_info* arr,
                        int64_t new_dict_id) {
    state->insert_or_assign(
        func_id, std::tuple(cache_dict_id, std::shared_ptr<array_info>(arr),
                            new_dict_id));
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
    SetAttrStringFromVoidPtr(m, delete_dict_encoding_state);
    return m;
}
