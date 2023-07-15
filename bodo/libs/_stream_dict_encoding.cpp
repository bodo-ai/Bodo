#include "_stream_dict_encoding.h"

DictEncodingState::DictEncodingState() : num_set_calls(0) { }

std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
DictEncodingState::contains_and_value(int64_t func_id, int64_t dict_id) {
    if (dict_id < 0) {
        return std::tuple(false, std::tuple(nullptr, -1));
    }
    auto result = this->one_input_map.find(func_id);
    if (result != this->one_input_map.end()) {
        return std::tuple(std::get<0>(result->second) == dict_id,
                          std::tuple(std::get<1>(result->second),
                                     std::get<2>(result->second)));
    }
    return std::tuple(false, std::tuple(nullptr, -1));
}

std::tuple<bool, std::tuple<std::shared_ptr<array_info>, int64_t>>
DictEncodingState::contains_and_value_multi_input(
    int64_t func_id, std::vector<int64_t> dict_ids) {
    auto result = this->multi_input_map.find(func_id);
    if (result != this->multi_input_map.end()) {
        std::vector<int64_t>& cached_ids = std::get<0>(result->second);
        if (dict_ids.size() == cached_ids.size()) {
            for (size_t i = 0; i < dict_ids.size(); i++) {
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

bool DictEncodingState::contains(int64_t func_id, int64_t dict_id) {
    return std::get<0>(this->contains_and_value(func_id, dict_id));
}

std::tuple<std::shared_ptr<array_info>, int64_t> DictEncodingState::get_array(
    int64_t func_id, int64_t cache_dict_id) {
    auto [is_valid, payload] = this->contains_and_value(func_id, cache_dict_id);
    if (!is_valid) {
        throw std::runtime_error(
            "DictEncodingState::get_array: Missing cache entry");
    }
    return payload;
}

void DictEncodingState::set_array(int64_t func_id, int64_t cache_dict_id,
                                  std::shared_ptr<array_info> arr,
                                  int64_t new_dict_id) {
    this->one_input_map.insert_or_assign(
        func_id, std::tuple(cache_dict_id, arr, new_dict_id));
    this->num_set_calls += 1;
}

bool DictEncodingState::contains_multi_input(int64_t func_id,
                                             std::vector<int64_t> dict_ids) {
    return std::get<0>(this->contains_and_value_multi_input(func_id, dict_ids));
}

std::tuple<std::shared_ptr<array_info>, int64_t>
DictEncodingState::get_array_multi_input(int64_t func_id,
                                         std::vector<int64_t> cache_dict_ids) {
    auto [is_valid, payload] =
        this->contains_and_value_multi_input(func_id, cache_dict_ids);
    if (!is_valid) {
        throw std::runtime_error(
            "DictEncodingState::get_array_multi_input: Missing cache entry");
    }
    return payload;
}

void DictEncodingState::set_array_multi_input(
    int64_t func_id, std::vector<int64_t> cache_dict_ids,
    std::shared_ptr<array_info> arr, int64_t new_dict_id) {
    this->multi_input_map.insert_or_assign(
        func_id, std::tuple(cache_dict_ids, arr, new_dict_id));
    this->num_set_calls += 1;
}

int64_t DictEncodingState::get_num_set_calls() { return this->num_set_calls; }

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
 * @param dict_id The dictionary id to check.
 * @return Has the state definitely cached this dictionary array.
 */
bool state_contains_dict_array(DictEncodingState* state, int64_t func_id,
                               int64_t dict_id) {
    return state->contains(func_id, dict_id);
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
        auto [arr, out_dict_id] = state->get_array(func_id, cache_dict_id);
        *new_dict_id = out_dict_id;
        return new array_info(*arr);
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
    state->set_array(func_id, cache_dict_id, std::shared_ptr<array_info>(arr),
                     new_dict_id);
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
    std::vector<int64_t> dict_ids_copy(dict_ids, dict_ids + num_dict_ids);
    return state->contains_multi_input(func_id, std::move(dict_ids_copy));
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
        std::vector<int64_t> cache_dict_ids_copy(
            cache_dict_ids, cache_dict_ids + num_cached_dict_ids);
        auto [arr, out_dict_id] = state->get_array_multi_input(
            func_id, std::move(cache_dict_ids_copy));
        *new_dict_id = out_dict_id;
        return new array_info(*arr);
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
    state->set_array_multi_input(func_id, std::move(cache_dict_ids_copy),
                                 std::shared_ptr<array_info>(arr), new_dict_id);
}

/**
 * @brief Return how many times one of the set APIs was called on this state.
 * This is used in place of an actual hit/miss rate.
 *
 * @param state DictEncodingState
 * @return int64_t num_set_calls field.
 */
int64_t get_state_num_set_calls(DictEncodingState* state) {
    return state->get_num_set_calls();
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
    SetAttrStringFromVoidPtr(m, get_state_num_set_calls);
    SetAttrStringFromVoidPtr(m, delete_dict_encoding_state);
    return m;
}
