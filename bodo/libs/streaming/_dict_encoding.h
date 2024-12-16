#pragma once

#include "../_bodo_common.h"
#include "../_stl.h"

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

class DictEncodingState {
   public:
    DictEncodingState();

    /**
     * @brief Determine if the state contains the output of the dictionary array
     * for the given function.
     *
     * @param func_id The id of the function to find.
     * @param dict_id The dictionary id to check.
     * @return The length of the input dictaionary if the output is found,
     * otherwise return std::nullopt.
     */
    std::optional<size_t> contains(int64_t func_id, int64_t dict_id);

    /**
     * @brief Get the cached dictionary array. This code assumes that
     * contains has already returned true.
     *
     * @param func_id The function id for finding the array.
     * @param cache_dict_id The id that should be cached.
     * @return std::tuple<std::shared_ptr<array_info>, int64_t, int64_t> The
     * array to return, the new dict id, and the length of the dict at the time
     * of caching.
     */
    std::tuple<std::shared_ptr<array_info>, int64_t, int64_t> get_array(
        int64_t func_id, int64_t cache_dict_id);

    /**
     * @brief Update state with a new cached value for the given function.
     *
     * @param func_id id of the function.
     * @param cache_dict_id id of the dictionary that produced this array.
     * @param cache_dict_len length of the dictionary that produced this array.
     * @param arr Output array to cache.
     * @param new_dict_id id of the new array. -1 if this is not a string array.
     */
    void set_array(int64_t func_id, int64_t cache_dict_id,
                   size_t cache_dict_len, std::shared_ptr<array_info> arr,
                   int64_t new_dict_id);

    /**
     * @brief Determine if the state contains the output of the dictionary array
     * for the given function.
     *
     * @param func_id The id of the function to find.
     * @param dict_ids The dictionary ids to check.
     * @param dict_lengths The dictionary lengths to check.
     * @return Has the state definitely cached this dictionary array.
     */
    bool contains_multi_input(int64_t func_id, std::vector<int64_t> dict_ids,
                              std::vector<int64_t> dict_lengths);

    /**
     * @brief Get the cached dictionary array for a kernels that depends
     * on multiple input dictionaries. This code assumes that
     * contains_multi_input has already returned true.
     *
     * @param func_id The function id for finding the array.
     * @param cache_dict_ids The ids that should be cached.
     * @param cache_dict_lengths Lengths of the dictionary IDs being cached.
     * @return std::tuple<std::shared_ptr<array_info>, int64_t> The array to
     * return and the new dict id.
     */
    std::tuple<std::shared_ptr<array_info>, int64_t> get_array_multi_input(
        int64_t func_id, std::vector<int64_t> cache_dict_ids,
        std::vector<int64_t> cache_dict_lengths);

    /**
     * @brief Update state with a new cached value for the given function.
     *
     * @param func_id id of the function.
     * @param cache_dict_ids The ids of the dictionaries that produced this
     * array.
     * @param cache_dict_lengths The lengths of the dictionaries that produced
     * this array.
     * @param arr Output array to cache.
     * @param new_dict_id id of the new array. -1 if this is not a string array.
     */
    void set_array_multi_input(int64_t func_id,
                               std::vector<int64_t> cache_dict_ids,
                               std::vector<int64_t> cache_dict_lengths,
                               std::shared_ptr<array_info> arr,
                               int64_t new_dict_id);

    /**
     * @brief Return the number of times set_array or set_array_multi_input was
     * called.
     *
     * @return int64_t num_set_calls
     */
    int64_t get_num_set_calls();

   private:
    struct CacheEntry {
        // output array
        std::shared_ptr<array_info> array;
        // dict id of the output dictionary
        int64_t output_dict_id;
        // dict id of the input dictionary
        int64_t dict_id;
        // length of the input dictionary when the function initially ran (not
        // necessarily the same length as array)
        size_t dict_len;
    };

    struct MultiCacheEntry {
        // output array
        std::shared_ptr<array_info> array;
        // dict ids of all input dictionaries
        std::vector<int64_t> dict_ids;
        // lengths of all input dictionaries when the function was initially
        // ran.
        std::vector<int64_t> dict_lengths;
        // dict id of the output dictionary
        int64_t output_dict_id;
    };

    /**
     * @brief Validate that the one_input_map contains the requested dictionary
     * output and return the output. If the dictionary is not contained the
     * output is garbage.
     *
     * @param func_id The id of the function to find.
     * @param dict_id The dictionary id to check. This function will also
     * validate the id.
     * @return std::optional<CacheEntry> Is the dictionary found in the given
     * cache? If so return the relevant entry. If not found return std::nullopt.
     */
    std::optional<CacheEntry> contains_and_value(int64_t func_id,
                                                 int64_t dict_id);

    /**
     * @brief Validate that the multi_input_map contains the requested
     * dictionary output and return the cached output. If the dictionary is not
     * contained the output is garbage.
     *
     * @param state The state to check.
     * @param func_id The id of the function to find.
     * @param dict_ids The dictionary ids to check. This function will also
     * validate the ids.
     * @return std::optional<MultiCacheEntry> Is the dictionary found in the
     * given cache? If so return the cached output. If not found return
     * std::nullopt.
     */
    std::optional<MultiCacheEntry> contains_and_value_multi_input(
        int64_t func_id, std::vector<int64_t> dict_ids,
        std::vector<int64_t> dict_lengths);

    // Hashmap for functions to operate on a single array. This to avoid paying
    // the vector penalty in the common case.
    bodo::unord_map_container<int64_t, CacheEntry> one_input_map;
    // Uncommon functions that can work with multiple dictionaries (e.g. concat,
    // coalesce)
    bodo::unord_map_container<int64_t, MultiCacheEntry> multi_input_map;
    // Count the number of times set is called. Used for testing.
    int64_t num_set_calls;
};
