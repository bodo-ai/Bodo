#pragma once
#include <deque>
#include <memory>
#include <vector>

#include "_bodo_common.h"
#include "_query_profile_collector.h"
#include "_stl.h"

struct ArrayBuildBuffer;

/**
 * @brief Metrics for a DictionaryBuilder. This tracks statistics such as cache
 * misses, time spent unifying, time spent transposing, etc.
 *
 */
struct DictBuilderMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    // Track the number of times we can't cache the dictionary
    // unification because we've never seen the input dictionary before
    stat_t unify_cache_id_misses = 0;
    // Track the number of times we can't cache the dictionary unification
    // because the input dictionary is longer than when we last saw it
    stat_t unify_cache_length_misses = 0;

    // Same, but for TransposeExisting:
    stat_t transpose_filter_cache_id_misses = 0;
    stat_t transpose_filter_cache_length_misses = 0;

    /// Timers for UnifyDictionaryArray
    time_t unify_build_transpose_map_time = 0;
    time_t unify_transpose_time = 0;
    // Time spent transposing STRING arrays (instead of the regular DICT case)
    time_t unify_string_arr_time = 0;

    /// Timers for TransposeExisting
    time_t transpose_filter_build_transpose_map_time = 0;
    time_t transpose_filter_transpose_time = 0;
    // Time spent transposing STRING arrays (instead of the regular DICT case)
    time_t transpose_filter_string_arr_time = 0;

    /**
     * @brief Add metrics from another set of dictionary-builder metrics. This
     * is useful in two situations:
     * 1. Combining metrics from children dict builders.
     * 2. Combining metrics from multiple columns to get metrics across all
     * columns of a table.
     *
     * @param src_metrics The metrics to add.
     */
    void add_metrics(const DictBuilderMetrics& src_metrics);

    /**
     * @brief Subtract 'src_metrics' from our metrics. All src_metrics must be
     * smaller or equal to our corresponding metrics. This is useful when we may
     * want to get a "diff" of metrics at two points in time. e.g. In case of
     * join, we take a snapshot of metrics for key columns at the end of the
     * build stage and then subtract that from the metrics at the end of the
     * probe stage. This gives us the "true" metrics for the probe phase.
     *
     * @param src_metrics The metrics to subtract.
     */
    void subtract_metrics(const DictBuilderMetrics& src_metrics);
};

/**
 * @brief Similar to ArrayBuildBuffer, but for incrementally building a
 * dictionary (the string portion of a dictionary encoded array).
 * This is meant to be shared between multiple arrays.
 * e.g. In Hash Join, for DICT key columns, this dictionary is shared
 * between the build_table_buffer and probe_table_buffer of all
 * partitions, as well as the shuffle buffers for both tables.
 *
 * NOTE: The id of a dictionary will change at most once during it's lifetime,
 * which is when it goes from being an empty to a non-empty dictionary. As a
 * result to record that a dictionary has previously been seen, you MUST record
 * both the id and the length.
 */
struct DictionaryBuilder {
    // NOTE: dictionary state can be shared across buffers that require the same
    // dictionary hash table used for dictionary unification of new batches
    // (only for dictionary-encoded string arrays)
    std::shared_ptr<bodo::unord_map_container<std::string, dict_indices_t,
                                              string_hash, std::equal_to<>>>
        dict_str_to_ind;
    // dictionary buffer to allow appending new dictionary values (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_buff;
    // hashes of dictionary elements (not actual array elements) to allow
    // consistent hashing (only for dictionary-encoded string arrays that are
    // key columns)
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes;
    // Input is a key column of its table. This essentially decides
    // whether or not we maintain 'dict_hashes'.
    const bool is_key;

    // Dictionary builders for children of nested arrays in case there are
    // dictionary-encoded string arrays
    std::vector<std::shared_ptr<DictionaryBuilder>> child_dict_builders;

    DictBuilderMetrics metrics;

    /**
     * @brief Construct a new Dictionary Builder.
     *
     * @param dict String array to store the dictionary in. This
     * expects an empty array.
     * @param is_key_ Whether this is a key column. This essentially
     * decided whether we will compute and maintain the hashes for
     * the values in this dictionary.
     * @param child_dict_builders_ Dictionary builders for children of nested
     * arrays in case there are dictionary-encoded string arrays
     * @param transpose_cache_size The number of array ids to cache transpose
     * information for.
     * @param filter_transpose_cache_size The number of array ids to cache
     * transpose filter information for (TransposeExisting API used by Runtime
     * Join Filters).
     *   NOTE: A size 3 cache is sufficient since for a given
     *   column there can only ever be 1 column-level PandasJoinFilter that can
     *   be generated. The 2nd one is for the case where all the columns are
     *   available and we need to transpose for calculating the hashes for the
     *   bloom filter. The 3rd one is for use by the Join Probe step (which will
     *   essentially replace the regular transpose cache usage).
     */
    DictionaryBuilder(std::shared_ptr<array_info> dict, bool is_key_,
                      std::vector<std::shared_ptr<DictionaryBuilder>>
                          child_dict_builders_ = {},
                      size_t transpose_cache_size = 2,
                      size_t filter_transpose_cache_size = 3);

    ~DictionaryBuilder() {}

    /**
     * @brief Unify dictionary of input array with buffer by appending its new
     * dictionary values to buffer's dictionary and transposing input's indices.
     * It is guaranteed that the output array will be locally unique.
     *
     * @param in_arr input array
     * @return std::shared_ptr<array_info> input array with its dictionary
     * replaced and indices transposed
     */
    std::shared_ptr<array_info> UnifyDictionaryArray(
        const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Transform the input array by replacing its dictionary with the
     * dictionary of this Dictionary builder and transposing the indices
     * accordingly. Unlike UnifyDictionaryArray, the DictionaryBuilder's
     * dictionary isn't modified. Instead, if the input contains a string that
     * is not already in the DictionaryBuilder's dictionary, we set the row as
     * null in the output array. Because of this, this essentially acts like a
     * filtering transpose.
     * NOTE: The function cannot be marked as 'const' because we modify the
     * cache.
     *
     * @param in_arr Input array to transpose.
     * @return std::shared_ptr<array_info> Transposed array (with nulls where
     * there were entries that weren't in the DictionaryBuilder).
     */
    std::shared_ptr<array_info> TransposeExisting(
        const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Get dictionary hashes
     *
     * @return std::shared_ptr<bodo::vector<uint32_t>> dictionary hashes or null
     * pointer
     */
    std::shared_ptr<bodo::vector<uint32_t>> GetDictionaryHashes();

    /**
     * @brief Get the index for idx'th row in 'in_arr'. Returns -1 if the string
     * doesn't exist in the DictionaryBuilder.
     *
     * @param in_arr STRING array to get the element from.
     * @param idx Row ID to get the index of.
     * @return dict_indices_t Index of the string in the dictionary or -1 if it
     * doesn't exist in the dictionary.
     */
    dict_indices_t GetIndex(const std::shared_ptr<array_info>& in_arr,
                            size_t idx) const noexcept;
    /**
     * @brief Ensure in_arr[idx] is in the dictionary and return the index for
     * it
     * @param in_arr array of strings
     * @param idx index into in_arr for the value we want in the dictionary
     * @return index of the existing entry for in_arr[idx] or the index of the
     * newly created entry for it.
     */
    dict_indices_t InsertIfNotExists(const std::shared_ptr<array_info>& in_arr,
                                     size_t idx);

    /**
     * @brief Ensure value is in the dictionary and return the index for it
     * @param value string value to insert
     * @return index of the existing entry for value or the index of the newly
     * created entry for it.
     */
    dict_indices_t InsertIfNotExists(const std::string_view& value);

    /**
     * @brief Get the Metrics for this DictionaryBuilder. In case of nested
     * types, this will combine metrics from all children dict builders and
     * return a single set of metrics.
     *
     * @return DictBuilderMetrics
     */
    DictBuilderMetrics GetMetrics() const;

   private:
    /**
     * @brief a unique identifier for a dictionary at a point in time
     */
    struct DictionaryID {
        // array ID of the underlying data array
        int64_t arr_id;
        // length of the data array
        size_t length;
    };
    using DictionaryCache =
        std::deque<std::pair<DictionaryID, std::vector<dict_indices_t>>>;
    // Caching information for UnifyDictionaryArray. Snowflake Batches
    // should follow a pattern where several input batches in a row all use
    // the same dictionary. When we encounter a new array we will
    // cache the transpose information to limit the compute per batch.
    // The first element of the pair is the array id, the second is the
    // transpose map.
    // This must be kept sorted in eviction order to ensure the oldest
    // entry is evicted.
    DictionaryCache cached_array_transposes;

    // Add cache_entry to the cache.
    const std::vector<int>& _AddToCache(
        std::pair<DictionaryID, std::vector<int>> cache_entry);

    // Move an element from any position in the cache to the front.
    const std::vector<int>& _MoveToFrontOfCache(
        DictionaryCache::iterator cache_entry);

    // Caching information for TransposeExisting. This is used by runtime join
    // filters (see HashJoinState::RuntimeFilter).
    // Inputs should follow a pattern where several input batches in a row all
    // use the same dictionary. When we encounter a new array we will cache the
    // transpose information to limit the compute per batch. The first element
    // of the pair is the array id, the second is the transpose map. This must
    // be kept sorted in eviction order to ensure the oldest entry is evicted.
    DictionaryCache cached_filter_array_transposes;

    // Add cache_entry to the cache.
    const std::vector<int>& _AddToFilterCache(
        std::pair<DictionaryID, std::vector<int>> cache_entry);

    // Move an element from any position in the cache to the front.
    const std::vector<int>& _MoveToFrontOfFilterCache(
        DictionaryCache::iterator cache_entry);

    /**
     * @brief Helper function to transpose the input array as per the provided
     * transpose map.
     *
     * @param in_arr Input DICT array to transpose.
     * @param transpose_map Transpose map to use for the transpose operation.
     * @return std::shared_ptr<array_info> Transposed DICT array.
     */
    std::shared_ptr<array_info> transpose_input_helper(
        const std::shared_ptr<array_info>& in_arr,
        const std::vector<int>& transpose_map) const;
};

/**
 * @brief Creates a dictionary builder for dict-encoded array types or nested
 * arrays but returns nullptr for other types.
 *
 * @param t array type
 * @param is_key flag for key arrays (e.g. in join), which enables storing
 * hashes in DictionaryBuilder
 * @return std::shared_ptr<DictionaryBuilder> dictionary builder or nullptr
 */
std::shared_ptr<DictionaryBuilder> create_dict_builder_for_array(
    const std::shared_ptr<bodo::DataType>& t, bool is_key);

/// @brief same as above but takes array_info as input
std::shared_ptr<DictionaryBuilder> create_dict_builder_for_array(
    const std::shared_ptr<array_info>& arr, bool is_key);

/**
 * @brief Set dictionary of dictionary-encoded array to internal dictionary of
 * DictionaryBuilder. Handles nested arrays as well. This is used in streaming
 * join for dummy probe table.
 *
 * @param arr input array
 * @param builder input dictionary builder
 */
void set_array_dict_from_builder(
    std::shared_ptr<array_info>& arr,
    const std::shared_ptr<DictionaryBuilder>& builder);

/**
 * @brief Set dictionary of dictionary-encoded array to dictionary of another
 * array. Handles nested arrays as well.
 *
 * @param out_arr output array to set dictionary
 * @param in_arr input array to copy dictionary
 */
void set_array_dict_from_array(std::shared_ptr<array_info>& out_arr,
                               const std::shared_ptr<array_info>& in_arr);
