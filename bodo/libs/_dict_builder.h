#pragma once
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"

struct ArrayBuildBuffer;

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
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes = nullptr;
    // Input is a key column of its table. This essentially decides
    // whether or not we maintain 'dict_hashes'.
    const bool is_key;

    /// @brief Tracing event for this dictionary builder.
    tracing::ResumableEvent dict_builder_event;

    // Track the number of times we can't cache the dictionary
    // unification because we've never seen the input dictionary before
    int64_t unify_cache_id_misses = 0;

    // Track the number of times we can't cache the dictionary
    // unification because the input dictionary is longer than when we last saw
    // it
    int64_t unify_cache_length_misses = 0;

    /**
     * @brief Construct a new Dictionary Builder.
     *
     * @param dict String array to store the dictionary in. This
     * expects an empty array.
     * @param is_key_ Whether this is a key column. This essentially
     * decided whether we will compute and maintain the hashes for
     * the values in this dictionary.
     * @param transpose_cache_size The number of array ids to cache transpose
     * information for.
     */
    DictionaryBuilder(std::shared_ptr<array_info> dict, bool is_key_,
                      size_t transpose_cache_size = 2);

    ~DictionaryBuilder() {
        dict_builder_event.add_attribute("Unify_Cache_ID_Misses",
                                         this->unify_cache_id_misses);
        dict_builder_event.add_attribute("Unify_Cache_Length_Misses",
                                         this->unify_cache_length_misses);
    }

    /**
     * @brief Unify dictionary of input array with buffer by appending its new
     * dictionary values to buffer's dictionary and transposing input's indices.
     * Is is guaranteed that the output array will be locally unique.
     *
     * @param in_arr input array
     * @return std::shared_ptr<array_info> input array with its dictionary
     * replaced and indices transposed
     */
    std::shared_ptr<array_info> UnifyDictionaryArray(
        const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Get dictionary hashes
     *
     * @return std::shared_ptr<bodo::vector<uint32_t>> dictionary hashes or null
     * pointer
     */
    std::shared_ptr<bodo::vector<uint32_t>> GetDictionaryHashes();

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
};
