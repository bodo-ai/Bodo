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
 */
struct DictionaryBuilder {
    // NOTE: dictionary state can be shared across buffers that require the same
    // dictionary hash table used for dictionary unification of new batches
    // (only for dictionary-encoded string arrays)
    std::shared_ptr<std::unordered_map<std::string, dict_indices_t, string_hash,
                                       std::equal_to<>>>
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

    /**
     * @brief Construct a new Dictionary Builder.
     *
     * @param dict String array to store the dictionary in. This
     * expects an empty array.
     * @param is_key_ Whether this is a key column. This essentially
     * decided whether we will compute and maintain the hashes for
     * the values in this dictionary.
     */
    DictionaryBuilder(std::shared_ptr<array_info> dict, bool is_key_)
        : is_key(is_key_) {
        this->dict_buff = std::make_shared<ArrayBuildBuffer>(dict);
        this->dict_hashes = std::make_shared<bodo::vector<uint32_t>>();
        this->dict_str_to_ind = std::make_shared<std::unordered_map<
            std::string, dict_indices_t, string_hash, std::equal_to<>>>();
    }

    /**
     * @brief Unify dictionary of input array with buffer by appending its new
     * dictionary values to buffer's dictionary and transposing input's indices.
     *
     * @param in_arr input array
     * @param is_key input is a key column of its table
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
};
