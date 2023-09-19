#include "_dict_builder.h"
#include "_array_hash.h"
#include "_bodo_common.h"
#include "_table_builder.h"

/* -------------------------- DictionaryBuilder --------------------------- */

inline dict_indices_t DictionaryBuilder::InsertIfNotExists(
    const std::shared_ptr<array_info>& in_arr, size_t idx) {
    char* in_data = in_arr->data1();
    offset_t* in_offsets = (offset_t*)in_arr->data2();

    offset_t start_offset = in_offsets[idx];
    offset_t end_offset = in_offsets[idx + 1];
    int64_t len = end_offset - start_offset;
    std::string_view val(&in_data[start_offset], len);

    dict_indices_t ind;

    // get existing index if already in hash table
    if (auto it = this->dict_str_to_ind->find(val);
        it != this->dict_str_to_ind->end()) {
        ind = it->second;
    } else {
        // insert into hash table if not exists
        ind = this->dict_str_to_ind->size();
        // TODO: remove std::string() after upgrade to C++23
        (*this->dict_str_to_ind)[std::string(val)] = ind;
        this->dict_buff->UnsafeAppendRow(in_arr, idx);
        if (this->is_key) {
            uint32_t hash;
            hash_string_32(&in_data[start_offset], (const int)len,
                           SEED_HASH_PARTITION, &hash);
            this->dict_hashes->emplace_back(hash);
        }
    }
    return ind;
}

DictionaryBuilder::DictionaryBuilder(std::shared_ptr<array_info> dict,
                                     bool is_key_, size_t transpose_cache_size)
    : is_key(is_key_),
      // Note: We cannot guarantee all DictionaryBuilders are created
      // the same number of times on each rank. Right now we do, but
      // in the future this could change.
      dict_builder_event("DictionaryBuilder::UnifyDictionaryArray", false),
      cached_array_transposes(
          transpose_cache_size,
          std::pair(DictionaryID{-1, 0}, std::vector<dict_indices_t>())) {
    // Dictionary build dictionaries are always unique.
    dict->is_locally_unique = true;
    this->dict_buff = std::make_shared<ArrayBuildBuffer>(dict);
    this->dict_hashes = std::make_shared<bodo::vector<uint32_t>>();
    this->dict_str_to_ind = std::make_shared<bodo::unord_map_container<
        std::string, dict_indices_t, string_hash, std::equal_to<>>>();
}

// TODO: Template this based on the type of the array
std::shared_ptr<array_info> DictionaryBuilder::UnifyDictionaryArray(
    const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != bodo_array_type::DICT &&
        in_arr->arr_type != bodo_array_type::STRING) {
        throw std::runtime_error(
            "UnifyDictionaryArray: DICT or STRING array expected");
    }

    bool empty_builder = this->dict_buff->data_array->array_id <= 0;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        // TODO(aneesh) move this out into it's own function
        std::shared_ptr<array_info> out_indices_arr =
            alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
        dict_indices_t* out_inds = (dict_indices_t*)out_indices_arr->data1();

        // copy null bitmask from input to output
        char* src_null_bitmask = in_arr->null_bitmask();
        char* dst_null_bitmask = out_indices_arr->null_bitmask();
        size_t bytes_in_bitmask = arrow::bit_util::BytesForBits(in_arr->length);
        memcpy(dst_null_bitmask, src_null_bitmask, bytes_in_bitmask);

        for (size_t i = 0; i < in_arr->length; i++) {
            // handle nulls in the dictionary
            if (!in_arr->get_null_bit(i)) {
                out_inds[i] = -1;
            } else {
                // We reserve space here instead of in InsertIfNotExists because
                // it shouldn't really make a big difference. The downside is
                // that we're checking the capacity and computing the length for
                // each element.
                this->dict_buff->ReserveArrayRow(in_arr, i);
                out_inds[i] = this->InsertIfNotExists(in_arr, i);
            }
        }

        // We only update the ID if this dictionary went from being empty to
        // non-empty.
        if (empty_builder && this->dict_buff->size > 0) {
            this->dict_buff->data_array->array_id =
                generate_array_id(this->dict_buff->data_array->length);
        }

        return create_dict_string_array(this->dict_buff->data_array,
                                        out_indices_arr);
    }

    auto iterationEvent(this->dict_builder_event.iteration());

    std::shared_ptr<array_info> batch_dict = in_arr->child_arrays[0];
    bool valid_arr_id = batch_dict->array_id >= 0;
    bool empty_arr = batch_dict->array_id == 0;
    // An empty array is automatically unified with all arrays
    if (empty_arr) {
        // Note: dict_buff->data_array is always unique.
        // Since the dictionaries match, we can set
        // is_locally_unique = True for the input.
        // This is always safe because either empty_arr = True (and
        // 0 element dictionaries are unique) or we checked unique
        // in a previous UnifyDictionaryArray call.
        batch_dict->is_locally_unique = true;
        // If the dictionaries already match and we don't need
        // to update either dictionary, we can just return without
        // transposing. Otherwise if we need to update the
        // builder we need to insert into the metadata.
        return in_arr;
    }

    auto cache_entry = std::find_if(
        this->cached_array_transposes.begin(),
        this->cached_array_transposes.end(),
        [batch_dict](const std::pair<DictionaryID, std::vector<int>> entry) {
            return entry.first.arr_id == batch_dict->array_id;
        });

    auto do_unify = [&](std::vector<int>& new_transpose_map) {
        // TODO make this only reserve for new elements
        this->dict_buff->ReserveArray(batch_dict);

        // Check/update dictionary hash table and create transpose map
        // if new_transpose_map is already populated, only consider elements
        // that come after it ends
        for (size_t i = new_transpose_map.size(); i < batch_dict->length; i++) {
            // handle nulls in the dictionary
            if (!batch_dict->get_null_bit(i)) {
                new_transpose_map.emplace_back(-1);
                continue;
            }
            new_transpose_map.emplace_back(
                this->InsertIfNotExists(batch_dict, i));
        }
    };
    std::vector<int> new_transpose_map;
    const std::vector<int>* active_transpose_map = nullptr;

    // Compute the cached_transpose_map if it is not already cached.
    if (!valid_arr_id || (cache_entry == this->cached_array_transposes.end())) {
        this->unify_cache_id_misses++;
        // Create new transpose map
        do_unify(new_transpose_map);

        if (valid_arr_id) {
            auto dict_id =
                DictionaryID{batch_dict->array_id, batch_dict->length};
            // Update the cached id and transpose map
            active_transpose_map = &this->_AddToCache(
                std::make_pair(dict_id, std::move(new_transpose_map)));
        } else {
            active_transpose_map = &new_transpose_map;
        }

        // We only update the ID if this dictionary went from being empty to
        // non-empty.
        if (empty_builder && this->dict_buff->size > 0) {
            // Update the dict id if there was any change to the dictionary.
            this->dict_buff->data_array->array_id =
                generate_array_id(this->dict_buff->data_array->length);
        }
    } else {
        if (cache_entry->first.length < batch_dict->length) {
            this->unify_cache_length_misses++;
            auto& new_transpose_map = cache_entry->second;
            do_unify(new_transpose_map);
            cache_entry->first.length = batch_dict->length;
        }
        active_transpose_map = &this->_MoveToFrontOfCache(cache_entry);
    }

    // create output batch array with common dictionary and new transposed
    // indices
    const std::shared_ptr<array_info>& in_indices_arr = in_arr->child_arrays[1];
    std::shared_ptr<array_info> out_indices_arr =
        alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
    dict_indices_t* in_inds = (dict_indices_t*)in_indices_arr->data1();
    dict_indices_t* out_inds = (dict_indices_t*)out_indices_arr->data1();
    for (size_t i = 0; i < in_indices_arr->length; i++) {
        if (!in_indices_arr->get_null_bit(i)) {
            out_indices_arr->set_null_bit(i, false);
            out_inds[i] = -1;
            continue;
        }
        dict_indices_t ind = (*active_transpose_map)[in_inds[i]];
        out_inds[i] = ind;
        if (ind == -1) {
            out_indices_arr->set_null_bit(i, false);
        } else {
            out_indices_arr->set_null_bit(i, true);
        }
    }
    return create_dict_string_array(this->dict_buff->data_array,
                                    out_indices_arr);
}

std::shared_ptr<bodo::vector<uint32_t>>
DictionaryBuilder::GetDictionaryHashes() {
    return this->dict_hashes;
}

const std::vector<int>& DictionaryBuilder::_AddToCache(
    std::pair<DictionaryID, std::vector<int>> cache_entry) {
    this->cached_array_transposes.pop_back();
    this->cached_array_transposes.emplace_front(std::move(cache_entry));
    return this->cached_array_transposes.front().second;
}

const std::vector<int>& DictionaryBuilder::_MoveToFrontOfCache(
    DictionaryCache::iterator cache_entry) {
    if (cache_entry != this->cached_array_transposes.begin()) {
        auto new_entry = std::move(*cache_entry);
        this->cached_array_transposes.erase(cache_entry);
        this->cached_array_transposes.emplace_front(std::move(new_entry));
    }
    return this->cached_array_transposes.front().second;
}

/* ------------------------------------------------------------------------
 */
