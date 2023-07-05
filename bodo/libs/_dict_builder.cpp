#include "_dict_builder.h"
#include "_array_hash.h"
#include "_bodo_common.h"
#include "_table_builder.h"

/* -------------------------- DictionaryBuilder --------------------------- */

std::shared_ptr<array_info> DictionaryBuilder::UnifyDictionaryArray(
    const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != bodo_array_type::DICT) {
        throw std::runtime_error("UnifyDictionaryArray: DICT array expected");
    }

    std::shared_ptr<array_info> batch_dict = in_arr->child_arrays[0];
    bool valid_arr_id = in_arr->dict_id >= 0;
    bool dicts_match =
        valid_arr_id && (this->dict_id == 0 || in_arr->dict_id == 0 ||
                         this->dict_id == in_arr->dict_id);
    if (dicts_match) {
        if (this->dict_id == 0 && in_arr->dict_id != 0) {
            // The dictionary is empty and we are adding entries.
            // We need to append the whole new dictionary.
            this->dict_buff->ReserveArray(batch_dict);
            // Append all entries
            this->dict_buff
                ->AppendBatch<bodo_array_type::STRING, Bodo_CTypes::STRING>(
                    batch_dict);
            // Update the id
            this->dict_id = in_arr->dict_id;
        }
        // If the dictionaries already match we can just return without
        // transposing.
        return in_arr;
    }

    // Compute the cached_transpose_map if this isn't the same as the
    // previous batch
    if (!valid_arr_id || (this->cached_dict_id != in_arr->dict_id)) {
        this->dict_buff->ReserveArray(batch_dict);
        // Clear the cache.
        this->cached_transpose_map.clear();
        this->cached_transpose_map.reserve(batch_dict->length);

        // Check/update dictionary hash table and create transpose map
        char* data = batch_dict->data1();
        offset_t* offsets = (offset_t*)batch_dict->data2();
        bool found_change = false;
        for (size_t i = 0; i < batch_dict->length; i++) {
            // handle nulls in the dictionary
            if (!batch_dict->get_null_bit(i)) {
                this->cached_transpose_map.emplace_back(-1);
                continue;
            }
            offset_t start_offset = offsets[i];
            offset_t end_offset = offsets[i + 1];
            int64_t len = end_offset - start_offset;
            std::string_view val(&data[start_offset], len);
            // get existing index if already in hash table
            if (this->dict_str_to_ind->contains(val)) {
                dict_indices_t ind = this->dict_str_to_ind->find(val)->second;
                this->cached_transpose_map.emplace_back(ind);
            } else {
                found_change = true;
                // insert into hash table if not exists
                dict_indices_t ind = this->dict_str_to_ind->size();
                // TODO: remove std::string() after upgrade to C++23
                (*(this->dict_str_to_ind))[std::string(val)] = ind;
                this->cached_transpose_map.emplace_back(ind);
                this->dict_buff->AppendRow(batch_dict, i);
                if (this->is_key) {
                    uint32_t hash;
                    hash_string_32(&data[start_offset], (const int)len,
                                   SEED_HASH_PARTITION, &hash);
                    this->dict_hashes->emplace_back(hash);
                }
            }
        }
        // Update the cached id
        this->cached_dict_id = in_arr->dict_id;
        if (found_change) {
            // Update the dict id if there was any change to the dictionary.
            this->dict_id =
                generate_dict_id(this->dict_buff->data_array->length);
        }
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
        dict_indices_t ind = this->cached_transpose_map[in_inds[i]];
        out_inds[i] = ind;
        if (ind == -1) {
            out_indices_arr->set_null_bit(i, false);
        } else {
            out_indices_arr->set_null_bit(i, true);
        }
    }
    return create_dict_string_array(
        this->dict_buff->data_array, out_indices_arr, false,
        /*_has_deduped_local_dictionary=*/true, false, this->dict_id);
}

std::shared_ptr<bodo::vector<uint32_t>>
DictionaryBuilder::GetDictionaryHashes() {
    return this->dict_hashes;
}

/* ------------------------------------------------------------------------ */
