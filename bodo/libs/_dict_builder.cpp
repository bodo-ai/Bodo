#include "_dict_builder.h"
#include <fmt/format.h>
#include <cassert>
#include <cstddef>
#include <memory>

#include "_array_hash.h"
#include "_bodo_common.h"
#include "_query_profile_collector.h"
#include "_table_builder.h"

/* -------------------------- DictBuilderMetrics -------------------------- */

void DictBuilderMetrics::add_metrics(const DictBuilderMetrics& src_metrics) {
    this->unify_cache_id_misses += src_metrics.unify_cache_id_misses;
    this->unify_cache_length_misses += src_metrics.unify_cache_length_misses;
    this->transpose_filter_cache_id_misses +=
        src_metrics.transpose_filter_cache_id_misses;
    this->transpose_filter_cache_length_misses +=
        src_metrics.transpose_filter_cache_length_misses;
    this->unify_build_transpose_map_time +=
        src_metrics.unify_build_transpose_map_time;
    this->unify_transpose_time += src_metrics.unify_transpose_time;
    this->unify_string_arr_time += src_metrics.unify_string_arr_time;
    this->transpose_filter_build_transpose_map_time +=
        src_metrics.transpose_filter_build_transpose_map_time;
    this->transpose_filter_transpose_time +=
        src_metrics.transpose_filter_transpose_time;
    this->transpose_filter_string_arr_time +=
        src_metrics.transpose_filter_string_arr_time;
}

void DictBuilderMetrics::subtract_metrics(
    const DictBuilderMetrics& src_metrics) {
    assert(this->unify_cache_id_misses >= src_metrics.unify_cache_id_misses);
    assert(this->unify_cache_length_misses >=
           src_metrics.unify_cache_length_misses);
    assert(this->transpose_filter_cache_id_misses >=
           src_metrics.transpose_filter_cache_id_misses);
    assert(this->transpose_filter_cache_length_misses >=
           src_metrics.transpose_filter_cache_length_misses);
    assert(this->unify_build_transpose_map_time >=
           src_metrics.unify_build_transpose_map_time);
    assert(this->unify_transpose_time >= src_metrics.unify_transpose_time);
    assert(this->unify_string_arr_time >= src_metrics.unify_string_arr_time);
    assert(this->transpose_filter_build_transpose_map_time >=
           src_metrics.transpose_filter_build_transpose_map_time);
    assert(this->transpose_filter_transpose_time >=
           src_metrics.transpose_filter_transpose_time);
    assert(this->transpose_filter_string_arr_time >=
           src_metrics.transpose_filter_string_arr_time);

    this->unify_cache_id_misses -= src_metrics.unify_cache_id_misses;
    this->unify_cache_length_misses -= src_metrics.unify_cache_length_misses;
    this->transpose_filter_cache_id_misses -=
        src_metrics.transpose_filter_cache_id_misses;
    this->transpose_filter_cache_length_misses -=
        src_metrics.transpose_filter_cache_length_misses;
    this->unify_build_transpose_map_time -=
        src_metrics.unify_build_transpose_map_time;
    this->unify_transpose_time -= src_metrics.unify_transpose_time;
    this->unify_string_arr_time -= src_metrics.unify_string_arr_time;
    this->transpose_filter_build_transpose_map_time -=
        src_metrics.transpose_filter_build_transpose_map_time;
    this->transpose_filter_transpose_time -=
        src_metrics.transpose_filter_transpose_time;
    this->transpose_filter_string_arr_time -=
        src_metrics.transpose_filter_string_arr_time;
}

void DictBuilderMetrics::add_to_metrics(std::vector<MetricBase>& metrics,
                                        const std::string_view prefix) {
    metrics.emplace_back(
        StatMetric(fmt::format("{}unify_cache_id_misses", prefix),
                   this->unify_cache_id_misses));
    metrics.emplace_back(
        StatMetric(fmt::format("{}unify_cache_length_misses", prefix),
                   this->unify_cache_length_misses));
    metrics.emplace_back(
        StatMetric(fmt::format("{}transpose_filter_cache_id_misses", prefix),
                   this->transpose_filter_cache_id_misses));
    metrics.emplace_back(StatMetric(
        fmt::format("{}transpose_filter_cache_length_misses", prefix),
        this->transpose_filter_cache_length_misses));
    metrics.emplace_back(
        TimerMetric(fmt::format("{}unify_build_transpose_map_time", prefix),
                    this->unify_build_transpose_map_time));
    metrics.emplace_back(
        TimerMetric(fmt::format("{}unify_transpose_time", prefix),
                    this->unify_transpose_time));
    metrics.emplace_back(
        TimerMetric(fmt::format("{}unify_string_arr_time", prefix),
                    this->unify_string_arr_time));
    metrics.emplace_back(TimerMetric(
        fmt::format("{}transpose_filter_build_transpose_map_time", prefix),
        this->transpose_filter_build_transpose_map_time));
    metrics.emplace_back(
        TimerMetric(fmt::format("{}transpose_filter_transpose_time", prefix),
                    this->transpose_filter_transpose_time));
    metrics.emplace_back(
        TimerMetric(fmt::format("{}transpose_filter_string_arr_time", prefix),
                    this->transpose_filter_string_arr_time));
}

/* ------------------------------------------------------------------------ */

/* -------------------------- DictionaryBuilder --------------------------- */

inline dict_indices_t DictionaryBuilder::GetIndex(
    const std::shared_ptr<array_info>& in_arr, size_t idx) const noexcept {
    assert(in_arr->arr_type == bodo_array_type::STRING);
    char* in_data = in_arr->data1<bodo_array_type::STRING>();
    offset_t* in_offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();

    offset_t start_offset = in_offsets[idx];
    offset_t end_offset = in_offsets[idx + 1];
    int64_t len = end_offset - start_offset;
    std::string_view val(&in_data[start_offset], len);
    auto it = this->dict_str_to_ind->find(val);
    return (it != this->dict_str_to_ind->end()) ? it->second : -1;
}

std::shared_ptr<array_info> DictionaryBuilder::transpose_input_helper(
    const std::shared_ptr<array_info>& in_arr,
    const std::vector<int>& transpose_map) const {
    assert(in_arr->arr_type == bodo_array_type::DICT);
    const std::shared_ptr<array_info>& in_indices_arr = in_arr->child_arrays[1];
    std::shared_ptr<array_info> out_indices_arr =
        alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
    const dict_indices_t* in_inds =
        (dict_indices_t*)
            in_indices_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
    dict_indices_t* out_inds =
        (dict_indices_t*)
            out_indices_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
    for (size_t i = 0; i < in_indices_arr->length; i++) {
        if (!in_indices_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                i)) {
            out_indices_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                i, false);
            out_inds[i] = -1;
        } else {
            dict_indices_t ind = transpose_map[in_inds[i]];
            out_indices_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                i, (ind != -1));
            out_inds[i] = ind;
        }
    }
    return create_dict_string_array(this->dict_buff->data_array,
                                    out_indices_arr);
}

std::shared_ptr<array_info> DictionaryBuilder::TransposeExisting(
    const std::shared_ptr<array_info>& in_arr) {
    // Handle nested arrays (child_dict_builders)
    if (this->child_dict_builders.size() > 0) {
        assert(in_arr->arr_type == bodo_array_type::ARRAY_ITEM ||
               in_arr->arr_type == bodo_array_type::STRUCT ||
               in_arr->arr_type == bodo_array_type::MAP);
        assert(this->child_dict_builders.size() == in_arr->child_arrays.size());
        std::vector<std::shared_ptr<array_info>> transposed_children;
        for (size_t i = 0; i < this->child_dict_builders.size(); i++) {
            std::shared_ptr<DictionaryBuilder> child_builder =
                this->child_dict_builders[i];
            const std::shared_ptr<array_info>& child_arr =
                in_arr->child_arrays[i];
            if (child_builder == nullptr) {
                transposed_children.emplace_back(child_arr);
            } else {
                transposed_children.emplace_back(
                    child_builder->TransposeExisting(child_arr));
            }
        }

        // Recreate array with transposed children
        // NOTE: assuming that input array buffers can be reused in output
        // (aren't modified)
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, in_arr->length,
                in_arr->buffers, transposed_children);
        }
        if (in_arr->arr_type == bodo_array_type::STRUCT) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, in_arr->length,
                in_arr->buffers, transposed_children, 0, 0, 0, -1, false, false,
                false, 0, in_arr->field_names);
        }
        if (in_arr->arr_type == bodo_array_type::MAP) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::MAP, Bodo_CTypes::MAP, in_arr->length,
                std::vector<std::shared_ptr<BodoBuffer>>({}),
                transposed_children);
        }
        return out_arr;
    }

    if (in_arr->arr_type != bodo_array_type::DICT &&
        in_arr->arr_type != bodo_array_type::STRING) {
        throw std::runtime_error(fmt::format(
            "DictionaryBuilder::TransposeExisting: DICT or STRING array "
            "expected, but got {} instead",
            GetArrType_as_string(in_arr->arr_type)));
    }

    assert(this->dict_buff != nullptr);
    if (in_arr->arr_type == bodo_array_type::STRING) {
        time_pt start = start_timer();
        // TODO Move this out into it's own function
        std::shared_ptr<array_info> out_indices_arr =
            alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
        dict_indices_t* out_inds =
            (dict_indices_t*)
                out_indices_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        void* out_null_bitmask =
            out_indices_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();

        for (size_t i = 0; i < in_arr->length; i++) {
            // Handle nulls in the dictionary
            dict_indices_t out_ind =
                in_arr->get_null_bit<bodo_array_type::STRING>(i)
                    ? this->GetIndex(in_arr, i)
                    : -1;
            out_inds[i] = out_ind;
            // Handles both the cases: either the entry was already null or
            // is now being set to null since the corresponding string doesn't
            // exist in the DictionaryBuilder.
            arrow::bit_util::SetBitTo(static_cast<uint8_t*>(out_null_bitmask),
                                      i, (out_ind != -1));
        }

        // Unlike UnifyDictionaryArray, we don't need to update the
        // dictionary/array id of the dictionary-builder's dictionary since we
        // haven't modified it (by design).

        std::unique_ptr<array_info> out = create_dict_string_array(
            this->dict_buff->data_array, std::move(out_indices_arr));
        this->metrics.transpose_filter_string_arr_time += end_timer(start);
        return out;
    }

    assert(in_arr->arr_type == bodo_array_type::DICT);
    time_pt start = start_timer();
    std::shared_ptr<array_info> batch_dict = in_arr->child_arrays[0];
    bool valid_arr_id = batch_dict->array_id >= 0;
    bool empty_arr = batch_dict->array_id == 0;
    // An empty array doesn't require any transposing.
    if (empty_arr) {
        return in_arr;
    }

    auto cache_entry = std::find_if(
        this->cached_filter_array_transposes.begin(),
        this->cached_filter_array_transposes.end(),
        [batch_dict](const std::pair<DictionaryID, std::vector<int>> entry) {
            return entry.first.arr_id == batch_dict->array_id;
        });

    auto update_transpose_map = [&](std::vector<int>& new_transpose_map) {
        // Check/update dictionary hash table and create transpose map
        // if new_transpose_map is already populated, only consider elements
        // that come after it ends
        size_t existing_size = new_transpose_map.size();
        for (size_t i = existing_size; i < batch_dict->length; i++) {
            // Handle nulls in the dictionary
            int idx = batch_dict->get_null_bit<bodo_array_type::STRING>(i)
                          ? this->GetIndex(batch_dict, i)
                          : -1;
            new_transpose_map.emplace_back(idx);
        }
    };

    std::vector<int> new_transpose_map;
    new_transpose_map.reserve(batch_dict->length);
    const std::vector<int>* active_transpose_map = nullptr;

    // Compute the cached_transpose_map if it is not already cached.
    if (!valid_arr_id ||
        (cache_entry == this->cached_filter_array_transposes.end())) {
        this->metrics.transpose_filter_cache_id_misses++;
        // Create transpose map
        update_transpose_map(new_transpose_map);

        if (valid_arr_id) {
            auto dict_id =
                DictionaryID{batch_dict->array_id, batch_dict->length};
            // Update the cached id and transpose map
            active_transpose_map = &this->_AddToFilterCache(
                std::make_pair(dict_id, std::move(new_transpose_map)));
        } else {
            active_transpose_map = &new_transpose_map;
        }
    } else {
        if (cache_entry->first.length < batch_dict->length) {
            this->metrics.transpose_filter_cache_length_misses++;
            auto& new_transpose_map = cache_entry->second;
            update_transpose_map(new_transpose_map);
            cache_entry->first.length = batch_dict->length;
        }
        active_transpose_map = &this->_MoveToFrontOfFilterCache(cache_entry);
    }
    this->metrics.transpose_filter_build_transpose_map_time += end_timer(start);

    // Create output batch array with common dictionary and new transposed
    // indices
    time_pt start_transpose = start_timer();
    std::shared_ptr<array_info> out =
        this->transpose_input_helper(in_arr, *active_transpose_map);
    this->metrics.transpose_filter_transpose_time += end_timer(start_transpose);
    return out;
}

inline dict_indices_t DictionaryBuilder::InsertIfNotExists(
    const std::shared_ptr<array_info>& in_arr, size_t idx) {
    assert(in_arr->arr_type == bodo_array_type::STRING);
    char* in_data = in_arr->data1<bodo_array_type::STRING>();
    offset_t* in_offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();

    offset_t start_offset = in_offsets[idx];
    offset_t end_offset = in_offsets[idx + 1];
    int64_t len = end_offset - start_offset;
    std::string_view val(&in_data[start_offset], len);
    return this->InsertIfNotExists(val);
}

inline dict_indices_t DictionaryBuilder::InsertIfNotExists(
    const std::string_view& val) {
    dict_indices_t ind;
    // get existing index if already in hash table
    if (auto it = this->dict_str_to_ind->find(val);
        it != this->dict_str_to_ind->end()) {
        ind = it->second;
    } else {
        this->dict_buff->ReserveSize(1);
        this->dict_buff->ReserveSpaceForStringAppend(val.size());

        // Resize buffers
        CHECK_ARROW_MEM(
            this->dict_buff->data_array->buffers[1]->SetSize(
                (this->dict_buff->size + 2) * sizeof(offset_t)),
            "DictionaryBuilder::InsertIfNotExists: SetSize failed!");
        CHECK_ARROW_MEM(
            this->dict_buff->data_array->buffers[2]->SetSize(
                arrow::bit_util::BytesForBits(this->dict_buff->size + 1)),
            "DictionaryBuilder::InsertIfNotExists: SetSize failed!");

        CHECK_ARROW_MEM(
            this->dict_buff->data_array->buffers[0]->SetSize(
                this->dict_buff->data_array->n_sub_elems() + val.size()),
            "DictionaryBuilder::InsertIfNotExists: SetSize failed!");
        // insert into hash table if not exists
        ind = this->dict_str_to_ind->size();
        // TODO: remove std::string() after upgrade to C++23
        (*this->dict_str_to_ind)[std::string(val)] = ind;

        // copy string into buffer
        memcpy(this->dict_buff->data_array->data1<bodo_array_type::STRING>() +
                   this->dict_buff->data_array
                       ->data2<bodo_array_type::STRING, offset_t>()[ind],
               val.data(), val.length());

        // Set the offsets
        this->dict_buff->data_array
            ->data2<bodo_array_type::STRING, offset_t>()[ind + 1] =
            this->dict_buff->data_array
                ->data2<bodo_array_type::STRING, offset_t>()[ind] +
            val.size();
        // Set the null bit
        this->dict_buff->data_array->set_null_bit<bodo_array_type::STRING>(
            ind, true);
        this->dict_buff->data_array->length++;

        if (this->is_key) {
            uint32_t hash;
            hash_string_32(val.data(), (const int)val.size(),
                           SEED_HASH_PARTITION, &hash);
            this->dict_hashes->emplace_back(hash);
        }
    }
    return ind;
}

DictionaryBuilder::DictionaryBuilder(
    std::shared_ptr<array_info> dict, bool is_key_,
    std::vector<std::shared_ptr<DictionaryBuilder>> child_dict_builders_,
    size_t transpose_cache_size, size_t filter_transpose_cache_size)
    : is_key(is_key_),
      child_dict_builders(child_dict_builders_),
      cached_array_transposes(
          transpose_cache_size,
          std::pair(DictionaryID{-1, 0}, std::vector<dict_indices_t>())),
      cached_filter_array_transposes(
          filter_transpose_cache_size,
          std::pair(DictionaryID{-1, 0}, std::vector<dict_indices_t>())) {
    // Nested arrays don't need other internal state
    if (child_dict_builders_.size() > 0) {
        assert(dict == nullptr);
        return;
    }
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
    // Unify child arrays for nested arrays
    if (this->child_dict_builders.size() > 0) {
        assert(in_arr->arr_type == bodo_array_type::ARRAY_ITEM ||
               in_arr->arr_type == bodo_array_type::STRUCT ||
               in_arr->arr_type == bodo_array_type::MAP);
        assert(this->child_dict_builders.size() == in_arr->child_arrays.size());

        std::vector<std::shared_ptr<array_info>> unified_children;
        for (size_t i = 0; i < this->child_dict_builders.size(); i++) {
            std::shared_ptr<DictionaryBuilder> child_builder =
                this->child_dict_builders[i];
            const std::shared_ptr<array_info>& child_arr =
                in_arr->child_arrays[i];
            if (child_builder == nullptr) {
                unified_children.emplace_back(child_arr);
            } else {
                unified_children.emplace_back(
                    child_builder->UnifyDictionaryArray(child_arr));
            }
        }

        // Recreate array with unified children
        // NOTE: assuming that input array buffers can be reused in output
        // (aren't modified)
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, in_arr->length,
                in_arr->buffers, unified_children);
        }
        if (in_arr->arr_type == bodo_array_type::STRUCT) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, in_arr->length,
                in_arr->buffers, unified_children, 0, 0, 0, -1, false, false,
                false, 0, in_arr->field_names);
        }
        if (in_arr->arr_type == bodo_array_type::MAP) {
            out_arr = std::make_shared<array_info>(
                bodo_array_type::MAP, Bodo_CTypes::MAP, in_arr->length,
                std::vector<std::shared_ptr<BodoBuffer>>({}), unified_children);
        }
        return out_arr;
    }

    if (in_arr->arr_type != bodo_array_type::DICT &&
        in_arr->arr_type != bodo_array_type::STRING) {
        throw std::runtime_error(
            "UnifyDictionaryArray: DICT or STRING array expected");
    }

    bool empty_builder = this->dict_buff->data_array->array_id <= 0;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        time_pt start = start_timer();
        // TODO(aneesh) move this out into it's own function
        std::shared_ptr<array_info> out_indices_arr =
            alloc_nullable_array(in_arr->length, Bodo_CTypes::INT32, 0);
        dict_indices_t* out_inds =
            (dict_indices_t*)
                out_indices_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();

        // copy null bitmask from input to output
        char* src_null_bitmask =
            in_arr->null_bitmask<bodo_array_type::STRING>();
        char* dst_null_bitmask =
            out_indices_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
        size_t bytes_in_bitmask = arrow::bit_util::BytesForBits(in_arr->length);
        memcpy(dst_null_bitmask, src_null_bitmask, bytes_in_bitmask);

        for (size_t i = 0; i < in_arr->length; i++) {
            // handle nulls in the dictionary
            if (!in_arr->get_null_bit<bodo_array_type::STRING>(i)) {
                out_inds[i] = -1;
            } else {
                out_inds[i] = this->InsertIfNotExists(in_arr, i);
            }
        }

        // We only update the ID if this dictionary went from being empty to
        // non-empty.
        if (empty_builder && this->dict_buff->size > 0) {
            this->dict_buff->data_array->array_id =
                generate_array_id(this->dict_buff->data_array->length);
        }

        std::unique_ptr<array_info> out = create_dict_string_array(
            this->dict_buff->data_array, std::move(out_indices_arr));
        this->metrics.unify_string_arr_time += end_timer(start);
        return out;
    }

    time_pt start = start_timer();
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
        // Check/update dictionary hash table and create transpose map
        // if new_transpose_map is already populated, only consider elements
        // that come after it ends
        for (size_t i = new_transpose_map.size(); i < batch_dict->length; i++) {
            // Handle nulls in the dictionary
            int idx = batch_dict->get_null_bit<bodo_array_type::STRING>(i)
                          ? this->InsertIfNotExists(batch_dict, i)
                          : -1;
            new_transpose_map.emplace_back(idx);
        }
    };
    std::vector<int> new_transpose_map;
    const std::vector<int>* active_transpose_map = nullptr;

    // Compute the cached_transpose_map if it is not already cached.
    if (!valid_arr_id || (cache_entry == this->cached_array_transposes.end())) {
        this->metrics.unify_cache_id_misses++;
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
            this->metrics.unify_cache_length_misses++;
            auto& new_transpose_map = cache_entry->second;
            do_unify(new_transpose_map);
            cache_entry->first.length = batch_dict->length;
        }
        active_transpose_map = &this->_MoveToFrontOfCache(cache_entry);
    }
    this->metrics.unify_build_transpose_map_time += end_timer(start);

    // Create output batch array with common dictionary and new transposed
    // indices
    time_pt start_transpose = start_timer();
    std::shared_ptr<array_info> out =
        this->transpose_input_helper(in_arr, *active_transpose_map);
    this->metrics.unify_transpose_time += end_timer(start_transpose);
    return out;
}

std::shared_ptr<bodo::vector<uint32_t>>
DictionaryBuilder::GetDictionaryHashes() {
    return this->dict_hashes;
}

DictBuilderMetrics DictionaryBuilder::GetMetrics() const {
    if (this->child_dict_builders.size() > 0) {
        // Combine metrics from all children dict-builders recursively.
        DictBuilderMetrics combined_metrics;
        for (size_t i = 0; i < this->child_dict_builders.size(); i++) {
            std::shared_ptr<DictionaryBuilder> child_builder =
                this->child_dict_builders[i];
            if (child_builder == nullptr) {
                continue;
            }
            combined_metrics.add_metrics(child_builder->GetMetrics());
        }
        return combined_metrics;
    }
    assert(this->dict_buff != nullptr);
    return this->metrics;
}

const std::vector<int>& DictionaryBuilder::_AddToCache(
    std::pair<DictionaryID, std::vector<int>> cache_entry) {
    this->cached_array_transposes.pop_back();
    this->cached_array_transposes.emplace_front(std::move(cache_entry));
    return this->cached_array_transposes.front().second;
}

const std::vector<int>& DictionaryBuilder::_AddToFilterCache(
    std::pair<DictionaryID, std::vector<int>> cache_entry) {
    this->cached_filter_array_transposes.pop_back();
    this->cached_filter_array_transposes.emplace_front(std::move(cache_entry));
    return this->cached_filter_array_transposes.front().second;
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

const std::vector<int>& DictionaryBuilder::_MoveToFrontOfFilterCache(
    DictionaryCache::iterator cache_entry) {
    if (cache_entry != this->cached_filter_array_transposes.begin()) {
        auto new_entry = std::move(*cache_entry);
        this->cached_filter_array_transposes.erase(cache_entry);
        this->cached_filter_array_transposes.emplace_front(
            std::move(new_entry));
    }
    return this->cached_filter_array_transposes.front().second;
}

/* ------------------------------------------------------------------------ */

std::shared_ptr<DictionaryBuilder> create_dict_builder_for_array(
    const std::shared_ptr<bodo::DataType>& t, bool is_key) {
    if (t->is_array()) {
        const bodo::ArrayType* t_as_array =
            static_cast<bodo::ArrayType*>(t.get());
        std::vector<std::shared_ptr<DictionaryBuilder>> child_dict_builders = {
            create_dict_builder_for_array(t_as_array->value_type->copy(),
                                          is_key)};
        return std::make_shared<DictionaryBuilder>(
            nullptr, is_key, std::move(child_dict_builders));
    }
    if (t->is_struct()) {
        const bodo::StructType* t_as_struct =
            static_cast<bodo::StructType*>(t.get());
        std::vector<std::shared_ptr<DictionaryBuilder>> child_dict_builders;
        for (const std::unique_ptr<bodo::DataType>& t :
             t_as_struct->child_types) {
            child_dict_builders.push_back(
                create_dict_builder_for_array(t->copy(), is_key));
        }
        return std::make_shared<DictionaryBuilder>(
            nullptr, is_key, std::move(child_dict_builders));
    }
    if (t->is_map()) {
        const bodo::MapType* t_as_map = static_cast<bodo::MapType*>(t.get());

        // Create array(struct(key, value)) type for proper nested dict builder
        // structure
        std::vector<std::unique_ptr<bodo::DataType>> struct_child_types;
        struct_child_types.push_back(t_as_map->key_type->copy());
        struct_child_types.push_back(t_as_map->value_type->copy());
        std::unique_ptr<bodo::StructType> t_struct =
            std::make_unique<bodo::StructType>(std::move(struct_child_types));
        std::shared_ptr<bodo::ArrayType> t_array =
            std::make_shared<bodo::ArrayType>(std::move(t_struct));
        std::vector<std::shared_ptr<DictionaryBuilder>> child_dict_builders = {
            create_dict_builder_for_array(t_array, is_key)};
        return std::make_shared<DictionaryBuilder>(
            nullptr, is_key, std::move(child_dict_builders));
    }
    if (t->array_type == bodo_array_type::DICT) {
        std::shared_ptr<array_info> dict = alloc_array_top_level(
            0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        return std::make_shared<DictionaryBuilder>(dict, is_key);
    }
    return nullptr;
}

std::shared_ptr<DictionaryBuilder> create_dict_builder_for_array(
    const std::shared_ptr<array_info>& arr, bool is_key) {
    return create_dict_builder_for_array(arr->data_type()->copy(), is_key);
}

void set_array_dict_from_builder(
    std::shared_ptr<array_info>& arr,
    const std::shared_ptr<DictionaryBuilder>& builder) {
    // Handle nested arrays
    if (arr->arr_type == bodo_array_type::ARRAY_ITEM ||
        arr->arr_type == bodo_array_type::STRUCT ||
        arr->arr_type == bodo_array_type::MAP) {
        for (size_t i = 0; i < arr->child_arrays.size(); i++) {
            std::shared_ptr<array_info>& child_arr = arr->child_arrays[i];
            const std::shared_ptr<DictionaryBuilder>& child_builder =
                builder->child_dict_builders[i];
            set_array_dict_from_builder(child_arr, child_builder);
        }
    }
    if (arr->arr_type == bodo_array_type::DICT) {
        arr->child_arrays[0] = builder->dict_buff->data_array;
    }
}

void set_array_dict_from_array(std::shared_ptr<array_info>& out_arr,
                               const std::shared_ptr<array_info>& in_arr) {
    // Handle nested arrays
    if (out_arr->arr_type == bodo_array_type::ARRAY_ITEM ||
        out_arr->arr_type == bodo_array_type::STRUCT ||
        out_arr->arr_type == bodo_array_type::MAP) {
        assert(out_arr->arr_type == in_arr->arr_type);
        for (size_t i = 0; i < out_arr->child_arrays.size(); i++) {
            set_array_dict_from_array(out_arr->child_arrays[i],
                                      in_arr->child_arrays[i]);
        }
    }
    if (out_arr->arr_type == bodo_array_type::DICT) {
        out_arr->child_arrays[0] = in_arr->child_arrays[0];
    }
}
