#include "_chunked_table_builder.h"

/* --------------------------- Helper Functions --------------------------- */

/**
 * @brief Get required buffer sizes for nullable array.
 *
 * @param dtype Dtype of the array.
 * @param size Number of elements.
 * @return std::tuple<int64_t, int64_t> Tuple of data buffer size and null
 * bitmask size.
 */
inline std::tuple<int64_t, int64_t> get_nullable_arr_alloc_sizes(
    Bodo_CTypes::CTypeEnum dtype, int64_t size) {
    if (dtype == Bodo_CTypes::_BOOL) {
        int64_t req_size = ::arrow::bit_util::BytesForBits(size);
        return std::tuple(req_size, req_size);
    } else {
        uint64_t size_type = numpy_item_size[dtype];
        int64_t data_buffer_req_size = static_cast<int64_t>(size * size_type);
        int64_t null_bitmap_buffer_req_size =
            ::arrow::bit_util::BytesForBits(size);
        return std::tuple(data_buffer_req_size, null_bitmap_buffer_req_size);
    }
}

/* ------------------------------------------------------------------------ */

/* ----------------------- ChunkedTableArrayBuilder ----------------------- */

ChunkedTableArrayBuilder::ChunkedTableArrayBuilder(
    std::shared_ptr<array_info> _data_array,
    std::shared_ptr<DictionaryBuilder> _dict_builder, size_t chunk_size,
    size_t _max_resize_count)
    : data_array(std::move(_data_array)),
      dict_builder(_dict_builder),
      capacity(chunk_size),
      max_resize_count(_max_resize_count) {
    if (this->data_array->length != 0) {
        throw std::runtime_error(
            "ChunkedTableArrayBuilder: Length of input array is not 0!");
    }

    // Get minimum frame size in BufferPool and set that as the
    // minimum size of any of the buffers.
    const int64_t min_buffer_allocation_size =
        bodo::BufferPool::Default()->GetSmallestSizeClassSize();

    // Reserve space in buffers based on type and capacity.
    // NOTE: We call Resize instead of Reserve so that we don't need
    // to call Resize during every AppendRow call separately.
    // Since the buffers are only used during the buffer build
    // phase, this should be safe, however, we should be careful
    // if using it elsewhere.
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            auto [data_buffer_alloc_size, null_bitmap_buffer_alloc_size] =
                get_nullable_arr_alloc_sizes(this->data_array->dtype,
                                             this->capacity);
            data_buffer_alloc_size =
                std::max(data_buffer_alloc_size, min_buffer_allocation_size);
            null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_alloc_size, min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[1]->Resize(
                                null_bitmap_buffer_alloc_size, false),
                            "Resize Failed!");

        } break;
        case bodo_array_type::STRING: {
            // For strings, allocate CHUNKED_TABLE_DEFAULT_STRING_PREALLOCATION
            // bytes per string for now.
            int64_t data_buffer_alloc_size =
                std::max(static_cast<int64_t>(
                             this->capacity *
                             CHUNKED_TABLE_DEFAULT_STRING_PREALLOCATION),
                         min_buffer_allocation_size);
            int64_t offset_buffer_alloc_size = std::max(
                static_cast<int64_t>((this->capacity + 1) * sizeof(offset_t)),
                min_buffer_allocation_size);
            int64_t null_bitmap_buffer_alloc_size =
                std::max(::arrow::bit_util::BytesForBits(this->capacity),
                         min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[1]->Resize(
                                offset_buffer_alloc_size, false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[2]->Resize(
                                null_bitmap_buffer_alloc_size, false),
                            "Resize Failed!");
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            int64_t data_buffer_alloc_size =
                std::max(static_cast<int64_t>(this->capacity * size_type),
                         min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, false),
                            "Resize Failed!");
        } break;
        case bodo_array_type::DICT: {
            if (_dict_builder == nullptr) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder: dict_builder is nullptr for a "
                    "dict-encoded string array!");
            }
            if (_dict_builder->dict_buff->data_array.get() !=
                this->data_array->child_arrays[0].get()) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder: specified dict_builder does not "
                    "match dictionary of _data_array!");
            }
            // Recursively call the constructor on the indices array.
            this->dict_indices = std::make_shared<ChunkedTableArrayBuilder>(
                this->data_array->child_arrays[1], nullptr, chunk_size,
                _max_resize_count);
        } break;
        default:
            throw std::runtime_error(
                "Invalid array type in ChunkedTableArrayBuilder: " +
                GetArrType_as_string(this->data_array->arr_type));
    }
}

size_t ChunkedTableArrayBuilder::GetTotalBytes() {
    // XXX This seems general, so could be moved into
    // array_info itself.
    size_t total = 0;
    std::shared_ptr<array_info> _data_array = this->data_array;
    if (this->data_array->arr_type == bodo_array_type::DICT) {
        // In case of dict encoded string array, we will get the
        // size from the indices array instead.
        _data_array = this->data_array->child_arrays[1];
    }
    for (const auto& buffer : _data_array->buffers) {
        total += buffer->getMeminfo()->size;
    }
    return total;
}

bool ChunkedTableArrayBuilder::CanResize() {
    if (this->data_array->arr_type == bodo_array_type::STRING) {
        return (this->resize_count < this->max_resize_count);
    }
    return false;
}

bool ChunkedTableArrayBuilder::CanAppendRowWithoutResizing(
    const std::shared_ptr<array_info>& in_arr, const int64_t row_ind) {
    // If size >= capacity, we cannot append regardless of the
    // array type and data type.
    if (this->size >= this->capacity) {
        return false;
    }

    // For non-string types, if size < capacity (checked above)
    // we can always append.
    if (this->data_array->arr_type != bodo_array_type::STRING) {
        return true;
    } else {
        // If it's just appending a null value, that should be fine
        // since it shouldn't require any additional space in the
        // data buffer itself.
        if (row_ind < 0) {
            return true;
        }
        // Compare current length with buffer size.
        const int64_t current_chars_len = this->data_array->n_sub_elems();
        const offset_t* offsets = (offset_t*)in_arr->data2();
        const int64_t new_str_len = offsets[row_ind + 1] - offsets[row_ind];
        const int64_t req_len = current_chars_len + new_str_len;
        return req_len <= this->data_array->buffers[0]->size();
    }
}

bool ChunkedTableArrayBuilder::CanAppendRow(
    const std::shared_ptr<array_info>& in_arr, const int64_t row_ind) {
    // If size >= capacity, we cannot append regardless of the
    // array type and data type.
    if (this->size >= this->capacity) {
        return false;
    }
    // For non-string types, if size < capacity (checked above)
    // we can always append.
    if (this->data_array->arr_type != bodo_array_type::STRING) {
        return true;
    } else {
        // If it's just appending a null value, that should be fine
        // since it shouldn't require any additional space in the
        // data buffer itself.
        if (row_ind < 0) {
            return true;
        }
        // Compare current length with buffer size to see if it
        // can be appended without resizing.
        const int64_t current_chars_len = this->data_array->n_sub_elems();
        const offset_t* offsets = (offset_t*)in_arr->data2();
        const int64_t new_str_len = offsets[row_ind + 1] - offsets[row_ind];
        const int64_t req_len = current_chars_len + new_str_len;
        if (req_len <= this->data_array->buffers[0]->size()) {
            return true;
        }
        // Check if resizing would enable us to do it:
        int64_t max_possible_buffer_size =
            this->data_array->buffers[0]->size() *
            std::pow(2, this->max_resize_count - this->resize_count);
        if (req_len <= max_possible_buffer_size) {
            return true;
        }
        // Check for the case where this string is very large
        // on its own. We have to support this case in some form
        // for reliability.
        // We will allow resizing this buffer endlessly if we
        // have resize attempts remaining, else we will let the
        // next chunk handle it.
        if ((new_str_len > max_possible_buffer_size) &&
            (this->resize_count < this->max_resize_count)) {
            return true;
        }
        return false;
    }
}

void ChunkedTableArrayBuilder::UnsafeAppendRow(
    const std::shared_ptr<array_info>& in_arr, const int64_t row_ind) {
    // TODO These functions are called in loops, so we eventually
    // need to template it based on array and c-type so that we don't
    // need to perform these checks every time.
    // (https://bodo.atlassian.net/browse/BSE-563)

    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            // In some cases (e.g. Outer Join, see
            // HashJoinState::InitOutputBuffer), we might have created the
            // buffer with a NULLABLE type while the input might still be NUMPY,
            // so we will handle that case here:
            if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                bool null_bit = (row_ind >= 0);
                if (null_bit) {
                    if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                        arrow::bit_util::SetBitTo(
                            (uint8_t*)this->data_array->data1(), this->size,
                            GetBit((uint8_t*)in_arr->data1(), row_ind));
                    } else {
                        uint64_t size_type =
                            numpy_item_size[this->data_array->dtype];
                        char* out_ptr =
                            this->data_array->data1() + size_type * size;
                        char* in_ptr = in_arr->data1() + size_type * row_ind;
                        memcpy(out_ptr, in_ptr, size_type);
                    }
                    null_bit =
                        GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                }
                SetBitTo((uint8_t*)this->data_array->null_bitmask(), size,
                         null_bit);
            } else if (in_arr->arr_type == bodo_array_type::NUMPY) {
                bool null_bit = (row_ind >= 0);
                if (null_bit) {
                    if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                        // Boolean needs a special implementation since NUMPY
                        // arrays use 1 byte per value whereas NULLABLE_INT_BOOL
                        // arrays use 1 bit per value.
                        bool data_bit = ((char*)in_arr->data1())[row_ind] != 0;
                        arrow::bit_util::SetBitTo(
                            (uint8_t*)this->data_array->data1(), this->size,
                            data_bit);
                    } else {
                        uint64_t size_type =
                            numpy_item_size[this->data_array->dtype];
                        char* out_ptr =
                            this->data_array->data1() + size_type * size;
                        char* in_ptr = in_arr->data1() + size_type * row_ind;
                        memcpy(out_ptr, in_ptr, size_type);
                    }
                }
                // Mark as not-null if row_ind >= 0, and null otherwise.
                // Replicates the behavior of
                // RetrieveArray_SingleColumn_F_numpy's nullable_arr case.
                SetBitTo((uint8_t*)this->data_array->null_bitmask(), size,
                         null_bit);
            } else {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::UnsafeAppendRow: Tried to "
                    "append row from array of type " +
                    GetArrType_as_string(in_arr->arr_type) +
                    " to a NULLABLE_INT_BOOL array buffer.");
            }
            this->size++;
            data_array->length = size;
        } break;
        case bodo_array_type::STRING: {
            offset_t* curr_offsets = (offset_t*)this->data_array->data2();
            bool null_bit = (row_ind >= 0);
            if (null_bit) {
                offset_t* in_offsets = (offset_t*)in_arr->data2();

                // append offset
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                curr_offsets[size + 1] = curr_offsets[size] + str_len;

                // copy characters
                char* out_ptr = this->data_array->data1() + curr_offsets[size];
                char* in_ptr = in_arr->data1() + in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);

                // update null bit based on in_arr
                null_bit = GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
            } else {
                // Still need to set these because otherwise it can lead
                // to segfaults.
                curr_offsets[size + 1] = curr_offsets[size];
            }
            SetBitTo((uint8_t*)this->data_array->null_bitmask(), size,
                     null_bit);

            // update size state
            this->size++;
            this->data_array->length = size;
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            char* out_ptr = this->data_array->data1() + size_type * size;
            if (row_ind >= 0) {
                char* in_ptr = in_arr->data1() + size_type * row_ind;
                memcpy(out_ptr, in_ptr, size_type);
            } else {
                std::vector<char> vectNaN =
                    RetrieveNaNentry(this->data_array->dtype);
                memcpy(out_ptr, vectNaN.data(), size_type);
            }
            this->size++;
            this->data_array->length = size;
        } break;
        case bodo_array_type::DICT: {
            if (this->data_array->child_arrays[0] != in_arr->child_arrays[0]) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::UnsafeAppendRow: Dictionaries "
                    "not unified!");
            }
            // Call recursively on the indices array instead.
            this->dict_indices->UnsafeAppendRow(in_arr->child_arrays[1],
                                                row_ind);
            this->size++;
            this->data_array->length = size;
        } break;
        default: {
            throw std::runtime_error(
                "Invalid array type in "
                "ChunkedTableArrayBuilder::UnsafeAppendRow: " +
                GetArrType_as_string(this->data_array->arr_type));
        }
    }
}

void ChunkedTableArrayBuilder::AppendRow(
    const std::shared_ptr<array_info>& in_arr, const int64_t row_ind) {
    // Fast path for most appends:
    if (this->CanAppendRowWithoutResizing(in_arr, row_ind)) {
        this->UnsafeAppendRow(in_arr, row_ind);
        return;
    }

    // Check if it can be appended after resizing:
    if (!this->CanAppendRow(in_arr, row_ind)) {
        throw std::runtime_error(
            "ChunkedTableArrayBuilder::AppendRow: Cannot append row!");
    }

    // If the row can be appended and it hasn't been already, this must be the
    // STRING case.
    if (this->data_array->arr_type != bodo_array_type::STRING) {
        throw std::runtime_error(
            "ChunkedTableArrayBuilder::AppendRow: Expected STRING column but "
            "encountered " +
            GetArrType_as_string(this->data_array->arr_type) + " instead.");
    }

    // Resize as many times as required:
    const int64_t current_chars_len = this->data_array->n_sub_elems();
    const offset_t* offsets = (offset_t*)in_arr->data2();
    const int64_t new_str_len = offsets[row_ind + 1] - offsets[row_ind];
    const int64_t req_len = current_chars_len + new_str_len;

    // TODO Make this more efficient by getting the string
    // resize information from CanAppend or similar helper so we don't have
    // to recompute that (https://bodo.atlassian.net/browse/BSE-556).
    int64_t buffer_size = this->data_array->buffers[0]->size();
    if (req_len > buffer_size) {
        while ((req_len > buffer_size) &&
               (this->resize_count < this->max_resize_count)) {
            buffer_size *= 2;
            this->resize_count += 1;
        }
        // In case it's still not sufficient (single very large string
        // case), just resize to required length.
        buffer_size = std::max(buffer_size, req_len);
        CHECK_ARROW_MEM(
            this->data_array->buffers[0]->Resize(buffer_size, false),
            "Resize Failed!");
    }

    // Now simply append into the buffer.
    this->UnsafeAppendRow(in_arr, row_ind);
}

void ChunkedTableArrayBuilder::Finalize(bool shrink_to_fit) {
    // Get minimum frame size in BufferPool and set that as the
    // minimum size of any of the buffers.
    const int64_t min_buffer_allocation_size =
        bodo::BufferPool::Default()->GetSmallestSizeClassSize();

    // The rest is very similar to the constructor, except this time
    // we use this->size (and number of chars in case of strings, etc.) instead
    // of this->capacity during the buffer size calculation and pass
    // the shrink_to_fit flag through to the Resize calls.
    // After the first Resize call, we do another Resize with `shrink_to_fit`
    // set to `false`. This is important since the `size_` attribute of
    // BodoBuffers is assumed to be the actual data size in some parts
    // of the code. Doing the second Resize will set the `size_` correctly
    // without forcing any re-allocations.
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            auto [data_buffer_req_size, null_bitmap_buffer_req_size] =
                get_nullable_arr_alloc_sizes(this->data_array->dtype,
                                             this->size);
            int64_t data_buffer_alloc_size =
                std::max(data_buffer_req_size, min_buffer_allocation_size);
            int64_t null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[1]->Resize(
                                null_bitmap_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_req_size, /*shrink_to_fit*/ false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(
                this->data_array->buffers[1]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "Resize Failed!");

        } break;
        case bodo_array_type::STRING: {
            int64_t data_buffer_req_size =
                static_cast<int64_t>(this->data_array->n_sub_elems());
            int64_t data_buffer_alloc_size =
                std::max(data_buffer_req_size, min_buffer_allocation_size);
            int64_t offset_buffer_req_size =
                static_cast<int64_t>((this->size + 1) * sizeof(offset_t));
            int64_t offset_buffer_alloc_size =
                std::max(offset_buffer_req_size, min_buffer_allocation_size);
            int64_t null_bitmap_buffer_req_size =
                ::arrow::bit_util::BytesForBits(this->size);
            int64_t null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[1]->Resize(
                                offset_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[2]->Resize(
                                null_bitmap_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_req_size, /*shrink_to_fit*/ false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(
                this->data_array->buffers[1]->Resize(offset_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "Resize Failed!");
            CHECK_ARROW_MEM(
                this->data_array->buffers[2]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "Resize Failed!");
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            int64_t data_buffer_req_size =
                static_cast<int64_t>(this->size * size_type);
            int64_t data_buffer_alloc_size =
                std::max(data_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_alloc_size, shrink_to_fit),
                            "Resize Failed!");
            CHECK_ARROW_MEM(this->data_array->buffers[0]->Resize(
                                data_buffer_req_size, /*shrink_to_fit*/ false),
                            "Resize Failed!");
        } break;
        case bodo_array_type::DICT: {
            // Call recursively on the indices array
            this->dict_indices->Finalize(shrink_to_fit);
        } break;
        default: {
            throw std::runtime_error(
                "Invalid array type in ChunkedTableArrayBuilder::Finalize: " +
                GetArrType_as_string(this->data_array->arr_type));
        }
    }
}

/* ------------------------------------------------------------------------ */

/* ------------------------- ChunkedTableBuilder -------------------------- */

ChunkedTableBuilder::ChunkedTableBuilder(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    size_t chunk_size, size_t max_resize_count_for_variable_size_dtypes_)
    : active_chunk(alloc_table(arr_c_types, arr_array_types)),
      active_chunk_capacity(chunk_size),
      max_resize_count_for_variable_size_dtypes(
          max_resize_count_for_variable_size_dtypes_) {
    this->active_chunk_array_builders.reserve(arr_c_types.size());
    for (size_t i = 0; i < arr_c_types.size(); i++) {
        if (arr_array_types[i] == bodo_array_type::DICT) {
            // Set the dictionary to the one from the dict builder:
            this->active_chunk->columns[i]->child_arrays[0] =
                dict_builders[i]->dict_buff->data_array;
        }
        this->active_chunk_array_builders.emplace_back(
            this->active_chunk->columns[i], dict_builders[i],
            this->active_chunk_capacity,
            max_resize_count_for_variable_size_dtypes);
    }
    this->dummy_output_chunk =
        alloc_table_like(this->active_chunk, /*reuse_dictionaries*/ true);
}

/**
 * @brief Helper function to extract the DictBuilder objects
 * from the provided ChunkedTableArrayBuilders.
 *
 * @param array_builders ChunkedTableArrayBuilder to extract
 * the DictBuilders from.
 * @return std::vector<std::shared_ptr<DictionaryBuilder>>
 */
std::vector<std::shared_ptr<DictionaryBuilder>>
get_dict_builders_from_chunked_table_array_builders(
    const std::vector<ChunkedTableArrayBuilder>& array_builders) {
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders(
        array_builders.size());
    for (size_t i = 0; i < array_builders.size(); i++) {
        dict_builders[i] = array_builders[i].dict_builder;
    }
    return dict_builders;
}

void ChunkedTableBuilder::FinalizeActiveChunk(bool shrink_to_fit) {
    // NOP in the empty chunk case
    if (this->active_chunk_size == 0) {
        return;
    }

    // Call Finalize on all the array builders of the active chunk:
    for (auto& builder : this->active_chunk_array_builders) {
        builder.Finalize(shrink_to_fit);
    }

    // New active chunk
    std::shared_ptr<table_info> new_active_chunk =
        alloc_table_like(this->active_chunk, /*reuse_dictionaries*/ true);
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
        get_dict_builders_from_chunked_table_array_builders(
            this->active_chunk_array_builders);
    // Add chunk to the deque
    this->chunks.push_back(std::move(this->active_chunk));
    // TODO (future) Unpin this chunk.
    // Reset state for active chunk:
    this->active_chunk = std::move(new_active_chunk);
    this->active_chunk_size = 0;
    this->active_chunk_array_builders.clear();
    this->active_chunk_array_builders.reserve(
        this->active_chunk->columns.size());
    for (size_t i = 0; i < this->active_chunk->columns.size(); i++) {
        this->active_chunk_array_builders.emplace_back(
            this->active_chunk->columns[i], dict_builders[i],
            this->active_chunk_capacity,
            this->max_resize_count_for_variable_size_dtypes);
    }
}

void ChunkedTableBuilder::AppendRow(const std::shared_ptr<table_info>& in_table,
                                    int64_t row_ind) {
    // Check if the row can be appended to all arrays. If any of them
    // return false, then finalize current chunk, start a new one
    // and append there.
    for (size_t i = 0; i < this->active_chunk_array_builders.size(); i++) {
        if (!this->active_chunk_array_builders[i].CanAppendRow(
                in_table->columns[i], row_ind)) {
            this->FinalizeActiveChunk();
            break;
        }
    }

    // Now append the row to all the arrays. This is guaranteed to work
    // since either this is a new chunk, or all builders returned true
    // for CanAppendRow.
    for (size_t i = 0; i < this->active_chunk_array_builders.size(); i++) {
        this->active_chunk_array_builders[i].AppendRow(in_table->columns[i],
                                                       row_ind);
    }
    this->active_chunk_size += 1;
    this->total_size += 1;
    this->total_remaining += 1;
}

void ChunkedTableBuilder::AppendRows(std::shared_ptr<table_info> in_table,
                                     const std::span<const int64_t> row_inds) {
    // NOTE: Currently this just iterates over the row indices
    // and append them one by one (due to complexities with
    // string handling). In the future, this can be optimized.
    // (https://bodo.atlassian.net/browse/BSE-555)
    for (int64_t row_ind : row_inds) {
        this->AppendRow(in_table, row_ind);
    }
}

void ChunkedTableBuilder::AppendJoinOutput(
    std::shared_ptr<table_info> build_table,
    std::shared_ptr<table_info> probe_table,
    const std::span<const int64_t> build_idxs,
    const std::span<const int64_t> probe_idxs,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    // NOTE: Currently this just iterates over the row indices
    // and append them one by one (due to complexities with
    // string handling). In the future, this can be optimized.
    // (https://bodo.atlassian.net/browse/BSE-555).

    if (build_idxs.size() != probe_idxs.size()) {
        throw std::runtime_error(
            "ChunkedTableBuilder::AppendJoinOutput: Length of build_idxs and "
            "probe_idxs does not match!");
    }
    if ((build_kept_cols.size() + probe_kept_cols.size()) !=
        this->active_chunk_array_builders.size()) {
        throw std::runtime_error(
            "ChunkedTableBuilder::AppendJoinOutput: build_kept_cols.size() + "
            "probe_kept_cols.size()) != "
            "this->active_chunk_array_builders.size()");
    }

    size_t build_ncols = build_kept_cols.size();

    for (size_t i_row = 0; i_row < build_idxs.size(); i_row++) {
        // Check if the row can be appended to all arrays. If any of them
        // return false, then finalize current chunk, start a new one
        // and append there.
        for (size_t i_col = 0; i_col < this->active_chunk_array_builders.size();
             i_col++) {
            if (i_col < build_ncols) {
                if (!this->active_chunk_array_builders[i_col].CanAppendRow(
                        build_table->columns[build_kept_cols[i_col]],
                        build_idxs[i_row])) {
                    this->FinalizeActiveChunk();
                    break;
                }
            } else {
                if (!this->active_chunk_array_builders[i_col].CanAppendRow(
                        probe_table
                            ->columns[probe_kept_cols[i_col - build_ncols]],
                        probe_idxs[i_row])) {
                    this->FinalizeActiveChunk();
                    break;
                }
            }
        }

        // Now append the row to all the arrays. This is guaranteed to work
        // since either this is a new chunk, or all builders returned true
        // for CanAppendRow.
        for (size_t i_col = 0; i_col < this->active_chunk_array_builders.size();
             i_col++) {
            if (i_col < build_ncols) {
                this->active_chunk_array_builders[i_col].AppendRow(
                    build_table->columns[build_kept_cols[i_col]],
                    build_idxs[i_row]);
            } else {
                this->active_chunk_array_builders[i_col].AppendRow(
                    probe_table->columns[probe_kept_cols[i_col - build_ncols]],
                    probe_idxs[i_row]);
            }
        }
        this->active_chunk_size += 1;
        this->total_size += 1;
        this->total_remaining += 1;
    }
}

void ChunkedTableBuilder::Finalize(bool shrink_to_fit) {
    // Finalize the active chunk:
    if (this->active_chunk_size > 0) {
        // Call Finalize on all the array builders of the active chunk:
        for (auto& builder : this->active_chunk_array_builders) {
            builder.Finalize(shrink_to_fit);
        }
        // Add chunk to the deque
        this->chunks.push_back(std::move(this->active_chunk));
        // TODO (future) Unpin this chunk.
    }
    // Reset state for active chunk:
    this->active_chunk = nullptr;
    this->active_chunk_size = 0;
    this->active_chunk_array_builders.clear();
}

std::tuple<std::shared_ptr<table_info>, int64_t> ChunkedTableBuilder::PopChunk(
    bool force_return) {
    // If there's no finalized chunks available and force_return = true,
    // then finalize the active chunk.
    if ((this->chunks.size() == 0) && force_return) {
        this->FinalizeActiveChunk();
    }

    // If there's a finalized chunk available, pop and return that.
    // Note that FinalizeActiveChunk would have been a NOP if the
    // active-chunk was empty, so we still need this check.
    if (this->chunks.size() > 0) {
        std::shared_ptr<table_info> chunk = this->chunks.front();
        this->chunks.pop_front();
        size_t chunk_nrows = chunk->nrows();
        if (this->dummy_output_chunk->ncols() == 0) {
            // In the all columns dead case, chunk->nrows() will be 0,
            // but it should actually be based on active_chunk_capacity
            // and total_remaining.
            chunk_nrows =
                std::min(this->active_chunk_capacity, this->total_remaining);
        }

        this->total_remaining -= chunk_nrows;
        return std::tuple(chunk, chunk_nrows);
    }
    return std::tuple(alloc_table_like(this->dummy_output_chunk,
                                       /*reuse_dictionaries*/ true),
                      0);
}

/* ------------------------------------------------------------------------ */
