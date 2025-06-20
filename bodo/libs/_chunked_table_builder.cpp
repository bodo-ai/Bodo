#include "_chunked_table_builder.h"
#include <numeric>
#include <utility>

#include "_dict_builder.h"
#include "_query_profile_collector.h"
#include "_table_builder_utils.h"
#include "arrow/util/bit_util.h"

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
      size(this->data_array->length),
      capacity(chunk_size),
      max_resize_count(_max_resize_count) {
    if (this->data_array->length != 0) {
        throw std::runtime_error(
            "ChunkedTableArrayBuilder::ChunkedTableArrayBuilder: Length of "
            "input array is not 0!");
    }

    // Get minimum frame size in BufferPool and set that as the
    // minimum size of any of the buffers (when spilling is available).
    // When spilling is not enabled, we don't want to over-allocate and
    // inflate the BufferPool statistics and cause early OOMs (triggered by
    // BufferPool not the OS).
    const int64_t min_buffer_allocation_size =
        bodo::BufferPool::Default()->is_spilling_enabled()
            ? bodo::BufferPool::Default()->GetSmallestSizeClassSize()
            : 0;

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
            // TODO XXX Use Reserve here instead of Resize?
            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 data_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[1]->Resize(
                                 null_bitmap_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");

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
            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 data_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[1]->Resize(
                                 offset_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[2]->Resize(
                                 null_bitmap_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            int64_t data_buffer_alloc_size =
                std::max(static_cast<int64_t>(this->capacity * size_type),
                         min_buffer_allocation_size);
            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 data_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            uint64_t utc_size_type = numpy_item_size[Bodo_CTypes::TIMESTAMPTZ];
            int64_t utc_buffer_alloc_size =
                static_cast<int64_t>(this->capacity * utc_size_type);
            utc_buffer_alloc_size =
                std::max(utc_buffer_alloc_size, min_buffer_allocation_size);

            uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
            int64_t offset_buffer_alloc_size =
                static_cast<int64_t>(this->capacity * offset_size_type);
            offset_buffer_alloc_size =
                std::max(offset_buffer_alloc_size, min_buffer_allocation_size);

            int64_t null_bitmap_buffer_alloc_size =
                ::arrow::bit_util::BytesForBits(this->capacity);
            null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_alloc_size, min_buffer_allocation_size);

            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 utc_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[1]->Resize(
                                 offset_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[2]->Resize(
                                 null_bitmap_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
        } break;
        case bodo_array_type::DICT: {
            if (_dict_builder == nullptr) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::ChunkedTableArrayBuilder: "
                    "dict_builder is nullptr for a "
                    "dict-encoded string array!");
            }
            if (_dict_builder->dict_buff->data_array.get() !=
                this->data_array->child_arrays[0].get()) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::ChunkedTableArrayBuilder: "
                    "specified dict_builder does not "
                    "match dictionary of _data_array!");
            }
            // Recursively call the constructor on the indices array.
            this->dict_indices = std::make_shared<ChunkedTableArrayBuilder>(
                this->data_array->child_arrays[1], nullptr, chunk_size,
                _max_resize_count);
        } break;
        case bodo_array_type::ARRAY_ITEM: {
            this->child_array_builders.emplace_back(
                this->data_array->child_arrays[0],
                this->dict_builder->child_dict_builders[0]);
            int64_t offset_buffer_alloc_size = std::max(
                static_cast<int64_t>((this->capacity + 1) * sizeof(offset_t)),
                min_buffer_allocation_size);
            int64_t null_bitmap_buffer_alloc_size =
                std::max(::arrow::bit_util::BytesForBits(this->capacity),
                         min_buffer_allocation_size);
            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 offset_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
            CHECK_ARROW_BASE(this->data_array->buffers[1]->Resize(
                                 null_bitmap_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
        } break;
        case bodo_array_type::MAP: {
            this->child_array_builders.emplace_back(
                this->data_array->child_arrays[0],
                this->dict_builder->child_dict_builders[0]);
        } break;
        case bodo_array_type::STRUCT: {
            for (size_t i = 0; i < this->data_array->child_arrays.size(); i++) {
                const std::shared_ptr<array_info>& child_array =
                    this->data_array->child_arrays[i];
                this->child_array_builders.emplace_back(
                    child_array, this->dict_builder->child_dict_builders[i]);
            }
            int64_t null_bitmap_buffer_alloc_size =
                std::max(::arrow::bit_util::BytesForBits(this->capacity),
                         min_buffer_allocation_size);
            CHECK_ARROW_BASE(this->data_array->buffers[0]->Resize(
                                 null_bitmap_buffer_alloc_size, false),
                             "ChunkedTableArrayBuilder::"
                             "ChunkedTableArrayBuilder: Resize failed!");
        } break;
        default:
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::ChunkedTableArrayBuilder: Invalid "
                "array type " +
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

template <bodo_array_type::arr_type_enum out_arr_type,
          bodo_array_type::arr_type_enum in_arr_type,
          Bodo_CTypes::CTypeEnum dtype, typename IndexT>
    requires(out_arr_type == bodo_array_type::ARRAY_ITEM &&
             in_arr_type == bodo_array_type::ARRAY_ITEM)
void ChunkedTableArrayBuilder::UnsafeAppendRows(
    const std::shared_ptr<array_info>& in_arr,
    const std::span<const IndexT> idxs, size_t idx_start, size_t idx_length) {
    offset_t* out_offsets = (offset_t*)this->data_array->data1<out_arr_type>();
    offset_t* in_offsets = (offset_t*)in_arr->data1<in_arr_type>();

    uint8_t* out_bitmask =
        (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
    const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<in_arr_type>();

    for (size_t i = 0; i < idx_length; i++) {
        int64_t row_idx = idxs[i + idx_start];
        // Copy the offsets
        offset_t new_length =
            row_idx < 0 ? 0 : in_offsets[row_idx + 1] - in_offsets[row_idx];
        out_offsets[this->size + i + 1] =
            out_offsets[this->size + i] + new_length;
        // Copy the null bitmap
        bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
        SetBitTo(out_bitmask, this->size + i, null_bit);
        // Append rows for inner array
        for (offset_t j = in_offsets[row_idx]; j < in_offsets[row_idx + 1];
             j++) {
            this->child_array_builders.front().ReserveArrayRow(
                in_arr->child_arrays[0], j);
            this->child_array_builders.front().UnsafeAppendRow(
                in_arr->child_arrays[0], static_cast<int64_t>(j));
        }
    }

    this->data_array->length += idx_length;
}

template <bodo_array_type::arr_type_enum out_arr_type,
          bodo_array_type::arr_type_enum in_arr_type,
          Bodo_CTypes::CTypeEnum dtype, typename IndexT>
    requires(out_arr_type == bodo_array_type::MAP &&
             in_arr_type == bodo_array_type::MAP)
void ChunkedTableArrayBuilder::UnsafeAppendRows(
    const std::shared_ptr<array_info>& in_arr,
    const std::span<const IndexT> idxs, size_t idx_start, size_t idx_length) {
    for (size_t i = 0; i < idx_length; i++) {
        int64_t row_idx = idxs[i + idx_start];
        // Append rows for child array
        this->child_array_builders[0].ReserveArrayRow(in_arr->child_arrays[0],
                                                      row_idx);
        this->child_array_builders[0].UnsafeAppendRow(in_arr->child_arrays[0],
                                                      row_idx);
    }

    this->data_array->length += idx_length;
}

template <bodo_array_type::arr_type_enum out_arr_type,
          bodo_array_type::arr_type_enum in_arr_type,
          Bodo_CTypes::CTypeEnum dtype, typename IndexT>
    requires(out_arr_type == bodo_array_type::STRUCT &&
             in_arr_type == bodo_array_type::STRUCT)
void ChunkedTableArrayBuilder::UnsafeAppendRows(
    const std::shared_ptr<array_info>& in_arr,
    const std::span<const IndexT> idxs, size_t idx_start, size_t idx_length) {
    uint8_t* out_bitmask =
        (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
    const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<in_arr_type>();

    for (size_t i = 0; i < idx_length; i++) {
        int64_t row_idx = idxs[i + idx_start];
        // Copy the null bitmap
        bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
        SetBitTo(out_bitmask, this->size + i, null_bit);
        // Append rows for child arrays
        for (size_t j = 0; j < this->child_array_builders.size(); ++j) {
            this->child_array_builders[j].ReserveArrayRow(
                in_arr->child_arrays[j], row_idx);
            this->child_array_builders[j].UnsafeAppendRow(
                in_arr->child_arrays[j], row_idx);
        }
    }
    if (this->data_array->field_names.size() == 0) {
        this->data_array->field_names = in_arr->field_names;
    }

    this->data_array->length += idx_length;
}

void ChunkedTableArrayBuilder::Finalize(bool shrink_to_fit) {
    // Get minimum frame size in BufferPool and set that as the
    // minimum size of any of the buffers (when spilling is available).
    // When spilling is not enabled, we don't want to over-allocate and
    // inflate the BufferPool statistics and cause early OOMs (triggered by
    // BufferPool not the OS).
    const int64_t min_buffer_allocation_size =
        bodo::BufferPool::Default()->is_spilling_enabled()
            ? bodo::BufferPool::Default()->GetSmallestSizeClassSize()
            : 0;

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
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(
                    null_bitmap_buffer_alloc_size, shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");

        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            uint64_t utc_size_type = numpy_item_size[Bodo_CTypes::TIMESTAMPTZ];
            int64_t utc_buffer_req_size =
                static_cast<int64_t>(this->size * utc_size_type);
            int64_t utc_buffer_alloc_size =
                std::max(utc_buffer_req_size, min_buffer_allocation_size);

            uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
            int64_t offset_buffer_req_size =
                static_cast<int64_t>(this->size * offset_size_type);
            int64_t offset_buffer_alloc_size =
                std::max(offset_buffer_req_size, min_buffer_allocation_size);

            int64_t null_bitmap_buffer_req_size =
                ::arrow::bit_util::BytesForBits(this->size);
            int64_t null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_req_size, min_buffer_allocation_size);

            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(utc_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(offset_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[2]->Resize(
                    null_bitmap_buffer_alloc_size, shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(utc_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(offset_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[2]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");

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
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(offset_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[2]->Resize(
                    null_bitmap_buffer_alloc_size, shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            // TODO Replace these Resize calls with SetSize
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(offset_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[2]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            int64_t data_buffer_req_size =
                static_cast<int64_t>(this->size * size_type);
            int64_t data_buffer_alloc_size =
                std::max(data_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(data_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
        } break;
        case bodo_array_type::DICT: {
            // Call recursively on the indices array
            this->dict_indices->Finalize(shrink_to_fit);
        } break;
        case bodo_array_type::ARRAY_ITEM: {
            int64_t offset_buffer_req_size =
                static_cast<int64_t>((this->size + 1) * sizeof(offset_t));
            int64_t offset_buffer_alloc_size =
                std::max(offset_buffer_req_size, min_buffer_allocation_size);
            int64_t null_bitmap_buffer_req_size =
                ::arrow::bit_util::BytesForBits(this->size);
            int64_t null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(offset_buffer_alloc_size,
                                                     shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(
                    null_bitmap_buffer_alloc_size, shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(offset_buffer_req_size,
                                                     /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[1]->Resize(
                    null_bitmap_buffer_req_size, /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
        } break;
        case bodo_array_type::MAP: {
            // Map doesn't have any buffers to resize
        } break;
        case bodo_array_type::STRUCT: {
            int64_t null_bitmap_buffer_req_size =
                ::arrow::bit_util::BytesForBits(this->size);
            int64_t null_bitmap_buffer_alloc_size = std::max(
                null_bitmap_buffer_req_size, min_buffer_allocation_size);
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(
                    null_bitmap_buffer_alloc_size, shrink_to_fit),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
            CHECK_ARROW_BASE(
                this->data_array->buffers[0]->Resize(
                    null_bitmap_buffer_alloc_size,
                    /*shrink_to_fit*/ false),
                "ChunkedTableArrayBuilder::Finalize: Resize failed!");
        } break;
        default: {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::Finalize: Invalid array type " +
                GetArrType_as_string(this->data_array->arr_type));
        }
    }
}

void ChunkedTableArrayBuilder::Reset() {
    this->resize_count = this->data_array->length = 0;
    // Reset the dictionary to point to the one
    // in the dict builder:
    set_array_dict_from_builder(this->data_array, this->dict_builder);
    switch (this->data_array->arr_type) {
        // TODO XXX Use SetSize here instead of Resize?
        case bodo_array_type::NULLABLE_INT_BOOL: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[2]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        case bodo_array_type::NUMPY: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        case bodo_array_type::STRING: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[2]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->Reset();
        } break;
        case bodo_array_type::ARRAY_ITEM: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        case bodo_array_type::MAP: {
            // Map doesn't have any buffers to reset
        } break;
        case bodo_array_type::STRUCT: {
            CHECK_ARROW_BASE(data_array->buffers[0]->Resize(0, false),
                             "ChunkedTableArrayBuilder::Reset: Resize failed!");
        } break;
        default: {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::Reset: Invalid array type in "
                "Clear " +
                GetArrType_as_string(data_array->arr_type));
        }
    }
}

/* ------------------------------------------------------------------------ */

/* --------------------- AbstractChunkedTableBuilder ---------------------- */

AbstractChunkedTableBuilder::AbstractChunkedTableBuilder(
    const std::shared_ptr<bodo::Schema>& schema,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    size_t chunk_size, size_t max_resize_count_for_variable_size_dtypes_,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm)
    : active_chunk(alloc_table(schema, pool, mm)),
      active_chunk_capacity(chunk_size),
      max_resize_count_for_variable_size_dtypes(
          max_resize_count_for_variable_size_dtypes_),
      dict_builders(dict_builders_),
      pool(pool),
      mm(mm) {
    assert(chunk_size > 0);
    this->active_chunk_array_builders.reserve(active_chunk->ncols());
    for (size_t i = 0; i < active_chunk->ncols(); i++) {
        // Set the dictionary to the one from the dict builder
        set_array_dict_from_builder(active_chunk->columns[i], dict_builders[i]);
        this->active_chunk_array_builders.emplace_back(
            this->active_chunk->columns[i], dict_builders[i],
            this->active_chunk_capacity,
            max_resize_count_for_variable_size_dtypes);
    }
    this->dummy_output_chunk = alloc_table_like(
        this->active_chunk, /*reuse_dictionaries*/ true, pool, mm);
}

void AbstractChunkedTableBuilder::FinalizeActiveChunk(bool shrink_to_fit) {
    // NOP in the empty chunk case
    if (this->active_chunk_size == 0) {
        return;
    }

    // Call Finalize on all the array builders of the active chunk:
    for (auto& builder : this->active_chunk_array_builders) {
        builder.Finalize(shrink_to_fit);
    }

    // New active chunk
    std::shared_ptr<table_info> new_active_chunk = alloc_table_like(
        this->active_chunk, /*reuse_dictionaries*/ true, pool, mm);
    this->PushActiveChunk();
    // Reset state for active chunk:
    this->active_chunk = std::move(new_active_chunk);
    this->active_chunk_size = 0;
    this->active_chunk_array_builders.clear();
    this->active_chunk_array_builders.reserve(
        this->active_chunk->columns.size());
    for (size_t i = 0; i < this->active_chunk->columns.size(); i++) {
        this->active_chunk_array_builders.emplace_back(
            this->active_chunk->columns[i], this->dict_builders[i],
            this->active_chunk_capacity,
            this->max_resize_count_for_variable_size_dtypes);
    }
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows, const size_t num_append_rows,
    const int64_t in_table_start_offset) {
    // Convert the bit-vector to a vector of indices. Offset the
    // entries by in_table_start_offset.
    // We do the "+1" since we need to have at least 1 element in the array for
    // the branchless loop to work.
    std::vector<int64_t> idxs(num_append_rows + 1);
    size_t next_idx = 0;
    for (size_t i_row = 0; i_row < append_rows.size(); i_row++) {
        idxs[next_idx] = in_table_start_offset + i_row;
        size_t delta = append_rows[i_row] ? 1 : 0;
        next_idx += delta;
    }
    assert(num_append_rows == next_idx);
    idxs.resize(next_idx);

    this->AppendBatch(in_table, idxs);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows, const int64_t in_table_start_offset) {
    // Calculate number of rows to append
    size_t num_append_rows =
        std::accumulate(append_rows.begin(), append_rows.end(), (size_t)0);
    this->AppendBatch(in_table, append_rows, num_append_rows,
                      in_table_start_offset);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::unique_ptr<uint8_t[]>& append_rows, const size_t num_append_rows,
    const int64_t in_table_start_offset) {
    std::vector<int64_t> idxs;
    idxs.reserve(num_append_rows);

    // Convert the bit-vector to a vector of indices. Offset the
    // entries by in_table_start_offset.
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (arrow::bit_util::GetBit(append_rows.get(), i_row)) {
            idxs.emplace_back(in_table_start_offset + i_row);
        }
    }

    this->AppendBatch(in_table, idxs);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::unique_ptr<uint8_t[]>& append_rows,
    const int64_t in_table_start_offset) {
    // Count the number of rows to append.
    // This won't be exact for the last word, but that's fine, it will be within
    // 63 and will always overestimate.
    size_t num_append_rows = 0;
    for (int64_t i = 0;
         i < (arrow::bit_util::BytesForBits(in_table->nrows()) / 8); i++) {
        num_append_rows +=
            arrow::bit_util::PopCount(((uint64_t*)append_rows.get())[i]);
    }
    this->AppendBatch(in_table, append_rows, num_append_rows,
                      in_table_start_offset);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table) {
    // Build all indices vector
    std::vector<int64_t> idxs;
    idxs.reserve(in_table->nrows());

    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        idxs.emplace_back(i_row);
    }

    this->AppendBatch(in_table, idxs);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::span<const int64_t> idxs) {
    std::vector<uint64_t> columns(in_table->columns.size());
    std::iota(columns.begin(), columns.end(), 0);
    AppendBatch(in_table, idxs, columns);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::span<const int32_t> idxs, const std::span<const uint64_t> cols) {
    this->AppendBatch<int32_t>(in_table, idxs, cols);
}

void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::span<const int64_t> idxs, const std::span<const uint64_t> cols) {
    this->AppendBatch<int64_t>(in_table, idxs, cols);
}

template <typename IndexT>
void AbstractChunkedTableBuilder::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::span<const IndexT> idxs, const std::span<const uint64_t> cols) {
#ifndef NUM_ROWS_CAN_APPEND_COL
#define NUM_ROWS_CAN_APPEND_COL(ARR_TYPE)                                    \
    batch_length =                                                           \
        this->active_chunk_array_builders[i_col].NumRowsCanAppend<ARR_TYPE>( \
            in_arr, idxs, curr_row, batch_length)
#endif

#ifndef APPEND_ROWS_COL
#define APPEND_ROWS_COL(OUT_ARR_TYPE, IN_ARR_TYPE, DTYPE)                     \
    found_match = true;                                                       \
    this->active_chunk_array_builders[i_col]                                  \
        .AppendRows<OUT_ARR_TYPE, IN_ARR_TYPE, DTYPE>(in_arr, idxs, curr_row, \
                                                      batch_length)
#endif

    // See comment in AppendJoinOutput for a description of the procedure.
    time_pt start = start_timer();
    size_t curr_row = 0;
    size_t n_rows = idxs.size();
    while (curr_row < n_rows) {
        // initialize the batch end to all rows.
        size_t batch_length = n_rows - curr_row;
        // Determine a consistent batch end across all columns. This value will
        // be the min of any column.
        for (size_t i_col = 0; i_col < cols.size(); i_col++) {
            std::shared_ptr<array_info>& in_arr =
                in_table->columns[cols[i_col]];
            if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::NULLABLE_INT_BOOL);
            } else if (in_arr->arr_type == bodo_array_type::NUMPY) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::NUMPY);
            } else if (in_arr->arr_type == bodo_array_type::STRING) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::STRING);
            } else if (in_arr->arr_type == bodo_array_type::DICT) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::DICT);
            } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::ARRAY_ITEM);
            } else if (in_arr->arr_type == bodo_array_type::MAP) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::MAP);
            } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::STRUCT);
            } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::TIMESTAMPTZ);
            } else {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::AppendJoinOutput: invalid "
                    "array type" +
                    GetArrType_as_string(in_arr->arr_type));
            }
        }
        // Append the actual rows.
        for (size_t i_col = 0; i_col < in_table->ncols(); i_col++) {
            bool found_match = false;
            std::shared_ptr<array_info>& in_arr = in_table->columns[i_col];
            if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                this->dummy_output_chunk->columns[i_col]->arr_type ==
                    bodo_array_type::NULLABLE_INT_BOOL) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DATE:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT128);
                        break;
                    default:
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                       this->dummy_output_chunk->columns[i_col]->arr_type ==
                           bodo_array_type::NUMPY) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT128);
                        break;
                    default:
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::NUMPY &&
                       this->dummy_output_chunk->columns[i_col]->arr_type ==
                           bodo_array_type::NUMPY) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::_BOOL);
                        break;
                    // TODO: Remove when nullable timestamp is supported
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DATE:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT128);
                        break;
                    default:
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::NUMPY &&
                       this->dummy_output_chunk->columns[i_col]->arr_type ==
                           bodo_array_type::NULLABLE_INT_BOOL) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DATE:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT128);
                        break;
                    default:
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::STRING) {
                if (in_arr->dtype == Bodo_CTypes::STRING) {
                    APPEND_ROWS_COL(bodo_array_type::STRING,
                                    bodo_array_type::STRING,
                                    Bodo_CTypes::STRING);
                } else if (in_arr->dtype == Bodo_CTypes::BINARY) {
                    APPEND_ROWS_COL(bodo_array_type::STRING,
                                    bodo_array_type::STRING,
                                    Bodo_CTypes::BINARY);
                }
            } else if (in_arr->arr_type == bodo_array_type::DICT) {
                if (in_arr->dtype == Bodo_CTypes::STRING) {
                    APPEND_ROWS_COL(bodo_array_type::DICT,
                                    bodo_array_type::DICT, Bodo_CTypes::STRING);
                }
            } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
                APPEND_ROWS_COL(bodo_array_type::TIMESTAMPTZ,
                                bodo_array_type::TIMESTAMPTZ,
                                Bodo_CTypes::TIMESTAMPTZ);
            } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
                if (in_arr->dtype == Bodo_CTypes::LIST) {
                    APPEND_ROWS_COL(bodo_array_type::ARRAY_ITEM,
                                    bodo_array_type::ARRAY_ITEM,
                                    Bodo_CTypes::LIST);
                }
            } else if (in_arr->arr_type == bodo_array_type::MAP) {
                if (in_arr->dtype == Bodo_CTypes::MAP) {
                    APPEND_ROWS_COL(bodo_array_type::MAP, bodo_array_type::MAP,
                                    Bodo_CTypes::MAP);
                }
            } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
                if (in_arr->dtype == Bodo_CTypes::STRUCT) {
                    APPEND_ROWS_COL(bodo_array_type::STRUCT,
                                    bodo_array_type::STRUCT,
                                    Bodo_CTypes::STRUCT);
                }
            }
            if (!found_match) {
                throw std::runtime_error(
                    "AbstractChunkedTableBuilder::AppendBatch: Could not "
                    "append "
                    "column " +
                    std::to_string(i_col));
            }
        }

        // Update the metadata.
        this->active_chunk_size += batch_length;
        this->total_size += batch_length;
        this->total_remaining += batch_length;
        this->max_reached_size =
            std::max(this->max_reached_size, this->total_remaining);
        // Update the curr_row
        curr_row += batch_length;
        // Check if we need to finalize
        if (curr_row < n_rows) {
            this->FinalizeActiveChunk();
        }
    }
    this->append_time += end_timer(start);
#undef NUM_ROWS_CAN_APPEND_COL
#undef APPEND_ROWS_COL
}

void AbstractChunkedTableBuilder::AppendJoinOutput(
    std::shared_ptr<table_info> build_table,
    std::shared_ptr<table_info> probe_table,
    const std::span<const int64_t> build_idxs,
    const std::span<const int64_t> probe_idxs,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    if (build_idxs.size() != probe_idxs.size()) {
        throw std::runtime_error(
            "AbstractChunkedTableBuilder::AppendJoinOutput: Length of "
            "build_idxs and "
            "probe_idxs does not match!");
    }
    if ((build_kept_cols.size() + probe_kept_cols.size()) !=
        this->active_chunk_array_builders.size()) {
        throw std::runtime_error(
            "AbstractChunkedTableBuilder::AppendJoinOutput: "
            "build_kept_cols.size() + "
            "probe_kept_cols.size()) != "
            "this->active_chunk_array_builders.size()");
    }

#ifndef NUM_ROWS_CAN_APPEND_COL
// The max value returned is batch_length so we don't need to do
// a min here.
#define NUM_ROWS_CAN_APPEND_COL(ARR_TYPE)                    \
    batch_length = this->active_chunk_array_builders[i_col]  \
                       .NumRowsCanAppend<ARR_TYPE, int64_t>( \
                           col, idxs, curr_row, batch_length)
#endif

#ifndef APPEND_ROWS_COL
#define APPEND_ROWS_COL(OUT_ARR_TYPE, IN_ARR_TYPE, DTYPE)       \
    this->active_chunk_array_builders[i_col]                    \
        .AppendRows<OUT_ARR_TYPE, IN_ARR_TYPE, DTYPE, int64_t>( \
            col, idxs, curr_row, batch_length)
#endif

    time_pt start = start_timer();
    size_t probe_ncols = probe_kept_cols.size();

    // We want to append rows in a columnar process. To do this we split an
    // append into three steps:
    //
    // Step 1: Check how many rows can be appended to active chunk from each
    // array.
    //    We will append the minimum of these.
    // Step 2: Append a batch of rows to the output in a columnar step.
    // Step 3: Finalize the active chunk if there are more rows to append.
    //
    // In this way we can process a table in "batches" of rows based on the
    // available space.
    size_t curr_row = 0;
    size_t max_rows = build_idxs.size();
    while (curr_row < max_rows) {
        // initialize the batch end to all rows.
        size_t batch_length = max_rows - curr_row;
        // Determine a consistent batch end across all columns. This value will
        // be the min of any column.
        for (size_t i_col = 0; i_col < this->active_chunk_array_builders.size();
             i_col++) {
            bool is_probe = i_col < probe_ncols;

            std::shared_ptr<array_info> col;
            std::span<const int64_t> idxs;
            if (is_probe) {
                col = probe_table->columns[probe_kept_cols[i_col]];
                idxs = probe_idxs;
            } else {
                col =
                    build_table->columns[build_kept_cols[i_col - probe_ncols]];
                idxs = build_idxs;
            }

            if (col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::NULLABLE_INT_BOOL);
            } else if (col->arr_type == bodo_array_type::NUMPY) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::NUMPY);
            } else if (col->arr_type == bodo_array_type::STRING) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::STRING);
            } else if (col->arr_type == bodo_array_type::DICT) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::DICT);
            } else if (col->arr_type == bodo_array_type::ARRAY_ITEM) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::ARRAY_ITEM);
            } else if (col->arr_type == bodo_array_type::MAP) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::MAP);
            } else if (col->arr_type == bodo_array_type::STRUCT) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::STRUCT);
            } else if (col->arr_type == bodo_array_type::TIMESTAMPTZ) {
                NUM_ROWS_CAN_APPEND_COL(bodo_array_type::TIMESTAMPTZ);
            } else {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::AppendJoinOutput: invalid "
                    "array type" +
                    GetArrType_as_string(col->arr_type));
            }
        }
        // Append the actual rows.
        for (size_t i_col = 0; i_col < this->active_chunk_array_builders.size();
             i_col++) {
            bool is_probe = i_col < probe_ncols;

            std::shared_ptr<array_info> col;
            std::span<const int64_t> idxs;
            if (is_probe) {
                col = probe_table->columns[probe_kept_cols[i_col]];
                idxs = probe_idxs;
            } else {
                col =
                    build_table->columns[build_kept_cols[i_col - probe_ncols]];
                idxs = build_idxs;
            }

            bodo_array_type::arr_type_enum out_arr_type =
                this->active_chunk_array_builders[i_col].data_array->arr_type;
            bodo_array_type::arr_type_enum in_arr_type = col->arr_type;

            if (out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                in_arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                switch (col->dtype) {
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::DATE:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NULLABLE_INT_BOOL,
                                        Bodo_CTypes::DECIMAL);
                        break;
                    default:
                        break;
                }
            }

            // Input array may be NUMPY and converted to NULLABLE_INT_BOOL
            // because of an outer join. This will be removed when we remove
            // Numpy arrays. Once that is done we can also remove references
            // to the arr_type of the data array.
            else if (out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                     in_arr_type == bodo_array_type::NUMPY) {
                switch (col->dtype) {
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    // TODO: Remove when nullable timestamp is supported
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NULLABLE_INT_BOOL,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATETIME);
                        break;
                    default:
                        break;
                }
            }

            // BOTH NUMPY
            else if (out_arr_type == bodo_array_type::NUMPY &&
                     in_arr_type == bodo_array_type::NUMPY) {
                switch (col->dtype) {
                    case Bodo_CTypes::_BOOL:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::INT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::UINT8:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::INT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::UINT16:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::INT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::UINT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::INT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::FLOAT64);
                        break;
                    // TODO: Remove when nullable timestamp is supported
                    case Bodo_CTypes::DATETIME:
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        // TODO(njriasan): Support TIMEDELTA array as a
                        // nullable array
                        APPEND_ROWS_COL(bodo_array_type::NUMPY,
                                        bodo_array_type::NUMPY,
                                        Bodo_CTypes::TIMEDELTA);
                        break;
                    default:
                        break;
                }
            }

            // String + Binary
            else if (out_arr_type == bodo_array_type::STRING &&
                     in_arr_type == bodo_array_type::STRING) {
                if (col->dtype == Bodo_CTypes::STRING) {
                    APPEND_ROWS_COL(bodo_array_type::STRING,
                                    bodo_array_type::STRING,
                                    Bodo_CTypes::STRING);
                } else if (col->dtype == Bodo_CTypes::BINARY) {
                    APPEND_ROWS_COL(bodo_array_type::STRING,
                                    bodo_array_type::STRING,
                                    Bodo_CTypes::BINARY);
                }
            }

            // DICT ENCODING
            else if (out_arr_type == bodo_array_type::DICT &&
                     in_arr_type == bodo_array_type::DICT) {
                if (col->dtype == Bodo_CTypes::STRING) {
                    APPEND_ROWS_COL(bodo_array_type::DICT,
                                    bodo_array_type::DICT, Bodo_CTypes::STRING);
                }
            }

            // NESTED ARRAY
            else if (out_arr_type == bodo_array_type::ARRAY_ITEM &&
                     in_arr_type == bodo_array_type::ARRAY_ITEM) {
                if (col->dtype == Bodo_CTypes::LIST) {
                    APPEND_ROWS_COL(bodo_array_type::ARRAY_ITEM,
                                    bodo_array_type::ARRAY_ITEM,
                                    Bodo_CTypes::LIST);
                }
            }

            // STRUCT ARRAY
            else if (out_arr_type == bodo_array_type::STRUCT &&
                     in_arr_type == bodo_array_type::STRUCT) {
                if (col->dtype == Bodo_CTypes::STRUCT) {
                    APPEND_ROWS_COL(bodo_array_type::STRUCT,
                                    bodo_array_type::STRUCT,
                                    Bodo_CTypes::STRUCT);
                }
            }

            // MAP ARRAY
            else if (out_arr_type == bodo_array_type::MAP &&
                     in_arr_type == bodo_array_type::MAP) {
                if (col->dtype == Bodo_CTypes::MAP) {
                    APPEND_ROWS_COL(bodo_array_type::MAP, bodo_array_type::MAP,
                                    Bodo_CTypes::MAP);
                }
            }

            // TIMESTAMPTZ ARRAY
            else if (out_arr_type == bodo_array_type::TIMESTAMPTZ &&
                     in_arr_type == bodo_array_type::TIMESTAMPTZ) {
                if (col->dtype == Bodo_CTypes::TIMESTAMPTZ) {
                    APPEND_ROWS_COL(bodo_array_type::TIMESTAMPTZ,
                                    bodo_array_type::TIMESTAMPTZ,
                                    Bodo_CTypes::TIMESTAMPTZ);
                }
            }

            else {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::AppendJoinOutput: invalid "
                    "array types " +
                    GetArrType_as_string(in_arr_type) + " and " +
                    GetArrType_as_string(out_arr_type));
            }
        }
        // Update the metadata.
        this->active_chunk_size += batch_length;
        this->total_size += batch_length;
        this->total_remaining += batch_length;
        this->max_reached_size =
            std::max(this->max_reached_size, this->total_remaining);
        // Update the curr_row
        curr_row += batch_length;
        // Check if we need to finalize
        if (curr_row < max_rows) {
            this->FinalizeActiveChunk();
        }
    }

    this->append_time += end_timer(start);

#undef NUM_ROWS_CAN_APPEND_COL
#undef APPEND_ROWS_COL
}

void AbstractChunkedTableBuilder::Finalize(bool shrink_to_fit) {
    // Finalize the active chunk:
    if (this->active_chunk_size > 0) {
        // Call Finalize on all the array builders of the active
        // chunk:
        for (auto& builder : this->active_chunk_array_builders) {
            builder.Finalize(shrink_to_fit);
        }
        this->PushActiveChunk();
    }
    // Reset state for active chunk:
    this->active_chunk = nullptr;
    this->active_chunk_size = 0;
    this->active_chunk_array_builders.clear();
}

std::tuple<std::shared_ptr<table_info>, int64_t>
AbstractChunkedTableBuilder::PopChunk(bool force_return) {
    // If there's no finalized chunks available and force_return =
    // true, then finalize the active chunk.
    if ((this->NumReadyChunks() == 0) && force_return) {
        this->FinalizeActiveChunk();
    }

    // If there's a finalized chunk available, pop and return that.
    // Note that FinalizeActiveChunk would have been a NOP if the
    // active_chunk was empty, so we still need this check.
    if (this->NumReadyChunks() > 0) {
        std::shared_ptr<table_info> chunk = this->PopFront();
        size_t chunk_nrows = chunk->nrows();
        if (this->dummy_output_chunk->ncols() == 0) {
            // In the all columns dead case, chunk->nrows() will be
            // 0, but it should actually be based on
            // active_chunk_capacity and total_remaining.
            chunk_nrows =
                std::min(this->active_chunk_capacity, this->total_remaining);
        }
        this->total_remaining -= chunk_nrows;
        return std::tuple(chunk, chunk_nrows);
    }
    return std::tuple(this->dummy_output_chunk, 0);
}

void AbstractChunkedTableBuilder::Reset() {
    this->ResetInternal();
    for (auto& active_chunk_array_builder : this->active_chunk_array_builders) {
        active_chunk_array_builder.Reset();
    }
    this->active_chunk_size = 0;
    this->total_size = 0;
    this->total_remaining = 0;
    this->max_reached_size = 0;
}

bool AbstractChunkedTableBuilder::empty() const {
    return this->total_remaining == 0;
}

void AbstractChunkedTableBuilder::UnifyDictionariesAndAppend(
    const std::shared_ptr<table_info>& in_table) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (uint64_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info> col = this->dummy_output_chunk->columns[i];
        if (this->dict_builders[i] != nullptr) {
            out_arrs.emplace_back(this->dict_builders[i]->UnifyDictionaryArray(
                in_table->columns[i]));
        } else {
            out_arrs.emplace_back(in_table->columns[i]);
        }
    }
    this->AppendBatch(
        std::make_shared<table_info>(out_arrs, in_table->nrows()));
}
/* ------------------------------------------------------------------------
 */

void ChunkedTableBuilder::PushActiveChunk() {
    // Unpin the chunk and add it to the list of finalized chunks:
    this->active_chunk->unpin();
    this->chunks.emplace_back(std::move(this->active_chunk));
}

size_t ChunkedTableBuilder::NumReadyChunks() { return this->chunks.size(); }

std::shared_ptr<table_info> ChunkedTableBuilder::PopFront() {
    std::shared_ptr<table_info> chunk = this->chunks.front();
    // Pin the chunk before returning it
    chunk->pin();
    this->chunks.pop_front();
    return chunk;
}

void ChunkedTableBuilder::ResetInternal() { this->chunks.clear(); }
