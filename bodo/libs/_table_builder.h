#pragma once
#include "_chunked_table_builder.h"
#include "_dict_builder.h"
#include "_table_builder_utils.h"

/**
 * @brief Wrapper around array_info to turn it into build buffer.
 * It allows appending elements while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct ArrayBuildBuffer {
    // internal array with data values
    std::shared_ptr<array_info> data_array;
    // Current number of elements in the buffer
    int64_t size;
    // Total capacity for data elements (including current elements,
    // capacity>=size should always be true)
    int64_t capacity;

    // Shared dictionary builder.
    std::shared_ptr<DictionaryBuilder> dict_builder = nullptr;
    // dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_indices;

    /**
     * @brief Construct a new ArrayBuildBuffer for the provided data array.
     *
     * @param _data_array Data array that we will be appending to. This is
     * expected to be an empty array.
     * @param dict_builder If this is a dictionary encoded string array,
     * a DictBuilder must be provided that will be used as the dictionary.
     * The dictionary of the data_array (_data_array->child_arrays[0]) must
     * be the dictionary in dict_builder (_dict_builder->dict_buff->data_array).
     */
    ArrayBuildBuffer(
        std::shared_ptr<array_info> _data_array,
        std::shared_ptr<DictionaryBuilder> _dict_builder = nullptr);

    size_t EstimatedSize() const;

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     * @param append_row bitmask indicating whether to append the row
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType == Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");

        uint8_t* out_ptr = (uint8_t*)this->data_array->data1();
        const uint8_t* in_ptr = (uint8_t*)in_arr->data1();
        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();

        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                arrow::bit_util::SetBitTo(out_ptr, this->size,
                                          GetBit(in_ptr, row_ind));
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->size, bit);
                this->size++;
            }
        }
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data batch to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data
     * @param append_rows bitmask indicating whether to append the row
     * @param append_rows_sum number of rows to append
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType != Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        using T = typename dtype_to_type<DType>::type;

        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(sizeof(T) *
                                            (size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");

        T* out_ptr = (T*)this->data_array->data1();
        const T* in_ptr = (T*)in_arr->data1();
        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();

        int64_t data_size = this->size;
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                out_ptr[data_size] = in_ptr[row_ind];
                data_size++;
            }
        }
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->size, bit);
                this->size++;
            }
        }
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data batch to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data
     * @param append_rows bitmask indicating whether to append the row
     * @param append_rows_sum number of rows to append
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::STRING)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        // Set size and copy offsets
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize((size + 1 + append_rows_sum) *
                                            sizeof(offset_t)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        offset_t* curr_offsets = (offset_t*)this->data_array->data2();
        offset_t* in_offsets = (offset_t*)in_arr->data2();
        uint64_t offset_size = this->size;
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                // append offset
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                curr_offsets[offset_size + 1] =
                    curr_offsets[offset_size] + str_len;
                offset_size++;
            }
        }

        // Set size and copy characters
        CHECK_ARROW_MEM(
            // data_array->n_sub_elems() is correct because we set offsets above
            // and n_sub_elems is based on the offsets array
            data_array->buffers[0]->SetSize(this->data_array->n_sub_elems()),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        uint64_t character_size = this->size;
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            // TODO If subsequent rows are to be appended, combine the memcpy
            if (append_rows[row_ind]) {
                // copy characters
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                char* out_ptr =
                    this->data_array->data1() + curr_offsets[character_size];
                const char* in_ptr = in_arr->data1() + in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);
                character_size++;
            }
        }

        CHECK_ARROW_MEM(
            data_array->buffers[2]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                // set null bit
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->size, bit);
                this->size++;
            }
        }
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data batch to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data
     * @param append_rows bitmask indicating whether to append the row
     * @param append_rows_sum number of rows to append
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::DICT)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        if (!is_matching_dictionary(this->data_array->child_arrays[0],
                                    in_arr->child_arrays[0])) {
            throw std::runtime_error(
                "dictionary not unified in UnsafeAppendBatch");
        }

        this->dict_indices->UnsafeAppendBatch<
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32>(
            in_arr->child_arrays[1], append_rows, append_rows_sum);
        this->size += append_rows_sum;
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data batch to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data
     * @param append_rows bitmask indicating whether to append the row
     * @param append_rows_sum number of rows to append
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NUMPY)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        using T = typename dtype_to_type<DType>::type;

        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(sizeof(T) *
                                            (size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        T* out_ptr = (T*)this->data_array->data1();
        const T* in_ptr = (T*)in_arr->data1();
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                out_ptr[size] = in_ptr[row_ind];
                size++;
            }
        }
        this->data_array->length = size;
    }

    /**
     * @brief Copy a bitmap from src to dest with length bits.
     */
    void _copy_bitmap(uint8_t* dest, const uint8_t* src, uint64_t length) {
        uint64_t bytes_to_copy = (length + 7) >> 3;
        memcpy(dest, src, bytes_to_copy);
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType == Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");

        uint8_t* out_ptr = (uint8_t*)this->data_array->data1();
        const uint8_t* in_ptr = (uint8_t*)in_arr->data1();
        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();

        // Fast path if our buffer is byte aligned
        if ((this->size & 7) == 0) {
            _copy_bitmap(out_ptr + (this->size >> 3), in_ptr, in_arr->length);
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
            this->size += in_arr->length;
        } else {
            for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
                arrow::bit_util::SetBitTo(out_ptr, this->size,
                                          GetBit(in_ptr, row_ind));
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->size, bit);
                this->size++;
            }
        }
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType != Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        uint64_t size_type = numpy_item_size[in_arr->dtype];
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize((size + in_arr->length) *
                                            size_type),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");

        char* out_ptr = this->data_array->data1() + size_type * size;
        const char* in_ptr = in_arr->data1();
        memcpy(out_ptr, in_ptr, size_type * in_arr->length);

        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();
        if ((size & 7) == 0) {
            // Fast path for byte aligned null bitmask
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
            this->size += in_arr->length;
        } else {
            // Slow path for non-byte aligned null bitmask
            for (uint64_t i = 0; i < in_arr->length; i++) {
                bool bit = GetBit(in_bitmask, i);
                SetBitTo(out_bitmask, this->size, bit);
                this->size++;
            }
        }
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::STRING)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        offset_t* curr_offsets = (offset_t*)this->data_array->data2();
        offset_t* in_offsets = (offset_t*)in_arr->data2();
        // Determine the new data sizes
        offset_t added_data_size = in_offsets[in_arr->length];
        offset_t old_data_size = curr_offsets[this->size];
        offset_t new_data_size = old_data_size + added_data_size;
        // Determine the new offset size
        size_t new_offset_size = (this->size + 1) + in_arr->length;
        // Determine the new bitmap size
        size_t new_bitmap_size =
            arrow::bit_util::BytesForBits(this->size + in_arr->length);

        // Set new buffer sizes. Required space should've been reserved
        // beforehand.
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(new_data_size),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(new_offset_size * sizeof(offset_t)),
            "ArrayBuildBuffer::UnsafeAppendBatch1: SetSize Failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[2]->SetSize(new_bitmap_size),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");

        // Copy data
        char* out_ptr = this->data_array->data1() + old_data_size;
        const char* in_ptr = in_arr->data1();
        memcpy(out_ptr, in_ptr, added_data_size);

        // Copy offsets
        for (uint64_t row_ind = 1; row_ind < in_arr->length + 1; row_ind++) {
            offset_t offset_val =
                curr_offsets[this->size] + in_offsets[row_ind];
            curr_offsets[size + row_ind] = offset_val;
        }

        // Copy bitmap
        uint8_t* out_bitmask = (uint8_t*)this->data_array->null_bitmask();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask();
        if ((size & 7) == 0) {
            // Fast path for byte aligned null bitmask
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
        } else {
            // Slow path for non-byte aligned null bitmask
            for (uint64_t i = 0; i < in_arr->length; i++) {
                bool bit = GetBit(in_bitmask, i);
                SetBitTo(out_bitmask, this->size + i, bit);
            }
        }
        this->size += in_arr->length;
        this->data_array->length = this->size;
    }

    // Needs optimized
    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::DICT)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        if (!is_matching_dictionary(this->data_array->child_arrays[0],
                                    in_arr->child_arrays[0])) {
            throw std::runtime_error(
                "dictionary not unified in UnsafeAppendBatch");
        }
        this->dict_indices->UnsafeAppendBatch<
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32>(
            in_arr->child_arrays[1]);
        // Update the size + length which won't be handled by the recursive
        // case.
        this->size += in_arr->length;
        this->data_array->length = this->size;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam is_bool whether the data array is boolean (only used
     *  if arr_type is NULLABLE_INT_BOOL)
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NUMPY)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        uint64_t size_type = numpy_item_size[in_arr->dtype];
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize((size + in_arr->length) *
                                            size_type),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize Failed!:");
        char* out_ptr = this->data_array->data1() + size_type * size;
        const char* in_ptr = in_arr->data1();
        memcpy(out_ptr, in_ptr, size_type * in_arr->length);
        this->size += in_arr->length;
        this->data_array->length = this->size;
    }

    void AppendRow(const std::shared_ptr<array_info>& in_arr, int64_t row_ind) {
        bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
        switch (arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL: {
                if (dtype == Bodo_CTypes::_BOOL) {
                    arrow::bit_util::SetBitTo(
                        (uint8_t*)data_array->data1(), size,
                        GetBit((uint8_t*)in_arr->data1(), row_ind));
                    bool bit =
                        GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                    SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);
                    size++;
                    data_array->length = size;
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                } else {
                    uint64_t size_type = numpy_item_size[in_arr->dtype];
                    char* out_ptr = data_array->data1() + size_type * size;
                    const char* in_ptr = in_arr->data1() + size_type * row_ind;
                    memcpy(out_ptr, in_ptr, size_type);
                    bool bit =
                        GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                    SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);
                    size++;
                    data_array->length = size;
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->Resize(size * size_type, false),
                        "Resize Failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                }
            } break;
            case bodo_array_type::STRING: {
                offset_t* curr_offsets = (offset_t*)data_array->data2();
                offset_t* in_offsets = (offset_t*)in_arr->data2();

                // append offset
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                curr_offsets[size + 1] = curr_offsets[size] + str_len;

                // copy characters
                char* out_ptr = data_array->data1() + curr_offsets[size];
                const char* in_ptr = in_arr->data1() + in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);

                // set null bit
                bool bit = GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);

                // update size state
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(
                                    data_array->n_sub_elems(), false),
                                "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                    (size + 1) * sizeof(offset_t), false),
                                "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[2]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
            } break;
            case bodo_array_type::DICT: {
                if (!is_matching_dictionary(this->data_array->child_arrays[0],
                                            in_arr->child_arrays[0])) {
                    throw std::runtime_error(
                        "dictionary not unified in AppendRow");
                }
                this->dict_indices->AppendRow(in_arr->child_arrays[1], row_ind);
                size++;
                data_array->length = size;
            } break;
            case bodo_array_type::NUMPY: {
                uint64_t size_type = numpy_item_size[in_arr->dtype];
                char* out_ptr = data_array->data1() + size_type * size;
                const char* in_ptr = in_arr->data1() + size_type * row_ind;
                memcpy(out_ptr, in_ptr, size_type);
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(size * size_type, false),
                    "Resize Failed!");
            } break;
            default:
                throw std::runtime_error(
                    "invalid array type in AppendRow " +
                    GetArrType_as_string(in_arr->arr_type));
        }
    }

    /**
     * @brief Utility function for type check before ReserveArray
     *
     * @param in_table input table used for finding new buffer sizes to reserve
     */
    void ReserveArrayTypeCheck(const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * array to buffer. NOTE: This requires reserving space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param in_arr input array used for finding new buffer sizes to reserve
     * @param reserve_rows bitmask indicating whether to reserve the row
     * @param reserve_rows_sum number of rows to reserve
     */
    void ReserveArray(const std::shared_ptr<array_info>& in_arr,
                      const std::vector<bool>& reserve_rows,
                      uint64_t reserve_rows_sum);

    void ReserveArray(const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Reserve enough space to be able to append the selected column
     * of all finalized chunks of a ChunkedTableBuilder.
     * NOTE: This requires reserving space for variable-sized elements like
     * strings and nested arrays.
     *
     * @param chunked_tb The ChunkedTableBuilder whose chunks we must copy over.
     * @param array_idx Index of the array to reserve space for.
     */
    void ReserveArray(const ChunkedTableBuilder& chunked_tb,
                      const size_t array_idx);

    /**
     * @brief Reserve enough space to potentially append new_data_len new rows
     * to buffer.
     * NOTE: This does not reserve space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param new_data_len number of new rows that need reserved
     */
    void ReserveSize(uint64_t new_data_len);

    /**
     * @brief increment the size of the buffer by one to allow a new row to be
     * appended.
     * NOTE: This does not resize data buffers of variable-sized
     * elements like strings and nested arrays.
     */
    void IncrementSize();

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionary state is also reset.
     * In particular, the reset to point to the dictionary of the original
     * dictionary-builder which was provided during creation and the
     * dictionary related flags are reset.
     */
    void Reset();
};

/**
 * @brief Wrapper around table_info to turn it into build buffer.
 * It allows appending rows while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct TableBuildBuffer {
    // internal data table with values
    std::shared_ptr<table_info> data_table;
    // buffer wrappers around arrays of data table
    std::vector<ArrayBuildBuffer> array_buffers;
    /// @brief Whether the table is currently pinned.
    bool pinned_ = true;

    // Only used for temporary objects. In particular,
    // in HashJoinState constructor, we cannot initialize
    // the shuffle buffers in the initialization list since
    // we need to build the dict_builders first. So we need
    // to provide this default constructor so that it
    // is initialized to an empty buffer by default and then
    // we can create and replace it with the actual TableBuildBuffer
    // later in the constructor.
    TableBuildBuffer() = default;

    /**
     * @brief Constructor for a TableBuildBuffer.
     *
     * @param arr_c_types Data Types for the columns.
     * @param arr_array_types Array Types for the columns.
     * @param dict_builders DictBuilders for the columns.
     * Element corresponding to a column must be provided in the
     * DICT array case and should be nullptr otherwise.
     * @param pool IBufferPool to use for allocating the underlying data
     * buffers.
     * @param mm MemoryManager for the 'pool'.
     */
    TableBuildBuffer(
        const std::vector<int8_t>& arr_c_types,
        const std::vector<int8_t>& arr_array_types,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    size_t EstimatedSize() const;

    void UnifyTablesAndAppend(
        const std::shared_ptr<table_info>& in_table,
        std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders);

    /**
     * @brief Append a batch of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table Input table with the new rows
     * @param append_rows Bit vector indicating which rows to append
     * @param append_rows_sum Total number of rows to append. This is just the
     * number of 'true' values in append_rows.
     */
    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum);
    /**
     * @brief Append a batch of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new rows
     * @param append_rows bit vector indicating which rows to append
     */
    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table,
                           const std::vector<bool>& append_rows);

    void UnsafeAppendBatch(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append key columns of a row of input table, assuming there is
     * already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new row
     * @param row_ind index of new row in input table
     * @param n_keys number of key columns
     */
    void AppendRowKeys(const std::shared_ptr<table_info>& in_table,
                       int64_t row_ind, uint64_t n_keys);

    /**
     * @brief increment size of data columns by one to allow appending a new
     * data row
     *
     * @param n_keys number of key columns
     */
    void IncrementSizeDataColumns(uint64_t n_keys);

    /**
     * @brief Reserve enough space to potentially append rows from
     * in_table (based on reserve_rows bitmap).
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to
     * reserve
     * @param reserve_rows bit vector indicating which rows to reserve
     * @param reserve_rows_sum Total number of rows. This is just the number of
     * 'true' values in reserve_rows.
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table,
                      const std::vector<bool>& reserve_rows,
                      uint64_t reserve_rows_sum);

    /**
     * @brief Reserve enough space to potentially append all columns of
     * input table to buffer (the rows specified using the 'reserve_rows'
     * bitmap).
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to
     * reserve
     * @param reserve_rows bit vector indicating which rows to reserve
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table,
                      const std::vector<bool>& reserve_rows);

    void ReserveTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Reserve enough space to be able to append all the
     * finalized chunks of a ChunkedTableBuilder.
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param chunked_tb ChunkedTableBuilder whose chunks we want to append to
     * this TableBuildBuffer.
     */
    void ReserveTable(const ChunkedTableBuilder& chunked_tb);

    /**
     * @brief Same as ReserveTable but reserves space only for key columns
     *
     * @param in_table input table used for finding new buffer sizes to reserve
     * @param n_keys number of keys
     */
    void ReserveTableKeys(const std::shared_ptr<table_info>& in_table,
                          uint64_t n_keys);

    /**
     * @brief Reserve enough space to potentially append new_data_len new rows
     * to data columns.
     * NOTE: This does not reserve space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param new_data_len number of new rows that need reserved
     * @param n_keys number of keys
     */
    void ReserveSizeDataColumns(uint64_t new_data_len, uint64_t n_keys);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionaries are reset to
     * use the original dictionary builders provided during
     * creation.
     */
    void Reset();

    /**
     * @brief Pin this table buffer. This is idempotent.
     * Currently, this simply calls 'pin' on the underlying
     * table_info.
     *
     */
    void pin();

    /**
     * @brief Unpin this table buffer. This is idempotent.
     * Currently, this simply calls 'unpin' on the underlying
     * table_info.
     *
     */
    void unpin();
};
