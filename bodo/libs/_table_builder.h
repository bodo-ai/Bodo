#pragma once
#include "_chunked_table_builder.h"
#include "_dict_builder.h"

/**
 * @brief Wrapper around array_info to turn it into build buffer.
 * It allows appending elements while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct ArrayBuildBuffer {
    // Internal array with data values
    std::shared_ptr<array_info> data_array;

    // Current number of elements in the buffer
    const uint64_t& size;
    // Total capacity for data elements (including current elements,
    // capacity>=size should always be true)
    int64_t capacity;

    // Child array builders
    std::vector<ArrayBuildBuffer> child_array_builders;

    // Shared dictionary builder
    std::shared_ptr<DictionaryBuilder> dict_builder;
    // dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_indices;

    /**
     * @brief Construct a new ArrayBuildBuffer for the provided data array.
     *
     * @param _data_array Data array that we will be appending to. This is
     * expected to be an empty array.
     * @param _dict_builder If this is a dictionary encoded string array,
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
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     * @param append_rows bitmask indicating whether to append the row
     * @param append_rows_sum number of rows to append
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
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        uint8_t* out_ptr = (uint8_t*)this->data_array->data1<arr_type>();
        const uint8_t* in_ptr = (uint8_t*)in_arr->data1<arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                arrow::bit_util::SetBitTo(out_ptr, this->size,
                                          GetBit(in_ptr, row_ind));
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
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

        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            sizeof(T) * (size + append_rows_sum)),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        T* out_ptr = (T*)this->data_array->data1<arr_type>();
        const T* in_ptr = (T*)in_arr->data1<arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

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
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
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
        CHECK_ARROW_MEM(data_array->buffers[1]->SetSize(
                            (size + 1 + append_rows_sum) * sizeof(offset_t)),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        offset_t* curr_offsets = (offset_t*)this->data_array->data2<arr_type>();
        offset_t* in_offsets = (offset_t*)in_arr->data2<arr_type>();
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
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        uint64_t character_size = this->size;
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            // TODO If subsequent rows are to be appended, combine the memcpy
            if (append_rows[row_ind]) {
                // copy characters
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                char* out_ptr = this->data_array->data1<arr_type>() +
                                curr_offsets[character_size];
                const char* in_ptr =
                    in_arr->data1<arr_type>() + in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);
                character_size++;
            }
        }

        CHECK_ARROW_MEM(
            data_array->buffers[2]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                // set null bit
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
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
        this->data_array->length += append_rows_sum;
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

        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            sizeof(T) * (size + append_rows_sum)),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        T* out_ptr = (T*)this->data_array->data1<arr_type>();
        const T* in_ptr = (T*)in_arr->data1<arr_type>();
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                out_ptr[this->data_array->length++] = in_ptr[row_ind];
            }
        }
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
        requires(arr_type == bodo_array_type::ARRAY_ITEM)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(sizeof(offset_t) *
                                            (this->size + 1 + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");

        offset_t* out_offsets = (offset_t*)this->data_array->data1<arr_type>();
        const offset_t* in_offsets = (offset_t*)in_arr->data1<arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

        std::vector<bool> inner_array_append_rows(
            in_arr->child_arrays[0]->length, false);
        uint64_t inner_array_append_rows_sum = 0;

        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                out_offsets[size + 1] = out_offsets[size] +
                                        in_offsets[row_ind + 1] -
                                        in_offsets[row_ind];
                inner_array_append_rows_sum +=
                    in_offsets[row_ind + 1] - in_offsets[row_ind];
                SetBitTo(out_bitmask, this->data_array->length++,
                         GetBit(in_bitmask, row_ind));
            }
            for (offset_t i = in_offsets[row_ind]; i < in_offsets[row_ind + 1];
                 i++) {
                inner_array_append_rows[i] = append_rows[row_ind];
            }
        }

        this->child_array_builders.front().ReserveArray(
            in_arr->child_arrays[0], inner_array_append_rows,
            inner_array_append_rows_sum);
        this->child_array_builders.front().UnsafeAppendBatch(
            in_arr->child_arrays[0], inner_array_append_rows,
            inner_array_append_rows_sum);
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
        requires(arr_type == bodo_array_type::MAP)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        this->child_array_builders.front().UnsafeAppendBatch(
            in_arr->child_arrays[0], append_rows, append_rows_sum);
        this->data_array->length += append_rows_sum;
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
        requires(arr_type == bodo_array_type::STRUCT)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");

        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }

        for (size_t i = 0; i < in_arr->child_arrays.size(); ++i) {
            this->child_array_builders[i].ReserveArray(
                in_arr->child_arrays[i], append_rows, append_rows_sum);
            this->child_array_builders[i].UnsafeAppendBatch(
                in_arr->child_arrays[i], append_rows, append_rows_sum);
        }
        // Copy field names if not set
        if (this->data_array->field_names.size() == 0) {
            this->data_array->field_names = in_arr->field_names;
        }
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
        requires(arr_type == bodo_array_type::TIMESTAMPTZ)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum) {
        using T1 = typename dtype_to_type<Bodo_CTypes::INT64>::type;
        using T2 = typename dtype_to_type<Bodo_CTypes::INT16>::type;

        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            sizeof(T1) * (size + append_rows_sum)),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(data_array->buffers[1]->SetSize(
                            sizeof(T2) * (size + append_rows_sum)),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[2]->SetSize(
                arrow::bit_util::BytesForBits(size + append_rows_sum)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        T1* out_ts_ptr = (T1*)this->data_array->data1<arr_type>();
        const T1* in_ts_ptr = (T1*)in_arr->data1<arr_type>();
        T2* out_offset_ptr = (T2*)this->data_array->data2<arr_type>();
        const T2* in_offset_ptr = (T2*)in_arr->data2<arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

        int64_t data_size = this->size;
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                out_ts_ptr[data_size] = in_ts_ptr[row_ind];
                out_offset_ptr[data_size] = in_offset_ptr[row_ind];
                data_size++;
            }
        }
        for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
            if (append_rows[row_ind]) {
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
    }

    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr,
                           const std::vector<bool>& append_rows,
                           uint64_t append_rows_sum);

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
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType == Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        uint8_t* out_ptr = (uint8_t*)this->data_array->data1<arr_type>();
        const uint8_t* in_ptr = (uint8_t*)in_arr->data1<arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();

        // Fast path if our buffer is byte aligned
        if ((this->size & 7) == 0) {
            _copy_bitmap(out_ptr + (this->size >> 3), in_ptr, in_arr->length);
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
            this->data_array->length += in_arr->length;
        } else {
            for (uint64_t row_ind = 0; row_ind < in_arr->length; row_ind++) {
                arrow::bit_util::SetBitTo(out_ptr, this->size,
                                          GetBit(in_ptr, row_ind));
                bool bit = GetBit(in_bitmask, row_ind);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType != Bodo_CTypes::_BOOL)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        uint64_t size_type = numpy_item_size[in_arr->dtype];
        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            (size + in_arr->length) * size_type),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        char* out_ptr = this->data_array->data1<arr_type>() + size_type * size;
        const char* in_ptr = in_arr->data1<arr_type>();
        memcpy(out_ptr, in_ptr, size_type * in_arr->length);

        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
        if ((size & 7) == 0) {
            // Fast path for byte aligned null bitmask
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
            this->data_array->length += in_arr->length;
        } else {
            // Slow path for non-byte aligned null bitmask
            for (uint64_t i = 0; i < in_arr->length; i++) {
                bool bit = GetBit(in_bitmask, i);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::STRING)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        offset_t* curr_offsets = (offset_t*)this->data_array->data2<arr_type>();
        offset_t* in_offsets = (offset_t*)in_arr->data2<arr_type>();
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
        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(new_data_size),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(new_offset_size * sizeof(offset_t)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(data_array->buffers[2]->SetSize(new_bitmap_size),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        // Copy data
        char* out_ptr = this->data_array->data1<arr_type>() + old_data_size;
        const char* in_ptr = in_arr->data1<arr_type>();
        memcpy(out_ptr, in_ptr, added_data_size);

        // Copy offsets
        for (uint64_t row_ind = 1; row_ind < in_arr->length + 1; row_ind++) {
            offset_t offset_val =
                curr_offsets[this->size] + in_offsets[row_ind];
            curr_offsets[size + row_ind] = offset_val;
        }

        // Copy bitmap
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
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
        this->data_array->length += in_arr->length;
    }

    // Needs optimized
    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
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
        this->data_array->length += in_arr->length;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::NUMPY)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        uint64_t size_type = numpy_item_size[in_arr->dtype];
        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            (size + in_arr->length) * size_type),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        char* out_ptr = this->data_array->data1<arr_type>() + size_type * size;
        const char* in_ptr = in_arr->data1<arr_type>();
        memcpy(out_ptr, in_ptr, size_type * in_arr->length);
        this->data_array->length += in_arr->length;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::ARRAY_ITEM)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        offset_t* curr_offsets = (offset_t*)this->data_array->data1<arr_type>();
        offset_t* in_offsets = (offset_t*)in_arr->data1<arr_type>();

        // Reserve space for and append inner array
        this->child_array_builders.front().ReserveArray(
            in_arr->child_arrays[0]);
        this->child_array_builders.front().UnsafeAppendBatch(
            in_arr->child_arrays[0]);

        // Set new buffer sizes. Required space should've been reserved
        // beforehand.
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize((this->size + 1 + in_arr->length) *
                                            sizeof(offset_t)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");
        CHECK_ARROW_MEM(
            data_array->buffers[1]->SetSize(
                arrow::bit_util::BytesForBits(this->size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");

        // Copy offsets
        for (uint64_t row_ind = 1; row_ind <= in_arr->length; row_ind++) {
            offset_t offset_val =
                curr_offsets[this->size] + in_offsets[row_ind];
            curr_offsets[size + row_ind] = offset_val;
        }

        // Copy bitmap
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
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
        this->data_array->length += in_arr->length;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::MAP)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        this->child_array_builders[0].UnsafeAppendBatch(
            in_arr->child_arrays[0]);
        this->data_array->length += in_arr->length;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::STRUCT)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        // Reserve space for and append child arrays
        for (size_t i = 0; i < in_arr->child_arrays.size(); ++i) {
            this->child_array_builders[i].ReserveArray(in_arr->child_arrays[i]);
            this->child_array_builders[i].UnsafeAppendBatch(
                in_arr->child_arrays[i]);
        }

        // Set new buffer sizes. Required space should've been reserved
        // beforehand.
        CHECK_ARROW_MEM(
            data_array->buffers[0]->SetSize(
                arrow::bit_util::BytesForBits(this->size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!:");

        // Copy bitmap
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
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
        // Copy field names if not set
        if (this->data_array->field_names.size() == 0) {
            this->data_array->field_names = in_arr->field_names;
        }
        this->data_array->length += in_arr->length;
    }

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum DType>
        requires(arr_type == bodo_array_type::TIMESTAMPTZ)
    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr) {
        uint64_t ts_size_type = numpy_item_size[Bodo_CTypes::INT64];
        uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
        CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(
                            (size + in_arr->length) * ts_size_type),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(data_array->buffers[1]->SetSize(
                            (size + in_arr->length) * offset_size_type),
                        "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");
        CHECK_ARROW_MEM(
            data_array->buffers[2]->SetSize(
                arrow::bit_util::BytesForBits(size + in_arr->length)),
            "ArrayBuildBuffer::UnsafeAppendBatch: SetSize failed!");

        char* out_ts_ptr =
            this->data_array->data1<arr_type>() + ts_size_type * size;
        const char* in_ts_ptr = in_arr->data1<arr_type>();
        memcpy(out_ts_ptr, in_ts_ptr, ts_size_type * in_arr->length);

        char* out_offset_ptr =
            this->data_array->data2<arr_type>() + offset_size_type * size;
        const char* in_offset_ptr = in_arr->data2<arr_type>();
        memcpy(out_offset_ptr, in_offset_ptr,
               offset_size_type * in_arr->length);

        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<arr_type>();
        const uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<arr_type>();
        if ((size & 7) == 0) {
            // Fast path for byte aligned null bitmask
            _copy_bitmap(out_bitmask + (this->size >> 3), in_bitmask,
                         in_arr->length);
            this->data_array->length += in_arr->length;
        } else {
            // Slow path for non-byte aligned null bitmask
            for (uint64_t i = 0; i < in_arr->length; i++) {
                bool bit = GetBit(in_bitmask, i);
                SetBitTo(out_bitmask, this->data_array->length++, bit);
            }
        }
    }

    void UnsafeAppendBatch(const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Append a new row to the buffer, assuming there is already enough
     * space reserved (with ReserveArrayRow).
     *
     * @tparam arr_type type of the data array
     * @tparam DType data type of the data array
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    void UnsafeAppendRow(const std::shared_ptr<array_info>& in_arr,
                         int64_t row_ind) {
        bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
        switch (arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL: {
                if (dtype == Bodo_CTypes::_BOOL) {
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->SetSize(
                            arrow::bit_util::BytesForBits(size + 1)),
                        "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->SetSize(
                            arrow::bit_util::BytesForBits(size + 1)),
                        "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                    arrow::bit_util::SetBitTo(
                        (uint8_t*)data_array
                            ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                        size,
                        GetBit(
                            (uint8_t*)in_arr
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            row_ind));
                    bool bit = GetBit((uint8_t*)in_arr->null_bitmask<
                                          bodo_array_type::NULLABLE_INT_BOOL>(),
                                      row_ind);
                    SetBitTo((uint8_t*)data_array->null_bitmask<
                                 bodo_array_type::NULLABLE_INT_BOOL>(),
                             size, bit);
                } else {
                    uint64_t size_type = numpy_item_size[in_arr->dtype];
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->SetSize((size + 1) * size_type),
                        "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->SetSize(
                            arrow::bit_util::BytesForBits(size + 1)),
                        "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                    char* out_ptr =
                        data_array
                            ->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                        size_type * size;
                    const char* in_ptr =
                        in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                        size_type * row_ind;
                    memcpy(out_ptr, in_ptr, size_type);
                    bool bit = GetBit((uint8_t*)in_arr->null_bitmask<
                                          bodo_array_type::NULLABLE_INT_BOOL>(),
                                      row_ind);
                    SetBitTo((uint8_t*)data_array->null_bitmask<
                                 bodo_array_type::NULLABLE_INT_BOOL>(),
                             size, bit);
                }
            } break;
            case bodo_array_type::TIMESTAMPTZ: {
                uint64_t utc_size_type =
                    numpy_item_size[Bodo_CTypes::TIMESTAMPTZ];
                uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->SetSize((size + 1) * utc_size_type),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(size + 1) *
                        offset_size_type),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[2]->SetSize(
                        arrow::bit_util::BytesForBits(size + 1)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                char* utc_out_ptr =
                    data_array->data1<bodo_array_type::TIMESTAMPTZ>() +
                    utc_size_type * size;
                const char* utc_in_ptr =
                    in_arr->data1<bodo_array_type::TIMESTAMPTZ>() +
                    utc_size_type * row_ind;
                char* offset_out_ptr =
                    data_array->data2<bodo_array_type::TIMESTAMPTZ>() +
                    offset_size_type * size;
                const char* offset_in_ptr =
                    in_arr->data2<bodo_array_type::TIMESTAMPTZ>() +
                    offset_size_type * row_ind;
                memcpy(utc_out_ptr, utc_in_ptr, utc_size_type);
                memcpy(offset_out_ptr, offset_in_ptr, offset_size_type);
                bool bit = GetBit(
                    (uint8_t*)
                        in_arr->null_bitmask<bodo_array_type::TIMESTAMPTZ>(),
                    row_ind);
                SetBitTo((uint8_t*)data_array
                             ->null_bitmask<bodo_array_type::TIMESTAMPTZ>(),
                         size, bit);
            } break;
            case bodo_array_type::STRING: {
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->SetSize((size + 2) *
                                                    sizeof(offset_t)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[2]->SetSize(
                        arrow::bit_util::BytesForBits(size + 1)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");

                offset_t* curr_offsets =
                    (offset_t*)data_array->data2<bodo_array_type::STRING>();
                offset_t* in_offsets =
                    (offset_t*)in_arr->data2<bodo_array_type::STRING>();

                // append offset
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                curr_offsets[size + 1] = curr_offsets[size] + str_len;

                CHECK_ARROW_MEM(
                    data_array->buffers[0]->SetSize(data_array->n_sub_elems() +
                                                    str_len),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");

                // copy characters
                char* out_ptr = data_array->data1<bodo_array_type::STRING>() +
                                curr_offsets[size];
                const char* in_ptr = in_arr->data1<bodo_array_type::STRING>() +
                                     in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);

                // set null bit
                bool bit = GetBit(
                    (uint8_t*)in_arr->null_bitmask<bodo_array_type::STRING>(),
                    row_ind);
                SetBitTo((uint8_t*)data_array
                             ->null_bitmask<bodo_array_type::STRING>(),
                         size, bit);
            } break;
            case bodo_array_type::DICT: {
                if (!is_matching_dictionary(this->data_array->child_arrays[0],
                                            in_arr->child_arrays[0])) {
                    throw std::runtime_error(
                        "dictionary not unified in AppendRow");
                }
                this->dict_indices->UnsafeAppendRow(in_arr->child_arrays[1],
                                                    row_ind);
            } break;
            case bodo_array_type::NUMPY: {
                uint64_t size_type = numpy_item_size[in_arr->dtype];
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->SetSize((size + 1) * size_type),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                char* out_ptr = data_array->data1<bodo_array_type::NUMPY>() +
                                size_type * size;
                const char* in_ptr = in_arr->data1<bodo_array_type::NUMPY>() +
                                     size_type * row_ind;
                memcpy(out_ptr, in_ptr, size_type);
            } break;
            case bodo_array_type::ARRAY_ITEM: {
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->SetSize((size + 1) *
                                                    sizeof(offset_t)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(size + 1)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");

                // append offset
                offset_t* curr_offsets =
                    (offset_t*)this->data_array
                        ->data1<bodo_array_type::ARRAY_ITEM>();
                offset_t* in_offsets =
                    (offset_t*)in_arr->data1<bodo_array_type::ARRAY_ITEM>();
                curr_offsets[size + 1] = curr_offsets[size] +
                                         in_offsets[row_ind + 1] -
                                         in_offsets[row_ind];

                // append inner array
                for (offset_t i = in_offsets[row_ind];
                     i < in_offsets[row_ind + 1]; i++) {
                    this->child_array_builders.front().ReserveArrayRow(
                        in_arr->child_arrays[0], i);
                    this->child_array_builders.front().UnsafeAppendRow(
                        in_arr->child_arrays[0], i);
                }

                // set null bit
                bool bit = GetBit(
                    (uint8_t*)
                        in_arr->null_bitmask<bodo_array_type::ARRAY_ITEM>(),
                    row_ind);
                SetBitTo((uint8_t*)data_array
                             ->null_bitmask<bodo_array_type::ARRAY_ITEM>(),
                         size, bit);
            } break;
            case bodo_array_type::STRUCT: {
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->SetSize(
                        arrow::bit_util::BytesForBits(size + 1)),
                    "ArrayBuildBuffer::UnsafeAppendRow: SetSize failed!");

                // append child array
                for (size_t i = 0; i < in_arr->child_arrays.size(); ++i) {
                    this->child_array_builders[i].ReserveArrayRow(
                        in_arr->child_arrays[i], row_ind);
                    this->child_array_builders[i].UnsafeAppendRow(
                        in_arr->child_arrays[i], row_ind);
                }

                // set null bit
                bool bit = GetBit(
                    (uint8_t*)in_arr->null_bitmask<bodo_array_type::STRUCT>(),
                    row_ind);
                SetBitTo((uint8_t*)data_array
                             ->null_bitmask<bodo_array_type::STRUCT>(),
                         size, bit);
            } break;
            case bodo_array_type::MAP: {
                this->child_array_builders[0].UnsafeAppendRow(
                    in_arr->child_arrays[0], row_ind);
            } break;
            default:
                throw std::runtime_error(
                    "ArrayBuildBuffer::UnsafeAppendRow: Invalid array type " +
                    GetArrType_as_string(in_arr->arr_type));
        }
        ++this->data_array->length;
    }

    /**
     * @brief Utility function for type check before ReserveArray
     *
     * @param in_table input table used for finding new buffer sizes to reserve
     */
    void ReserveArrayTypeCheck(const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * array to buffer. This requires reserving space for variable-sized
     * elements like strings.
     *
     * NOTE: For semi-structured data array (ARRAY_ITEM, STRUCT and MAP),
     * ReserveArray only reserve space for the buffers and NOT the child arrays.
     * Reserving space for inner array seperately is required before appending.
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
     *
     * @param chunked_tb The ChunkedTableBuilder whose chunks we must copy over.
     * @param array_idx Index of the array to reserve space for.
     */
    void ReserveArray(const ChunkedTableBuilder& chunked_tb,
                      const size_t array_idx);

    /**
     * @brief Reserved enouch space to append in_arr[row_idx] as a row.
     *
     * @param in_arr input array used for finding new buffer sizes to reserve
     * @param row_idx index of input array used for finding new buffer sizes to
     * reserve
     */
    void ReserveArrayRow(const std::shared_ptr<array_info>& in_arr,
                         size_t row_idx);

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
     * @brief Reserve enough space to append additional characters
     * NOTE: This function assumes that this ArrayBuildBuffer is building a
     * string array.
     *
     * @param new_char_count the minimum number of characters that we should
     * reserve space for.
     */
    void ReserveSpaceForStringAppend(size_t new_char_count);

    /**
     * @brief increment the size of the buffer to allow new rows to be
     * appended.
     * NOTE: The array should have enough capactiy before making
     * this call
     */
    void IncrementSize(size_t addln_size = 1);

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
     * @param schema Schema of the table
     * @param dict_builders DictBuilders for the columns.
     * Element corresponding to a column must be provided in the
     * DICT array case and should be nullptr otherwise.
     * @param pool IBufferPool to use for allocating the underlying data
     * buffers.
     * @param mm MemoryManager for the 'pool'.
     */
    TableBuildBuffer(
        const std::shared_ptr<bodo::Schema>& schema,
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
     * @brief increment size of all columns by addln_size.
     * NOTE: The table should have enough capacity already
     * reserved before this call.
     */
    void IncrementSize(size_t new_size);

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
     * @brief Reserve enough space to hold size + new_data_len elements
     */
    void ReserveTableSize(const size_t new_data_len);

    /**
     * @brief Reserve enough space to be able to append all the
     * finalized chunks of a ChunkedTableBuilder.
     * NOTE: This requires reserving space for
     * variable-sized elements like strings and nested arrays.
     *
     * @param chunked_tb ChunkedTableBuilder whose chunks we want to append
     * to this TableBuildBuffer.
     */
    void ReserveTable(const ChunkedTableBuilder& chunked_tb);

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
