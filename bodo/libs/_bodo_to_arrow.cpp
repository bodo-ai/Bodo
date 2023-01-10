// Copyright (C) 2022 Bodo Inc. All rights reserved.

#include "_bodo_to_arrow.h"
#include <cassert>
#include "_array_utils.h"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs)                  \
    if (!(res.status().ok())) {                                \
        std::string err_msg = std::string("Error in arrow ") + \
                              " write: " + msg + " " +         \
                              res.status().ToString();         \
        throw std::runtime_error(err_msg);                     \
    }                                                          \
    lhs = std::move(res).ValueOrDie();

/// Convert Bodo date (year, month, day) from int64 to Arrow date32
static inline int32_t bodo_date64_to_arrow_date32(int64_t date) {
    int64_t year = date >> 32;
    int64_t month = (date >> 16) & 0xFFFF;
    int64_t day = date & 0xFFFF;
    // NOTE that get_days_from_date returns int64 and we are downcasting to
    // int32
    return get_days_from_date(year, month, day);
}

/// Convert Bodo date array (year, month, day elements) to Arrow date32 array
static void CastBodoDateToArrowDate32(const int64_t *input, int64_t length,
                                      int32_t *output) {
    for (int64_t i = 0; i < length; ++i) {
        *output++ = bodo_date64_to_arrow_date32(*input++);
    }
}

/**
 * @brief Create Arrow Chunked Array from Bodo's array_info
 *
 * @param pool Arrow memory pool
 * @param array Bodo array to create Arrow array from
 * @param[out] out Arrow array created from Bodo Array
 * @param convert_timedelta_to_int64 : cast timedelta to int64.
 * This is required for writing to Snowflake.
 * When this is false and a timedelta column is passed, the function will error
 * out.
 * TODO: [BE-4102] Full support for this type (i.e. converting to Arrow
 * 'duration' type).
 * @param tz Timezone to use for Datetime (/timestamp) arrays. Provide an empty
 * string ("") to not specify one. This is primarily required for Iceberg, for
 * which we specify "UTC".
 * @param time_unit Time-Unit (NANO / MICRO / MILLI / SECOND) to use for
 * Datetime (/timestamp) arrays. Bodo arrays store information in nanoseconds.
 * When this is not nanoseconds, the data is converted to the specified type
 * before being copied to the Arrow array. Note that in case it's not
 * nanoseconds, we make a copy of the integer array (array->data1) since we
 * cannot modify the existing array, as it might be used elsewhere. This is
 * primarily required for Iceberg which requires data to be written in
 * microseconds.
 * @param downcast_time_ns_to_us (default False): Is time data required to be
 * written in microseconds? NOTE: this is needed for snowflake write operation.
 * See gen_snowflake_schema comments.
 * @return Arrow DataType of output array
 */
std::shared_ptr<arrow::DataType> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    std::shared_ptr<arrow::Array> *out, bool convert_timedelta_to_int64,
    const std::string &tz, arrow::TimeUnit::type &time_unit, bool copy,
    bool downcast_time_ns_to_us) {
    // Return DataType value
    std::shared_ptr<arrow::DataType> ret_type = nullptr;

    // Allocate null bitmap
    std::shared_ptr<arrow::ResizableBuffer> null_bitmap;
    int64_t null_bytes = arrow::bit_util::BytesForBits(array->length);
    arrow::Result<std::unique_ptr<arrow::ResizableBuffer>> res =
        AllocateResizableBuffer(null_bytes, pool);
    CHECK_ARROW_AND_ASSIGN(res, "AllocateResizableBuffer", null_bitmap);
    // Padding zeroed by AllocateResizableBuffer
    memset(null_bitmap->mutable_data(), 0, static_cast<size_t>(null_bytes));

    int64_t null_count_ = 0;
    if (array->arr_type == bodo_array_type::ARROW) {
        ret_type = array->array->type();
        std::shared_ptr<arrow::ArrayData> typ_data = array->array->data();
        *out = array->array;
    }

    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        array->arr_type == bodo_array_type::STRING) {
        // set arrow bit mask based on bodo bitmask
        for (size_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                null_count_++;
                SetBitTo(null_bitmap->mutable_data(), i, false);
            } else {
                SetBitTo(null_bitmap->mutable_data(), i, true);
            }
        }
        if (array->dtype == Bodo_CTypes::_BOOL) {
            // special case: nullable bool column are bit vectors in Arrow
            ret_type = arrow::boolean();

            int64_t nbytes = ::arrow::bit_util::BytesForBits(array->length);
            std::shared_ptr<::arrow::Buffer> buffer;
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(nbytes, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", buffer);

            int64_t i = 0;
            uint8_t *in_data = (uint8_t *)array->data1;
            const auto generate = [&in_data, &i]() -> bool {
                return in_data[i++] != 0;
            };
            ::arrow::internal::GenerateBitsUnrolled(buffer->mutable_data(), 0,
                                                    array->length, generate);

            auto arr_data =
                arrow::ArrayData::Make(arrow::boolean(), array->length,
                                       {null_bitmap, buffer}, null_count_, 0);
            *out = arrow::MakeArray(arr_data);
        }
    }
    // TODO: Reuse some of this code to enable to_parquet with Categorical
    // Arrays?
    if (array->arr_type == bodo_array_type::DICT) {
        // C++ dictionary arrays in Bodo are dictionary arrays in Arrow.
        // We construct the array using arrow::DictionaryArray::FromArrays
        // https://arrow.apache.org/docs/cpp/api/array.html#_CPPv4N5arrow15DictionaryArray10FromArraysERKNSt10shared_ptrI8DataTypeEERKNSt10shared_ptrI5ArrayEERKNSt10shared_ptrI5ArrayEE

        std::shared_ptr<arrow::Array> dictionary;
        std::shared_ptr<arrow::Array> index_array;

        // Recurse on the dictionary
        auto dict_type = bodo_array_to_arrow(
            pool, array->info1, &dictionary, convert_timedelta_to_int64, tz,
            time_unit, copy, downcast_time_ns_to_us);
        // Recurse on the index array
        auto index_type = bodo_array_to_arrow(
            pool, array->info2, &index_array, convert_timedelta_to_int64, tz,
            time_unit, copy, downcast_time_ns_to_us);

        // Extract the types from the dictionary call.
        // TODO: Can we provide ordered?
        auto type = arrow::dictionary(index_type, dict_type);
        ret_type = type;

        auto result =
            arrow::DictionaryArray::FromArrays(type, index_array, dictionary);
        std::shared_ptr<arrow::Array> dict_array;
        CHECK_ARROW_AND_ASSIGN(result, "arrow::DictionaryArray::FromArrays",
                               dict_array)
        *out = dict_array;
    }

    if (array->arr_type == bodo_array_type::NUMPY ||
        (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
         array->dtype != Bodo_CTypes::_BOOL)) {
        int64_t in_num_bytes;
        std::shared_ptr<arrow::DataType> type;
        arrow::Result<std::shared_ptr<arrow::DataType>> type_res;
        switch (array->dtype) {
            case Bodo_CTypes::INT8:
                in_num_bytes = sizeof(int8_t) * array->length;
                type = arrow::int8();
                break;
            case Bodo_CTypes::UINT8:
                in_num_bytes = sizeof(uint8_t) * array->length;
                type = arrow::uint8();
                break;
            case Bodo_CTypes::INT16:
                in_num_bytes = sizeof(int16_t) * array->length;
                type = arrow::int16();
                break;
            case Bodo_CTypes::UINT16:
                in_num_bytes = sizeof(uint16_t) * array->length;
                type = arrow::uint16();
                break;
            case Bodo_CTypes::INT32:
                in_num_bytes = sizeof(int32_t) * array->length;
                type = arrow::int32();
                break;
            case Bodo_CTypes::UINT32:
                in_num_bytes = sizeof(uint32_t) * array->length;
                type = arrow::uint32();
                break;
            case Bodo_CTypes::INT64:
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::int64();
                break;
            case Bodo_CTypes::UINT64:
                in_num_bytes = sizeof(uint64_t) * array->length;
                type = arrow::uint64();
                break;
            case Bodo_CTypes::FLOAT32:
                in_num_bytes = sizeof(float) * array->length;
                type = arrow::float32();
                break;
            case Bodo_CTypes::FLOAT64:
                in_num_bytes = sizeof(double) * array->length;
                type = arrow::float64();
                break;
            case Bodo_CTypes::DECIMAL:
                in_num_bytes = BYTES_PER_DECIMAL * array->length;
                type_res =
                    arrow::Decimal128Type::Make(array->precision, array->scale);
                CHECK_ARROW_AND_ASSIGN(type_res, "arrow::Decimal128Type::Make",
                                       type);
                break;
            case Bodo_CTypes::DATE:
                // input from Bodo uses int64 for dates
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::date32();
                break;
            case Bodo_CTypes::TIME:
                in_num_bytes = sizeof(int64_t) * array->length;
                switch (array->precision) {
                    case 0:
                        type = arrow::time32(arrow::TimeUnit::SECOND);
                        break;
                    case 3:
                        type = arrow::time32(arrow::TimeUnit::MILLI);
                        break;
                    case 6:
                        type = arrow::time64(arrow::TimeUnit::MICRO);
                        break;
                    case 9:
                        if (downcast_time_ns_to_us)
                            type = arrow::time64(arrow::TimeUnit::MICRO);
                        else
                            type = arrow::time64(arrow::TimeUnit::NANO);
                        break;
                    default:
                        throw std::runtime_error(
                            "Unrecognized precision passed to "
                            "bodo_array_to_arrow." +
                            std::to_string(array->precision));
                }
                break;
            case Bodo_CTypes::TIMEDELTA:
                // Convert timedelta to ns frequency in case of snowflake write
                if (convert_timedelta_to_int64) {
                    in_num_bytes = sizeof(int64_t) * array->length;
                    type = arrow::int64();
                }
                // NOTE: Parquet/Iceberg will raise an error on Python side
                // when _get_numba_typ_from_pa_typ is reached.
                // raise error for any other operation.
                else
                    throw std::runtime_error(
                        "Converting Bodo arrays to Arrow format is currently "
                        "not supported for Timedelta.");
                break;
            case Bodo_CTypes::DATETIME:
                // input from Bodo uses int64 for datetimes (datetime64[ns])
                in_num_bytes = sizeof(int64_t) * array->length;
                if (tz.length() > 0) {
                    type = arrow::timestamp(time_unit, tz);
                } else {
                    type = arrow::timestamp(time_unit);
                }

                // convert Bodo NaT to Arrow null bitmap
                for (size_t i = 0; i < array->length; i++) {
                    if (array->at<int64_t>(i) ==
                        std::numeric_limits<int64_t>::min()) {
                        // if value is NaT (equals
                        // std::numeric_limits<int64_t>::min()) we set it as a
                        // null element in output Arrow array
                        null_count_++;
                        SetBitTo(null_bitmap->mutable_data(), i, false);
                    } else {
                        SetBitTo(null_bitmap->mutable_data(), i, true);
                    }
                }
                break;
            default:
                std::cerr << "Fatal error: invalid dtype found in conversion"
                             " of numeric Bodo array to Arrow"
                          << std::endl;
                exit(1);
        }

        ret_type = type;
        std::shared_ptr<arrow::Buffer> out_buffer;

        if (array->dtype == Bodo_CTypes::DATE) {
            // allocate buffer to store date32 values in Arrow format
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(sizeof(int32_t) * array->length, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);
            CastBodoDateToArrowDate32((int64_t *)array->data1, array->length,
                                      (int32_t *)out_buffer->mutable_data());
        } else if (array->dtype == Bodo_CTypes::TIME &&
                   (array->precision != 9 || downcast_time_ns_to_us)) {
            if (array->precision == 6 ||
                (array->precision == 9 && downcast_time_ns_to_us)) {
                int64_t divide_factor = 1000;
                arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                    AllocateBuffer(in_num_bytes, pool);
                CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);

                int64_t *new_data1 = (int64_t *)out_buffer->mutable_data();
                for (size_t i = 0; i < array->length; i++) {
                    // convert to the specified time unit
                    new_data1[i] = array->at<int64_t>(i) / divide_factor;
                }
            } else {
                int64_t divide_factor =
                    array->precision == 3 ? 1000000 : 1000000000;
                arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                    AllocateBuffer(in_num_bytes / 2, pool);
                CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);

                int32_t *new_data1 = (int32_t *)out_buffer->mutable_data();
                for (size_t i = 0; i < array->length; i++) {
                    // convert to the specified time unit
                    new_data1[i] = array->at<int64_t>(i) / divide_factor;
                }
            }

        } else if (array->dtype == Bodo_CTypes::DATETIME &&
                   time_unit != arrow::TimeUnit::NANO) {
            // For datetime arrays, Bodo stores information in nanoseconds.
            // If the Arrow arrays should store them in a different time unit,
            // we need to convert the nanoseconds into the specified time unit.
            // This is primarily used for Iceberg, which requires data to be
            // written in microseconds.
            int64_t divide_factor;
            switch (time_unit) {
                case arrow::TimeUnit::MICRO:
                    divide_factor = 1000;
                    break;
                case arrow::TimeUnit::MILLI:
                    divide_factor = 1000000;
                    break;
                case arrow::TimeUnit::SECOND:
                    divide_factor = 1000000000;
                    break;
                default:
                    throw std::runtime_error(
                        "Unrecognized time_unit passed to "
                        "bodo_array_to_arrow.");
            };

            // Allocate buffer to store timestamp (int64) values in Arrow
            // format. We cannot reuse Bodo buffers, since we don't want to
            // modify values there as they might be used elsewhere.
            // in_num_bytes in this case is (sizeof(int64_t) * array->length)
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(in_num_bytes, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);

            int64_t *new_data1 = (int64_t *)out_buffer->mutable_data();
            for (size_t i = 0; i < array->length; i++) {
                // convert to the specified time unit
                new_data1[i] = array->at<int64_t>(i) / divide_factor;
            }
        } else {
            // we can use the same input buffer (no need to cast or convert)
            out_buffer = std::make_shared<arrow::Buffer>(
                (uint8_t *)array->data1, in_num_bytes);

            // copy buffers if necessary (resulting buffers will be
            // managed/de-allocated by Arrow)
            // TODO: eliminate the need to copy in callers as much as possible
            if (copy) {
                auto out_buffer_copy_res = arrow::Buffer::Copy(
                    out_buffer, arrow::default_cpu_memory_manager());
                CHECK_ARROW_AND_ASSIGN(out_buffer_copy_res, "Buffer::Copy",
                                       out_buffer);
            }
        }

        auto arr_data = arrow::ArrayData::Make(
            type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
        *out = arrow::MakeArray(arr_data);

    } else if (array->arr_type == bodo_array_type::STRING) {
        std::shared_ptr<arrow::DataType> arrow_type;
        if (array->dtype == Bodo_CTypes::BINARY) {
#if OFFSET_BITWIDTH == 64
            arrow_type = arrow::large_binary();
#else
            arrow_type = arrow::binary();
#endif
        } else {
#if OFFSET_BITWIDTH == 64
            arrow_type = arrow::large_utf8();
#else
            arrow_type = arrow::utf8();
#endif
        }

        ret_type = arrow_type;

        // We use the same input Bodo buffers (no need to copy to new buffers)
        const int64_t n_strings = array->length;
        const int64_t n_chars = ((offset_t *)array->data2)[n_strings];

        std::shared_ptr<arrow::Buffer> chars_buffer =
            std::make_shared<arrow::Buffer>((uint8_t *)array->data1, n_chars);

        std::shared_ptr<arrow::Buffer> offsets_buffer =
            std::make_shared<arrow::Buffer>((uint8_t *)array->data2,
                                            sizeof(offset_t) * (n_strings + 1));

        // copy buffers if necessary (resulting buffers will be
        // managed/de-allocated by Arrow)
        // TODO: eliminate the need to copy in callers as much as possible
        if (copy) {
            auto chars_buff_copy_res = arrow::Buffer::Copy(
                chars_buffer, arrow::default_cpu_memory_manager());
            CHECK_ARROW_AND_ASSIGN(chars_buff_copy_res, "Buffer::Copy",
                                   chars_buffer);

            auto offsets_buff_copy_res = arrow::Buffer::Copy(
                offsets_buffer, arrow::default_cpu_memory_manager());
            CHECK_ARROW_AND_ASSIGN(offsets_buff_copy_res, "Buffer::Copy",
                                   offsets_buffer);
        }

        auto arr_data = arrow::ArrayData::Make(
            arrow_type, n_strings, {null_bitmap, offsets_buffer, chars_buffer},
            null_count_, /*offset=*/0);
        *out = arrow::MakeArray(arr_data);

    } else if (array->arr_type == bodo_array_type::LIST_STRING) {
        // Track statuses of arrow operations.
        arrow::Status arrowOpStatus;

        // TODO: Try to have Arrow reuse Bodo buffers instead of copying to
        // new buffers
        ret_type = arrow::list(arrow::utf8());

        int64_t num_lists = array->length;
        char *chars = (char *)array->data1;
        int64_t *char_offsets = (int64_t *)array->data2;
        int64_t *string_offsets = (int64_t *)array->data3;
        arrow::ListBuilder list_builder(
            pool, std::make_shared<arrow::StringBuilder>(pool));
        arrow::StringBuilder &string_builder = *(
            static_cast<arrow::StringBuilder *>(list_builder.value_builder()));
        bool failed = false;

        for (int64_t i = 0; i < num_lists; i++) {
            bool is_null = !GetBit((uint8_t *)array->null_bitmask, i);
            if (is_null) {
                arrowOpStatus = list_builder.AppendNull();
                failed = failed || !arrowOpStatus.ok();
            } else {
                arrowOpStatus = list_builder.Append();
                failed = failed || !arrowOpStatus.ok();
                int64_t l_string = string_offsets[i];
                int64_t r_string = string_offsets[i + 1];
                for (int64_t j = l_string; j < r_string; j++) {
                    bool is_null =
                        !GetBit((uint8_t *)array->sub_null_bitmask, j);
                    if (is_null) {
                        arrowOpStatus = string_builder.AppendNull();
                    } else {
                        int64_t l_char = char_offsets[j];
                        int64_t r_char = char_offsets[j + 1];
                        int64_t length = r_char - l_char;
                        arrowOpStatus = string_builder.Append(
                            (uint8_t *)(chars + l_char), length);
                    }
                    failed = failed || !arrowOpStatus.ok();
                }
            }
        }

        std::shared_ptr<arrow::Array> result;
        arrowOpStatus = list_builder.Finish(&result);
        failed = failed || !arrowOpStatus.ok();
        if (failed) {
            throw std::runtime_error(
                "Error occured while creating arrow string list array");
        }

        *out = result;
    } else if (array->arr_type == bodo_array_type::CATEGORICAL) {
        // convert Bodo categorical array to corresponding Arrow dictionary
        // array. array_info doesn't store category values right now so we just
        // set dummy values. Category values are not needed in our C++ kernels.
        const size_t siztype = numpy_item_size[array->dtype];
        int64_t in_num_bytes = array->length * siztype;
        std::shared_ptr<arrow::Buffer> out_buffer =
            std::make_shared<arrow::Buffer>((uint8_t *)array->data1,
                                            in_num_bytes);

        // set arrow bit mask using category index values (-1 for nulls)
        int64_t null_count_ = 0;
        for (size_t i = 0; i < array->length; i++) {
            char *ptr = array->data1 + (i * siztype);
            if (isnan_categorical_ptr(array->dtype, ptr)) {
                null_count_++;
                SetBitTo(null_bitmap->mutable_data(), i, false);
            } else {
                SetBitTo(null_bitmap->mutable_data(), i, true);
            }
        }
        std::shared_ptr<arrow::DataType> index_type;
        switch (array->dtype) {
            case Bodo_CTypes::INT8:
                index_type = arrow::int8();
                break;
            case Bodo_CTypes::INT16:
                index_type = arrow::int16();
                break;
            case Bodo_CTypes::INT32:
                index_type = arrow::int32();
                break;
            case Bodo_CTypes::INT64:
                index_type = arrow::int64();
                break;
            default:
                throw std::runtime_error(
                    "bodo_array_to_arrow(): invalid categorical dtype");
        }
        auto arr_data =
            arrow::ArrayData::Make(index_type, array->length,
                                   {null_bitmap, out_buffer}, null_count_, 0);
        std::shared_ptr<arrow::Array> index_array = arrow::MakeArray(arr_data);

        // if the number of categories is unknown and set to 0, use maximum
        // index to find number of categories
        uint64_t n_cats = array->num_categories;
        if (n_cats == 0) {
            uint64_t max_ind;
            switch (array->dtype) {
                case Bodo_CTypes::INT8:
                    max_ind = *std::max_element(
                        (int8_t *)array->data1,
                        ((int8_t *)array->data1) + array->length);
                    break;
                case Bodo_CTypes::INT16:
                    max_ind = *std::max_element(
                        (int16_t *)array->data1,
                        ((int16_t *)array->data1) + array->length);
                    break;
                case Bodo_CTypes::INT32:
                    max_ind = *std::max_element(
                        (int32_t *)array->data1,
                        ((int32_t *)array->data1) + array->length);
                    break;
                case Bodo_CTypes::INT64:
                    max_ind = *std::max_element(
                        (int64_t *)array->data1,
                        ((int64_t *)array->data1) + array->length);
                    break;
                default:
                    throw std::runtime_error(
                        "bodo_array_to_arrow(): invalid categorical dtype");
            }
            n_cats = (uint64_t)(max_ind + 1);
        }

        // create dictionary with dummy values
        std::shared_ptr<arrow::Array> dictionary;
        arrow::Int64Builder builder;
        std::vector<int64_t> values(n_cats);
        std::iota(std::begin(values), std::end(values), 0);
        (void)builder.AppendValues(values);
        (void)builder.Finish(&dictionary);

        auto type = arrow::dictionary(index_array->type(), dictionary->type());

        auto result =
            arrow::DictionaryArray::FromArrays(type, index_array, dictionary);
        std::shared_ptr<arrow::Array> dict_array;
        CHECK_ARROW_AND_ASSIGN(result, "arrow::DictionaryArray::FromArrays",
                               dict_array)
        ret_type = dict_array->type();
        *out = dict_array;
    }

    assert(ret_type != nullptr);
    return ret_type;
}

std::shared_ptr<arrow::Table> bodo_table_to_arrow(table_info *table) {
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    std::vector<std::shared_ptr<arrow::Array>> arrow_arrs(
        table->columns.size());

    for (int i = 0; i < (int)table->ncols(); i++) {
        array_info *arr = table->columns[i];
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        auto arrow_type = bodo_array_to_arrow(
            ::arrow::default_memory_pool(), arr, &arrow_arrs[i],
            false /*convert_timedelta_to_int64*/, "", time_unit, false,
            false /*downcast_time_ns_to_us*/);
        schema_vector.push_back(
            arrow::field("A" + std::to_string(i), arrow_type, true));
    }
    std::shared_ptr<arrow::KeyValueMetadata> schema_metadata;
    std::shared_ptr<arrow::Schema> schema =
        std::make_shared<arrow::Schema>(schema_vector, schema_metadata);
    std::shared_ptr<arrow::Table> arrow_table =
        arrow::Table::Make(schema, arrow_arrs, table->nrows());
    return arrow_table;
}
