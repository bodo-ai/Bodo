// Copyright (C) 2022 Bodo Inc. All rights reserved.
#include "_bodo_to_arrow.h"

#include <cassert>
#include <iostream>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/compute/cast.h>
#include <arrow/table.h>

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_datetime_utils.h"

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
 * nanoseconds, we make a copy of the integer array (array->data1()) since we
 * cannot modify the existing array, as it might be used elsewhere. This is
 * primarily required for Iceberg which requires data to be written in
 * microseconds.
 * @param downcast_time_ns_to_us (default False): Is time data required to be
 * written in microseconds? NOTE: this is needed for snowflake write operation.
 * See gen_snowflake_schema comments.
 * @return Arrow DataType of output array
 */
std::shared_ptr<arrow::DataType> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const std::shared_ptr<array_info> array,
    std::shared_ptr<arrow::Array> *out, bool convert_timedelta_to_int64,
    const std::string &tz, arrow::TimeUnit::type &time_unit,
    bool downcast_time_ns_to_us) {
    // Return DataType value
    std::shared_ptr<arrow::DataType> ret_type =
        std::shared_ptr<arrow::DataType>(nullptr);

    // Allocate null bitmap
    // TODO: Switch to 0 copy
    std::shared_ptr<arrow::Buffer> null_bitmap;
    int64_t null_count_ = 0;
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        array->arr_type == bodo_array_type::STRING ||
        array->arr_type == bodo_array_type::ARRAY_ITEM ||
        array->arr_type == bodo_array_type::MAP ||
        array->arr_type == bodo_array_type::STRUCT ||
        array->arr_type == bodo_array_type::TIMESTAMPTZ) {
        if (array->arr_type == bodo_array_type::STRING ||
            array->arr_type == bodo_array_type::TIMESTAMPTZ) {
            null_bitmap = array->buffers[2];
        } else if (array->arr_type == bodo_array_type::STRUCT) {
            null_bitmap = array->buffers[0];
        } else if (array->arr_type == bodo_array_type::MAP) {
            null_bitmap = array->child_arrays[0]->buffers[1];
        } else {
            null_bitmap = array->buffers[1];
        }
        const uint8_t *null_bitmask = (uint8_t *)array->null_bitmask();
        for (size_t i = 0; i < array->length; i++) {
            null_count_ += !GetBit(null_bitmask, i);
        }
    } else {
        // TODO: Remove for arrays that don't need it
        int64_t null_bytes = arrow::bit_util::BytesForBits(array->length);
        arrow::Result<std::unique_ptr<arrow::Buffer>> res =
            AllocateBuffer(null_bytes, pool);
        CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", null_bitmap);
        // Padding zeroed by AllocateBuffer
        memset(null_bitmap->mutable_data(), 0xFF,
               static_cast<size_t>(null_bytes));
    }
    if (array->arr_type == bodo_array_type::STRUCT) {
        std::vector<std::shared_ptr<arrow::Array>> child_arrays;
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (size_t i = 0; i < array->child_arrays.size(); i++) {
            std::shared_ptr<array_info> child_arr = array->child_arrays[i];
            std::string field_name = array->field_names.empty()
                                         ? std::string()
                                         : array->field_names[i];
            std::shared_ptr<arrow::Array> inner_array;
            std::shared_ptr<arrow::DataType> inner_type = bodo_array_to_arrow(
                pool, child_arr, &inner_array, convert_timedelta_to_int64, tz,
                time_unit, downcast_time_ns_to_us);
            child_arrays.push_back(inner_array);
            // TODO [BE-3247] We should specify nullability of the fields here.
            fields.push_back(
                std::make_shared<arrow::Field>(field_name, inner_type));
        }

        std::shared_ptr<arrow::StructType> struct_type =
            std::make_shared<arrow::StructType>(fields);

        std::shared_ptr<arrow::Array> arrow_array =
            std::make_shared<arrow::StructArray>(struct_type, array->length,
                                                 child_arrays, null_bitmap);
        ret_type = arrow_array->type();
        *out = arrow_array;
    }

    if (array->arr_type == bodo_array_type::ARRAY_ITEM) {
        // wrap offset buffer
        std::shared_ptr<BodoBuffer> offsets_buffer = array->buffers[0];

        // convert inner array
        std::shared_ptr<arrow::Array> inner_array;
        std::shared_ptr<arrow::DataType> inner_type = bodo_array_to_arrow(
            pool, array->child_arrays[0], &inner_array,
            convert_timedelta_to_int64, tz, time_unit, downcast_time_ns_to_us);

        // We use `element` for consistency.
        // TODO [BE-3247] We should specify nullability of the fields here.
        std::shared_ptr<arrow::Field> field =
            std::make_shared<arrow::Field>("element", inner_type);
        static_assert(OFFSET_BITWIDTH == 64);

        *out = std::make_shared<arrow::LargeListArray>(
            arrow::large_list(field), array->length, offsets_buffer,
            inner_array, null_bitmap);
        return (*out)->type();
    }

    if (array->arr_type == bodo_array_type::MAP) {
        std::shared_ptr<array_info> array_item_arr = array->child_arrays[0];
        std::shared_ptr<array_info> struct_arr =
            array_item_arr->child_arrays[0];

        // Convert offset buffer to 32-bits as required by Arrow MapArray
        std::shared_ptr<BodoBuffer> offsets_buffer = array_item_arr->buffers[0];
        offset_t *offsets_ptr = (offset_t *)offsets_buffer->mutable_data();
        std::unique_ptr<BodoBuffer> offsets_buffer_32 = AllocateBodoBuffer(
            array_item_arr->length + 1, Bodo_CTypes::CTypeEnum::UINT32);
        uint32_t *offsets_ptr_32 =
            (uint32_t *)offsets_buffer_32->mutable_data();
        for (size_t i = 0; i < (array_item_arr->length + 1); i++) {
            if (offsets_ptr[i] > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(
                    "bodo_array_to_arrow: Map array offset too large to "
                    "convert to 32-bit Arrow offset: " +
                    std::to_string(offsets_ptr[i]));
            }
            offsets_ptr_32[i] = static_cast<uint32_t>(offsets_ptr[i]);
        }

        // Convert key array
        std::shared_ptr<arrow::Array> key_array;
        std::shared_ptr<arrow::DataType> key_type = bodo_array_to_arrow(
            pool, struct_arr->child_arrays[0], &key_array,
            convert_timedelta_to_int64, tz, time_unit, downcast_time_ns_to_us);

        // Convert item array
        std::shared_ptr<arrow::Array> item_array;
        std::shared_ptr<arrow::DataType> item_type = bodo_array_to_arrow(
            pool, struct_arr->child_arrays[1], &item_array,
            convert_timedelta_to_int64, tz, time_unit, downcast_time_ns_to_us);

        *out = std::make_shared<arrow::MapArray>(
            arrow::map(key_type, item_type, false), array_item_arr->length,
            std::move(offsets_buffer_32), key_array, item_array, null_bitmap);
        return (*out)->type();
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
            pool, array->child_arrays[0], &dictionary,
            convert_timedelta_to_int64, tz, time_unit, downcast_time_ns_to_us);
        // Recurse on the index array
        auto index_type = bodo_array_to_arrow(
            pool, array->child_arrays[1], &index_array,
            convert_timedelta_to_int64, tz, time_unit, downcast_time_ns_to_us);

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

    if (array->arr_type == bodo_array_type::NUMPY &&
        array->dtype == Bodo_CTypes::_BOOL) {
        // Special case: Numpy boolean cannot do 0 copy.
        ret_type = arrow::boolean();

        int64_t nbytes = ::arrow::bit_util::BytesForBits(array->length);

        std::shared_ptr<::arrow::Buffer> buffer;
        arrow::Result<std::unique_ptr<arrow::Buffer>> res =
            AllocateBuffer(nbytes, pool);
        CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", buffer);

        bool *in_data = (bool *)array->data1();
        for (size_t i = 0; i < array->length; i++) {
            bool bit = in_data[i];
            SetBitTo(buffer->mutable_data(), i, bit);
        }

        auto arr_data =
            arrow::ArrayData::Make(arrow::boolean(), array->length,
                                   {null_bitmap, buffer}, null_count_, 0);
        *out = arrow::MakeArray(arr_data);
    }

    if ((array->arr_type == bodo_array_type::NUMPY &&
         array->dtype != Bodo_CTypes::_BOOL) ||
        array->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        int64_t in_num_bytes;
        std::shared_ptr<arrow::DataType> type;
        arrow::Result<std::shared_ptr<arrow::DataType>> type_res;
        switch (array->dtype) {
            case Bodo_CTypes::_BOOL:
                // Boolean arrays store 1 bit per boolean value.
                in_num_bytes = (array->length + 7) >> 3;
                type = arrow::boolean();
                break;
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
                in_num_bytes = sizeof(int32_t) * array->length;
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
                    if ((array->at<int64_t>(i) ==
                         std::numeric_limits<int64_t>::min()) &&
                        GetBit(null_bitmap->mutable_data(), i)) {
                        // if value is NaT (equals
                        // std::numeric_limits<int64_t>::min()) we set it as a
                        // null element in output Arrow array
                        null_count_++;
                        SetBitTo(null_bitmap->mutable_data(), i, false);
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

        if (array->dtype == Bodo_CTypes::TIME &&
            (array->precision != 9 || downcast_time_ns_to_us)) {
            std::shared_ptr<arrow::Buffer> out_buffer;
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

            auto arr_data = arrow::ArrayData::Make(
                type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
            *out = arrow::MakeArray(arr_data);

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
            std::shared_ptr<arrow::Buffer> out_buffer;
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(in_num_bytes, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);

            int64_t *new_data1 = (int64_t *)out_buffer->mutable_data();
            for (size_t i = 0; i < array->length; i++) {
                // convert to the specified time unit
                new_data1[i] = array->at<int64_t>(i) / divide_factor;
            }

            auto arr_data = arrow::ArrayData::Make(
                type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
            *out = arrow::MakeArray(arr_data);

        } else {
            // Can't reuse the same BodoBuffer because Bodo arrays have
            // data offsets due to slicing, which is applied in data1().
            // Can't pass the offset directly to Arrow since Bodo only applies
            // the offset to the data array, not null bitmap:
            // https://github.com/Bodo-inc/Bodo/blob/338f8ea3c11016bd560e5158b1ec0abf732856ed/bodo/utils/indexing.py
            std::shared_ptr<BodoBuffer> out_buffer =
                std::make_shared<BodoBuffer>((uint8_t *)array->data1(),
                                             in_num_bytes,
                                             array->buffers[0]->getMeminfo());

            auto arr_data = arrow::ArrayData::Make(
                type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
            *out = arrow::MakeArray(arr_data);
        }
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

        // We use the same input Bodo buffers wrapped in BodoBuffers, which
        // track refcounts and deallocate if necessary.
        const int64_t n_strings = array->length;

        // get buffers of characters and offsets arrays to wrap in BodoBuffers.
        std::shared_ptr<BodoBuffer> chars_buffer = array->buffers[0];
        std::shared_ptr<BodoBuffer> offsets_buffer = array->buffers[1];

        auto arr_data = arrow::ArrayData::Make(
            arrow_type, n_strings, {null_bitmap, offsets_buffer, chars_buffer},
            null_count_, /*offset=*/0);
        *out = arrow::MakeArray(arr_data);

    } else if (array->arr_type == bodo_array_type::CATEGORICAL) {
        // convert Bodo categorical array to corresponding Arrow dictionary
        // array. array_info doesn't store category values right now so we just
        // set dummy values. Category values are not needed in our C++ kernels.
        const size_t siztype = numpy_item_size[array->dtype];
        int64_t in_num_bytes = array->length * siztype;
        std::shared_ptr<arrow::Buffer> out_buffer =
            std::make_shared<arrow::Buffer>(
                (uint8_t *)array->data1(), in_num_bytes,
                bodo::default_buffer_memory_manager());

        // set arrow bit mask using category index values (-1 for nulls)
        int64_t null_count_ = 0;
        for (size_t i = 0; i < array->length; i++) {
            char *ptr = array->data1() + (i * siztype);
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
                        (int8_t *)array->data1(),
                        ((int8_t *)array->data1()) + array->length);
                    break;
                case Bodo_CTypes::INT16:
                    max_ind = *std::max_element(
                        (int16_t *)array->data1(),
                        ((int16_t *)array->data1()) + array->length);
                    break;
                case Bodo_CTypes::INT32:
                    max_ind = *std::max_element(
                        (int32_t *)array->data1(),
                        ((int32_t *)array->data1()) + array->length);
                    break;
                case Bodo_CTypes::INT64:
                    max_ind = *std::max_element(
                        (int64_t *)array->data1(),
                        ((int64_t *)array->data1()) + array->length);
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
    } else if (array->arr_type == bodo_array_type::TIMESTAMPTZ) {
        // Convert Bodo TimestampTZ array to Arrow String array
        std::shared_ptr<arrow::DataType> arrow_type = arrow::utf8();
        auto builder = arrow::StringBuilder();
        auto ts_buffer = (int64_t *)array->data1();
        auto tz_buffer = (int16_t *)array->data2();

        // Allocate buffer for timestamp string - we know that the timestamp
        // string can only ever be 38 characters, but the compiler we use in CI
        // throws an error because it doesn't correctly parse that the format
        // string restricts the number of characters for each field, so we use a
        // larger buffer.
        char ts_str[128];

        // convert Bodo TimestampTZ to Arrow String
        for (size_t i = 0; i < array->length; i++) {
            if (!GetBit(null_bitmap->mutable_data(), i)) {
                auto res = builder.AppendNull();
                if (!res.ok()) [[unlikely]] {
                    throw std::runtime_error(
                        "bodo_array_to_arrow(): failed to append null to "
                        "StringBuilder");
                }
            } else {
                auto ts = ts_buffer[i];
                auto offset = tz_buffer[i];

                auto offset_sign = offset < 0 ? '-' : '+';
                auto abs_offset = std::abs(offset);
                auto offset_hrs = abs_offset / 60;
                auto offset_mins = abs_offset % 60;

                // We need to add the offset here to convert UTC to the local
                // timestamp
                time_t seconds = ts / 1000000000 + offset * 60;
                size_t ns = ts % 1000000000;
                struct tm *ptm;
                ptm = gmtime(&seconds);

                snprintf(ts_str, 128,
                         "%04d-%02d-%02d %02d:%02d:%02d.%09zu %c%02d:%02d",
                         ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
                         ptm->tm_hour, ptm->tm_min, ptm->tm_sec, ns,
                         offset_sign, offset_hrs, offset_mins);

                auto res = builder.Append(ts_str);
                if (!res.ok()) [[unlikely]] {
                    throw std::runtime_error(
                        "bodo_array_to_arrow(): failed to append string to "
                        "StringBuilder");
                }
            }
        }

        *out = builder.Finish().ValueOrDie();
        ret_type = arrow_type;
    }

    if (!ret_type) {
        throw std::runtime_error(
            "bodo_array_to_arrow(): unexpected null return type");
    }
    return ret_type;
}

std::shared_ptr<arrow::Table> bodo_table_to_arrow(
    std::shared_ptr<table_info> table) {
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    std::vector<std::shared_ptr<arrow::Array>> arrow_arrs(
        table->columns.size());

    for (int i = 0; i < (int)table->ncols(); i++) {
        std::shared_ptr<array_info> arr = table->columns[i];
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        auto arrow_type = bodo_array_to_arrow(
            bodo::BufferPool::DefaultPtr(), arr, &arrow_arrs[i],
            false /*convert_timedelta_to_int64*/, "", time_unit,
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

extern "C" {

// meminfo destructor function that releases the meminfo's Arrow buffer
// by deleting a pointer to buffer shared_ptr that holds a reference.
void arrow_buffer_dtor(void *ptr, size_t size, void *info) {
    std::shared_ptr<arrow::Buffer> *arrow_buf =
        (std::shared_ptr<arrow::Buffer> *)info;
    delete arrow_buf;
}

}  // extern "C"

/**
 * @brief Helper function to pass an Arrow buffer to Bodo with zero-copy
 * @param buf Shared pointer to the Arrow buffer
 * @param ptr Pointer to the raw data
 * @param n_bytes buffer length in bytes
 * @param typ_enum Underlying dtype of elements
 * @return std::shared_ptr<BodoBuffer> Output Bodo buffer
 */
std::shared_ptr<BodoBuffer> arrow_buffer_to_bodo(
    std::shared_ptr<arrow::Buffer> buf, void *ptr, int64_t n_bytes,
    Bodo_CTypes::CTypeEnum typ_enum) {
    // Create a pointer to the Buffer shared_ptr to pass to the meminfo
    // destructor function for manual deletion. This essentially creates a
    // reference to the shared_ptr for Bodo to hold.
    std::shared_ptr<arrow::Buffer> *dtor_data =
        new std::shared_ptr<arrow::Buffer>(buf);

    // Create a meminfo holding Arrow data, which has a custom destructor that
    // deletes the Arrow buffer.
    NRT_MemInfo *buf_meminfo = NRT_MemInfo_allocate();
    NRT_MemInfo_init(buf_meminfo, ptr, 0, (NRT_dtor_function)arrow_buffer_dtor,
                     (void *)dtor_data, NULL);

    return std::make_shared<BodoBuffer>((uint8_t *)ptr, n_bytes, buf_meminfo,
                                        false);
}

/**
 * @brief Helper function to pass an Arrow null bitmap to Bodo with zero-copy
 * @param buf Shared pointer to the Arrow null bitmap buffer
 * @param data Pointer to the raw null bitmap data
 *  Using the buffer's data() method causes a segfault
 * @param offset Number of elements in the buffers to start from
 * @param null_count Number of null elements in the array
 *  Arrow doesn't allocate null bitmap if there are no nulls or all nulls
 * @param n_bits Number of bits / Number of elements in Array
 * @return null_bitmap_buffer Output Bodo meminfo
 */
std::shared_ptr<BodoBuffer> arrow_null_bitmap_to_bodo(
    std::shared_ptr<arrow::Buffer> buf, const uint8_t *data,
    const int64_t offset, int64_t null_count, int64_t n_bits) {
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_bits);
    int64_t offset_bytes = offset / 8;

    // Arrow doesn't allocate null bitmap if there are no nulls in the array
    if (data != nullptr) {
        if (offset == 0) {
            // Pass null bitmap buffer to Bodo with zero-copy
            return arrow_buffer_to_bodo(buf, (void *)data, n_bytes,
                                        Bodo_CTypes::UINT8);
        } else if (offset % 8 == 0) {
            // Allocate null bitmap and efficiently copy
            // n_bits when aligned on bytes
            std::shared_ptr<BodoBuffer> null_bitmap_buffer =
                AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
            memcpy(null_bitmap_buffer->mutable_data(), data + offset_bytes,
                   n_bytes);
            return null_bitmap_buffer;
        } else {
            // Allocate null bitmap and copy n_bits elements from offset
            std::shared_ptr<BodoBuffer> null_bitmap_buffer =
                AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
            // Initialize to all nulls:
            memset(null_bitmap_buffer->mutable_data(), 0x00, n_bytes);
            // Set the non-null bits:
            for (int64_t i = 0; i < n_bits; i++) {
                if (arrow::bit_util::GetBit(data, offset + i)) {
                    arrow::bit_util::SetBit(null_bitmap_buffer->mutable_data(),
                                            i);
                }
            }
            return null_bitmap_buffer;
        }
    } else {
        // Allocate null bitmap and set all elements to non-null or all null
        // based on the null_count (see
        // https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/array/array_base.h#L58)
        std::shared_ptr<BodoBuffer> null_bitmap_buffer =
            AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
        memset(null_bitmap_buffer->mutable_data(),
               null_count == 0 ? 0xff : 0x00, n_bytes);
        return null_bitmap_buffer;
    }
}

/**
 * @brief Helper function to construct a Bodo array from an Arrow NullArray
 * In Bodo, NullArray is represented as a string array with all nulls
 * So we have allocate a new string array
 * @param arrow_null_arr Input Arrow NullArray
 * @return out_array Output Bodo array
 */
std::shared_ptr<array_info> arrow_null_array_to_bodo(
    std::shared_ptr<arrow::NullArray> arrow_null_arr) {
    int64_t n = arrow_null_arr->length();

    auto out_array = alloc_array_top_level(n, 0, 0, bodo_array_type::STRING,
                                           Bodo_CTypes::STRING);
    // set offsets to zero
    memset(out_array->data2(), 0, sizeof(offset_t) * (n + 1));
    // setting all to null
    int64_t n_null_bytes = ((n + 7) >> 3);
    memset(out_array->null_bitmask(), 0, n_null_bytes);

    return out_array;
}

/**
 * @brief Helper function to construct a Bodo array from an Arrow ExtensionArray
 * @param arrow_ext_arr Input Arrow ExtensionArray
 * @return out_array Output Bodo array
 */
std::shared_ptr<array_info> arrow_timestamptz_array_to_bodo(
    std::shared_ptr<arrow::ExtensionArray> arrow_ext_arr) {
    auto arr = arrow_ext_arr->storage();
    auto arrow_struct_arr = std::static_pointer_cast<arrow::StructArray>(arr);
    int64_t n = arrow_struct_arr->length();
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_struct_arr->null_bitmap(), arrow_struct_arr->null_bitmap_data(),
        arrow_struct_arr->offset(), arrow_struct_arr->null_count(), n);

    std::shared_ptr<array_info> inner_arr_0 =
        arrow_array_to_bodo(arrow_struct_arr->field(0), -1, nullptr);
    std::shared_ptr<array_info> inner_arr_1 =
        arrow_array_to_bodo(arrow_struct_arr->field(1), -1, nullptr);
    // get the buffer from arr0 and arr1
    std::shared_ptr<BodoBuffer> arr0_buf = inner_arr_0->buffers[0];
    std::shared_ptr<BodoBuffer> arr1_buf = inner_arr_1->buffers[0];

    return std::make_shared<array_info>(
        bodo_array_type::TIMESTAMPTZ, Bodo_CTypes::TIMESTAMPTZ, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {arr0_buf, arr1_buf, null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({}));
}

/**
 * @brief Convert Arrow struct array to Bodo array_info (STRUCT
 * type) with zero-copy. The output Bodo array holds references to the Arrow
 * array's buffers and releases them when deleted.
 *
 * @param arrow_struct_arr Input Arrow StructArray
 * @param dicts_ref_arr Array used for setting dictionaries if provided. Should
 * have the same structure as input (i.e. number and type of fields)
 * @return std::shared_ptr<array_info> Output Bodo array (STRUCT type)
 */
std::shared_ptr<array_info> arrow_struct_array_to_bodo(
    std::shared_ptr<arrow::StructArray> arrow_struct_arr,
    std::shared_ptr<array_info> dicts_ref_arr) {
    int64_t n = arrow_struct_arr->length();

    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_struct_arr->null_bitmap(), arrow_struct_arr->null_bitmap_data(),
        arrow_struct_arr->offset(), arrow_struct_arr->null_count(), n);

    size_t n_fields = arrow_struct_arr->fields().size();
    std::vector<std::shared_ptr<array_info>> inner_arrs_vec;
    std::vector<std::string> field_names_vec;

    std::shared_ptr<arrow::DataType> struct_type = arrow_struct_arr->type();

    for (size_t i = 0; i < n_fields; i++) {
        std::shared_ptr<array_info> inner_arr = arrow_array_to_bodo(
            arrow_struct_arr->field(i), -1,
            dicts_ref_arr == nullptr ? nullptr
                                     : dicts_ref_arr->child_arrays[i]);
        std::string field_name = struct_type->field(i)->name();
        inner_arrs_vec.push_back(inner_arr);
        field_names_vec.push_back(field_name);
    }

    return std::make_shared<array_info>(
        bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, n,
        std::vector<std::shared_ptr<BodoBuffer>>({null_bitmap_buffer}),
        inner_arrs_vec, 0, 0, 0, -1, false, false, false, 0, field_names_vec);
}

/**
 * @brief Convert Arrow list array to Bodo array item array with zero-copy. The
 * output Bodo array holds references to the Arrow array's buffers and releases
 * them when deleted.
 *
 * @param arrow_list_arr Input Arrow LargeListArray
 * @param dicts_ref_arr Array used for setting dictionaries if provided. Should
 * have the same structure as input (i.e. child array type)
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_list_array_to_bodo(
    std::shared_ptr<arrow::LargeListArray> arrow_list_arr,
    std::shared_ptr<array_info> dicts_ref_arr) {
    int64_t n = arrow_list_arr->length();

    std::shared_ptr<BodoBuffer> offset_buffer =
        arrow_buffer_to_bodo(arrow_list_arr->value_offsets(),
                             (void *)arrow_list_arr->raw_value_offsets(),
                             (n + 1) * sizeof(offset_t), Bodo_CType_offset);
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_list_arr->null_bitmap(), arrow_list_arr->null_bitmap_data(),
        arrow_list_arr->offset(), arrow_list_arr->null_count(), n);

    std::shared_ptr<array_info> inner_arr = arrow_array_to_bodo(
        arrow_list_arr->values(), -1,
        dicts_ref_arr == nullptr ? nullptr : dicts_ref_arr->child_arrays[0]);

    return std::make_shared<array_info>(
        bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {offset_buffer, null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({inner_arr}));
}

/**
 * @brief Convert Arrow map array to Bodo map array. The
 * output Bodo array holds references to the Arrow array's buffers and releases
 * them when deleted.
 * This is not entirely zero-copy since Arrow has 32-bit offsets
 * which need to be converted to 64-bit for Bodo.
 *
 * @param arrow_map_arr Input Arrow MapArray
 * @param dicts_ref_arr Array used for setting dictionaries if provided. Should
 * have the same structure as input (i.e. child array types)
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_map_array_to_bodo(
    std::shared_ptr<arrow::MapArray> arrow_map_arr,
    std::shared_ptr<array_info> dicts_ref_arr) {
    int64_t n = arrow_map_arr->length();

    // Cast offsets from int32 to int64 required by Bodo
    std::shared_ptr<arrow::Array> offsets_arr = arrow_map_arr->offsets();
    auto res = arrow::compute::Cast(*offsets_arr, arrow::int64(),
                                    arrow::compute::CastOptions::Safe(),
                                    bodo::default_buffer_exec_context());
    std::shared_ptr<arrow::Array> casted_arr;
    CHECK_ARROW_AND_ASSIGN(res, "Cast", casted_arr);

    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> offsets_64_arr =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
            casted_arr);

    // Convert offset buffer to Bodo
    std::shared_ptr<BodoBuffer> offset_buffer = arrow_buffer_to_bodo(
        offsets_64_arr->values(), (void *)offsets_64_arr->raw_values(),
        (n + 1) * sizeof(offset_t), Bodo_CType_offset);

    // Convert null bitmap buffer to Bodo
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_map_arr->null_bitmap(), arrow_map_arr->null_bitmap_data(),
        arrow_map_arr->offset(), arrow_map_arr->null_count(), n);

    // Convert key array to Bodo
    std::shared_ptr<array_info> key_arr = arrow_array_to_bodo(
        arrow_map_arr->keys(), -1,
        dicts_ref_arr == nullptr
            ? nullptr
            : dicts_ref_arr->child_arrays[0]->child_arrays[0]->child_arrays[0]);

    // Convert item array to Bodo
    std::shared_ptr<array_info> item_arr = arrow_array_to_bodo(
        arrow_map_arr->items(), -1,
        dicts_ref_arr == nullptr
            ? nullptr
            : dicts_ref_arr->child_arrays[0]->child_arrays[0]->child_arrays[1]);

    // Get struct array null bitmap buffer (same as key array since Arrow
    // doesn't have a separate null bitmap)
    std::shared_ptr<BodoBuffer> struct_null_bitmap_buffer =
        arrow_null_bitmap_to_bodo(arrow_map_arr->keys()->null_bitmap(),
                                  arrow_map_arr->keys()->null_bitmap_data(),
                                  arrow_map_arr->keys()->offset(),
                                  arrow_map_arr->keys()->null_count(),
                                  arrow_map_arr->keys()->length());

    // Create inner struct array
    std::shared_ptr<array_info> struct_arr = std::make_shared<array_info>(
        bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, key_arr->length,
        std::vector<std::shared_ptr<BodoBuffer>>({struct_null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({key_arr, item_arr}), 0, 0, 0,
        -1, false, false, false, 0, std::vector<std::string>({"key", "value"}));

    // Create inner array(item) array
    std::shared_ptr<array_info> array_item_arr = std::make_shared<array_info>(
        bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {offset_buffer, null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({struct_arr}));

    return std::make_shared<array_info>(
        bodo_array_type::MAP, Bodo_CTypes::MAP, n,
        std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>({std::move(array_item_arr)}));
}

/**
 * @brief Convert Arrow decimal128 array to Bodo array_info (DECIMAL type) with
 * zero-copy. The output Bodo array holds references to the Arrow array's
 * buffers and releases them when deleted.
 *
 * @param arrow_decimal_arr Input Arrow Decimal128Array
 * @param force_aligned Whether to force a 16 byte alignment
 * for the underlying data. If this is set and the data is not 16 byte aligned
 * this will not be zero-copy. This can occur when we get the arrow data from
 * the Snowflake connector and is necessary because an accessing an pointer that
 * is not properly aligned can lead to undefined behavior in other kernels,
 * sometimes resulting in a segfault.
 * @return std::shared_ptr<array_info> Output Bodo array
 * (NULLABLE_INT_BOOL/DECIMAL type)
 */
std::shared_ptr<array_info> arrow_decimal_array_to_bodo(
    std::shared_ptr<arrow::Decimal128Array> arrow_decimal_arr,
    bool force_aligned) {
    int64_t n = arrow_decimal_arr->length();
    // Pass Arrow null bitmap and data buffer to Bodo
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_decimal_arr->null_bitmap(), arrow_decimal_arr->null_bitmap_data(),
        arrow_decimal_arr->offset(), arrow_decimal_arr->null_count(), n);
    const uint8_t *raw_data = arrow_decimal_arr->raw_values();
    int64_t n_bytes = n * numpy_item_size[Bodo_CTypes::DECIMAL];
    std::shared_ptr<BodoBuffer> data_buf_buffer;
    if (force_aligned && (reinterpret_cast<uintptr_t>(raw_data) % 16 != 0)) {
        // Allocate a new 16 byte aligned buffer (default alignment is 64B, so
        // it should always be 16B aligned)
        data_buf_buffer = AllocateBodoBuffer(n_bytes, Bodo_CTypes::DECIMAL);
        // Copy the data into the new buffer.
        memcpy(data_buf_buffer->mutable_data(), raw_data, n_bytes);
    } else {
        data_buf_buffer =
            arrow_buffer_to_bodo(arrow_decimal_arr->values(), (void *)raw_data,
                                 n_bytes, Bodo_CTypes::DECIMAL);
    }
    // get precision/scale info
    std::shared_ptr<arrow::Decimal128Type> dtype =
        std::static_pointer_cast<arrow::Decimal128Type>(
            arrow_decimal_arr->type());

    int32_t precision = dtype->precision();
    int32_t scale = dtype->scale();

    return std::make_shared<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::DECIMAL, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {data_buf_buffer, null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({}), precision, scale);
}

/**
 * @brief Convert Arrow nullable numeric array to Bodo array_info with zero-copy
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted.
 *
 * @param arrow_num_arr Input Arrow NumericArray
 * @param typ_enum Underlying dtype of elements
 * @return std::shared_ptr<array_info> Output Bodo array
 */
template <typename T>
std::shared_ptr<array_info> arrow_numeric_array_to_bodo(
    std::shared_ptr<T> arrow_num_arr, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t n = arrow_num_arr->length();

    // Pass Arrow null bitmap and data buffer to Bodo
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_num_arr->null_bitmap(), arrow_num_arr->null_bitmap_data(),
        arrow_num_arr->offset(), arrow_num_arr->null_count(), n);
    std::shared_ptr<BodoBuffer> data_buf_buffer = arrow_buffer_to_bodo(
        arrow_num_arr->values(), (void *)arrow_num_arr->raw_values(),
        n * numpy_item_size[typ_enum], typ_enum);

    return std::make_shared<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, typ_enum, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {data_buf_buffer, null_bitmap_buffer}));
}

/**
 * @brief Convert Arrow boolean array to Bodo array_info with zero-copy
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted.
 *
 * @param arrow_num_arr Input Arrow BooleanArray
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_boolean_array_to_bodo(
    std::shared_ptr<arrow::BooleanArray> arrow_bool_arr) {
    int64_t n_bits = arrow_bool_arr->length();
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_bits);

    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_bool_arr->null_bitmap(), arrow_bool_arr->null_bitmap_data(),
        arrow_bool_arr->offset(), arrow_bool_arr->null_count(), n_bits);

    // As BooleanArray does not expose the raw_values ptr, we cannot easily
    // copy the underlying boolean array starting from the offset. If offset
    // is zero, we cast to UInt8Array as a hack to access the buffer, otherwise
    // we avoid zero-copy.
    int64_t offset_bits = arrow_bool_arr->offset();
    std::shared_ptr<BodoBuffer> data_buf_buffer;
    if (offset_bits == 0) {
        arrow::UInt8Array *arrow_uint8_arr =
            reinterpret_cast<arrow::UInt8Array *>(arrow_bool_arr.get());
        data_buf_buffer = arrow_buffer_to_bodo(
            arrow_bool_arr->values(), (void *)arrow_uint8_arr->raw_values(),
            n_bytes, Bodo_CTypes::UINT8);
    } else {
        data_buf_buffer = AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
        uint8_t *data_buf = (uint8_t *)data_buf_buffer->mutable_data();
        for (int64_t i_bit = 0; i_bit < n_bits; i_bit++) {
            SetBitTo(data_buf, i_bit, arrow_bool_arr->Value(i_bit));
        }
    }
    return std::make_shared<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL, n_bits,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {data_buf_buffer, null_bitmap_buffer}));
}

/**
 * @brief Convert Arrow string or binary array to Bodo array_info with zero-copy
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted. A LargeStringArray is a LargeBinaryArray, so
 * this method works for both types.
 *
 * @param arrow_bin_arr Input Arrow string or binary array
 * @param typ_enum Underlying dtype of elements (string or binary)
 * @param array_id The identifier for equivalent arrays. If < 0 we need
 * to generate a new id.
 * @return array_info* Output Bodo array
 */
std::shared_ptr<array_info> arrow_string_binary_array_to_bodo(
    std::shared_ptr<arrow::LargeBinaryArray> arrow_bin_arr,
    Bodo_CTypes::CTypeEnum typ_enum, int64_t array_id) {
    int64_t n = arrow_bin_arr->length();

    std::shared_ptr<BodoBuffer> char_buf_buffer = arrow_buffer_to_bodo(
        arrow_bin_arr->value_data(), (void *)arrow_bin_arr->raw_data(),
        arrow_bin_arr->total_values_length(), Bodo_CTypes::UINT8);
    std::shared_ptr<BodoBuffer> offset_buffer =
        arrow_buffer_to_bodo(arrow_bin_arr->value_offsets(),
                             (void *)arrow_bin_arr->raw_value_offsets(),
                             (n + 1) * sizeof(offset_t), Bodo_CType_offset);
    std::shared_ptr<BodoBuffer> null_bitmap_buffer = arrow_null_bitmap_to_bodo(
        arrow_bin_arr->null_bitmap(), arrow_bin_arr->null_bitmap_data(),
        arrow_bin_arr->offset(), arrow_bin_arr->null_count(), n);

    if (array_id < 0) {
        array_id = generate_array_id(n);
    }

    return std::make_shared<array_info>(
        bodo_array_type::STRING, typ_enum, n,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {char_buf_buffer, offset_buffer, null_bitmap_buffer}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, 0, array_id);
}

/**
 * @brief Convert Arrow dict-encoded array to Bodo array_info with zero-copy.
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases thm when deleted.
 *
 * @param arrow_dict_arr Input Arrow DictionaryArray
 * @param dicts_ref_arr Array used for setting dictionary if provided
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_dictionary_array_to_bodo(
    std::shared_ptr<arrow::DictionaryArray> arrow_dict_arr,
    std::shared_ptr<array_info> dicts_ref_arr) {
    // Recurse on the dictionary and index arrays
    std::shared_ptr<array_info> dict_array;
    if (dicts_ref_arr != nullptr) {
        dict_array = dicts_ref_arr->child_arrays[0];
    } else {
        dict_array = arrow_array_to_bodo(arrow_dict_arr->dictionary());
    }
    std::shared_ptr<array_info> idx_array =
        arrow_array_to_bodo(arrow_dict_arr->indices());

    if (dict_array->dtype != Bodo_CTypes::STRING) {
        throw std::runtime_error(
            "arrow_dictionary_array_to_bodo(): Expected dict_array->dtype to "
            "be string, but found " +
            std::to_string(dict_array->dtype));
    }
    return create_dict_string_array(dict_array, idx_array);
}

std::shared_ptr<array_info> arrow_array_to_bodo(
    std::shared_ptr<arrow::Array> arrow_arr, int64_t array_id,
    std::shared_ptr<array_info> dicts_ref_arr) {
    switch (arrow_arr->type_id()) {
        case arrow::Type::LARGE_STRING:
            return arrow_string_binary_array_to_bodo(
                std::static_pointer_cast<arrow::LargeStringArray>(arrow_arr),
                Bodo_CTypes::STRING, array_id);
        // convert 32-bit offset array to 64-bit offset array to match Bodo data
        // layout
        case arrow::Type::STRING: {
            static_assert(OFFSET_BITWIDTH == 64);
            auto res =
                arrow::compute::Cast(*arrow_arr, arrow::large_utf8(),
                                     arrow::compute::CastOptions::Safe(),
                                     bodo::default_buffer_exec_context());
            std::shared_ptr<arrow::Array> casted_arr;
            CHECK_ARROW_AND_ASSIGN(res, "Cast", casted_arr);
            return arrow_string_binary_array_to_bodo(
                std::static_pointer_cast<arrow::LargeStringArray>(casted_arr),
                Bodo_CTypes::STRING, array_id);
        }
        case arrow::Type::LARGE_BINARY:
            return arrow_string_binary_array_to_bodo(
                std::static_pointer_cast<arrow::LargeBinaryArray>(arrow_arr),
                Bodo_CTypes::BINARY, array_id);
        // convert 32-bit offset array to 64-bit offset array to match Bodo data
        // layout
        case arrow::Type::BINARY: {
            static_assert(OFFSET_BITWIDTH == 64);
            auto res = arrow::compute::Cast(*arrow_arr, arrow::large_binary());
            std::shared_ptr<arrow::Array> casted_arr;
            CHECK_ARROW_AND_ASSIGN(res, "Cast", casted_arr);
            return arrow_string_binary_array_to_bodo(
                std::static_pointer_cast<arrow::LargeBinaryArray>(casted_arr),
                Bodo_CTypes::BINARY, array_id);
        }
        case arrow::Type::LARGE_LIST:
            return arrow_list_array_to_bodo(
                std::static_pointer_cast<arrow::LargeListArray>(arrow_arr),
                dicts_ref_arr);
        // convert 32-bit offset array to 64-bit offset array to match Bodo data
        // layout
        case arrow::Type::LIST: {
            static_assert(OFFSET_BITWIDTH == 64);
            auto res = arrow::compute::Cast(
                *arrow_arr, arrow::large_list(arrow_arr->type()->field(0)),
                arrow::compute::CastOptions::Safe(),
                bodo::default_buffer_exec_context());
            std::shared_ptr<arrow::Array> casted_arr;
            CHECK_ARROW_AND_ASSIGN(res, "Cast", casted_arr);
            return arrow_list_array_to_bodo(
                std::static_pointer_cast<arrow::LargeListArray>(casted_arr),
                dicts_ref_arr);
        }
        case arrow::Type::MAP: {
            return arrow_map_array_to_bodo(
                std::static_pointer_cast<arrow::MapArray>(arrow_arr),
                dicts_ref_arr);
        }
        case arrow::Type::STRUCT:
            return arrow_struct_array_to_bodo(
                std::static_pointer_cast<arrow::StructArray>(arrow_arr),
                dicts_ref_arr);
        case arrow::Type::DECIMAL128:
            return arrow_decimal_array_to_bodo(
                std::static_pointer_cast<arrow::Decimal128Array>(arrow_arr),
                /*force_aligned*/ true);
        case arrow::Type::DOUBLE:
            return arrow_numeric_array_to_bodo<arrow::DoubleArray>(
                std::static_pointer_cast<arrow::DoubleArray>(arrow_arr),
                Bodo_CTypes::FLOAT64);
        case arrow::Type::FLOAT:
            return arrow_numeric_array_to_bodo<arrow::FloatArray>(
                std::static_pointer_cast<arrow::FloatArray>(arrow_arr),
                Bodo_CTypes::FLOAT32);
        case arrow::Type::BOOL:
            return arrow_boolean_array_to_bodo(
                std::static_pointer_cast<arrow::BooleanArray>(arrow_arr));
        case arrow::Type::UINT64:
            return arrow_numeric_array_to_bodo<arrow::UInt64Array>(
                std::static_pointer_cast<arrow::UInt64Array>(arrow_arr),
                Bodo_CTypes::UINT64);
        case arrow::Type::INT64:
            return arrow_numeric_array_to_bodo<arrow::Int64Array>(
                std::static_pointer_cast<arrow::Int64Array>(arrow_arr),
                Bodo_CTypes::INT64);
        case arrow::Type::UINT32:
            return arrow_numeric_array_to_bodo<arrow::UInt32Array>(
                std::static_pointer_cast<arrow::UInt32Array>(arrow_arr),
                Bodo_CTypes::UINT32);
        case arrow::Type::DATE32:
            return arrow_numeric_array_to_bodo<arrow::Date32Array>(
                std::static_pointer_cast<arrow::Date32Array>(arrow_arr),
                Bodo_CTypes::DATE);
        case arrow::Type::TIMESTAMP:
            return arrow_numeric_array_to_bodo<arrow::TimestampArray>(
                std::static_pointer_cast<arrow::TimestampArray>(arrow_arr),
                Bodo_CTypes::DATETIME);
        case arrow::Type::INT32:
            return arrow_numeric_array_to_bodo<arrow::Int32Array>(
                std::static_pointer_cast<arrow::Int32Array>(arrow_arr),
                Bodo_CTypes::INT32);
        case arrow::Type::UINT16:
            return arrow_numeric_array_to_bodo<arrow::UInt16Array>(
                std::static_pointer_cast<arrow::UInt16Array>(arrow_arr),
                Bodo_CTypes::UINT16);
        case arrow::Type::INT16:
            return arrow_numeric_array_to_bodo<arrow::Int16Array>(
                std::static_pointer_cast<arrow::Int16Array>(arrow_arr),
                Bodo_CTypes::INT16);
        case arrow::Type::UINT8:
            return arrow_numeric_array_to_bodo<arrow::UInt8Array>(
                std::static_pointer_cast<arrow::UInt8Array>(arrow_arr),
                Bodo_CTypes::UINT8);
        case arrow::Type::INT8:
            return arrow_numeric_array_to_bodo<arrow::Int8Array>(
                std::static_pointer_cast<arrow::Int8Array>(arrow_arr),
                Bodo_CTypes::INT8);
        case arrow::Type::TIME64:
            return arrow_numeric_array_to_bodo<arrow::Time64Array>(
                std::static_pointer_cast<arrow::Time64Array>(arrow_arr),
                Bodo_CTypes::TIME);
        case arrow::Type::DICTIONARY:
            return arrow_dictionary_array_to_bodo(
                std::static_pointer_cast<arrow::DictionaryArray>(arrow_arr),
                dicts_ref_arr);
        case arrow::Type::NA:
            return arrow_null_array_to_bodo(
                std::static_pointer_cast<arrow::NullArray>(arrow_arr));
        case arrow::Type::EXTENSION: {
            // Cast the type to an ExtensionArray to access the extension name
            auto ext_type = std::static_pointer_cast<arrow::ExtensionType>(
                arrow_arr->type());
            auto name = ext_type->extension_name();
            if (name == "arrow_timestamp_tz") {
                return arrow_timestamptz_array_to_bodo(
                    std::static_pointer_cast<arrow::ExtensionArray>(arrow_arr));
            }
            // fallthrough
        }
        default:
            throw std::runtime_error("arrow_array_to_bodo(): Array type " +
                                     arrow_arr->type()->ToString() +
                                     " not supported");
    }
}

std::shared_ptr<table_info> arrow_recordbatch_to_bodo(
    std::shared_ptr<arrow::RecordBatch> arrow_rb) {
    std::vector<std::shared_ptr<array_info>> cols;
    cols.reserve(arrow_rb->num_columns());

    for (auto col : arrow_rb->columns()) {
        cols.push_back(arrow_array_to_bodo(col));
    }

    return std::make_shared<table_info>(cols);
}
