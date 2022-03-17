// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ArrowDataframeReader, ColumnBuilder subclasses and
// helper code to read Arrow data into Bodo.

#include "arrow_reader.h"
#include "../libs/_distributed.h"

using arrow::Type;

#define kNanosecondsInDay 86400000000000LL  // TODO: reuse from type_traits.h

// similar to arrow/python/arrow_to_pandas.cc ConvertDatetimeNanos except with
// just buffer
// TODO: reuse from arrow
template <typename T, int64_t SHIFT>
inline void convertArrowToDT64(const uint8_t* buff, uint8_t* out_data,
                               int64_t rows_to_skip, int64_t rows_to_read) {
    int64_t* out_values = (int64_t*)out_data;
    const T* in_values = (const T*)buff;
    for (int64_t i = 0; i < rows_to_read; ++i) {
        *out_values++ =
            (static_cast<int64_t>(in_values[rows_to_skip + i]) * SHIFT);
    }
}

// copied from Arrow since not in exported APIs
// https://github.com/apache/arrow/blob/329c9944554ddb142b0a2ac26a4abdf477636e37/cpp/src/arrow/python/datetime.cc#L150
// Extracts the month and year and day number from a number of days
static void get_date_from_days(int64_t days, int64_t* date_year,
                               int64_t* date_month, int64_t* date_day) {
    int64_t i;

    *date_year = days_to_yearsdays(&days);
    const int* month_lengths = days_per_month_table[is_leapyear(*date_year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            *date_month = i + 1;
            *date_day = days + 1;
            return;
        } else {
            days -= month_lengths[i];
        }
    }

    // Should never get here
    return;
}

/**
 * @brief copy date32 data into our packed datetime.date arrays
 *
 * @param out_data output data
 * @param buff date32 value buffer from Arrow
 * @param rows_to_skip number of items to skipp in buff
 * @param rows_to_read number of items to read after skipping
 */
inline void copy_data_dt32(uint64_t* out_data, const int32_t* buff,
                           int64_t rows_to_skip, int64_t rows_to_read) {
    for (int64_t i = 0; i < rows_to_read; i++) {
        int32_t val = buff[rows_to_skip + i];
        // convert date32 into packed datetime.date value
        int64_t
            year = -1,
            month = -1,
            day =
                -1;  // assigned to non-realized value to make any error crash.
        get_date_from_days(val, &year, &month, &day);
        out_data[i] = (year << 32) + (month << 16) + day;
    }
}

bool arrowBodoTypesEqual(std::shared_ptr<arrow::DataType> arrow_type,
                         Bodo_CTypes::CTypeEnum pq_type) {
    if (arrow_type->id() == Type::BOOL && pq_type == Bodo_CTypes::_BOOL)
        return true;
    if (arrow_type->id() == Type::UINT8 && pq_type == Bodo_CTypes::UINT8)
        return true;
    if (arrow_type->id() == Type::INT8 && pq_type == Bodo_CTypes::INT8)
        return true;
    if (arrow_type->id() == Type::UINT16 && pq_type == Bodo_CTypes::UINT16)
        return true;
    if (arrow_type->id() == Type::INT16 && pq_type == Bodo_CTypes::INT16)
        return true;
    if (arrow_type->id() == Type::UINT32 && pq_type == Bodo_CTypes::UINT32)
        return true;
    if (arrow_type->id() == Type::INT32 && pq_type == Bodo_CTypes::INT32)
        return true;
    if (arrow_type->id() == Type::UINT64 && pq_type == Bodo_CTypes::UINT64)
        return true;
    if (arrow_type->id() == Type::INT64 && pq_type == Bodo_CTypes::INT64)
        return true;
    if (arrow_type->id() == Type::FLOAT && pq_type == Bodo_CTypes::FLOAT32)
        return true;
    if (arrow_type->id() == Type::DOUBLE && pq_type == Bodo_CTypes::FLOAT64)
        return true;
    if (arrow_type->id() == Type::DECIMAL && pq_type == Bodo_CTypes::DECIMAL)
        return true;
    if (arrow_type->id() == Type::STRING && pq_type == Bodo_CTypes::STRING)
        return true;
    if (arrow_type->id() == Type::BINARY && pq_type == Bodo_CTypes::BINARY)
        return true;
    // TODO: add timestamp[ns]

    // Dictionary array's codes are always read into proper integer array type,
    // so buffer data types are the same
    if (arrow_type->id() == Type::DICTIONARY) return true;
    return false;
}

inline void copy_data_dispatch(uint8_t* out_data, const uint8_t* buff,
                               int64_t rows_to_skip, int64_t rows_to_read,
                               std::shared_ptr<arrow::DataType> arrow_type,
                               int out_dtype) {
    // read date32 values into datetime.date arrays, default from Arrow >= 0.13
    if (arrow_type->id() == Type::DATE32 && out_dtype == Bodo_CTypes::DATE) {
        copy_data_dt32((uint64_t*)out_data, (int32_t*)buff, rows_to_skip,
                       rows_to_read);
    }
    // datetime64 cases
    else if (out_dtype == Bodo_CTypes::DATETIME) {
        // similar to arrow_to_pandas.cc
        if (arrow_type->id() == Type::DATE32) {
            // days since epoch
            convertArrowToDT64<int32_t, kNanosecondsInDay>(
                buff, out_data, rows_to_skip, rows_to_read);
        } else if (arrow_type->id() == Type::DATE64) {
            // Date64Type is millisecond timestamp stored as int64_t
            convertArrowToDT64<int64_t, 1000000L>(buff, out_data, rows_to_skip,
                                                  rows_to_read);
        } else if (arrow_type->id() == Type::TIMESTAMP) {
            const auto& ts_type =
                static_cast<const arrow::TimestampType&>(*arrow_type);

            if (ts_type.unit() == arrow::TimeUnit::NANO) {
                int dtype_size = sizeof(int64_t);
                memcpy(out_data, buff + rows_to_skip * dtype_size,
                       rows_to_read * dtype_size);
            } else if (ts_type.unit() == arrow::TimeUnit::MICRO) {
                convertArrowToDT64<int64_t, 1000L>(buff, out_data, rows_to_skip,
                                                   rows_to_read);
            } else if (ts_type.unit() == arrow::TimeUnit::MILLI) {
                convertArrowToDT64<int64_t, 1000000L>(
                    buff, out_data, rows_to_skip, rows_to_read);
            } else if (ts_type.unit() == arrow::TimeUnit::SECOND) {
                convertArrowToDT64<int64_t, 1000000000L>(
                    buff, out_data, rows_to_skip, rows_to_read);
            } else {
                throw std::runtime_error(
                    "arrow read: Invalid datetime timeunit " +
                    arrow_type->ToString());
            }
        } else {
            throw std::runtime_error(
                "arrow read: Invalid datetime conversion " +
                arrow_type->ToString());
        }
    } else {
        throw std::runtime_error("arrow read: invalid dtype conversion for " +
                                 arrow_type->ToString());
    }
}

inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                      int64_t rows_to_skip, int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff,
                      Bodo_CTypes::CTypeEnum out_dtype) {
    // unpack booleans from bits
    if (out_dtype == Bodo_CTypes::_BOOL) {
        if (arrow_type->id() != Type::BOOL)
            std::cerr << "boolean type error" << '\n';

        for (int64_t i = 0; i < rows_to_read; i++) {
            out_data[i] =
                (uint8_t)::arrow::bit_util::GetBit(buff, i + rows_to_skip);
        }
        return;
    }

    if (arrowBodoTypesEqual(arrow_type, out_dtype)) {
        int dtype_size = numpy_item_size[out_dtype];
        // fast path if no conversion required
        memcpy(out_data, buff + rows_to_skip * dtype_size,
               rows_to_read * dtype_size);
    } else {
        copy_data_dispatch(out_data, buff, rows_to_skip, rows_to_read,
                           arrow_type, out_dtype);
    }
    // set NaNs for double values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::FLOAT64) {
        double* double_data = (double*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::bit_util::GetBit(null_bitmap_buff,
                                           i + rows_to_skip)) {
                // TODO: use NPY_NAN
                double_data[i] = std::nan("");
            }
        }
    }
    // set NaNs for float values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::FLOAT32) {
        float* float_data = (float*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::bit_util::GetBit(null_bitmap_buff,
                                           i + rows_to_skip)) {
                // TODO: use NPY_NAN
                float_data[i] = std::nanf("");
            }
        }
    }
    // set NaTs for datetime null values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::DATETIME) {
        int64_t* data = (int64_t*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::bit_util::GetBit(null_bitmap_buff,
                                           i + rows_to_skip)) {
                data[i] = std::numeric_limits<int64_t>::min();
            }
        }
    }
    return;
}

inline void copy_nulls(uint8_t* out_nulls, const uint8_t* null_bitmap_buff,
                       int64_t skip, int64_t num_values, int64_t null_offset) {
    if (out_nulls != nullptr) {
        if (null_bitmap_buff == nullptr) {
            for (size_t i = 0; i < size_t(num_values); i++) {
                // set all to not null
                ::arrow::bit_util::SetBit(out_nulls, null_offset + i);
            }
        } else {
            for (size_t i = 0; i < size_t(num_values); i++) {
                auto bit =
                    ::arrow::bit_util::GetBit(null_bitmap_buff, skip + i);
                SetBitTo(out_nulls, null_offset + i, bit);
            }
        }
    }
}

template <typename T>
inline void copy_nulls_categorical_inner(uint8_t* out_data,
                                         const uint8_t* null_bitmap_buff,
                                         int64_t skip, int64_t num_values) {
    T* data = (T*)out_data;
    for (size_t i = 0; i < size_t(num_values); i++) {
        auto bit = ::arrow::bit_util::GetBit(null_bitmap_buff, skip + i);
        if (!bit) data[i] = -1;
    }
}

/**
 * @brief set -1 code for null positions in categorical array from
 * Arrow's null bitmap.
 *
 * @param out_data output codes array for categoricals
 * @param null_bitmap_buff null bitmap from Arrow
 * @param skip skip previous rows in null buffer
 * @param num_values number of values to read
 * @param out_dtype data type for codes array
 */
inline void copy_nulls_categorical(uint8_t* out_data,
                                   const uint8_t* null_bitmap_buff,
                                   int64_t skip, int64_t num_values,
                                   int out_dtype) {
    // codes array can only be signed int 8/16/32/64
    if (out_dtype == Bodo_CTypes::INT8)
        copy_nulls_categorical_inner<int8_t>(out_data, null_bitmap_buff, skip,
                                             num_values);
    if (out_dtype == Bodo_CTypes::INT16)
        copy_nulls_categorical_inner<int16_t>(out_data, null_bitmap_buff, skip,
                                              num_values);
    if (out_dtype == Bodo_CTypes::INT32)
        copy_nulls_categorical_inner<int32_t>(out_data, null_bitmap_buff, skip,
                                              num_values);
    if (out_dtype == Bodo_CTypes::INT64)
        copy_nulls_categorical_inner<int64_t>(out_data, null_bitmap_buff, skip,
                                              num_values);
}

// -------------------- TableBuilder --------------------

/**
 * Column builder for datatypes that can be represented by fundamental C++
 * datatypes like int, double, etc. See TableBuilder constructor for details.
 */
class PrimitiveBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     * @param length : final output length on this process
     * @param is_nullable : true if array is nullable
     * @param is_categorical : true if column is categorical
     */
    PrimitiveBuilder(std::shared_ptr<arrow::DataType> type, int64_t length,
                     bool is_nullable, bool is_categorical)
        : is_nullable(is_nullable), is_categorical(is_categorical) {
        if (is_nullable)
            out_array =
                alloc_array(length, -1, -1, bodo_array_type::NULLABLE_INT_BOOL,
                            arrow_to_bodo_type(type), 0, -1);
        else
            out_array = alloc_array(length, -1, -1, bodo_array_type::NUMPY,
                                    arrow_to_bodo_type(type), 0, -1);
        dtype_size = numpy_item_size[out_array->dtype];
    }

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {
        std::shared_ptr<arrow::DataType> arrow_type = chunked_arr->type();

        for (int64_t i = 0; i < chunked_arr->num_chunks(); i++) {
            std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(i);
            // read the categorical codes for the DictionaryArray case.
            // category values are read during typing already
            if (is_categorical) {
                arr = reinterpret_cast<arrow::DictionaryArray*>(arr.get())
                          ->indices();
            }

            // because the input array might be sliced, we need to get the
            // range that we are reading (offset and length)
            int64_t in_offset = arr->data()->offset;
            int64_t in_length = arr->data()->length;
            auto buffers = arr->data()->buffers;
            if (buffers.size() != 2)
                throw std::runtime_error(
                    "PrimitiveBuilder: invalid number of array buffers");

            const uint8_t* buff = buffers[1]->data();
            const uint8_t* null_bitmap_buff =
                arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();

            uint8_t* data_ptr = reinterpret_cast<uint8_t*>(
                out_array->data1 + cur_offset * dtype_size);
            copy_data(data_ptr, buff, in_offset, in_length, arrow_type,
                      null_bitmap_buff, out_array->dtype);
            if (is_nullable)
                copy_nulls(reinterpret_cast<uint8_t*>(out_array->null_bitmask),
                           null_bitmap_buff, in_offset, in_length, cur_offset);

            // Arrow uses nullable arrays for categorical codes, but we use
            // regular numpy arrays and store -1 for null, so nulls have to be
            // set in data array
            if (is_categorical && arr->null_count() != 0) {
                copy_nulls_categorical(data_ptr, null_bitmap_buff, in_offset,
                                       in_length, out_array->dtype);
            }

            cur_offset += in_length;
        }
    }

   private:
    const bool is_nullable;
    const bool is_categorical;
    int dtype_size;  // sizeof dtype
    size_t cur_offset = 0;
};

/// Column builder for string arrays
class StringBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     */
    StringBuilder(std::shared_ptr<arrow::DataType> type)
        : dtype(arrow_to_bodo_type(type)) {}

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {
        // XXX hopefully keeping the string arrays around doesn't prevent other
        // intermediate Arow arrays and tables from being deleted when not
        // needed anymore
        arrays.insert(arrays.end(), chunked_arr->chunks().begin(),
                      chunked_arr->chunks().end());
    }

    virtual array_info* get_output() {
        if (out_array != nullptr) return out_array;
        // copy from Arrow arrays to Bodo array
        int64_t total_n_strings = 0;
        int64_t total_n_chars = 0;
        // This code works for both Binary and String arrays.
        // NOTE: arrow::StringArray is a subclass of arrow::BinaryArray.
        // We don't need StringArray specific methods or attributes, so we
        // use BinaryArray everywhere.
        for (auto arr : arrays) {
            auto str_arr = std::dynamic_pointer_cast<arrow::BinaryArray>(arr);
            const int64_t n_strings = str_arr->length();
            const int64_t str_start_offset = str_arr->data()->offset;
            const uint32_t* in_offsets =
                (uint32_t*)str_arr->value_offsets()->data();
            total_n_strings += n_strings;
            total_n_chars += in_offsets[str_start_offset + n_strings] -
                             in_offsets[str_start_offset];
        }
        out_array = alloc_array(total_n_strings, total_n_chars, -1,
                                bodo_array_type::STRING, dtype, 0, -1);
        int64_t n_null_bytes = (total_n_strings + 7) >> 3;
        memset(out_array->null_bitmask, 0, n_null_bytes);
        int64_t n_strings_copied = 0;
        int64_t n_chars_copied = 0;
        offset_t* out_offsets = (offset_t*)out_array->data2;
        out_offsets[0] = 0;
        for (auto arr : arrays) {
            auto str_arr = std::dynamic_pointer_cast<arrow::BinaryArray>(arr);
            const int64_t n_strings = str_arr->length();
            const int64_t str_start_offset = str_arr->data()->offset;
            const uint32_t* in_offsets =
                (uint32_t*)str_arr->value_offsets()->data();
            const int64_t n_chars = in_offsets[str_start_offset + n_strings] -
                                    in_offsets[str_start_offset];
            memcpy(out_array->data1 + n_chars_copied,
                   str_arr->value_data()->data() + in_offsets[str_start_offset],
                   sizeof(char) * n_chars);  // data
            for (int64_t i = 0; i < n_strings; i++) {
                out_offsets[n_strings_copied + i + 1] =
                    out_offsets[n_strings_copied + i] +
                    in_offsets[str_start_offset + i + 1] -
                    in_offsets[str_start_offset + i];
                if (!str_arr->IsNull(i))
                    SetBitTo((uint8_t*)out_array->null_bitmask,
                             n_strings_copied + i, true);
            }
            n_strings_copied += n_strings;
            n_chars_copied += n_chars;
        }
        out_offsets[total_n_strings] = static_cast<offset_t>(total_n_chars);
        arrays.clear();  // free Arrow memory
        return out_array;
    }

   private:
    const Bodo_CTypes::CTypeEnum dtype;  // STRING or BINARY
    arrow::ArrayVector arrays;
};

/// Column builder for dictionary-encoded string arrays
class DictionaryEncodedStringBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     */
    DictionaryEncodedStringBuilder(std::shared_ptr<arrow::DataType> type,
                                   int64_t length)
        : length(length) {
        //  'type' comes from the schema returned from
        //  bodo.io.parquet_pio.get_parquet_dataset() which always has string
        //  columns as STRING (not DICT)
        Bodo_CTypes::CTypeEnum dtype = arrow_to_bodo_type(type);
        if (dtype != Bodo_CTypes::CTypeEnum::STRING &&
            dtype != Bodo_CTypes::CTypeEnum::BINARY) {
            throw std::runtime_error(
                "DictionaryEncodedStringBuilder only supports STRING and "
                "BINARY data types");
        }
    }

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {
        // Store the chunks
        this->all_chunks.insert(this->all_chunks.end(),
                                chunked_arr->chunks().begin(),
                                chunked_arr->chunks().end());
    }

    virtual array_info* get_output() {
        if (out_array != nullptr) return out_array;

        if (length == 0) {
            this->out_array = alloc_dict_string_array(0, 0, 0, /*has_global_dictionary=*/false);
            return this->out_array;
        }

        // Unify all the chunks
        arrow::Result<std::shared_ptr<::arrow::ChunkedArray>> chunked_arr_res =
            ::arrow::ChunkedArray::Make(this->all_chunks);
        if (!chunked_arr_res.ok()) {
            throw std::runtime_error(
                "Runtime error in creation of chunked array...");
        }
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr =
            chunked_arr_res.ValueOrDie();
        arrow::Result<std::shared_ptr<::arrow::ChunkedArray>> unified_arr_res =
            arrow::DictionaryUnifier::UnifyChunkedArray(chunked_arr);
        if (!unified_arr_res.ok()) {
            throw std::runtime_error(
                "Runtime error in chunk array dictionary unification...");
        }
        std::shared_ptr<::arrow::ChunkedArray> unified_arr =
            unified_arr_res.ValueOrDie();

        // After unification, all chunks should have the same dictionary
        // TODO Verify
        std::shared_ptr<arrow::Array> first_chunk = (unified_arr->chunk(0));
        std::shared_ptr<arrow::Array> dictionary =
            reinterpret_cast<arrow::DictionaryArray*>(first_chunk.get())
                ->dictionary();

        // copy from Arrow arrays to Bodo array

        // copy dictionary
        StringBuilder dictionary_builder(dictionary->type());
        arrow::ArrayVector dict_v{dictionary};
        dictionary_builder.append(
            std::make_shared<arrow::ChunkedArray>(dict_v));
        array_info* bodo_dictionary = dictionary_builder.get_output();

        // copy indices
        PrimitiveBuilder indices_builder(arrow::int32(), this->length,
                                         /*is_nullable=*/true,
                                         /*is_categorical=*/false);
        arrow::ArrayVector indices_chunks;
        for (auto chunk : unified_arr->chunks()) {
            std::shared_ptr<arrow::DictionaryArray> dict_array_chunk =
                std::dynamic_pointer_cast<arrow::DictionaryArray>(chunk);
            // TODO assert that dict_array_chunk->indices() type is int32
            indices_chunks.push_back(dict_array_chunk->indices());
        }
        indices_builder.append(
            std::make_shared<arrow::ChunkedArray>(indices_chunks));
        array_info* bodo_indices = indices_builder.get_output();

        out_array = new array_info(
            bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length, -1,
            -1, NULL, NULL, NULL, bodo_indices->null_bitmask, NULL, NULL, NULL,
            NULL, 0, 0, 0, false, bodo_dictionary, bodo_indices);

        all_chunks.clear();
        return out_array;
    }

   private:
    const int64_t length;  // number of indices of output array
    arrow::ArrayVector all_chunks;
};

/// Column builder for list of string arrays
class ListStringBuilder : public TableBuilder::BuilderColumn {
   public:
    ListStringBuilder(std::shared_ptr<arrow::DataType> type) {
        string_builder = new StringBuilder(type->field(0)->type());
    }

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {
        // get child (StringArray) chunks and pass them to StringBuilder
        arrow::ArrayVector child_chunks;
        for (auto arr : chunked_arr->chunks()) {
            std::shared_ptr<arrow::ListArray> list_array =
                std::dynamic_pointer_cast<arrow::ListArray>(arr);
            // propagate slice information to child (string) arrays
            auto values_offset = list_array->raw_value_offsets()[0];
            auto values_length =
                list_array->raw_value_offsets()[list_array->length()] -
                values_offset;
            child_chunks.push_back(
                list_array->values()->Slice(values_offset, values_length));
        }
        string_builder->append(
            std::make_shared<arrow::ChunkedArray>(child_chunks));
        arrays.insert(arrays.end(), chunked_arr->chunks().begin(),
                      chunked_arr->chunks().end());
    }

    virtual array_info* get_output() {
        if (out_array != nullptr) return out_array;

        // get output Bodo string array
        array_info* out_str_array = string_builder->get_output();
        int64_t total_n_strings = out_str_array->length;
        delete string_builder;
        int64_t total_n_lists = 0;
        for (auto arr : arrays) total_n_lists += arr->length();

        // allocate Bodo list of string array with preexisting string array
        // alloc_list_string_array deletes the string array_info struct
        array_info* out_array =
            alloc_list_string_array(total_n_lists, out_str_array, 0);

        // copy list offsets and null bitmap
        int64_t n_lists_copied = 0;
        offset_t* out_offsets = (offset_t*)out_array->data3;  // list offsets
        out_offsets[0] = 0;
        for (auto arr : arrays) {
            auto list_arr = std::dynamic_pointer_cast<arrow::ListArray>(arr);
            const int64_t n_lists = list_arr->length();
            const int64_t list_start_offset = list_arr->data()->offset;
            const uint32_t* in_offsets =
                (uint32_t*)list_arr->value_offsets()->data();
            for (int64_t i = 0; i < n_lists; i++) {
                out_offsets[n_lists_copied + i + 1] =
                    out_offsets[n_lists_copied + i] +
                    in_offsets[list_start_offset + i + 1] -
                    in_offsets[list_start_offset + i];
                if (!list_arr->IsNull(i))
                    SetBitTo((uint8_t*)out_array->null_bitmask,
                             n_lists_copied + i, true);
            }
            n_lists_copied += n_lists;
        }
        out_offsets[total_n_lists] = static_cast<offset_t>(total_n_strings);
        arrays.clear();  // free Arrow memory
        return out_array;
    }

   private:
    StringBuilder* string_builder;
    arrow::ArrayVector arrays;
};

/**
 * Column builder for general nested Arrow arrays:
 * https://github.com/apache/arrow/blob/02e1da410cf24950b7ddb962de6598308690e369/cpp/src/arrow/type_traits.h#L1014
 * except LIST<STRING> which we handle separately for performance.
 */
class ArrowBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     */
    ArrowBuilder(std::shared_ptr<arrow::DataType> type) {}

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {
        // XXX hopefully keeping the arrays around doesn't prevent other
        // intermediate Arrow arrays and tables from being deleted when not
        // needed anymore
        arrays.insert(arrays.end(), chunked_arr->chunks().begin(),
                      chunked_arr->chunks().end());
    }

    virtual array_info* get_output() {
        if (out_array != nullptr) return out_array;
        std::shared_ptr<::arrow::Array> out_arrow_array;
        // TODO make this more efficient:
        // This copies to new buffers managed by Arrow, and then we copy again
        // to our own buffers in nested_array_to_c called by info_to_array
        // https://bodo.atlassian.net/browse/BE-1426
        out_arrow_array =
            arrow::Concatenate(arrays, arrow::default_memory_pool())
                .ValueOrDie();
        arrays.clear();  // memory of each array will be freed now
        out_array = new array_info(
            bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/,
            out_arrow_array->length(), -1, -1, NULL, NULL, NULL, NULL, NULL,
            /*meminfo TODO*/ NULL, NULL, out_arrow_array);
        return out_array;
    }

   private:
    arrow::ArrayVector arrays;
};

/// Column builder for Arrow arrays with all null values
class AllNullsBuilder : public TableBuilder::BuilderColumn {
   public:
    AllNullsBuilder(int64_t length) {
        // Arrow null arrays are typed as string in parquet_pio.py
        out_array = alloc_array(length, 0, 0, bodo_array_type::STRING,
                                Bodo_CTypes::STRING, 0, -1);
        // set offsets to zero
        memset(out_array->data2, 0, sizeof(offset_t) * length);
        // setting all to null
        int64_t n_null_bytes = ((length + 7) >> 3);
        memset(out_array->null_bitmask, 0, n_null_bytes);
    }

    virtual void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) {}
};

TableBuilder::TableBuilder(std::shared_ptr<arrow::Schema> schema,
                           std::set<int>& selected_fields,
                           const int64_t num_rows,
                           std::vector<bool>& is_nullable,
                           const std::set<std::string>& str_as_dict_cols) {
    int j = 0;
    for (int i : selected_fields) {
        const bool nullable_field = is_nullable[j++];
        // NOTE: these correspond to fields in arrow::Schema (a nested type
        // is a single field)
        auto field = schema->field(i);
        auto type = field->type()->id();
        // 'type' comes from the schema returned from
        // bodo.io.parquet_pio.get_parquet_dataset() which always has string
        // columns as STRING (not DICT)
        bool is_categorical = arrow::is_dictionary(type);
        if (arrow::is_primitive(type) || arrow::is_decimal(type) ||
            is_categorical) {
            if (is_categorical) {
                auto dict_type =
                    std::dynamic_pointer_cast<arrow::DictionaryType>(
                        field->type());
                columns.push_back(new PrimitiveBuilder(
                    dict_type->index_type(), num_rows, false, is_categorical));
            } else {
                columns.push_back(new PrimitiveBuilder(
                    field->type(), num_rows, nullable_field, is_categorical));
            }
        } else if (arrow::is_binary_like(type) &&
                   (str_as_dict_cols.count(field->name()) > 0)) {
            columns.push_back(
                new DictionaryEncodedStringBuilder(field->type(), num_rows));
        } else if (arrow::is_binary_like(type)) {
            columns.push_back(new StringBuilder(field->type()));
        } else if (type == arrow::Type::LIST &&
                   arrow::is_binary_like(
                       field->type()->field(0)->type()->id())) {
            columns.push_back(
                new ListStringBuilder(field->type()));  // list of string
        } else if (type == arrow::Type::NA) {
            columns.push_back(new AllNullsBuilder(num_rows));
        } else {
            columns.push_back(new ArrowBuilder(field->type()));
        }
    }
}

void TableBuilder::append(std::shared_ptr<::arrow::Table> table) {
    // NOTE table could be sliced, so the column builders need to take into
    // account the offset and length attributes of the Arrow arrays in the table
    for (size_t i = 0; i < columns.size(); i++) {
        columns[i]->append(table->column(i));
    }
}

// -------------------- ArrowDataframeReader --------------------
void ArrowDataframeReader::init(const std::vector<int32_t>& str_as_dict_cols) {
    if (initialized)
        throw std::runtime_error("ArrowDataframeReader already initialized");
    tracing::Event ev("reader::init", parallel);
    ev.add_attribute("g_parallel", parallel);
    ev.add_attribute("g_tot_rows_to_read", tot_rows_to_read);

    gilstate = PyGILState_Ensure();  // XXX is this needed??
    gil_held = true;

    PyObject* ds = get_dataset();
    schema = get_schema(ds);

    for (auto i : str_as_dict_cols) {
        auto field = schema->field(i);
        str_as_dict_colnames.emplace(field->name());
    }
    if (ev.is_tracing() && str_as_dict_cols.size() > 0) {
        std::string str_as_dict_colnames_str = "[";
        size_t i = 0;
        for (auto colname : str_as_dict_colnames) {
            if (i < str_as_dict_colnames.size() - 1)
                str_as_dict_colnames_str += colname + ", ";
            else
                str_as_dict_colnames_str += colname + "]";
            i++;
        }
        ev.add_attribute("g_str_as_dict_cols", str_as_dict_colnames_str);
    }

    // total_rows = ds.total_rows
    PyObject* total_rows_py = PyObject_GetAttrString(ds, "_bodo_total_rows");
    this->total_rows = PyLong_AsLongLong(total_rows_py);
    Py_DECREF(total_rows_py);

    // all_pieces = ds.pieces
    PyObject* all_pieces = PyObject_GetAttrString(ds, "pieces");
    ev.add_attribute("g_total_num_pieces", (size_t)PyObject_Length(all_pieces));
    Py_DECREF(ds);

    // iterate through pieces next
    PyObject* iterator = PyObject_GetIter(all_pieces);
    Py_DECREF(all_pieces);
    PyObject* piece;

    if (iterator == NULL)
        throw std::runtime_error(
            "ArrowDataframeReader::init(): error getting pieces iterator");

    if (!parallel) {
        // The process will read the whole dataset

        // head-only optimization ("limit pushdown"): user code may only use
        // df.head() so we just read the necessary rows
        this->count = (tot_rows_to_read != -1)
                          ? std::min(tot_rows_to_read, total_rows)
                          : total_rows;

        if (total_rows > 0) {
            // total number of rows of all the pieces we iterate through
            int64_t count_rows = 0;
            while ((piece = PyIter_Next(iterator))) {
                PyObject* num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                if (num_rows_piece_py == NULL)
                    throw std::runtime_error(
                        "_bodo_num_rows attribute not in piece");
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);
                if (num_rows_piece > 0) add_piece(piece, num_rows_piece);
                Py_DECREF(piece);
                count_rows += num_rows_piece;
                // finish when number of rows of my pieces covers my chunk
                if (count_rows >= this->count) break;
            }
        }
    } else {
        // is parallel (this process will read a chunk of dataset)

        // calculate the portion of rows that this process needs to read
        size_t rank = dist_get_rank();
        size_t nranks = dist_get_size();
        int64_t start_row_global = dist_get_start(total_rows, nranks, rank);
        this->count = dist_get_node_portion(total_rows, nranks, rank);

        // head-only optimization ("limit pushdown"): user code may only use
        // df.head() so we just read the necessary rows. This reads the data in
        // an imbalanced way. The assumed use case is printing df.head() which
        // looks better if the data is in fewer ranks.
        // TODO: explore if balancing data is necessary for some use cases
        if (tot_rows_to_read != -1) {
            // no rows to read on this process
            if (start_row_global >= tot_rows_to_read) {
                start_row_global = 0;
                this->count = 0;
            }
            // there may be fewer rows to read
            else {
                int64_t new_end =
                    std::min(start_row_global + this->count, tot_rows_to_read);
                this->count = new_end - start_row_global;
            }
        }

        // get those pieces that correspond to my chunk
        if (this->count > 0) {
            // total number of rows of all the pieces we iterate through
            int64_t count_rows = 0;
            // track total rows that this rank will read from pieces we iterate
            // through
            int64_t rows_added = 0;
            while ((piece = PyIter_Next(iterator))) {
                PyObject* num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                if (num_rows_piece_py == NULL)
                    throw std::runtime_error(
                        "_bodo_num_rows attribute not in piece");
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);

                // we skip all initial pieces whose total row count is less than
                // start_row_global (first row of my chunk). After that, we get
                // all subsequent pieces until the number of rows is greater or
                // equal to number of rows in my chunk
                if ((num_rows_piece > 0) &&
                    (start_row_global < count_rows + num_rows_piece)) {
                    int64_t rows_added_from_piece;
                    if (get_num_pieces() == 0) {
                        // this is the first piece
                        this->start_row_first_piece =
                            start_row_global - count_rows;
                        rows_added_from_piece = std::min(
                            num_rows_piece - this->start_row_first_piece,
                            this->count);
                    } else {
                        rows_added_from_piece =
                            std::min(num_rows_piece, this->count - rows_added);
                    }
                    rows_added += rows_added_from_piece;
                    this->add_piece(piece, rows_added_from_piece);
                }
                Py_DECREF(piece);

                count_rows += num_rows_piece;
                // finish when number of rows of my pieces covers my chunk
                if (rows_added == this->count) break;
            }
        }
    }
    Py_DECREF(iterator);

    if (PyErr_Occurred()) throw std::runtime_error("python");
    release_gil();

    if (ev.is_tracing()) {
        ev.add_attribute("g_schema", schema->ToString());
        std::string selected_fields_str;
        for (auto i : selected_fields) {
            selected_fields_str += schema->field(i)->ToString() + "\n";
        }
        ev.add_attribute("g_selected_fields", selected_fields_str);
        ev.add_attribute("num_pieces", get_num_pieces());
        ev.add_attribute("num_rows", size_t(count));
        ev.add_attribute("g_total_rows", total_rows);
    }
    initialized = true;
}
