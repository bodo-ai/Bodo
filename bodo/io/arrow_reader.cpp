
// Implementation of ArrowReader, ColumnBuilder subclasses and
// helper code to read Arrow data into Bodo.
#include "arrow_reader.h"

#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/dataset/scanner.h>
#include <arrow/io/interfaces.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <arrow/util/future.h>
#include <fmt/format.h>
#include <listobject.h>
#include <pyerrors.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_distributed.h"
#include "../libs/_stl.h"

#include "arrow_compat.h"
#include "json_col_parser.h"
#include "timestamptz_parser.h"

using arrow::Type;

#define kNanosecondsInDay 86400000000000LL  // TODO: reuse from type_traits.h

// If the arrow type is a time type, get the time unit associated with it
inline arrow::TimeUnit::type getTimeUnit(
    const std::shared_ptr<arrow::DataType>& type) {
    if (type->id() == arrow::Type::TIMESTAMP) {
        const auto& ts_type = static_cast<const arrow::TimestampType&>(*type);
        return ts_type.unit();
    } else {
        return arrow::TimeUnit::NANO;
    }
}

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

template <typename T, int64_t MULTIPLIER>
inline void convertArrowToTime64(const uint8_t* buff, uint8_t* out_data,
                                 int64_t rows_to_skip, int64_t rows_to_read) {
    int64_t* out_values = (int64_t*)out_data;
    const T* in_values = (const T*)buff;
    for (int64_t i = 0; i < rows_to_read; ++i) {
        *out_values++ =
            (static_cast<int64_t>(in_values[rows_to_skip + i]) * MULTIPLIER);
    }
}

bool arrowBodoTypesEqual(std::shared_ptr<arrow::DataType> arrow_type,
                         Bodo_CTypes::CTypeEnum pq_type) {
    switch (arrow_type->id()) {
        case Type::BOOL:
            return pq_type == Bodo_CTypes::_BOOL;
        case Type::UINT8:
            return pq_type == Bodo_CTypes::UINT8;
        case Type::INT8:
            return pq_type == Bodo_CTypes::INT8;
        case Type::UINT16:
            return pq_type == Bodo_CTypes::UINT16;
        case Type::INT16:
            return pq_type == Bodo_CTypes::INT16;
        case Type::UINT32:
            return pq_type == Bodo_CTypes::UINT32;
        case Type::INT32:
            return pq_type == Bodo_CTypes::INT32;
        case Type::UINT64:
            return pq_type == Bodo_CTypes::UINT64;
        case Type::INT64:
            return pq_type == Bodo_CTypes::INT64;
        case Type::FLOAT:
            return pq_type == Bodo_CTypes::FLOAT32;
        case Type::DOUBLE:
            return pq_type == Bodo_CTypes::FLOAT64;
        case Type::DECIMAL:
            return pq_type == Bodo_CTypes::DECIMAL;
        case Type::STRING:
        case Type::LARGE_STRING:
            return pq_type == Bodo_CTypes::STRING;
        case Type::BINARY:
            return pq_type == Bodo_CTypes::BINARY;
        case Type::DATE32:
            return pq_type == Bodo_CTypes::DATE;
        case Type::TIMESTAMP:
            return pq_type == Bodo_CTypes::DATETIME &&
                   getTimeUnit(arrow_type) == arrow::TimeUnit::NANO;
        case Type::DICTIONARY:
            // Dictionary array's codes are always read into proper integer
            // array type, so buffer data types are the same
            return true;
        default:
            return false;
    }
    return false;
}

inline void copy_data_dispatch(uint8_t* out_data, const uint8_t* buff,
                               int64_t rows_to_skip, int64_t rows_to_read,
                               std::shared_ptr<arrow::DataType> arrow_type,
                               Bodo_CTypes::CTypeEnum out_dtype) {
    // datetime64 cases
    if (out_dtype == Bodo_CTypes::DATETIME) {
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
    } else if (arrow_type->id() == Type::TIME32) {
        const auto& t_type = static_cast<const arrow::Time32Type&>(*arrow_type);
        if (t_type.unit() == arrow::TimeUnit::MILLI) {
            convertArrowToTime64<int32_t, 1000000>(buff, out_data, rows_to_skip,
                                                   rows_to_read);
        } else if (t_type.unit() == arrow::TimeUnit::SECOND) {
            convertArrowToTime64<int32_t, 1000000000>(
                buff, out_data, rows_to_skip, rows_to_read);
        } else {
            throw std::runtime_error("arrow read: Invalid time unit " +
                                     arrow_type->ToString());
        }
    } else if (arrow_type->id() == Type::TIME64) {
        const auto& t_type = static_cast<const arrow::Time64Type&>(*arrow_type);
        if (t_type.unit() == arrow::TimeUnit::NANO) {
            convertArrowToTime64<int64_t, 1>(buff, out_data, rows_to_skip,
                                             rows_to_read);
        } else if (t_type.unit() == arrow::TimeUnit::MICRO) {
            convertArrowToTime64<int64_t, 1000>(buff, out_data, rows_to_skip,
                                                rows_to_read);
        } else {
            throw std::runtime_error("arrow read: Invalid time unit " +
                                     arrow_type->ToString());
        }
    } else {
        throw std::runtime_error("arrow read: invalid dtype conversion for " +
                                 arrow_type->ToString() + " to " +
                                 GetDtype_as_string(out_dtype));
    }
}

inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                      int64_t rows_to_skip, int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff,
                      bodo_array_type::arr_type_enum array_type,
                      Bodo_CTypes::CTypeEnum out_dtype, int64_t curr_offset,
                      int dtype_size) {
    // unpack booleans from bits
    if (out_dtype == Bodo_CTypes::_BOOL) {
        if (arrow_type->id() != Type::BOOL) {
            throw std::runtime_error(
                "arrow read: invalid dtype conversion for " +
                arrow_type->ToString() + " to " +
                GetDtype_as_string(out_dtype));
        }
        // TODO: Replace with zero copy when the buffer can be copied
        // entirely (or copied via a view).
        if (array_type == bodo_array_type::NULLABLE_INT_BOOL) {
            for (int64_t i = 0; i < rows_to_read; i++) {
                auto bit = ::arrow::bit_util::GetBit(buff, i + rows_to_skip);
                SetBitTo(out_data, curr_offset + i, bit);
            }
        } else {
            for (int64_t i = 0; i < rows_to_read; i++) {
                auto bit = ::arrow::bit_util::GetBit(buff, i + rows_to_skip);
                out_data[curr_offset + i] = bit;
            }
        }
        return;
    }

    if (arrowBodoTypesEqual(arrow_type, out_dtype)) {
        // fast path if no conversion required
        memcpy(out_data + (curr_offset * dtype_size),
               buff + rows_to_skip * dtype_size, rows_to_read * dtype_size);
    } else {
        copy_data_dispatch(out_data + (curr_offset * dtype_size), buff,
                           rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // set NaNs for double values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::FLOAT64) {
        double* double_data = (double*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::bit_util::GetBit(null_bitmap_buff,
                                           i + rows_to_skip)) {
                // TODO: use NPY_NAN
                double_data[i + curr_offset] = std::nan("");
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
                float_data[i + curr_offset] = std::nanf("");
            }
        }
    }
    // set NaTs for datetime null values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::DATETIME) {
        int64_t* data = (int64_t*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::bit_util::GetBit(null_bitmap_buff,
                                           i + rows_to_skip)) {
                data[i + curr_offset] = std::numeric_limits<int64_t>::min();
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
        if (!bit) {
            data[i] = -1;
        }
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
                                   int64_t curr_offset, int64_t dtype_size,
                                   int out_dtype) {
    // codes array can only be signed int 8/16/32/64
    if (out_dtype == Bodo_CTypes::INT8) {
        copy_nulls_categorical_inner<int8_t>(
            out_data + (curr_offset * dtype_size), null_bitmap_buff, skip,
            num_values);
    } else if (out_dtype == Bodo_CTypes::INT16) {
        copy_nulls_categorical_inner<int16_t>(
            out_data + (curr_offset * dtype_size), null_bitmap_buff, skip,
            num_values);
    } else if (out_dtype == Bodo_CTypes::INT32) {
        copy_nulls_categorical_inner<int32_t>(
            out_data + (curr_offset * dtype_size), null_bitmap_buff, skip,
            num_values);
    } else if (out_dtype == Bodo_CTypes::INT64) {
        copy_nulls_categorical_inner<int64_t>(
            out_data + (curr_offset * dtype_size), null_bitmap_buff, skip,
            num_values);
    }
}

// -------------------- TableBuilder --------------------

/**
 * Column builder for datatypes that can be represented by fundamental C++
 * datatypes like int, double, etc. See TableBuilder constructor for details.
 */
class PrimitiveBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param dtype : Bodo type of input array
     * @param timeUnit: Time unit if dtype is Bodo_CTypes::DATETIME
     * @param length : final output length on this process
     * @param is_nullable : true if array is nullable
     * @param is_categorical : true if column is categorical
     */
    PrimitiveBuilder(Bodo_CTypes::CTypeEnum dtype, int64_t length,
                     bool is_nullable, bool is_categorical,
                     arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO)
        : is_nullable(is_nullable),
          is_categorical(is_categorical),
          dtype(dtype) {
        if (is_nullable && !is_categorical) {
            switch (dtype) {
                case Bodo_CTypes::FLOAT64:
                case Bodo_CTypes::FLOAT32:
                case Bodo_CTypes::_BOOL:
                case Bodo_CTypes::UINT64:
                case Bodo_CTypes::INT64:
                case Bodo_CTypes::UINT32:
                case Bodo_CTypes::INT32:
                case Bodo_CTypes::UINT16:
                case Bodo_CTypes::INT16:
                case Bodo_CTypes::UINT8:
                case Bodo_CTypes::INT8:
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DECIMAL:
                    return;

                case Bodo_CTypes::DATETIME:
                    if (time_unit == arrow::TimeUnit::NANO) {
                        return;
                    }
                    [[fallthrough]];
                default:
                    // Fallthrough for unsupported zero-copy cases
                    // TODO support timestamp types
                    ;
            }
        }
        // Only used in fallback if zero-copy is not supported
        temp_zero_copy_fallback = true;
        bodo_array_type::arr_type_enum out_array_type =
            is_nullable ? bodo_array_type::NULLABLE_INT_BOOL
                        : bodo_array_type::NUMPY;
        out_array =
            alloc_array_top_level(length, -1, -1, out_array_type, dtype);
    }

    PrimitiveBuilder(std::shared_ptr<arrow::DataType> type, int64_t length,
                     bool is_nullable, bool is_categorical)
        : PrimitiveBuilder(arrow_to_bodo_type(type->id()), length, is_nullable,
                           is_categorical, getTimeUnit(type)) {}

    void append(std::shared_ptr<arrow::ChunkedArray> chunked_arr) override {
        if (!temp_zero_copy_fallback) {
            // Accumulate chunked_arr's in an ArrayVector. Concatenate in Arrow
            // and convert to Bodo with zero-copy in get_output
            arrays.insert(arrays.end(), chunked_arr->chunks().begin(),
                          chunked_arr->chunks().end());
            return;
        }

        // Fallback if zero-copy is not supported
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
            if (buffers.size() != 2) {
                throw std::runtime_error(
                    "PrimitiveBuilder: invalid number of array buffers");
            }

            const uint8_t* buff = buffers[1]->data();
            const uint8_t* null_bitmap_buff =
                arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();

            int dtype_size = numpy_item_size[dtype];
            uint8_t* data_ptr = reinterpret_cast<uint8_t*>(out_array->data1());
            copy_data(data_ptr, buff, in_offset, in_length, arrow_type,
                      null_bitmap_buff, out_array->arr_type, out_array->dtype,
                      cur_offset, dtype_size);
            if (is_nullable) {
                copy_nulls(
                    reinterpret_cast<uint8_t*>(out_array->null_bitmask()),
                    null_bitmap_buff, in_offset, in_length, cur_offset);
            }

            // Arrow uses nullable arrays for categorical codes, but we use
            // regular numpy arrays and store -1 for null, so nulls have to be
            // set in data array
            if (is_categorical && arr->null_count() != 0) {
                copy_nulls_categorical(data_ptr, null_bitmap_buff, in_offset,
                                       in_length, cur_offset, dtype_size,
                                       out_array->dtype);
            }

            cur_offset += in_length;
        }
    }

    std::shared_ptr<array_info> get_output() override {
        if (out_array == nullptr && !temp_zero_copy_fallback) {
            if (arrays.empty()) {
                // Avoid empty call to concatenate
                out_array = is_nullable ? alloc_nullable_array(0, dtype)
                                        : alloc_numpy(0, dtype);
                return out_array;
            }
            auto* pool = bodo::BufferPool::DefaultPtr();
            arrow::Result<std::shared_ptr<arrow::Array>> res =
                arrow::Concatenate(arrays, pool);
            std::shared_ptr<arrow::Array> concat_res;
            CHECK_ARROW_READER_AND_ASSIGN(res, "Concatenate", concat_res);
            out_array = arrow_array_to_bodo(concat_res, pool);
        }
        return out_array;
    }

   private:
    const bool is_nullable;
    const bool is_categorical;
    Bodo_CTypes::CTypeEnum dtype;
    size_t cur_offset = 0;
    // TODO remove fallback once zero-copy is supported everywhere
    bool temp_zero_copy_fallback = false;
    arrow::ArrayVector arrays;
};

/// Column builder for string arrays
class StringBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param dtype : Bodo type of input array. Either STRING or BINARY.
     */
    StringBuilder(Bodo_CTypes::CTypeEnum dtype, int64_t array_id = -1)
        : dtype(dtype), array_id(array_id) {}

    void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) override {
        // Reserve some space up front in case of multiple chunks.
        arrays.reserve(arrays.size() + chunked_arr->chunks().size());
        for (const std::shared_ptr<arrow::Array>& chunk :
             chunked_arr->chunks()) {
            if (chunk->type_id() == arrow::Type::STRING) {
                static_assert(OFFSET_BITWIDTH == 64);
                // Convert 32-bit offset array to 64-bit offset array to match
                // Bodo data layout. This avoids overflow errors during the
                // Concatenate step in get_output.
                ::arrow::Result<std::shared_ptr<::arrow::Array>> res =
                    arrow::compute::Cast(*chunk, arrow::large_utf8(),
                                         arrow::compute::CastOptions::Safe(),
                                         bodo::default_buffer_exec_context());
                std::shared_ptr<arrow::Array> casted_arr;
                CHECK_ARROW_READER_AND_ASSIGN(res, "Cast", casted_arr);
                arrays.push_back(std::move(casted_arr));
            } else if (chunk->type_id() == arrow::Type::BINARY) {
                static_assert(OFFSET_BITWIDTH == 64);
                // Convert 32-bit offset array to 64-bit offset array to match
                // Bodo data layout. This avoids overflow errors during the
                // Concatenate step in get_output.
                ::arrow::Result<std::shared_ptr<::arrow::Array>> res =
                    arrow::compute::Cast(*chunk, arrow::large_binary(),
                                         arrow::compute::CastOptions::Safe(),
                                         bodo::default_buffer_exec_context());
                std::shared_ptr<arrow::Array> casted_arr;
                CHECK_ARROW_READER_AND_ASSIGN(res, "Cast", casted_arr);
                arrays.push_back(std::move(casted_arr));
            } else {  // LARGE_STRING or LARGE_BINARY
                arrays.push_back(chunk);
            }
        }
    }

    std::shared_ptr<array_info> get_output() override {
        if (out_array == nullptr) {
            if (arrays.empty()) {
                out_array = alloc_string_array(dtype, 0, 0);
                return out_array;
            }
            auto* pool = bodo::BufferPool::DefaultPtr();
            arrow::Result<std::shared_ptr<arrow::Array>> res =
                arrow::Concatenate(arrays, pool);
            std::shared_ptr<arrow::Array> concat_res;
            CHECK_ARROW_READER_AND_ASSIGN(res, "Concatenate", concat_res);
            out_array = arrow_array_to_bodo(concat_res, pool, array_id);
        }
        return out_array;
    }

   private:
    Bodo_CTypes::CTypeEnum dtype;
    arrow::ArrayVector arrays;
    int64_t array_id;
};

/// Column builder for dictionary-encoded string arrays
class DictionaryEncodedStringBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     */
    DictionaryEncodedStringBuilder(std::shared_ptr<arrow::DataType> type,
                                   int64_t length, int64_t dict_id = -1)
        : length(length), dict_id(dict_id) {
        //  'type' comes from the schema returned from
        //  bodo.io.parquet_pio.get_parquet_dataset() which always has
        //  string columns as STRING (not DICT)
        Bodo_CTypes::CTypeEnum dtype = arrow_to_bodo_type(type->id());
        if (dtype != Bodo_CTypes::CTypeEnum::STRING &&
            dtype != Bodo_CTypes::CTypeEnum::BINARY) {
            throw std::runtime_error(
                "DictionaryEncodedStringBuilder only supports STRING and "
                "BINARY data types");
        }
    }

    DictionaryEncodedStringBuilder(int64_t length, int64_t dict_id = -1)
        : length(length), dict_id(dict_id) {}

    void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) override {
        // Store the chunks
        this->all_chunks.insert(this->all_chunks.end(),
                                chunked_arr->chunks().begin(),
                                chunked_arr->chunks().end());
    }

    std::shared_ptr<array_info> get_output() override {
        if (out_array != nullptr) {
            return out_array;
        }

        if (length == 0) {
            this->out_array = alloc_dict_string_array(0, 0, 0);
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
        StringBuilder dictionary_builder(Bodo_CTypes::STRING, dict_id);
        arrow::ArrayVector dict_v{dictionary};
        dictionary_builder.append(
            std::make_shared<arrow::ChunkedArray>(dict_v));
        std::shared_ptr<array_info> bodo_dictionary =
            dictionary_builder.get_output();

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
        std::shared_ptr<array_info> bodo_indices = indices_builder.get_output();
        out_array = create_dict_string_array(bodo_dictionary, bodo_indices);
        all_chunks.clear();
        return out_array;
    }

   private:
    const int64_t length;  // number of indices of output array
    arrow::ArrayVector all_chunks;
    int64_t dict_id;  // ID used for generating the dictionaries. This is used
                      // by streaming to classify dictionaries as the same.
};

/// Column builder for constructing dictionary-encoded string arrays from string
/// arrays
/// TODO: Minimize duplicated code and shared logic from str_to_dict_str_array
/// function in _str_ext.cpp
class DictionaryEncodedFromStringBuilder : public TableBuilder::BuilderColumn {
   public:
    /**
     * @param type : Arrow type of input array
     */
    DictionaryEncodedFromStringBuilder(std::shared_ptr<arrow::DataType> type,
                                       int64_t length)
        : dtype(arrow_to_bodo_type(type->id())), length(length) {
        //  This builder is currently only used for Snowflake, where we
        //  determine whether or not to dictionary-encode ourselves.
        if (dtype != Bodo_CTypes::CTypeEnum::STRING &&
            dtype != Bodo_CTypes::CTypeEnum::BINARY) {
            throw std::runtime_error(
                "DictionaryEncodedFromStringBuilder only supports STRING and "
                "BINARY data types");
        }
        // initialize the indices array
        if (length > 0) {
            this->indices_arr =
                alloc_nullable_array(length, Bodo_CTypes::INT32, 0);
        }
    }

    void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) override {
        // unlike in other builders where we store the chunks, we incrementally
        // build the dictionary encoded array to save memory
        for (auto arr : chunked_arr->chunks()) {
            if (arrow::is_binary_like(arr->type_id())) {
                this->update_dict_and_fill_indices_from_chunk<
                    arrow::BinaryArray, uint32_t>(arr);
            } else if (arrow::is_large_binary_like(arr->type_id())) {
                this->update_dict_and_fill_indices_from_chunk<
                    arrow::LargeBinaryArray, uint64_t>(arr);
            } else {
                throw std::runtime_error(
                    "Unsupported array type provided to "
                    "DictionaryEncodedFromStringBuilder.");
            }
        }
        // XXX hopefully keeping the string arrays around doesn't prevent other
        // intermediate Arrow arrays and tables from being deleted when not
        // needed anymore
    }

    std::shared_ptr<array_info> get_output() override {
        if (out_array != nullptr) {
            return out_array;
        }

        if (length == 0) {
            this->out_array = alloc_dict_string_array(0, 0, 0);
            return this->out_array;
        }
        // We set is_locally_unique=true since we constructed the
        // dictionary ourselves and made sure not to put nulls in the
        // dictionary.
        std::shared_ptr<array_info> dict_arr =
            alloc_string_array(dtype, total_distinct_strings,
                               total_distinct_chars, -1, 0, false, true);
        int64_t n_null_bytes = (total_distinct_strings + 7) >> 3;
        offset_t* out_offsets =
            (offset_t*)dict_arr->data2<bodo_array_type::STRING>();
        // We know there's no nulls in the dictionary, so memset the
        // null_bitmask
        memset(dict_arr->null_bitmask<bodo_array_type::STRING>(), 0xFF,
               n_null_bytes);
        out_offsets[0] = 0;
        for (auto& it : str_to_ind) {
            memcpy(
                dict_arr->data1<bodo_array_type::STRING>() + it.second.second,
                it.first.c_str(), it.first.size());
            out_offsets[it.second.first] = it.second.second;
        }
        out_offsets[total_distinct_strings] =
            static_cast<offset_t>(total_distinct_chars);
        out_array = create_dict_string_array(dict_arr, indices_arr);
        return out_array;
    }

   private:
    template <typename ARROW_ARRAY_TYPE, typename OFFSET_TYPE>
    void update_dict_and_fill_indices_from_chunk(
        const std::shared_ptr<arrow::Array>& arr) {
        auto str_arr = std::dynamic_pointer_cast<ARROW_ARRAY_TYPE>(arr);
        const int64_t n_strings = str_arr->length();
        const int64_t str_start_offset = str_arr->data()->offset;
        const OFFSET_TYPE* in_offsets =
            (OFFSET_TYPE*)str_arr->value_offsets()->data();
        for (int64_t i = 0; i < n_strings; i++) {
            if (str_arr->IsNull(i)) {
                indices_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    n_strings_copied + i, false);
                continue;
            }
            indices_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                n_strings_copied + i, true);
            const uint64_t length = in_offsets[str_start_offset + i + 1] -
                                    in_offsets[str_start_offset + i];
            std::string_view val = str_arr->GetView(i);
            if (auto it = str_to_ind.find(val); it != str_to_ind.end()) {
                indices_arr
                    ->at<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(
                        n_strings_copied + i) = it->second.first;
            } else {
                indices_arr
                    ->at<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(
                        n_strings_copied + i) = count;
                std::pair<dict_indices_t, uint64_t> ind_offset_len =
                    std::make_pair(count++, total_distinct_chars);
                // TODO: remove std::string() after upgrade to C++23
                str_to_ind[std::string(val)] = ind_offset_len;
                total_distinct_chars += length;
                total_distinct_strings += 1;
            }
        }
        n_strings_copied += n_strings;
    }

    const Bodo_CTypes::CTypeEnum dtype;  // STRING or BINARY
    const int64_t length;                // number of indices of output array
    std::shared_ptr<array_info> indices_arr;
    // str_to_ind maps string to its new index, new offset in the new
    // dictionary array. the 0th offset maps to 0.
    // Supports access with string_view. See comments for string_hash.
    bodo::unord_map_container<std::string, std::pair<int32_t, uint64_t>,
                              string_hash, std::equal_to<>>
        str_to_ind;
    int64_t count = 0;  // cumulative counter, used for reindexing the unique
                        // strings in the inner dict array
    int64_t n_strings_copied =
        0;  // used for mapping the new index back to the indices array
    int64_t total_distinct_strings =
        0;  // number of strings in the inner dict array
    int64_t total_distinct_chars =
        0;  // total number of chars in the inner dict array
};

/**
 * Column builder for general nested Arrow arrays:
 * https://github.com/apache/arrow/blob/02e1da410cf24950b7ddb962de6598308690e369/cpp/src/arrow/type_traits.h#L1014
 * except LIST<STRING> which we handle separately for performance.
 */
class ArrowBuilder : public TableBuilder::BuilderColumn {
   public:
    ArrowBuilder(std::shared_ptr<arrow::DataType> _dtype)
        : dtype(std::move(_dtype)) {}

    void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) override {
        // XXX hopefully keeping the arrays around doesn't prevent other
        // intermediate Arrow arrays and tables from being deleted when not
        // needed anymore
        arrays.insert(arrays.end(), chunked_arr->chunks().begin(),
                      chunked_arr->chunks().end());
    }

    std::shared_ptr<array_info> get_output() override {
        auto* pool = bodo::BufferPool::DefaultPtr();
        if (out_array != nullptr) {
            return out_array;
        }

        if (arrays.empty()) {
            auto out_arrow_array =
                arrow::MakeEmptyArray(dtype, pool).ValueOrDie();
            out_array = arrow_array_to_bodo(out_arrow_array, pool);
            return out_array;
        }

        arrow::Result<std::shared_ptr<arrow::Array>> res =
            arrow::Concatenate(arrays, pool);
        std::shared_ptr<arrow::Array> concat_res;
        CHECK_ARROW_READER_AND_ASSIGN(res, "Concatenate", concat_res);
        arrays.clear();  // memory of each array will be freed now

        out_array = arrow_array_to_bodo(concat_res, pool);

        return out_array;
    }

   private:
    arrow::ArrayVector arrays;
    std::shared_ptr<arrow::DataType> dtype;
};

/// Column builder for Arrow arrays with all null values
class AllNullsBuilder : public TableBuilder::BuilderColumn {
   public:
    AllNullsBuilder(int64_t length) {
        // Arrow null arrays are typed as string in parquet_pio.py
        out_array = alloc_array_top_level(length, 0, 0, bodo_array_type::STRING,
                                          Bodo_CTypes::STRING);
        // set offsets to zero
        memset(out_array->data2<bodo_array_type::STRING>(), 0,
               sizeof(offset_t) * length);
        // setting all to null
        int64_t n_null_bytes = ((length + 7) >> 3);
        memset(out_array->null_bitmask<bodo_array_type::STRING>(), 0,
               n_null_bytes);
    }

    void append(std::shared_ptr<::arrow::ChunkedArray> chunked_arr) override {}
};

TableBuilder::TableBuilder(std::shared_ptr<arrow::Schema> schema,
                           std::vector<int>& selected_fields,
                           const int64_t num_rows,
                           const std::vector<bool>& is_nullable,
                           const std::set<std::string>& str_as_dict_cols,
                           const bool create_dict_from_string,
                           const std::vector<int64_t>& dict_ids) {
    this->total_rows = num_rows;
    this->rem_rows = num_rows;

    int j = 0;
    for (int i : selected_fields) {
        const bool nullable_field = is_nullable[j++];
        // NOTE: these correspond to fields in arrow::Schema (a nested type
        // is a single field)
        auto field = schema->field(i);
        auto type = field->type()->id();
        if (arrow::is_primitive(type) || arrow::is_decimal(type)) {
            columns.push_back(std::make_unique<PrimitiveBuilder>(
                field->type(), num_rows, nullable_field, false));
        } else if (arrow::is_string(type) &&
                   (str_as_dict_cols.count(field->name()) > 0)) {
            if (create_dict_from_string) {
                columns.push_back(
                    std::make_unique<DictionaryEncodedFromStringBuilder>(
                        field->type(), num_rows));
            } else {
                // Only Snowflake streaming will pass in dict ids.
                int64_t dict_id = dict_ids.size() > 0 ? dict_ids[i] : -1;
                columns.push_back(
                    std::make_unique<DictionaryEncodedStringBuilder>(
                        field->type(), num_rows, dict_id));
            }
        } else if (arrow::is_base_binary_like(type)) {
            Bodo_CTypes::CTypeEnum dtype;
            if (type == arrow::Type::STRING ||
                type == arrow::Type::LARGE_STRING) {
                dtype = Bodo_CTypes::STRING;
            } else {
                dtype = Bodo_CTypes::BINARY;
            }
            columns.push_back(std::make_unique<StringBuilder>(dtype));
        } else if (type == arrow::Type::NA) {
            columns.push_back(std::make_unique<AllNullsBuilder>(num_rows));
        } else {
            columns.push_back(std::make_unique<ArrowBuilder>(field->type()));
        }
    }
}

/**
 * @brief Convert Bodo Array Type to Arrow Data Type
 * TODO: Move this function to _bodo_to_arrow.cpp and extract similar code
 * from bodo_array_to_arrow
 *
 * @param arr Array to determine type of
 * @return Output Arrow DataType
 */
std::shared_ptr<arrow::DataType> bodo_arr_type_to_arrow_dtype(
    const std::shared_ptr<array_info>& arr) {
    if (arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        return arrow::large_list(
            bodo_arr_type_to_arrow_dtype(arr->child_arrays[0]));
    } else if (arr->arr_type == bodo_array_type::STRUCT) {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (size_t i = 0; i < arr->child_arrays.size(); i++) {
            auto field_name =
                arr->field_names.empty() ? "" : arr->field_names[i];
            fields.push_back(arrow::field(
                field_name,
                bodo_arr_type_to_arrow_dtype(arr->child_arrays[i])));
        }
        return arrow::struct_(fields);
    } else if (arr->arr_type == bodo_array_type::MAP) {
        return arrow::map(
            bodo_arr_type_to_arrow_dtype(arr->child_arrays[0]->child_arrays[0]),
            bodo_arr_type_to_arrow_dtype(arr->child_arrays[0]->child_arrays[1]),
            false);
    } else if (arr->arr_type == bodo_array_type::DICT) {
        return arrow::dictionary(arrow::int32(), arrow::large_utf8());
    } else if (arr->arr_type == bodo_array_type::STRING) {
        return arr->dtype == Bodo_CTypes::STRING ? arrow::large_utf8()
                                                 : arrow::large_binary();
    } else if (arr->arr_type == bodo_array_type::NUMPY ||
               arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        switch (arr->dtype) {
            case Bodo_CTypes::INT8:
                return arrow::int8();
            case Bodo_CTypes::INT16:
                return arrow::int16();
            case Bodo_CTypes::INT32:
                return arrow::int32();
            case Bodo_CTypes::INT64:
                return arrow::int64();
            case Bodo_CTypes::UINT8:
                return arrow::uint8();
            case Bodo_CTypes::UINT16:
                return arrow::uint16();
            case Bodo_CTypes::UINT32:
                return arrow::uint32();
            case Bodo_CTypes::UINT64:
                return arrow::uint64();
            case Bodo_CTypes::FLOAT32:
                return arrow::float32();
            case Bodo_CTypes::FLOAT64:
                return arrow::float64();
            case Bodo_CTypes::_BOOL:
                return arrow::boolean();
            case Bodo_CTypes::DATE:
                return arrow::date32();
            case Bodo_CTypes::DATETIME:
                return arrow::timestamp(arrow::TimeUnit::NANO);
            case Bodo_CTypes::DECIMAL:
                return arrow::decimal128(arr->precision, arr->scale);
            case Bodo_CTypes::TIME:
                switch (arr->precision) {
                    case 0:
                        return arrow::time32(arrow::TimeUnit::SECOND);
                    case 3:
                        return arrow::time32(arrow::TimeUnit::MILLI);
                    case 6:
                        return arrow::time64(arrow::TimeUnit::MICRO);
                    case 9:
                        return arrow::time64(arrow::TimeUnit::NANO);
                    default:
                        throw std::runtime_error(
                            "Unrecognized precision passed to "
                            "bodo_array_to_arrow: " +
                            std::to_string(arr->precision));
                }
            default:
                throw std::runtime_error(
                    "bodo_arr_type_to_arrow_dtype: unsupported dtype (" +
                    GetDtype_as_string(arr->dtype) + ")");
        }
    } else {
        throw std::runtime_error("bodo_arr_type_to_arrow_dtype: array type (" +
                                 GetArrType_as_string(arr->arr_type) +
                                 ") not supported");
    }
}

TableBuilder::TableBuilder(std::shared_ptr<table_info> table,
                           const int64_t num_rows) {
    total_rows = num_rows;
    rem_rows = num_rows;

    for (std::shared_ptr<array_info> arr : table->columns) {
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            columns.push_back(std::make_unique<PrimitiveBuilder>(
                arr->dtype, num_rows, true, false));
        } else if (arr->arr_type == bodo_array_type::NUMPY) {
            columns.push_back(std::make_unique<PrimitiveBuilder>(
                arr->dtype, num_rows, false, false));
        } else if (arr->arr_type == bodo_array_type::CATEGORICAL) {
            columns.push_back(std::make_unique<PrimitiveBuilder>(
                arr->dtype, num_rows, false, true));
        } else if (arr->arr_type == bodo_array_type::DICT) {
            columns.push_back(
                std::make_unique<DictionaryEncodedStringBuilder>(num_rows));
        } else if (arr->arr_type == bodo_array_type::STRING) {
            columns.push_back(std::make_unique<StringBuilder>(arr->dtype));
        } else if (arr->arr_type == bodo_array_type::STRUCT ||
                   arr->arr_type == bodo_array_type::ARRAY_ITEM ||
                   arr->arr_type == bodo_array_type::MAP) {
            columns.push_back(std::make_unique<ArrowBuilder>(
                bodo_arr_type_to_arrow_dtype(arr)));
        } else {
            throw std::runtime_error("TableBuilder: array type (" +
                                     GetArrType_as_string(arr->arr_type) +
                                     ") not supported");
        }
    }
}

void TableBuilder::append(std::shared_ptr<::arrow::Table> table) {
    // NOTE table could be sliced, so the column builders need to take into
    // account the offset and length attributes of the Arrow arrays in the
    // table
    rem_rows -= table->num_rows();
    for (size_t i = 0; i < this->columns.size(); i++) {
        this->columns[i]->append(table->column(i));
    }
}

// -------------------- ArrowReader --------------------

void ArrowReader::distribute_rows(PyObject* pieces_py) {
    assert(this->row_level);
    if (!this->parallel) {
        // The process will read the whole dataset.

        // head-only optimization ("limit pushdown"): user code may only use
        // df.head() so we just read the necessary rows
        this->count = (this->tot_rows_to_read != -1)
                          ? std::min(this->tot_rows_to_read, this->total_rows)
                          : this->total_rows;

        if (this->total_rows > 0) {
            // total number of rows of all the pieces we iterate through
            int64_t count_rows = 0;
            PyObject* piece;
            // iterate through pieces next
            PyObject* iterator = PyObject_GetIter(pieces_py);
            if (iterator == nullptr) {
                throw std::runtime_error(
                    "ArrowReader::distribute_rows(): error getting pieces "
                    "iterator");
            }
            while ((piece = PyIter_Next(iterator))) {
                PyObject* num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                if (num_rows_piece_py == nullptr) {
                    throw std::runtime_error(
                        "ArrowReader::distribute_rows: _bodo_num_rows "
                        "attribute not in piece");
                }
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);
                if (num_rows_piece > 0) {
                    this->add_piece(piece, num_rows_piece);
                    this->metrics.local_n_pieces_to_read_from++;
                }
                Py_DECREF(piece);
                count_rows += num_rows_piece;
                // finish when number of rows of my pieces covers my chunk
                if (count_rows >= this->count) {
                    break;
                }
            }
            Py_DECREF(iterator);
        }
    } else {
        // Is parallel (this process will read a chunk of dataset)

        // calculate the portion of rows that this process needs to read
        size_t rank = dist_get_rank();
        size_t nranks = dist_get_size();
        int64_t start_row_global =
            dist_get_start(this->total_rows, nranks, rank);
        this->count = dist_get_node_portion(this->total_rows, nranks, rank);

        // head-only optimization ("limit pushdown"): user code may only use
        // df.head() so we just read the necessary rows. This reads the data
        // in an imbalanced way. The assumed use case is printing df.head()
        // which looks better if the data is in fewer ranks.
        // TODO: explore if balancing data is necessary for some use cases
        if (this->tot_rows_to_read != -1) {
            // no rows to read on this process
            if (start_row_global >= this->tot_rows_to_read) {
                start_row_global = 0;
                this->count = 0;
            }
            // there may be fewer rows to read
            else {
                int64_t new_end = std::min(start_row_global + this->count,
                                           this->tot_rows_to_read);
                this->count = new_end - start_row_global;
            }
        }

        // get those pieces that correspond to my chunk
        if (this->count > 0) {
            // total number of rows of all the pieces we iterate through
            int64_t count_rows = 0;
            // track total rows that this rank will read from pieces we
            // iterate through
            int64_t rows_added = 0;
            PyObject* piece;
            // iterate through pieces next
            PyObject* iterator = PyObject_GetIter(pieces_py);
            if (iterator == nullptr) {
                throw std::runtime_error(
                    "ArrowReader::distribute_rows(): error getting "
                    "pieces iterator");
            }
            while ((piece = PyIter_Next(iterator))) {
                PyObject* num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                if (num_rows_piece_py == nullptr) {
                    throw std::runtime_error(
                        "ArrowReader::distribute_rows(): _bodo_num_rows "
                        "attribute not in piece");
                }
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);

                // We skip all initial pieces whose total row count is less
                // than start_row_global (first row of my chunk). After
                // that, we get all subsequent pieces until the number of
                // rows is greater or equal to number of rows in my chunk
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
                    this->metrics.local_n_pieces_to_read_from++;
                }
                Py_DECREF(piece);

                count_rows += num_rows_piece;
                // finish when number of rows of my pieces covers my chunk
                if (rows_added == this->count) {
                    break;
                }
            }
            Py_DECREF(iterator);
        }
    }
    this->metrics.local_rows_to_read = this->count;
}

void ArrowReader::distribute_pieces(PyObject* pieces_py) {
    throw std::runtime_error("ArrowReader::distribute_pieces: Unsupported!");
}

void ArrowReader::init_arrow_reader(std::span<int32_t> str_as_dict_cols,
                                    const bool create_dict_from_string) {
    auto start = start_timer();
    if (initialized) {
        throw std::runtime_error("ArrowReader already initialized");
    }

    tracing::Event ev("reader::init", this->parallel);
    ev.add_attribute("g_parallel", this->parallel);
    ev.add_attribute("g_tot_rows_to_read", this->tot_rows_to_read);

    gilstate = PyGILState_Ensure();  // XXX is this needed??
    gil_held = true;

    this->create_dict_encoding_from_strings = create_dict_from_string;
    this->metrics.n_str_as_dict_cols = str_as_dict_cols.size();

    // 'this->str_as_dict_colnames' must be initialized before calling
    // 'get_dataset()' since it may depend on it.
    for (auto i : str_as_dict_cols) {
        auto field = this->schema->field(i);
        this->str_as_dict_colnames.emplace(field->name());
    }
    if (ev.is_tracing()) {
        std::string str_as_dict_colnames_str = "[";
        size_t i = 0;
        for (auto colname : this->str_as_dict_colnames) {
            if (i < str_as_dict_colnames.size() - 1) {
                str_as_dict_colnames_str += colname + ", ";
            } else {
                str_as_dict_colnames_str += colname;
            }
            i++;
        }
        str_as_dict_colnames_str += "]";
        ev.add_attribute("g_str_as_dict_cols", str_as_dict_colnames_str);
    }

    time_pt start_get_ds = start_timer();
    PyObject* ds = get_dataset();
    this->metrics.get_ds_time += end_timer(start_get_ds);

    // row_level = ds.row_level
    PyObject* row_level_py = PyObject_GetAttrString(ds, "row_level");
    this->row_level = PyObject_IsTrue(row_level_py);
    Py_DECREF(row_level_py);

    // total_rows = ds.total_rows
    // In the row-level case, this is the exact count of the rows that will be
    // output after applying filters. In the piece-level case, this is the
    // amount of rows that will read from source before filtering.
    PyObject* total_rows_py = PyObject_GetAttrString(ds, "_bodo_total_rows");
    this->total_rows = PyLong_AsLongLong(total_rows_py);
    Py_DECREF(total_rows_py);

    // all_pieces = ds.pieces
    PyObject* all_pieces = PyObject_GetAttrString(ds, "pieces");

    // Total number of pieces globally.
    this->metrics.global_n_pieces = PyObject_Length(all_pieces);
    ev.add_attribute("g_total_num_pieces",
                     static_cast<size_t>(this->metrics.global_n_pieces));

    Py_DECREF(ds);

    if (this->force_row_level_read() && !this->row_level) {
        throw std::runtime_error(
            "ArrowReader::init_arrow_reader: Must do a row-level read but the "
            "dataset returned row_level=false. This is likely a bug!");
    }

    auto start_dist = start_timer();
    if (this->row_level) {
        this->distribute_rows(all_pieces);
    } else {
        this->distribute_pieces(all_pieces);
    }
    this->metrics.distribute_pieces_or_rows_time += end_timer(start_dist);

    Py_DECREF(all_pieces);

    if (PyErr_Occurred()) {
        throw std::runtime_error("python");
    }
    release_gil();

    if (ev.is_tracing()) {
        ev.add_attribute("g_schema", this->schema->ToString());
        std::string selected_fields_str;
        for (auto i : selected_fields) {
            selected_fields_str += this->schema->field(i)->ToString() + "\n";
        }
        ev.add_attribute("g_selected_fields", selected_fields_str);
        ev.add_attribute("num_pieces", static_cast<size_t>(get_num_pieces()));
        ev.add_attribute("num_rows", count);
        ev.add_attribute("g_total_rows", total_rows);
    }

    // Initialize the number of rows left to read
    this->rows_left_to_emit = count;
    this->rows_left_to_read = count;
    this->initialized = true;
    this->metrics.init_arrow_reader_total_time += end_timer(start);
}

std::shared_ptr<arrow::Table> ArrowReader::cast_arrow_table(
    std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Schema> schema,
    bool downcast_decimal_to_double) {
    if (table->schema()->Equals(schema)) {
        return table;
    }

    arrow::ChunkedArrayVector new_cols;
    for (int i = 0; i < table->num_columns(); i++) {
        //  We should do a quick check to ensure that we only perform
        //  upcasting and no nullability changes
        auto col = table->column(i);

        auto exp_type = schema->field(i)->type();
        auto exp_nullable = schema->field(i)->nullable();
        auto act_type = col->type();
        auto act_nullable = table->schema()->field(i)->nullable();

        // TODO: Simplify and update for nullable float and timestamps
        // Either
        // 1) Expected nullable and actually nullable
        // 2) Expected not nullable and actually not nullable
        // 3) Expected nullable and actually not nullable
        auto nullable_eq = exp_nullable == act_nullable || !act_nullable;
        // TableBuilder currently will allow for nullable floating-point or
        // timestamp arrays to be appended to non-nullable variants (by
        // converting NA to NaN or NaT) Thus, we shouldn't bother checking
        // for nullability in these cases
        if (exp_type->id() == Type::FLOAT || exp_type->id() == Type::DOUBLE ||
            exp_type->id() == Type::TIMESTAMP ||
            exp_type->id() == Type::TIME64 || exp_type->id() == Type::TIME32) {
            nullable_eq = true;
        } else if (exp_type->id() == Type::INT8 ||
                   exp_type->id() == Type::INT16 ||
                   exp_type->id() == Type::INT32 ||
                   exp_type->id() == Type::INT64) {
            // Integer types are sometimes misreported as being
            // nullable when they are not. In particular, COUNT(*)
            // is not nullable but the snowflake connector says it is.
            // If we've been told that the expected is not nullable
            // and the actual is nullable, confirm whether the actual
            // is compatible.
            if (!exp_nullable && act_nullable && col->null_count() == 0) {
                // There are no nulls so just mark this conversion as ok.
                nullable_eq = true;
            }
        }

        // We need to be able to cast between time types of the same
        // width with different units
        bool same_time_type = false;
        if ((exp_type->id() == Type::TIME64 &&
             act_type->id() == Type::TIME64) ||
            (exp_type->id() == Type::TIME32 &&
             act_type->id() == Type::TIME32)) {
            same_time_type = true;
        }

        if (act_type->Equals(exp_type) && nullable_eq) {
            new_cols.push_back(col);
        }

        // Parse Array of JSON Strings into List Array
        else if ((act_type->id() == Type::STRING ||
                  act_type->id() == Type::LARGE_STRING) &&
                 (exp_type->id() == Type::LIST ||
                  exp_type->id() == Type::LARGE_LIST)) {
            std::vector<std::shared_ptr<arrow::Array>> chunks;
            std::transform(
                col->chunks().begin(), col->chunks().end(),
                std::back_inserter(chunks),
                [exp_type](std::shared_ptr<arrow::Array> chunk) {
                    return string_to_list_arr(
                        std::dynamic_pointer_cast<arrow::StringArray>(chunk),
                        exp_type);
                });
            new_cols.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
        }

        // Parse null array
        else if ((act_type->id() == Type::STRING ||
                  act_type->id() == Type::LARGE_STRING) &&
                 exp_type->id() == Type::NA) {
            std::vector<std::shared_ptr<arrow::Array>> chunks;
            std::transform(
                col->chunks().begin(), col->chunks().end(),
                std::back_inserter(chunks),
                [](std::shared_ptr<arrow::Array> chunk) {
                    return std::make_shared<arrow::NullArray>(chunk->length());
                });
            new_cols.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
        }

        // Parse Array of JSON Strings into Map Array
        else if ((act_type->id() == Type::STRING ||
                  act_type->id() == Type::LARGE_STRING) &&
                 exp_type->id() == Type::MAP) {
            std::vector<std::shared_ptr<arrow::Array>> chunks;
            std::transform(
                col->chunks().begin(), col->chunks().end(),
                std::back_inserter(chunks),
                [exp_type](std::shared_ptr<arrow::Array> chunk) {
                    return string_to_map_arr(
                        std::dynamic_pointer_cast<arrow::StringArray>(chunk),
                        exp_type);
                });
            new_cols.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
        }

        // Parse Array of JSON Strings into Struct Array
        else if ((act_type->id() == Type::STRING ||
                  act_type->id() == Type::LARGE_STRING) &&
                 exp_type->id() == Type::STRUCT) {
            std::vector<std::shared_ptr<arrow::Array>> chunks;
            std::transform(
                col->chunks().begin(), col->chunks().end(),
                std::back_inserter(chunks),
                [exp_type](std::shared_ptr<arrow::Array> chunk) {
                    return string_to_struct_arr(
                        std::dynamic_pointer_cast<arrow::StringArray>(chunk),
                        exp_type);
                });
            new_cols.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
        }

        else if (exp_type->id() == Type::EXTENSION) {
            if (act_type->id() != Type::STRING &&
                act_type->id() != Type::LARGE_STRING) {
                throw std::runtime_error(
                    "Reading ExtensionTypes is only supported for strings");
            }

            auto ext_name =
                std::static_pointer_cast<arrow::ExtensionType>(exp_type)
                    ->extension_name();
            // Parse Array of TimestampTZ Strings into TimestampTZ Array
            if (ext_name == "arrow_timestamp_tz") {
                std::vector<std::shared_ptr<arrow::Array>> chunks;
                std::transform(
                    col->chunks().begin(), col->chunks().end(),
                    std::back_inserter(chunks),
                    [exp_type](std::shared_ptr<arrow::Array> chunk) {
                        return string_to_timestamptz_arr(
                            std::dynamic_pointer_cast<arrow::StringArray>(
                                chunk),
                            exp_type);
                    });
                new_cols.push_back(
                    std::make_shared<arrow::ChunkedArray>(chunks));
            } else {
                throw std::runtime_error("ExtensionType " + ext_name +
                                         " not supported");
            }
        }

        // Float -> Double
        // Int -> Wider Int
        // Int -> Wider Decimal
        // Float / Double -> Wider Decimal
        else if ((act_type->bit_width() < exp_type->bit_width() ||
                  same_time_type) &&
                 nullable_eq) {
            // Check if upcast is possible in this case
            // Dont bother checking if types are compatible, should
            // be done at compile-time. Only sizes can change.

            // bit-width is well-defined for all types with a fixed width.
            // For other types, such as strings it's always set to -1, and
            // hence should be safe to compare.
            // (https://arrow.apache.org/docs/cpp/api/datatype.html#_CPPv4NK5arrow8DataType9bit_widthEv)

            auto res = arrow::compute::Cast(
                col, exp_type, arrow::compute::CastOptions::Safe(),
                bodo::default_buffer_exec_context());
            // TODO: Use fmt::format or std::format on C++23
            CHECK_ARROW_READER(res.status(),
                               "Failed to safely cast from " +
                                   col->type()->ToString() + " to " +
                                   exp_type->ToString() +
                                   " before appending to TableBuilder");
            auto casted_datum = res.ValueOrDie();
            new_cols.push_back(casted_datum.chunked_array());

        } else if (downcast_decimal_to_double &&
                   act_type->id() == Type::DECIMAL128 &&
                   exp_type->id() == Type::DOUBLE && nullable_eq) {
            // Bodo has limited support for decimal columns right now
            // We can attempt to get around it by unsafely downcasting to
            // double
            auto opt = arrow::compute::CastOptions::Safe();
            opt.allow_decimal_truncate = true;

            auto res = arrow::compute::Cast(
                col, exp_type, opt, bodo::default_buffer_exec_context());
            CHECK_ARROW_READER(res.status(), "Failed to downcast from " +
                                                 col->type()->ToString() +
                                                 " to " + exp_type->ToString());

            auto casted_datum = res.ValueOrDie();
            new_cols.push_back(casted_datum.chunked_array());

        } else if (act_type->id() == Type::DECIMAL128 &&
                   exp_type->id() == Type::DECIMAL128 && nullable_eq) {
            int32_t act_scale =
                std::reinterpret_pointer_cast<arrow::Decimal128Type>(act_type)
                    ->scale();
            int32_t exp_scale =
                std::reinterpret_pointer_cast<arrow::Decimal128Type>(exp_type)
                    ->scale();
            int32_t exp_precision =
                std::reinterpret_pointer_cast<arrow::Decimal128Type>(exp_type)
                    ->precision();
            if (act_scale == exp_scale) {
                // Cast between decimal types that have the same scale, but
                // first verify that the values can all fit in the new
                // precision.
                int n_chunks = col->num_chunks();
                for (int cur_chunk = 0; cur_chunk < n_chunks; cur_chunk++) {
                    const std::shared_ptr<arrow::Array>& chunk =
                        col->chunk(cur_chunk);
                    size_t n_rows = chunk->length();
                    std::shared_ptr<arrow::Decimal128Array> array =
                        std::reinterpret_pointer_cast<arrow::Decimal128Array>(
                            chunk);
                    const uint8_t* raw_data = array->values()->data();
                    auto* decimal_data =
                        reinterpret_cast<const arrow::Decimal128*>(raw_data);
                    for (size_t row = 0; row < n_rows; row++) {
                        if (array->IsValid(row) &&
                            !decimal_data[row].FitsInPrecision(exp_precision)) {
                            throw std::runtime_error(
                                "cast_arrow_table: number out of representable "
                                "range when casted to type " +
                                exp_type->ToString());
                        }
                    }
                }

                auto opt = arrow::compute::CastOptions::Safe();
                opt.allow_decimal_truncate = true;

                auto res = arrow::compute::Cast(
                    col, exp_type, opt, bodo::default_buffer_exec_context());
                CHECK_ARROW_READER(res.status(), "Failed to cast from " +
                                                     col->type()->ToString() +
                                                     " to " +
                                                     exp_type->ToString());

                auto casted_datum = res.ValueOrDie();
                new_cols.push_back(casted_datum.chunked_array());
            } else {
                throw std::runtime_error(
                    "Unsupported cast from " + col->type()->ToString() +
                    " to " + exp_type->ToString() +
                    " (decimal casting currently requires the same scale)");
            }
        } else {
            // TODO: Use fmt::format or std::format on C++23
            throw std::runtime_error("Invalid Downcast from " +
                                     col->type()->ToString() + " to " +
                                     exp_type->ToString());
        }
    }

    return arrow::Table::Make(schema, new_cols, table->num_rows());
}

void ArrowReader::ReportInitStageMetrics(std::vector<MetricBase>& metrics_out) {
    if ((this->op_id == -1) || this->reported_init_stage_metrics) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 16);

    // Globals
    MetricBase::StatValue limit_nrows_ = this->tot_rows_to_read;
    metrics_out.emplace_back(StatMetric("limit_nrows", limit_nrows_, true));
    MetricBase::StatValue force_row_level_ =
        this->force_row_level_read() ? 1 : 0;
    metrics_out.emplace_back(
        StatMetric("force_row_level", force_row_level_, true));
    MetricBase::StatValue row_level_ = this->row_level ? 1 : 0;
    metrics_out.emplace_back(StatMetric("row_level", row_level_, true));
    // In the row-level case, this is the exact count of the rows that will be
    // output after applying filters. In the piece-level case, this is the
    // amount of rows that will read from source before filtering.
    MetricBase::StatValue global_nrows_to_read = this->total_rows;
    metrics_out.emplace_back(
        StatMetric("global_nrows_to_read", global_nrows_to_read, true));
    metrics_out.emplace_back(
        StatMetric("global_n_pieces", this->metrics.global_n_pieces, true));
    MetricBase::StatValue create_dict_from_str =
        this->create_dict_encoding_from_strings ? 1 : 0;
    metrics_out.emplace_back(StatMetric("create_dict_encoding_from_strings",
                                        create_dict_from_str, true));
    metrics_out.emplace_back(StatMetric(
        "n_str_as_dict_cols", this->metrics.n_str_as_dict_cols, true));

    // Locals
    metrics_out.emplace_back(
        TimerMetric("get_ds_time", this->metrics.get_ds_time, false));
    metrics_out.emplace_back(
        TimerMetric("init_arrow_reader_total_time",
                    this->metrics.init_arrow_reader_total_time, false));
    metrics_out.emplace_back(
        TimerMetric("distribute_pieces_or_rows_time",
                    this->metrics.distribute_pieces_or_rows_time, false));
    metrics_out.emplace_back(
        StatMetric("local_rows_to_read", this->metrics.local_rows_to_read));
    metrics_out.emplace_back(
        StatMetric("local_n_pieces_to_read_from",
                   this->metrics.local_n_pieces_to_read_from));

    this->reported_init_stage_metrics = true;
}

void ArrowReader::ReportReadStageMetrics(std::vector<MetricBase>& metrics_out) {
    if ((this->op_id == -1) || this->reported_read_stage_metrics) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 2);

    metrics_out.emplace_back(TimerMetric("read_batch_total_time",
                                         this->metrics.read_batch_total_time));

    this->reported_read_stage_metrics = true;
}

/**
 * @brief Unify the given table with the dictionary builders
 * for its dictionary columns. This should only be used in the streaming case.
 *
 * @param table Input table.
 * @return table_info* New combined table with unified dictionary columns.
 */
table_info* ArrowReader::unify_table_with_dictionary_builders(
    std::shared_ptr<table_info> table) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(table->ncols());
    for (uint64_t i = 0; i < table->ncols(); i++) {
        if (this->dict_builders[i] != nullptr) {
            out_arrs.emplace_back(this->dict_builders[i]->UnifyDictionaryArray(
                table->columns[i]));
        } else {
            out_arrs.emplace_back(table->columns[i]);
        }
    }
    return new table_info(out_arrs, table->nrows());
}

/**
 * @brief Py Entry Function to Call read_batch(...) on ArrowReaders
 *
 * @param[in] reader ArrowReader object to get next batch of
 * @param[out] is_last_out Bool to pass to Python if is last batch
 * @param[out] total_rows_out uint64 to pass to Python for the # of rows
 *        in the output batch
 * @param produce_output Bool to indicate whether to produce output
 * @return table_info* Output Bodo table representing the batch
 */
table_info* arrow_reader_read_py_entry(ArrowReader* reader, bool* is_last_out,
                                       uint64_t* total_rows_out,
                                       bool produce_output) {
    try {
        bool is_last_out_ = false;
        uint64_t total_rows_out_ = 0;

        table_info* table =
            reader->read_batch(is_last_out_, total_rows_out_, produce_output);

        *total_rows_out = total_rows_out_;
        *is_last_out = is_last_out_;
        reader->metrics.output_row_count += total_rows_out_;
        if (is_last_out_) {
            if (reader->op_id != -1 &&
                !reader->get_reported_read_stage_metrics()) {
                QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                    QueryProfileCollector::MakeOperatorStageID(
                        reader->op_id, QUERY_PROFILE_READ_STAGE_ID),
                    reader->metrics.output_row_count);

                std::vector<MetricBase> metrics;
                reader->ReportReadStageMetrics(metrics);
                QueryProfileCollector::Default().RegisterOperatorStageMetrics(
                    QueryProfileCollector::MakeOperatorStageID(
                        reader->op_id, QUERY_PROFILE_READ_STAGE_ID),
                    std::move(metrics));
            }
        }
        return table;
    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return nullptr;
    }
}

/**
 * @brief Delete / deallocate an ArrowReader object
 * @param[in] reader ArrowReader object to delete
 */
void arrow_reader_del_py_entry(ArrowReader* reader) { delete reader; }

/**
 * @brief Impl for PyCFunction arrow_cpp.fetch_parquet_frags_metadata
 * Unwraps a python list of pyarrow ParquetFileFragments into
 * arrow::dataset::ParquetFileFragments and dispatches an async task to fetch
 * there metadata. Returns upon completion of all tasks
 * @param self: unused, will be NULL
 * @param args: list of python arguments, should always be 1 list of pyarrow
 * ParquetFileFragments
 * @param nargs: number of args, should always be 1
 */
PyObject* fetch_parquet_frags_metadata(PyObject* self, PyObject* const* args,
                                       Py_ssize_t nargs) {
    arrow::py::import_pyarrow_wrappers();

    // Ensure there is only one argument
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError,
                        "fetch_frags_metadata takes 1 argument");
        return nullptr;
    }

    // Make sure the argument is a list
    PyObject* arg = args[0];
    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return nullptr;
    }
    std::vector<std::shared_ptr<arrow::dataset::ParquetFileFragment>> fragments;
    for (Py_ssize_t i = 0; i < PyList_Size(arg); i++) {
        PyObject* py_frag = PyList_GET_ITEM(arg, i);
        auto res = arrow::py::unwrap_fragment(py_frag);
        if (!res.ok()) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to unwrap fragment at index %s",
                         std::to_string(i).c_str());
            return nullptr;
        } else {
            fragments.push_back(
                std::static_pointer_cast<arrow::dataset::ParquetFileFragment>(
                    res.ValueOrDie()));
        }
    }
    std::vector<arrow::Future<>> futures;
    for (auto& frag : fragments) {
        arrow::io::IOContext io_context = arrow::io::default_io_context();
        auto res = io_context.executor()->Submit([frag]() {
            auto status = frag->EnsureCompleteMetadata();
            return status;
        });
        if (!res.ok()) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to submit task");
            return nullptr;
        }
        futures.push_back(std::move(res.ValueOrDie()));
    }

    // Again we don't care about the result of the futures, we just want to make
    auto all_future = arrow::AllComplete(futures);
    all_future.Wait();

    auto res = all_future.MoveResult();
    if (!res.ok()) {
        PyErr_Format(PyExc_RuntimeError, "Failed to fetch metadata: %s",
                     res.status().message().c_str());
        return nullptr;
    }

    Py_RETURN_NONE;
}

/**
 * @brief Impl for PyCFunction arrow_cpp.fetch_parquet_frag_row_counts
 * Unwraps a python list of pyarrow ParquetFileFragments into
 * arrow::dataset::ParquetFileFragments and dispatches an async task to fetch
 * there metadata. Returns upon completion of all tasks
 * @param self: unused, will be NULL
 * @param args: list of python arguments, should always be:
 *  - List of pyarrow.dataset.ParquetFileFragments
 *  - A pyarrow.compute.Expression representing the filter
 *  - A pyarrow.Schema representing the datasets / final schema
 * @param nargs: number of args, should always be 3
 * @return PyObject* List of row counts for each fragment
 */
PyObject* fetch_parquet_frag_row_counts(PyObject* self, PyObject* const* args,
                                        Py_ssize_t nargs) {
    arrow::py::import_pyarrow_wrappers();

    // Ensure there are 3 arguments
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError,
                        "fetch_parquet_frag_row_counts takes 3 arguments");
        return nullptr;
    }

    // Make sure the first argument is a list
    PyObject* frag_arg = args[0];
    if (!PyList_Check(frag_arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return nullptr;
    }
    // Convert the first argument into a list of Parquet fragments
    std::vector<std::shared_ptr<arrow::dataset::ParquetFileFragment>> fragments;
    for (Py_ssize_t i = 0; i < PyList_Size(frag_arg); i++) {
        // Returns a borrowed reference, no DECREF
        PyObject* py_frag = PyList_GET_ITEM(frag_arg, i);
        auto res = arrow::py::unwrap_fragment(py_frag);
        if (!res.ok()) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to unwrap fragment at index %s",
                         std::to_string(i).c_str());
            return nullptr;
        }

        fragments.push_back(
            std::static_pointer_cast<arrow::dataset::ParquetFileFragment>(
                res.ValueOrDie()));
    }
    // Convert the second argument into a filter expression
    PyObject* filter_arg = args[1];
    auto filter_res = arrow::py::unwrap_expression(filter_arg);
    if (!filter_res.ok()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to unwrap filter expression");
        return nullptr;
    }
    auto filter = filter_res.ValueOrDie();
    // Convert the third argument into a schema
    PyObject* schema_arg = args[2];
    auto schema_res = arrow::py::unwrap_schema(schema_arg);
    if (!schema_res.ok()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to unwrap schema");
        return nullptr;
    }
    auto schema = schema_res.ValueOrDie();

    // Set up scanner options for each count rows
    auto scan_options = std::make_shared<arrow::dataset::ScanOptions>();
    scan_options->filter = filter;
    scan_options->use_threads = true;
    scan_options->pool = bodo::BufferPool::DefaultPtr();

    std::vector<arrow::Future<int64_t>> futures;
    for (auto& frag : fragments) {
        auto scanner_res = std::make_shared<arrow::dataset::ScannerBuilder>(
                               schema, frag, scan_options)
                               ->Finish();
        if (!scanner_res.ok()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to construct scanner for counting rows");
            return nullptr;
        }
        futures.push_back(scanner_res.ValueOrDie()->CountRowsAsync());
    }

    // Wait for all futures to finish
    auto all_future = arrow::All(futures);
    all_future.Wait();

    auto res = all_future.MoveResult();
    if (!res.ok()) {
        PyErr_Format(PyExc_RuntimeError, "Failed to fetch metadata: %s",
                     res.status().message().c_str());
        return nullptr;
    }
    auto row_counts = res.ValueOrDie();

    auto out_list = PyList_New(row_counts.size());
    for (size_t i = 0; i < row_counts.size(); i++) {
        if (!row_counts[i].ok()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to count rows for a fragment");
            return nullptr;
        }
        PyList_SET_ITEM(out_list, i,
                        PyLong_FromLongLong(row_counts[i].ValueOrDie()));
    }
    return out_list;
}
