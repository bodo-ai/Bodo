// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "../libs/_bodo_common.h"
#include "../libs/_datetime_ext.h"
#include "_fs_io.h"
#include "_fsspec_reader.h"
#include "_hdfs_reader.h"
#include "_parquet_reader.h"
#include "_s3_reader.h"
#include "arrow/array.h"
#include "arrow/io/hdfs.h"
#include "arrow/python/arrow_to_pandas.h"
#include "arrow/python/pyarrow.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"

using arrow::Type;
using parquet::ParquetFileReader;

void pack_null_bitmap(uint8_t* out_nulls, std::vector<bool>& null_vec,
                      int64_t n_all_vals);
std::shared_ptr<arrow::DataType> get_arrow_type(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx);
bool arrowBodoTypesEqual(std::shared_ptr<arrow::DataType> arrow_type,
                         Bodo_CTypes::CTypeEnum pq_type);
inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                      int64_t rows_to_skip, int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff, int out_dtype);

template <typename T, int64_t SHIFT>
inline void convertArrowToDT64(const uint8_t* buff, uint8_t* out_data,
                               int64_t rows_to_skip, int64_t rows_to_read);
void append_bits_to_vec(std::vector<bool>* null_vec, const uint8_t* null_buff,
                        int64_t null_size, int64_t offset, int64_t num_values);

void set_null_buff(uint8_t** out_nulls, const uint8_t* null_buff,
                   int64_t null_size);

#define CHECK(expr, msg)                                                      \
    if (!(expr)) {                                                            \
        std::string err_msg = std::string("Error in parquet reader: ") + msg; \
        throw std::runtime_error(err_msg);                                    \
    }

#define CHECK_ARROW(expr, msg)                                                 \
    if (!(expr.ok())) {                                                        \
        std::string err_msg = std::string("Error in arrow parquet reader: ") + \
                              msg + " " + expr.ToString();                     \
        throw std::runtime_error(err_msg);                                     \
    }

#define PQ_DT64_TYPE 3  // using INT96 value as dt64, TODO: refactor
#define kNanosecondsInDay 86400000000000LL  // TODO: reuse from type_traits.h

inline void copy_nulls(uint8_t* out_nulls, const uint8_t* null_bitmap_buff,
                       int64_t skip, int64_t num_values, int64_t null_offset) {
    if (out_nulls != nullptr) {
        if (null_bitmap_buff == nullptr) {
            for (size_t i = 0; i < size_t(num_values); i++) {
                ::arrow::BitUtil::SetBit(out_nulls, null_offset + i);
            }
        } else {
            for (size_t i = 0; i < size_t(num_values); i++) {
                auto bit = ::arrow::BitUtil::GetBit(null_bitmap_buff, skip + i);
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
        auto bit = ::arrow::BitUtil::GetBit(null_bitmap_buff, skip + i);
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

int64_t pq_get_size_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx) {
    int64_t nrows = arrow_reader->parquet_reader()->metadata()->num_rows();
    return nrows;
}

/**
 * Read a set of rows from column of basic type and return the associated
 * vector of data and nullbits that are needed to describe the data.
 * @param arrow_reader : Arrow reader for a specific parquet file (already
 * opened).
 * @param column_idx : index of column to read
 * @param[out] out_data : vector of data (elements of type out_dtype)
 * @param[out] out_dtype : dtype of elements
 * @param start : starting row in column to read
 * @param count : num rows from start to read
 * @param[out] out_nulls : vector specifying if rows contain nulls or not
 * @param null_offset : offset to start writing into out_nulls
 */
int pq_read_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx, int64_t column_idx, uint8_t* out_data,
    int out_dtype, int64_t start, int64_t count, uint8_t* out_nulls,
    int64_t null_offset, int is_categorical) {
    if (count == 0) {
        return 0;
    }

    int64_t n_row_groups =
        arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;  // number of rows skipped so far
                               // (start-skipped_rows is rows left to skip to
                               // reach starting point)
    int64_t read_rows = 0;     // number of rows read so far

    auto rg_metadata =
        arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, real_column_idx);
    int dtype_size = numpy_item_size[out_dtype];

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group) {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(
            row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    while (read_rows < count) {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow::Status status =
            arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        CHECK_ARROW(status, "arrow_reader->ReadRowGroup");
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        if (chunked_arr->num_chunks() != 1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);

        // read the categorical codes for the DictionaryArray case
        // category values are read during typing already
        if (is_categorical) {
            arr =
                reinterpret_cast<arrow::DictionaryArray*>(arr.get())->indices();
        }

        auto buffers = arr->data()->buffers;
        if (buffers.size() != 2) {
            std::cerr << "invalid parquet number of array buffers" << std::endl;
        }
        const uint8_t* buff = buffers[1]->data();
        const uint8_t* null_bitmap_buff =
            arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();
        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read =
            std::min(count - read_rows, nrows_in_group - rows_to_skip);

        uint8_t* data_ptr = out_data + read_rows * dtype_size;
        copy_data(data_ptr, buff, rows_to_skip, rows_to_read, arrow_type,
                  null_bitmap_buff, out_dtype);
        copy_nulls(out_nulls, null_bitmap_buff, rows_to_skip, rows_to_read,
                   read_rows + null_offset);

        // Arrow uses nullable arrays for categorical codes, but we use
        // regular numpy arrays and store -1 for null, so nulls have to be
        // set in data array
        if (is_categorical && arr->null_count() != 0) {
            copy_nulls_categorical(data_ptr, null_bitmap_buff, rows_to_skip,
                                   rows_to_read, out_dtype);
        }

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups) {
            auto rg_metadata =
                arrow_reader->parquet_reader()->metadata()->RowGroup(
                    row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        } else
            break;
    }
    if (read_rows != count) std::cerr << "parquet read incomplete" << '\n';
    return 0;
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

template <typename T_in, typename T_out>
inline void copy_data_cast(uint8_t* out_data, const uint8_t* buff,
                           int64_t rows_to_skip, int64_t rows_to_read,
                           std::shared_ptr<arrow::DataType> arrow_type,
                           int out_dtype) {
    T_out* out_data_cast = (T_out*)out_data;
    T_in* in_data_cast = (T_in*)buff;
    for (int64_t i = 0; i < rows_to_read; i++) {
        out_data_cast[i] = (T_out)in_data_cast[rows_to_skip + i];
    }
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
                std::cerr << "Invalid datetime timeunit" << out_dtype << " "
                          << arrow_type << std::endl;
            }
        } else {
            //
            std::cerr << "Invalid datetime conversion" << out_dtype << " "
                      << arrow_type << std::endl;
        }
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "parquet read: invalid dtype conversion");
    }
}

inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                      int64_t rows_to_skip, int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff, int out_dtype) {
    // unpack booleans from bits
    if (out_dtype == Bodo_CTypes::_BOOL) {
        if (arrow_type->id() != Type::BOOL)
            std::cerr << "boolean type error" << '\n';

        for (int64_t i = 0; i < rows_to_read; i++) {
            out_data[i] =
                (uint8_t)::arrow::BitUtil::GetBit(buff, i + rows_to_skip);
        }
        return;
    }

    if (arrowBodoTypesEqual(arrow_type, (Bodo_CTypes::CTypeEnum)out_dtype)) {
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
            if (!::arrow::BitUtil::GetBit(null_bitmap_buff, i + rows_to_skip)) {
                // TODO: use NPY_NAN
                double_data[i] = std::nan("");
            }
        }
    }
    // set NaNs for float values
    if (null_bitmap_buff != nullptr && out_dtype == Bodo_CTypes::FLOAT32) {
        float* float_data = (float*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++) {
            if (!::arrow::BitUtil::GetBit(null_bitmap_buff, i + rows_to_skip)) {
                // TODO: use NPY_NAN
                float_data[i] = std::nanf("");
            }
        }
    }
    return;
}

/**
 * Read a set of rows from column of strings and return the associated
 * vector of offsets, data and nullbits that are needed to describe string data.
 * @param arrow_reader : Arrow reader for a specific parquet file (already
 * opened).
 * @param column_idx : index of column to read
 * @param start : starting row in column to read
 * @param count : num rows from start to read
 * @param[out] offset_vec : vector of offsets (where each string starts in data)
 * @param[out] data_vec : vector of data (elements of type out_dtype)
 * @param[out] null_vec : vector specifying if rows contain nulls or not
 */
int pq_read_string_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx, int64_t column_idx, int64_t start, int64_t count,
    std::vector<offset_t>* offset_vec, std::vector<uint8_t>* data_vec,
    std::vector<bool>* null_vec) {
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, real_column_idx);
    if (arrow_type->id() != Type::STRING)
        std::cerr << "Invalid Parquet string data type: "
                  << arrow_type->ToString() << '\n';

    int64_t n_row_groups =
        arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;

    auto rg_metadata =
        arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group) {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(
            row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    offset_t curr_offset = 0;

    /* ------- read offsets and data ------ */
    while (read_rows < count) {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow::Status status =
            arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        CHECK_ARROW(status, "arrow_reader->ReadRowGroup");
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        if (chunked_arr->num_chunks() != 1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        auto buffers = arr->data()->buffers;
        if (buffers.size() != 3) {
            std::cerr << "invalid parquet string number of array buffers"
                      << std::endl;
        }

        int64_t null_size = -1;
        if (buffers[0]) null_size = buffers[0]->size();
        // TODO check type of Arrow offsets buffer
        const uint32_t* offsets_buff = (const uint32_t*)buffers[1]->data();
        const uint8_t* data_buff = buffers[2]->data();
        const uint8_t* null_buff = arr->null_bitmap_data();

        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read =
            std::min(count - read_rows, nrows_in_group - rows_to_skip);

        for (int64_t i = 0; i < rows_to_read; i++) {
            uint32_t str_size = offsets_buff[rows_to_skip + i + 1] -
                                offsets_buff[rows_to_skip + i];
            offset_vec->push_back(curr_offset);
            curr_offset += str_size;
        }

        int data_size = offsets_buff[rows_to_skip + rows_to_read] -
                        offsets_buff[rows_to_skip];

        data_vec->insert(data_vec->end(),
                         data_buff + offsets_buff[rows_to_skip],
                         data_buff + offsets_buff[rows_to_skip] + data_size);

        append_bits_to_vec(null_vec, null_buff, null_size, rows_to_skip,
                           rows_to_read);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups) {
            auto rg_metadata =
                arrow_reader->parquet_reader()->metadata()->RowGroup(
                    row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        } else
            break;
    }
    if (read_rows != count) std::cerr << "parquet read incomplete" << '\n';

    offset_vec->push_back(curr_offset);
    return 0;
}

/**
 * Read a set of rows from column of list of string and return the associated
 * vector of offsets, index offsets, data and nullbits that are needed to
 * describe list of string data.
 * @param arrow_reader : Arrow reader for a specific parquet file (already
 * opened).
 * @param column_idx : index of column to read
 * @param start : starting row in column to read
 * @param count : num rows from start to read
 * @param[out] offset_vec : vector of offsets (where each string starts in data)
 * @param[out] index_offset_vec : vector of offsets (where each list starts in
 * data)
 * @param[out] data_vec : vector of data (elements of type out_dtype)
 * @param[out] null_vec : vector specifying if rows contain nulls or not
 */
int64_t pq_read_list_string_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx, int64_t column_idx, int64_t start, int64_t count,
    std::vector<offset_t>* offset_vec, std::vector<offset_t>* index_offset_vec,
    std::vector<uint8_t>* data_vec, std::vector<bool>* null_vec) {
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, real_column_idx);
    if (arrow_type->id() != Type::LIST)
        std::cerr << "Invalid Parquet list data type" << '\n';
    // TODO check that list of string

    int64_t n_row_groups =
        arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices = {static_cast<int>(column_idx)};

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;

    auto rg_metadata =
        arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group) {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(
            row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    offset_t curr_offset = 0;
    offset_t curr_index_offset = 0;
    int64_t num_strings = 0;

    /* ------- read offsets and data ------ */
    while (read_rows < count) {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow::Status status =
            arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        CHECK_ARROW(status, "arrow_reader->ReadRowGroup");
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        if (chunked_arr->num_chunks() != 1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        auto parent_buffers = arr->data()->buffers;
        if (parent_buffers.size() != 2) {
            std::cerr
                << "invalid parquet list string number of parent array buffers "
                << parent_buffers.size() << std::endl;
        }

        std::vector<std::shared_ptr<arrow::ArrayData>>& child_data =
            arr->data()->child_data;
        if (child_data.size() != 1) {
            std::cerr
                << "arrow list array of strings must contain a child array"
                << std::endl;
        }

        auto child_buffers = child_data[0]->buffers;
        if (child_buffers.size() != 3) {
            std::cerr << "invalid parquet string number of array buffers "
                      << child_buffers.size() << std::endl;
        }

        const uint32_t* parent_offsets_buf =
            (const uint32_t*)parent_buffers[1]->data();
        const uint32_t* offsets_buff =
            (const uint32_t*)child_buffers[1]->data();
        const uint8_t* data_buff = child_buffers[2]->data();
        const uint8_t* null_buff = arr->null_bitmap_data();

        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read =
            std::min(count - read_rows, nrows_in_group - rows_to_skip);

        int64_t parent_null_size;
        if (parent_buffers[0]) {
            parent_null_size = parent_buffers[0]->size();
        } else {
            parent_null_size = (rows_to_read + 7) >> 3;
        }

        int data_size = 0;
        for (int64_t i = 0; i < rows_to_read; i++) {
            uint32_t str_offset_start = parent_offsets_buf[rows_to_skip + i];
            uint32_t str_offset_end = parent_offsets_buf[rows_to_skip + i + 1];
            data_size +=
                offsets_buff[str_offset_end] - offsets_buff[str_offset_start];
            for (int64_t j = str_offset_start; j < str_offset_end; j++) {
                offset_vec->push_back(curr_offset);
                uint32_t str_size = offsets_buff[j + 1] - offsets_buff[j];
                curr_offset += str_size;
                num_strings++;
            }
            index_offset_vec->push_back(curr_index_offset);
            curr_index_offset += str_offset_end - str_offset_start;
        }

        uint32_t data_start = offsets_buff[parent_offsets_buf[rows_to_skip]];
        data_vec->insert(data_vec->end(), data_buff + data_start,
                         data_buff + data_start + data_size);
        append_bits_to_vec(null_vec, null_buff, parent_null_size, rows_to_skip,
                           rows_to_read);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups) {
            auto rg_metadata =
                arrow_reader->parquet_reader()->metadata()->RowGroup(
                    row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        } else
            break;
    }
    if (read_rows != count) std::cerr << "parquet read incomplete" << '\n';

    offset_vec->push_back(curr_offset);
    index_offset_vec->push_back(curr_index_offset);
    return num_strings;
}

/**
 * Read a set of rows from column of list of item and return the associated
 * vector of offsets, data and nullbits that are needed to describe list of
 * item data.
 * @param arrow_reader : Arrow reader for a specific parquet file (already
 * opened).
 * @param column_idx : index of column to read
 * @param out_dtype : dtype of item (for list of item)
 * @param start : starting row in column to read
 * @param count : num rows from start to read
 * @param[out] offset_vec : vector of offsets (where each list starts in data)
 * @param[out] data_vec : vector of data (elements of type out_dtype)
 * @param[out] null_vec : vector specifying if rows contain nulls or not
 */
int64_t pq_read_array_item_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx, int64_t column_idx, int out_dtype, int64_t start,
    int64_t count, std::vector<offset_t>* offset_vec,
    std::vector<uint8_t>* data_vec, std::vector<bool>* null_vec) {
    Bodo_CTypes::CTypeEnum out_dtype_ct = Bodo_CTypes::CTypeEnum(out_dtype);
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, real_column_idx);
    if (arrow_type->id() != Type::LIST)
        std::cerr << "Invalid Parquet list data type" << '\n';

    int64_t n_row_groups =
        arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices = {static_cast<int>(column_idx)};

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;
    int dtype_size = numpy_item_size[out_dtype];
    std::vector<char> vectNaN = RetrieveNaNentry(out_dtype_ct);

    auto rg_metadata =
        arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group) {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(
            row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    offset_t curr_offset = 0;
    int64_t num_items = 0;

    /* ------- read offsets and data ------ */
    while (read_rows < count) {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow::Status status =
            arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        CHECK_ARROW(status, "arrow_reader->ReadRowGroup");
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        if (chunked_arr->num_chunks() != 1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        auto parent_buffers = arr->data()->buffers;
        if (parent_buffers.size() != 2) {
            std::cerr
                << "invalid parquet list item number of parent array buffers "
                << parent_buffers.size() << std::endl;
        }

        std::vector<std::shared_ptr<arrow::ArrayData>>& child_data =
            arr->data()->child_data;
        if (child_data.size() != 1) {
            std::cerr << "arrow list array of item must contain a child array"
                      << std::endl;
        }

        std::vector<std::shared_ptr<arrow::Buffer>> child_buffers =
            child_data[0]->buffers;
        if (child_buffers.size() != 2) {
            std::cerr << "invalid parquet item number of array buffers "
                      << child_buffers.size() << std::endl;
        }

        const uint32_t* offsets_buff =
            (const uint32_t*)parent_buffers[1]->data();
        const uint8_t* data_null_buff = child_buffers[0]->data();
        const uint8_t* data_buff = child_buffers[1]->data();
        const uint8_t* null_buff = arr->null_bitmap_data();

        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read =
            std::min(count - read_rows, nrows_in_group - rows_to_skip);

        int64_t parent_null_size;
        if (parent_buffers[0]) {
            parent_null_size = parent_buffers[0]->size();
        } else {
            parent_null_size = (rows_to_read + 7) >> 3;
        }

        for (int64_t i = 0; i < rows_to_read; i++) {
            uint32_t list_size = offsets_buff[rows_to_skip + i + 1] -
                                 offsets_buff[rows_to_skip + i];
            offset_vec->push_back(curr_offset);
            curr_offset += list_size;
        }

        int data_size = offsets_buff[rows_to_skip + rows_to_read] -
                        offsets_buff[rows_to_skip];
        // We convert the missing entry floating point values to NAN.
        // Howver if any of following condition is satisfied:
        // ---data_null_buff is null pointer
        // ---data is not a float
        // Then we return directly the values.
        if (data_null_buff == nullptr ||
            (out_dtype_ct != Bodo_CTypes::FLOAT32 &&
             out_dtype_ct != Bodo_CTypes::FLOAT64)) {
            data_vec->insert(
                data_vec->end(),
                data_buff + offsets_buff[rows_to_skip] * dtype_size,
                data_buff +
                    (offsets_buff[rows_to_skip] + data_size) * dtype_size);
        } else {
            for (int i = 0; i < rows_to_read; i++) {
                std::vector<uint8_t> V(dtype_size);
                for (uint32_t j = offsets_buff[rows_to_skip + i];
                     j < offsets_buff[rows_to_skip + i + 1]; j++) {
                    bool bit = ::arrow::BitUtil::GetBit(data_null_buff, j);
                    if (!bit) {
                        memcpy(V.data(), vectNaN.data(), dtype_size);
                    } else {
                        memcpy(V.data(), data_buff + j * dtype_size,
                               dtype_size);
                    }
                    data_vec->insert(data_vec->end(), V.begin(), V.end());
                }
            }
        }
        append_bits_to_vec(null_vec, null_buff, parent_null_size, rows_to_skip,
                           rows_to_read);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups) {
            auto rg_metadata =
                arrow_reader->parquet_reader()->metadata()->RowGroup(
                    row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        } else
            break;
    }
    if (read_rows != count) std::cerr << "parquet read incomplete" << '\n';

    offset_vec->push_back(curr_offset);
    return num_items;
}

void pq_read_arrow_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    const std::vector<int>& column_indices, int64_t start, int64_t count,
    arrow::ArrayVector& parts) {
    int64_t n_row_groups =
        arrow_reader->parquet_reader()->metadata()->num_row_groups();

    int row_group_index = 0;
    int64_t skipped_rows = 0;  // number of rows skipped so far
                               // (start-skipped_rows is rows left to skip to
                               // reach starting point)
    int64_t read_rows = 0;     // number of rows read so far for this reader

    auto rg_metadata =
        arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group =
        rg_metadata->ColumnChunk(column_indices[0])->num_values();

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group) {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(
            row_group_index);
        nrows_in_group =
            rg_metadata->ColumnChunk(column_indices[0])->num_values();
    }

    while (read_rows < count) {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow::Status status =
            arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        CHECK_ARROW(status, "arrow_reader->ReadRowGroup");
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        if (chunked_arr->num_chunks() != 1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read =
            std::min(count - read_rows, nrows_in_group - rows_to_skip);

        parts.push_back(arr->Slice(rows_to_skip, rows_to_read));

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups) {
            auto rg_metadata =
                arrow_reader->parquet_reader()->metadata()->RowGroup(
                    row_group_index);
            nrows_in_group =
                rg_metadata->ColumnChunk(column_indices[0])->num_values();
        } else
            break;
    }
    if (read_rows != count) std::cerr << "parquet read incomplete" << '\n';
}

void pq_init_reader(const char* file_name,
                    std::shared_ptr<parquet::arrow::FileReader>* a_reader,
                    const char* bucket_region, bool s3fs_anon) {
    PyObject* f_mod;
    std::string f_name(file_name);
    auto pool = ::arrow::default_memory_pool();
    arrow::Status status;

    size_t colon = f_name.find_first_of(':');
    std::string protocol = f_name.substr(0, colon);
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    // HDFS if starts with hdfs://
    if (f_name.find("hdfs://") == 0 || f_name.find("abfs://") == 0 ||
        f_name.find("abfss://") == 0) {
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        // open Parquet file
        hdfs_open_file(file_name, &file);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    } else if (f_name.find("s3://") == 0) {
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        // remove s3://
        f_name = f_name.substr(strlen("s3://"));
        // get s3 opener function
        // open Parquet file
        s3_open_file(f_name.c_str(), &file, bucket_region, s3fs_anon);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    } else if (protocol == "gcs" || pyfs.count(protocol)) {
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        fsspec_open_file(f_name, protocol, &file);
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    } else  // regular file system
    {
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::OpenFile(f_name, false), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    }
    return;
}

// get type as enum values defined in arrow/cpp/src/arrow/type.h
// TODO: handle more complex types
std::shared_ptr<arrow::DataType> get_arrow_type(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t real_column_idx) {
    // TODO: error checking
    // GetSchema supported as of arrow version=0.15.1
    std::shared_ptr<::arrow::Schema> col_schema;
    arrow::Status status = arrow_reader->GetSchema(&col_schema);
    CHECK_ARROW(status, "arrow_reader->GetSchema");
    return col_schema->field(real_column_idx)->type();
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
    // TODO: add timestamp[ns]

    // Dictionary array's codes are always read into proper integer array type,
    // so buffer data types are the same
    if (arrow_type->id() == Type::DICTIONARY) return true;
    return false;
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

void append_bits_to_vec(std::vector<bool>* null_vec, const uint8_t* null_buff,
                        int64_t null_size, int64_t offset, int64_t num_values) {
    if (null_buff != nullptr && null_size > 0) {
        // to make packing portions of data easier, add data to vector in
        // unpacked format then repack
        for (int64_t i = offset; i < offset + num_values; i++) {
            bool val = ::arrow::BitUtil::GetBit(null_buff, i);
            null_vec->push_back(val);
        }
        // null_vec->insert(null_vec->end(), null_buff, null_buff+null_size);
    } else {
        // arrow returns nullptr if noen of the data is null
        for (int64_t i = 0; i < num_values; i++) {
            null_vec->push_back(1);
        }
    }
}

void pack_null_bitmap(uint8_t* out_nulls, std::vector<bool>& null_vec,
                      int64_t n_all_vals) {
    assert(null_vec.size() > 0);
    int64_t n_bytes = (null_vec.size() + 7) >> 3;
    memset(out_nulls, 0, n_bytes);
    for (int64_t i = 0; i < n_all_vals; i++) {
        if (null_vec[i]) ::arrow::BitUtil::SetBit(out_nulls, i);
    }
}

void set_null_buff(uint8_t** out_nulls, const uint8_t* null_buff,
                   int64_t null_size) {
    if (null_buff != nullptr && null_size > 0) {
        memcpy(*out_nulls, null_buff, null_size);
    } else {
        memset(*out_nulls, 0xFF, null_size);
    }
}

#undef CHECK
