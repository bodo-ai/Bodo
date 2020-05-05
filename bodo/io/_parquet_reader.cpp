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
#include "_parquet_reader.h"
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

void pack_null_bitmap(uint8_t** out_nulls, std::vector<bool>& null_vec,
                      int64_t n_all_vals);
std::shared_ptr<arrow::DataType> get_arrow_type(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx);
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

#define CHECK(expr, msg)                                              \
    if (!(expr)) {                                                    \
        std::cerr << "Error in parquet reader: " << msg << std::endl; \
    }

#define CHECK_ARROW(expr, msg)                                               \
    if (!(expr.ok())) {                                                      \
        std::cerr << "Error in arrow parquet reader: " << msg << " " << expr \
                  << std::endl;                                              \
    }

typedef void (*s3_opener_t)(const char*,
                            std::shared_ptr<::arrow::io::RandomAccessFile>*);

typedef void (*hdfs_open_file_t)(
    const char*, std::shared_ptr<::arrow::io::HdfsReadableFile>*);

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

int64_t pq_get_size_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx) {
    int64_t nrows = arrow_reader->parquet_reader()->metadata()->num_rows();
    return nrows;
}

int64_t pq_read_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint8_t* out_data, int out_dtype, uint8_t* out_nulls,
    int64_t null_offset) {
    std::shared_ptr<::arrow::ChunkedArray> chunked_array;
    arrow::Status stat = arrow_reader->ReadColumn(column_idx, &chunked_array);
    CHECK_ARROW(stat, "arrow_reader->ReadColumn");
    if (chunked_array == NULL) return 0;
    auto arr = chunked_array->chunk(0);
    int64_t num_values = arr->length();
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, column_idx);

    auto buffers = arr->data()->buffers;
    if (buffers.size() != 2) {
        std::cerr << "invalid parquet number of array buffers" << std::endl;
    }
    const uint8_t* buff = buffers[1]->data();
    const uint8_t* null_bitmap_buff =
        arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();

    copy_data(out_data, buff, 0, num_values, arrow_type, null_bitmap_buff,
              out_dtype);
    copy_nulls(out_nulls, null_bitmap_buff, 0, num_values, null_offset);

    return num_values;
}

int pq_read_parallel_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint8_t* out_data, int out_dtype, int64_t start,
    int64_t count, uint8_t* out_nulls, int64_t null_offset) {
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
        get_arrow_type(arrow_reader, column_idx);
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

        copy_data(out_data + read_rows * dtype_size, buff, rows_to_skip,
                  rows_to_read, arrow_type, null_bitmap_buff, out_dtype);
        copy_nulls(out_nulls, null_bitmap_buff, rows_to_skip, rows_to_read,
                   read_rows + null_offset);

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
        int64_t year, month, day;
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

int64_t pq_read_string_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint32_t** out_offsets, uint8_t** out_data,
    uint8_t** out_nulls, std::vector<uint32_t>* offset_vec,
    std::vector<uint8_t>* data_vec, std::vector<bool>* null_vec) {
    std::shared_ptr<::arrow::ChunkedArray> chunked_arr;
    arrow::Status status = arrow_reader->ReadColumn(column_idx, &chunked_arr);
    CHECK_ARROW(status, "arrow_reader->ReadColumn");
    if (chunked_arr == NULL) return -1;
    auto arr = chunked_arr->chunk(0);
    int64_t num_values = arr->length();
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::STRING)
        std::cerr << "Invalid Parquet string data type" << '\n';

    auto buffers = arr->data()->buffers;
    if (buffers.size() != 3) {
        std::cerr << "invalid parquet string number of array buffers"
                  << std::endl;
    }

    int64_t null_size;
    if (buffers[0])
        null_size = buffers[0]->size();
    else
        null_size = (num_values + 7) >> 3;
    int64_t offsets_size = buffers[1]->size();
    int64_t data_size = buffers[2]->size();

    const uint32_t* offsets_buff = (const uint32_t*)buffers[1]->data();
    const uint8_t* data_buff = buffers[2]->data();
    const uint8_t* null_buff = arr->null_bitmap_data();

    if (offset_vec == NULL) {
        if (data_vec != NULL)
            std::cerr << "parquet read string input error" << '\n';

        *out_offsets = new uint32_t[offsets_size / sizeof(uint32_t)];
        *out_data = new uint8_t[data_size];
        *out_nulls = new uint8_t[null_size];

        set_null_buff(out_nulls, null_buff, null_size);
        memcpy(*out_offsets, offsets_buff, offsets_size);
        memcpy(*out_data, data_buff, data_size);
    } else {
        offset_vec->insert(offset_vec->end(), offsets_buff,
                           offsets_buff + offsets_size / sizeof(uint32_t));
        data_vec->insert(data_vec->end(), data_buff, data_buff + data_size);
        append_bits_to_vec(null_vec, null_buff, null_size, 0, num_values);
    }

    return num_values;
}

int pq_read_string_parallel_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint32_t** out_offsets, uint8_t** out_data,
    uint8_t** out_nulls, int64_t start, int64_t count,
    std::vector<uint32_t>* offset_vec, std::vector<uint8_t>* data_vec,
    std::vector<bool>* null_vec) {
    if (count == 0) {
        if (offset_vec == NULL) {
            *out_offsets = NULL;
            *out_data = NULL;
        }
        return 0;
    }

    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::STRING)
        std::cerr << "Invalid Parquet string data type" << '\n';

    if (offset_vec == NULL) {
        *out_offsets = new uint32_t[count + 1];
        data_vec = new std::vector<uint8_t>();
        null_vec = new std::vector<bool>();
    }

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

    uint32_t curr_offset = 0;

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
            if (offset_vec == NULL)
                (*out_offsets)[read_rows + i] = curr_offset;
            else
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

    if (offset_vec == NULL) {
        (*out_offsets)[count] = curr_offset;
        *out_data = new uint8_t[curr_offset];
        memcpy(*out_data, data_vec->data(), curr_offset);
        pack_null_bitmap(out_nulls, *null_vec, count);
        delete data_vec;
        delete null_vec;
    } else
        offset_vec->push_back(curr_offset);

    return 0;
}

std::pair<int64_t, int64_t> pq_read_list_string_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint32_t** out_offsets, uint32_t** index_offsets,
    uint8_t** out_data, uint8_t** out_nulls, std::vector<uint32_t>* offset_vec,
    std::vector<uint32_t>* index_offset_vec, std::vector<uint8_t>* data_vec,
    std::vector<bool>* null_vec) {
    std::shared_ptr<::arrow::ChunkedArray> chunked_arr;
    arrow::Status status;
    status = arrow_reader->ReadColumn(column_idx, &chunked_arr);
    CHECK_ARROW(status, "arrow_reader->ReadColumn");
    if (chunked_arr == NULL) return {-1, -1};
    auto arr = chunked_arr->chunk(0);
    int64_t num_values = arr->length();
    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::LIST)
        std::cerr << "Invalid Parquet list data type" << '\n';

    std::vector<std::shared_ptr<arrow::ArrayData>>& child_data =
        arr->data()->child_data;
    if (child_data.size() != 1) {
        std::cerr << "arrow list array of strings must contain a child array"
                  << std::endl;
        return {-1, -1};
    }

    auto parent_buffers = arr->data()->buffers;
    if (parent_buffers.size() != 2) {
        std::cerr
            << "invalid parquet list string number of parent array buffers "
            << parent_buffers.size() << std::endl;
        return {-1, -1};
    }
    int64_t parent_offsets_size = parent_buffers[1]->size();
    int64_t parent_null_size;
    if (parent_buffers[0]) {
        parent_null_size = parent_buffers[0]->size();
    } else {
        parent_null_size = (num_values+ 7) >> 3;
    }

    auto child_buffers = child_data[0]->buffers;
    if (child_buffers.size() != 3) {
        std::cerr << "invalid parquet string number of array buffers "
                  << child_buffers.size() << std::endl;
        return {-1, -1};
    }
    int64_t num_strings = child_data[0]->length;
    int64_t offsets_size = child_buffers[1]->size();
    int64_t data_size = child_buffers[2]->size();
    const uint32_t* index_offsets_buff =
        (const uint32_t*)parent_buffers[1]->data();
    const uint32_t* offsets_buff = (const uint32_t*)child_buffers[1]->data();
    const uint8_t* data_buff = child_buffers[2]->data();
    const uint8_t* null_buff = arr->null_bitmap_data();

    if (offset_vec == NULL) {
        if (data_vec != NULL)
            std::cerr << "parquet read string input error" << '\n';

        *out_offsets = new uint32_t[offsets_size / sizeof(uint32_t)];
        *index_offsets = new uint32_t[parent_offsets_size / sizeof(uint32_t)];
        *out_data = new uint8_t[data_size];
        *out_nulls = new uint8_t[parent_null_size];

        set_null_buff(out_nulls, null_buff, parent_null_size);
        memcpy(*out_offsets, offsets_buff, offsets_size);
        memcpy(*index_offsets, index_offsets_buff, parent_offsets_size);
        memcpy(*out_data, data_buff, data_size);
    } else {
        offset_vec->insert(offset_vec->end(), offsets_buff,
                           offsets_buff + offsets_size / sizeof(uint32_t));
        index_offset_vec->insert(
            index_offset_vec->end(), index_offsets_buff,
            index_offsets_buff + parent_offsets_size / sizeof(uint32_t));
        // we are reading a dataset split into multiple parquet files. some
        // files might not have any string data at all (because all strings are
        // empty or there are nan lists). In that case, we don't want to insert
        // nulls in the middle of the data buffer
        if ((offsets_size / sizeof(uint32_t) > 0) &&
            (offsets_buff[offsets_size / sizeof(uint32_t) - 1] > 0))
            data_vec->insert(data_vec->end(), data_buff, data_buff + data_size);
        append_bits_to_vec(null_vec, null_buff, parent_null_size, 0,
                           num_values);
    }

    return {num_values, num_strings};
}

int64_t pq_read_list_string_parallel_single_file(
    std::shared_ptr<parquet::arrow::FileReader> arrow_reader,
    int64_t column_idx, uint32_t** out_offsets, uint32_t** out_index_offsets,
    uint8_t** out_data, uint8_t** out_nulls, int64_t start, int64_t count,
    std::vector<uint32_t>* offset_vec, std::vector<uint32_t>* index_offset_vec,
    std::vector<uint8_t>* data_vec, std::vector<bool>* null_vec) {
    if (count == 0) {
        if (offset_vec == NULL) {
            *out_offsets = NULL;
            *out_index_offsets = NULL;
            *out_data = NULL;
        }
        return 0;
    }

    std::shared_ptr<arrow::DataType> arrow_type =
        get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::LIST)
        std::cerr << "Invalid Parquet string data type" << '\n';
    // TODO check that list of string

    bool output_vectors = true;
    if (offset_vec == NULL) {
        output_vectors = false;
        *out_index_offsets = new uint32_t[count + 1];
        // don't know how many strings there are
        offset_vec = new std::vector<uint32_t>();
        data_vec = new std::vector<uint8_t>();
        null_vec = new std::vector<bool>();
    }

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

    uint32_t curr_offset = 0;
    uint32_t curr_index_offset = 0;
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
            if (!output_vectors) {
                (*out_index_offsets)[read_rows + i] = curr_index_offset;
            } else {
                index_offset_vec->push_back(curr_index_offset);
            }
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

    if (!output_vectors) {
        offset_vec->push_back(curr_offset);
        *out_offsets = new uint32_t[offset_vec->size()];
        memcpy(*out_offsets, offset_vec->data(),
               offset_vec->size() * sizeof(uint32_t));

        (*out_index_offsets)[count] = curr_index_offset;
        *out_data = new uint8_t[curr_offset];
        memcpy(*out_data, data_vec->data(), curr_offset);
        pack_null_bitmap(out_nulls, *null_vec, count);
        delete offset_vec;
        delete data_vec;
        delete null_vec;
    } else {
        offset_vec->push_back(curr_offset);
        index_offset_vec->push_back(curr_index_offset);
    }

    return num_strings;
}

void pq_init_reader(const char* file_name,
                    std::shared_ptr<parquet::arrow::FileReader>* a_reader) {
    PyObject* f_mod;
    std::string f_name(file_name);
    auto pool = ::arrow::default_memory_pool();
    arrow::Status status;

    // HDFS if starts with hdfs://
    if (f_name.find("hdfs://") == 0) {
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        std::shared_ptr<::arrow::io::HdfsReadableFile> file;
        // get hdfs opener function
        import_fs_module(Bodo_Fs::hdfs, "parquet", f_mod);
        PyObject* func_obj = PyObject_GetAttrString(f_mod, "hdfs_open_file");
        CHECK(func_obj, "getting hdfs_open_file func_obj failed");
        hdfs_open_file_t hdfs_open_file =
            (hdfs_open_file_t)PyNumber_AsSsize_t(func_obj, NULL);
        // open Parquet file
        hdfs_open_file(file_name, &file);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (f_name.find("s3://") == 0) {
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        // remove s3://
        f_name = f_name.substr(strlen("s3://"));
        // get s3 opener function
        import_fs_module(Bodo_Fs::s3, "parquet", f_mod);
        PyObject* func_obj = PyObject_GetAttrString(f_mod, "s3_open_file");
        CHECK(func_obj, "getting s3_open_file func_obj failed");
        s3_opener_t s3_open_file =
            (s3_opener_t)PyNumber_AsSsize_t(func_obj, NULL);
        // open Parquet file
        s3_open_file(f_name.c_str(), &file);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else  // regular file system
    {
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
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
    int64_t column_idx) {
    // TODO: error checking
    // GetSchema supported as of arrow version=0.15.1
    std::shared_ptr<::arrow::Schema> col_schema;
    arrow::Status status = arrow_reader->GetSchema(&col_schema);
    CHECK_ARROW(status, "arrow_reader->GetSchema");
    return col_schema->field(column_idx)->type();
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

void pack_null_bitmap(uint8_t** out_nulls, std::vector<bool>& null_vec,
                      int64_t n_all_vals) {
    assert(null_vec.size() > 0);
    int64_t n_bytes = (null_vec.size() + sizeof(uint8_t) - 1) / sizeof(uint8_t);
    *out_nulls = new uint8_t[n_bytes];
    memset(*out_nulls, 0, n_bytes);
    for (int64_t i = 0; i < n_all_vals; i++) {
        if (null_vec[i]) ::arrow::BitUtil::SetBit(*out_nulls, i);
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
