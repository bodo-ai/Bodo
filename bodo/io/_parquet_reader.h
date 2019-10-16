// Copyright (C) 2019 Bodo Inc.
#ifndef _PARQUET_READER_H_INCLUDED
#define _PARQUET_READER_H_INCLUDED

#include "parquet/arrow/reader.h"
using parquet::arrow::FileReader;

extern "C" {

// parquet type sizes (NOT arrow), parquet/types.h
// boolean, int32, int64, int96, float, double, byte
// XXX assuming int96 is always converted to int64 since it's timestamp
static int pq_type_sizes[] = {1, 4, 8, 8, 4, 8, 1};

void pq_init_reader(const char *file_name,
                    std::shared_ptr<FileReader> *a_reader);
int64_t pq_get_size_single_file(std::shared_ptr<FileReader>,
                                int64_t column_idx);
int64_t pq_read_single_file(std::shared_ptr<FileReader>, int64_t column_idx,
                            uint8_t *out, int out_dtype,
                            uint8_t *out_nulls = nullptr,
                            int64_t null_offset = 0);
int pq_read_parallel_single_file(std::shared_ptr<FileReader>,
                                 int64_t column_idx, uint8_t *out_data,
                                 int out_dtype, int64_t start, int64_t count,
                                 uint8_t *out_nulls = nullptr,
                                 int64_t null_offset = 0);
int64_t pq_read_string_single_file(std::shared_ptr<FileReader>,
                                   int64_t column_idx, uint32_t **out_offsets,
                                   uint8_t **out_data, uint8_t **out_nulls,
                                   std::vector<uint32_t> *offset_vec = NULL,
                                   std::vector<uint8_t> *data_vec = NULL,
                                   std::vector<bool> *null_vec = NULL);
int pq_read_string_parallel_single_file(
    std::shared_ptr<FileReader>, int64_t column_idx, uint32_t **out_offsets,
    uint8_t **out_data, uint8_t **out_nulls, int64_t start, int64_t count,
    std::vector<uint32_t> *offset_vec = NULL,
    std::vector<uint8_t> *data_vec = NULL, std::vector<bool> *null_vec = NULL);

}  // extern "C"

#endif  // _PARQUET_READER_H_INCLUDED
