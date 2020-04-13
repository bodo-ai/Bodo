// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _PARQUET_READER_H_INCLUDED
#define _PARQUET_READER_H_INCLUDED
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"

void pq_init_reader(const char *file_name,
                    std::shared_ptr<parquet::arrow::FileReader> *a_reader);
int64_t pq_get_size_single_file(std::shared_ptr<parquet::arrow::FileReader>,
                                int64_t column_idx);
int64_t pq_read_single_file(std::shared_ptr<parquet::arrow::FileReader>,
                            int64_t column_idx, uint8_t *out, int out_dtype,
                            uint8_t *out_nulls = nullptr,
                            int64_t null_offset = 0);
int pq_read_parallel_single_file(std::shared_ptr<parquet::arrow::FileReader>,
                                 int64_t column_idx, uint8_t *out_data,
                                 int out_dtype, int64_t start, int64_t count,
                                 uint8_t *out_nulls = nullptr,
                                 int64_t null_offset = 0);
int64_t pq_read_string_single_file(std::shared_ptr<parquet::arrow::FileReader>,
                                   int64_t column_idx, uint32_t **out_offsets,
                                   uint8_t **out_data, uint8_t **out_nulls,
                                   std::vector<uint32_t> *offset_vec = NULL,
                                   std::vector<uint8_t> *data_vec = NULL,
                                   std::vector<bool> *null_vec = NULL);
int pq_read_string_parallel_single_file(
    std::shared_ptr<parquet::arrow::FileReader>, int64_t column_idx,
    uint32_t **out_offsets, uint8_t **out_data, uint8_t **out_nulls,
    int64_t start, int64_t count, std::vector<uint32_t> *offset_vec = NULL,
    std::vector<uint8_t> *data_vec = NULL, std::vector<bool> *null_vec = NULL);

std::pair<int64_t, int64_t> pq_read_list_string_single_file(
    std::shared_ptr<parquet::arrow::FileReader>, int64_t column_idx,
    uint32_t **out_offsets, uint32_t **index_offsets, uint8_t **out_data,
    uint8_t **out_nulls, std::vector<uint32_t> *offset_vec = NULL,
    std::vector<uint32_t> *index_offset_vec = NULL,
    std::vector<uint8_t> *data_vec = NULL, std::vector<bool> *null_vec = NULL);

int64_t pq_read_list_string_parallel_single_file(
    std::shared_ptr<parquet::arrow::FileReader>, int64_t column_idx,
    uint32_t **out_offsets, uint32_t **index_offsets, uint8_t **out_data,
    uint8_t **out_nulls, int64_t start, int64_t count,
    std::vector<uint32_t> *offset_vec = NULL,
    std::vector<uint32_t> *index_offset_vec = NULL,
    std::vector<uint8_t> *data_vec = NULL, std::vector<bool> *null_vec = NULL);

#endif  // _PARQUET_READER_H_INCLUDED
