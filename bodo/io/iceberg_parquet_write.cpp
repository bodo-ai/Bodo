// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Functions to write Bodo arrays to Iceberg table (parquet format)

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "parquet_write.h"

/**
 * Write the Bodo table (the chunk in this process) to a parquet file
 * as part of an Iceberg table
 * @param fname name of the parquet file to write
 * @param table_data_loc path to the table location (i.e. the data folder for
 * the table)
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param[out] record_count Number of records in this file
 * @param[out] file_size_in_bytes Size of the file in bytes
 */
void iceberg_pq_write(const char *fname, const char *table_data_loc,
                      const table_info *table, const array_info *col_names_arr,
                      const char *compression, bool is_parallel,
                      const char *bucket_region, int64_t row_group_size,
                      char *iceberg_metadata, int64_t *record_count,
                      int64_t *file_size_in_bytes) {
    if (!is_parallel)
        throw std::runtime_error(
            "iceberg_pq_write not implemented in sequential mode");
    std::unordered_map<std::string, std::string> md = {
        {"iceberg.schema", std::string(iceberg_metadata)}};

    // For Iceberg, all timestamp data needs to be written
    // as microseconds, so that's the type we
    // specify. `pq_write` will convert the nanoseconds to
    // microseconds during `bodo_array_to_arrow`.
    // See https://iceberg.apache.org/spec/#primitive-types,
    // https://iceberg.apache.org/spec/#parquet.
    // We've also made the decision to always
    // write the `timestamptz` type when writing
    // Iceberg data, similar to Spark.
    // The underlying already is in UTC already
    // for timezone aware types, and for timezone
    // naive, it won't matter.
    *file_size_in_bytes = pq_write(
        table_data_loc, table, col_names_arr, nullptr, false, "", compression,
        is_parallel, false, 0, 0, 0, "", bucket_region, row_group_size, "", md,
        std::string(fname), "UTC", arrow::TimeUnit::MICRO);
    *record_count = table->nrows();
}

void iceberg_pq_write_py_entry(
    const char *fname, const char *table_data_loc, const table_info *table,
    const array_info *col_names_arr, const char *compression, bool is_parallel,
    const char *bucket_region, int64_t row_group_size, char *iceberg_metadata,
    int64_t *record_count, int64_t *file_size_in_bytes) {
    try {
        iceberg_pq_write(fname, table_data_loc, table, col_names_arr,
                         compression, is_parallel, bucket_region,
                         row_group_size, iceberg_metadata, record_count,
                         file_size_in_bytes);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}
