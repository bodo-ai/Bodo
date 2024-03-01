// Copyright (C) 2019 Bodo Inc. All rights reserved.

// Functions to write Bodo arrays to parquet

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "../libs/_bodo_common.h"
#include "_fs_io.h"

/**
 * Struct used during pq_write_partitioned_py_entry and iceberg_pq_write to
 * store the information for a partition that this process is going to write:
 * the file path of the parquet file for this partition (e.g.
 * sales_date=2020-01-01/part-00.parquet), and the rows in the table that
 * correspond to this partition.
 * In case of Iceberg, we also store a Python tuple with the information
 * about the file that gets written for each partition.
 */
struct partition_write_info {
    std::string fpath;          // path and filename
    std::vector<int64_t> rows;  // rows in this partition

    // Iceberg only

    // Python tuple consisting of (file_name, number_of_records, file_size,
    // *partition_values)
    PyObject *iceberg_file_info_py;
};

Bodo_Fs::FsEnum filesystem_type(const char *fname);

/**
 * Write the Bodo table (the chunk in this process) to a parquet file.
 * @param _path_name path of output file or directory
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param index array containing the table index
 * @param write_index true if we need to write index passed in 'index', false
 * otherwise
 * @param metadata string containing table metadata
 * @param is_parallel true if the table is part of a distributed table (in this
 *        case, this process writes a file named "part-000X.parquet" where X is
 *        my rank into the directory specified by 'path_name'
 * @param write_rangeindex_to_metadata : true if writing a RangeIndex to
 * metadata
 * @param ri_start,ri_stop,ri_step start,stop,step parameters of given
 * RangeIndex
 * @param idx_name name of the given index
 * @param row_group_size Row group size in number of rows
 * @param prefix Prefix for parquet files written in distributed case
 * @param schema_metadata_pairs Additional metadata to include in the parquet
 * schema as key value pairs (string to string). The pandas index metadata will
 * be added automatically if relevant and doesn't need to be included.
 * @param filename Name of the file to be written. Currently this is only used
 * by Iceberg -- this will be refactored in future PRs.
 * @param convert_timedelta_to_int64 : cast timedelta to int64.
 * This is required for writing to Snowflake.
 * @param tz Timezone to use for Datetime (/timestamp) arrays. Provide an empty
 * string ("") to not specify one. This is primarily required for
 * Iceberg/Snowflake, for which we specify "UTC".
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
 * @param create_dir (default true): flag to create directory for Parquet
 * parallel write if it doesn't exist. Set to false in streaming Parquet write
 * since the writer init function creates the directory.
 * @returns size of the written file (in bytes)
 */
int64_t pq_write(
    const char *_path_name, const std::shared_ptr<table_info> table,
    const std::shared_ptr<array_info> col_names_arr,
    const std::shared_ptr<array_info> index, bool write_index,
    const char *metadata, const char *compression, bool is_parallel,
    bool write_rangeindex_to_metadata, const int ri_start, const int ri_stop,
    const int ri_step, const char *idx_name, const char *bucket_region,
    int64_t row_group_size, const char *prefix,
    bool convert_timedelta_to_int64 = false, std::string tz = "",
    bool downcast_time_ns_to_us = false, bool create_dir = true,
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO,
    std::unordered_map<std::string, std::string> schema_metadata_pairs = {},
    std::string filename = "",
    std::shared_ptr<arrow::Schema> expected_schema = nullptr);

int64_t pq_write_py_entry(const char *_path_name, table_info *table,
                          array_info *col_names_arr, array_info *index,
                          bool write_index, const char *metadata,
                          const char *compression, bool is_parallel,
                          bool write_rangeindex_to_metadata, const int ri_start,
                          const int ri_stop, const int ri_step,
                          const char *idx_name, const char *bucket_region,
                          int64_t row_group_size, const char *prefix,
                          bool convert_timedelta_to_int64, const char *tz,
                          bool downcast_time_ns_to_us, bool create_dir);

/**
 * @brief Create a directory for streaming Parquet write if not exists (called
 * from streaming Parquet writer init function)
 *
 * @param _path_name directory path
 */
void pq_write_create_dir_py_entry(const char *_path_name);

/**
 * Write the Bodo table (this process' chunk) to a partitioned directory of
 * parquet files. This process will write N files if it has N partitions in its
 * local data.
 * @param _path_name path of base output directory for partitioned dataset
 * @param table table to write to parquet files
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param col_names_arr_no_partitions array containing the table's column names
 * (index and partition columns not included)
 * @param categories_table table containing categories arrays for each partition
 * column that is a categorical array. Categories could be (for example) strings
 * like "2020-01-01", "2020-01-02", etc.
 * @param partition_cols_idx indices of partition columns in table
 * @param num_partition_cols number of partition columns
 * @param is_parallel true if the table is part of a distributed table
 * @param row_group_size Row group size in number of rows
 * @param prefix Prefix to use for each file created in distributed case
 * @param tz Timezone to use for Datetime (/timestamp) arrays. Provide an empty
 * string ("") to not specify one. NOTE: this will be applied for all datetime
 * columns.
 */
void pq_write_partitioned_py_entry(
    const char *_path_name, table_info *in_table, array_info *in_col_names_arr,
    array_info *in_col_names_arr_no_partitions, table_info *in_categories_table,
    int *partition_cols_idx, int num_partition_cols, const char *compression,
    bool is_parallel, const char *bucket_region, int64_t row_group_size,
    const char *prefix, const char *tz);
