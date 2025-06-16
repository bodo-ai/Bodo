
// Functions to write Bodo arrays to parquet
#pragma once

#include <aws/core/auth/AWSCredentialsProvider.h>
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
 * @param table Arrow table to write to parquet file
 * @param is_parallel true if the table is part of a distributed table (in this
 *        case, this process writes a file named "part-000X.parquet" where X is
 *        my rank into the directory specified by 'path_name'
 * @param row_group_size Row group size in number of rows
 * @param prefix Prefix for parquet files written in distributed case
 * @param bodo_array_types Bodo array types of the columns in the table. This is
 * used to determine parquet metadata information.
 * @param filename Name of the file to be written. Currently this is only used
 * by Iceberg -- this will be refactored in future PRs.
 * @param create_dir (default true): flag to create directory for Parquet
 * parallel write if it doesn't exist. Set to false in streaming Parquet write
 * since the writer init function creates the directory.
 * @param arrow_fs (default nullptr): Arrow filesystem to write to, fall back to
 * parsing the path if not present
 * @param force_hdfs (default false): Force HDFS filesystem type
 * @returns size of the written file (in bytes)
 */
int64_t pq_write(const char *_path_name,
                 const std::shared_ptr<arrow::Table> &table,
                 const char *compression, bool is_parallel,
                 const char *bucket_region, int64_t row_group_size,
                 const char *prefix,
                 std::vector<bodo_array_type::arr_type_enum> bodo_array_types,
                 bool create_dir = true, std::string filename = "",
                 arrow::fs::FileSystem *arrow_fs = nullptr);

int64_t pq_write_py_entry(const char *_path_name, table_info *table,
                          array_info *col_names_arr, const char *metadata,
                          const char *compression, bool is_parallel,
                          const char *bucket_region, int64_t row_group_size,
                          const char *prefix, bool convert_timedelta_to_int64,
                          const char *tz, bool downcast_time_ns_to_us,
                          bool create_dir);

/**
 * @brief Create a directory for streaming Parquet write if not exists (called
 * from streaming Parquet writer init function)
 *
 * @param _path_name directory path
 */
void pq_write_create_dir_py_entry(const char *_path_name);
void pq_write_create_dir(const char *_path_name);

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
