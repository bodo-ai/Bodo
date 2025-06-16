#include <arrow/filesystem/filesystem.h>
#include <string>
#include <variant>
#include <vector>
#include "../libs/_bodo_common.h"

/**
 * @brief Main function for Iceberg write which can handle partition-spec
 * and sort-orders.
 *
 * @param table_data_loc Location of the Iceberg warehouse (the data folder)
 * @param table Bodo table to write
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param partition_spec Python list of tuples containing description of the
 * partition fields (in the order that the partitions should be applied)
 * @param sort_order Python list of tuples containing description of the
 * sort fields (in the order that the sort should be performed)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param[out] iceberg_files_info_py List of tuples for each of the files
 consisting of (file_name, record_count, file_size, *partition_values). Should
 be passed in as an empty list which will be filled during execution.
 * @param iceberg_schema Expected Arrow schema of files written to Iceberg
 table, if given.
* @param arrow_fs Arrow file system object for writing the data, can be nullptr
in which case the filesystem is inferred from the path.
 * @param sketches collection of theta sketches accumulating ndv for the table
 * as it is being written.
 * NOTE: passing sketches_ptr as void* since including the header for
 * UpdateSketchCollection causes compilation issues (TODO: fix).
 */
void iceberg_pq_write(
    const char *table_data_loc, std::shared_ptr<table_info> table,
    const std::vector<std::string> col_names_arr, PyObject *partition_spec,
    PyObject *sort_order, const char *compression, bool is_parallel,
    const char *bucket_region, int64_t row_group_size,
    const char *iceberg_metadata, PyObject *iceberg_files_info_py,
    std::shared_ptr<arrow::Schema> iceberg_schema,
    std::shared_ptr<arrow::fs::FileSystem> arrow_fs, void *sketches_ptr);
