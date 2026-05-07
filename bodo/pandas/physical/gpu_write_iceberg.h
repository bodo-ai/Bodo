#pragma once

/**
 * @file gpu_write_iceberg.h
 * @brief GPU-accelerated Iceberg table write operator for the Bodo DataFrame
 * library.
 *
 * Mirrors the CPU-side `PhysicalWriteIceberg` (write_iceberg.h) but operates
 * on `cudf::table` batches via the `PhysicalGPUSink` interface, following the
 * pattern established by `PhysicalGPUWriteParquet` (gpu_write_parquet.h).
 *
 * Data flow:
 *   1. Accumulate incoming `GPU_DATA` batches into a buffer of cudf tables.
 *   2. On flush: concatenate, sort by sort_spec + partition_spec, detect
 *      partition boundaries, and write each partition group to a Parquet file.
 *   3. On finalize: MPI-gather file metadata tuples across all ranks.
 */

#include <Python.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <arrow/scalar.h>
#include <arrow/util/key_value_metadata.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/types.hpp>
#include <memory>
#include <string>
#include <vector>

#include "../../libs/_utils.h"
#include "../../libs/gpu_utils.h"
#include "../../libs/streaming/_shuffle.h"
#include "../libs/_query_profile_collector.h"
#include "_bodo_write_function.h"
#include "operator.h"

/// Helper macro for Arrow status checks in GPU Iceberg write operations.
/// Throws `std::runtime_error` with a descriptive message on failure.
#define CHECK_ARROW_GPU_ICEBERG(expr, msg)                                  \
    if (!(expr.ok())) {                                                     \
        std::string err_msg = std::string("Error in GPU iceberg write: ") + \
                              msg + " " + expr.ToString();                  \
        throw std::runtime_error(err_msg);                                  \
    }

/**
 * @brief Describes a single Iceberg partition field parsed from the Python
 * partition spec.
 *
 * The partition spec is a list of `(transform_name, col_name, [arg])` tuples.
 * The constructor of `PhysicalGPUWriteIceberg` parses these into
 * `PartitionField` structs.
 *
 */
struct PartitionField {
    cudf::size_type col_idx;     ///< Index of this column in the input table.
    std::string col_name;        ///< Column name (from Arrow schema).
    std::string transform;       ///< Transform name: "identity", "void", etc.
    long arg;                    ///< Transform argument (N for bucket, W for
                                 ///< truncate; 0 if unused).
    std::string partition_name;  ///< Name used in the partition directory path
                                 ///< (e.g. "region" for "region=US/").
};

/**
 * @brief Describes a single Iceberg sort field parsed from the Python sort
 * spec.
 *
 * The sort spec is a list of `(col_idx, transform_name, arg, is_asc,
 * nulls_last)` tuples produced by `build_partition_sort_tuples`.
 */
struct SortField {
    cudf::size_type col_idx;  ///< Index of this column in the input table.
    bool is_asc;              ///< true for ascending, false for descending.
    bool nulls_last;          ///< true to place nulls after all values, false
                              ///< to place nulls first.
    std::string transform;    ///< Transform name: "identity", "bucket", etc.
    long arg;                 ///< Transform argument (N for bucket, W for
                              ///< truncate; 0 if unused).

    /// Returns true if the transform acts as a pass-through (identity or
    /// void), meaning no column transformation is needed before sorting.
    [[nodiscard]] bool is_noop_transform() const {
        return transform == "identity" || transform == "void";
    }
};

/**
 * @brief Holds the rows and partition values for one partition group detected
 * during flush.
 *
 * Produced by `detect_partition_groups` and consumed by `write_group`.
 */
struct PartitionGroup {
    cudf::size_type start_row;  ///< First row index in the working table for
                                ///< this partition group.
    cudf::size_type end_row;    ///< One past the last row index.
    std::vector<std::shared_ptr<arrow::Scalar>>
        xfrm_values;  ///< Transformed partition values at the first row.
    std::vector<std::shared_ptr<arrow::Scalar>>
        orig_values;  ///< Original partition values at the first row.
};

/**
 * @brief Tracks per-operator metrics for `PhysicalGPUWriteIceberg`.
 *
 * Reported to `QueryProfileCollector` during `FinalizeSink`.
 */
struct PhysicalGPUWriteIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t max_buffer_rows = 0;
    stat_t n_files_written = 0;
    stat_t n_partition_groups = 0;

    time_t accumulate_time = 0;
    time_t sort_time = 0;
    time_t file_write_time = 0;
    time_t finalize_time = 0;
};

/**
 * @brief GPU-accelerated Iceberg table write operator.
 *
 * Implements `PhysicalGPUSink` to consume `GPU_DATA` batches from the
 * execution pipeline and write them as partitioned, sorted Parquet files to
 * an Iceberg table location.
 *
 * @see PhysicalWriteIceberg (CPU counterpart)
 * @see PhysicalGPUWriteParquet (GPU Parquet writer, shares BodoDataSink)
 * @see gpu_read_iceberg.h (GPU Iceberg reader)
 */
class PhysicalGPUWriteIceberg : public PhysicalGPUSink {
   public:
    /**
     * @brief Construct from schema and Iceberg write bind data.
     *
     * Parses partition and sort tuples from Python, builds column-name
     * to index mappings, and initializes accumulation state.
     *
     * @param in_bodo_schema Bodo schema of the input table columns.
     * @param bind_data Iceberg write function data (table location, filesystem,
     *                  compression, partition/sort specs, Iceberg schema,
     * etc.).
     */
    explicit PhysicalGPUWriteIceberg(
        std::shared_ptr<bodo::Schema> in_bodo_schema,
        IcebergWriteFunctionData& bind_data);

    virtual ~PhysicalGPUWriteIceberg() = default;

    /**
     * @brief Consume a GPU-side batch and accumulate into the write buffer.
     *
     * Non-GPU ranks immediately return `NEED_MORE_INPUT` until
     * `prev_op_result == FINISHED`, then return `FINISHED`. GPU ranks
     * accumulate non-empty tables and flush when the estimated byte count
     * exceeds `max_pq_chunksize` or the pipeline signals completion.
     *
     * Uses `sync_is_last_non_blocking` to synchronize the FINISHED flag
     * across all ranks (same as the CPU writer).
     *
     * @param input_batch Input GPU batch (cudf::table + Arrow schema +
     *                    stream/event).
     * @param prev_op_result Result from the preceding operator.
     * @param se Stream and event for CUDA stream ordering.
     * @return `FINISHED` if the sink is done, `NEED_MORE_INPUT` otherwise.
     */
    OperatorResult ConsumeBatchGPU(GPU_DATA input_batch,
                                   OperatorResult prev_op_result,
                                   std::shared_ptr<StreamAndEvent> se) override;

    /**
     * @brief Return the accumulated Iceberg file metadata.
     *
     * Returns `iceberg_files_info_py`, a Python list of
     * `(file_name, record_count, file_size, *partition_values)` tuples,
     * one per file written. Before finalization this is the local rank's
     * list; after `FinalizeSink` it is the MPI-gathered list from all ranks.
     *
     * @return Pointer to the Python list object.
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override;

    /**
     * @brief Gather file metadata across all MPI ranks and report metrics.
     *
     * Imports `mpi4py.MPI`, calls `COMM_WORLD.gather(iceberg_files_info_py)`,
     * replaces the local list with the gathered result, and registers
     * operator metrics with `QueryProfileCollector`.
     */
    void FinalizeSink() override;

   private:
    /**
     * @brief Parse the Python partition spec into `PartitionField` structs.
     *
     * The partition spec is a Python list of
     * `(col_idx, transform_name, arg, partition_name)` tuples produced by
     * `build_partition_sort_tuples`.
     *
     * @param schema Arrow schema providing column names.
     * @param partition_tuples_py Python list of partition spec tuples.
     * @throws std::runtime_error if the spec is malformed.
     */
    void parse_partition_spec(const std::shared_ptr<arrow::Schema>& schema,
                              PyObject* partition_tuples_py);

    /**
     * @brief Parse the Python sort order into `SortField` structs.
     *
     * The sort spec is a Python list of `(col_idx, transform_name, arg,
     * is_asc, nulls_last)` tuples produced by `build_partition_sort_tuples`.
     * Column names are resolved against `schema` to produce column indices.
     * Sort direction, null ordering, and transform info are stored in each
     * `SortField`.
     *
     * @param schema Arrow schema providing column names.
     * @param sort_tuples_py Python list of sort spec tuples.
     * @throws std::runtime_error if a transform is unsupported or the spec
     *         is malformed.
     */
    void parse_sort_order(const std::shared_ptr<arrow::Schema>& schema,
                          PyObject* sort_tuples_py);

    /**
     * @brief Concatenate all accumulated tables into one working table.
     *
     * Handles the single-table fast-path (avoiding a copy) and clears
     * `accumulated_tables`.
     *
     * @param se Stream and event for CUDA operations.
     * @return Concatenated cuDF table.
     */
    std::unique_ptr<cudf::table> concatenate_accumulated(
        std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Apply sort and partition transforms, prepending the resulting
     * columns to the working table.
     *
     * The output table has layout:
     *   [sort_xfrm_0, ..., sort_xfrm_N, part_xfrm_0, ..., part_xfrm_M,
     *    orig_col_0, ...]
     *
     * @param working_table Concatenated table (consumed, columns released).
     * @param se Stream and event for CUDA operations.
     * @return Table with prepended transform columns.
     */
    std::unique_ptr<cudf::table> prepend_transform_columns(
        std::unique_ptr<cudf::table> working_table,
        std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Sort the working table by sort keys then partition-transform
     * keys.
     *
     * Builds sort-key indices, column orders, and null precedences from
     * `sort_fields` and `partition_fields`. Void partition columns are
     * skipped. Calls `cudf::sort_by_key` and replaces `working_table` in
     * place.
     *
     * @param working_table Table to sort (mutated in place).
     * @param se Stream and event for CUDA operations.
     */
    void sort_working_table(std::unique_ptr<cudf::table>& working_table,
                            std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Detect partition boundaries in a sorted working table using
     * GPU group-by.
     *
     * For non-partitioned tables, produces a single group covering all
     * rows. For partitioned tables, uses `cudf::groupby(sorted::YES)` on
     * the transformed partition columns, gathers the first-row values,
     * and converts them to Arrow scalars.
     *
     * @param working_table Sorted working table (with prepended transform
     *        columns).
     * @param groups Output vector of partition groups.
     * @param se Stream and event for CUDA operations.
     */
    void detect_partition_groups(
        const std::unique_ptr<cudf::table>& working_table,
        std::vector<PartitionGroup>& groups,
        std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Write a single partition group as a Parquet file and record
     * its metadata.
     *
     * Handles: partition directory path building, directory creation,
     * slicing the working table, Parquet metadata (iceberg.schema +
     * field IDs), the `cudf::io::write_parquet` call, per-column
     * statistics via `compute_field_metrics_gpu`, and appending the
     * file-info tuple to `iceberg_files_info_py`.
     *
     * @param working_tv Full working table view (with prepended columns).
     * @param num_prepended Number of prepended transform columns.
     * @param output_col_names Names of output columns.
     * @param output_col_indices Original column indices for output.
     * @param group Partition group to write.
     * @param se Stream and event for CUDA operations.
     */
    void write_group(const cudf::table_view& working_tv,
                     cudf::size_type num_prepended,
                     const std::vector<std::string>& output_col_names,
                     const std::vector<cudf::size_type>& output_col_indices,
                     const PartitionGroup& group,
                     std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Flush all accumulated tables to Parquet files.
     *
     * Orchestrates: concatenation, transform prepending, sorting,
     * partition-boundary detection, and per-group file writing.
     *
     * @param se Stream and event for CUDA operations.
     * @param is_last Whether this is the final flush (pipeline done).
     */
    void flush_buffer(std::shared_ptr<StreamAndEvent> se, bool is_last);

    /**
     * @brief Build a Hive-style partition directory path.
     *
     * For each partition field, appends `<col_name>=<value>`, separated by
     * `/`. Null values produce `__HIVE_DEFAULT_PARTITION__`. Uses the
     * scalar's `ToString()` for rendering (prefer `render_partition_value`
     * for transform-aware path building in `write_group`).
     *
     * @param partition_values Arrow scalars for this partition group's
     *                        identity, in partition-spec order.
     * @return Relative path from the table location (e.g.,
     *         `region=US/date=2024-01-01`), or empty string if no partitions.
     */
    std::string build_partition_path(
        const std::vector<std::shared_ptr<arrow::Scalar>>& partition_values);

    /**
     * @brief Map a compression string to a cuDF compression type.
     *
     * Supported values: `snappy`, `gzip`, `brotli`, `lz4`, `zstd`.
     * Throws for unsupported codecs
     *
     * @param c Compression codec name.
     * @return Corresponding `cudf::io::compression_type`.
     */
    cudf::io::compression_type pq_compression_from_string(const std::string& c);

    /**
     * @brief Compute per-column Iceberg file statistics on GPU and return
     * the four Python stat dicts.
     *
     * For each output column in the partition group slice:
     * - `value_counts` = col.size()
     * - `null_count`   = col.null_count()
     * - lower/upper bounds from `cudf::minmax(col)` (skipped for all-null
     *   columns and nested types).  Results are serialized to Iceberg
     *   binary format on the host side after a single batched GPU→host
     *   transfer of the min/max scalars.
     *
     * @param group_view     Sliced view of the partition group's output
     *                       columns (on GPU, stays on GPU).
     * @param output_col_names Names of the output columns (drives Iceberg
     *                         field ID lookups).
     * @param se             Stream and event for CUDA operations.
     * @param value_counts_dict Output dict: {field_id: value_count}.
     * @param null_count_dict   Output dict: {field_id: null_count}.
     * @param lower_bound_dict  Output dict: {field_id: bytes}.
     * @param upper_bound_dict  Output dict: {field_id: bytes}.
     */
    void compute_field_metrics_gpu(
        cudf::table_view group_view,
        const std::vector<std::string>& output_col_names,
        std::shared_ptr<StreamAndEvent> se, PyObject* value_counts_dict,
        PyObject* null_count_dict, PyObject* lower_bound_dict,
        PyObject* upper_bound_dict);

    /**
     * @brief Serialize an Arrow scalar to Iceberg binary format.
     *
     * Follows
     * https://iceberg.apache.org/spec/#appendix-d-single-value-serialization
     *
     * @param scalar Arrow scalar.
     * @return Python bytes object.
     */
    static PyObject* arrow_scalar_to_iceberg_bytes(
        const std::shared_ptr<arrow::Scalar>& scalar);

    /**
     * @brief Convert an Arrow scalar to a Python object (new reference).
     *
     * Handles all Arrow types used in partition values: integers (signed
     * and unsigned), floats, strings, booleans, DATE32, TIMESTAMP, BINARY,
     * and DECIMAL128.
     *
     * @param scalar Arrow scalar.
     * @return New Python object reference (PyLong, PyFloat, PyUnicode,
     *         Py_True/Py_False, or Py_None).
     * @throws std::runtime_error on unsupported types.
     */
    static PyObject* arrow_scalar_to_pyobject(
        const std::shared_ptr<arrow::Scalar>& scalar);

    /**
     * @brief Serialize a native value to little-endian Python bytes.
     */
    template <typename T>
    static PyObject* buffer_to_little_endian_bytes(T value);

    /**
     * @brief Generate a UUID-based Iceberg data file name.
     *
     * Format: `{rank:05d}-{rank}-{uuid}.parquet`
     * (matches the CPU writer's `generate_iceberg_file_name`).
     */
    static std::string generate_iceberg_file_name();

    /**
     * @brief Populate a metrics vector for reporting to
     * `QueryProfileCollector`.
     *
     * Includes buffer, file, and partition-group statistics along with
     * per-stage timers.
     *
     * @param metrics_out Output vector to append metrics to.
     */
    void ReportMetrics(std::vector<MetricBase>& metrics_out);

    /**
     * @brief Apply an Iceberg transform to a GPU column and return the
     * transformed column.
     *
     * Dispatch function that routes to the appropriate transform
     * implementation based on `transform_name`.
     *
     * @param col Input column view (on GPU).
     * @param transform_name Transform name: "identity", "bucket", "truncate",
     *        "year", "month", "day", "hour", or "void".
     * @param arg Transform argument (N for bucket, W for truncate; -1 if
     *        unused).
     * @param stream CUDA stream.
     * @return New cudf column with the transformed values.
     * @throws std::runtime_error on unsupported transforms or types.
     */
    static std::unique_ptr<cudf::column> apply_iceberg_transform_gpu(
        cudf::column_view col, const std::string& transform_name, long arg,
        rmm::cuda_stream_view stream);

    /**
     * @brief Render a partition value for use in a Hive-style partition
     * directory path.
     *
     * Uses both the original value (for identity-transform rendering of
     * complex types like DATE) and the transformed value (for non-identity
     * transforms like year, month, bucket, etc.).
     *
     * @param transform_name Transform name.
     * @param arg Transform argument (N for bucket, W for truncate; -1 if
     *        unused).
     * @param orig_scalar Arrow scalar from the original (untransformed)
     *        column at the partition group's first row.
     * @param transformed_scalar Arrow scalar from the transformed column
     *        at the partition group's first row.
     * @return String value for the partition directory component (e.g.,
     *         "2024" for a year transform).
     */
    static std::string render_partition_value(
        const std::string& transform_name, long arg,
        const std::shared_ptr<arrow::Scalar>& orig_scalar,
        const std::shared_ptr<arrow::Scalar>& transformed_scalar);

    // ---- Bind data (from IcebergWriteFunctionData) ----

    const std::shared_ptr<arrow::Schema> in_schema;  ///< Input Arrow schema
                                                     ///< (column names/types).
    const std::string table_loc;     ///< Iceberg table data root directory.
    const int64_t max_pq_chunksize;  ///< Flush threshold in bytes.
    const std::string compression;   ///< Compression codec name.
    const std::string iceberg_schema_str;  ///< Iceberg schema JSON string for
                                           ///< Parquet key-value metadata.
    const std::shared_ptr<arrow::Schema> iceberg_schema;  ///< Expected Iceberg
                                                          ///< table schema.
    std::shared_ptr<arrow::fs::FileSystem> fs;  ///< Arrow filesystem for
                                                ///< writing (e.g. local or S3).

    // ---- Partition and sort specs ----

    PyObjectPtr partition_tuples;  ///< Python partition spec (kept alive
                                   ///< via Py_INCREF).
    PyObjectPtr sort_tuples;       ///< Python sort spec (kept alive via
                                   ///< Py_INCREF).
    std::vector<PartitionField> partition_fields;  ///< Parsed partition spec.
    std::vector<SortField> sort_fields;            ///< Parsed sort spec.
    std::vector<cudf::size_type> sort_cols;        ///< Concatenated sort +
                                                   ///< partition column indices
                                                   ///< for `sort_by_key`.
    std::vector<cudf::size_type> partition_col_idxs;  ///< Partition column
                                                      ///< indices in input
                                                      ///< table.

    // ---- Accumulation state ----

    std::vector<std::shared_ptr<cudf::table>>
        accumulated_tables;  ///< Pending tables not yet flushed.
    int64_t total_rows;      ///< Sum of rows across all pending tables.
    int64_t total_bytes;     ///< Estimated sum of bytes across all pending
                             ///< tables (approximate).
    int64_t row_sz_est;  ///< Estimated average row size in bytes (for updating
                         ///< total_bytes).

    // ---- Stream ordering ----

    std::shared_ptr<StreamAndEvent>
        prev_batch_se;  ///< Stream/event of the previous batch; waited
                        ///< on before consuming the next batch to
                        ///< serialize writes.

    // ---- Termination ----

    std::shared_ptr<IsLastState> is_last_state;  ///< State for non-blocking
                                                 ///< is-last synchronization.
    bool finished;  ///< Sink has finished and will reject new batches.

    // ---- File tracking ----

    PyObject* iceberg_files_info_py;  ///< Python list of
                                      ///< `(file_name, record_count,
                                      ///< file_size, value_counts_dict,
                                      ///< null_count_dict,
                                      ///< lower_bound_dict,
                                      ///< upper_bound_dict,
                                      ///< *partition_values)` tuples.
                                      ///< After finalization, holds
                                      ///< the MPI-gathered result.

    // ---- Counters and metrics ----

    int64_t iter;                            ///< Flush iteration counter
                                             ///< (used for file naming).
    PhysicalGPUWriteIcebergMetrics metrics;  ///< Per-operator metrics.
};
