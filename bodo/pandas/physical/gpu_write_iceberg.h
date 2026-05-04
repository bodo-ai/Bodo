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
#include <arrow/util/key_value_metadata.h>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
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
 * Supported transforms (initial scope): `identity`, `void`.
 */
struct PartitionField {
    cudf::size_type col_idx;  ///< Index of this column in the input table.
    std::string col_name;     ///< Column name (from Arrow schema).
    std::string transform;    ///< Transform name: "identity", "void", etc.
    long arg;                 ///< Transform argument (N for bucket, W for
                              ///< truncate; 0 if unused).
};

/**
 * @brief Describes a single Iceberg sort field parsed from the Python sort
 * spec.
 *
 * The sort spec is a list of `(col_name, nulls_last?)` tuples. Currently only
 * column index is stored; sort direction and null ordering are deferred to
 * follow-up work.
 */
struct SortField {
    cudf::size_type col_idx;  ///< Index of this column in the input table.
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
     * The partition spec is a Python list of `(transform_name, col_name,
     * [arg])` tuples. Column names are resolved against `schema` to produce
     * column indices. Currently only `identity` and `void` transforms are
     * processed; other transforms are stored but not handled during flush.
     *
     * @param schema Arrow schema providing column names.
     * @param partition_tuples_py Python list of partition spec tuples.
     * @throws std::runtime_error if a column name is not found in the schema
     *         or the spec is malformed.
     */
    void parse_partition_spec(const std::shared_ptr<arrow::Schema>& schema,
                              PyObject* partition_tuples_py);

    /**
     * @brief Parse the Python sort order into `SortField` structs.
     *
     * The sort spec is a Python list of `(col_name, nulls_last?)` tuples.
     * Column names are resolved against `schema` to produce column indices.
     * Sort direction (`asc`/`desc`) and null ordering are deferred to
     * follow-up work; all sorts currently use ascending + nulls last.
     *
     * @param schema Arrow schema providing column names.
     * @param sort_tuples_py Python list of sort spec tuples.
     * @throws std::runtime_error if a column name is not found in the schema
     *         or the spec is malformed.
     */
    void parse_sort_order(const std::shared_ptr<arrow::Schema>& schema,
                          PyObject* sort_tuples_py);

    /**
     * @brief Flush all accumulated tables to Parquet files.
     *
     * Steps:
     * 1. Concatenate all accumulated tables into one.
     * 2. Sort by sort_cols + partition_cols (ascending, nulls last) using
     *    `cudf::sort_by_key`.
     * 3. Extract partition columns to host Arrow arrays, walk rows to
     *    detect partition boundaries, and build partition groups.
     * 4. For each partition group:
     *    - Build the partition directory path.
     *    - Slice the sorted table and select non-void output columns.
     *    - Write a Parquet file via `BodoDataSink` with `iceberg.schema`
     *      in the key-value metadata.
     *    - Record `(file_name, record_count, file_size, *partition_values)`
     *      in `iceberg_files_info_py`.
     * 5. Clear the accumulation buffer.
     *
     * @param se Stream and event for CUDA operations.
     * @param is_last Whether this is the final flush (pipeline done).
     */
    void flush_buffer(std::shared_ptr<StreamAndEvent> se, bool is_last);

    /**
     * @brief Build a Hive-style partition directory path.
     *
     * For each partition field, appends `<col_name>=<value>`, separated by
     * `/`. Null values produce `__HIVE_DEFAULT_PARTITION__`.
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
     * @brief Populate a metrics vector for reporting to
     * `QueryProfileCollector`.
     *
     * Includes buffer, file, and partition-group statistics along with
     * per-stage timers and dictionary-builder metrics (placeholder).
     *
     * @param metrics_out Output vector to append metrics to.
     */
    void ReportMetrics(std::vector<MetricBase>& metrics_out);

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
                                                ///< writing (local or S3).

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
                                        ///< file_size, *partition_values)`
                                        ///< tuples. After finalization, holds
                                        ///< the MPI-gathered result.

    // ---- Counters and metrics ----

    int64_t iter;                            ///< Flush iteration counter
                                             ///< (used for file naming).
    PhysicalGPUWriteIcebergMetrics metrics;  ///< Per-operator metrics.
};
