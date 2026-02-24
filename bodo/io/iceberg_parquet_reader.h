#pragma once

#include <memory>

#include <arrow/compute/expression.h>
#include <arrow/compute/type_fwd.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/python/pyarrow.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>
#include <arrow/util/thread_pool.h>
#include <fmt/format.h>
#include <object.h>

#include "../libs/_io_cpu_thread_pool.h"
#include "arrow_reader.h"

struct IcebergParquetReaderMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    //// Metrics from the initialization step

    /// During get_dataset:

    // Time to get the file list from the Iceberg Connector
    time_t get_ds_file_list_time = 0;
    // Time to map the list of files to schema ID in the Iceberg Connector
    time_t get_ds_file_to_schema_time_us = 0;
    // Time to get the filesystem.
    time_t get_ds_get_fs_time = 0;
    // Number of files that we're analyzing (schema validation, row counts,
    // etc.)
    stat_t get_ds_n_files_analyzed = 0;
    // Time to create the file fragments (only applicable in the row_level case)
    time_t get_ds_file_frags_creation_time = 0;
    // Time to get Schema Group identifiers for all fragments
    time_t get_ds_get_sg_id_time = 0;
    // Time to sort the fragments by their schema group identifier
    time_t get_ds_sort_by_sg_id_time = 0;
    // Number of unique schema groups in files analyzed by this rank.
    stat_t get_ds_nunique_sgs_seen = 0;
    // Time to get exact row counts (only applicable in the row_level case).
    time_t get_ds_exact_row_counts_time = 0;
    // Number of row groups in files assigned to this rank for analysis
    // (only applicable in the row_level case)
    stat_t get_ds_get_row_counts_nrgs = 0;
    // Number of rows in files assigned to this rank for scan planning. In the
    // row-level case, this is the row count after filtering. In the piece level
    // case, this is the total number of rows in all the row groups that survive
    // the row group level filtering.
    stat_t get_ds_get_row_counts_nrows = 0;
    // Total size in bytes of row groups of files analyzed by this file.
    // (only applicable in the row_level case)
    stat_t get_ds_get_row_counts_total_bytes = 0;
    // Time spent in all-gathering all the pieces to create the global
    // IcebergParquetDataset after the initial parallel analysis.
    time_t get_ds_pieces_allgather_time = 0;
    // Time spent sorting all the pieces based on their schema group identifier
    // (after the allgather)
    time_t get_ds_sort_all_pieces_time = 0;
    // Time spent assembling the final IcebergParquetDataset object from all the
    // pieces.
    time_t get_ds_assemble_ds_time = 0;

    /// Metrics about the dataset (global)

    // Number of unique schema groups overall.
    stat_t ds_nunique_schema_groups = 0;

    /// During init_scanners:

    // Time to get the PyArrow datasets
    time_t init_scanners_get_pa_datasets_time = 0;
    // Number of scanners this rank will use. There's one per schema group, so
    // this is also the number of unique schema groups that this rank will read
    // files from.
    stat_t n_scanners = 0;
    // Time spent creating scanners from the pa datasets.
    time_t create_scanners_time = 0;

    //// Metrics from the the actual read step:

    // Time to get the next batches.
    time_t get_batch_time = 0;
    // Time to evolve the schema of the record batch.
    time_t evolve_time = 0;
    // Time to convert an Arrow RecordBatch to a Bodo table.
    time_t arrow_rb_to_bodo_time = 0;
    // Time to unify dictionaries full batches.
    time_t unify_time = 0;
    // Time to unify and append small batches.
    time_t unify_append_small_time = 0;
    // Number of batches read
    stat_t n_batches = 0;
    // Number of small batches read.
    stat_t n_small_batches = 0;
    // Time spent popping a chunk from 'out_batches'.
    time_t output_pop_chunk_time = 0;
};

class IcebergParquetReader : public ArrowReader {
   public:
    /**
     * Initialize IcebergParquetReader.
     * See iceberg_pq_read_py_entry function below for description of arguments.
     */
    IcebergParquetReader(PyObject* catalog, const char* _table_id,
                         bool _parallel, int64_t tot_rows_to_read,
                         PyObject* _iceberg_filter,
                         std::string _expr_filter_f_str,
                         PyObject* _filter_scalars,
                         std::vector<int> _selected_fields,
                         std::vector<bool> is_nullable,
                         PyObject* _pyarrow_schema, int64_t batch_size,
                         int64_t op_id, int64_t _snapshot_id);
    ~IcebergParquetReader() override;

    void init_iceberg_reader(std::span<int32_t> str_as_dict_cols,
                             bool create_dict_from_string);

    // Return and incref the file list.
    PyObject* get_file_list();

    int64_t get_snapshot_id() { return this->snapshot_id; }

    size_t get_num_pieces() const override { return this->file_paths.size(); }

    IcebergParquetReaderMetrics iceberg_reader_metrics;

    /**
     * @brief Report Read Stage metrics if they haven't already been reported.
     * Note that this will only report the metrics to the QueryProfileCollector
     * the first time it's called.
     *
     */
    void ReportReadStageMetrics(std::vector<MetricBase>& metrics_out) override;

   protected:
    bool force_row_level_read() const override;

    PyObject* get_dataset() override;

    void distribute_pieces(PyObject* pieces_py) override;

    /**
     * @brief Add an Iceberg Parquet piece/file for this rank
     * to read. The pieces must be added in the order of the
     * schema groups they belong to.
     *
     * @param piece Python object (IcebergPiece)
     * @param num_rows Number of rows to read from this file.
     */
    void add_piece(PyObject* piece, int64_t num_rows) override;

    std::shared_ptr<table_info> get_empty_out_table() override;

    std::tuple<table_info*, bool, uint64_t> read_inner_row_level() override;

    std::tuple<table_info*, bool, uint64_t> read_inner_piece_level() override;

    /**
     * @brief Initialize the Arrow Dataset Scanners. This will create
     * one Scanner per unique Schema Group that the rank will read from.
     *
     */
    void init_scanners();

    /**
     * @brief Report Init Stage metrics if they haven't already been reported.
     * Note that this will only report the metrics to the QueryProfileCollector
     * the first time it's called.
     *
     */
    void ReportInitStageMetrics(std::vector<MetricBase>& metrics_out) override;

   private:
    // Pyiceberg catalog to read table metadata
    PyObject* catalog;
    // Table identifiers for the iceberg table
    // provided by the user.
    const char* table_id;

    /// @brief Executor to use for CPU bound tasks.
    arrow::internal::Executor* cpu_executor() const {
        if (this->st_cpu_executor_.has_value()) {
            return this->st_cpu_executor_.value().get();
        } else {
            return ::arrow::internal::GetCpuThreadPool();
        }
    }

    // CPU Thread Pool
    std::optional<std::shared_ptr<bodo::SingleThreadedCpuThreadPool>>
        st_cpu_executor_ = std::nullopt;

    // Average number of files that each rank will read.
    double avg_num_pieces = 0;

    /// Filter information to be passed to create
    /// the IcebergParquetDataset.

    // Filters to use for file pruning using Iceberg metadata.
    PyObject* iceberg_filter;
    // Information for constructing the filters dynamically
    // for each schema group based on their schema. See
    // description of bodo.io.iceberg.generate_expr_filter
    // for more details.
    std::string expr_filter_f_str;
    PyObject* filter_scalars;

    // Filesystem to use for reading the files.
    PyObject* filesystem = nullptr;

    // Memoized empty out table.
    std::shared_ptr<table_info> empty_out_table;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;
    // Number of rows to read from the files.
    std::vector<int64_t> pieces_nrows;
    // Index of the schema group corresponding to the files.
    std::vector<int64_t> pieces_schema_group_idx;
    // List of IcebergSchemaGroup objects.
    PyObject* schema_groups_py;

    // Scanners to use. There's one per IcebergSchemaGroup
    // that this rank will read from.
    std::vector<std::shared_ptr<arrow::dataset::Scanner>> scanners;
    // The schemas that the scanners will use for reading.
    // These will be used as the 'source' schemas when evolving
    // the record batches returned by the corresponding scanners.
    std::vector<std::shared_ptr<arrow::Schema>> scanner_read_schemas;
    // Next scanner to use.
    size_t next_scanner_idx = 0;
    // Arrow Batched Reader to get next batch iteratively.
    std::shared_ptr<arrow::RecordBatchReader> curr_reader = nullptr;
    // Corresponding read schema.
    std::shared_ptr<arrow::Schema> curr_read_schema = nullptr;
    // Number of remaining rows to skip outputting
    // from the first file assigned to this process. Only applicable in the
    // row-level read case.
    int64_t rows_to_skip = -1;

    // List of the original Iceberg file names as relative paths.
    // For example if the absolute path was
    // /Users/bodo/iceberg_db/my_table/part01.pq and the iceberg directory is
    // iceberg_db, then the path in the list would be
    // iceberg_db/my_table/part01.pq. These are used by merge/delete and are not
    // the same as the files we read, which are absolute paths.
    PyObject* file_list = nullptr;
    // Iceberg snapshot id for read.
    int64_t snapshot_id;

    /**
     * @brief Helper function to get the next available
     * RecordBatch. Note that in the row-level case, this must only be called
     * when there are rows left to read.
     *
     * @return std::tuple<bool, std::shared_ptr<arrow::RecordBatch>>: Boolean
     * indicating if we're out of scanners/pieces, next batch (potentially
     * nullptr). If we're out of pieces, the batch is guaranteed to be nullptr.
     * If we return a valid batch (i.e. not nullptr), the bool is guaranteed to
     * be false. However, we may return {false, nullptr}, which indicates that
     * the user should call this function again to get the next batch.
     */
    std::tuple<bool, std::shared_ptr<arrow::RecordBatch>> get_next_batch();
};
