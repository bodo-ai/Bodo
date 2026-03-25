#pragma once

#include <Python.h>
#include <arrow/util/key_value_metadata.h>
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/scalar/scalar.hpp>
#include <memory>
#include <random>
#include <utility>
#include "../_util.h"
#include "../io/parquet_reader.h"
#include "../libs/gpu_utils.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "gpu_expression.h"
#include "operator.h"

#include <mpi.h>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>  // cudf::io::read_parquet, options
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <parquet/arrow/reader.h>  // parquet::ParquetFileReader or parquet::arrow::FileReader
#include <parquet/exception.h>
#include "../io/arrow_compat.h"

#include <glob.h>
#include <parquet/metadata.h>
#include <unistd.h>
#include <cstddef>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Helper to ensure that the pyarrow wrappers have been imported.
// We use a static variable to make sure we only do the import once.
static bool imported_pyarrow_wrappers = false;
static void ensure_pa_wrappers_imported() {
#define CHECK(expr, msg)                                        \
    if (expr) {                                                 \
        throw std::runtime_error(std::string("fs_io: ") + msg); \
    }
    if (imported_pyarrow_wrappers) {
        return;
    }
    CHECK(arrow::py::import_pyarrow_wrappers(),
          "importing pyarrow_wrappers failed!");
    imported_pyarrow_wrappers = true;

#undef CHECK
}

// Limit the total number of uncompressed bytes that can be processed by a
// single chunked reader. This is a soft limit applied to multi-part datasets.
const int64_t CHUNKED_READER_TOTAL_BYTES_LIMIT =
    4LL * 1024 * 1024 * 1024;  // 4GB

// Whether to use libkvikio for S3 reads (better for large reads) but off by
// default for now since it requires setting a lot of extra environment
// variables.
const bool USE_KVIKIO_REMOTE_SOURCE = false;

// Parquet metadata estimation parameters.
const int PARQUET_SAMPLING_RANDOM_SEED =
    42;  // Seed to control random sampling of parquet files for bytes/row and
         // bytes/file estimates.
const double PARQUET_SAMPLING_FRACTION =
    0.001;  // Sample 0.1% of files in the dataset for metadata estimation
const size_t PARQUET_SAMPLING_MIN_FILES =
    3;  // Minimum number of parquet files to sample from (if <3 parts in the
        // dataset, all files will be read).

/**
 * @brief A datasource implementation for reading from Arrow files.
 *
 */
class arrow_file_datasource : public cudf::io::datasource {
   public:
    explicit arrow_file_datasource(
        std::shared_ptr<arrow::io::RandomAccessFile> arrow_file)
        : arrow_file_(std::move(arrow_file)) {
        // Start read immediately using the file's IO context and save the
        // result for better performance. This can potentially consume a lot of
        // CPU memory if file sizes are large, though the number of files read
        // at once is limited by CHUNKED_READER_TOTAL_BYTES_LIMIT.
        file_buffer_future_ =
            arrow_file_->ReadAsync(0, arrow_file_->GetSize().ValueOrDie());
    }

    std::unique_ptr<buffer> host_read(size_t offset, size_t size) override {
        ensure_buffer();

        auto ptr = reinterpret_cast<const uint8_t *>(buffer_->data()) + offset;

        return std::make_unique<datasource::non_owning_buffer>(ptr, size);
    }

    size_t host_read(size_t offset, size_t size, uint8_t *dst) override {
        ensure_buffer();

        auto ptr = reinterpret_cast<const uint8_t *>(buffer_->data()) + offset;

        memcpy(dst, ptr, size);
        return size;
    }

    size_t size() const override { return arrow_file_->GetSize().ValueOrDie(); }

   private:
    void ensure_buffer() {
        std::call_once(buffer_init_, [&] {
            buffer_ = file_buffer_future_.result().ValueOrDie();
        });
    }

    std::shared_ptr<arrow::io::RandomAccessFile> arrow_file_;

    arrow::Future<std::shared_ptr<arrow::Buffer>> file_buffer_future_;

    std::shared_ptr<arrow::Buffer> buffer_;
    std::once_flag buffer_init_;
};

struct FilePart {
    std::string path;
    int start_row_group;               // inclusive
    std::optional<int> end_row_group;  // exclusive (if nullopt, read all row
                                       // groups in the file)
};

struct RankBatchGeneratorMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t bytes_per_part_estimate = 0;
    stat_t chunked_reader_limit_bytes = 0;
    stat_t num_readers = 0;
    time_t get_dataset_time = 0;
    time_t estimate_parquet_metadata_time = 0;
    time_t get_next_chunked_reader_time = 0;
};

class RankBatchGenerator {
   public:
    RankBatchGenerator(PyObject *dataset_path,
                       duckdb::unique_ptr<duckdb::TableFilterSet> &filter_exprs,
                       std::size_t target_rows,
                       const std::vector<std::string> &_selected_columns,
                       std::shared_ptr<arrow::Schema> _arrow_schema,
                       std::shared_ptr<arrow::Schema> _output_arrow_schema,
                       MPI_Comm comm)
        : path_(dataset_path),
          filesystem_(nullptr),
          target_rows_(target_rows),
          selected_columns(_selected_columns),
          arrow_schema(std::move(_arrow_schema)),
          output_arrow_schema(std::move(_output_arrow_schema)) {
        tableFilterSetToCudfAST(*filter_exprs, arrow_schema->field_names(),
                                filter_ast_tree, filter_scalars);

        get_dataset();

        // Only assign parts to GPU-pinned ranks
        if (files_.empty() || !is_gpu_rank()) {
            // nothing to do
            parts_ = {};
            return;
        }

        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);

        if (files_.size() == 1) {
            // single file -> partition by row groups
            parts_ =
                partition_by_row_groups(files_[0], filesystem_, rank_, size_);
        } else {
            // multi-file dataset -> partition by file (block partition)
            parts_ = partition_by_files(files_, filesystem_, rank_, size_);
        }

        estimate_parquet_metadata();

        // Set common parquet reader options.
        chunked_reader_opts.set_columns(selected_columns);
        if (filter_ast_tree.size() > 0) {
            chunked_reader_opts.set_filter(filter_ast_tree.back());
        }
        // Disable Pandas metadata to avoid reordering the index columns.
        chunked_reader_opts.enable_use_pandas_metadata(false);

        chunked_reader_se = make_stream_and_event(g_use_async);
    }

    std::pair<std::unique_ptr<cudf::table>, bool> next(
        std::shared_ptr<StreamAndEvent> se) {
        if (parts_.empty()) {
            // nothing assigned to this rank
            return {empty_table_from_arrow_schema(output_arrow_schema), true};
        }

        std::vector<std::unique_ptr<cudf::table>> gpu_tables;
        size_t rows_accum = 0;
        if (leftover_tbl) {
            std::size_t n = leftover_tbl->num_rows();
            if (n <= target_rows_) {
                rows_accum += n;
                gpu_tables.emplace_back(std::move(leftover_tbl));
                leftover_tbl = nullptr;
            } else {
                cudf::table_view tv = leftover_tbl->view();
                auto batch = cudf::slice(tv, {0, (int)target_rows_},
                                         chunked_reader_se->stream)[0];
                auto remain = cudf::slice(tv, {(int)target_rows_, (int)n},
                                          chunked_reader_se->stream)[0];
                gpu_tables.push_back(std::make_unique<cudf::table>(batch));
                leftover_tbl = std::make_unique<cudf::table>(remain);
                rows_accum = target_rows_;
            }
        }

        // If leftover satisfied the batch, return early
        if (rows_accum >= target_rows_) {
            // Sync batch stream with chunked reader stream.
            chunked_reader_se->event.record(chunked_reader_se->stream);
            se->event.wait(chunked_reader_se->stream);
            return {std::move(gpu_tables[0]), read_finished()};
        }

        if (!curr_reader && has_next_reader()) {
            auto start_get_next_reader = start_timer();
            curr_reader = next_reader();
            this->metrics.get_next_chunked_reader_time +=
                end_timer(start_get_next_reader);
        }

        while (curr_reader && rows_accum < target_rows_) {
            while (curr_reader->has_next() && rows_accum < target_rows_) {
                try {
                    auto table_and_metadata = curr_reader->read_chunk();
                    auto tbl = std::move(table_and_metadata.tbl);

                    if (!tbl || tbl->num_rows() == 0) {
                        // no rows in this row group (rare) — advance
                        continue;
                    }

                    std::size_t tbl_n = tbl->num_rows();
                    if (rows_accum + tbl_n <= target_rows_) {
                        rows_accum += tbl_n;
                        gpu_tables.push_back(std::move(tbl));
                    } else {
                        std::size_t need = target_rows_ - rows_accum;
                        cudf::table_view tv = tbl->view();
                        auto batch = cudf::slice(tv, {0, (int)need},
                                                 chunked_reader_se->stream)[0];
                        auto remain = cudf::slice(tv, {(int)need, (int)tbl_n},
                                                  chunked_reader_se->stream)[0];
                        gpu_tables.push_back(
                            std::make_unique<cudf::table>(batch));
                        leftover_tbl = std::make_unique<cudf::table>(remain);
                        rows_accum = target_rows_;
                    }
                } catch (const std::exception &e) {
                    // reading failed: propagate or print and stop
                    throw std::runtime_error(
                        "PhysicalGPUReadParquet(): read_chunk failed " +
                        parts_[0].path + " " + std::string(e.what()));
                }
            }

            if (!curr_reader->has_next()) {
                // Free the chunked reader memory as soon as we know we won't
                // read from it again
                curr_reader.reset();
                if (has_next_reader()) {
                    auto start_get_next_reader = start_timer();
                    curr_reader = next_reader();
                    this->metrics.get_next_chunked_reader_time +=
                        end_timer(start_get_next_reader);
                }
            }
        }
        // Sync batch stream with chunked reader stream.
        chunked_reader_se->event.record(chunked_reader_se->stream);
        se->event.wait(chunked_reader_se->stream);

        // If we collected no tables (e.g., parts exhausted), return EOF
        if (gpu_tables.empty()) {
            // If we've exhausted all parts, EOF true; otherwise false but no
            // data (shouldn't happen)
            return {empty_table_from_arrow_schema(output_arrow_schema),
                    read_finished()};
        }

        // Concatenate gpu_tables into a single cudf::table
        // Build vector of table_views
        std::vector<cudf::table_view> views;
        views.reserve(gpu_tables.size());
        for (auto &tptr : gpu_tables) {
            views.emplace_back(tptr->view());
        }

        std::unique_ptr<cudf::table> batch;
        if (views.size() == 1) {
            // move the single table out
            batch = std::move(gpu_tables[0]);
        } else {
            // concatenate returns a std::unique_ptr<cudf::table>
            batch = cudf::concatenate(views, se->stream);
        }

        return {std::move(batch), read_finished()};
    }

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        std::string prefix = "batch_gen";
        metrics_out.emplace_back(TimerMetric(prefix + "_get_dataset_time",
                                             this->metrics.get_dataset_time));
        metrics_out.emplace_back(
            TimerMetric(prefix + "_get_next_chunked_reader_time",
                        this->metrics.get_next_chunked_reader_time));
        metrics_out.emplace_back(
            TimerMetric(prefix + "_estimate_parquet_metadata_time",
                        this->metrics.estimate_parquet_metadata_time));
        metrics_out.emplace_back(
            StatMetric(prefix + "_bytes_per_part_estimate",
                       this->metrics.bytes_per_part_estimate));
        metrics_out.emplace_back(
            StatMetric(prefix + "_chunked_reader_limit_bytes",
                       this->metrics.chunked_reader_limit_bytes));
        metrics_out.emplace_back(
            StatMetric(prefix + "_num_readers", this->metrics.num_readers));
    }

   private:
    bool has_next_reader() { return next_part_idx < parts_.size(); }

    bool read_finished() {
        return ((!curr_reader && !has_next_reader()) &&
                (!leftover_tbl || leftover_tbl->num_rows() == 0));
    }

    /**
     * @brief Whether to use Arrow as the datasource for reading parquet files.
     *
     * @return true Use arrow_datasource (read Parquet files into arrow
     * buffers).
     * @return false Let libcudf decide what datasource to use based on path.
     */
    bool use_arrow_source() {
        return (filesystem_->type_name() != "local") &&
               !(USE_KVIKIO_REMOTE_SOURCE && filesystem_->type_name() == "s3");
    }

    std::unique_ptr<cudf::io::chunked_parquet_reader> next_reader() {
        if (next_part_idx >= parts_.size()) {
            // No parts assigned to this rank or we exhausted all parts already
            return nullptr;
        } else if (parts_.size() == 1 && parts_[0].end_row_group.has_value()) {
            // Partition row groups case:
            // TODO: break up single file with multiple row groups that are, all
            // together, larger than CHUNKED_READER_TOTAL_BYTES_LIMIT into
            // multiple readers.
            int start_row_group = parts_[0].start_row_group;
            int end_row_group = parts_[0].end_row_group.value();
            std::vector<int> single_file_row_groups(end_row_group -
                                                    start_row_group);
            std::iota(single_file_row_groups.begin(),
                      single_file_row_groups.end(), start_row_group);
            std::vector<std::vector<int>> row_groups;
            row_groups.push_back(single_file_row_groups);
            chunked_reader_opts.set_row_groups(row_groups);
        }

        this->metrics.num_readers++;

        size_t remaining_parts = parts_.size() - next_part_idx;
        size_t num_parts =
            bytes_per_part_estimate > 0
                ? std::min(remaining_parts,
                           (size_t)std::ceil(
                               (double)CHUNKED_READER_TOTAL_BYTES_LIMIT /
                               bytes_per_part_estimate))
                : remaining_parts;
        std::vector<std::unique_ptr<cudf::io::datasource>> sources;
        for (size_t i = 0; i < num_parts; i++) {
            const auto &part = parts_[next_part_idx + i];
            if (use_arrow_source()) {
                // Use our filesystem to read all parquet parts associated with
                // this reader into arrow buffers.
                // TODO: read only required row groups if single file case and
                // potentially selected columns in all cases.
                std::shared_ptr<arrow::io::RandomAccessFile> arrow_file =
                    filesystem_->OpenInputFile(part.path).ValueOrDie();
                sources.push_back(
                    std::make_unique<arrow_file_datasource>(arrow_file));
            } else {
                std::string path = part.path;
                if (filesystem_->type_name() == "s3") {
                    // Required for kvikio remote source to recognize S3 paths.
                    path = "s3://" + path;
                }
                sources.push_back(cudf::io::datasource::create(path));
            }
        }
        next_part_idx = next_part_idx + num_parts;

        // Parquet metadata will be populated by cudf.
        std::vector<cudf::io::parquet::FileMetaData> metadata;
        return std::make_unique<cudf::io::chunked_parquet_reader>(
            chunked_reader_limit, std::move(sources), std::move(metadata),
            chunked_reader_opts, chunked_reader_se->stream);
    }

    /**
     * @brief Estimate parquet metadata such as bytes per row and bytes per part
     * by sampling a few row groups from the dataset and use estimates to set
     * appropriate chunk sizes for the chunked reader.
     */
    void estimate_parquet_metadata() {
        auto start_estimate_parquet_metadata = start_timer();

        std::mt19937 rng(PARQUET_SAMPLING_RANDOM_SEED);
        std::uniform_int_distribution<size_t> dist(0, parts_.size() - 1);
        size_t num_samples = std::min(
            parts_.size(),
            std::max(PARQUET_SAMPLING_MIN_FILES,
                     (size_t)(parts_.size() * PARQUET_SAMPLING_FRACTION)));
        std::vector<FilePart> sampled_parts;
        std::ranges::sample(parts_.begin(), parts_.end(),
                            std::back_inserter(sampled_parts), num_samples,
                            rng);

        int64_t total_sample_rows = 0;
        int64_t total_sample_bytes = 0;
        for (const auto &part : sampled_parts) {
            std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
            CHECK_ARROW_AND_ASSIGN(
                filesystem_->OpenInputFile(part.path),
                "Error opening file via Arrow filesystem: " + part.path,
                arrow_file);
            std::unique_ptr<parquet::ParquetFileReader> pf =
                parquet::ParquetFileReader::Open(arrow_file);
            std::shared_ptr<parquet::FileMetaData> metadata = pf->metadata();
            int start_row_group = part.start_row_group;
            int end_row_group =
                part.end_row_group.value_or(metadata->num_row_groups());
            for (int i = start_row_group; i < end_row_group; i++) {
                std::unique_ptr<parquet::RowGroupMetaData> rg_meta =
                    metadata->RowGroup(i);
                total_sample_rows += rg_meta->num_rows();
                total_sample_bytes += rg_meta->total_byte_size();
            }
        }

        bytes_per_part_estimate =
            (sampled_parts.size() > 0)
                ? total_sample_bytes / sampled_parts.size()
                : 0;
        chunked_reader_limit =
            (total_sample_rows > 0)
                ? (total_sample_bytes / total_sample_rows) * target_rows_
                : 0;

        this->metrics.estimate_parquet_metadata_time +=
            end_timer(start_estimate_parquet_metadata);
        this->metrics.bytes_per_part_estimate = bytes_per_part_estimate;
        this->metrics.chunked_reader_limit_bytes = chunked_reader_limit;
    }

    /** Get PyArrow parquet dataset from path and populate files_ and
     * filesystem_.
     */
    void get_dataset() {
        auto start_get_dataset = start_timer();

        // import bodo.io.parquet_pio
        PyObjectPtr pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

        // ds = bodo.io.parquet_pio.get_parquet_dataset(path,
        // get_row_counts=False)
        // TODO(Ehsan): pass other options filters, storage_options,
        // read_categories, tot_rows_to_read, schema, partitioning
        PyObjectPtr ds = PyObject_CallMethod(pq_mod, "get_parquet_dataset",
                                             "OO", this->path_, Py_False);
        if (ds == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        PyObjectPtr py_filesystem = PyObject_GetAttrString(ds, "filesystem");

        ensure_pa_wrappers_imported();

        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_filesystem(py_filesystem.get()),
            "Error during Parquet read: Failed to unwrap Arrow filesystem",
            this->filesystem_);

        // all_pieces = ds.pieces
        PyObjectPtr all_pieces = PyObject_GetAttrString(ds, "pieces");

        PyObject *piece;
        // iterate through pieces next
        PyObject *iterator = PyObject_GetIter(all_pieces.get());
        if (iterator == nullptr) {
            throw std::runtime_error(
                "RankBatchGenerator::get_dataset(): error getting pieces "
                "iterator");
        }
        while ((piece = PyIter_Next(iterator))) {
            // p = piece.path
            PyObjectPtr p = PyObject_GetAttrString(piece, "path");
            const char *c_path = PyUnicode_AsUTF8(p.get());
            files_.emplace_back(c_path);
            Py_DECREF(piece);
        }
        Py_DECREF(iterator);

        this->metrics.get_dataset_time += end_timer(start_get_dataset);
    }

    // Partition a single file by row groups into contiguous chunks for each
    // rank
    static std::vector<FilePart> partition_by_row_groups(
        const std::string &file,
        std::shared_ptr<arrow::fs::FileSystem> filesystem, int rank, int size) {
        std::vector<FilePart> result;
        // Use Arrow/Parquet to get num_row_groups cheaply
        int total_rg = 0;
        try {
            // Use Arrow filesystem to read the file into a buffer
            std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
            CHECK_ARROW_AND_ASSIGN(
                filesystem->OpenInputFile(file),
                "Error opening file via Arrow filesystem: " + file, arrow_file);

            std::unique_ptr<parquet::ParquetFileReader> pf =
                parquet::ParquetFileReader::Open(arrow_file);
            total_rg = pf->metadata()->num_row_groups();
        } catch (const std::exception &e) {
            std::cerr << "Failed to read parquet metadata for " << file << ": "
                      << e.what() << "\n";
            return result;
        }

        int base = total_rg / size;
        int rem = total_rg % size;
        int start = rank * base + std::min(rank, rem);
        int end = start + base + (rank < rem ? 1 : 0);
        if (start >= end) {
            return result;
        }
        result.push_back(FilePart{
            .path = file, .start_row_group = start, .end_row_group = end});
        return result;
    }

    // Partition files across ranks by block partitioning
    static std::vector<FilePart> partition_by_files(
        const std::vector<std::string> &files,
        std::shared_ptr<arrow::fs::FileSystem> filesystem, int rank, int size) {
        std::vector<FilePart> result;
        int total = static_cast<int>(files.size());
        int base = total / size;
        int rem = total % size;
        int start = rank * base + std::min(rank, rem);
        int end = start + base + (rank < rem ? 1 : 0);
        for (int i = start; i < end; ++i) {
            const std::string &f = files[i];
            result.push_back(FilePart{.path = f,
                                      .start_row_group = 0,
                                      .end_row_group = std::nullopt});
        }
        return result;
    }

   private:
    PyObject *path_;
    std::shared_ptr<arrow::fs::FileSystem> filesystem_;
    std::size_t target_rows_;
    int rank_{0}, size_{1};

    RankBatchGeneratorMetrics metrics;

    // Filter expressions to apply to read_parquet()
    // NOTE: all expressions and scalars must be kept alive in these data
    // structures since cudf APIs take in references.
    cudf::ast::tree filter_ast_tree;
    std::vector<std::unique_ptr<cudf::scalar>> filter_scalars;

    std::vector<std::string> files_;
    std::vector<FilePart> parts_;
    const std::vector<std::string> &selected_columns;
    // Arrow schema of all columns (not just the selected columns)
    std::shared_ptr<arrow::Schema> arrow_schema;
    // Arrow schema of the selected columns
    std::shared_ptr<arrow::Schema> output_arrow_schema;
    // Current chunked reader responsible for reading a subset of parts assigned
    // to this rank.
    std::unique_ptr<cudf::io::chunked_parquet_reader> curr_reader;
    std::shared_ptr<StreamAndEvent> chunked_reader_se;
    cudf::io::parquet_reader_options chunked_reader_opts;

    // leftover rows from previous oversized row-group read
    std::unique_ptr<cudf::table> leftover_tbl;
    // Keep track of which part we're processing for chunked reader
    size_t next_part_idx = 0;

    // Estimate of bytes per row and bytes per part for determining how many
    // parts can be read by a single chunked reader while respecting the
    // CHUNKED_READER_TOTAL_BYTES_LIMIT.
    int64_t bytes_per_part_estimate = 0;
    // Limit for each chunk read by a single chunked reader.
    size_t chunked_reader_limit = 0;
};

struct PhysicalGPUReadParquetMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

std::shared_ptr<arrow::Schema> MakeNullableSchema(
    std::shared_ptr<arrow::Schema> s) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(s->num_fields());
    for (auto &f : s->fields()) {
        fields.push_back(f->WithNullable(true));
    }
    return arrow::schema(fields, s->metadata());
}

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalGPUReadParquet : public PhysicalGPUSource {
   private:
    // Schema of the output (just the selected columns)
    std::shared_ptr<bodo::Schema> output_schema;

    JoinFilterColStats join_filter_col_stats;

    PyObject *path;
    PyObject *storage_options;
    PyObject *schema_fields;
    const std::vector<int> selected_columns;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    int64_t total_rows_to_read = -1;  // Default to read everything.

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalGPUReadParquet(
        PyObject *py_path, PyObject *pyarrow_schema, PyObject *storage_options,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats)
        : join_filter_col_stats(std::move(join_filter_col_stats)),
          path(py_path),
          storage_options(storage_options),
          selected_columns(selected_columns),
          filter_exprs(filter_exprs.Copy()) {
        time_pt start_init = start_timer();

        Py_INCREF(this->storage_options);

        // Extract metadata from pyarrow schema (for Pandas Index reconstruction
        // of dataframe later)
        arrow_schema = MakeNullableSchema(unwrap_schema(pyarrow_schema));
        this->out_metadata = std::make_shared<bodo::TableMetadata>(
            arrow_schema->metadata()->keys(),
            arrow_schema->metadata()->values());
        this->output_schema = bodo::Schema::FromArrowSchema(arrow_schema)
                                  ->Project(selected_columns);
        this->output_arrow_schema = output_schema->ToArrowSchema();

        this->schema_fields = PyObject_GetAttrString(pyarrow_schema, "names");
        if (!this->schema_fields || !PyList_Check(this->schema_fields)) {
            throw std::runtime_error(
                "PhysicalGPUReadParquet(): failed to get schema fields from "
                "pyarrow schema");
        }

        // ----------------------------------------------------------
        // Handle limit.
        // ----------------------------------------------------------
        if (limit_val) {
            // If the limit option is present...
            if (limit_val->Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
                throw std::runtime_error(
                    "PhysicalGPUReadParquet unsupported limit type");
            }
            // Limit the rows to read to the limit value.
            total_rows_to_read = limit_val->GetConstantValue();
        }

        // Extract column names from pyarrow schema using selected columns
        int num_fields = PyList_Size(this->schema_fields);
        out_column_names.reserve(selected_columns.size());

        for (int col_idx : selected_columns) {
            if (!(col_idx >= 0 && col_idx < num_fields)) {
                throw std::runtime_error(
                    "PhysicalGPUReadParquet(): invalid column index " +
                    std::to_string(col_idx) + " for schema with " +
                    std::to_string(num_fields) + " fields");
            }
            PyObject *name = PyList_GetItem(this->schema_fields, col_idx);
            if (name && PyUnicode_Check(name)) {
                out_column_names.emplace_back(PyUnicode_AsUTF8(name));
            } else {
                out_column_names.push_back("column_" + std::to_string(col_idx));
            }
        }

        this->metrics.init_time += end_timer(start_init);

        this->comm = get_gpu_mpi_comm(get_gpu_id());
    }
    virtual ~PhysicalGPUReadParquet() {
        Py_XDECREF(this->storage_options);
        Py_XDECREF(this->schema_fields);
        if (this->comm != MPI_COMM_NULL) {
            MPI_Comm_free(&this->comm);
        }
    }

    void FinalizeSource() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            this->metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.produce_time);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.rows_read);
    }

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        if (!batch_gen) {
            time_pt start_init = start_timer();
            init_batch_gen();
            this->metrics.init_time += end_timer(start_init);
        }

        // Non-GPU ranks return nullptr to avoid any GPU work
        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, output_arrow_schema, se),
                    OperatorResult::FINISHED};
        }

        time_pt start_produce = start_timer();

        std::pair<std::unique_ptr<cudf::table>, bool> next_batch_tup =
            batch_gen->next(se);

        auto result = next_batch_tup.second ? OperatorResult::FINISHED
                                            : OperatorResult::HAVE_MORE_OUTPUT;
        std::pair<GPU_DATA, OperatorResult> ret = std::make_pair(
            GPU_DATA(std::move(next_batch_tup.first), output_arrow_schema, se),
            result);

        this->metrics.produce_time += end_timer(start_produce);
        return ret;
    }

    /**
     * @brief Get the physical schema of the Parquet data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    // Column names and metadata (Pandas Index info) used for dataframe
    // construction
    std::shared_ptr<bodo::TableMetadata> out_metadata;
    std::vector<std::string> out_column_names;

   private:
    PhysicalGPUReadParquetMetrics metrics;
    std::shared_ptr<RankBatchGenerator> batch_gen;
    // Arrow schema for all the columns in the dataset (not just selected
    // columns)
    std::shared_ptr<arrow::Schema> arrow_schema;
    std::shared_ptr<arrow::Schema> output_arrow_schema;

    // Communicator for GPU ranks (for part assignments)
    MPI_Comm comm;

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
        if (this->batch_gen) {
            this->batch_gen->ReportMetrics(metrics_out);
        }
    }

    void init_batch_gen() {
        auto batch_size = get_gpu_streaming_batch_size();

        this->filter_exprs = join_filter_col_stats.insert_filters(
            std::move(this->filter_exprs), this->selected_columns);

        batch_gen = std::make_shared<RankBatchGenerator>(
            path, filter_exprs, batch_size, output_schema->column_names,
            arrow_schema, output_arrow_schema, comm);
    }
};
