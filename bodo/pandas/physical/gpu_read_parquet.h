#pragma once

#include <Python.h>
#include <arrow/util/key_value_metadata.h>
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/scalar/scalar.hpp>
#include <memory>
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
#include <mpi_proto.h>
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

const bool PARQUET_READ_ALL = false;

struct FilePart {
    std::string path;
    int start_row_group;  // inclusive
    int end_row_group;    // exclusive
    size_t total_bytes;   // Total number of uncompressed bytes
    size_t total_rows;
};

class RankBatchGenerator {
   public:
    struct {
        using time_t = MetricBase::TimerValue;
        time_t get_dataset_time = 0;
        time_t partition_file_time = 0;
    } metrics;

    RankBatchGenerator(PyObject *dataset_path,
                       duckdb::unique_ptr<duckdb::TableFilterSet> &filter_exprs,
                       std::size_t target_rows,
                       const std::vector<std::string> &_selected_columns,
                       std::shared_ptr<arrow::Schema> _arrow_schema,
                       MPI_Comm comm)
        : path_(dataset_path),
          filesystem_(nullptr),
          target_rows_(target_rows),
          selected_columns(_selected_columns),
          arrow_schema(std::move(_arrow_schema)) {
        tableFilterSetToCudfAST(*filter_exprs, arrow_schema->field_names(),
                                filter_ast_tree, filter_scalars);

        get_dataset();

        // Only assign parts to GPU-pinned ranks
        if (files_.empty() || comm == MPI_COMM_NULL) {
            // nothing to do
            parts_ = {};
            return;
        }

        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);

        auto start_partition_dataset = start_timer();
        if (files_.size() == 1) {
            // single file -> partition by row groups
            parts_ =
                partition_by_row_groups(files_[0], filesystem_, rank_, size_);

            int start_row_group = parts_[0].start_row_group;
            int end_row_group = parts_[0].end_row_group;
            std::vector<int> single_file_row_groups(end_row_group -
                                                    start_row_group);
            std::iota(single_file_row_groups.begin(),
                      single_file_row_groups.end(), start_row_group);
            std::vector<std::vector<int>> row_groups;
            row_groups.push_back(single_file_row_groups);
            chunked_reader_opts.set_row_groups(row_groups);
        } else {
            // multi-file dataset -> partition by file (block partition)
            parts_ = partition_by_files(files_, filesystem_, rank_, size_);
        }
        this->metrics.partition_file_time += end_timer(start_partition_dataset);

        chunked_reader_stream = cudf::get_default_stream();
        size_t total_rows = 0;
        size_t total_bytes = 0;
        std::vector<std::string> paths;
        for (const auto &part : parts_) {
            std::string path = part.path;
            total_rows += part.total_rows;
            total_bytes += part.total_bytes;
            // Use KVIO for S3
            // TODO: Custom data source for better performance (e.g. Arrow
            // filesystem input stream) ?
            if (filesystem_->type_name() == "s3") {
                path = "s3://" + path;
            }
            paths.push_back(path);
        }
        // Estimate bytes per row
        // TODO: better estimate or use row_group selector + read_parquet
        // The 2x multiple accounts for GPU expansion factor.
        size_t chunked_reader_limit =
            2 * (total_bytes / total_rows) * target_rows_;
        std::cout << "Limit : " << chunked_reader_limit << std::endl;
        cudf::io::source_info src_info(paths);
        chunked_reader_opts.set_source(src_info);

        chunked_reader_opts.set_columns(selected_columns);
        if (filter_ast_tree.size() > 0) {
            chunked_reader_opts.set_filter(filter_ast_tree.back());
        }
        chunked_reader = std::make_unique<cudf::io::chunked_parquet_reader>(
            chunked_reader_limit, chunked_reader_opts, chunked_reader_stream);
    }

    // Read multiple parts at a time for better performance
    std::pair<std::unique_ptr<cudf::table>, bool> read_all(
        std::shared_ptr<StreamAndEvent> se) {
        std::cout << " Reading dataset with " << parts_.size()
                  << " parts assigned to this rank\n";
        if (parts_.empty()) {
            // nothing assigned to this rank
            return {empty_table_from_arrow_schema(arrow_schema), true};
        }

        cudf::io::parquet_reader_options reader_opts;
        std::vector<std::string> paths;
        for (const auto &part : parts_) {
            std::string path = part.path;
            // Use KVIO for S3 (TODO: KVIO too slow?)
            if (filesystem_->type_name() == "s3") {
                path = "s3://" + path;
            }
            paths.push_back(path);
        }
        cudf::io::source_info src_info(paths);
        reader_opts.set_source(src_info);

        reader_opts.set_columns(selected_columns);
        if (filter_ast_tree.size() > 0) {
            reader_opts.set_filter(filter_ast_tree.back());
        }
        std::cout << "  Performing read" << std::endl;
        auto result = cudf::io::read_parquet(reader_opts, se->stream);

        std::unique_ptr<cudf::table> tbl = std::move(result.tbl);
        std::cout << "  Read complete" << std::endl;
        return {std::move(tbl), true};
    }

    bool read_finished() {
        return (!chunked_reader->has_next()) &&
               (!leftover_tbl || leftover_tbl->num_rows() == 0);
    }

    std::pair<std::unique_ptr<cudf::table>, bool> next(
        std::shared_ptr<StreamAndEvent> se) {
        if (parts_.empty()) {
            // nothing assigned to this rank
            return {empty_table_from_arrow_schema(arrow_schema), true};
        }

        if (!this->chunked_reader) {
            throw std::runtime_error("Chunked reader not initialized");
        }

        std::vector<std::unique_ptr<cudf::table>> gpu_tables;
        size_t rows_accum = 0;
        if (leftover_tbl) {
            std::size_t n = leftover_tbl->num_rows();
            std::cout << " leftover table nrows " << n << std::endl;
            if (n <= target_rows_) {
                rows_accum += n;
                gpu_tables.emplace_back(std::move(leftover_tbl));
            } else {
                cudf::table_view tv = leftover_tbl->view();
                auto batch = cudf::slice(tv, {0, (int)target_rows_})[0];
                auto remain = cudf::slice(tv, {(int)target_rows_, (int)n})[0];
                gpu_tables.push_back(std::make_unique<cudf::table>(batch));
                leftover_tbl = std::make_unique<cudf::table>(remain);
                rows_accum = target_rows_;
            }
        }

        // If leftover satisfied the batch, return early
        if (rows_accum >= target_rows_) {
            std::cout << " returning leftover table rows | read finished = "
                      << read_finished() << std::endl;
            return {std::move(gpu_tables[0]), read_finished()};
        }

        while (chunked_reader->has_next() && rows_accum < target_rows_) {
            try {
                auto table_and_metadata = chunked_reader->read_chunk();
                auto tbl = std::move(table_and_metadata.tbl);
                std::cout << "##### Read " << tbl->num_rows()
                          << " rows from: " << parts_[0].path << "#####"
                          << std::endl;

                if (!tbl || tbl->num_rows() == 0) {
                    // no rows in this row group (rare) — advance
                    continue;
                }

                std::size_t tbl_n = tbl->num_rows();
                if (rows_accum + tbl_n <= target_rows_) {
                    std::cout
                        << " Adding table to GPU tables until batch is filled "
                        << std::endl;
                    rows_accum += tbl_n;
                    gpu_tables.push_back(std::move(tbl));
                } else {
                    std::cout << " Read too many rows, slicing tables"
                              << std::endl;
                    std::size_t need = target_rows_ - rows_accum;
                    cudf::table_view tv = tbl->view();
                    auto batch = cudf::slice(tv, {0, (int)need})[0];
                    auto remain = cudf::slice(tv, {(int)need, (int)tbl_n})[0];
                    gpu_tables.push_back(std::make_unique<cudf::table>(batch));
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

        // If we collected no tables (e.g., parts exhausted), return EOF
        if (gpu_tables.empty()) {
            // If we've exhausted all parts, EOF true; otherwise false but no
            // data (shouldn't happen)
            std::cout << " No GPU tables read | read finished = "
                      << read_finished() << std::endl;
            return {std::make_unique<cudf::table>(cudf::table_view{}),
                    read_finished()};
        }

        // Concatenate gpu_tables into a single cudf::table
        // Build vector of table_views
        std::cout << "concatenating " << gpu_tables.size() << " GPU tables "
                  << std::endl;
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

        std::cout << " read parquet returning batch | read finished =  "
                  << read_finished() << std::endl;
        return {std::move(batch), read_finished()};
    }

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        std::string prefix = "gpu_read_parquet_batch_generator";
        metrics_out.emplace_back(TimerMetric(prefix + "_get_dataset_time",
                                             this->metrics.get_dataset_time));
        metrics_out.emplace_back(
            TimerMetric(prefix + "_partition_file_time",
                        this->metrics.partition_file_time));
    }

   private:
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

    /**
     * @brief
     *
     */
    static std::tuple<size_t, size_t> get_num_rows_bytes(
        const parquet::FileMetaData &metadata, int start_row_group,
        int end_row_group) {
        size_t total_rows = 0;
        size_t total_bytes = 0;

        for (int i = start_row_group; i < end_row_group; i++) {
            std::unique_ptr<parquet::RowGroupMetaData> rg_meta =
                metadata.RowGroup(i);
            total_rows += rg_meta->num_rows();
            total_bytes += rg_meta->total_byte_size();
        }

        return {total_rows, total_bytes};
    }

    // Partition a single file by row groups into contiguous chunks for each
    // rank
    static std::vector<FilePart> partition_by_row_groups(
        const std::string &file,
        std::shared_ptr<arrow::fs::FileSystem> filesystem, int rank, int size) {
        std::vector<FilePart> result;
        // Use Arrow/Parquet to get num_row_groups cheaply
        int total_rg = 0;
        std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
        std::unique_ptr<parquet::ParquetFileReader> pf;
        std::shared_ptr<parquet::FileMetaData> metadata;
        try {
            // Use Arrow filesystem to read the file into a buffer
            CHECK_ARROW_AND_ASSIGN(
                filesystem->OpenInputFile(file),
                "Error opening file via Arrow filesystem: " + file, arrow_file);

            pf = parquet::ParquetFileReader::Open(arrow_file);
            metadata = pf->metadata();
            total_rg = metadata->num_row_groups();
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
        auto num_rows_bytes = get_num_rows_bytes(*metadata, start, end);
        size_t part_num_rows = std::get<0>(num_rows_bytes);
        size_t part_num_bytes = std::get<1>(num_rows_bytes);
        result.push_back(FilePart{.path = file,
                                  .start_row_group = start,
                                  .end_row_group = end,
                                  .total_bytes = part_num_bytes,
                                  .total_rows = part_num_rows});
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
            int num_rg = 0;
            size_t num_rows = 0;
            size_t num_bytes = 0;
            try {
                std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
                CHECK_ARROW_AND_ASSIGN(
                    filesystem->OpenInputFile(f),
                    "Error opening file via Arrow filesystem: " + f,
                    arrow_file);
                std::unique_ptr<parquet::ParquetFileReader> pf =
                    parquet::ParquetFileReader::Open(arrow_file);
                std::shared_ptr<parquet::FileMetaData> metadata =
                    pf->metadata();
                num_rg = metadata->num_row_groups();
                auto num_rows_bytes = get_num_rows_bytes(*metadata, 0, num_rg);
                num_rows = std::get<0>(num_rows_bytes);
                num_bytes = std::get<1>(num_rows_bytes);
            } catch (const std::exception &e) {
                std::cerr << "Failed to read parquet metadata for " << f << ": "
                          << e.what() << "\n";
                continue;
            }
            result.push_back(FilePart{.path = f,
                                      .start_row_group = 0,
                                      .end_row_group = num_rg,
                                      .total_bytes = num_bytes,
                                      .total_rows = num_rows});
        }
        return result;
    }

   private:
    PyObject *path_;
    std::shared_ptr<arrow::fs::FileSystem> filesystem_;
    std::size_t target_rows_;
    int rank_{0}, size_{1};

    // Filter expressions to apply to read_parquet()
    // NOTE: all expressions and scalars must be kept alive in these data
    // structures since cudf APIs take in references.
    cudf::ast::tree filter_ast_tree;
    std::vector<std::unique_ptr<cudf::scalar>> filter_scalars;

    std::vector<std::string> files_;
    std::vector<FilePart> parts_;
    const std::vector<std::string> &selected_columns;
    std::shared_ptr<arrow::Schema> arrow_schema;
    std::unique_ptr<cudf::io::chunked_parquet_reader> chunked_reader;
    cudaStream_t chunked_reader_stream;
    cudf::io::parquet_reader_options chunked_reader_opts;

    // leftover rows from previous oversized row-group read
    std::unique_ptr<cudf::table> leftover_tbl;
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

        time_pt start_produce = start_timer();

        std::pair<std::unique_ptr<cudf::table>, bool> next_batch_tup;
        if (PARQUET_READ_ALL) {
            next_batch_tup = batch_gen->read_all(se);
        } else {
            next_batch_tup = batch_gen->next(se);
        }

        auto result = next_batch_tup.second ? OperatorResult::FINISHED
                                            : OperatorResult::HAVE_MORE_OUTPUT;
        std::pair<GPU_DATA, OperatorResult> ret = std::make_pair(
            GPU_DATA(std::move(next_batch_tup.first), arrow_schema, se),
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
    std::shared_ptr<arrow::Schema> arrow_schema;

    // Communicator for GPU ranks (for part assignments)
    MPI_Comm comm;

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
        this->batch_gen->ReportMetrics(metrics_out);
    }

    void init_batch_gen() {
        auto batch_size = get_gpu_streaming_batch_size();

        this->filter_exprs = join_filter_col_stats.insert_filters(
            std::move(this->filter_exprs), this->selected_columns);

        batch_gen = std::make_shared<RankBatchGenerator>(
            path, filter_exprs, batch_size, output_schema->column_names,
            arrow_schema, comm);
    }
};
