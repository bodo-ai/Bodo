#pragma once

#include <Python.h>
#include <arrow/util/key_value_metadata.h>
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

#include <arrow/io/file.h>
#include <mpi_proto.h>
#include <parquet/arrow/reader.h>  // parquet::ParquetFileReader or parquet::arrow::FileReader
#include <parquet/exception.h>

#include <glob.h>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct FilePart {
    std::string path;
    int start_row_group;  // inclusive
    int end_row_group;    // exclusive
};

class RankBatchGenerator {
   public:
    RankBatchGenerator(std::string dataset_path, std::size_t target_rows,
                       const std::vector<std::string> &_selected_columns,
                       std::shared_ptr<arrow::Schema> _arrow_schema,
                       MPI_Comm comm)
        : path_(std::move(dataset_path)),
          target_rows_(target_rows),
          selected_columns(_selected_columns),
          arrow_schema(_arrow_schema) {
        files_ = list_parquet_files(path_);

        // Only assign parts to GPU-pinned ranks
        if (files_.empty() || comm == MPI_COMM_NULL) {
            // nothing to do
            parts_ = {};
            current_part_idx_ = 0;
            return;
        }

        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);

        if (files_.size() == 1) {
            // single file -> partition by row groups
            parts_ = partition_by_row_groups(files_[0], rank_, size_);
        } else {
            // multi-file dataset -> partition by file (block partition)
            parts_ = partition_by_files(files_, rank_, size_);
        }

        current_part_idx_ = 0;
        current_rg_ = (parts_.empty() ? 0 : parts_[0].start_row_group);
    }

    // next() returns a pair: (cudf::table, eof_flag)
    // If no data assigned to this rank, returns empty table and eof=true.
    std::pair<std::unique_ptr<cudf::table>, bool> next(
        std::shared_ptr<StreamAndEvent> se) {
        if (parts_.empty()) {
            // nothing assigned to this rank
            se->event.record(se->stream);
            return {empty_table_from_arrow_schema(arrow_schema), true};
        }

        // If we've exhausted all parts, signal EOF
        if (current_part_idx_ >= static_cast<int>(parts_.size())) {
            se->event.record(se->stream);
            return {empty_table_from_arrow_schema(arrow_schema), true};
        }

        std::vector<std::unique_ptr<cudf::table>> gpu_tables;
        std::size_t rows_accum = 0;

        if (leftover_tbl) {
            std::size_t n = leftover_tbl->num_rows();
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
            se->event.record(se->stream);
            return {std::move(gpu_tables[0]), false};
        }

        // accumulate whole row groups until target reached or parts exhausted
        while (current_part_idx_ < static_cast<int>(parts_.size()) &&
               rows_accum < target_rows_) {
            const FilePart &part = parts_[current_part_idx_];

            // If current_rg_ reached end of this part, advance to next part
            if (current_rg_ >= part.end_row_group) {
                ++current_part_idx_;
                if (current_part_idx_ < static_cast<int>(parts_.size())) {
                    current_rg_ = parts_[current_part_idx_].start_row_group;
                }
                continue;
            }

            // Read whole row group current_rg_ from part.path into GPU using
            // cuDF Build cudf::io::parquet_reader_options to request a single
            // row group.
            try {
                cudf::io::parquet_reader_options reader_opts =
                    cudf::io::parquet_reader_options::builder(
                        cudf::io::source_info(part.path))
                        .row_groups(std::vector<std::vector<int>>{
                            std::vector<int>{current_rg_}})
                        .build();

                if (selected_columns.size() > 0) {
                    reader_opts.set_columns(selected_columns);
                }
                auto result = cudf::io::read_parquet(reader_opts, se->stream);

                // result.tbl is typically a std::unique_ptr<cudf::table> or
                // cudf::table Adjust the following to match your cuDF version:
                std::unique_ptr<cudf::table> tbl = std::move(result.tbl);

                // advance to next row group in this part
                ++current_rg_;

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
                    auto batch = cudf::slice(tv, {0, (int)need})[0];
                    auto remain = cudf::slice(tv, {(int)need, (int)tbl_n})[0];
                    gpu_tables.push_back(std::make_unique<cudf::table>(batch));
                    leftover_tbl = std::make_unique<cudf::table>(remain);
                    rows_accum = target_rows_;
                }
            } catch (const std::exception &e) {
                // reading failed: propagate or print and stop
                throw std::runtime_error(
                    "PhysicalGPUReadParquet(): read_parquet failed " +
                    part.path + " " + std::string(e.what()));
            }
        }

        // Determine EOF: true if we've consumed all parts and current_rg_ >=
        // end of last part
        bool eof = (current_part_idx_ >= static_cast<int>(parts_.size()) &&
                    (!leftover_tbl || leftover_tbl->num_rows() == 0));

        // If we collected no tables (e.g., parts exhausted), return EOF
        if (gpu_tables.empty()) {
            // If we've exhausted all parts, EOF true; otherwise false but no
            // data (shouldn't happen)
            se->event.record(se->stream);
            return {std::make_unique<cudf::table>(cudf::table_view{}), eof};
        }

        // Concatenate gpu_tables into a single cudf::table
        // Build vector of table_views
        std::vector<cudf::table_view> views;
        views.reserve(gpu_tables.size());
        for (auto &tptr : gpu_tables)
            views.emplace_back(tptr->view());

        std::unique_ptr<cudf::table> batch;
        if (views.size() == 1) {
            // move the single table out
            batch = std::move(gpu_tables[0]);
        } else {
            // concatenate returns a std::unique_ptr<cudf::table>
            batch = cudf::concatenate(views, se->stream);
        }

        se->event.record(se->stream);
        return {std::move(batch), eof};
    }

   private:
    // list parquet files in path (if path is a file, return single element)
    static std::vector<std::string> list_parquet_files(
        const std::string &path) {
        std::vector<std::string> out;
        fs::path p(path);
        if (fs::is_regular_file(p)) {
            out.push_back(p.string());
            return out;
        }
        if (fs::is_directory(p)) {
            for (auto &entry : fs::directory_iterator(p)) {
                if (!entry.is_regular_file())
                    continue;
                auto ext = entry.path().extension().string();
                if (ext == ".parquet" || ext == ".PARQUET" || ext == ".pq" ||
                    ext == ".PQ") {
                    out.push_back(entry.path().string());
                }
            }
            std::sort(out.begin(), out.end());
            return out;
        }
        if (fs::exists(p)) {
            out.push_back(p.string());
        }
        if (path.find('*') != std::string::npos ||
            path.find('?') != std::string::npos ||
            path.find('[') != std::string::npos) {
            // Handle globs.

            glob_t g;
            if (glob(path.c_str(), 0, NULL, &g) == 0) {
                for (size_t i = 0; i < g.gl_pathc; i++) {
                    out.push_back(g.gl_pathv[i]);
                }
            }
            globfree(&g);
            std::sort(out.begin(), out.end());
            return out;
        }

        return out;
    }

    // Partition a single file by row groups into contiguous chunks for each
    // rank
    static std::vector<FilePart> partition_by_row_groups(
        const std::string &file, int rank, int size) {
        std::vector<FilePart> result;
        // Use Arrow/Parquet to get num_row_groups cheaply
        int total_rg = 0;
        try {
            std::shared_ptr<arrow::io::ReadableFile> infile;
            PARQUET_ASSIGN_OR_THROW(infile,
                                    arrow::io::ReadableFile::Open(file));
            std::unique_ptr<parquet::ParquetFileReader> pf =
                parquet::ParquetFileReader::Open(infile);
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
        result.push_back(FilePart{file, start, end});
        return result;
    }

    // Partition files across ranks by block partitioning
    static std::vector<FilePart> partition_by_files(
        const std::vector<std::string> &files, int rank, int size) {
        std::vector<FilePart> result;
        int total = static_cast<int>(files.size());
        int base = total / size;
        int rem = total % size;
        int start = rank * base + std::min(rank, rem);
        int end = start + base + (rank < rem ? 1 : 0);
        for (int i = start; i < end; ++i) {
            const std::string &f = files[i];
            int num_rg = 0;
            try {
                std::shared_ptr<arrow::io::ReadableFile> infile;
                PARQUET_ASSIGN_OR_THROW(infile,
                                        arrow::io::ReadableFile::Open(f));
                std::unique_ptr<parquet::ParquetFileReader> pf =
                    parquet::ParquetFileReader::Open(infile);
                num_rg = pf->metadata()->num_row_groups();
            } catch (const std::exception &e) {
                std::cerr << "Failed to read parquet metadata for " << f << ": "
                          << e.what() << "\n";
                continue;
            }
            result.push_back(FilePart{f, 0, num_rg});
        }
        return result;
    }

   private:
    std::string path_;
    std::size_t target_rows_;
    int rank_{0}, size_{1};

    std::vector<std::string> files_;
    std::vector<FilePart> parts_;
    const std::vector<std::string> &selected_columns;
    std::shared_ptr<arrow::Schema> arrow_schema;

    // current position
    int current_part_idx_{0};
    int current_rg_{0};

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

    std::string path;
    PyObject *storage_options;
    PyObject *schema_fields;
    const std::vector<int> selected_columns;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    int64_t total_rows_to_read = -1;  // Default to read everything.
    std::unique_ptr<CudfExpr> cudfExprTree;

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalGPUReadParquet(
        PyObject *py_path, PyObject *pyarrow_schema, PyObject *storage_options,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats)
        : join_filter_col_stats(std::move(join_filter_col_stats)),
          storage_options(storage_options),
          selected_columns(selected_columns),
          filter_exprs(filter_exprs.Copy()) {
        time_pt start_init = start_timer();

        std::map<int, int> old_to_new_column_map;
        // Generate map of original column indices to selected column indices.
        for (size_t i = 0; i < selected_columns.size(); ++i) {
            old_to_new_column_map.insert({selected_columns[i], i});
        }

        if (filter_exprs.filters.size() != 0) {
            cudfExprTree =
                tableFilterSetToCudf(filter_exprs, old_to_new_column_map);
        }

        if (py_path && PyUnicode_Check(py_path)) {
            path = PyUnicode_AsUTF8(py_path);
        } else {
            throw std::runtime_error(
                "PhysicalGPUReadParquet(): path not a Python unicode string");
        }

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

        std::pair<std::unique_ptr<cudf::table>, bool> next_batch_tup =
            batch_gen->next(se);
        if (cudfExprTree) {
            next_batch_tup.first = cudfExprTree->eval(*(next_batch_tup.first));
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
    }

    void init_batch_gen() {
        auto batch_size = get_gpu_streaming_batch_size();

        batch_gen = std::make_shared<RankBatchGenerator>(
            path, batch_size, output_schema->column_names, arrow_schema, comm);
    }
};
