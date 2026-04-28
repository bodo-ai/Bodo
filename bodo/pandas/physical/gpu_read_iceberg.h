#pragma once

#include <Python.h>
#include <arrow/util/key_value_metadata.h>
#include <cudf/io/datasource.hpp>
#include <memory>
#include <unordered_map>
#include <utility>
#include "../../libs/_utils.h"
#include "../../libs/gpu_utils.h"
#include "../_util.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "gpu_expression.h"
#include "operator.h"

#include <mpi.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#include "../io/arrow_compat.h"
#include "../io/iceberg_helpers.h"
#include "physical/gpu_read_parquet.h"

#include <parquet/metadata.h>
#include <string>
#include <vector>

/**
 * @brief Batch generator for Iceberg on GPU.
 */
class GPUIcebergRankBatchGenerator {
   public:
    GPUIcebergRankBatchGenerator(
        PyObject* catalog, const std::string& table_id,
        PyObject* iceberg_filter, PyObject* iceberg_schema,
        const std::shared_ptr<arrow::Schema>& arrow_schema,
        const int64_t snapshot_id, const std::vector<int>& selected_columns,
        duckdb::TableFilterSet& filter_exprs, std::size_t target_rows,
        MPI_Comm comm, std::shared_ptr<arrow::Schema> output_arrow_schema)
        : target_rows_(target_rows),
          filter_exprs_(filter_exprs.Copy()),
          py_pieces(nullptr),
          py_schema_groups(nullptr),
          selected_columns_(selected_columns),
          output_arrow_schema(std::move(output_arrow_schema)) {
        // Only assign parts to GPU-pinned ranks
        if (!is_gpu_rank()) {
            return;
        }

        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);

        // Initialize CUDF AST filters for row-level filtering
        tableFilterSetToCudfAST(*filter_exprs_, arrow_schema->field_names(),
                                filter_ast_tree, filter_scalars);

        get_dataset(catalog, table_id, arrow_schema, iceberg_filter,
                    iceberg_schema, snapshot_id, target_rows);

        // Distribute pieces
        distribute_pieces();
    }

    std::pair<std::unique_ptr<cudf::table>, bool> next(
        std::shared_ptr<StreamAndEvent> se) {
        if (!is_gpu_rank() ||
            (curr_piece_idx >= pieces_.size() && !leftover_tbl &&
             (!curr_reader || !curr_reader->has_next()))) {
            return {empty_table_from_arrow_schema(output_arrow_schema), true};
        }

        std::vector<std::unique_ptr<cudf::table>> gpu_tables;
        size_t rows_accum = 0;

        // Handle leftover from previous read
        if (leftover_tbl) {
            std::size_t n = leftover_tbl->num_rows();
            if (n <= target_rows_) {
                rows_accum += n;
                gpu_tables.emplace_back(std::move(leftover_tbl));
                leftover_tbl = nullptr;
            } else {
                cudf::table_view tv = leftover_tbl->view();
                auto sliced =
                    cudf::slice(tv, {0, (int)target_rows_, (int)n}, se->stream);
                gpu_tables.push_back(std::make_unique<cudf::table>(sliced[0]));
                leftover_tbl = std::make_unique<cudf::table>(sliced[1]);
                rows_accum = target_rows_;
            }
        }

        while (rows_accum < target_rows_ && curr_piece_idx < pieces_.size()) {
            if (!curr_reader) {
                init_next_reader(se->stream);
            }

            while (curr_reader && curr_reader->has_next() &&
                   rows_accum < target_rows_) {
                auto table_and_metadata = curr_reader->read_chunk();
                auto tbl = std::move(table_and_metadata.tbl);

                if (!tbl || tbl->num_rows() == 0) {
                    continue;
                }

                // Evolve table to match output schema
                auto evolved_tbl = evolve_table(std::move(tbl), se->stream);

                std::size_t tbl_n = evolved_tbl->num_rows();
                if (rows_accum + tbl_n <= target_rows_) {
                    rows_accum += tbl_n;
                    gpu_tables.push_back(std::move(evolved_tbl));
                } else {
                    std::size_t need = target_rows_ - rows_accum;
                    cudf::table_view tv = evolved_tbl->view();
                    auto sliced =
                        cudf::slice(tv, {0, (int)need, (int)tbl_n}, se->stream);
                    gpu_tables.push_back(
                        std::make_unique<cudf::table>(sliced[0]));
                    leftover_tbl = std::make_unique<cudf::table>(sliced[1]);
                    rows_accum = target_rows_;
                }
            }

            if (curr_reader && !curr_reader->has_next()) {
                curr_reader.reset();
                // Move to next set of pieces (already handled by
                // init_next_reader)
            }
        }

        if (gpu_tables.empty()) {
            return {empty_table_from_arrow_schema(output_arrow_schema), true};
        }

        std::vector<cudf::table_view> views;
        views.reserve(gpu_tables.size());
        for (auto& tptr : gpu_tables) {
            views.emplace_back(tptr->view());
        }

        std::unique_ptr<cudf::table> batch;
        if (views.size() == 1) {
            batch = std::move(gpu_tables[0]);
        } else {
            batch = cudf::concatenate(views, se->stream);
        }

        bool finished = (curr_piece_idx >= pieces_.size() && !leftover_tbl &&
                         (!curr_reader || !curr_reader->has_next()));

        return {std::move(batch), finished};
    }

    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        MetricBase::StatValue pieces = pieces_.size();
        metrics_out.emplace_back(StatMetric("num_pieces", pieces));
        metrics_out.emplace_back(
            TimerMetric("evolve_time", this->evolve_time_));
    }

   private:
    void init_next_reader(rmm::cuda_stream_view stream) {
        if (curr_piece_idx >= pieces_.size()) {
            return;
        }

        int64_t schema_group_idx = pieces_[curr_piece_idx].schema_group_idx;
        std::vector<std::string> group_paths;
        while (curr_piece_idx < pieces_.size() &&
               pieces_[curr_piece_idx].schema_group_idx == schema_group_idx) {
            group_paths.push_back(pieces_[curr_piece_idx].path);
            curr_piece_idx++;
        }

        // Get schema group info
        PyObject* schema_group =
            PyList_GetItem(py_schema_groups.get(), schema_group_idx);
        PyObjectPtr read_schema_py =
            PyObject_GetAttrString(schema_group, "read_schema");
        curr_read_schema =
            arrow::py::unwrap_schema(read_schema_py.get()).ValueOrDie();

        // Identify existing columns and their file names
        std::vector<std::string> file_col_names;
        curr_col_mapping.clear();

        for (size_t i = 0; i < selected_columns_.size(); i++) {
            int col_idx = selected_columns_[i];
            std::string name = curr_read_schema->field(col_idx)->name();
            if (name.starts_with("_BODO_TEMP_")) {
                // Column doesn't exist in file
                curr_col_mapping.emplace_back(-1, i);
            } else {
                curr_col_mapping.emplace_back((int)file_col_names.size(), i);
                file_col_names.push_back(name);
            }
        }

        cudf::io::parquet_reader_options opts =
            cudf::io::parquet_reader_options::builder(cudf::io::source_info());
        opts.set_column_names(file_col_names);
        if (filter_ast_tree.size() > 0) {
            opts.set_filter(filter_ast_tree.back());
        }

        std::vector<std::unique_ptr<cudf::io::datasource>> sources;
        for (const auto& path : group_paths) {
            if (filesystem_->type_name() != "local") {
                std::shared_ptr<arrow::io::RandomAccessFile> arrow_file =
                    filesystem_->OpenInputFile(path).ValueOrDie();
                sources.push_back(
                    std::make_unique<arrow_file_datasource>(arrow_file));
            } else {
                sources.push_back(cudf::io::datasource::create(path));
            }
        }

        curr_reader = std::make_unique<cudf::io::chunked_parquet_reader>(
            0,  // chunk_read_limit, 0 for default
            std::move(sources), std::vector<cudf::io::parquet::FileMetaData>(),
            opts, stream);
    }

    std::unique_ptr<cudf::column> evolve_column(
        std::unique_ptr<cudf::column> col,
        const std::shared_ptr<arrow::Field>& source_field,
        const std::shared_ptr<arrow::Field>& target_field,
        rmm::cuda_stream_view stream) {
        // Verify that the Iceberg field ID is the same.
        int source_field_iceberg_field_id = get_iceberg_field_id(source_field);
        int target_field_iceberg_field_id = get_iceberg_field_id(target_field);
        if (source_field_iceberg_field_id != target_field_iceberg_field_id) {
            throw std::runtime_error(
                "GPUIcebergRankBatchGenerator::evolve_column: Iceberg field ID "
                "of the source (" +
                std::to_string(source_field_iceberg_field_id) +
                ") and target (" +
                std::to_string(target_field_iceberg_field_id) +
                ") fields do not match!");
        }

        if (target_field->type()->id() == arrow::Type::STRUCT) {
            // Recursive evolution for Struct
            auto source_type = std::dynamic_pointer_cast<arrow::StructType>(
                source_field->type());
            auto target_type = std::dynamic_pointer_cast<arrow::StructType>(
                target_field->type());

            cudf::size_type num_row = col->size();
            cudf::size_type null_count = col->null_count();
            cudf::column::contents col_data = col->release();

            std::vector<std::unique_ptr<cudf::column>> source_children =
                std::move(col_data.children);
            std::vector<std::unique_ptr<cudf::column>> evolved_children;

            std::unordered_map<int, int> source_id_to_idx;
            for (int i = 0; i < source_type->num_fields(); i++) {
                source_id_to_idx[get_iceberg_field_id(source_type->field(i))] =
                    i;
            }

            for (int i = 0; i < target_type->num_fields(); i++) {
                auto target_child_field = target_type->field(i);
                int id = get_iceberg_field_id(target_child_field);
                if (source_id_to_idx.contains(id)) {
                    int source_idx = source_id_to_idx[id];
                    evolved_children.push_back(
                        evolve_column(std::move(source_children[source_idx]),
                                      source_type->field(source_idx),
                                      target_child_field, stream));
                } else {
                    // Add null child
                    auto null_scalar = arrow_scalar_to_cudf(
                        arrow::MakeNullScalar(target_child_field->type()),
                        stream);
                    evolved_children.push_back(cudf::make_column_from_scalar(
                        *null_scalar, num_row, stream));
                }
            }

            rmm::device_buffer null_mask = col_data.null_mask
                                               ? std::move(*col_data.null_mask)
                                               : rmm::device_buffer{0, stream};

            return cudf::make_structs_column(
                num_row, std::move(evolved_children), null_count,
                std::move(null_mask), stream);
        }

        // For other types (List, Map, Primitive), we might need more logic
        // For now, return as is if types match
        return col;
    }

    std::unique_ptr<cudf::table> evolve_table(std::unique_ptr<cudf::table> tbl,
                                              rmm::cuda_stream_view stream) {
        time_pt start_evolve = start_timer();
        // We need the read_schema to get the source fields for evolution
        // We can store it in curr_read_schema
        std::vector<std::unique_ptr<cudf::column>> output_cols;
        output_cols.reserve(selected_columns_.size());

        auto released_cols = tbl->release();

        for (size_t i = 0; i < selected_columns_.size(); i++) {
            int col_idx = selected_columns_[i];
            const auto& mapping = curr_col_mapping[i];
            auto target_field = output_arrow_schema->field(i);

            if (mapping.first == -1) {
                // Add null column
                auto arrow_type = target_field->type();
                auto null_scalar = arrow_scalar_to_cudf(
                    arrow::MakeNullScalar(arrow_type), stream);
                output_cols.push_back(cudf::make_column_from_scalar(
                    *null_scalar, tbl->num_rows(), stream));
            } else {
                auto source_field = curr_read_schema->field(col_idx);
                output_cols.push_back(
                    evolve_column(std::move(released_cols[mapping.first]),
                                  source_field, target_field, stream));
            }
        }

        auto ret = std::make_unique<cudf::table>(std::move(output_cols));
        this->evolve_time_ += end_timer(start_evolve);
        return ret;
    }

    void get_dataset(PyObject* catalog, const std::string& table_id,
                     const std::shared_ptr<arrow::Schema>& arrow_schema,
                     PyObject* iceberg_filter, PyObject* iceberg_schema,
                     int64_t snapshot_id, int64_t tot_rows_to_read) {
        // import bodo.io.iceberg
        PyObjectPtr iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        // For now, we don't support dict-encoded strings on GPU yet (or we'll
        // handle them as strings)
        PyObjectPtr str_as_dict_cols_py = PyList_New(0);

        // We use piece-level read for GPU to avoid expensive exact row counting
        // on CPU if possible.
        PyObject* force_row_level_py = Py_False;

        PyObjectPtr pyarrow_schema = arrow::py::wrap_schema(arrow_schema);

        // We need to & the iceberg_filter with converted duckdb table filters
        // to apply the filters at the file level.
        PyObjectPtr duckdb_iceberg_filter =
            duckdbFilterSetToPyicebergFilter(*filter_exprs_, arrow_schema);

        // Perform the python & to combine the filters
        PyObjectPtr py_iceberg_filter_and_duckdb_filter =
            PyObjectPtr(PyObject_CallMethod(duckdb_iceberg_filter.get(),
                                            "__and__", "O", iceberg_filter));
        if (!py_iceberg_filter_and_duckdb_filter) {
            throw std::runtime_error(
                "failed to combine iceberg filter with duckdb table filters");
        }

        // We need to convert the combined pyiceberg iceberg filter to an Arrow
        // filter format string so it can be applied at the row level in
        // addition to the file level.
        //  import bodo.io.iceberg.common
        PyObjectPtr iceberg_common_mod =
            PyImport_ImportModule("bodo.io.iceberg.common");
        if (PyErr_Occurred()) {
            throw std::runtime_error(
                "failed to import bodo.io.iceberg.common module");
        }

        PyObjectPtr convert_func = PyObject_GetAttrString(
            iceberg_common_mod.get(),
            "pyiceberg_filter_to_pyarrow_format_str_and_scalars");
        if (!convert_func || !PyCallable_Check(convert_func)) {
            throw std::runtime_error(
                "failed to get convert_iceberg_filter_to_arrow function from "
                "bodo.io.iceberg.common module");
        }

        // call python function to convert the combined pyiceberg filter
        PyObjectPtr filter_f_str_and_scalars =
            PyObjectPtr(PyObject_CallFunctionObjArgs(
                convert_func.get(), py_iceberg_filter_and_duckdb_filter.get(),
                iceberg_schema, Py_True, nullptr));
        if (!filter_f_str_and_scalars) {
            throw std::runtime_error(
                "failed to convert pyiceberg filter to arrow filter format "
                "string");
        }

        // The result is a tuple of (iceberg_filter_f_str, filter_scalars)
        PyObject* iceberg_filter_f_str_py =
            PyTuple_GetItem(filter_f_str_and_scalars.get(), 0);
        PyObject* filter_scalars_py =
            PyTuple_GetItem(filter_f_str_and_scalars.get(), 1);

        PyObjectPtr ds = PyObject_CallMethod(
            iceberg_mod.get(), "get_iceberg_pq_dataset", "OsOOOsOOLL", catalog,
            table_id.c_str(), pyarrow_schema.get(), str_as_dict_cols_py.get(),
            py_iceberg_filter_and_duckdb_filter.get(),
            iceberg_filter_f_str_py != Py_None
                ? PyUnicode_AsUTF8(iceberg_filter_f_str_py)
                : "",
            filter_scalars_py, force_row_level_py,
            static_cast<long long>(snapshot_id),
            static_cast<long long>(tot_rows_to_read));

        if (ds == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        PyObjectPtr py_filesystem = PyObject_GetAttrString(ds, "filesystem");
        ensure_pa_wrappers_imported();
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_filesystem(py_filesystem.get()),
            "Error during Iceberg read: Failed to unwrap Arrow filesystem",
            this->filesystem_);

        this->py_pieces = PyObject_GetAttrString(ds.get(), "pieces");
        this->py_schema_groups =
            PyObject_GetAttrString(ds.get(), "schema_groups");
    }

    void distribute_pieces() {
        // import bodo.io.iceberg.read_parquet
        PyObjectPtr iceberg_mod =
            PyImport_ImportModule("bodo.io.iceberg.read_parquet");
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        // pieces_myrank_py =
        // bodo.io.iceberg.read_parquet.distribute_pieces(pieces_py)
        PyObjectPtr pieces_myrank_py = PyObject_CallMethod(
            iceberg_mod.get(), "distribute_pieces", "O", this->py_pieces.get());
        if (pieces_myrank_py == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        PyObject* piece;
        PyObject* iterator = PyObject_GetIter(pieces_myrank_py.get());
        if (iterator == nullptr) {
            throw std::runtime_error(
                "GPUIcebergRankBatchGenerator::distribute_pieces(): error "
                "getting pieces iterator");
        }

        while ((piece = PyIter_Next(iterator))) {
            // Add piece to our local list
            // Extract path, num_rows, schema_group_idx
            PyObjectPtr p = PyObject_GetAttrString(piece, "path");
            const char* c_path = PyUnicode_AsUTF8(p.get());

            PyObjectPtr num_rows_piece_py =
                PyObject_GetAttrString(piece, "_bodo_num_rows");
            int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py.get());

            PyObjectPtr schema_group_idx_py =
                PyObject_GetAttrString(piece, "schema_group_idx");
            int64_t schema_group_idx =
                PyLong_AsLongLong(schema_group_idx_py.get());

            this->pieces_.push_back({c_path, num_rows_piece, schema_group_idx});

            Py_DECREF(piece);
        }
        Py_DECREF(iterator);
    }

    struct IcebergPieceInfo {
        std::string path;
        int64_t num_rows;
        int64_t schema_group_idx;
    };

    std::size_t target_rows_;
    int rank_{0}, size_{1};

    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs_;
    cudf::ast::tree filter_ast_tree;
    std::vector<std::unique_ptr<cudf::scalar>> filter_scalars;

    std::shared_ptr<arrow::fs::FileSystem> filesystem_;
    PyObjectPtr py_pieces;
    PyObjectPtr py_schema_groups;

    std::vector<IcebergPieceInfo> pieces_;
    const std::vector<int>& selected_columns_;
    std::shared_ptr<arrow::Schema> output_arrow_schema;

    size_t curr_piece_idx = 0;
    std::unique_ptr<cudf::io::chunked_parquet_reader> curr_reader;
    std::shared_ptr<arrow::Schema> curr_read_schema;
    std::unique_ptr<cudf::table> leftover_tbl;
    std::vector<std::pair<int, int>> curr_col_mapping;
    MetricBase::TimerValue evolve_time_ = 0;
};

struct PhysicalGPUReadIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
    time_t evolve_time = 0;
};

class PhysicalGPUReadIceberg : public PhysicalGPUSource {
   private:
    PyObject* catalog;
    const std::string table_id;
    PyObject* iceberg_filter;
    PyObject* iceberg_schema;
    const int64_t snapshot_id;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    const std::vector<int> selected_columns;

    std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> output_arrow_schema;
    std::vector<std::string> out_column_names;
    std::shared_ptr<bodo::TableMetadata> out_metadata;

    PhysicalGPUReadIcebergMetrics metrics;
    std::shared_ptr<GPUIcebergRankBatchGenerator> batch_gen;
    MPI_Comm comm;

   public:
    explicit PhysicalGPUReadIceberg(
        PyObject* catalog, const std::string table_id, PyObject* iceberg_filter,
        PyObject* iceberg_schema,
        const std::shared_ptr<arrow::Schema> arrow_schema,
        const int64_t snapshot_id, const std::vector<int>& selected_columns,
        duckdb::TableFilterSet& filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode>& limit_val,
        JoinFilterColStats join_filter_col_stats)
        : catalog(catalog),
          table_id(table_id),
          iceberg_filter(iceberg_filter),
          iceberg_schema(iceberg_schema),
          snapshot_id(snapshot_id),
          filter_exprs(filter_exprs.Copy()),
          arrow_schema(arrow_schema),
          selected_columns(selected_columns) {
        Py_INCREF(this->catalog);
        Py_INCREF(this->iceberg_filter);
        Py_INCREF(this->iceberg_schema);

        this->filter_exprs = join_filter_col_stats.insert_filters(
            std::move(this->filter_exprs), this->selected_columns);

        // Initialize schemas and metadata
        this->output_schema = bodo::Schema::FromArrowSchema(arrow_schema)
                                  ->Project(selected_columns);
        this->output_arrow_schema = output_schema->ToArrowSchema();
        this->out_metadata = std::make_shared<bodo::TableMetadata>(
            arrow_schema->metadata()->keys(),
            arrow_schema->metadata()->values());

        for (int i : selected_columns) {
            out_column_names.push_back(arrow_schema->field(i)->name());
        }

        this->comm = get_gpu_mpi_comm(get_gpu_id());
    }

    virtual ~PhysicalGPUReadIceberg() {
        Py_XDECREF(this->catalog);
        Py_XDECREF(this->iceberg_filter);
        Py_XDECREF(this->iceberg_schema);
        if (this->comm != MPI_COMM_NULL) {
            MPI_Comm_free(&this->comm);
        }
    }

    void FinalizeSource() override {
        std::vector<MetricBase> metrics_out;
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
        if (this->batch_gen) {
            this->batch_gen->ReportMetrics(metrics_out);
        }

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

        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, output_arrow_schema, se),
                    OperatorResult::FINISHED};
        }

        time_pt start_produce = start_timer();
        auto next_batch_tup = batch_gen->next(se);
        auto result = next_batch_tup.second ? OperatorResult::FINISHED
                                            : OperatorResult::HAVE_MORE_OUTPUT;

        GPU_DATA ret(std::move(next_batch_tup.first), output_arrow_schema, se);
        this->metrics.produce_time += end_timer(start_produce);
        return {std::move(ret), result};
    }

    const std::shared_ptr<bodo::Schema> getOutputSchemaInternal() override {
        return output_schema;
    }

   private:
    void init_batch_gen() {
        auto batch_size = get_gpu_streaming_batch_size();
        batch_gen = std::make_shared<GPUIcebergRankBatchGenerator>(
            catalog, table_id, iceberg_filter, iceberg_schema, arrow_schema,
            snapshot_id, selected_columns, *filter_exprs, batch_size, comm,
            output_arrow_schema);
    }
};
