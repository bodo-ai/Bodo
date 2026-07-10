#include "physical/gpu_read_iceberg.h"

// GPU-side reader for Iceberg tables. Each GPU rank processes a subset of
// Parquet files, applying DuckDB and Iceberg filters at the row level via
// cuDF AST, then evolves column schemas to match the Iceberg table schema.

#include <Python.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <longobject.h>
#include <mpi.h>
#include <parquet/metadata.h>
#include <parquet/schema.h>
#include <algorithm>
#include <atomic>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../_util.h"
#include "../io/arrow_compat.h"
#include "../io/iceberg_helpers.h"
#include "../libs/_query_profile_collector.h"
#include "physical/gpu_expression.h"
#include "physical/gpu_read_parquet.h"
#include "physical/operator.h"

GPUIcebergRankBatchGenerator::GPUIcebergRankBatchGenerator(
    PyObject* catalog, const std::string& table_id, PyObject* iceberg_filter,
    PyObject* iceberg_schema,
    const std::shared_ptr<arrow::Schema>& arrow_schema,
    const int64_t snapshot_id, const std::vector<int>& selected_columns,
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs,
    std::size_t target_rows, MPI_Comm comm,
    std::shared_ptr<arrow::Schema> output_arrow_schema)
    : target_rows_(target_rows),
      filter_exprs_(std::move(filter_exprs)),
      py_pieces(nullptr),
      py_schema_groups(nullptr),
      pyarrow_schema(nullptr),
      py_filesystem(nullptr),
      selected_columns_(selected_columns),
      output_arrow_schema(std::move(output_arrow_schema)) {
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);

    get_dataset(catalog, table_id, arrow_schema, iceberg_filter, iceberg_schema,
                snapshot_id, target_rows, comm);

    // Non-GPU ranks only participate in collective MPI ops inside get_dataset.
    if (!is_gpu_rank()) {
        return;
    }

    distribute_pieces();
    init_scanners(comm);

    // Subdivide schema groups by physical Parquet schema so that files
    // with mismatched column types (e.g. int32 vs int64 from schema
    // evolution) are not batched into the same chunked reader.
    time_pt start_fp = start_timer();
    compute_physical_schema_fingerprints();
    this->fingerprint_time_ += end_timer(start_fp);

    // Estimate bytes per piece and compute an appropriate chunk size
    // by sampling parquet file metadata, reusing the logic from
    // estimate_parquet_metadata in gpu_read_parquet.h.
    if (!pieces_.empty()) {
        std::vector<FilePart> file_parts;
        file_parts.reserve(pieces_.size());
        for (const auto& piece : pieces_) {
            file_parts.push_back(FilePart{
                .path = piece.path,
                .start_row_group = 0,
                .end_row_group = std::nullopt,
            });
        }
        ParquetMetadataEstimate est =
            estimate_parquet_metadata(filesystem_, file_parts, target_rows_);
        bytes_per_part_estimate_ = est.bytes_per_part;
        chunked_reader_limit_ = est.chunked_reader_limit;
    }
    chunked_reader_se = make_stream_and_event(g_use_async);
}

std::pair<std::unique_ptr<cudf::table>, bool>
GPUIcebergRankBatchGenerator::next(std::shared_ptr<StreamAndEvent> se) {
    if (!is_gpu_rank() || (curr_piece_idx >= pieces_.size() && !leftover_tbl &&
                           (!curr_reader || !curr_reader->has_next()))) {
        return {empty_table_from_arrow_schema(output_arrow_schema), true};
    }

    std::vector<std::unique_ptr<cudf::table>> gpu_tables;
    size_t rows_accum = 0;

    // Consume leftover rows from the previous batch first.
    if (leftover_tbl) {
        std::size_t n = leftover_tbl->num_rows();
        if (n <= target_rows_) {
            rows_accum += n;
            gpu_tables.emplace_back(std::move(leftover_tbl));
            leftover_tbl = nullptr;
        } else {
            cudf::table_view tv = leftover_tbl->view();
            std::vector<cudf::table_view> sliced =
                cudf::slice(tv, {0, (int)target_rows_, (int)n},
                            this->chunked_reader_se->stream);
            gpu_tables.push_back(std::make_unique<cudf::table>(sliced[0]));
            leftover_tbl = std::make_unique<cudf::table>(sliced[1]);
            rows_accum = target_rows_;
        }
    }

    while (rows_accum < target_rows_ && curr_piece_idx < pieces_.size()) {
        if (!curr_reader) {
            init_next_reader(this->chunked_reader_se->stream);
        }

        while (curr_reader && curr_reader->has_next() &&
               rows_accum < target_rows_) {
            cudf::io::table_with_metadata table_and_metadata =
                curr_reader->read_chunk();
            std::unique_ptr<cudf::table> tbl =
                std::move(table_and_metadata.tbl);

            if (!tbl || tbl->num_rows() == 0) {
                continue;
            }

            if (rows_to_skip > 0) {
                // Some pieces may have a row offset
                // discard leading rows until the skip count is
                // exhausted.
                size_t n = tbl->num_rows();
                size_t to_skip = std::min((size_t)rows_to_skip, n);
                if (to_skip == n) {
                    rows_to_skip -= (int64_t)n;
                    continue;
                }
                std::vector<cudf::size_type> splits = {(cudf::size_type)to_skip,
                                                       (cudf::size_type)n};
                std::vector<cudf::table_view> sliced = cudf::slice(
                    tbl->view(), splits, this->chunked_reader_se->stream);
                tbl = std::make_unique<cudf::table>(sliced[1]);
                rows_to_skip -= (int64_t)to_skip;
            }

            std::unique_ptr<cudf::table> evolved_tbl =
                evolve_table(std::move(tbl), this->chunked_reader_se->stream);

            std::size_t tbl_n = evolved_tbl->num_rows();
            if (rows_accum + tbl_n <= target_rows_) {
                rows_accum += tbl_n;
                gpu_tables.push_back(std::move(evolved_tbl));
            } else {
                std::size_t need = target_rows_ - rows_accum;
                cudf::table_view tv = evolved_tbl->view();
                std::vector<cudf::table_view> sliced =
                    cudf::slice(tv, {0, (int)need, (int)tbl_n},
                                this->chunked_reader_se->stream);
                gpu_tables.push_back(std::make_unique<cudf::table>(sliced[0]));
                leftover_tbl = std::make_unique<cudf::table>(sliced[1]);
                rows_accum = target_rows_;
            }
        }

        if (curr_reader && !curr_reader->has_next()) {
            curr_reader.reset();
        }
    }
    // Sync batch stream with chunked reader stream.
    chunked_reader_se->event.record(chunked_reader_se->stream);
    se->event.wait(chunked_reader_se->stream);

    if (gpu_tables.empty()) {
        return {empty_table_from_arrow_schema(output_arrow_schema), true};
    }

    std::vector<cudf::table_view> views;
    views.reserve(gpu_tables.size());
    for (std::unique_ptr<cudf::table>& tptr : gpu_tables) {
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

void GPUIcebergRankBatchGenerator::ReportMetrics(
    std::vector<MetricBase>& metrics_out) {
    MetricBase::StatValue pieces = pieces_.size();
    metrics_out.emplace_back(StatMetric("num_pieces", pieces));
    metrics_out.emplace_back(TimerMetric("evolve_time", this->evolve_time_));
    metrics_out.emplace_back(
        TimerMetric("fingerprint_time", this->fingerprint_time_));
}

void GPUIcebergRankBatchGenerator::init_next_reader(
    rmm::cuda_stream_view stream) {
    if (curr_piece_idx >= pieces_.size()) {
        return;
    }

    int64_t schema_group_idx = pieces_[curr_piece_idx].schema_group_idx;

    if (schema_group_idx < 0 ||
        (size_t)schema_group_idx >= scanner_read_schemas.size()) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::init_next_reader: schema group "
            "index out of bounds");
    }
    curr_read_schema = scanner_read_schemas[schema_group_idx];

    std::set<int> columns_to_read_set(selected_columns_.begin(),
                                      selected_columns_.end());
    for (const std::pair<const duckdb::idx_t,
                         duckdb::unique_ptr<duckdb::TableFilter>>& pair :
         filter_exprs_->filters) {
        columns_to_read_set.insert(pair.first);
    }

    std::vector<std::string> file_col_names;
    curr_col_mapping.clear();
    std::map<int, int> col_to_file_idx;

    for (int col_idx : columns_to_read_set) {
        std::string name = curr_read_schema->field(col_idx)->name();
        // _BODO_TEMP_ columns are placeholders for fields absent from
        // this particular file.
        if (!name.starts_with("_BODO_TEMP_")) {
            col_to_file_idx[col_idx] = (int)file_col_names.size();
            file_col_names.push_back(name);
        }
    }

    for (size_t i = 0; i < selected_columns_.size(); i++) {
        int col_idx = selected_columns_[i];
        if (col_to_file_idx.count(col_idx)) {
            curr_col_mapping.emplace_back(col_to_file_idx[col_idx], i);
        } else {
            curr_col_mapping.emplace_back(-1, i);
        }
    }

    if (last_filter_schema_group_idx != schema_group_idx) {
        // Rebuild the filter AST only when the schema group changes
        // (which determines column name -> index mappings).
        filter_ast_tree = cudf::ast::tree();
        filter_scalars.clear();

        bool filter_col_missing = false;
        for (const std::pair<const duckdb::idx_t,
                             duckdb::unique_ptr<duckdb::TableFilter>>& pair :
             filter_exprs_->filters) {
            if (!col_to_file_idx.count(pair.first)) {
                filter_col_missing = true;
                break;
            }
        }

        if (filter_col_missing) {
            std::unique_ptr<cudf::numeric_scalar<bool>> false_scalar =
                std::make_unique<cudf::numeric_scalar<bool>>(false, true);
            filter_scalars.push_back(std::move(false_scalar));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<bool>*>(
                    filter_scalars.back().get())));
            filter_ast_tree.push(cudf::ast::operation(
                cudf::ast::ast_operator::IDENTITY, filter_ast_tree.back()));
        } else {
            duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs_copy =
                filter_exprs_->Copy();
            tableFilterSetToCudfAST(*filter_exprs_copy,
                                    curr_read_schema->field_names(),
                                    filter_ast_tree, filter_scalars);
            const cudf::ast::expression* duckdb_filter_root =
                filter_ast_tree.size() > 0 ? &filter_ast_tree.back() : nullptr;

            std::map<int, int> field_id_to_col_idx;
            PyObject* field_ids_tuple =
                this->scanner_field_ids[schema_group_idx].get();
            for (int i = 0; i < (int)PyTuple_Size(field_ids_tuple); i++) {
                PyObject* item = PyTuple_GetItem(field_ids_tuple, i);
                int field_id = -1;
                if (PyLong_Check(item)) {
                    field_id = (int)PyLong_AsLong(item);
                } else if (PyTuple_Check(item)) {
                    field_id = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
                }
                if (field_id != -1) {
                    field_id_to_col_idx[field_id] = i;
                }
            }

            build_pyiceberg_cudf_ast_node(
                this->iceberg_filter_cudf_ast.get(), field_id_to_col_idx,
                curr_read_schema, filter_ast_tree, filter_scalars, stream);
            const cudf::ast::expression* iceberg_filter_root =
                &filter_ast_tree.back();

            if (duckdb_filter_root) {
                filter_ast_tree.push(cudf::ast::operation(
                    cudf::ast::ast_operator::LOGICAL_AND, *duckdb_filter_root,
                    *iceberg_filter_root));
            } else {
                // No duckdb filter, just use iceberg filter
            }
        }
        last_filter_schema_group_idx = schema_group_idx;
    }

    cudf::io::parquet_reader_options opts =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info());
    opts.set_column_names(file_col_names);
    opts.enable_allow_mismatched_pq_schemas(true);

    if (filter_ast_tree.size() > 0) {
        opts.set_filter(filter_ast_tree.back());
    }

    std::vector<std::unique_ptr<cudf::io::datasource>> sources;

    // Cap the number of files given to a single chunked reader to
    // avoid OOM when many files share the same schema group and
    // physical schema.
    size_t max_parts_for_reader =
        bytes_per_part_estimate_ > 0
            ? std::max(size_t{1}, (size_t)std::ceil(
                                      (double)CHUNKED_READER_TOTAL_BYTES_LIMIT /
                                      bytes_per_part_estimate_))
            : pieces_.size() - curr_piece_idx;

    const PiecePhysicalSchema& curr_schema =
        piece_physical_schemas_[curr_piece_idx];
    size_t parts_in_reader = 0;
    while (curr_piece_idx < pieces_.size() &&
           parts_in_reader < max_parts_for_reader &&
           pieces_[curr_piece_idx].schema_group_idx == schema_group_idx &&
           piece_physical_schemas_[curr_piece_idx] == curr_schema) {
        std::string path = pieces_[curr_piece_idx].path;
        if (filesystem_->type_name() != "local") {
            std::shared_ptr<arrow::io::RandomAccessFile> arrow_file =
                filesystem_->OpenInputFile(path).ValueOrDie();
            sources.push_back(
                std::make_unique<arrow_file_datasource>(arrow_file));
        } else {
            sources.push_back(cudf::io::datasource::create(path));
        }
        curr_piece_idx++;
        parts_in_reader++;
    }

    curr_reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        chunked_reader_limit_, std::move(sources),
        std::vector<cudf::io::parquet::FileMetaData>(), opts, stream);
}

std::unique_ptr<cudf::column>
GPUIcebergRankBatchGenerator::make_column_from_contents(
    cudf::column::contents&& col_data, cudf::size_type num_row,
    cudf::size_type null_count, cudf::data_type type,
    std::vector<std::unique_ptr<cudf::column>>&& children,
    rmm::cuda_stream_view stream) {
    rmm::device_buffer null_mask = col_data.null_mask
                                       ? std::move(*col_data.null_mask)
                                       : rmm::device_buffer{0, stream};
    return std::make_unique<cudf::column>(
        type, num_row, std::move(*col_data.data), std::move(null_mask),
        null_count, std::move(children));
}

std::unique_ptr<cudf::column> GPUIcebergRankBatchGenerator::evolve_column(
    std::unique_ptr<cudf::column> col,
    const std::shared_ptr<arrow::Field>& source_field,
    const std::shared_ptr<arrow::Field>& target_field,
    rmm::cuda_stream_view stream) {
    int source_field_iceberg_field_id = get_iceberg_field_id(source_field);
    int target_field_iceberg_field_id = get_iceberg_field_id(target_field);
    if (source_field_iceberg_field_id != target_field_iceberg_field_id) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::evolve_column: Iceberg field ID "
            "of the source (" +
            std::to_string(source_field_iceberg_field_id) + ") and target (" +
            std::to_string(target_field_iceberg_field_id) +
            ") fields do not match!");
    }

    if (arrow::is_list(source_field->type()->id())) {
        const std::shared_ptr<arrow::Field>& source_value_field =
            std::dynamic_pointer_cast<arrow::BaseListType>(source_field->type())
                ->value_field();
        const std::shared_ptr<arrow::Field>& target_value_field =
            std::dynamic_pointer_cast<arrow::BaseListType>(target_field->type())
                ->value_field();

        cudf::size_type num_row = col->size();
        cudf::size_type null_count = col->null_count();
        cudf::data_type type = col->type();
        cudf::column::contents col_data = col->release();

        std::unique_ptr<cudf::column> evolved_values_column =
            evolve_column(std::move(col_data.children[0]), source_value_field,
                          target_value_field, stream);

        std::vector<std::unique_ptr<cudf::column>> children;
        children.push_back(std::move(evolved_values_column));

        return make_column_from_contents(std::move(col_data), num_row,
                                         null_count, type, std::move(children),
                                         stream);

    } else if (source_field->type()->id() == arrow::Type::MAP) {
        const std::shared_ptr<arrow::Field>& source_key_field =
            std::dynamic_pointer_cast<arrow::MapType>(source_field->type())
                ->key_field();
        const std::shared_ptr<arrow::Field>& target_key_field =
            std::dynamic_pointer_cast<arrow::MapType>(target_field->type())
                ->key_field();
        const std::shared_ptr<arrow::Field>& source_item_field =
            std::dynamic_pointer_cast<arrow::MapType>(source_field->type())
                ->item_field();
        const std::shared_ptr<arrow::Field>& target_item_field =
            std::dynamic_pointer_cast<arrow::MapType>(target_field->type())
                ->item_field();

        cudf::size_type num_row = col->size();
        cudf::size_type null_count = col->null_count();
        cudf::data_type type = col->type();
        cudf::column::contents col_data = col->release();

        std::unique_ptr<cudf::column> struct_col =
            std::move(col_data.children[0]);
        cudf::size_type struct_num_row = struct_col->size();
        cudf::size_type struct_null_count = struct_col->null_count();
        cudf::column::contents struct_data = struct_col->release();

        std::unique_ptr<cudf::column> evolved_keys_column =
            evolve_column(std::move(struct_data.children[0]), source_key_field,
                          target_key_field, stream);
        std::unique_ptr<cudf::column> evolved_items_column =
            evolve_column(std::move(struct_data.children[1]), source_item_field,
                          target_item_field, stream);

        std::vector<std::unique_ptr<cudf::column>> evolved_struct_children;
        evolved_struct_children.push_back(std::move(evolved_keys_column));
        evolved_struct_children.push_back(std::move(evolved_items_column));

        rmm::device_buffer struct_null_mask =
            struct_data.null_mask ? std::move(*struct_data.null_mask)
                                  : rmm::device_buffer{0, stream};

        std::unique_ptr<cudf::column> evolved_struct_col =
            cudf::make_structs_column(
                struct_num_row, std::move(evolved_struct_children),
                struct_null_count, std::move(struct_null_mask), stream);

        std::vector<std::unique_ptr<cudf::column>> children;
        children.push_back(std::move(evolved_struct_col));

        return make_column_from_contents(std::move(col_data), num_row,
                                         null_count, type, std::move(children),
                                         stream);

    } else if (target_field->type()->id() == arrow::Type::STRUCT) {
        std::shared_ptr<arrow::StructType> source_type =
            std::dynamic_pointer_cast<arrow::StructType>(source_field->type());
        std::shared_ptr<arrow::StructType> target_type =
            std::dynamic_pointer_cast<arrow::StructType>(target_field->type());

        cudf::size_type num_row = col->size();
        cudf::size_type null_count = col->null_count();
        cudf::column::contents col_data = col->release();

        std::vector<std::unique_ptr<cudf::column>> source_children =
            std::move(col_data.children);
        std::vector<std::unique_ptr<cudf::column>> evolved_children;

        std::unordered_map<int, int> source_id_to_idx;
        for (int i = 0; i < source_type->num_fields(); i++) {
            source_id_to_idx[get_iceberg_field_id(source_type->field(i))] = i;
        }

        for (int i = 0; i < target_type->num_fields(); i++) {
            std::shared_ptr<arrow::Field> target_child_field =
                target_type->field(i);
            int id = get_iceberg_field_id(target_child_field);
            if (source_id_to_idx.contains(id)) {
                int source_idx = source_id_to_idx[id];
                evolved_children.push_back(
                    evolve_column(std::move(source_children[source_idx]),
                                  source_type->field(source_idx),
                                  target_child_field, stream));
            } else {
                std::unique_ptr<cudf::scalar> null_scalar =
                    arrow_scalar_to_cudf(
                        arrow::MakeNullScalar(target_child_field->type()),
                        stream);
                evolved_children.push_back(cudf::make_column_from_scalar(
                    *null_scalar, num_row, stream));
            }
        }

        rmm::device_buffer null_mask = col_data.null_mask
                                           ? std::move(*col_data.null_mask)
                                           : rmm::device_buffer{0, stream};

        return cudf::make_structs_column(num_row, std::move(evolved_children),
                                         null_count, std::move(null_mask),
                                         stream);
    }

    cudf::data_type target_cudf_type = arrow_to_cudf_type(target_field->type());
    if (col->type() != target_cudf_type) {
        return cudf::cast(col->view(), target_cudf_type, stream);
    }
    return col;
}

std::unique_ptr<cudf::table> GPUIcebergRankBatchGenerator::evolve_table(
    std::unique_ptr<cudf::table> tbl, rmm::cuda_stream_view stream) {
    time_pt start_evolve = start_timer();
    std::vector<std::unique_ptr<cudf::column>> output_cols;
    output_cols.reserve(selected_columns_.size());

    size_t num_rows = tbl->num_rows();
    std::vector<std::unique_ptr<cudf::column>> released_cols = tbl->release();

    for (size_t i = 0; i < selected_columns_.size(); i++) {
        int col_idx = selected_columns_[i];
        const std::pair<int, int>& mapping = curr_col_mapping[i];
        std::shared_ptr<arrow::Field> target_field =
            output_arrow_schema->field(i);

        if (mapping.first == -1) {
            // Column doesn't exist in this file — fill with nulls.
            std::shared_ptr<arrow::DataType> arrow_type = target_field->type();
            std::unique_ptr<cudf::scalar> null_scalar =
                arrow_scalar_to_cudf(arrow::MakeNullScalar(arrow_type), stream);
            output_cols.push_back(
                cudf::make_column_from_scalar(*null_scalar, num_rows, stream));
        } else {
            std::shared_ptr<arrow::Field> source_field =
                curr_read_schema->field(col_idx);
            output_cols.push_back(
                evolve_column(std::move(released_cols[mapping.first]),
                              source_field, target_field, stream));
        }
    }

    std::unique_ptr<cudf::table> ret =
        std::make_unique<cudf::table>(std::move(output_cols));
    this->evolve_time_ += end_timer(start_evolve);
    return ret;
}

void GPUIcebergRankBatchGenerator::push_cudf_identity(
    bool value, cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars) {
    std::unique_ptr<cudf::numeric_scalar<bool>> literal_value =
        std::make_unique<cudf::numeric_scalar<bool>>(value, true);
    filter_scalars.push_back(std::move(literal_value));
    filter_ast_tree.push(
        cudf::ast::literal(*static_cast<cudf::numeric_scalar<bool>*>(
            filter_scalars.back().get())));
    filter_ast_tree.push(cudf::ast::operation(cudf::ast::ast_operator::IDENTITY,
                                              filter_ast_tree.back()));
}

void GPUIcebergRankBatchGenerator::push_literal_to_cudf_ast(
    PyObject* value_py, const std::shared_ptr<arrow::DataType>& type,
    cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars) {
    bool is_null = (value_py == Py_None);
    switch (type->id()) {
        case arrow::Type::BOOL: {
            std::unique_ptr<cudf::numeric_scalar<bool>> literal_value =
                std::make_unique<cudf::numeric_scalar<bool>>(
                    is_null ? false : (value_py == Py_True), !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<bool>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::INT8: {
            std::unique_ptr<cudf::numeric_scalar<int8_t>> literal_value =
                std::make_unique<cudf::numeric_scalar<int8_t>>(
                    is_null ? 0 : (int8_t)PyLong_AsLong(value_py), !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int8_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::INT16: {
            std::unique_ptr<cudf::numeric_scalar<int16_t>> literal_value =
                std::make_unique<cudf::numeric_scalar<int16_t>>(
                    is_null ? 0 : (int16_t)PyLong_AsLong(value_py), !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int16_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::INT32: {
            std::unique_ptr<cudf::numeric_scalar<int32_t>> literal_value =
                std::make_unique<cudf::numeric_scalar<int32_t>>(
                    is_null ? 0 : (int32_t)PyLong_AsLong(value_py), !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int32_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::INT64: {
            std::unique_ptr<cudf::numeric_scalar<int64_t>> literal_value =
                std::make_unique<cudf::numeric_scalar<int64_t>>(
                    is_null ? 0 : (int64_t)PyLong_AsLongLong(value_py),
                    !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int64_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::FLOAT: {
            std::unique_ptr<cudf::numeric_scalar<float>> literal_value =
                std::make_unique<cudf::numeric_scalar<float>>(
                    is_null ? 0.0f : (float)PyFloat_AsDouble(value_py),
                    !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<float>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::DOUBLE: {
            std::unique_ptr<cudf::numeric_scalar<double>> literal_value =
                std::make_unique<cudf::numeric_scalar<double>>(
                    is_null ? 0.0 : PyFloat_AsDouble(value_py), !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<double>*>(
                    filter_scalars.back().get())));
            return;
        }
        case arrow::Type::STRING: {
            const char* str = is_null ? "" : PyUnicode_AsUTF8(value_py);
            std::unique_ptr<cudf::string_scalar> literal_value =
                std::make_unique<cudf::string_scalar>(std::string(str),
                                                      !is_null);
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::string_scalar*>(
                    filter_scalars.back().get())));
            return;
        }
        default:
            throw std::runtime_error(
                "push_literal_to_cudf_ast: unsupported type: " +
                type->ToString());
    }
}

void GPUIcebergRankBatchGenerator::build_pyiceberg_cudf_ast_node(
    PyObject* node, const std::map<int, int>& field_id_to_col_idx,
    const std::shared_ptr<arrow::Schema>& read_schema,
    cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars,
    rmm::cuda_stream_view stream) {
    if (!PyTuple_Check(node)) {
        push_cudf_identity(true, filter_ast_tree, filter_scalars);
        return;
    }

    PyObject* op_py = PyTuple_GetItem(node, 0);
    if (!op_py || !PyUnicode_Check(op_py)) {
        push_cudf_identity(true, filter_ast_tree, filter_scalars);
        return;
    }
    const char* op = PyUnicode_AsUTF8(op_py);

    if (strcmp(op, "true") == 0) {
        push_cudf_identity(true, filter_ast_tree, filter_scalars);
        return;
    }

    if (strcmp(op, "false") == 0) {
        push_cudf_identity(false, filter_ast_tree, filter_scalars);
        return;
    }

    auto get_col_info =
        [&](PyObject* node_inner) -> std::pair<int, std::string> {
        PyObject* field_id_py = PyTuple_GetItem(node_inner, 1);
        int field_id = (int)PyLong_AsLong(field_id_py);
        std::map<int, int>::const_iterator it =
            field_id_to_col_idx.find(field_id);
        if (it == field_id_to_col_idx.end()) {
            throw std::runtime_error(
                "build_pyiceberg_cudf_ast_node: field ID " +
                std::to_string(field_id) + " not found in schema group");
        }
        int col_idx = it->second;
        return {col_idx, read_schema->field(col_idx)->name()};
    };

    if (strcmp(op, "is_null") == 0 || strcmp(op, "is_not_null") == 0) {
        std::pair<int, std::string> col_info = get_col_info(node);
        cudf::ast::column_name_reference col_ref(col_info.second);
        filter_ast_tree.push(col_ref);
        cudf::ast::operation expr = cudf::ast::operation(
            cudf::ast::ast_operator::IS_NULL, filter_ast_tree.back());
        filter_ast_tree.push(expr);
        if (strcmp(op, "is_not_null") == 0) {
            filter_ast_tree.push(cudf::ast::operation(
                cudf::ast::ast_operator::NOT, filter_ast_tree.back()));
        }
        return;
    }

    static const std::unordered_map<std::string, cudf::ast::ast_operator>
        cmp_ops = {
            {"eq", cudf::ast::ast_operator::EQUAL},
            {"neq", cudf::ast::ast_operator::NOT_EQUAL},
            {"gt", cudf::ast::ast_operator::GREATER},
            {"gte", cudf::ast::ast_operator::GREATER_EQUAL},
            {"lt", cudf::ast::ast_operator::LESS},
            {"lte", cudf::ast::ast_operator::LESS_EQUAL},
        };
    std::unordered_map<std::string, cudf::ast::ast_operator>::const_iterator
        cmp_it = cmp_ops.find(op);
    if (cmp_it != cmp_ops.end()) {
        std::pair<int, std::string> col_info = get_col_info(node);
        int col_idx = col_info.first;
        std::string col_name = col_info.second;
        PyObject* value_py = PyTuple_GetItem(node, 2);
        std::shared_ptr<arrow::DataType> col_type =
            read_schema->field(col_idx)->type();

        cudf::ast::column_name_reference col_ref(col_name);
        filter_ast_tree.push(col_ref);
        push_literal_to_cudf_ast(value_py, col_type, filter_ast_tree,
                                 filter_scalars);

        cudf::ast::operation expr = cudf::ast::operation(
            cmp_it->second, filter_ast_tree[filter_ast_tree.size() - 2],
            filter_ast_tree.back());
        filter_ast_tree.push(expr);
        return;
    }

    if (strcmp(op, "in") == 0 || strcmp(op, "not_in") == 0) {
        std::pair<int, std::string> col_info = get_col_info(node);
        int col_idx = col_info.first;
        std::string col_name = col_info.second;
        std::shared_ptr<arrow::DataType> col_type =
            read_schema->field(col_idx)->type();
        PyObject* values_list = PyTuple_GetItem(node, 2);
        Py_ssize_t n_values = PyList_Size(values_list);

        if (n_values == 0) {
            push_cudf_identity(false, filter_ast_tree, filter_scalars);
            return;
        }

        PyObject* first_val = PyList_GetItem(values_list, 0);
        cudf::ast::column_name_reference col_ref(col_name);
        filter_ast_tree.push(col_ref);
        push_literal_to_cudf_ast(first_val, col_type, filter_ast_tree,
                                 filter_scalars);

        const cudf::ast::expression* acc = &filter_ast_tree.push(
            cudf::ast::operation(cudf::ast::ast_operator::EQUAL,
                                 filter_ast_tree[filter_ast_tree.size() - 3],
                                 filter_ast_tree.back()));

        for (Py_ssize_t i = 1; i < n_values; i++) {
            PyObject* val = PyList_GetItem(values_list, i);
            cudf::ast::column_name_reference col_ref_i(col_name);
            filter_ast_tree.push(col_ref_i);
            push_literal_to_cudf_ast(val, col_type, filter_ast_tree,
                                     filter_scalars);

            const cudf::ast::expression* eq_node =
                &filter_ast_tree.push(cudf::ast::operation(
                    cudf::ast::ast_operator::EQUAL,
                    filter_ast_tree[filter_ast_tree.size() - 3],
                    filter_ast_tree.back()));

            acc = &filter_ast_tree.push(cudf::ast::operation(
                cudf::ast::ast_operator::LOGICAL_OR, *acc, *eq_node));
        }

        if (strcmp(op, "not_in") == 0) {
            filter_ast_tree.push(cudf::ast::operation(
                cudf::ast::ast_operator::NOT, filter_ast_tree.back()));
        }
        return;
    }

    if (strcmp(op, "not") == 0) {
        PyObject* child = PyTuple_GetItem(node, 1);
        build_pyiceberg_cudf_ast_node(child, field_id_to_col_idx, read_schema,
                                      filter_ast_tree, filter_scalars, stream);
        filter_ast_tree.push(cudf::ast::operation(cudf::ast::ast_operator::NOT,
                                                  filter_ast_tree.back()));
        return;
    }

    if (strcmp(op, "and") == 0 || strcmp(op, "or") == 0) {
        PyObject* left = PyTuple_GetItem(node, 1);
        PyObject* right = PyTuple_GetItem(node, 2);

        build_pyiceberg_cudf_ast_node(left, field_id_to_col_idx, read_schema,
                                      filter_ast_tree, filter_scalars, stream);
        const cudf::ast::expression* left_root = &filter_ast_tree.back();
        build_pyiceberg_cudf_ast_node(right, field_id_to_col_idx, read_schema,
                                      filter_ast_tree, filter_scalars, stream);
        const cudf::ast::expression* right_root = &filter_ast_tree.back();

        cudf::ast::ast_operator comb_op =
            (strcmp(op, "and") == 0) ? cudf::ast::ast_operator::LOGICAL_AND
                                     : cudf::ast::ast_operator::LOGICAL_OR;

        filter_ast_tree.push(
            cudf::ast::operation(comb_op, *left_root, *right_root));
        return;
    }

    push_cudf_identity(true, filter_ast_tree, filter_scalars);
}

void GPUIcebergRankBatchGenerator::get_dataset(
    PyObject* catalog, const std::string& table_id,
    const std::shared_ptr<arrow::Schema>& arrow_schema,
    PyObject* iceberg_filter, PyObject* iceberg_schema, int64_t snapshot_id,
    int64_t tot_rows_to_read, MPI_Comm comm) {
    PyObjectPtr iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
    if (PyErr_Occurred()) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::get_dataset: failed to import "
            "bodo.io.iceberg module");
    }

    PyObjectPtr str_as_dict_cols_py = PyList_New(0);

    PyObject* force_row_level_py = Py_False;

    this->pyarrow_schema = arrow::py::wrap_schema(arrow_schema);

    PyObjectPtr duckdb_iceberg_filter =
        duckdbFilterSetToPyicebergFilter(*filter_exprs_, arrow_schema);

    PyObjectPtr py_iceberg_filter_and_duckdb_filter =
        PyObjectPtr(PyObject_CallMethod(duckdb_iceberg_filter.get(), "__and__",
                                        "O", iceberg_filter));
    if (!py_iceberg_filter_and_duckdb_filter) {
        throw std::runtime_error(
            "failed to combine iceberg filter with duckdb table filters");
    }

    PyObjectPtr iceberg_common_mod =
        PyImport_ImportModule("bodo.io.iceberg.common");
    if (PyErr_Occurred()) {
        throw std::runtime_error(
            "failed to import bodo.io.iceberg.common module");
    }

    PyObjectPtr cudf_ast_func = PyObject_GetAttrString(
        iceberg_common_mod.get(), "pyiceberg_filter_to_cudf_ast");
    if (!cudf_ast_func || !PyCallable_Check(cudf_ast_func)) {
        throw std::runtime_error(
            "failed to get pyiceberg_filter_to_cudf_ast function");
    }
    this->iceberg_filter_cudf_ast = PyObjectPtr(PyObject_CallFunctionObjArgs(
        cudf_ast_func.get(), iceberg_filter, iceberg_schema, Py_True, nullptr));
    if (!this->iceberg_filter_cudf_ast) {
        throw std::runtime_error(
            "failed to convert iceberg filter to cuDF AST");
    }

    PyObjectPtr convert_func = PyObject_GetAttrString(
        iceberg_common_mod.get(),
        "pyiceberg_filter_to_pyarrow_format_str_and_scalars");
    if (!convert_func || !PyCallable_Check(convert_func)) {
        throw std::runtime_error(
            "failed to get convert_iceberg_filter_to_arrow function from "
            "bodo.io.iceberg.common module");
    }

    PyObjectPtr filter_f_str_and_scalars =
        PyObjectPtr(PyObject_CallFunctionObjArgs(
            convert_func.get(), py_iceberg_filter_and_duckdb_filter.get(),
            iceberg_schema, Py_True, nullptr));
    if (!filter_f_str_and_scalars) {
        throw std::runtime_error(
            "failed to convert pyiceberg filter to arrow filter format "
            "string");
    }

    PyObject* iceberg_filter_f_str_py =
        PyTuple_GetItem(filter_f_str_and_scalars.get(), 0);
    PyObject* filter_scalars_py =
        PyTuple_GetItem(filter_f_str_and_scalars.get(), 1);

    PyObjectPtr ds = PyObject_CallMethod(
        iceberg_mod.get(), "get_iceberg_pq_dataset", "OsOOOsOOLL", catalog,
        table_id.c_str(), this->pyarrow_schema.get(), str_as_dict_cols_py.get(),
        py_iceberg_filter_and_duckdb_filter.get(),
        iceberg_filter_f_str_py != Py_None
            ? PyUnicode_AsUTF8(iceberg_filter_f_str_py)
            : "",
        filter_scalars_py, force_row_level_py,
        static_cast<long long>(snapshot_id),
        static_cast<long long>(tot_rows_to_read));

    if (ds == nullptr || PyErr_Occurred()) {
        throw std::runtime_error(
            "Error during Iceberg read: Failed to get dataset from Python");
    }

    this->py_filesystem = PyObject_GetAttrString(ds.get(), "filesystem");
    ensure_pa_wrappers_imported();
    if (!(this->py_filesystem == nullptr || this->py_filesystem == Py_None)) {
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_filesystem(this->py_filesystem.get()),
            "Error during Iceberg read: Failed to unwrap Arrow filesystem",
            this->filesystem_);
    }

    this->py_pieces = PyObject_GetAttrString(ds.get(), "pieces");
    this->py_schema_groups = PyObject_GetAttrString(ds.get(), "schema_groups");
}

void GPUIcebergRankBatchGenerator::distribute_pieces() {
    PyObjectPtr iceberg_mod =
        PyImport_ImportModule("bodo.io.iceberg.read_parquet");
    if (PyErr_Occurred()) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::distribute_pieces: failed to "
            "import bodo.io.iceberg.read_parquet module");
    }

    PyObjectPtr rank_py = PyLong_FromLong(rank_);
    PyObjectPtr size_py = PyLong_FromLong(size_);

    PyObjectPtr pieces_myrank_py = PyObject_CallMethod(
        iceberg_mod.get(), "distribute_pieces", "OOO", this->py_pieces.get(),
        rank_py.get(), size_py.get());
    if (pieces_myrank_py == nullptr && PyErr_Occurred()) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::distribute_pieces: "
            "distribute_pieces call failed");
    }

    PyObject* piece;
    PyObjectPtr iterator = PyObject_GetIter(pieces_myrank_py.get());
    if (iterator == nullptr) {
        throw std::runtime_error(
            "GPUIcebergRankBatchGenerator::distribute_pieces(): error "
            "getting pieces iterator");
    }

    while ((piece = PyIter_Next(iterator))) {
        PyObjectPtr p = PyObject_GetAttrString(piece, "path");
        const char* c_path = PyUnicode_AsUTF8(p.get());

        PyObjectPtr num_rows_piece_py =
            PyObject_GetAttrString(piece, "_bodo_num_rows");
        int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py.get());

        PyObjectPtr schema_group_idx_py =
            PyObject_GetAttrString(piece, "schema_group_idx");
        int64_t schema_group_idx = PyLong_AsLongLong(schema_group_idx_py.get());

        this->pieces_.push_back({c_path, num_rows_piece, schema_group_idx});

        Py_DECREF(piece);
    }
}

void GPUIcebergRankBatchGenerator::init_scanners(MPI_Comm comm) {
    if (pieces_.empty()) {
        return;
    }

    size_t n_schema_groups = PyList_Size(this->py_schema_groups.get());
    this->scanner_read_schemas.reserve(n_schema_groups);
    for (size_t i = 0; i < n_schema_groups; i++) {
        PyObject* schema_group =
            PyList_GetItem(this->py_schema_groups.get(), i);
        PyObjectPtr read_schema_py =
            PyObjectPtr(PyObject_GetAttrString(schema_group, "read_schema"));
        if (!read_schema_py) {
            throw std::runtime_error(
                "GPUIcebergRankBatchGenerator::init_scanners: failed to "
                "get read_schema from schema group");
        }
        std::shared_ptr<arrow::Schema> read_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(read_schema_py.get()),
            "GPUIcebergRankBatchGenerator::init_scanners: failed to unwrap "
            "schema",
            read_schema);
        this->scanner_read_schemas.push_back(read_schema);
        this->scanner_field_ids.emplace_back(
            PyObject_GetAttrString(schema_group, "iceberg_field_ids"));
    }
}

void GPUIcebergRankBatchGenerator::compute_physical_schema_fingerprints() {
    if (pieces_.empty()) {
        return;
    }

    size_t n = pieces_.size();
    piece_physical_schemas_.resize(n);

    // Read Parquet footers in parallel to compute schema fingerprints.
    int num_threads =
        std::max(std::min(arrow::io::GetIOThreadPoolCapacity(), int(n)), 1);
    std::vector<std::thread> threads;
    std::atomic<size_t> next_idx{0};
    std::atomic<bool> has_error{false};
    std::string error_msg;

    // Build the column-read set once — same for all files.
    std::set<int> columns_to_read(selected_columns_.begin(),
                                  selected_columns_.end());
    for (const auto& pair : filter_exprs_->filters) {
        columns_to_read.insert(pair.first);
    }

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&]() {
            while (true) {
                size_t i = next_idx.fetch_add(1);
                if (i >= n || has_error.load(std::memory_order_acquire)) {
                    break;
                }

                try {
                    const auto& piece = pieces_[i];
                    std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
                    CHECK_ARROW_AND_ASSIGN(
                        filesystem_->OpenInputFile(piece.path),
                        "compute_physical_schema_fingerprints: failed to "
                        "open file",
                        arrow_file);
                    std::unique_ptr<parquet::ParquetFileReader> pf =
                        parquet::ParquetFileReader::Open(arrow_file);
                    std::shared_ptr<parquet::FileMetaData> metadata =
                        pf->metadata();
                    auto* schema = metadata->schema();
                    auto read_schema =
                        scanner_read_schemas[piece.schema_group_idx];

                    PiecePhysicalSchema fp;
                    for (int col_idx : columns_to_read) {
                        std::string name = read_schema->field(col_idx)->name();
                        if (name.starts_with("_BODO_TEMP_")) {
                            fp.properties.push_back(-1);
                            continue;
                        }
                        int pq_idx = schema->ColumnIndex(name);
                        if (pq_idx < 0) {
                            fp.properties.push_back(-2);
                            continue;
                        }
                        auto* col = schema->Column(pq_idx);
                        fp.properties.push_back(
                            static_cast<int32_t>(col->physical_type()));
                        fp.properties.push_back(
                            static_cast<int32_t>(col->converted_type()));
                        fp.properties.push_back(
                            col->logical_type()
                                ? static_cast<int32_t>(
                                      col->logical_type()->type())
                                : -1);
                        fp.properties.push_back(col->max_repetition_level());
                        fp.properties.push_back(col->max_definition_level());
                        fp.properties.push_back(col->type_length());
                    }
                    piece_physical_schemas_[i] = std::move(fp);
                } catch (const std::exception& e) {
                    if (!has_error.exchange(true, std::memory_order_acq_rel)) {
                        error_msg = e.what();
                    }
                } catch (...) {
                    if (!has_error.exchange(true, std::memory_order_acq_rel)) {
                        error_msg = "unknown error reading parquet metadata";
                    }
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (has_error.load(std::memory_order_acquire)) {
        throw std::runtime_error("compute_physical_schema_fingerprints: " +
                                 error_msg);
    }

    // Sort pieces by (schema_group_idx, fingerprint, path) so that files
    // with identical physical schemas are contiguous for batching.
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::sort(indices, [&](size_t a, size_t b) {
        if (pieces_[a].schema_group_idx != pieces_[b].schema_group_idx) {
            return pieces_[a].schema_group_idx < pieces_[b].schema_group_idx;
        }
        if (piece_physical_schemas_[a] != piece_physical_schemas_[b]) {
            return piece_physical_schemas_[a] < piece_physical_schemas_[b];
        }
        return pieces_[a].path < pieces_[b].path;
    });

    std::vector<IcebergPieceInfo> sorted_pieces(n);
    std::vector<PiecePhysicalSchema> sorted_schemas(n);
    for (size_t i = 0; i < n; i++) {
        sorted_pieces[i] = std::move(pieces_[indices[i]]);
        sorted_schemas[i] = std::move(piece_physical_schemas_[indices[i]]);
    }
    pieces_ = std::move(sorted_pieces);
    piece_physical_schemas_ = std::move(sorted_schemas);
}

PhysicalGPUReadIceberg::PhysicalGPUReadIceberg(
    PyObject* catalog, const std::string table_id, PyObject* iceberg_filter,
    PyObject* iceberg_schema, const std::shared_ptr<arrow::Schema> arrow_schema,
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
      selected_columns(selected_columns),
      join_filter_col_stats(std::move(join_filter_col_stats)) {
    Py_INCREF(this->catalog);
    Py_INCREF(this->iceberg_filter);
    Py_INCREF(this->iceberg_schema);

    this->output_schema =
        bodo::Schema::FromArrowSchema(arrow_schema)->Project(selected_columns);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(selected_columns.size());
    for (int i : selected_columns) {
        fields.push_back(arrow_schema->field(i));
    }
    this->output_arrow_schema = arrow::schema(fields, arrow_schema->metadata());

    this->comm = get_gpu_mpi_comm(get_gpu_id());
}

PhysicalGPUReadIceberg::~PhysicalGPUReadIceberg() {
    Py_XDECREF(this->catalog);
    Py_XDECREF(this->iceberg_filter);
    Py_XDECREF(this->iceberg_schema);
    if (this->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&this->comm);
    }
}

void PhysicalGPUReadIceberg::FinalizeSource() {
    std::vector<MetricBase> metrics_out;
    metrics_out.emplace_back(
        TimerMetric("produce_time", this->metrics.produce_time));
    if (this->batch_gen) {
        this->batch_gen->ReportMetrics(metrics_out);
    }

    QueryProfileCollector::Default().SubmitOperatorName(getOpId(), ToString());
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

std::pair<GPU_DATA, OperatorResult> PhysicalGPUReadIceberg::ProduceBatchGPU(
    std::shared_ptr<StreamAndEvent> se) {
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
    std::pair<std::unique_ptr<cudf::table>, bool> next_batch_tup =
        batch_gen->next(se);
    OperatorResult result = next_batch_tup.second
                                ? OperatorResult::FINISHED
                                : OperatorResult::HAVE_MORE_OUTPUT;

    if (next_batch_tup.first) {
        this->metrics.rows_read += next_batch_tup.first->num_rows();
    }

    GPU_DATA ret(std::move(next_batch_tup.first), output_arrow_schema, se);
    this->metrics.produce_time += end_timer(start_produce);
    return {std::move(ret), result};
}

const std::shared_ptr<bodo::Schema>
PhysicalGPUReadIceberg::getOutputSchemaInternal() {
    return output_schema;
}

void PhysicalGPUReadIceberg::init_batch_gen() {
    int batch_size = get_gpu_streaming_batch_size();
    this->filter_exprs = this->join_filter_col_stats.insert_filters(
        std::move(this->filter_exprs), this->selected_columns);

    log_rtjf_expressions(join_filter_col_stats, arrow_schema);

    batch_gen = std::make_shared<GPUIcebergRankBatchGenerator>(
        catalog, table_id, iceberg_filter, iceberg_schema, arrow_schema,
        snapshot_id, selected_columns, std::move(filter_exprs), batch_size,
        comm, output_arrow_schema);
}
