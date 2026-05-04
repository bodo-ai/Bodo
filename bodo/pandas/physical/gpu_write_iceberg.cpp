#include "physical/gpu_write_iceberg.h"

#include <Python.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <arrow/table.h>
#include <mpi.h>
#include <pyerrors.h>
#include <tupleobject.h>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include "../libs/_query_profile_collector.h"
#include "physical/gpu_write_parquet.h"

PhysicalGPUWriteIceberg::PhysicalGPUWriteIceberg(
    std::shared_ptr<bodo::Schema> in_bodo_schema,
    IcebergWriteFunctionData& bind_data)
    : in_schema(std::move(bind_data.in_schema)),
      table_loc(std::move(bind_data.table_loc)),
      max_pq_chunksize(bind_data.max_pq_chunksize),
      compression(std::move(bind_data.compression)),
      iceberg_schema_str(std::move(bind_data.iceberg_schema_str)),
      iceberg_schema(std::move(bind_data.iceberg_schema)),
      fs(std::move(bind_data.fs)),
      partition_tuples(nullptr),
      sort_tuples(nullptr),
      total_rows(0),
      total_bytes(0),
      is_last_state(std::make_shared<IsLastState>()),
      finished(false),
      iceberg_files_info_py(PyList_New(0)),
      iter(0) {
    Py_INCREF(bind_data.partition_tuples);
    Py_INCREF(bind_data.sort_tuples);
    partition_tuples = PyObjectPtr(bind_data.partition_tuples);
    sort_tuples = PyObjectPtr(bind_data.sort_tuples);

    parse_partition_spec(in_schema, partition_tuples);
    parse_sort_order(in_schema, sort_tuples);

    sort_cols.clear();
    for (const auto& sf : sort_fields) {
        sort_cols.push_back(sf.col_idx);
    }
    for (const auto& pf : partition_fields) {
        sort_cols.push_back(pf.col_idx);
        partition_col_idxs.push_back(pf.col_idx);
    }
    // Estimate row size based on the input schema
    row_sz_est = 0;
    for (const auto& field : in_schema->fields()) {
        if (field->type()->byte_width() > 0) {
            row_sz_est += field->type()->byte_width();
        } else {
            // For variable-length types, use an arbitrary average size
            row_sz_est += 100;
        }
    }
}
void PhysicalGPUWriteIceberg::parse_partition_spec(
    const std::shared_ptr<arrow::Schema>& schema,
    PyObject* partition_tuples_py) {
    if (!partition_tuples_py || partition_tuples_py == Py_None) {
        return;
    }

    if (!PyList_Check(partition_tuples_py)) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg: partition_tuples is not a list");
    }

    std::map<std::string, int> name_to_idx;
    for (int i = 0; i < schema->num_fields(); i++) {
        name_to_idx[schema->field(i)->name()] = i;
    }

    Py_ssize_t n = PyList_Size(partition_tuples_py);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* tup = PyList_GetItem(partition_tuples_py, i);
        if (!PyTuple_Check(tup) || PyTuple_Size(tup) < 2) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: invalid partition tuple");
        }

        const char* transform = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0));
        const char* col_name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1));

        auto it = name_to_idx.find(col_name);
        if (it == name_to_idx.end()) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: partition column '" +
                std::string(col_name) + "' not found in schema");
        }

        long arg = 0;
        if (PyTuple_Size(tup) >= 3) {
            PyObject* arg_obj = PyTuple_GetItem(tup, 2);
            if (arg_obj && arg_obj != Py_None) {
                arg = static_cast<long>(PyLong_AsLong(arg_obj));
            }
        }

        partition_fields.push_back({it->second, col_name, transform, arg});
    }
}

void PhysicalGPUWriteIceberg::parse_sort_order(
    const std::shared_ptr<arrow::Schema>& schema, PyObject* sort_tuples_py) {
    if (!sort_tuples_py || sort_tuples_py == Py_None) {
        return;
    }

    if (!PyList_Check(sort_tuples_py)) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg: sort_tuples is not a list");
    }

    std::map<std::string, int> name_to_idx;
    for (int i = 0; i < schema->num_fields(); i++) {
        name_to_idx[schema->field(i)->name()] = i;
    }

    Py_ssize_t n = PyList_Size(sort_tuples_py);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* tup = PyList_GetItem(sort_tuples_py, i);
        if (!PyTuple_Check(tup) || PyTuple_Size(tup) != 1) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: invalid sort tuple");
        }

        const char* col_name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0));

        auto it = name_to_idx.find(col_name);
        if (it == name_to_idx.end()) {
            throw std::runtime_error("PhysicalGPUWriteIceberg: sort column '" +
                                     std::string(col_name) +
                                     "' not found in schema");
        }

        sort_fields.push_back({it->second});
    }
}

cudf::io::compression_type PhysicalGPUWriteIceberg::pq_compression_from_string(
    const std::string& c) {
    if (c == "snappy") {
        return cudf::io::compression_type::SNAPPY;
    }
    if (c == "gzip") {
        return cudf::io::compression_type::GZIP;
    }
    if (c == "brotli") {
        return cudf::io::compression_type::BROTLI;
    }
    if (c == "lz4") {
        return cudf::io::compression_type::LZ4;
    }
    if (c == "zstd") {
        return cudf::io::compression_type::ZSTD;
    }
    throw std::runtime_error(
        "PhysicalGPUWriteIceberg: Unsupported compression codec '" + c + "'");
}

OperatorResult PhysicalGPUWriteIceberg::ConsumeBatchGPU(
    GPU_DATA input_batch, OperatorResult prev_op_result,
    std::shared_ptr<StreamAndEvent> se) {
    if (!is_gpu_rank()) {
        if (prev_op_result == OperatorResult::FINISHED) {
            finished = true;
        }
        return finished ? OperatorResult::FINISHED
                        : OperatorResult::NEED_MORE_INPUT;
    }

    if (prev_batch_se) {
        prev_batch_se->event.wait(se->stream);
    }
    prev_batch_se = se;

    if (this->finished) {
        return OperatorResult::FINISHED;
    }

    bool is_last = (prev_op_result == OperatorResult::FINISHED);
    is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(is_last)));

    time_pt start_accumulate = start_timer();

    std::shared_ptr<cudf::table> incoming_table = input_batch.table;
    if (incoming_table && incoming_table->num_rows() > 0) {
        accumulated_tables.push_back(incoming_table);
        total_rows += incoming_table->num_rows();
        cudf::table_view tv = incoming_table->view();
        if (tv.num_rows() > 0 && tv.num_columns() > 0) {
            total_bytes += tv.num_rows() * this->row_sz_est;
        }
    }

    metrics.accumulate_time += end_timer(start_accumulate);
    metrics.max_buffer_rows = std::max(metrics.max_buffer_rows, total_rows);

    bool should_flush = is_last || (total_bytes >= max_pq_chunksize);

    if (should_flush && total_rows > 0) {
        flush_buffer(se, is_last);
    }

    if (is_last) {
        this->finished = true;
        return OperatorResult::FINISHED;
    }

    return OperatorResult::NEED_MORE_INPUT;
}

void PhysicalGPUWriteIceberg::flush_buffer(std::shared_ptr<StreamAndEvent> se,
                                           bool is_last) {
    time_pt start_sort = start_timer();

    // Concatenate all accumulated tables into one
    std::unique_ptr<cudf::table> sorted_table;
    if (accumulated_tables.size() == 1) {
        sorted_table =
            std::make_unique<cudf::table>(accumulated_tables[0]->release());
    } else {
        std::vector<cudf::table_view> views;
        views.reserve(accumulated_tables.size());
        for (auto& t : accumulated_tables) {
            views.push_back(t->view());
        }
        sorted_table = cudf::concatenate(views, se->stream);
    }

    // Sort by sort_cols + partition_cols
    if (!sort_cols.empty()) {
        std::vector<cudf::order> column_order(sort_cols.size(),
                                              cudf::order::ASCENDING);
        std::vector<cudf::null_order> null_precedence(sort_cols.size(),
                                                      cudf::null_order::AFTER);
        cudf::table_view key_tv = sorted_table->view().select(sort_cols);
        std::unique_ptr<cudf::table> sorted =
            cudf::sort_by_key(sorted_table->view(), key_tv, column_order,
                              null_precedence, se->stream);
        sorted_table = std::move(sorted);
    }

    metrics.sort_time += end_timer(start_sort);
    time_pt start_file_write = start_timer();

    // Detect partition boundaries by transferring partition columns
    // to host and walking rows
    // TODO: If this becomes a bottleneck, we could consider doing this on the
    // GPU
    std::vector<std::pair<cudf::size_type, cudf::size_type>> partition_groups;
    std::vector<std::vector<std::shared_ptr<arrow::Scalar>>>
        partition_group_values;

    if (partition_col_idxs.empty()) {
        partition_groups.emplace_back(
            0, static_cast<cudf::size_type>(sorted_table->num_rows()));
        partition_group_values.emplace_back();
    } else {
        // Extract partition columns as owned table and convert to Arrow
        std::vector<std::unique_ptr<cudf::column>> part_cols;
        for (int idx : partition_col_idxs) {
            part_cols.push_back(std::make_unique<cudf::column>(
                sorted_table->view().column(idx), se->stream));
        }
        auto part_table = std::make_unique<cudf::table>(std::move(part_cols));

        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (int idx : partition_col_idxs) {
            fields.push_back(in_schema->field(idx));
        }
        auto part_arrow_schema = arrow::schema(fields);

        std::shared_ptr<arrow::Table> part_arrow_table = convertGPUToArrow(
            GPU_DATA(std::shared_ptr<cudf::table>(std::move(part_table)),
                     part_arrow_schema, se));

        cudf::size_type n_rows = sorted_table->num_rows();
        cudf::size_type grp_start = 0;
        int n_part_cols = (int)partition_col_idxs.size();

        for (cudf::size_type r = 1; r < n_rows; r++) {
            bool same = true;
            for (int ci = 0; ci < n_part_cols; ci++) {
                std::shared_ptr<arrow::Scalar> prev =
                    part_arrow_table->column(ci)->GetScalar(r - 1).ValueOrDie();
                std::shared_ptr<arrow::Scalar> curr =
                    part_arrow_table->column(ci)->GetScalar(r).ValueOrDie();
                if (!prev->Equals(*curr)) {
                    same = false;
                    break;
                }
            }
            if (!same) {
                partition_groups.emplace_back(grp_start, r);
                std::vector<std::shared_ptr<arrow::Scalar>> vals;
                for (int ci = 0; ci < n_part_cols; ci++) {
                    vals.push_back(part_arrow_table->column(ci)
                                       ->GetScalar(grp_start)
                                       .ValueOrDie());
                }
                partition_group_values.push_back(std::move(vals));
                grp_start = r;
            }
        }
        // Final group
        partition_groups.emplace_back(grp_start, n_rows);
        {
            std::vector<std::shared_ptr<arrow::Scalar>> vals;
            for (int ci = 0; ci < n_part_cols; ci++) {
                vals.push_back(part_arrow_table->column(ci)
                                   ->GetScalar(grp_start)
                                   .ValueOrDie());
            }
            partition_group_values.push_back(std::move(vals));
        }
    }

    metrics.n_partition_groups +=
        static_cast<MetricBase::StatValue>(partition_groups.size());

    int myrank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::string fname_prefix = get_fname_prefix(iter);

    // Determine which columns to include in output (exclude void columns)
    std::vector<cudf::size_type> output_col_indices;
    std::vector<std::string> output_col_names;
    for (int i = 0; i < in_schema->num_fields(); i++) {
        bool is_void_col = false;
        for (const auto& pf : partition_fields) {
            if (pf.col_idx == i && pf.transform == "void") {
                is_void_col = true;
                break;
            }
        }
        if (!is_void_col) {
            output_col_indices.push_back(i);
            output_col_names.push_back(in_schema->field_names()[i]);
        }
    }

    for (size_t gi = 0; gi < partition_groups.size(); gi++) {
        auto [start, end] = partition_groups[gi];
        if (end <= start) {
            continue;
        }

        // Build partition directory path
        std::string part_path =
            build_partition_path(partition_group_values[gi]);

        std::filesystem::path dir_path(table_loc);
        if (!part_path.empty()) {
            dir_path /= part_path;
        }

        std::string dir_path_str = fs->type_name() == "local"
                                       ? dir_path.string()
                                       : dir_path.generic_string();
        arrow::Status mkdir_status = fs->CreateDir(dir_path_str);
        if (!mkdir_status.ok() && !mkdir_status.IsAlreadyExists()) {
            CHECK_ARROW_GPU_ICEBERG(
                mkdir_status, "PhysicalGPUWriteIceberg: CreateDir failed");
        }

        // Build file name
        std::string fname =
            fname_prefix +
            std::string(5 - std::to_string(myrank).length(), '0') +
            std::to_string(myrank) + ".parquet";
        std::filesystem::path out_path = dir_path / fname;

        // Slice sorted table to this partition group's rows
        cudf::table_view stv = sorted_table->view();
        std::vector<cudf::table_view> sliced = cudf::slice(stv, {start, end});
        cudf::table_view group_view = sliced[0];

        // Select output columns from the sliced view
        cudf::table_view out_tv = group_view.select(output_col_indices);

        // Build metadata with iceberg.schema
        std::map<std::string, std::string> kv_meta;
        kv_meta["iceberg.schema"] = iceberg_schema_str;

        cudf::io::table_input_metadata write_meta{out_tv};
        for (int i = 0; i < (int)output_col_names.size(); i++) {
            write_meta.column_metadata[i].set_name(output_col_names[i]);
        }

        // Build writer options with key-value metadata
        BodoDataSink bodo_data_sink(fs, out_path);
        auto sink = cudf::io::sink_info(&bodo_data_sink);
        auto builder =
            cudf::io::parquet_writer_options::builder(sink, out_tv)
                .metadata(write_meta)
                .key_value_metadata(
                    std::vector<std::map<std::string, std::string>>{kv_meta});
        builder =
            builder.stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
        try {
            builder.compression(static_cast<cudf::io::compression_type>(
                pq_compression_from_string(compression)));
        } catch (...) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: "
                "pq_compression_from_string failed.");
        }

        auto options = builder.build();
        cudf::io::write_parquet(options, se->stream);

        // Record file metadata
        cudf::size_type record_count = end - start;
        size_t file_size = bodo_data_sink.bytes_written();

        std::string out_path_str = fs->type_name() == "local"
                                       ? out_path.string()
                                       : out_path.generic_string();

        auto& partition_vals = partition_group_values[gi];
        PyObjectPtr file_info_tuple =
            PyTuple_New(3 + static_cast<Py_ssize_t>(partition_vals.size()));
        PyTuple_SetItem(file_info_tuple, 0,
                        PyUnicode_FromString(out_path_str.c_str()));
        PyTuple_SetItem(file_info_tuple, 1, PyLong_FromLongLong(record_count));
        PyTuple_SetItem(file_info_tuple, 2, PyLong_FromLongLong(file_size));

        for (size_t pi = 0; pi < partition_vals.size(); pi++) {
            std::shared_ptr<arrow::Scalar> scalar = partition_vals[pi];
            PyObject* py_val = nullptr;
            if (!scalar->is_valid) {
                py_val = Py_None;
                Py_INCREF(py_val);
            } else {
                switch (scalar->type->id()) {
                    case arrow::Type::INT8:
                        py_val = PyLong_FromLong(
                            std::static_pointer_cast<arrow::Int8Scalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::INT16:
                        py_val = PyLong_FromLong(
                            std::static_pointer_cast<arrow::Int16Scalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::INT32:
                        py_val = PyLong_FromLong(
                            std::static_pointer_cast<arrow::Int32Scalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::INT64:
                        py_val = PyLong_FromLongLong(
                            std::static_pointer_cast<arrow::Int64Scalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::UINT8:
                        py_val = PyLong_FromUnsignedLong(
                            std::static_pointer_cast<arrow::UInt8Scalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::UINT16:
                        py_val = PyLong_FromUnsignedLong(
                            std::static_pointer_cast<arrow::UInt16Scalar>(
                                scalar)
                                ->value);
                        break;
                    case arrow::Type::UINT32:
                        py_val = PyLong_FromUnsignedLong(
                            std::static_pointer_cast<arrow::UInt32Scalar>(
                                scalar)
                                ->value);
                        break;
                    case arrow::Type::UINT64:
                        py_val = PyLong_FromUnsignedLongLong(
                            std::static_pointer_cast<arrow::UInt64Scalar>(
                                scalar)
                                ->value);
                        break;
                    case arrow::Type::FLOAT:
                        py_val = PyFloat_FromDouble(
                            std::static_pointer_cast<arrow::FloatScalar>(scalar)
                                ->value);
                        break;
                    case arrow::Type::DOUBLE:
                        py_val = PyFloat_FromDouble(
                            std::static_pointer_cast<arrow::DoubleScalar>(
                                scalar)
                                ->value);
                        break;
                    case arrow::Type::STRING:
                        py_val = PyUnicode_FromString(
                            std::static_pointer_cast<arrow::StringScalar>(
                                scalar)
                                ->value->ToString()
                                .c_str());
                        break;
                    case arrow::Type::BOOL:
                        py_val = std::static_pointer_cast<arrow::BooleanScalar>(
                                     scalar)
                                         ->value
                                     ? Py_True
                                     : Py_False;
                        Py_INCREF(py_val);
                        break;
                    case arrow::Type::DATE32:
                        py_val = PyLong_FromLong(
                            std::static_pointer_cast<arrow::Date32Scalar>(
                                scalar)
                                ->value);
                        break;
                    case arrow::Type::TIMESTAMP:
                        py_val = PyLong_FromLongLong(
                            std::static_pointer_cast<arrow::TimestampScalar>(
                                scalar)
                                ->value);
                        break;
                    default:
                        std::string str_val = scalar->ToString();
                        py_val = PyUnicode_FromString(str_val.c_str());
                        break;
                }
            }
            PyTuple_SetItem(file_info_tuple, 3 + static_cast<Py_ssize_t>(pi),
                            py_val);
        }

        PyList_Append(iceberg_files_info_py, file_info_tuple);

        metrics.n_files_written++;
    }

    metrics.file_write_time += end_timer(start_file_write);

    accumulated_tables.clear();
    total_rows = 0;
    total_bytes = 0;
    iter++;
}

std::string PhysicalGPUWriteIceberg::build_partition_path(
    const std::vector<std::shared_ptr<arrow::Scalar>>& partition_values) {
    if (partition_values.empty()) {
        return "";
    }

    std::string path;
    for (size_t i = 0; i < partition_values.size(); i++) {
        if (!path.empty()) {
            path += "/";
        }
        path += partition_fields[i].col_name + "=";
        std::shared_ptr<arrow::Scalar> scalar = partition_values[i];
        if (!scalar->is_valid) {
            path += "__HIVE_DEFAULT_PARTITION__";
        } else {
            path += scalar->ToString();
        }
    }
    return path;
}

void PhysicalGPUWriteIceberg::FinalizeSink() {
    time_pt start_finalize = start_timer();

    PyObjectPtr mpi4py_module = PyImport_ImportModule("mpi4py");
    if (!mpi4py_module || PyErr_Occurred()) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg::Finalize: Failed to import "
            "mpi4py");
    }
    PyObjectPtr mpi_module = PyObject_GetAttrString(mpi4py_module, "MPI");
    if (!mpi_module || PyErr_Occurred()) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg::Finalize: Failed to get MPI "
            "module");
    }

    PyObjectPtr comm_world = PyObject_GetAttrString(mpi_module, "COMM_WORLD");
    if (!comm_world || PyErr_Occurred()) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg::Finalize: Failed to get "
            "COMM_WORLD");
    }

    PyObjectPtr gather_method = PyObject_GetAttrString(comm_world, "gather");
    if (!gather_method || PyErr_Occurred()) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg::Finalize: Failed to get "
            "gather method");
    }

    PyObjectPtr args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, iceberg_files_info_py);

    PyObject* all_infos = PyObject_CallObject(gather_method, args);

    if (!all_infos || PyErr_Occurred()) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg::Finalize: Iceberg write: MPI "
            "gather operation failed");
    }

    iceberg_files_info_py = PyObjectPtr(all_infos);
    metrics.finalize_time = end_timer(start_finalize);

    std::vector<MetricBase> metrics_out;
    ReportMetrics(metrics_out);
    QueryProfileCollector::Default().RegisterOperatorStageMetrics(
        QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
        std::move(metrics_out));
    QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
        QueryProfileCollector::MakeOperatorStageID(getOpId(), 1), 0);
}

std::variant<std::shared_ptr<table_info>, PyObject*>
PhysicalGPUWriteIceberg::GetResult() {
    return iceberg_files_info_py;
}

void PhysicalGPUWriteIceberg::ReportMetrics(
    std::vector<MetricBase>& metrics_out) {
    metrics_out.emplace_back(
        StatMetric("max_buffer_rows", metrics.max_buffer_rows));
    metrics_out.emplace_back(
        StatMetric("n_files_written", metrics.n_files_written));
    metrics_out.emplace_back(
        StatMetric("n_partition_groups", metrics.n_partition_groups));
    metrics_out.emplace_back(
        TimerMetric("accumulate_time", metrics.accumulate_time));
    metrics_out.emplace_back(TimerMetric("sort_time", metrics.sort_time));
    metrics_out.emplace_back(
        TimerMetric("file_write_time", metrics.file_write_time));
    metrics_out.emplace_back(
        TimerMetric("finalize_time", metrics.finalize_time));
}
