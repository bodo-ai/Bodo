#include "physical/gpu_write_iceberg.h"

#include <Python.h>
#include <arrow/io/file.h>
#include <arrow/python/api.h>
#include <arrow/table.h>
#include <mpi.h>
#include <pyerrors.h>
#include <tupleobject.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include "../io/iceberg_helpers.h"
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

    Py_ssize_t n = PyList_Size(partition_tuples_py);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* tup = PyList_GetItem(partition_tuples_py, i);
        // Tuple format from build_partition_sort_tuples:
        //   (col_idx, transform_name, arg, partition_name)
        if (!PyTuple_Check(tup) || PyTuple_Size(tup) < 4) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: invalid partition tuple "
                "(expected 4 elements)");
        }

        int64_t col_idx = PyLong_AsLongLong(PyTuple_GET_ITEM(tup, 0));
        const char* transform = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1));
        long arg = PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        const char* partition_name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 3));

        std::string col_name = schema->field(col_idx)->name();
        partition_fields.push_back({static_cast<cudf::size_type>(col_idx),
                                    col_name, transform, arg, partition_name});
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

    Py_ssize_t n = PyList_Size(sort_tuples_py);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* tup = PyList_GetItem(sort_tuples_py, i);
        // Tuple format from build_partition_sort_tuples:
        //   (col_idx, transform_name, arg, is_asc, nulls_last)
        if (!PyTuple_Check(tup) || PyTuple_Size(tup) < 5) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: invalid sort tuple "
                "(expected 5 elements)");
        }

        int64_t col_idx = PyLong_AsLongLong(PyTuple_GET_ITEM(tup, 0));
        const char* transform = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1));
        long arg = PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        bool is_asc = PyObject_IsTrue(PyTuple_GET_ITEM(tup, 3));
        bool nulls_last = PyObject_IsTrue(PyTuple_GET_ITEM(tup, 4));

        SortField sf{.col_idx = static_cast<cudf::size_type>(col_idx),
                     .is_asc = is_asc,
                     .nulls_last = nulls_last,
                     .transform = transform,
                     .arg = arg};
        if (!sf.is_noop_transform()) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: sort transform '" + sf.transform +
                "' is not yet supported on GPU");
        }
        sort_fields.push_back(std::move(sf));
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
    // All ranks must participate in sync_is_last_non_blocking (it uses
    // MPI_Ibarrier on MPI_COMM_WORLD).  Do this before the GPU rank
    // check so non-GPU ranks don't skip the barrier and deadlock.
    if (this->finished) {
        return OperatorResult::FINISHED;
    }

    bool is_last = (prev_op_result == OperatorResult::FINISHED);
    is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(is_last)));

    if (!is_gpu_rank()) {
        if (is_last) {
            finished = true;
        }
        return finished ? OperatorResult::FINISHED
                        : OperatorResult::NEED_MORE_INPUT;
    }
    // ---- GPU rank only below ----

    if (prev_batch_se) {
        prev_batch_se->event.wait(se->stream);
    }
    prev_batch_se = se;

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
        std::vector<cudf::order> column_order;
        std::vector<cudf::null_order> null_precedence;
        column_order.reserve(sort_cols.size());
        null_precedence.reserve(sort_cols.size());

        // Per-sort-field direction and null placement
        for (const auto& sf : sort_fields) {
            column_order.push_back(sf.is_asc ? cudf::order::ASCENDING
                                             : cudf::order::DESCENDING);
            null_precedence.push_back(sf.nulls_last ? cudf::null_order::AFTER
                                                    : cudf::null_order::BEFORE);
        }
        // Partition columns: sort direction doesn't affect group
        // membership, only contiguity. Use ascending + nulls last
        // as a stable default (partition fields carry no sort
        // direction in the Iceberg spec).
        for (size_t i = 0; i < partition_fields.size(); i++) {
            column_order.push_back(cudf::order::ASCENDING);
            null_precedence.push_back(cudf::null_order::AFTER);
        }

        cudf::table_view key_tv = sorted_table->view().select(sort_cols);
        std::unique_ptr<cudf::table> sorted =
            cudf::sort_by_key(sorted_table->view(), key_tv, column_order,
                              null_precedence, se->stream);
        sorted_table = std::move(sorted);
    }

    metrics.sort_time += end_timer(start_sort);
    time_pt start_file_write = start_timer();

    // Detect partition boundaries on GPU using groupby on the already-sorted
    // partition columns.  groupby(keys_are_sorted=YES) scans in O(n) without
    // re-sorting.
    std::vector<std::pair<cudf::size_type, cudf::size_type>> partition_groups;
    std::vector<std::vector<std::shared_ptr<arrow::Scalar>>>
        partition_group_values;

    if (partition_col_idxs.empty()) {
        partition_groups.emplace_back(
            0, static_cast<cudf::size_type>(sorted_table->num_rows()));
        partition_group_values.emplace_back();
    } else {
        cudf::table_view part_tv =
            sorted_table->view().select(partition_col_idxs);

        // Keys are already sorted by sort_cols + partition_cols, so
        // partition columns are contiguous and sorted.
        cudf::groupby::groupby gb(part_tv, cudf::null_policy::INCLUDE,
                                  cudf::sorted::YES);

        auto groups = gb.get_groups(
            /*values=*/{}, se->stream);

        const auto& offsets = groups.offsets;
        if (offsets.size() < 2) {
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: groupby returned fewer "
                "than 2 offsets");
        }
        cudf::size_type n_groups =
            static_cast<cudf::size_type>(offsets.size() - 1);

        // Build partition_groups from offsets
        partition_groups.reserve(n_groups);
        for (cudf::size_type gi = 0; gi < n_groups; gi++) {
            partition_groups.emplace_back(offsets[gi], offsets[gi + 1]);
        }

        // Gather the first row of each group from the sorted table's
        // partition columns — guarantees column types match the input
        // schema and avoids relying on groupby's internal key
        // representation.
        std::vector<cudf::size_type> group_starts;
        group_starts.reserve(n_groups);
        for (cudf::size_type gi = 0; gi < n_groups; gi++) {
            group_starts.push_back(static_cast<cudf::size_type>(offsets[gi]));
        }
        auto gather_map = cudf::make_numeric_column(
            cudf::data_type{cudf::type_id::INT32},
            static_cast<cudf::size_type>(group_starts.size()),
            cudf::mask_state::UNALLOCATED, se->stream);
        auto gather_view = gather_map->mutable_view();
        CHECK_CUDA(
            cudaMemcpyAsync(gather_view.head<int32_t>(), group_starts.data(),
                            group_starts.size() * sizeof(cudf::size_type),
                            cudaMemcpyHostToDevice, se->stream.value()));

        auto gathered =
            cudf::gather(part_tv, gather_map->view(),
                         cudf::out_of_bounds_policy::DONT_CHECK, se->stream);

        // Transfer the gathered rows (one per group) to Arrow
        std::vector<std::shared_ptr<arrow::Field>> key_fields;
        for (int idx : partition_col_idxs) {
            key_fields.push_back(in_schema->field(idx));
        }
        auto key_arrow_schema = arrow::schema(key_fields);

        std::shared_ptr<arrow::Table> key_arrow_table = convertGPUToArrow(
            GPU_DATA(std::shared_ptr<cudf::table>(std::move(gathered)),
                     key_arrow_schema, se));

        int n_part_cols = static_cast<int>(partition_col_idxs.size());
        partition_group_values.reserve(n_groups);
        for (cudf::size_type gi = 0; gi < n_groups; gi++) {
            std::vector<std::shared_ptr<arrow::Scalar>> vals;
            vals.reserve(n_part_cols);
            for (int ci = 0; ci < n_part_cols; ci++) {
                vals.push_back(
                    key_arrow_table->column(ci)->GetScalar(gi).ValueOrDie());
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
        if (fs->type_name() == "local") {
            std::filesystem::create_directories(dir_path);
        } else {
            // S3 / HDFS create parent dirs automatically when writing.
            arrow::Status mkdir_status = fs->CreateDir(dir_path_str);
            if (!mkdir_status.ok() && !mkdir_status.IsAlreadyExists()) {
                CHECK_ARROW_GPU_ICEBERG(
                    mkdir_status, "PhysicalGPUWriteIceberg: CreateDir failed");
            }
        }

        // Build file name
        std::string fname = generate_iceberg_file_name();
        std::filesystem::path out_path = dir_path / fname;

        // Relative path from table_loc (for the file-info tuple).
        std::string rel_path;
        if (part_path.empty()) {
            rel_path = fname;
        } else {
            rel_path = part_path + "/" + fname;
        }

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
            auto field = iceberg_schema->GetFieldByName(output_col_names[i]);
            int64_t field_id = get_iceberg_field_id(field);
            write_meta.column_metadata[i].set_parquet_field_id(
                static_cast<int32_t>(field_id));
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

        // Compute per-column stats on GPU via cudf::minmax.
        PyObject* value_counts_dict = PyDict_New();
        PyObject* null_count_dict = PyDict_New();
        PyObject* lower_bound_dict = PyDict_New();
        PyObject* upper_bound_dict = PyDict_New();
        compute_field_metrics_gpu(out_tv, output_col_names, se,
                                  value_counts_dict, null_count_dict,
                                  lower_bound_dict, upper_bound_dict);

        auto& partition_vals = partition_group_values[gi];
        int n_part_vals = static_cast<int>(partition_vals.size());

        // Tuple: file_name, record_count, file_size,
        //        value_counts_dict, null_count_dict,
        //        lower_bound_dict,  upper_bound_dict,
        //        *partition_values
        PyObject* file_info_tuple = PyTuple_New(7 + n_part_vals);
        PyTuple_SetItem(file_info_tuple, 0,
                        PyUnicode_FromString(rel_path.c_str()));
        PyTuple_SetItem(file_info_tuple, 1, PyLong_FromLongLong(record_count));
        PyTuple_SetItem(file_info_tuple, 2, PyLong_FromLongLong(file_size));
        PyTuple_SetItem(file_info_tuple, 3, value_counts_dict);
        PyTuple_SetItem(file_info_tuple, 4, null_count_dict);
        PyTuple_SetItem(file_info_tuple, 5, lower_bound_dict);
        PyTuple_SetItem(file_info_tuple, 6, upper_bound_dict);

        for (int pi = 0; pi < n_part_vals; pi++) {
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
                    case arrow::Type::LARGE_STRING: {
                        auto s =
                            std::static_pointer_cast<arrow::LargeStringScalar>(
                                scalar);
                        py_val =
                            PyUnicode_FromString(s->value->ToString().c_str());
                        break;
                    }
                    case arrow::Type::BINARY: {
                        auto b = std::static_pointer_cast<arrow::BinaryScalar>(
                            scalar);
                        py_val = PyUnicode_FromStringAndSize(
                            reinterpret_cast<const char*>(b->value->data()),
                            static_cast<Py_ssize_t>(b->value->size()));
                        break;
                    }
                    case arrow::Type::LARGE_BINARY: {
                        auto b =
                            std::static_pointer_cast<arrow::LargeBinaryScalar>(
                                scalar);
                        py_val = PyUnicode_FromStringAndSize(
                            reinterpret_cast<const char*>(b->value->data()),
                            static_cast<Py_ssize_t>(b->value->size()));
                        break;
                    }
                    case arrow::Type::DECIMAL128: {
                        py_val =
                            PyUnicode_FromString(scalar->ToString().c_str());
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            "PhysicalGPUWriteIceberg: unsupported "
                            "partition value type: " +
                            scalar->type->ToString() +
                            " (name=" + scalar->type->name() + ")");
                }
            }
            PyTuple_SetItem(file_info_tuple, 7 + pi, py_val);
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
        path += partition_fields[i].partition_name + "=";
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

    iceberg_files_info_py = all_infos;
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

// ========================================================================
// Iceberg binary serialization and GPU-based field metrics
// ========================================================================

template <typename T>
PyObject* PhysicalGPUWriteIceberg::buffer_to_little_endian_bytes(T value) {
    const char* initial_bytes = reinterpret_cast<const char*>(&value);
    if constexpr (std::endian::native == std::endian::little) {
        return PyBytes_FromStringAndSize(initial_bytes, sizeof(T));
    } else {
        std::string str = std::string(initial_bytes, sizeof(T));
        std::ranges::reverse(str);
        const char* bytes = str.c_str();
        return PyBytes_FromStringAndSize(bytes, sizeof(T));
    }
}

PyObject* PhysicalGPUWriteIceberg::arrow_scalar_to_iceberg_bytes(
    const std::shared_ptr<arrow::Scalar>& scalar) {
    switch (scalar->type->id()) {
        case arrow::Type::BOOL: {
            auto b = std::static_pointer_cast<arrow::BooleanScalar>(scalar);
            char byte_value = b->value ? 0x1 : 0x0;
            return PyBytes_FromStringAndSize(&byte_value, 1);
        }
        case arrow::Type::INT32:
            return buffer_to_little_endian_bytes<int32_t>(
                std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value);
        case arrow::Type::INT64:
            return buffer_to_little_endian_bytes<int64_t>(
                std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
        case arrow::Type::FLOAT:
            return buffer_to_little_endian_bytes<float>(
                std::static_pointer_cast<arrow::FloatScalar>(scalar)->value);
        case arrow::Type::DOUBLE:
            return buffer_to_little_endian_bytes<double>(
                std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value);
        case arrow::Type::DATE32:
            return buffer_to_little_endian_bytes<int32_t>(
                std::static_pointer_cast<arrow::Date32Scalar>(scalar)->value);
        case arrow::Type::TIME64:
            return buffer_to_little_endian_bytes<int64_t>(
                std::static_pointer_cast<arrow::Time64Scalar>(scalar)->value);
        case arrow::Type::TIMESTAMP:
            return buffer_to_little_endian_bytes<int64_t>(
                std::static_pointer_cast<arrow::TimestampScalar>(scalar)
                    ->value);
        case arrow::Type::STRING: {
            auto s = std::static_pointer_cast<arrow::StringScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char*>(s->value->data()),
                s->value->size());
        }
        case arrow::Type::LARGE_STRING: {
            auto s = std::static_pointer_cast<arrow::LargeStringScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char*>(s->value->data()),
                s->value->size());
        }
        case arrow::Type::BINARY: {
            auto b = std::static_pointer_cast<arrow::BinaryScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char*>(b->value->data()),
                b->value->size());
        }
        case arrow::Type::LARGE_BINARY: {
            auto b = std::static_pointer_cast<arrow::LargeBinaryScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char*>(b->value->data()),
                b->value->size());
        }
        default:
            throw std::runtime_error(
                "arrow_scalar_to_iceberg_bytes: unsupported type " +
                scalar->type->ToString());
    }
}

void PhysicalGPUWriteIceberg::compute_field_metrics_gpu(
    cudf::table_view group_view,
    const std::vector<std::string>& output_col_names,
    std::shared_ptr<StreamAndEvent> se, PyObject* value_counts_dict,
    PyObject* null_count_dict, PyObject* lower_bound_dict,
    PyObject* upper_bound_dict) {
    // Collect per-column stats on GPU.
    std::vector<int64_t> value_counts;
    std::vector<int64_t> null_counts;
    std::vector<std::unique_ptr<cudf::scalar>> min_scalars;
    std::vector<std::unique_ptr<cudf::scalar>> max_scalars;
    std::vector<bool> has_bounds;
    std::vector<int64_t> field_ids;
    std::vector<cudf::type_id> col_type_ids;

    for (size_t i = 0; i < output_col_names.size(); i++) {
        cudf::column_view col = group_view.column(i);
        std::string col_name = output_col_names[i];
        auto field = iceberg_schema->GetFieldByName(col_name);
        int64_t fid = get_iceberg_field_id(field);
        field_ids.push_back(fid);

        int64_t vc = col.size();
        int64_t nc = col.null_count();
        value_counts.push_back(vc);
        null_counts.push_back(nc);

        // Skip min/max for nested types on GPU; will store
        // Py_None in the bounds dicts for these fields.
        cudf::type_id tid = col.type().id();
        col_type_ids.push_back(tid);
        bool is_nested =
            (tid == cudf::type_id::LIST || tid == cudf::type_id::STRUCT);
        bool can_bound = !is_nested && (nc < vc);
        has_bounds.push_back(can_bound);

        if (can_bound) {
            auto [min_s, max_s] = cudf::minmax(col);
            min_scalars.push_back(std::move(min_s));
            max_scalars.push_back(std::move(max_s));
        } else {
            min_scalars.push_back(nullptr);
            max_scalars.push_back(nullptr);
        }
    }

    // Populate value_counts and null_counts dicts on host.
    for (size_t i = 0; i < output_col_names.size(); i++) {
        PyObjectPtr fid_py = PyLong_FromLongLong(field_ids[i]);
        PyObjectPtr vc_py = PyLong_FromLongLong(value_counts[i]);
        PyDict_SetItem(value_counts_dict, fid_py, vc_py);
        PyObjectPtr nc_py = PyLong_FromLongLong(null_counts[i]);
        PyDict_SetItem(null_count_dict, fid_py, nc_py);
    }

    // Batch-transfer all min/max scalars to host at once.
    // Build a 1-row cudf table of mins and another of maxes, then
    // convert each to Arrow to extract the scalars.
    std::vector<std::unique_ptr<cudf::column>> min_cols;
    std::vector<std::unique_ptr<cudf::column>> max_cols;
    std::vector<std::shared_ptr<arrow::Field>> bound_fields;
    std::vector<size_t> bound_indices;

    for (size_t i = 0; i < output_col_names.size(); i++) {
        if (!has_bounds[i])
            continue;
        min_cols.push_back(cudf::make_column_from_scalar(*min_scalars[i], 1));
        max_cols.push_back(cudf::make_column_from_scalar(*max_scalars[i], 1));
        auto field = iceberg_schema->GetFieldByName(output_col_names[i]);
        bound_fields.push_back(field);
        bound_indices.push_back(i);
    }

    if (!min_cols.empty()) {
        auto min_table = std::make_unique<cudf::table>(std::move(min_cols));
        auto max_table = std::make_unique<cudf::table>(std::move(max_cols));
        auto bound_schema = arrow::schema(bound_fields);

        auto min_arrow = convertGPUToArrow(
            GPU_DATA(std::shared_ptr<cudf::table>(std::move(min_table)),
                     bound_schema, se));
        auto max_arrow = convertGPUToArrow(
            GPU_DATA(std::shared_ptr<cudf::table>(std::move(max_table)),
                     bound_schema, se));

        for (size_t bi = 0; bi < bound_indices.size(); bi++) {
            size_t orig_i = bound_indices[bi];
            PyObjectPtr fid_py = PyLong_FromLongLong(field_ids[orig_i]);
            auto min_s = min_arrow->column(bi)->GetScalar(0).ValueOrDie();
            auto max_s = max_arrow->column(bi)->GetScalar(0).ValueOrDie();
            PyObjectPtr min_obj = arrow_scalar_to_iceberg_bytes(min_s);
            PyObjectPtr max_obj = arrow_scalar_to_iceberg_bytes(max_s);
            PyDict_SetItem(lower_bound_dict, fid_py, min_obj);
            PyDict_SetItem(upper_bound_dict, fid_py, max_obj);
        }
    }

    // Columns that cannot be bounded (nested types or all-null
    // columns) are simply omitted from the bounds dicts, matching
    // the CPU writer's behavior (generate_iceberg_field_metrics).
    // PyIceberg expects map values to be bytes, not None.
}

std::string PhysicalGPUWriteIceberg::generate_iceberg_file_name() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    boost::uuids::uuid _uuid = boost::uuids::random_generator()();
    std::string uuid = boost::uuids::to_string(_uuid);
    // Format: {rank:05d}-{rank}-{uuid}.parquet
    // (matches the CPU writer in bodo/io/iceberg_parquet_write.cpp)
    std::vector<char> fname;
    fname.resize(20 + uuid.length());
    int check = snprintf(fname.data(), fname.size(), "%05d-%d-%s.parquet", rank,
                         rank, uuid.c_str());
    if (size_t(check + 1) > fname.size()) {
        throw std::runtime_error(
            "generate_iceberg_file_name: snprintf overflow");
    }
    return std::string(fname.data());
}
