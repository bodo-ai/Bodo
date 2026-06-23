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
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include "../io/iceberg_helpers.h"
#include "../libs/_query_profile_collector.h"
#include "physical/gpu_write_parquet.h"
#include "physical/write_iceberg.h"

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
    if (this->finished) {
        return OperatorResult::FINISHED;
    }

    bool is_last = (prev_op_result == OperatorResult::FINISHED);
    // All ranks must participate in sync_is_last_non_blocking (it uses
    // MPI_Ibarrier on MPI_COMM_WORLD).  Do this before the GPU rank
    // check so non-GPU ranks don't skip the barrier and deadlock.
    is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(is_last)));

    if (!is_gpu_rank()) {
        if (is_last) {
            finished = true;
        }
        return finished ? OperatorResult::FINISHED
                        : OperatorResult::NEED_MORE_INPUT;
    }

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
        total_bytes += incoming_table->alloc_size();
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

std::unique_ptr<cudf::table> PhysicalGPUWriteIceberg::concatenate_accumulated(
    std::shared_ptr<StreamAndEvent> se) {
    std::unique_ptr<cudf::table> working_table;
    if (accumulated_tables.size() == 1) {
        working_table =
            std::make_unique<cudf::table>(accumulated_tables[0]->release());
    } else {
        std::vector<cudf::table_view> views;
        views.reserve(accumulated_tables.size());
        for (auto& t : accumulated_tables) {
            views.push_back(t->view());
        }
        working_table = cudf::concatenate(views, se->stream);
    }
    accumulated_tables.clear();
    return working_table;
}

std::unique_ptr<cudf::table> PhysicalGPUWriteIceberg::prepend_transform_columns(
    std::unique_ptr<cudf::table> working_table,
    std::shared_ptr<StreamAndEvent> se) {
    cudf::size_type n_sort = static_cast<cudf::size_type>(sort_fields.size());
    cudf::size_type n_part =
        static_cast<cudf::size_type>(partition_fields.size());
    cudf::size_type num_prepended = n_sort + n_part;

    if (num_prepended == 0) {
        return working_table;
    }

    std::vector<std::unique_ptr<cudf::column>> prepended_cols;
    prepended_cols.reserve(num_prepended);

    // Sort transform columns (first in prepended group)
    for (cudf::size_type si = 0; si < n_sort; si++) {
        const auto& sf = sort_fields[si];
        cudf::column_view orig_col = working_table->view().column(sf.col_idx);
        auto xfrm_col = apply_iceberg_transform_gpu(orig_col, sf.transform,
                                                    sf.arg, se->stream);
        prepended_cols.push_back(std::move(xfrm_col));
    }

    // Partition transform columns (second in prepended group)
    for (cudf::size_type pi = 0; pi < n_part; pi++) {
        const auto& pf = partition_fields[pi];
        cudf::column_view orig_col = working_table->view().column(pf.col_idx);
        auto xfrm_col = apply_iceberg_transform_gpu(orig_col, pf.transform,
                                                    pf.arg, se->stream);
        prepended_cols.push_back(std::move(xfrm_col));
    }

    // Build new table: prepended columns + original columns.
    auto released_cols = working_table->release();
    std::vector<std::unique_ptr<cudf::column>> all_cols;
    all_cols.reserve(num_prepended + released_cols.size());
    for (auto& pc : prepended_cols) {
        all_cols.push_back(std::move(pc));
    }
    for (auto& col : released_cols) {
        all_cols.push_back(std::move(col));
    }
    return std::make_unique<cudf::table>(std::move(all_cols));
}

void PhysicalGPUWriteIceberg::sort_working_table(
    std::unique_ptr<cudf::table>& working_table,
    std::shared_ptr<StreamAndEvent> se) {
    cudf::size_type n_sort = static_cast<cudf::size_type>(sort_fields.size());
    cudf::size_type n_part =
        static_cast<cudf::size_type>(partition_fields.size());

    std::vector<cudf::size_type> sort_key_indices;
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;

    for (cudf::size_type si = 0; si < n_sort; si++) {
        sort_key_indices.push_back(si);
        column_order.push_back(sort_fields[si].is_asc
                                   ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
        null_precedence.push_back(sort_fields[si].nulls_last
                                      ? cudf::null_order::AFTER
                                      : cudf::null_order::BEFORE);
    }
    for (cudf::size_type pi = 0; pi < n_part; pi++) {
        if (partition_fields[pi].transform == "void") {
            continue;
        }
        sort_key_indices.push_back(n_sort + pi);
        column_order.push_back(cudf::order::ASCENDING);
        null_precedence.push_back(cudf::null_order::AFTER);
    }

    if (!sort_key_indices.empty()) {
        cudf::table_view key_tv =
            working_table->view().select(sort_key_indices);
        std::unique_ptr<cudf::table> sorted =
            cudf::sort_by_key(working_table->view(), key_tv, column_order,
                              null_precedence, se->stream);
        working_table = std::move(sorted);
    }
}

void PhysicalGPUWriteIceberg::detect_partition_groups(
    const std::unique_ptr<cudf::table>& working_table,
    std::vector<PartitionGroup>& groups, std::shared_ptr<StreamAndEvent> se) {
    cudf::size_type n_sort = static_cast<cudf::size_type>(sort_fields.size());
    cudf::size_type n_part =
        static_cast<cudf::size_type>(partition_fields.size());
    cudf::size_type num_prepended = n_sort + n_part;

    if (n_part == 0) {
        groups.push_back(
            {0,
             static_cast<cudf::size_type>(working_table->num_rows()),
             {},
             {}});
        return;
    }

    // Partition transform columns start at index n_sort
    std::vector<cudf::size_type> part_xfrm_idxs(n_part);
    for (cudf::size_type pi = 0; pi < n_part; pi++) {
        part_xfrm_idxs[pi] = n_sort + pi;
    }
    cudf::table_view part_tv = working_table->view().select(part_xfrm_idxs);

    cudf::groupby::groupby gb(part_tv, cudf::null_policy::INCLUDE,
                              cudf::sorted::YES);

    auto gb_groups = gb.get_groups(/*values=*/{}, se->stream);

    const auto& offsets = gb_groups.offsets;
    if (offsets.size() < 2) {
        throw std::runtime_error(
            "PhysicalGPUWriteIceberg: groupby returned fewer "
            "than 2 offsets");
    }
    cudf::size_type n_groups = static_cast<cudf::size_type>(offsets.size() - 1);

    // Gather the first row of each group from BOTH:
    //   - the transformed partition columns (for render_partition_value)
    //   - the original partition columns (for identity-transform
    //     rendering of complex types like DATE)
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
    CHECK_CUDA(cudaMemcpyAsync(gather_view.head<int32_t>(), group_starts.data(),
                               group_starts.size() * sizeof(cudf::size_type),
                               cudaMemcpyHostToDevice, se->stream.value()));

    // Gather from transformed partition columns
    auto gathered_xfrm =
        cudf::gather(part_tv, gather_map->view(),
                     cudf::out_of_bounds_policy::DONT_CHECK, se->stream);

    // Gather from original partition columns
    std::vector<cudf::size_type> orig_part_idxs;
    orig_part_idxs.reserve(n_part);
    for (const auto& pf : partition_fields) {
        orig_part_idxs.push_back(num_prepended + pf.col_idx);
    }
    cudf::table_view orig_part_tv =
        working_table->view().select(orig_part_idxs);
    auto gathered_orig =
        cudf::gather(orig_part_tv, gather_map->view(),
                     cudf::out_of_bounds_policy::DONT_CHECK, se->stream);

    // Build Arrow schemas for the gathered columns
    std::vector<std::shared_ptr<arrow::Field>> xfrm_fields;
    std::vector<std::shared_ptr<arrow::Field>> orig_fields;
    for (const auto& pf : partition_fields) {
        orig_fields.push_back(in_schema->field(pf.col_idx));
        if (pf.transform == "truncate" &&
            in_schema->field(pf.col_idx)->type()->id() == arrow::Type::STRING) {
            xfrm_fields.push_back(arrow::field(pf.col_name, arrow::utf8()));
        } else if (pf.transform == "bucket") {
            xfrm_fields.push_back(arrow::field(pf.col_name, arrow::uint32()));
        } else if (pf.transform == "day") {
            xfrm_fields.push_back(arrow::field(pf.col_name, arrow::int64()));
        } else if (pf.transform == "identity" || pf.transform == "void") {
            xfrm_fields.push_back(in_schema->field(pf.col_idx));
        } else {
            // year, month, hour, and others produce int32
            xfrm_fields.push_back(arrow::field(pf.col_name, arrow::int32()));
        }
    }
    auto xfrm_arrow_schema = arrow::schema(xfrm_fields);
    auto orig_arrow_schema = arrow::schema(orig_fields);

    auto xfrm_arrow_table =
        convertGPUToArrow(gathered_xfrm->view(), xfrm_arrow_schema);
    auto orig_arrow_table =
        convertGPUToArrow(gathered_orig->view(), orig_arrow_schema);

    groups.reserve(n_groups);
    for (cudf::size_type gi = 0; gi < n_groups; gi++) {
        std::vector<std::shared_ptr<arrow::Scalar>> xfrm_vals;
        std::vector<std::shared_ptr<arrow::Scalar>> orig_vals;
        xfrm_vals.reserve(n_part);
        orig_vals.reserve(n_part);
        for (int pi = 0; pi < static_cast<int>(n_part); pi++) {
            xfrm_vals.push_back(
                xfrm_arrow_table->column(pi)->GetScalar(gi).ValueOrDie());
            orig_vals.push_back(
                orig_arrow_table->column(pi)->GetScalar(gi).ValueOrDie());
        }
        groups.push_back({static_cast<cudf::size_type>(offsets[gi]),
                          static_cast<cudf::size_type>(offsets[gi + 1]),
                          std::move(xfrm_vals), std::move(orig_vals)});
    }
}

void PhysicalGPUWriteIceberg::write_group(
    const cudf::table_view& working_tv, cudf::size_type num_prepended,
    const std::vector<std::string>& output_col_names,
    const std::vector<cudf::size_type>& output_col_indices,
    const PartitionGroup& group, std::shared_ptr<StreamAndEvent> se) {
    cudf::size_type start = group.start_row;
    cudf::size_type end = group.end_row;
    if (end <= start) {
        return;
    }

    cudf::size_type n_part =
        static_cast<cudf::size_type>(partition_fields.size());

    // Build partition directory path
    std::string part_path;
    if (n_part > 0) {
        for (size_t pi = 0; pi < static_cast<size_t>(n_part); pi++) {
            if (!part_path.empty()) {
                part_path += "/";
            }
            const auto& pf = partition_fields[pi];
            std::string val_str = render_partition_value(pf.transform, pf.arg,
                                                         group.orig_values[pi],
                                                         group.xfrm_values[pi]);
            part_path += pf.partition_name + "=" + val_str;
        }
    }

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
        arrow::Status mkdir_status = fs->CreateDir(dir_path_str);
        if (!mkdir_status.ok() && !mkdir_status.IsAlreadyExists()) {
            CHECK_ARROW_GPU_ICEBERG(
                mkdir_status, "PhysicalGPUWriteIceberg: CreateDir failed");
        }
    }

    // Build file name
    std::string fname = generate_iceberg_file_name();
    std::filesystem::path out_path = dir_path / fname;

    // Relative path from table_loc
    std::string rel_path;
    if (part_path.empty()) {
        rel_path = fname;
    } else {
        rel_path = part_path + "/" + fname;
    }

    // Slice working table to this partition group's rows
    std::vector<cudf::table_view> sliced =
        cudf::slice(working_tv, {start, end});
    cudf::table_view group_view = sliced[0];

    // Output column indices are offset by num_prepended
    std::vector<cudf::size_type> out_idxs;
    out_idxs.reserve(output_col_indices.size());
    for (auto idx : output_col_indices) {
        out_idxs.push_back(num_prepended + idx);
    }
    cudf::table_view out_tv = group_view.select(out_idxs);

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

    // Compute per-column stats on GPU
    PyObject* value_counts_dict = PyDict_New();
    PyObject* null_count_dict = PyDict_New();
    PyObject* lower_bound_dict = PyDict_New();
    PyObject* upper_bound_dict = PyDict_New();
    compute_field_metrics_gpu(out_tv, output_col_names, se, value_counts_dict,
                              null_count_dict, lower_bound_dict,
                              upper_bound_dict);

    int n_part_vals = static_cast<int>(group.xfrm_values.size());

    // Tuple: file_name, record_count, file_size,
    //        value_counts_dict, null_count_dict,
    //        lower_bound_dict,  upper_bound_dict,
    //        *partition_values
    PyObjectPtr file_info_tuple = PyTuple_New(7 + n_part_vals);
    PyTuple_SetItem(file_info_tuple, 0, PyUnicode_FromString(rel_path.c_str()));
    PyTuple_SetItem(file_info_tuple, 1, PyLong_FromLongLong(record_count));
    PyTuple_SetItem(file_info_tuple, 2, PyLong_FromLongLong(file_size));
    PyTuple_SetItem(file_info_tuple, 3, value_counts_dict);
    PyTuple_SetItem(file_info_tuple, 4, null_count_dict);
    PyTuple_SetItem(file_info_tuple, 5, lower_bound_dict);
    PyTuple_SetItem(file_info_tuple, 6, upper_bound_dict);

    for (int pi = 0; pi < n_part_vals; pi++) {
        PyObject* py_val = arrow_scalar_to_pyobject(group.xfrm_values[pi]);
        PyTuple_SetItem(file_info_tuple, 7 + pi, py_val);
    }

    PyList_Append(iceberg_files_info_py, file_info_tuple);

    metrics.n_files_written++;
}

void PhysicalGPUWriteIceberg::flush_buffer(std::shared_ptr<StreamAndEvent> se,
                                           bool is_last) {
    time_pt start_sort = start_timer();

    auto working_table = concatenate_accumulated(se);
    working_table = prepend_transform_columns(std::move(working_table), se);
    sort_working_table(working_table, se);

    metrics.sort_time += end_timer(start_sort);
    time_pt start_file_write = start_timer();

    cudf::size_type n_sort = static_cast<cudf::size_type>(sort_fields.size());
    cudf::size_type n_part =
        static_cast<cudf::size_type>(partition_fields.size());
    cudf::size_type num_prepended = n_sort + n_part;

    std::vector<std::string> output_col_names;
    std::vector<cudf::size_type> output_col_indices;
    for (int i = 0; i < in_schema->num_fields(); i++) {
        output_col_indices.push_back(i);
        output_col_names.push_back(in_schema->field_names()[i]);
    }

    std::vector<PartitionGroup> groups;
    detect_partition_groups(working_table, groups, se);

    metrics.n_partition_groups +=
        static_cast<MetricBase::StatValue>(groups.size());

    for (const auto& g : groups) {
        write_group(working_table->view(), num_prepended, output_col_names,
                    output_col_indices, g, se);
    }

    metrics.file_write_time += end_timer(start_file_write);

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
    iceberg_files_info_py = gather_iceberg_files_info(iceberg_files_info_py);
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

// Iceberg transform kernels (GPU)

std::unique_ptr<cudf::column>
PhysicalGPUWriteIceberg::apply_iceberg_transform_gpu(
    cudf::column_view col, const std::string& transform_name, long arg,
    rmm::cuda_stream_view stream) {
    cudf::size_type n = col.size();

    // Helper: create a filled numeric column from a scalar value.
    auto make_filled_col = [&](auto value) {
        auto s = std::make_unique<cudf::numeric_scalar<decltype(value)>>(
            value, true, stream);
        return cudf::make_column_from_scalar(*s, n, stream);
    };

    if (transform_name == "identity") {
        // identity transform: pass-through for most types, but DATETIME
        // (TIMESTAMP_NANOSECONDS) must be converted to microseconds for
        // partition correctness (two different ns values may map to the
        // same µs).
        if (col.type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS) {
            auto ts_i64 =
                cudf::cast(col, cudf::data_type{cudf::type_id::INT64}, stream);
            auto col_1000 = make_filled_col(static_cast<int64_t>(1000));
            auto us_i64 = cudf::binary_operation(
                ts_i64->view(), col_1000->view(), cudf::binary_operator::DIV,
                cudf::data_type{cudf::type_id::INT64}, stream);
            // Keep as INT64 (µs since epoch), matching CPU writer's
            // convert_datetime_ns_to_us output type.
            return us_i64;
        }
        return std::make_unique<cudf::column>(col);
    }
    if (transform_name == "void") {
        auto null_scalar =
            std::make_unique<cudf::numeric_scalar<int32_t>>(0, false, stream);
        return cudf::make_column_from_scalar(*null_scalar, n, stream);
    }
    if (transform_name == "year") {
        cudf::type_id tid = col.type().id();
        if (tid == cudf::type_id::TIMESTAMP_NANOSECONDS ||
            tid == cudf::type_id::TIMESTAMP_MICROSECONDS ||
            tid == cudf::type_id::TIMESTAMP_MILLISECONDS ||
            tid == cudf::type_id::TIMESTAMP_SECONDS ||
            tid == cudf::type_id::TIMESTAMP_DAYS) {
            auto year_col = cudf::datetime::extract_datetime_component(
                col, cudf::datetime::datetime_component::YEAR, stream);
            auto yr_int32 =
                cudf::cast(year_col->view(),
                           cudf::data_type{cudf::type_id::INT32}, stream);
            auto col_1970 = make_filled_col(static_cast<int32_t>(1970));
            auto result = cudf::binary_operation(
                yr_int32->view(), col_1970->view(), cudf::binary_operator::SUB,
                cudf::data_type{cudf::type_id::INT32}, stream);
            return result;
        }
        throw std::runtime_error(
            "apply_iceberg_transform_gpu: year transform requires "
            "TIMESTAMP or DATE type, got " +
            std::to_string(static_cast<int>(tid)));
    }
    if (transform_name == "month") {
        cudf::type_id tid = col.type().id();
        if (tid == cudf::type_id::TIMESTAMP_NANOSECONDS ||
            tid == cudf::type_id::TIMESTAMP_MICROSECONDS ||
            tid == cudf::type_id::TIMESTAMP_MILLISECONDS ||
            tid == cudf::type_id::TIMESTAMP_SECONDS ||
            tid == cudf::type_id::TIMESTAMP_DAYS) {
            auto year_col = cudf::datetime::extract_datetime_component(
                col, cudf::datetime::datetime_component::YEAR, stream);
            auto month_col = cudf::datetime::extract_datetime_component(
                col, cudf::datetime::datetime_component::MONTH, stream);
            // months_since_epoch = (year - 1970) * 12 + (month - 1)
            auto yr_int32 =
                cudf::cast(year_col->view(),
                           cudf::data_type{cudf::type_id::INT32}, stream);
            auto mo_int32 =
                cudf::cast(month_col->view(),
                           cudf::data_type{cudf::type_id::INT32}, stream);
            auto col_1970 = make_filled_col(static_cast<int32_t>(1970));
            auto yr_minus = cudf::binary_operation(
                yr_int32->view(), col_1970->view(), cudf::binary_operator::SUB,
                cudf::data_type{cudf::type_id::INT32}, stream);
            auto col_12 = make_filled_col(static_cast<int32_t>(12));
            auto yr_mul = cudf::binary_operation(
                yr_minus->view(), col_12->view(), cudf::binary_operator::MUL,
                cudf::data_type{cudf::type_id::INT32}, stream);
            auto col_1 = make_filled_col(static_cast<int32_t>(1));
            auto mo_minus = cudf::binary_operation(
                mo_int32->view(), col_1->view(), cudf::binary_operator::SUB,
                cudf::data_type{cudf::type_id::INT32}, stream);
            auto result = cudf::binary_operation(
                yr_mul->view(), mo_minus->view(), cudf::binary_operator::ADD,
                cudf::data_type{cudf::type_id::INT32}, stream);
            return result;
        }
        throw std::runtime_error(
            "apply_iceberg_transform_gpu: month transform requires "
            "TIMESTAMP or DATE type, got " +
            std::to_string(static_cast<int>(tid)));
    }
    if (transform_name == "day") {
        cudf::type_id tid = col.type().id();
        if (tid == cudf::type_id::TIMESTAMP_DAYS) {
            // cudf blocks timestamp→int cast, but allows
            // timestamp→duration and duration→int.
            auto dur = cudf::cast(
                col, cudf::data_type{cudf::type_id::DURATION_DAYS}, stream);
            return cudf::cast(dur->view(),
                              cudf::data_type{cudf::type_id::INT64}, stream);
        }
        if (tid == cudf::type_id::TIMESTAMP_NANOSECONDS) {
            constexpr int64_t ns_per_day =
                24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
            auto col_ns = make_filled_col(ns_per_day);
            auto ts_i64 =
                cudf::cast(col, cudf::data_type{cudf::type_id::INT64}, stream);
            auto result = cudf::binary_operation(
                ts_i64->view(), col_ns->view(), cudf::binary_operator::DIV,
                cudf::data_type{cudf::type_id::INT64}, stream);
            return result;
        }
        throw std::runtime_error(
            "apply_iceberg_transform_gpu: day transform requires "
            "TIMESTAMP or DATE type, got " +
            std::to_string(static_cast<int>(tid)));
    }
    if (transform_name == "hour") {
        cudf::type_id tid = col.type().id();
        if (tid == cudf::type_id::TIMESTAMP_NANOSECONDS) {
            constexpr int64_t ns_per_hour =
                60LL * 60LL * 1000LL * 1000LL * 1000LL;
            auto col_ns = make_filled_col(ns_per_hour);
            auto ts_i64 =
                cudf::cast(col, cudf::data_type{cudf::type_id::INT64}, stream);
            auto result_i64 = cudf::binary_operation(
                ts_i64->view(), col_ns->view(), cudf::binary_operator::DIV,
                cudf::data_type{cudf::type_id::INT64}, stream);
            return cudf::cast(result_i64->view(),
                              cudf::data_type{cudf::type_id::INT32}, stream);
        }
        throw std::runtime_error(
            "apply_iceberg_transform_gpu: hour transform requires "
            "TIMESTAMP type, got " +
            std::to_string(static_cast<int>(tid)));
    }
    if (transform_name == "truncate") {
        long W = arg;
        if (W <= 0) {
            throw std::runtime_error(
                "apply_iceberg_transform_gpu: truncate W must be > 0");
        }
        cudf::type_id tid = col.type().id();
        if (tid == cudf::type_id::STRING) {
            auto start_scalar =
                std::make_unique<cudf::numeric_scalar<int>>(0, true, stream);
            auto stop_scalar = std::make_unique<cudf::numeric_scalar<int>>(
                static_cast<int>(W), true, stream);
            auto step_scalar =
                std::make_unique<cudf::numeric_scalar<int>>(1, true, stream);
            return cudf::strings::slice_strings(cudf::strings_column_view{col},
                                                *start_scalar, *stop_scalar,
                                                *step_scalar, stream);
        }
        if (tid == cudf::type_id::INT32 || tid == cudf::type_id::INT64) {
            // v - (((v % W) + W) % W)
            auto col_w = (tid == cudf::type_id::INT32)
                             ? make_filled_col(static_cast<int32_t>(W))
                             : make_filled_col(static_cast<int64_t>(W));
            auto mod1 = cudf::binary_operation(col, col_w->view(),
                                               cudf::binary_operator::MOD,
                                               col.type(), stream);
            auto mod2 = cudf::binary_operation(mod1->view(), col_w->view(),
                                               cudf::binary_operator::ADD,
                                               col.type(), stream);
            auto mod3 = cudf::binary_operation(mod2->view(), col_w->view(),
                                               cudf::binary_operator::MOD,
                                               col.type(), stream);
            auto result = cudf::binary_operation(col, mod3->view(),
                                                 cudf::binary_operator::SUB,
                                                 col.type(), stream);
            return result;
        }
        throw std::runtime_error(
            "apply_iceberg_transform_gpu: truncate transform requires "
            "INT32, INT64, or STRING type, got " +
            std::to_string(static_cast<int>(tid)));
    }
    if (transform_name == "bucket") {
        long N = arg;
        if (N <= 0) {
            throw std::runtime_error(
                "apply_iceberg_transform_gpu: bucket N must be > 0");
        }
        // MurmurHash3_x86_32 requires a table_view, so wrap column.
        auto col_wrapper = std::make_unique<cudf::column>(col);
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(std::move(col_wrapper));
        auto tmp_table = std::make_unique<cudf::table>(std::move(cols));
        auto hash_col =
            cudf::hashing::murmurhash3_x86_32(tmp_table->view(), 0, stream);
        auto col_mask = make_filled_col(static_cast<uint32_t>(INT_MAX));
        auto masked = cudf::binary_operation(
            hash_col->view(), col_mask->view(),
            cudf::binary_operator::BITWISE_AND,
            cudf::data_type{cudf::type_id::UINT32}, stream);
        auto col_n = make_filled_col(static_cast<uint32_t>(N));
        auto result = cudf::binary_operation(
            masked->view(), col_n->view(), cudf::binary_operator::MOD,
            cudf::data_type{cudf::type_id::UINT32}, stream);
        return result;
    }
    throw std::runtime_error(
        "apply_iceberg_transform_gpu: unsupported transform '" +
        transform_name + "'");
}

/**
 * @brief Convert Arrow scalar to string for partition path rendering.
 *
 * Handles identity-transform cases where the Iceberg spec uses the
 * scalar's natural rendering (e.g., dates as "YYYY-MM-DD").
 * For non-identity transforms, uses the transformed scalar value
 * and applies transform-specific formatting.
 */
std::string PhysicalGPUWriteIceberg::render_partition_value(
    const std::string& transform_name, long arg,
    const std::shared_ptr<arrow::Scalar>& orig_scalar,
    const std::shared_ptr<arrow::Scalar>& transformed_scalar) {
    if (!transformed_scalar->is_valid) {
        return "__HIVE_DEFAULT_PARTITION__";
    }
    if (transform_name == "identity" || transform_name == "void") {
        // Identity/void: render the original scalar as-is
        // (void transforms produce all-null, which is caught above)
        return orig_scalar->is_valid ? orig_scalar->ToString()
                                     : "__HIVE_DEFAULT_PARTITION__";
    }
    if (transform_name == "bucket") {
        return std::to_string(
            std::static_pointer_cast<arrow::UInt32Scalar>(transformed_scalar)
                ->value);
    }
    if (transform_name == "truncate") {
        return transformed_scalar->ToString();
    }
    if (transform_name == "year") {
        int32_t v =
            std::static_pointer_cast<arrow::Int32Scalar>(transformed_scalar)
                ->value;
        return std::to_string(v + 1970);
    }
    if (transform_name == "month") {
        int32_t v =
            std::static_pointer_cast<arrow::Int32Scalar>(transformed_scalar)
                ->value;
        int32_t yr = (v / 12) + 1970;
        int32_t mo = (v % 12) + 1;
        // Lets use fmt instead
        return fmt::format("{:04}-{:02}", yr, mo);
    }
    if (transform_name == "day") {
        int64_t v =
            std::static_pointer_cast<arrow::Int64Scalar>(transformed_scalar)
                ->value;
        // days since epoch → date string
        // Compute year/month/day from days since 1970-01-01
        int64_t days = v;
        // Use the original scalar to render the date string if available
        // (the original holds the actual date value for DATE types)
        if (orig_scalar->is_valid &&
            (orig_scalar->type->id() == arrow::Type::DATE32 ||
             orig_scalar->type->id() == arrow::Type::TIMESTAMP)) {
            // For DATE inputs, the original scalar's ToString() gives
            // the properly formatted date.
            return orig_scalar->ToString();
        }
        // Fallback for DATETIME input: render as ISO date from days since
        // epoch.
        // Simple approach: the original datetime scalar's ToString() works.
        if (orig_scalar->is_valid) {
            return orig_scalar->ToString();
        }
        // Absolute fallback: days value as-is
        return std::to_string(days);
    }
    if (transform_name == "hour") {
        int32_t v =
            std::static_pointer_cast<arrow::Int32Scalar>(transformed_scalar)
                ->value;
        // hours since epoch: render as date string from original scalar
        if (orig_scalar->is_valid) {
            return orig_scalar->ToString();
        }
        return std::to_string(v);
    }
    // Unknown transform: fallback to string representation
    return transformed_scalar->ToString();
}

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

PyObject* PhysicalGPUWriteIceberg::arrow_scalar_to_pyobject(
    const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar->is_valid) {
        Py_RETURN_NONE;
    }
    switch (scalar->type->id()) {
        case arrow::Type::INT8:
            return PyLong_FromLong(
                std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value);
        case arrow::Type::INT16:
            return PyLong_FromLong(
                std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value);
        case arrow::Type::INT32:
            return PyLong_FromLong(
                std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value);
        case arrow::Type::INT64:
            return PyLong_FromLongLong(
                std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
        case arrow::Type::UINT8:
            return PyLong_FromUnsignedLong(
                std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value);
        case arrow::Type::UINT16:
            return PyLong_FromUnsignedLong(
                std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value);
        case arrow::Type::UINT32:
            return PyLong_FromUnsignedLong(
                std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value);
        case arrow::Type::UINT64:
            return PyLong_FromUnsignedLongLong(
                std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
        case arrow::Type::FLOAT:
            return PyFloat_FromDouble(
                std::static_pointer_cast<arrow::FloatScalar>(scalar)->value);
        case arrow::Type::DOUBLE:
            return PyFloat_FromDouble(
                std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value);
        case arrow::Type::STRING: {
            return PyUnicode_FromString(
                std::static_pointer_cast<arrow::StringScalar>(scalar)
                    ->value->ToString()
                    .c_str());
        }
        case arrow::Type::BOOL: {
            PyObject* res =
                std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value
                    ? Py_True
                    : Py_False;
            Py_INCREF(res);
            return res;
        }
        case arrow::Type::DATE32:
            return PyLong_FromLong(
                std::static_pointer_cast<arrow::Date32Scalar>(scalar)->value);
        case arrow::Type::TIMESTAMP:
            return PyLong_FromLongLong(
                std::static_pointer_cast<arrow::TimestampScalar>(scalar)
                    ->value);
        case arrow::Type::LARGE_STRING: {
            auto s = std::static_pointer_cast<arrow::LargeStringScalar>(scalar);
            return PyUnicode_FromString(s->value->ToString().c_str());
        }
        case arrow::Type::BINARY: {
            auto b = std::static_pointer_cast<arrow::BinaryScalar>(scalar);
            return PyUnicode_FromStringAndSize(
                reinterpret_cast<const char*>(b->value->data()),
                static_cast<Py_ssize_t>(b->value->size()));
        }
        case arrow::Type::LARGE_BINARY: {
            auto b = std::static_pointer_cast<arrow::LargeBinaryScalar>(scalar);
            return PyUnicode_FromStringAndSize(
                reinterpret_cast<const char*>(b->value->data()),
                static_cast<Py_ssize_t>(b->value->size()));
        }
        case arrow::Type::DECIMAL128:
            return PyUnicode_FromString(scalar->ToString().c_str());
        default:
            throw std::runtime_error(
                "PhysicalGPUWriteIceberg: unsupported "
                "partition value type: " +
                scalar->type->ToString() + " (name=" + scalar->type->name() +
                ")");
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
        if (!has_bounds[i]) {
            continue;
        }
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

        auto min_arrow = convertGPUToArrow(min_table->view(), bound_schema);
        auto max_arrow = convertGPUToArrow(max_table->view(), bound_schema);

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
