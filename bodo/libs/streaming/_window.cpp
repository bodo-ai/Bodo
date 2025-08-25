#include "_window.h"
#include <vector>

#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_memory_budget.h"
#include "../groupby/_groupby_ftypes.h"
#include "../window/_window_calculator.h"
#include "../window/_window_compute.h"

WindowStateSorter::WindowStateSorter(
    std::shared_ptr<bodo::Schema>& build_table_schema, size_t n_window_keys,
    const std::vector<bool>& order_by_asc, const std::vector<bool>& order_by_na,
    bool parallel)
    : build_table_schema(build_table_schema),
      num_keys(n_window_keys + order_by_asc.size()),
      skip_sorting(num_keys == 0),
      asc(num_keys),
      na_pos(num_keys) {
    // Set arbitrary values for sort properties for partition by
    // keys.
    for (size_t i = 0; i < n_window_keys; i++) {
        asc[i] = 0;
        na_pos[i] = 0;
    }
    // Use the sort properties for order by keys.
    for (size_t i = 0; i < order_by_asc.size(); i++) {
        size_t dest_index = i + n_window_keys;
        asc[dest_index] = order_by_asc[i];
        na_pos[dest_index] = order_by_na[i];
    }

    // TODO(aneesh): replace this with a CTB
    build_table_dict_builders.resize(build_table_schema->column_types.size());
    for (size_t i = 0; i < build_table_schema->column_types.size(); i++) {
        // Mark the partition by and order by columns as keys since it may
        // be needed or useful for sort.
        bool is_key = i < num_keys;
        build_table_dict_builders[i] = create_dict_builder_for_array(
            build_table_schema->column_types[i]->copy(), is_key);
    }

    if (skip_sorting) {
        build_table_buffer = std::make_unique<TableBuildBuffer>(
            build_table_schema, build_table_dict_builders);
        return;
    }

    stream_sorter = std::make_unique<StreamSortState>(
        -1, num_keys, std::move(asc), std::move(na_pos), build_table_schema,
        parallel);
}

void WindowStateSorter::AppendBatch(std::shared_ptr<table_info>& table,
                                    bool is_last) {
    if (skip_sorting) {
        assert(build_table_buffer != nullptr);
        build_table_buffer->UnifyTablesAndAppend(table,
                                                 build_table_dict_builders);

        return;
    }

    assert(stream_sorter != nullptr);
    stream_sorter->ConsumeBatch(table);
    if (is_last) {
        stream_sorter->FinalizeBuild();
    }
}

std::vector<std::shared_ptr<table_info>> WindowStateSorter::Finalize() {
    if (skip_sorting) {
        // TODO(aneesh) use a CTB instead to return a vector of unpinned chunks
        // Skip sort if there are no partition/order columns.
        assert(build_table_buffer != nullptr);
        return {build_table_buffer->data_table};
    }

    assert(stream_sorter != nullptr);
    std::vector<std::shared_ptr<table_info>> chunks =
        stream_sorter->GetAllOutputUnpinned();
    if (chunks.empty()) {
        return {stream_sorter->dummy_output_chunk};
    }
    return chunks;
}

WindowState::WindowState(const std::unique_ptr<bodo::Schema>& in_schema_,
                         std::vector<int32_t> window_ftypes_, uint64_t n_keys_,
                         std::vector<bool> order_by_asc_,
                         std::vector<bool> order_by_na_,
                         std::vector<bool> cols_to_keep_,
                         std::vector<int32_t> func_input_indices_,
                         std::vector<int32_t> func_input_offsets_,
                         int64_t output_batch_size_, bool parallel_,
                         int64_t sync_iter_, int64_t op_id_,
                         int64_t op_pool_size_bytes_,
                         bool allow_work_stealing_)
    :  // Create the operator buffer pool
      op_pool(std::make_unique<bodo::OperatorBufferPool>(
          op_id_,
          ((op_pool_size_bytes_ == -1)
               ? static_cast<uint64_t>(
                     bodo::BufferPool::Default()->get_memory_size_bytes() *
                     // TODO: Add a window specific value
                     GROUPBY_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL)
               : op_pool_size_bytes_),
          bodo::BufferPool::Default(),
          // TODO: Add a window specific value
          GROUPBY_OPERATOR_BUFFER_POOL_ERROR_THRESHOLD)),
      op_mm(bodo::buffer_memory_manager(op_pool.get())),
      op_scratch_pool(
          std::make_unique<bodo::OperatorScratchPool>(this->op_pool.get())),
      op_scratch_mm(bodo::buffer_memory_manager(this->op_scratch_pool.get())),
      n_keys(n_keys_),
      parallel(parallel_),
      output_batch_size(output_batch_size_),
      window_ftypes(std::move(window_ftypes_)),
      order_by_asc(std::move(order_by_asc_)),
      order_by_na(std::move(order_by_na_)),
      cols_to_keep_bitmask(std::move(cols_to_keep_)),
      func_input_indices(std::move(func_input_indices_)),
      func_input_offsets(std::move(func_input_offsets_)),
      sync_iter(sync_iter_),
      // Build schema always matches the input schema.
      build_table_schema(std::make_shared<bodo::Schema>(*in_schema_)),
      sorter(build_table_schema, n_keys, order_by_asc, order_by_na, parallel),
      op_id(op_id_) {
    // Enable/disable work stealing
    char* disable_output_work_stealing_env_ =
        std::getenv("BODO_STREAM_WINDOW_DISABLE_OUTPUT_WORK_STEALING");
    bool allow_work_stealing = allow_work_stealing_;
    if (disable_output_work_stealing_env_) {
        allow_work_stealing &=
            std::strcmp(disable_output_work_stealing_env_, "0") == 0;
    }
    this->enable_output_work_stealing = allow_work_stealing;

    // Build schema always matches the input schema.
    this->build_table_schema = std::make_shared<bodo::Schema>(*in_schema_);

    // Generate the output schema
    std::unique_ptr<bodo::Schema> output_schema =
        std::make_unique<bodo::Schema>();
    // Window outputs all the input columns - where partition_by_cols_to_keep or
    // order_by_cols_to_keep or input_cols_to_keep are false + the number of
    // function columns.

    size_t dropped_cols = std::count(cols_to_keep_bitmask.begin(),
                                     cols_to_keep_bitmask.end(), false);
    size_t num_output_cols = (this->build_table_schema->column_types.size() -
                              dropped_cols + window_ftypes.size());
    // Create separate dictionary builders for the output because the sort step
    // creates a global dictionary right now which would require transposing.
    std::vector<std::shared_ptr<DictionaryBuilder>> output_dict_builders(
        num_output_cols);
    size_t output_index = 0;
    for (size_t i = 0; i < cols_to_keep_bitmask.size(); i++) {
        if (cols_to_keep_bitmask[i]) {
            output_schema->append_column(
                this->build_table_schema->column_types[i]->copy());
            output_dict_builders[output_index++] =
                create_dict_builder_for_array(
                    this->build_table_schema->column_types[i]->copy(),
                    i < this->sorter.NumSortKeys() + n_keys);
        }
    }

    // Append the window function output types.

    for (size_t func_idx = 0; func_idx < window_ftypes.size(); func_idx++) {
        InferWindowOutputDataType(func_idx, output_schema);
        size_t out_idx = output_schema->column_types.size() - 1;
        output_dict_builders[output_index++] = create_dict_builder_for_array(
            output_schema->column_types[out_idx]->copy(), false);
    }
    this->output_state = std::make_shared<GroupbyOutputState>(
        std::move(output_schema), std::move(output_dict_builders),
        /*chunk_size*/ this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        /*enable_work_stealing_*/ this->parallel &&
            this->enable_output_work_stealing);

    if (this->op_id != -1) {
        std::vector<MetricBase> metrics;
        metrics.reserve(3);
        MetricBase::BlobValue agg_type =
            get_aggregation_type_string(AggregationType::WINDOW);
        metrics.emplace_back(BlobMetric("aggregation_type", agg_type, true));
        MetricBase::BlobValue acc_or_agg = "ACC";
        metrics.emplace_back(BlobMetric("acc_or_agg", acc_or_agg, true));
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }
    this->curr_stage_id++;

    if (char* debug_window_env_ =
            std::getenv("BODO_DEBUG_STREAM_GROUPBY_PARTITIONING")) {
        this->debug_window = !std::strcmp(debug_window_env_, "1");
    }
}

void WindowState::InferWindowOutputDataType(
    int32_t func_idx, std::unique_ptr<bodo::Schema>& out_schema) {
    int32_t window_ftype = window_ftypes[func_idx];
    bodo_array_type::arr_type_enum arr_type =
        bodo_array_type::NULLABLE_INT_BOOL;
    Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::INT64;
    switch (window_ftype) {
        case Bodo_FTypes::dense_rank:
        case Bodo_FTypes::rank:
        case Bodo_FTypes::row_number:
        case Bodo_FTypes::cume_dist:
        case Bodo_FTypes::percent_rank:
        case Bodo_FTypes::size:
        case Bodo_FTypes::count: {
            break;
        }
        case Bodo_FTypes::first:
        case Bodo_FTypes::last:
        case Bodo_FTypes::count_if:
        case Bodo_FTypes::booland_agg:
        case Bodo_FTypes::boolor_agg:
        case Bodo_FTypes::bitand_agg:
        case Bodo_FTypes::bitor_agg:
        case Bodo_FTypes::bitxor_agg:
        case Bodo_FTypes::sum:
        case Bodo_FTypes::min:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::max: {
            int32_t in_col_offset = func_input_offsets[func_idx];
            int32_t in_col_idx = func_input_indices[in_col_offset];
            arr_type = build_table_schema->column_types[in_col_idx]->array_type;
            dtype = build_table_schema->column_types[in_col_idx]->c_type;
            break;
        }
        default: {
            throw std::runtime_error("Unsupported streaming window ftype: " +
                                     (get_name_for_Bodo_FTypes(window_ftype)));
        }
    }

    std::tie(arr_type, dtype) =
        get_groupby_output_dtype(window_ftype, arr_type, dtype);
    out_schema->append_column(
        std::make_unique<bodo::DataType>(arr_type, dtype));
}

void WindowState::AllocWindowOutputColumn(
    int32_t func_idx, size_t output_rows,
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Use InferWindowOutputDataType to fetch the desired array/data type
    std::unique_ptr<bodo::Schema> dummy_schema =
        std::make_unique<bodo::Schema>();
    InferWindowOutputDataType(func_idx, dummy_schema);
    auto arr_type = dummy_schema->column_types[0]->array_type;
    auto dtype = dummy_schema->column_types[0]->c_type;
    // For string/dict arrays, allocate an empty array since the real array
    // will be created at the end.
    std::shared_ptr<array_info> out_arr;
    if (arr_type == bodo_array_type::STRING ||
        arr_type == bodo_array_type::DICT) {
        out_arr = alloc_string_array(Bodo_CTypes::STRING, 0, 0, 0, 0, false,
                                     false, false, pool, mm);
    } else {
        size_t alloc_rows = output_rows;
        if ((this->n_keys + this->order_by_asc.size()) == 0) {
            // If there is no partition or order, allocate just a single row
            // to store the the local/global value.
            alloc_rows = 1;
        }
        out_arr = alloc_array_top_level(alloc_rows, -1, -1, arr_type, dtype, -1,
                                        0, 0, false, false, false, pool, mm);
    }
    aggfunc_output_initialize(out_arr, this->window_ftypes[func_idx], true);
    out_arrs.push_back(out_arr);
}

std::shared_ptr<table_info> WindowState::UnifyDictionaryArrays(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (dict_builders[i] == nullptr) {
            out_arr = in_arr;
        } else {
            out_arr = dict_builders[i]->UnifyDictionaryArray(in_arr);
        }
        out_arrs.emplace_back(out_arr);
    }

    return std::make_shared<table_info>(out_arrs);
}

// TODO: Update these metrics to reflect the actual metrics that
// are useful/helpful for a sort based implementation. Right now
// we just reuse the group by stats and eliminate any non-window
// code paths.
void WindowState::ReportBuildMetrics() {
    std::vector<MetricBase> metrics;
    metrics.reserve(128);

    metrics.emplace_back(StatMetric("n_repartitions_in_append",
                                    this->metrics.n_repartitions_in_append));
    metrics.emplace_back(StatMetric("n_repartitions_in_finalize",
                                    this->metrics.n_repartitions_in_finalize));
    metrics.emplace_back(TimerMetric("repartitioning_time_total",
                                     this->metrics.repartitioning_time));
    metrics.emplace_back(
        TimerMetric("repartitioning_part_hashing_time",
                    this->metrics.repartitioning_part_hashing_time));
    metrics.emplace_back(
        StatMetric("repartitioning_part_hashing_nrows",
                   this->metrics.repartitioning_part_hashing_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_active_part1_append_time",
                    this->metrics.repartitioning_active_part1_append_time));
    metrics.emplace_back(
        StatMetric("repartitioning_active_part1_append_nrows",
                   this->metrics.repartitioning_active_part1_append_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_active_part2_append_time",
                    this->metrics.repartitioning_active_part2_append_time));
    metrics.emplace_back(
        StatMetric("repartitioning_active_part2_append_nrows",
                   this->metrics.repartitioning_active_part2_append_nrows));
    metrics.emplace_back(
        TimerMetric("repartitioning_inactive_pop_chunk_time",
                    this->metrics.repartitioning_inactive_pop_chunk_time));
    metrics.emplace_back(
        StatMetric("repartitioning_inactive_pop_chunk_n_chunks",
                   this->metrics.repartitioning_inactive_pop_chunk_n_chunks));
    metrics.emplace_back(
        TimerMetric("repartitioning_inactive_append_time",
                    this->metrics.repartitioning_inactive_append_time));

    metrics.emplace_back(
        TimerMetric("appends_active_time", this->metrics.appends_active_time));
    metrics.emplace_back(
        StatMetric("appends_active_nrows", this->metrics.appends_active_nrows));

    metrics.emplace_back(TimerMetric("input_part_hashing_time",
                                     this->metrics.input_part_hashing_time));
    metrics.emplace_back(
        StatMetric("input_hashing_nrows", this->metrics.input_hashing_nrows));
    metrics.emplace_back(TimerMetric("input_partition_check_time",
                                     this->metrics.input_partition_check_time));
    metrics.emplace_back(StatMetric("input_partition_check_nrows",
                                    this->metrics.input_partition_check_nrows));
    metrics.emplace_back(TimerMetric("appends_inactive_time",
                                     this->metrics.appends_inactive_time));
    metrics.emplace_back(StatMetric("appends_inactive_nrows",
                                    this->metrics.appends_inactive_nrows));

    // Final number of partitions
    metrics.emplace_back(
        StatMetric("n_partitions", this->metrics.n_partitions));
    metrics.emplace_back(
        TimerMetric("finalize_time_total", this->metrics.finalize_time));

    metrics.emplace_back(
        TimerMetric("finalize_window_compute_time",
                    this->metrics.finalize_window_compute_time));
    metrics.emplace_back(
        TimerMetric("finalize_colset_update_time",
                    this->metrics.finalize_update_metrics.colset_update_time));
    metrics.emplace_back(
        StatMetric("finalize_colset_update_nrows",
                   this->metrics.finalize_update_metrics.colset_update_nrows));
    metrics.emplace_back(TimerMetric(
        "finalize_hashing_time",
        this->metrics.finalize_update_metrics.grouping_metrics.hashing_time));
    metrics.emplace_back(StatMetric(
        "finalize_hashing_nrows",
        this->metrics.finalize_update_metrics.grouping_metrics.hashing_nrows));
    metrics.emplace_back(TimerMetric(
        "finalize_grouping_time",
        this->metrics.finalize_update_metrics.grouping_metrics.grouping_time));
    metrics.emplace_back(StatMetric(
        "finalize_grouping_nrows",
        this->metrics.finalize_update_metrics.grouping_metrics.grouping_nrows));
    metrics.emplace_back(TimerMetric(
        "finalize_hll_time",
        this->metrics.finalize_update_metrics.grouping_metrics.hll_time));
    metrics.emplace_back(StatMetric(
        "finalize_hll_nrows",
        this->metrics.finalize_update_metrics.grouping_metrics.hll_nrows));

    metrics.emplace_back(
        TimerMetric("finalize_eval_time", this->metrics.finalize_eval_time));
    metrics.emplace_back(
        StatMetric("finalize_eval_nrows", this->metrics.finalize_eval_nrows));
    metrics.emplace_back(
        TimerMetric("finalize_activate_partition_time",
                    this->metrics.finalize_activate_partition_time));
    metrics.emplace_back(
        TimerMetric("finalize_activate_pin_chunk_time",
                    this->metrics.finalize_activate_pin_chunk_time));
    metrics.emplace_back(
        StatMetric("finalize_activate_pin_chunk_n_chunks",
                   this->metrics.finalize_activate_pin_chunk_n_chunks));

    // TODO: Export Shuffle metrics for sort.

    // Dict Builders Stats for build
    // NOTE: When window functions can output string arrays this will need
    // to be updated.
    DictBuilderMetrics dict_builder_metrics;
    MetricBase::StatValue n_dict_builders = 0;
    for (const auto& dict_builder : this->sorter.GetDictBuilders()) {
        if (dict_builder != nullptr) {
            dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
            n_dict_builders++;
        }
    }
    metrics.emplace_back(StatMetric("n_dict_builders", n_dict_builders, true));
    dict_builder_metrics.add_to_metrics(metrics, "dict_builders_");
    // Output buffer append time and total size.
    metrics.emplace_back(TimerMetric("output_append_time",
                                     this->output_state->buffer.append_time));
    MetricBase::StatValue output_total_size =
        this->output_state->buffer.total_size;
    metrics.emplace_back(StatMetric("output_total_nrows", output_total_size));
    MetricBase::StatValue output_n_chunks =
        this->output_state->buffer.chunks.size();
    metrics.emplace_back(StatMetric("output_n_chunks", output_n_chunks));

    if (this->op_id != -1) {
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }
}

void WindowState::ReportOutputMetrics() {
    std::vector<MetricBase> metrics;
    metrics.reserve(32);

    this->output_state->ExportMetrics(metrics);

    if (this->op_id != -1) {
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                       this->curr_stage_id),
            std::move(metrics));
    }
}

/**
 * Takes in a collection of table infos, a collection of column indices, and
 * returns the common array type if there is one, or UNKNOWN otherwise.
 */
bodo_array_type::arr_type_enum get_common_arr_type(
    std::vector<std::shared_ptr<table_info>> chunks,
    std::vector<int32_t> col_indices) {
    if (col_indices.empty()) {
        return bodo_array_type::UNKNOWN;
    }
    bodo_array_type::arr_type_enum first_arr_type =
        chunks[0]->columns[col_indices[0]]->arr_type;
    for (int32_t col_idx : col_indices) {
        if (chunks[0]->columns[col_idx]->arr_type != first_arr_type) {
            return bodo_array_type::UNKNOWN;
        }
    }
    return first_arr_type;
}

void WindowState::FinalizeBuild() {
    time_pt start_finalize = start_timer();
    // We first sort the entire table and then compute any functions.
    size_t num_order_by_keys = this->order_by_asc.size();
    size_t num_keys = this->n_keys + num_order_by_keys;
    std::vector<int64_t> asc(num_keys);
    std::vector<int64_t> na_pos(num_keys);

    // Set arbitrary values for sort properties for partition by
    // keys.
    for (size_t i = 0; i < this->n_keys; i++) {
        asc[i] = 0;
        na_pos[i] = 0;
    }
    // Use the sort properties for order by keys.
    for (size_t i = 0; i < num_order_by_keys; i++) {
        size_t dest_index = i + this->n_keys;
        asc[dest_index] = this->order_by_asc[i];
        na_pos[dest_index] = this->order_by_na[i];
    }

    // TODO: Separate sort from compute.
    ScopedTimer window_timer(this->metrics.finalize_window_compute_time);
    std::vector<std::shared_ptr<table_info>> sorted_table_chunks =
        sorter.Finalize();

    std::vector<int32_t> partition_indices(this->n_keys);
    for (size_t col_idx = 0; col_idx < this->n_keys; col_idx++) {
        partition_indices.push_back(col_idx);
    }

    std::vector<int32_t> order_indices(num_order_by_keys);
    for (size_t col_idx = this->n_keys;
         col_idx < this->n_keys + num_order_by_keys; col_idx++) {
        order_indices.push_back(col_idx);
    }
    size_t n_funcs = window_ftypes.size();
    std::vector<std::vector<int32_t>> input_indices(n_funcs);
    for (size_t func_idx = 0; func_idx < n_funcs; func_idx++) {
        auto start = static_cast<size_t>(func_input_offsets[func_idx]);
        auto stop = static_cast<size_t>(func_input_offsets[func_idx + 1]);
        std::vector<int32_t>& indices = input_indices[func_idx];
        for (size_t offset = start; offset < stop; offset++) {
            indices.push_back(func_input_indices[offset]);
        }
    }

    std::vector<int32_t> keep_indices;
    for (size_t i = 0; i < sorted_table_chunks[0]->ncols(); i++) {
        if (cols_to_keep_bitmask[i]) {
            keep_indices.push_back(i);
        }
    }

    if (partition_indices.empty() && order_indices.empty()) {
        // If there is no partition/order column, re-route to the global
        // computation.
        size_t n_chunks = sorted_table_chunks.size();
        std::vector<std::shared_ptr<array_info>> out_arrs;
        for (size_t fn_idx = 0; fn_idx < this->window_ftypes.size(); fn_idx++) {
            AllocWindowOutputColumn(fn_idx, 1, out_arrs);
        }
        global_window_computation(sorted_table_chunks, this->window_ftypes,
                                  func_input_indices, func_input_offsets,
                                  out_arrs, this->parallel);

        // Append the table chunks to the output buffer.
        for (size_t chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
            size_t chunk_size = sorted_table_chunks[chunk_idx]->nrows();
            std::vector<std::shared_ptr<array_info>> cols_to_keep;
            // Append the pass-through columns
            for (size_t col_idx : keep_indices) {
                cols_to_keep.push_back(
                    sorted_table_chunks[chunk_idx]->columns[col_idx]);
            }
            // Append the window function outputs replicated to match the input
            // row count
            for (const std::shared_ptr<array_info>& out_col : out_arrs) {
                std::vector<int64_t> idxs(chunk_size, 0);
                cols_to_keep.push_back(
                    RetrieveArray_SingleColumn(out_col, idxs));
            }
            std::shared_ptr<table_info> data_table_w_cols_to_keep =
                std::make_shared<table_info>(cols_to_keep);
            data_table_w_cols_to_keep->pin();

            // Unify the dictionaries. This should be append only because the
            // output state dict builders should be empty right now.
            std::shared_ptr<table_info> dict_unified_table =
                this->UnifyDictionaryArrays(
                    std::move(data_table_w_cols_to_keep),
                    this->output_state->dict_builders);
            this->output_state->buffer.AppendBatch(dict_unified_table);
        }
    } else {
        // Otherwise, use the sort-based implementation with window calculators.
        bodo_array_type::arr_type_enum partition_arr_type =
            get_common_arr_type(sorted_table_chunks, partition_indices);
        bodo_array_type::arr_type_enum order_arr_type =
            get_common_arr_type(sorted_table_chunks, order_indices);
        std::vector<std::shared_ptr<table_info>> out_chunks;
        compute_window_functions_via_calculators(
            build_table_schema, sorted_table_chunks, partition_indices,
            order_indices, keep_indices, input_indices, window_ftypes,
            this->sorter.GetDictBuilders(), partition_arr_type, order_arr_type,
            out_chunks, this->parallel);
        for (auto& out_chunk : out_chunks) {
            out_chunk->pin();
            // Unify the dictionaries. This should be append only because the
            // output state dict builders should be empty right now.
            std::shared_ptr<table_info> dict_unified_table =
                this->UnifyDictionaryArrays(std::move(out_chunk),
                                            this->output_state->dict_builders);
            this->output_state->buffer.AppendBatch(dict_unified_table);
        }
    }
    window_timer.finalize();
    this->output_state->Finalize();
    this->build_input_finalized = true;
    this->metrics.finalize_time += end_timer(start_finalize);

    if (this->debug_window) {
        std::cerr << "[DEBUG] WindowState::FinalizeBuild: Finished"
                  << std::endl;
    }
}

void WindowState::AppendBatch(std::shared_ptr<table_info>& in_table,
                              bool is_last) {
    auto table = UnifyDictionaryArrays(in_table, sorter.GetDictBuilders());
    sorter.AppendBatch(table, is_last);
}

/**
 * @brief Initialize a new streaming window state for specified array types
 * and number of keys (called from Python)
 *
 * @param build_arr_c_types array types of build table columns (Bodo_CTypes
 * ints)
 * @param build_arr_array_types array types of build table columns
 * (bodo_array_type ints)
 * @param n_build_arrs number of build table columns
 * @param window_ftypes window function types (Bodo_FTypes ints)
 * @param n_keys number of partition by keys
 * @param order_by_asc Boolean bitmask specifying sort-direction for
 *  order-by columns. It should be 'n_order_by_keys' elements long.
 * @param order_by_na Boolean bitmask specifying whether nulls should be
 *  considered 'last' for the order-by columns. It should be
 * 'n_order_by_keys' elements long.
 * @param n_order_by_keys Number of order-by columns.
 * @param cols_to_keep Bitmask specifying the partition, order-by, and input
 * columns to keep. It must have the same number of elements as the total number
 * of columns.
 * @param n_input_cols The intended length of input_cols_to_keep
 * @param output_batch_size Batch size for reading output.
 * @param parallel Is the output parallel?
 * @param sync_iter How frequently should we sync to check if all input has
 * been accumulated
 * @return WindowState* window state to return to Python
 */
WindowState* window_state_init_py_entry(
    int64_t operator_id, int8_t* build_arr_c_types,
    int8_t* build_arr_array_types, int n_build_arrs, int32_t* window_ftypes,
    int32_t n_funcs, uint64_t n_keys, bool* order_by_asc, bool* order_by_na,
    uint64_t n_order_by_keys, bool* cols_to_keep, int32_t n_input_cols,
    int32_t* func_input_indices, int32_t* func_input_offsets,
    int64_t output_batch_size, bool parallel, int64_t sync_iter,
    bool allow_work_stealing) {
    // TODO: Consider allowing op pool size bytes to be set
    int64_t op_pool_size_bytes =
        OperatorComptroller::Default()->GetOperatorBudget(operator_id);
    // Create vectors for the MRNF arguments from the raw pointer arrays.
    std::vector<bool> order_by_asc_vec(order_by_asc,
                                       order_by_asc + n_order_by_keys);
    std::vector<bool> order_by_na_vec(order_by_na,
                                      order_by_na + n_order_by_keys);
    size_t n_cols = n_keys + n_order_by_keys + n_input_cols;
    std::vector<bool> cols_to_keep_vec(cols_to_keep, cols_to_keep + n_cols);
    std::vector<int32_t> func_input_offsets_vec(
        func_input_offsets, func_input_offsets + n_funcs + 1);
    int32_t n_inputs = func_input_offsets_vec[n_funcs];
    std::vector<int32_t> func_input_indices_vec(func_input_indices,
                                                func_input_indices + n_inputs);
    return new WindowState(
        bodo::Schema::Deserialize(
            std::vector<int8_t>(build_arr_array_types,
                                build_arr_array_types + n_build_arrs),
            std::vector<int8_t>(build_arr_c_types,
                                build_arr_c_types + n_build_arrs)),

        std::vector<int32_t>(window_ftypes, window_ftypes + n_funcs), n_keys,
        order_by_asc_vec, order_by_na_vec, cols_to_keep_vec,
        func_input_indices_vec, func_input_offsets_vec, output_batch_size,
        parallel, sync_iter, operator_id, op_pool_size_bytes,
        allow_work_stealing);
}

/**
 * @brief consume build table batch in streaming window by just
 * accumulating rows until all data has been received and then
 * performing a sort + computing function(s) in the finalize step.
 *
 * @param window_state window state pointer
 * @param in_table build table batch
 * @param is_last is last batch locally
 * @return updated is_last
 */
bool window_build_consume_batch(WindowState* window_state,
                                std::shared_ptr<table_info> in_table,
                                bool is_last) {
    // We require that all dictionary keys/values are unified before update
    window_state->AppendBatch(in_table, is_last);
    // Compute output when all input batches are accumulated.
    // Note: We don't need to be synchronized because this is a pipeline
    // breaking step without any "shuffle" that depends on the iteration count.
    // If we can this approach to be "incremental", this will need to change.
    if (is_last) {
        window_state->FinalizeBuild();
    }

    window_state->build_iter++;
    return is_last;
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param window_state window state pointer
 * @param in_table build table batch
 * @param is_last is last batch (in this pipeline) locally
 * @return updated global is_last with possibility of false negatives due to
 * iterations between syncs
 */
bool window_build_consume_batch_py_entry(WindowState* window_state,
                                         table_info* in_table, bool is_last) {
    try {
        std::unique_ptr<table_info> input_table(in_table);
        window_state->metrics.build_input_row_count += input_table->nrows();
        is_last = window_build_consume_batch(window_state,
                                             std::move(input_table), is_last);

        if (is_last) {
            // Report metrics
            window_state->ReportBuildMetrics();
            window_state->curr_stage_id++;
            // The build_table_dict_builders are no longer used after
            // finalize because the dict builders in the output state
            // are used instead.
            assert(window_state->build_input_finalized);
            window_state->sorter.GetDictBuilders().clear();
        }
        return is_last;

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return false;
}

/**
 * @brief return output of window computation
 *
 * @param window_state window state pointer
 * @param produce_output flag to indicate if output should be produced
 * @return std::tuple<std::shared_ptr<table_info>, bool> output data batch
 * and flag for last batch
 */
std::tuple<std::shared_ptr<table_info>, bool> window_produce_output_batch(
    WindowState* window_state, bool produce_output) {
    auto [batch, is_last] =
        window_state->output_state->PopBatch(produce_output);
    window_state->output_state->iter++;
    return std::make_tuple(batch, is_last);
}

/**
 * @brief Python wrapper to produce output table
 * batch
 *
 * @param window_state groupby state pointer
 * @param[out] out_is_last is last batch
 * @param produce_output whether to produce output
 * @return table_info* output table batch
 */
table_info* window_produce_output_batch_py_entry(WindowState* window_state,
                                                 bool* out_is_last,
                                                 bool produce_output) {
    try {
        bool is_last;
        std::shared_ptr<table_info> out;
        std::tie(out, is_last) =
            window_produce_output_batch(window_state, produce_output);
        *out_is_last = is_last;
        window_state->metrics.output_row_count += out->nrows();
        if (is_last) {
            if (window_state->op_id != -1) {
                QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                    QueryProfileCollector::MakeOperatorStageID(
                        window_state->op_id, window_state->curr_stage_id),
                    window_state->metrics.output_row_count);
            }
            window_state->ReportOutputMetrics();
        }
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief delete window state (called from Python after output loop is
 * finished)
 *
 * @param window_state window state pointer to delete
 */
void delete_window_state(WindowState* window_state) { delete window_state; }

#ifdef IS_TESTING
PyMODINIT_FUNC PyInit_test_cpp(void);
#endif

PyMODINIT_FUNC PyInit_stream_window_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_window_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, window_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, window_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, window_produce_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_window_state);

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    return m;
}
