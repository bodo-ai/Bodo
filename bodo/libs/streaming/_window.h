#pragma once
#include "../_operator_pool.h"
#include "_groupby.h"
#include "_sort.h"

class WindowStateSorter {
    std::shared_ptr<bodo::Schema> build_table_schema;
    size_t num_keys;
    // True when num_keys is 0 in which case we just collect the tables instead
    // of doing any sorting.
    bool skip_sorting;
    // Dictionary builders used for the build_table_buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;
    // Input state
    // TODO: Enable some type of partitioning. Ideally this can be
    // done by switching to a CTB and supporting that inside our sort.
    std::unique_ptr<TableBuildBuffer> build_table_buffer;

    std::vector<int64_t> asc;
    std::vector<int64_t> na_pos;

    std::unique_ptr<StreamSortState> stream_sorter;

   public:
    WindowStateSorter(std::shared_ptr<bodo::Schema>& build_table_schema,
                      size_t n_window_keys,
                      const std::vector<bool>& order_by_asc,
                      const std::vector<bool>& order_by_na, bool parallel);

    void AppendBatch(std::shared_ptr<table_info>& table, bool is_last);

    std::vector<std::shared_ptr<table_info>> Finalize();

    size_t NumSortKeys() { return num_keys; }

    std::vector<std::shared_ptr<DictionaryBuilder>>& GetDictBuilders() {
        return build_table_dict_builders;
    }
};

class WindowState {
   private:
    // NOTE: These need to be declared first so that they are
    // removed at the very end during destruction.

    /// @brief OperatorBufferPool for this operator.
    const std::unique_ptr<bodo::OperatorBufferPool> op_pool;
    /// @brief Memory manager for op_pool. This is used during buffer
    /// allocations.
    const std::shared_ptr<::arrow::MemoryManager> op_mm;

    /// @brief OperatorScratchPool corresponding to the op_pool.
    const std::unique_ptr<bodo::OperatorScratchPool> op_scratch_pool;
    /// @brief Memory manager for op_scratch_pool.
    const std::shared_ptr<::arrow::MemoryManager> op_scratch_mm;

   public:
    // Current stage ID. 0 is for Initialization, stage 1 is for
    // accumulating the input + finalize, and stage 2 is for
    // producing the output.
    uint32_t curr_stage_id = 0;
    const uint64_t n_keys;
    bool parallel;
    const int64_t output_batch_size;
    // Integer enum of window functions.
    const std::vector<int32_t> window_ftypes;
    const std::vector<bool> order_by_asc;
    const std::vector<bool> order_by_na;
    const std::vector<bool> cols_to_keep_bitmask;
    const std::vector<int32_t> func_input_indices;
    const std::vector<int32_t> func_input_offsets;

    // The number of iterations between syncs (used to check
    // when all input has been received).
    int64_t sync_iter;

    // Current iteration of build steps
    uint64_t build_iter = 0;

    /// The schema of the inputs
    std::shared_ptr<bodo::Schema> build_table_schema;
    WindowStateSorter sorter;

    // Output state
    std::shared_ptr<GroupbyOutputState> output_state = nullptr;
    // By default, enable work-stealing for window.
    // This can be overriden by explicitly setting
    // BODO_STREAM_DISABLE_ENABLE_OUTPUT_WORK_STEALING.
    bool enable_output_work_stealing = true;

    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the final steps.
    bool build_input_finalized = false;

    // Whether we should print debug information to verify sort path is taken
    bool debug_window = false;

    // TODO: Replace with window metrics.
    GroupbyMetrics metrics;
    const int64_t op_id;

    WindowState(const std::unique_ptr<bodo::Schema>& in_schema_,
                std::vector<int32_t> window_ftypes_, uint64_t n_keys_,
                std::vector<bool> order_by_asc_, std::vector<bool> order_by_na_,
                std::vector<bool> cols_to_keep_,
                std::vector<int32_t> func_input_indices_,
                std::vector<int32_t> func_input_offsets_,
                int64_t output_batch_size_, bool parallel_, int64_t sync_iter_,
                int64_t op_id_, int64_t op_pool_size_bytes_,
                bool allow_work_stealing_);

    /**
     * @brief infers the output data type of one of the window function calls
     * and appends to a schema.
     *
     * @param[in] func_idx: the index of the window function call.
     * @param[in] out_schema: the schema to append the result to.
     */
    void InferWindowOutputDataType(int32_t func_idx,
                                   std::unique_ptr<bodo::Schema>& out_schema);

    /**
     * @brief infers the output data type of one of the window function calls
     * and allocates a corresponding array to store the output.
     *
     * @param[in] func_idx: the index of the window function call.
     * @param[in] out_rows: the number of rows to allocate.
     * @param[in] out_arrs: the vector of columns to append to.
     */
    void AllocWindowOutputColumn(
        int32_t func_idx, size_t output_rows,
        std::vector<std::shared_ptr<array_info>>& out_arrs,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    /**
     * @brief Unify dictionaries of input table with the given dictionary
     * builder by appending its new dictionary values to buffer's dictionaries
     * and transposing input's indices.
     *
     * @param in_table input table
     * @param dict_builders The dictionary builders to update.
     *
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with build table dictionaries.
     */
    static std::shared_ptr<table_info> UnifyDictionaryArrays(
        const std::shared_ptr<table_info>& in_table,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders);

    /**
     * @brief Report the current set of build stage metrics.
     *
     */
    void ReportBuildMetrics();

    /**
     * @brief Report the metrics for the output production stage. This primarily
     * consists of metrics related to any work redistribution that might've been
     * performed during execution.
     *
     */
    void ReportOutputMetrics();

    /**
     * @brief Finalize the build step. This will finalize all data,
     * append their outputs to the output buffer, clear the build state and set
     * build_input_finalized to prevent future repetitions of the build step.
     *
     */
    void FinalizeBuild();

    void AppendBatch(std::shared_ptr<table_info>& in_table, bool is_last);
};
