#pragma once
#include "_memory_budget.h"
#include "_operator_pool.h"
#include "_stream_groupby.h"

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
    const std::vector<bool> partition_by_cols_to_keep;
    const std::vector<bool> order_by_cols_to_keep;

    // The number of iterations between syncs (used to check
    // when all input has been received).
    int64_t sync_iter;

    // Current iteration of build steps
    uint64_t build_iter = 0;

    // Dictionary builders used for the build_table_buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Input state
    // TODO: Enable some type of partitioning. Ideally this can be
    // done by switching to a CTB and supporting that inside our sort.
    std::unique_ptr<TableBuildBuffer> build_table_buffer;

    // Output state
    std::shared_ptr<GroupbyOutputState> output_state = nullptr;
    // By default, enable work-stealing for window.
    // This can be overriden by explicitly setting
    // BODO_STREAM_DISABLE_ENABLE_OUTPUT_WORK_STEALING.
    bool enable_output_work_stealing = true;

    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the final steps.
    bool build_input_finalized = false;

    // TODO: Replace with window metrics.
    GroupbyMetrics metrics;
    const int64_t op_id;

    WindowState(const std::unique_ptr<bodo::Schema>& in_schema_,
                std::vector<int32_t> window_ftypes_, uint64_t n_keys_,
                std::vector<bool> order_by_asc_, std::vector<bool> order_by_na_,
                std::vector<bool> partition_by_cols_to_keep_,
                std::vector<bool> order_by_cols_to_keep_,
                int64_t output_batch_size_, bool parallel_, int64_t sync_iter_,
                int64_t op_id_, int64_t op_pool_size_bytes_,
                bool allow_work_stealing_);

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
};
