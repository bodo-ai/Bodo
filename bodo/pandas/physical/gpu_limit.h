#pragma once

#include <memory>
#include <utility>
#include "../../libs/streaming/_join.h"
#include "../libs/_array_utils.h"
#include "../libs/_distributed.h"
#include "../libs/_table_builder.h"
#include "../libs/_table_builder_utils.h"
#include "operator.h"

inline bool gpu_capable(duckdb::LogicalLimit& logical_limit) { return true; }

/**
 * @brief Physical node for limit.
 *
 */
class PhysicalGPULimit : public PhysicalGPUSource, public PhysicalGPUSink {
   public:
    explicit PhysicalGPULimit(uint64_t nrows,
                              std::shared_ptr<bodo::Schema> input_schema)
        : n(nrows),
          local_remaining(nrows),
          collected_rows(GPU_DATA(empty_table_from_arrow_schema(
                                      input_schema->ToArrowSchema()),
                                  input_schema->ToArrowSchema(), nullptr),
                         get_gpu_streaming_batch_size()),
          output_schema(input_schema) {
        arrow_output_schema = output_schema->ToArrowSchema();
    }

    virtual ~PhysicalGPULimit() = default;

    /**
     * @brief Finalize - this function is a safe place to see how many rows each
     * rank collected and to determine how many from each rank is needed to meet
     * the overall goal of "n".  The collected_rows are reduced to
     * meet that goal.
     *
     */
    void FinalizeSink() override {
        GpuMpiManager gpu_mpi;

        if (!is_gpu_rank()) {
            return;
        }

        int n_pes = gpu_mpi.get_num_ranks();  // total gpus
        int myrank = gpu_mpi.get_rank();

        // Gather how many rows are on each rank
        std::vector<uint64_t> row_counts(n_pes);

        // Get total rows on this GPU.
        uint64_t cur_rows = collected_rows.collected_rows;
        CHECK_MPI(MPI_Allgather(&cur_rows, 1, MPI_UNSIGNED_LONG_LONG,
                                row_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                                gpu_mpi.get_mpi_comm()),
                  "PhysicalGPULimit: MPI error on MPI_Allgather:");

        local_remaining = 0;

        for (int32_t i = 0; i < n_pes && n > 0; ++i) {
            uint64_t num_from_rank = std::min(n, row_counts[i]);
            if (i == myrank) {
                local_remaining = num_from_rank;
            }
            n -= num_from_rank;
        }
    }

    void FinalizeSource() override {}

    /**
     * @brief Do limit.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        if (!is_gpu_rank()) {
            return OperatorResult::FINISHED;
        }
        // Every rank will collect n rows.  We remove extras in Finalize.
        uint64_t select_local =
            std::min(local_remaining,
                     static_cast<uint64_t>(input_batch.table->num_rows()));
        if (select_local > 0) {
            std::unique_ptr<cudf::table> first_select_local_rows =
                std::make_unique<cudf::table>(
                    cudf::slice(input_batch.table->view(),
                                {0, (int)select_local}, se->stream)[0]);
            collected_rows.append_batch(GPU_DATA(
                std::move(first_select_local_rows), arrow_output_schema, se));
            local_remaining -= select_local;
        }
        return (local_remaining == 0 ||
                prev_op_result == OperatorResult::FINISHED)
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // Limit should be between pipelines and act alternatively as a sink
        // then source but there should never be the need to ask for the result
        // all in one go.
        throw std::runtime_error("GetResult called on a limit node.");
    }

    /**
     * @brief ProduceBatch - act as a data source
     *
     * returns std::pair<GPU_DATA, OperatorResult>
     */
    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, arrow_output_schema, se),
                    OperatorResult::FINISHED};
        }

        auto next_batch = collected_rows.next(se, true);
        if (next_batch.table->num_rows() >=
            static_cast<cudf::size_type>(local_remaining)) {
            cudf::table_view tv = next_batch.table->view();
            std::vector<cudf::size_type> offsets = {
                0, static_cast<cudf::size_type>(local_remaining)};
            auto slices = cudf::slice(tv, offsets);
            next_batch.table = std::make_shared<cudf::table>(slices[0]);

            local_remaining = 0;
        } else {
            local_remaining -= next_batch.table->num_rows();
        }
        return {next_batch, local_remaining == 0
                                ? OperatorResult::FINISHED
                                : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    uint64_t n, local_remaining;
    GPUBatchGenerator collected_rows;
    const std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> arrow_output_schema;
};
