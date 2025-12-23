#pragma once

#include <memory>
#include <utility>
#include "../../libs/streaming/_join.h"
#include "../libs/_array_utils.h"
#include "../libs/_distributed.h"
#include "../libs/_table_builder.h"
#include "../libs/_table_builder_utils.h"
#include "operator.h"

/**
 * @brief Physical node for limit.
 *
 */
class PhysicalLimit : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalLimit(uint64_t nrows,
                           std::shared_ptr<bodo::Schema> input_schema)
        : n(nrows), local_remaining(nrows), output_schema(input_schema) {}

    virtual ~PhysicalLimit() = default;

    /**
     * @brief Finalize - this function is a safe place to see how many rows each
     * rank collected and to determine how many from each rank is needed to meet
     * the overall goal of "n".  The collected_rows table_infos are reduced to
     * meet that goal.
     *
     */
    void FinalizeSink() override {
        int n_pes = 0;
        int myrank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);   // total ranks
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);  // my rank

        // Gather how many rows are on each rank
        std::vector<uint64_t> row_counts(n_pes);

        // Sum up total rows this rank has collected.
        uint64_t cur_rows = 0;
        for (auto it = collected_rows->builder->begin();
             it != collected_rows->builder->end(); ++it) {
            cur_rows += (*it)->nrows();
        }
        CHECK_MPI(MPI_Allgather(&cur_rows, 1, MPI_UNSIGNED_LONG_LONG,
                                row_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                                MPI_COMM_WORLD),
                  "PhysicalLimit: MPI error on MPI_Allgather:");

        uint64_t local_remaining = 0;

        for (int32_t i = 0; i < n_pes && n > 0; ++i) {
            uint64_t num_from_rank = std::min(n, row_counts[i]);
            if (i == myrank) {
                local_remaining = num_from_rank;
            }
            n -= num_from_rank;
        }

        auto reduced_collected_rows =
            std::make_unique<ChunkedTableBuilderState>(
                collected_rows->table_schema,
                collected_rows->builder->active_chunk_capacity);
        while (!collected_rows->builder->empty()) {
            auto next_batch = collected_rows->builder->PopChunk();
            uint64_t select_local =
                std::min(local_remaining, (uint64_t)std::get<1>(next_batch));
            auto unified_table = unify_dictionary_arrays_helper(
                std::get<0>(next_batch), reduced_collected_rows->dict_builders,
                0);
            reduced_collected_rows->builder->AppendBatch(
                unified_table, get_n_rows(select_local));
            reduced_collected_rows->builder->FinalizeActiveChunk();
            local_remaining -= select_local;
        }

        collected_rows = std::move(reduced_collected_rows);
    }

    void FinalizeSource() override {}

    /**
     * @brief get_n_rows - utility function to get a fixed number of rows from a
     * table_info
     *
     * param input_batch - the table to return a subset of
     * param num_rows - the number of rows of the table to return
     *
     * returns std::shared_ptr<table_info> - the table restriced to num_rows
     * rows.
     */
    std::vector<int64_t> get_n_rows(uint64_t num_rows) {
        std::vector<int64_t> rowInds(num_rows);
        for (uint64_t i = 0; i < num_rows; ++i) {
            rowInds[i] = i;
        }
        return rowInds;
    }

    /**
     * @brief Do limit.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        std::cout << "PhysicalLimit::ConsumeBatch called with "
                  << input_batch->nrows() << " rows." << std::endl;
        if (!collected_rows) {
            collected_rows = std::make_unique<ChunkedTableBuilderState>(
                input_batch->schema(), get_streaming_batch_size());
        }
        // Every rank will collect n rows.  We remove extras in Finalize.
        uint64_t select_local = std::min(local_remaining, input_batch->nrows());
        if (select_local > 0) {
            auto unified_table = unify_dictionary_arrays_helper(
                input_batch, collected_rows->dict_builders, 0);
            collected_rows->builder->AppendBatch(unified_table,
                                                 get_n_rows(select_local));
            collected_rows->builder->FinalizeActiveChunk();
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
     * returns std::pair<std::shared_ptr<table_info>, OperatorResult>
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        auto next_batch = collected_rows->builder->PopChunk(true);
        return {std::get<0>(next_batch),
                collected_rows->builder->empty()
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
    std::unique_ptr<ChunkedTableBuilderState> collected_rows;
    const std::shared_ptr<bodo::Schema> output_schema;
};
