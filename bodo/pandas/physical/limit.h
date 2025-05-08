#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "../libs/_distributed.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalLimit : public PhysicalSourceSink {
   public:
    explicit PhysicalLimit(uint64_t nrows) : n(nrows) {}

    virtual ~PhysicalLimit() = default;

    void Finalize() override {}

    /**
     * @brief Do limit.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more output.
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        int n_pes = 0;
        int myrank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // Gather how many rows are on each rank
        std::vector<uint64_t> row_counts(n_pes);
        uint64_t cur_rows = input_batch->nrows();
        CHECK_MPI(MPI_Allgather(&cur_rows, 1, MPI_UNSIGNED_LONG_LONG, row_counts.data(), 1,
                                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD),
                  "PhysicalLimit: MPI error on MPI_Allgather:");

        uint64_t select_local = 0;

        for (int64_t i = 0; i < n_pes && n > 0; ++i) {
            uint64_t num_from_rank = std::min(n, row_counts[i]);
            if (i == myrank) {
                select_local = num_from_rank;
            }
            n -= num_from_rank;
        }

        std::vector<int64_t> rowInds(select_local);
        for (uint64_t i = 0; i < select_local; ++i) {
            rowInds[i] = i;
        }
        std::shared_ptr<table_info> out_table_info =
            RetrieveTable(input_batch, rowInds);
        return {out_table_info, n == 0 ? OperatorResult::FINISHED : OperatorResult::NEED_MORE_INPUT};
    }

   private:
    uint64_t n;
};
