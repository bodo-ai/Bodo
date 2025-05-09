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
class PhysicalLimit : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalLimit(uint64_t nrows) : n(nrows), local_remaining(nrows), produced(0) {}

    virtual ~PhysicalLimit() = default;

    /**
     * @brief Finalize - this function is a safe place to see how many rows each rank
     *        collected and to determine how many from each rank is needed to meet the
     *        overall goal of "n".  The collected_rows table_infos are reduced to meet
     *        that goal.
     *
     */
    void Finalize() override {
        int n_pes = 0;
        int myrank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);  // total ranks
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // my rank

        // Gather how many rows are on each rank
        std::vector<uint64_t> row_counts(n_pes);

        // Sum up total rows this rank has collected.
        uint64_t cur_rows = std::accumulate(
            collected_rows.begin(),
            collected_rows.end(),
            0,
            [](uint64_t acc, const std::shared_ptr<table_info>& item) {
                return acc + item->nrows();
            });
        CHECK_MPI(MPI_Allgather(&cur_rows, 1, MPI_UNSIGNED_LONG_LONG,
                                row_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                                MPI_COMM_WORLD),
                  "PhysicalLimit: MPI error on MPI_Allgather:");

        uint64_t select_local = 0;

        for (int64_t i = 0; i < n_pes && n > 0; ++i) {
            uint64_t num_from_rank = std::min(n, row_counts[i]);
            if (i == myrank) {
                select_local = num_from_rank;
            }
            n -= num_from_rank;
        }

        // Go back through the collected rows and remove ones we don't need.
        for (size_t i = 0; i < collected_rows.size(); ++i) {
            collected_rows[i] = get_n_rows(collected_rows[i], std::min(select_local, collected_rows[i]->nrows()));
            select_local -= collected_rows[i]->nrows();
            if (select_local == 0) {
                // Remove any subsequent sets of rows as they are unneeded.
                collected_rows.erase(collected_rows.begin() + i + 1, collected_rows.end());
                break;
            }
        }
    }

    /**
     * @brief get_n_rows - utility function to get a fixed number of rows from a table_info
     *
     * param input_batch - the table to return a subset of
     * param num_rows - the number of rows of the table to return
     *
     * returns std::shared_ptr<table_info> - the table restriced to num_rows rows.
     */
    std::shared_ptr<table_info> get_n_rows(std::shared_ptr<table_info> input_batch, uint64_t num_rows) {
        std::vector<int64_t> rowInds(num_rows);
        for (uint64_t i = 0; i < num_rows; ++i) {
            rowInds[i] = i;
        }
        return RetrieveTable(input_batch, rowInds);
    }

    /**
     * @brief Do limit.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    OperatorResult ConsumeBatch(
        std::shared_ptr<table_info> input_batch) override {
        // Every rank will collect n rows.  We remove extras in Finalize.
        uint64_t select_local = std::min(local_remaining, input_batch->nrows());
        collected_rows.emplace_back(get_n_rows(input_batch, select_local));
        local_remaining -= select_local;
        return local_remaining == 0 ? OperatorResult::FINISHED : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() {
        // Limit should be between pipelines and act alternatively as a sink then source
        // but there should never be the need to ask for the result all in one go.
        throw std::runtime_error(
            "GetResult called on a limit node.");
    }

    /**
     * @brief ProduceBatch - act as a data source
     *
     * returns std::pair<std::shared_ptr<table_info>, ProducerResult>
     */
    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch() {
        // Return the previously accumulated table_infos.  There should always be at least one.
        produced++;
        return {collected_rows[produced-1], produced >= collected_rows.size() ? ProducerResult::FINISHED : ProducerResult::HAVE_MORE_OUTPUT};
    }

   private:
    uint64_t n, local_remaining, produced;
    std::vector<std::shared_ptr<table_info>> collected_rows;
};
