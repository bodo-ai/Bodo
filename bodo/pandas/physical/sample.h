
#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalSample : public PhysicalSourceSink {
   public:
    explicit PhysicalSample(float percent) : percentage(percent) {}

    virtual ~PhysicalSample() = default;

    void Finalize() override {}

    /**
     * @brief Do limit.
     *
     * @return std::pair<int64_t, OperatorResult> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), state of operator after processing the
     * batch
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        uint64_t select_this_time = stochasticRound(input_batch->nrows());

        // Perhaps we should randomly select rather than just
        // take the first select_this_time rows.
        std::vector<int64_t> rowInds(select_this_time);
        for (uint64_t i = 0; i < select_this_time; ++i) {
            rowInds[i] = i;
        }
        std::shared_ptr<table_info> out_table_info =
            RetrieveTable(input_batch, rowInds);
        return {out_table_info, OperatorResult::NEED_MORE_INPUT};
    }

   private:
    float percentage;

    uint64_t stochasticRound(uint64_t nrows) {
        double scaled = nrows * percentage;
        int base = static_cast<int>(scaled);
        double remainder = scaled - base;

        // Decide whether to add 1 based on remainder probability
        return base + (static_cast<double>(rand()) / RAND_MAX < remainder ? 1 : 0);
    }
};
