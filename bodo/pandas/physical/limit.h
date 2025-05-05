#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalLimit : public PhysicalSourceSink {
   public:
    explicit PhysicalLimit(uint64_t nrows)
        : n(nrows) {}

    virtual ~PhysicalLimit() = default;

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
        uint64_t select_this_time = std::min(n, input_batch->nrows()); 
        n -= select_this_time;

        std::vector<int64_t> rowInds(select_this_time);
        for (uint64_t i = 0; i < select_this_time; ++i) {
            rowInds[i] = i;
        }
        std::shared_ptr<table_info> out_table_info =
            RetrieveTable(input_batch, rowInds);
        return {out_table_info, OperatorResult::FINISHED};
    }

   private:
    uint64_t n;
};
