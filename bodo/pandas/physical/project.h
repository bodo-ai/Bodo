#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalSourceSink {
   public:
    explicit PhysicalProjection(std::vector<int64_t> &cols)
        : selected_columns(cols) {}

    virtual ~PhysicalProjection() = default;

    void Finalize() override {}

    /**
     * @brief Do projection.
     *
     * @return std::pair<int64_t, OperatorResult> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), state of operator after processing the
     * batch
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        // Select columns from the actual data in Bodo table_info format.
        std::shared_ptr<table_info> out_table_info =
            ProjectTable(input_batch, selected_columns);
        return {out_table_info, OperatorResult::FINISHED};
    }

   private:
    std::vector<int64_t> selected_columns;
};
