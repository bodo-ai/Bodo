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
    PhysicalProjection(std::shared_ptr<PhysicalOperator> src,
                       std::vector<int64_t> &cols)
        : src(src), selected_columns(cols) {}

    void Finalize() override {}

    /**
     * @brief Do projection.
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        // Select columns from the actual data in Bodo table_info format.
        std::shared_ptr<table_info> out_table_info =
            ProjectTable(input_batch, selected_columns);
        return {out_table_info, OperatorResult::FINISHED};
    }

   private:
    std::shared_ptr<PhysicalOperator> src;
    std::vector<int64_t> selected_columns;
};
