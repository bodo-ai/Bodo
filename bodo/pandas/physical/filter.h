
#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "expression.h"
#include "operator.h"

/**
 * @brief Physical node for filter.
 *
 */
class PhysicalFilter : public PhysicalSourceSink {
   public:
    explicit PhysicalFilter(std::shared_ptr<PhysicalExpression> expr)
        : expression(expr) {}

    virtual ~PhysicalFilter() = default;

    void Finalize() override {}

    /**
     * @brief Do filter
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        std::shared_ptr<ExprResult> expr_output =
            expression->ProcessBatch(input_batch);
        std::shared_ptr<ArrayExprResult> arr_output =
            std::dynamic_pointer_cast<ArrayExprResult>(expr_output);
        if (!arr_output) {
            throw std::runtime_error(
                "Filter expression tree did not result in an array");
        }
        auto bitmask = arr_output->result;
        if (bitmask->dtype != Bodo_CTypes::_BOOL) {
            throw std::runtime_error(
                "Filter expression tree did not result in a boolean array");
        }

        auto out_table = RetrieveTable(input_batch, bitmask);

        return {out_table, OperatorResult::FINISHED};
    }

   private:
    std::shared_ptr<PhysicalExpression> expression;
};
