
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
     * @brief The logic for filtering input batches.
     *
     * @param input_batch - the input data to be filtered
     * @return the filtered batch from applying the expression to the input.
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        // Evaluate the Physical expression tree with the given input batch.
        std::shared_ptr<ExprResult> expr_output =
            expression->ProcessBatch(input_batch);
        // Make sure that the output of the expression tree is a bitmask in
        // the form of a boolean array.
        std::shared_ptr<ArrayExprResult> arr_output =
            std::dynamic_pointer_cast<ArrayExprResult>(expr_output);
        if (!arr_output) {
            throw std::runtime_error(
                "Filter expression tree did not result in an array");
        }
        std::shared_ptr<array_info> bitmask = arr_output->result;
        if (bitmask->dtype != Bodo_CTypes::_BOOL) {
            throw std::runtime_error(
                "Filter expression tree did not result in a boolean array");
        }

        if (!input_batch->column_names[input_batch->column_names.size() - 1]
                 .starts_with("__index_level_")) {
            // Filter will always generate an explicit index.  If an explicit
            // index does not already exist then create it and fill it is
            // sequentially.
            uint64_t col_length = input_batch->columns[0]->length;
            std::shared_ptr<array_info> idx_arr =
                alloc_nullable_array_no_nulls(col_length, Bodo_CTypes::INT64);
            // TODO: This has to be a bug in the streaming case.
            for (size_t i = 0; i < col_length; i++) {
                getv<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(idx_arr, i) =
                    i;
            }
            input_batch->columns.push_back(idx_arr);
            input_batch->column_names.push_back("__index_level_0__");
        }

        // Apply the bitmask to the input_batch to do row filtering.
        std::shared_ptr<table_info> out_table =
            RetrieveTable(input_batch, bitmask);

        return {out_table, OperatorResult::FINISHED};
    }

   private:
    std::shared_ptr<PhysicalExpression> expression;
};
