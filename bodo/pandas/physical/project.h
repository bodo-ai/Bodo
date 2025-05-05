#pragma once

#include <utility>
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalSourceSink {
   public:
    explicit PhysicalProjection(
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
        : exprs(std::move(exprs)) {}

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
        std::vector<std::shared_ptr<array_info>> out_cols;
        std::vector<std::string> col_names;

        for (const auto& expr : this->exprs) {
            if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
                size_t col_idx = colref.binding.column_index;
                out_cols.emplace_back(input_batch->columns[col_idx]);
                if (input_batch->column_names.size() > 0) {
                    col_names.emplace_back(input_batch->column_names[col_idx]);
                } else {
                    col_names.emplace_back(colref.GetName());
                }
            } else {
                // TODO: Python scalar function
            }
        }

        uint64_t out_size =
            out_cols.size() > 0 ? out_cols[0]->length : input_batch->nrows();
        if (out_size != input_batch->nrows()) {
            throw std::runtime_error(
                "Output size does not match input size in Projection");
        }

        std::shared_ptr<table_info> out_table_info =
            std::make_shared<table_info>(out_cols, out_size, col_names,
                                         input_batch->metadata);
        return {out_table_info, OperatorResult::FINISHED};
    }

   private:
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
};
