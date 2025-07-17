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
    explicit PhysicalFilter(duckdb::LogicalFilter& logical_filter,
                            std::shared_ptr<PhysicalExpression> expr,
                            std::shared_ptr<bodo::Schema> input_schema)
        : expression(expr) {
        this->output_schema = std::make_shared<bodo::Schema>();
        if (logical_filter.projection_map.empty()) {
            for (size_t i = 0; i < input_schema->ncols(); i++) {
                this->kept_cols.push_back(i);
            }
        } else {
            for (const auto& c : logical_filter.projection_map) {
                this->kept_cols.push_back(c);
            }
        }
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[this->kept_cols[i]]->copy();
            this->output_schema->append_column(std::move(col_type));
            this->output_schema->column_names.push_back(
                input_schema->column_names[this->kept_cols[i]]);
        }
        if (this->kept_cols.size() !=
            logical_filter.GetColumnBindings().size()) {
            throw std::runtime_error(
                "Filter output schema has different number of columns than "
                "LogicalFilter");
        }
    }

    virtual ~PhysicalFilter() = default;

    void Finalize() override {}

    /**
     * @brief The logic for filtering input batches.
     *
     * @param input_batch - the input data to be filtered
     * @return the filtered batch from applying the expression to the input.
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
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
                "Filter expression tree did not result in a boolean array " +
                std::to_string(static_cast<int>(bitmask->dtype)));
        }

        // Apply the bitmask to the input_batch to do row filtering.
        std::shared_ptr<table_info> filtered_table =
            RetrieveTable(input_batch, bitmask);

        std::vector<std::shared_ptr<array_info>> out_cols;
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            out_cols.emplace_back(filtered_table->columns[this->kept_cols[i]]);
        }
        std::shared_ptr<table_info> out_table = std::make_shared<table_info>(
            out_cols, filtered_table->nrows(), output_schema->column_names,
            input_batch->metadata);

        // Just propagate the FINISHED flag to other operators (like join) or
        // accept more input
        return {out_table, prev_op_result == OperatorResult::FINISHED
                               ? OperatorResult::FINISHED
                               : OperatorResult::NEED_MORE_INPUT};
    }

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        // Filter's output schema is the same as the input schema
        return output_schema;
    }

   private:
    std::shared_ptr<PhysicalExpression> expression;
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<uint64_t> kept_cols;
};
