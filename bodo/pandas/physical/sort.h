#pragma once

#include <cstdint>
#include "../../libs/streaming/_sort.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "operator.h"

/**
 * @brief Physical node for join.
 *
 */
class PhysicalSort : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalSort(
        duckdb::LogicalOrder& logical_order,
        std::shared_ptr<bodo::Schema> input_schema) : output_schema(input_schema) {
        std::vector<int64_t> ascending;
        std::vector<int64_t> na_first;
        std::vector<uint64_t> keys;

        // Convert BoundOrderByNode's to keys, asc, and na_first.
        for (const auto& order : logical_order.orders) {
            switch(order.type) {
            case duckdb::OrderType::ASCENDING:
                ascending.push_back(1);
                break;
            case duckdb::OrderType::DESCENDING:
                ascending.push_back(0);
                break;
            default:
                throw std::runtime_error(
                    "PhysicalSort order type not ascending or descending.");
            }
            switch(order.null_order) {
            case duckdb::OrderByNullType::NULLS_FIRST:
                na_first.push_back(1);
                break;
            case duckdb::OrderByNullType::NULLS_LAST:
                na_first.push_back(0);
                break;
            default:
                throw std::runtime_error(
                    "PhysicalSort orderbynull type not first or last.");
            }
            duckdb::ExpressionClass expr_class = order.expression->GetExpressionClass();
            if (expr_class != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "PhysicalSort expression is not column ref.");
            }
            auto& bce = order.expression->Cast<duckdb::BoundColumnRefExpression>();
            // Is this right or need col_ref_map?
            keys.push_back(bce.binding.column_index);
        }

        // Establish table reordering so key are at beginning.
        initInputColumnMapping(col_inds, keys, output_schema->ncols());

        std::shared_ptr<bodo::Schema> build_table_schema_reordered =
            output_schema->Project(col_inds);

        stream_sorter = std::make_unique<StreamSortState>(
            -1, keys.size(), std::move(ascending), std::move(na_first),
            build_table_schema_reordered, /*parallel*/ true);
    }

    virtual ~PhysicalSort() = default;

    void Finalize() override {
        stream_sorter->FinalizeBuild();
    }

    /**
     * @brief process input tables to sort
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, col_inds);

        stream_sorter->ConsumeBatch(input_batch_reordered);

        return local_is_last
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief Act as data source producing sorted rows.
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch() override {
        auto sorted_res = stream_sorter->GetOutput();
        return {sorted_res.first,
                sorted_res.second
                    ? OperatorResult::FINISHED
                    : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() override {
        // Sort build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    /**
     * @brief Get the output schema of join probe
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    static void initOutputColumnMapping(std::vector<uint64_t>& col_inds,
                                        const std::vector<uint64_t>& keys,
                                        uint64_t ncols) {
        // Map key column index to its position in keys vector
        std::unordered_map<uint64_t, size_t> key_positions;
        for (size_t i = 0; i < keys.size(); ++i) {
            key_positions[keys[i]] = i;
        }
        uint64_t data_offset = keys.size();

        for (uint64_t i = 0; i < ncols; i++) {
            if (key_positions.find(i) != key_positions.end()) {
                col_inds.push_back(key_positions[i]);
            } else {
                col_inds.push_back(data_offset++);
            }
        }
    }


    std::vector<int64_t> col_inds;
    std::shared_ptr<bodo::Schema> output_schema;
    std::unique_ptr<StreamSortState> stream_sorter;
};
