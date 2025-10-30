#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "_plan.h"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"

#include "_pipeline.h"
#include "physical/cte.h"

class PhysicalPlanBuilder {
   public:
    std::vector<std::shared_ptr<Pipeline>> locked_pipelines;
    // TODO: Make private properties later
    std::vector<std::shared_ptr<Pipeline>> finished_pipelines;
    std::shared_ptr<PipelineBuilder> active_pipeline;
    std::map<duckdb::idx_t, std::shared_ptr<PhysicalCTE>>& ctes;

    // Mapping of join ids to their JoinState pointers for join filter operators
    // (filled during physical plan construction). Using loose pointers since
    // PhysicalJoinFilter only needs to access the JoinState during execution
    std::shared_ptr<std::unordered_map<int, JoinState*>> join_filter_states;

    PhysicalPlanBuilder(
        std::map<duckdb::idx_t, std::shared_ptr<PhysicalCTE>>& _ctes)
        : active_pipeline(nullptr),
          ctes(_ctes),
          join_filter_states(
              std::make_shared<std::unordered_map<int, JoinState*>>()) {}

    /**
     * @brief Move finshed_pipelines into locked category
     * so that nothing can be inserted before them.
     */
    void lock_finished() {
        locked_pipelines.insert(
            locked_pipelines.end(),
            std::make_move_iterator(finished_pipelines.begin()),
            std::make_move_iterator(finished_pipelines.end()));
        finished_pipelines.clear();
    }

    void Visit(duckdb::LogicalGet& op);
    void Visit(duckdb::LogicalProjection& op);
    void Visit(duckdb::LogicalFilter& op);
    void Visit(duckdb::LogicalAggregate& op);
    void Visit(duckdb::LogicalOrder& op);
    void Visit(duckdb::LogicalComparisonJoin& op);
    void Visit(duckdb::LogicalCrossProduct& op);
    void Visit(duckdb::LogicalLimit& op);
    void Visit(duckdb::LogicalTopN& op);
    void Visit(duckdb::LogicalSample& op);
    void Visit(duckdb::LogicalSetOperation& op);
    void Visit(duckdb::LogicalCopyToFile& op);
    void Visit(duckdb::LogicalDistinct& op);
    void Visit(duckdb::LogicalMaterializedCTE& op);
    void Visit(duckdb::LogicalCTERef& op);
    void Visit(bodo::LogicalJoinFilter& op);

    void Visit(duckdb::LogicalOperator& op) {
        if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
            Visit(op.Cast<duckdb::LogicalGet>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
            Visit(op.Cast<duckdb::LogicalProjection>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
            Visit(op.Cast<duckdb::LogicalFilter>());
        } else if (op.type == duckdb::LogicalOperatorType::
                                  LOGICAL_AGGREGATE_AND_GROUP_BY) {
            Visit(op.Cast<duckdb::LogicalAggregate>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_ORDER_BY) {
            Visit(op.Cast<duckdb::LogicalOrder>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
            Visit(op.Cast<duckdb::LogicalComparisonJoin>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
            Visit(op.Cast<duckdb::LogicalCrossProduct>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_LIMIT) {
            Visit(op.Cast<duckdb::LogicalLimit>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_TOP_N) {
            Visit(op.Cast<duckdb::LogicalTopN>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_SAMPLE) {
            Visit(op.Cast<duckdb::LogicalSample>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_UNION) {
            Visit(op.Cast<duckdb::LogicalSetOperation>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE) {
            Visit(op.Cast<duckdb::LogicalCopyToFile>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_DISTINCT) {
            Visit(op.Cast<duckdb::LogicalDistinct>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
            Visit(op.Cast<duckdb::LogicalMaterializedCTE>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_CTE_REF) {
            Visit(op.Cast<duckdb::LogicalCTERef>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
            // TODO: add join filter to DuckDB operator types to allow more
            // extension types
            Visit(op.Cast<bodo::LogicalJoinFilter>());
        } else {
            throw std::runtime_error(
                "PhysicalPlanBuilder::Visit unsupported logical operator "
                "type " +
                std::to_string(static_cast<int>(op.type)));
        }
    }
};
