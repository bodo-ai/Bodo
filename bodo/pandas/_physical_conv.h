#pragma once

#include <memory>
#include <vector>

#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"

#include "_pipeline.h"

class PhysicalPlanBuilder {
   public:
    // TODO: Make private properties later
    std::vector<std::shared_ptr<Pipeline>> finished_pipelines;
    std::shared_ptr<PipelineBuilder> active_pipeline;

    PhysicalPlanBuilder() : active_pipeline(nullptr) {}

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
        } else {
            throw std::runtime_error(
                "PhysicalPlanBuilder::Visit unsupported logical operator "
                "type " +
                std::to_string(static_cast<int>(op.type)));
        }
    }
};
