#pragma once

#include <memory>
#include <vector>

#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"

#include "_pipeline.h"
#include "physical/operator.h"

class PhysicalPlanBuilder {
   private:
    std::vector<std::shared_ptr<Pipeline>> finished_pipelines;
    std::shared_ptr<PipelineBuilder> active_pipeline;

   public:
    PhysicalPlanBuilder() : active_pipeline(nullptr) {}

    void Visit(duckdb::LogicalGet& op);
    void Visit(duckdb::LogicalProjection& op);
    void Visit(duckdb::LogicalFilter& op);
    void Visit(duckdb::LogicalComparisonJoin& op);

    void Visit(duckdb::LogicalOperator& op) {
        if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
            Visit(op.Cast<duckdb::LogicalGet>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
            Visit(op.Cast<duckdb::LogicalProjection>());
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
            Visit(op.Cast<duckdb::LogicalFilter>());
        } else if (op.type ==
                   duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
            Visit(op.Cast<duckdb::LogicalComparisonJoin>());
        } else {
            throw std::runtime_error("Unsupported logical operator type");
        }
    }
};
