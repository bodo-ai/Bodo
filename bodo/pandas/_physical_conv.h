#pragma once

#include <memory>
#include <vector>

#include "_pipeline.h"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "physical/operator.h"

class PhysicalPlanBuilder {
   private:
    std::vector<std::shared_ptr<Pipeline>> finished_pipelines;
    std::shared_ptr<PipelineBuilder> active_pipelines;

   public:
    PhysicalPlanBuilder() : active_pipelines(nullptr) {}

    void Visit(duckdb::LogicalGet& op);
    void Visit(duckdb::LogicalProjection& op);
    void Visit(duckdb::LogicalFilter& op);
    void Visit(duckdb::LogicalComparisonJoin& op);

    void Visit(std::shared_ptr<duckdb::LogicalOperator> op) {
        if (op->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
            Visit(op->Cast<duckdb::LogicalGet>());
        } else if (op->type ==
                   duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
            Visit(op->Cast<duckdb::LogicalProjection>());
        } else if (op->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
            Visit(op->Cast<duckdb::LogicalFilter>());
        } else if (op->type ==
                   duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
            Visit(op->Cast<duckdb::LogicalComparisonJoin>());
        } else {
            throw std::runtime_error("Unsupported logical operator type");
        }
    }
};
