#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "_plan.h"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"

#include "_pipeline.h"
#include "physical/cte.h"

struct CTEInfo {
    std::shared_ptr<PhysicalCTE> physical_node;
    std::shared_ptr<Pipeline> cte_pipeline_root;
};

class PhysicalPlanBuilder {
   private:
    bool node_run_on_gpu(duckdb::LogicalOperator& op) {
        auto iter = run_on_gpu.find(&op);
        if (iter == run_on_gpu.end()) {
            return false;
        } else {
            return iter->second;
        }
    }

   public:
    std::shared_ptr<PipelineBuilder> active_pipeline;
    std::shared_ptr<Pipeline> terminal_pipeline;
    std::map<duckdb::idx_t, CTEInfo>& ctes;
    std::map<duckdb::LogicalOperator*, bool>& run_on_gpu;

    // Mapping of join ids to their JoinState pointers for join filter operators
    // (filled during physical plan construction). Using loose pointers since
    // PhysicalJoinFilter only needs to access the JoinState during execution
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states;
    // Mapping of join ids to the pipeline for build side of the join.
    std::shared_ptr<std::unordered_map<int, std::shared_ptr<Pipeline>>>
        join_filter_pipelines;
#ifdef USE_CUDF
    // Mapping of join ids to whether they are run on GPU.
    // If so, then associated JoinFilter nodes not used.
    std::shared_ptr<std::unordered_map<int, bool>> join_on_gpu;
#endif

    PhysicalPlanBuilder(
        std::map<duckdb::idx_t, CTEInfo>& _ctes,
        std::map<duckdb::LogicalOperator*, bool>& _run_on_gpu,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            _join_filter_states =
                std::make_shared<std::unordered_map<int, join_state_t>>(),
        std::shared_ptr<std::unordered_map<int, std::shared_ptr<Pipeline>>>
            _join_filter_pipelines = std::make_shared<
                std::unordered_map<int, std::shared_ptr<Pipeline>>>()
#ifdef USE_CUDF
            ,
        std::shared_ptr<std::unordered_map<int, bool>> _join_on_gpu =
            std::make_shared<std::unordered_map<int, bool>>()
#endif
            )
        : active_pipeline(nullptr),
          ctes(_ctes),
          run_on_gpu(_run_on_gpu),
          join_filter_states(std::move(_join_filter_states)),
          join_filter_pipelines(std::move(_join_filter_pipelines))
#ifdef USE_CUDF
          ,
          join_on_gpu(std::move(_join_on_gpu))
#endif
    {
    }

    template <typename T,
              std::enable_if_t<std::is_base_of_v<PhysicalSink, T> &&
                                   std::is_base_of_v<PhysicalSource, T>,
                               int> = 0>
    void FinishPipelineOneOperator(std::shared_ptr<T> obj) {
        std::shared_ptr<Pipeline> done_pipeline =
            active_pipeline->Build(std::static_pointer_cast<PhysicalSink>(obj));
        active_pipeline = std::make_shared<PipelineBuilder>(
            std::static_pointer_cast<PhysicalSource>(obj));
        active_pipeline->addRunBefore(done_pipeline);
    }

    template <typename T,
              std::enable_if_t<std::is_base_of_v<PhysicalGPUSink, T> &&
                                   std::is_base_of_v<PhysicalGPUSource, T>,
                               int> = 0>
    void FinishPipelineOneOperator(std::shared_ptr<T> obj) {
        std::shared_ptr<Pipeline> done_pipeline = active_pipeline->Build(
            std::static_pointer_cast<PhysicalGPUSink>(obj));
        active_pipeline = std::make_shared<PipelineBuilder>(
            std::static_pointer_cast<PhysicalGPUSource>(obj));
        active_pipeline->addRunBefore(done_pipeline);
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
    void Visit(duckdb::LogicalEmptyResult& op);
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
                   duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
            Visit(op.Cast<duckdb::LogicalEmptyResult>());
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
