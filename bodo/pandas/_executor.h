// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>

#include "duckdb/planner/logical_operator.hpp"

#include "_physical_conv.h"
#include "_pipeline.h"

/**
 * @brief Executor class for executing a DuckDB logical plan in streaming
 * fashion (push-based approach).
 *
 */
class Executor {
   private:
    std::vector<std::shared_ptr<Pipeline>> pipelines;

   public:
    explicit Executor(std::unique_ptr<duckdb::LogicalOperator> plan) {
        // Convert the logical plan to a physical plan
        PhysicalPlanBuilder builder;
        builder.Visit(*plan);
        pipelines = std::move(builder.finished_pipelines);

        if (builder.active_pipeline != nullptr) {
            pipelines.push_back(builder.active_pipeline->BuildEnd());
        }
    }

    /**
     * @brief Execute the plan and return the result (placeholder for now).
     */
    std::shared_ptr<table_info> ExecutePipelines() {
        // TODO: support multiple pipelines
        pipelines[0]->Execute();

        return pipelines[0]->GetResult();
    }
};
