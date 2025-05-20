// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>
#include "_physical_conv.h"
#include "_pipeline.h"
#include "duckdb/planner/logical_operator.hpp"

/**
 * @brief Executor class for executing a DuckDB logical plan in streaming
 * fashion (push-based approach).
 *
 */
class Executor {
   private:
    std::vector<std::shared_ptr<Pipeline>> pipelines;

   public:
    explicit Executor(std::unique_ptr<duckdb::LogicalOperator> plan,
                      std::shared_ptr<arrow::Schema> out_schema) {
        // Convert the logical plan to a physical plan
        PhysicalPlanBuilder builder;
        builder.Visit(*plan);
        pipelines = std::move(builder.finished_pipelines);
        assert(builder.active_pipeline != nullptr);
        std::shared_ptr<bodo::Schema> in_schema =
            builder.active_pipeline->getPrevOpOutputSchema();
        pipelines.push_back(builder.active_pipeline->BuildEnd(
            in_schema, bodo::Schema::FromArrowSchema(out_schema)));
    }

    /**
     * @brief Execute the plan and return the result.
     */
    std::shared_ptr<table_info> ExecutePipelines() {
        // Pipelines generation ensures that pipelines are in the right
        // order and that the dependencies are satisfied (e.g. join build
        // pipeline is before probe).
        for (size_t i = 0; i < pipelines.size(); ++i) {
            pipelines[i]->Execute();
        }
        return pipelines.back()->GetResult();
    }
};
