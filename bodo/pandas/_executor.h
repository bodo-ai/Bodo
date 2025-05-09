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
        assert(builder.active_pipelines.size() == 1);
        pipelines.push_back(builder.active_pipelines.top().first->BuildEnd(out_schema));
    }

    /**
     * @brief Execute the plan and return the result.
     */
    std::shared_ptr<table_info> ExecutePipelines() {
        // We must execute all pipelines once.
        // Though there are dependencies between them
        // the method of construction should guarantee that
        // earlier items in the vector are dependencies to
        // later items so we can still process them in order.
        for (size_t i = 0; i < pipelines.size(); ++i) {
            assert(pipelines[i]->dependencies_finished());
            // Run that pipeline.
            pipelines[i]->Execute();
        }
        return pipelines.back()->GetResult();
    }
};
