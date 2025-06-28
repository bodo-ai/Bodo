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

        // Write finalizes the active pipeline but others need result collection
        if (builder.active_pipeline != nullptr) {
            std::shared_ptr<bodo::Schema> in_schema =
                builder.active_pipeline->getPrevOpOutputSchema();
            pipelines.push_back(builder.active_pipeline->BuildEnd(
                in_schema, bodo::Schema::FromArrowSchema(out_schema)));
        }
    }

    /**
     * @brief Execute the plan and return the result.
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> ExecutePipelines() {
        // Pipelines generation ensures that pipelines are in the right
        // order and that the dependencies are satisfied (e.g. join build
        // pipeline is before probe).
#ifdef DEBUG_PIPELINE
        std::cout << "ExecutePipelines with " << pipelines.size()
                  << " pipelines." << std::endl;
#endif
        for (size_t i = 0; i < pipelines.size(); ++i) {
#ifdef DEBUG_PIPELINE
            std::cout << "Before execute pipeline " << i << std::endl;
#endif
            pipelines[i]->Execute();
#ifdef DEBUG_PIPELINE
            std::cout << "After execute pipeline " << i << std::endl;
#endif
        }
        return pipelines.back()->GetResult();
    }
};
