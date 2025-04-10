// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>

#include "_pipeline.h"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

/**
 * @brief Executor class for executing a DuckDB logical plan in streaming
 * fashion (push-based approach).
 *
 */
class Executor {
   private:
    std::vector<Pipeline> pipelines;

   public:
    explicit Executor(std::unique_ptr<duckdb::LogicalOperator> plan);

    /**
     * @brief Execute the plan and return the result (placeholder for now).
     */
    std::pair<int64_t, PyObject*> ExecutePipelines();
};
