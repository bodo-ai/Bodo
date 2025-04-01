// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>
#include "duckdb/planner/operator/logical_get.hpp"

/**
 * @brief Physical operators to be used in the execution pipelines (NOTE: they
 * are Bodo classes and not using DuckDB).
 *
 */
class PhysicalOperator {
   public:
    virtual std::pair<int64_t, PyObject*> execute() = 0;
    virtual ~PhysicalOperator() = default;
};

/**
 * @brief Physical node for reading Parquet files in pipelines.
 *
 */
class PhysicalReadParquet : public PhysicalOperator {
   public:
    PhysicalReadParquet(std::string path) : path(path) {}
    std::pair<int64_t, PyObject*> execute() override;

   private:
    std::string path;
};

/**
 * @brief Pipeline class for executing a sequence of physical operators.
 *
 */
class Pipeline {
   public:
    Pipeline(std::vector<std::shared_ptr<PhysicalOperator>> operators)
        : operators(operators) {}
    std::pair<int64_t, PyObject*> execute();

   private:
    std::vector<std::shared_ptr<PhysicalOperator>> operators;
};

/**
 * @brief Executor class for executing a DuckDB logical plan in streaming
 * fashion (push-based approach).
 *
 */
class Executor {
   public:
    Executor(std::unique_ptr<duckdb::LogicalOperator> plan);
    std::pair<int64_t, PyObject*> execute();

   private:
    std::vector<Pipeline> pipelines;
};
