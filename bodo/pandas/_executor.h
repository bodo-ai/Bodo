// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>
#include <object.h>
#include <pytypedefs.h>
#include "../io/parquet_reader.h"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

class Executor;
class Pipeline;

/**
 * @brief Physical operators to be used in the execution pipelines (NOTE: they
 * are Bodo classes and not using DuckDB).
 *
 */
class PhysicalOperator {
   public:
    /**
     * @brief Execute the physical operator and return the result (placeholder
     * for now).
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    virtual std::pair<int64_t, PyObject *> execute() = 0;
    virtual ~PhysicalOperator() = default;
    std::pair<int64_t, PyObject *> result;
};

/**
 * @brief Physical node for reading Parquet files in pipelines.
 *
 */
class PhysicalReadParquet : public PhysicalOperator {
   public:
    // TODO: Fill in the contents with info from the logical operator
    PhysicalReadParquet(std::string path, PyObject *_pyarrow_schema,
                        PyObject *storage_options)
        : path(path), pyarrow_schema(_pyarrow_schema) {
        py_path = PyUnicode_FromString(path.c_str());

        std::shared_ptr<arrow::Schema> arrow_schema =
            unwrap_schema(pyarrow_schema);
        int num_fields = arrow_schema->num_fields();
        std::vector<int> selected_fields(num_fields);
        std::vector<bool> is_nullable(num_fields);
        for (int i = 0; i < num_fields; i++) {
            selected_fields[i] = i;
            // setting nullable to true for all fields for now
            is_nullable[i] = true;
        }

        internal_reader = new ParquetReader(
            py_path, true, Py_None, storage_options, pyarrow_schema, -1,
            selected_fields, is_nullable, false, 4000);
        internal_reader->init_pq_reader({}, nullptr, nullptr, 0);
    }

    /**
     * @brief Read parquet and return the result (placeholder for now).
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<int64_t, PyObject *> execute() override;

    ~PhysicalReadParquet() { Py_DECREF(py_path); }

   private:
    std::string path;
    PyObject *py_path;
    PyObject *pyarrow_schema;
    ParquetReader *internal_reader;
};

/**
 * @brief Physical node for reading Parquet files in pipelines.
 *
 */
class PhysicalReadPandas : public PhysicalOperator {
   public:
    PhysicalReadPandas(PyObject *df) : df(df) {
        Py_INCREF(df);
        num_rows = PyObject_Length(df);
    }
    ~PhysicalReadPandas() { Py_DECREF(df); }

    /**
     * @brief Read parquet and return the result (placeholder for now).
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<int64_t, PyObject *> execute() override;

   private:
    PyObject *df;
    int64_t current_row = 0;
    int64_t num_rows;
};

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalOperator {
   public:
    PhysicalProjection(std::shared_ptr<PhysicalOperator> src,
                       std::vector<int64_t> &cols)
        : src(src), selected_columns(cols) {}

    /**
     * @brief Do projection.
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<int64_t, PyObject *> execute() override;

    static std::shared_ptr<PhysicalOperator> make(
        const duckdb::LogicalProjection &proj_plan,
        const std::shared_ptr<PhysicalOperator> &source);

   private:
    std::shared_ptr<PhysicalOperator> src;
    std::vector<int64_t> selected_columns;
};

/**
 * @brief Pipeline class for executing a sequence of physical operators.
 *
 */
class Pipeline {
   public:
    Pipeline(std::vector<std::shared_ptr<PhysicalOperator>> operators)
        : operators(operators) {}

    /**
     * @brief Execute the pipeline and return the result (placeholder for now).
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<int64_t, PyObject *> execute();

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

    /**
     * @brief Execute the plan and return the result (placeholder for now).
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<int64_t, PyObject *> execute();
    std::shared_ptr<PhysicalOperator> processNode(
        std::unique_ptr<duckdb::LogicalOperator> &plan);
    std::vector<Pipeline> pipelines;
};
