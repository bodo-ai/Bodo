
// Defines Bodo data structures and functionality for DuckDB plans

#pragma once

#include <Python.h>
#include <utility>
#include "../io/arrow_reader.h"
#include "_executor.h"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/optimizer/optimizer.hpp"

/**
 * @brief Superclass for Bodo's DuckDB TableFunction classes.
 *
 */
class BodoScanFunction : public duckdb::TableFunction {
   public:
    BodoScanFunction(std::string name)
        : TableFunction(name, {}, nullptr, nullptr, nullptr, nullptr) {}
};

/**
 * @brief Superclass for Bodo's DuckDB TableFunctionData classes.
 *
 */
class BodoScanFunctionData : public duckdb::TableFunctionData {
   public:
    BodoScanFunctionData() {}
    /**
     * @brief Create a PhysicalOperator for reading data from this source.
     *
     * @return std::shared_ptr<PhysicalOperator> read operator
     */
    virtual std::shared_ptr<PhysicalOperator> CreatePhysicalOperator() = 0;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading Parquet datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoParquetScanFunction : public BodoScanFunction {
   public:
    BodoParquetScanFunction() : BodoScanFunction("bodo_read_parquet") {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Parquet datasets.
 *
 */
class BodoParquetScanFunctionData : public BodoScanFunctionData {
   public:
    BodoParquetScanFunctionData(std::string path) : path(path) {}
    std::shared_ptr<PhysicalOperator> CreatePhysicalOperator() override {
        return std::make_shared<PhysicalReadParquet>(path);
    }
    // Parquet dataset path
    std::string path;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading dataframe rows
 * (used in LogicalGet).
 *
 */
class BodoDataFrameScanFunction : public BodoScanFunction {
   public:
    BodoDataFrameScanFunction() : BodoScanFunction("bodo_read_df") {
        projection_pushdown = true;
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * spawner sequentially.
 *
 */
class BodoDataFrameSeqScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameSeqScanFunctionData(PyObject *df) : df(df) { Py_INCREF(df); }
    ~BodoDataFrameSeqScanFunctionData() { Py_DECREF(df); }
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    std::shared_ptr<PhysicalOperator> CreatePhysicalOperator() override {
        return std::make_shared<PhysicalReadPandas>(df);
    }

    PyObject *df;
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * workers in parallel.
 *
 */
class BodoDataFrameParallelScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameParallelScanFunctionData(std::string result_id)
        : result_id(result_id) {}
    ~BodoDataFrameParallelScanFunctionData() {}
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    std::shared_ptr<PhysicalOperator> CreatePhysicalOperator() override {
        // Read the dataframe from the result registry using
        // sys.modules["__main__"].RESULT_REGISTRY since importing
        // bodo.spawn.worker creates a new module with new empty registry.

        // Import Python sys module
        PyObject *sys_module = PyImport_ImportModule("sys");
        if (!sys_module) {
            throw std::runtime_error("Failed to import sys module");
        }

        // Get sys.modules dictionary
        PyObject *modules_dict = PyObject_GetAttrString(sys_module, "modules");
        if (!modules_dict) {
            Py_DECREF(sys_module);
            throw std::runtime_error("Failed to get sys.modules");
        }

        // Get __main__ module
        PyObject *main_module = PyDict_GetItemString(modules_dict, "__main__");
        if (!main_module) {
            Py_DECREF(modules_dict);
            Py_DECREF(sys_module);
            throw std::runtime_error("Failed to get __main__ module");
        }

        // Get RESULT_REGISTRY[result_id]
        PyObject *result_registry =
            PyObject_GetAttrString(main_module, "RESULT_REGISTRY");
        PyObject *df = PyDict_GetItemString(result_registry, result_id.c_str());
        if (!df) {
            throw std::runtime_error("Result ID not found in result registry");
        }
        Py_DECREF(result_registry);
        Py_DECREF(modules_dict);
        Py_DECREF(sys_module);

        return std::make_shared<PhysicalReadPandas>(df);
    }
    std::string result_id;
};

/**
 * @brief Optimize a DuckDB logical plan by applying the DuckDB optimizer.
 *
 * @param plan input logical plan to be optimized
 * @return duckdb::unique_ptr<duckdb::LogicalOperator> optimized plan
 */
duckdb::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan);

/**
 * @brief Execute a DuckDB logical plan in our C++ runtime and return the
 * result.
 *
 * @param plan logical plan to execute (should be optimized unless if testing)
 * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to int64,
 * pyarrow schema object
 */
std::pair<int64_t, PyObject *> execute_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan);

/**
 * @brief Creates a LogicalComparisonJoin node.
 *
 * @param lhs - left-side of the join
 * @param rhs - right-side of the join
 * @param join_type - the type of the join, e.g., "INNER"
 * @return duckdb::unique_ptr<duckdb::LogicalComparisonJoin> output node
 */
duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_comparison_join(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, duckdb::JoinType join_type,
    std::vector<std::pair<int, int>> &cond_vec);

/**
 * @brief Creates a LogicalProjection node.
 *
 * @param source - the data source to project from
 * @param select_vec - vector of column indices to project
 * @param out_schema_py - the schema of data coming out of the projection
 * @return duckdb::unique_ptr<duckdb::LogicalProjection> output node
 */
duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> &select_vec, PyObject *out_schema_py);

/**
 * @brief Create an expression from a constant integer.
 *
 * @param val - the constant int for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const int expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int val);

/**
 * @brief Create an expression that references a specified column.
 *
 * @param field_py - the data type of the specified column
 * @param col_idx - the column index of the specified column
 * @return duckdb::unique_ptr<duckdb::Expression> - the column reference
 * expression
 */
duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr(PyObject *field_py,
                                                         int col_idx);

/**
 * @brief Create an expression from two sources and an operator.
 *
 * @param lhs - the left-hand side of the expression
 * @param rhs - the right-hand side of the expression
 * @param etype - the expression type comparing the two sources
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
duckdb::unique_ptr<duckdb::Expression> make_binop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype);

/**
 * @brief Create a filter node.
 *
 * @param source - the source of the data to be filtered
 * @param filter_expr - the filter expression to apply to the source
 * @return duckdb::unique_ptr<duckdb::LogicalFilter> - the filter node
 */
duckdb::unique_ptr<duckdb::LogicalFilter> make_filter(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::unique_ptr<duckdb::Expression> &filter_expr);

/**
 * @brief Creates a LogicalGet node for reading a Parquet dataset in DuckDB with
 * Bodo metadata.
 *
 * @param parquet_path path to the Parquet dataset
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    std::string parquet_path, PyObject *pyarrow_schema);

/**
 * @brief Create LogicalGet node for reading a dataframe sequentially
 *
 * @param df input dataframe to read
 * @param pyarrow_schema schema of the dataframe
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output DuckDB node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_seq_node(
    PyObject *df, PyObject *pyarrow_schema);

/**
 * @brief Create LogicalGet node for reading a dataframe in parallel
 *
 * @param result_id input dataframe id on workers to read
 * @param pyarrow_schema schema of the dataframe
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output DuckDB node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_parallel_node(
    std::string result_id, PyObject *pyarrow_schema);

/**
 * @brief Returns a statically created DuckDB client context.
 *
 * @return duckdb::ClientContext& static context object
 */
duckdb::ClientContext &get_duckdb_context();

/**
 * @brief Returns a statically created DuckDB binder.
 *
 * @return duckdb::shared_ptr<duckdb::Binder> static binder object
 */
duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder();

/**
 * @brief Returns a statically created duckdb optimizer
 *
 * @return duckdb::Optimizer& static optimizer object
 */
duckdb::Optimizer &get_duckdb_optimizer();

/**
 * @brief Convert an Arrow schema to DuckDB column names and data types to pass
 * to plan nodes.
 *
 * @param arrow_schema input Arrow schema
 * @return std::pair<duckdb::vector<duckdb::string>,
 * duckdb::vector<duckdb::LogicalType>> duckdb column names and types
 */
std::pair<duckdb::vector<duckdb::string>, duckdb::vector<duckdb::LogicalType>>
arrow_schema_to_duckdb(const std::shared_ptr<arrow::Schema> &arrow_schema);

/**
 * @brief Convert an Arrow field to a DuckDB column name and data type.
 *
 * @param field input Arrow field
 * @return std::pair<duckdb::string, duckdb::LogicalType> duckdb column name and
 * type
 */
std::pair<duckdb::string, duckdb::LogicalType> arrow_field_to_duckdb(
    const std::shared_ptr<arrow::Field> &field);
