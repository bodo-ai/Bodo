
// Defines Bodo data structures and functionality for DuckDB plans

#pragma once

#include <Python.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include <utility>
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"

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
    std::unique_ptr<duckdb::LogicalOperator> plan, PyObject *out_schema_py);

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
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py);

/**
 * @brief Creates a BoundOrderByNode object.
 *
 * @param asc - true if sorted ascending
 * @param na_first - true if null go first in ordering
 * @param col_ref_expr - column reference expr to sort on
 * @return duckdb::unique_ptr<duckdb::LogicalProjection> output node
 */
duckdb::unique_ptr<duckdb::BoundOrderByNode> make_order_by_node(
    bool asc,
    bool na_first,
    std::unique_ptr<duckdb::Expression> col_ref_expr);

/**
 * @brief Creates a LogicalOrder node.
 *
 * @param source - the data source to project from
 * @param order_vec - vector of BoundOrderByNode to say how to order
 * @param out_schema_py - the schema of data coming out of the order
 * @return duckdb::unique_ptr<duckdb::LogicalOrder> output node
 */
duckdb::unique_ptr<duckdb::LogicalOrder> make_order(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<std::unique_ptr<duckdb::BoundOrderByNode>> &order_vec,
    PyObject *out_schema_py);

/**
 * @brief Creates a LogicalAggregate node.
 *
 * @param source - the data source to aggregate
 * @param key_indices - key column indices to group by
 * @param exprs - vector of aggregate exprs
 * @param out_schema_py - the schema of data coming out of the aggregate
 * @return duckdb::unique_ptr<duckdb::LogicalAggregate> output node
 */
duckdb::unique_ptr<duckdb::LogicalAggregate> make_aggregate(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> &key_indices,
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py);

/**
 * @brief Get column indices that are pushed down from a projection node to its
 * source. Used for testing.
 *
 * @param proj input projection node
 * @return std::vector<int> pushed down column indices
 */
std::vector<int> get_projection_pushed_down_columns(
    std::unique_ptr<duckdb::LogicalOperator> &proj);

/**
 * @brief Creates an Expression node with a UDF inside.
 *
 * @param source input table plan
 * @param func UDF function to execute
 * @param out_schema_py output data type (single column for df.apply)
 * @param args arguments to the UDF
 * @param selected_columns column indices for input table columns to pass to the
 * UDF
 * @return duckdb::unique_ptr<duckdb::Expression> Expression node for UDF
 */
duckdb::unique_ptr<duckdb::Expression> make_python_scalar_func_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    PyObject *args, const std::vector<int> &selected_columns);

/**
 * @brief Create an expression from a constant integer.
 *
 * @param val - the constant int for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const int expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int val);

/**
 * @brief Create an expression from a constant float.
 *
 * @param val - the constant float for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const float expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_float_expr(float val);

/**
 * @brief Create an expression from a constant string.
 *
 * @param val - the constant string for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const string expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_string_expr(
    const std::string &val);

/**
 * @brief Create an expression from a constant timestamp with ns resolution.
 *
 * @param val - the constant timestamp for the expression in ns since epoch
 * @return duckdb::unique_ptr<duckdb::Expression> - the const timestamp expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_timestamp_ns_expr(
    int64_t val);

/**
 * @brief Create an expression that references a specified column.
 *
 * @param field_py - the data type of the specified column
 * @param col_idx - the column index of the specified column
 * @return duckdb::unique_ptr<duckdb::Expression> - the column reference
 * expression
 */
duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *field_py,
    int col_idx);

/**
 * @brief Create an aggregate expression for a given function name and input
 * source node
 *
 * @param source input source node to aggregate
 * @param field_py output field type for the aggregate function
 * @param function_name function name for matching in backend
 * @param input_column_indices argument column indices for the input source
 * @return duckdb::unique_ptr<duckdb::Expression> new BoundAggregateExpression
 * object
 */
duckdb::unique_ptr<duckdb::Expression> make_agg_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *field_py,
    std::string function_name, std::vector<int> input_column_indices);

/**
 * @brief Create an expression from two sources and an operator.
 *
 * @param lhs - the left-hand side of the expression
 * @param rhs - the right-hand side of the expression
 * @param etype - the expression type comparing the two sources
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
std::unique_ptr<duckdb::Expression> make_comparison_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype);

/**
 * @brief Create an expression from two sources and an operator.
 *
 * @param lhs - the left-hand side of the expression
 * @param rhs - the right-hand side of the expression
 * @param opstr - the name of the function combining the two sources
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
std::unique_ptr<duckdb::Expression> make_arithop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, std::string opstr);

/**
 * @brief Create an expression from a source and function as a string.
 *
 * @param source - the source of the expression
 * @param opstr - the name of the function to apply to the source
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
std::unique_ptr<duckdb::Expression> make_unaryop_expr(
    std::unique_ptr<duckdb::Expression> &source, std::string opstr);

/**
 * @brief Create a conjunction (and/or) expression from two sources.
 *
 * @param lhs - the left-hand side of the expression
 * @param rhs - the right-hand side of the expression
 * @param etype - the expression type (and/or) comparing the two sources
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
duckdb::unique_ptr<duckdb::Expression> make_conjunction_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype);

/**
 * @brief Create a unary op (e.g., not) expression from a source.
 *
 * @param lhs - the left-hand side of the expression
 * @param etype - the expression type (e.g., not) to apply to the source
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
duckdb::unique_ptr<duckdb::Expression> make_unary_expr(
    std::unique_ptr<duckdb::Expression> &lhs, duckdb::ExpressionType etype);

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
 * @brief Create a limit node.
 *
 * @param source - the source of the data to be filtered
 * @param n - the number of rows to return
 * @return duckdb::unique_ptr<duckdb::LogicalLimit> - the limit node
 */
duckdb::unique_ptr<duckdb::LogicalLimit> make_limit(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n);

/**
 * @brief Create a sample node.
 *
 * @param source - the source of the data to be filtered
 * @param n - the number of rows to return
 * @return duckdb::unique_ptr<duckdb::LogicalLimit> - the sample node
 */
duckdb::unique_ptr<duckdb::LogicalSample> make_sample(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n);

/**
 * @brief Creates a LogicalGet node for reading a Parquet dataset in DuckDB with
 * Bodo metadata.
 *
 * @param parquet_path path to the Parquet dataset
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    PyObject *parquet_path, PyObject *pyarrow_schema,
    PyObject *storage_options);

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
 * @brief Creates a LogicalGet node for reading an Iceberg dataset in DuckDB
 * with Bodo metadata.
 *
 * @param pyarrow_schema schema of the Iceberg dataset
 * @param table_name Identifier for the Iceberg table, includes schema and table
 * @param pyiceberg_catalog Iceberg catalog object
 * @param iceberg_filter Iceberg filter expression
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_iceberg_get_node(
    PyObject *pyarrow_schema, std::string table_identifier,
    PyObject *pyiceberg_catalog, PyObject *iceberg_filter);

/**
 * @brief Returns a statically created DuckDB database.
 *
 * @return duckdb::DuckDB& static context object
 */
duckdb::shared_ptr<duckdb::DuckDB> get_duckdb();

/**
 * @brief Returns a statically created DuckDB client context.
 *
 * @return duckdb::ClientContext& static context object
 */
duckdb::shared_ptr<duckdb::ClientContext> get_duckdb_context();

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
duckdb::shared_ptr<duckdb::Optimizer> get_duckdb_optimizer();

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

/**
 * @brief Convert a plan rooted at the given logical operator into text or
 * graphviz.
 *
 * @param plan - the root of the plan to convert to graphviz or text
 * @param graphviz_format - use graphviz format if true, otherwise text
 */
std::string plan_to_string(std::unique_ptr<duckdb::LogicalOperator> &plan,
                           bool graphviz_format);

/**
 * @brief Count the number of nodes in the expression tree.
 *
 * @param op root of expression tree
 * @return number of nodes in that tree
 */
int planCountNodes(std::unique_ptr<duckdb::LogicalOperator> &op);

/**
 * @brief Set the C++ table column names and metadata from PyArrow schema object
 *
 * @param table_pointer C++ table pointer
 * @param pyarrow_schema input PyArrow schema object
 */
void set_table_meta_from_arrow(int64_t table_pointer, PyObject *pyarrow_schema);
