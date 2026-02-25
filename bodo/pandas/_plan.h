
// Defines Bodo data structures and functionality for DuckDB plans

#pragma once

#include <Python.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include <cstdint>
#include <utility>
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

namespace bodo {

/**
 * @brief Logical join filter operator (extension of DuckDB logical operator).
 *
 */
class LogicalJoinFilter : public duckdb::LogicalOperator {
   public:
    // TODO: add join filter to DuckDB operator types to allow more extension
    // types
    static constexpr const duckdb::LogicalOperatorType TYPE =
        duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;

    LogicalJoinFilter(
        duckdb::unique_ptr<duckdb::LogicalOperator> source,
        const std::vector<int> filter_ids,
        const std::vector<std::vector<int64_t>> filter_columns,
        const std::vector<std::vector<bool>> is_first_locations,
        const std::vector<std::vector<int64_t>> orig_build_key_cols)
        : duckdb::LogicalOperator(
              duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR),
          filter_ids(std::move(filter_ids)),
          filter_columns(std::move(filter_columns)),
          is_first_locations(std::move(is_first_locations)),
          orig_build_key_cols(std::move(orig_build_key_cols)) {
        estimated_cardinality = source->estimated_cardinality;
        has_estimated_cardinality = source->has_estimated_cardinality;
        this->children.push_back(std::move(source));
    }

    duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override {
        return children[0]->GetColumnBindings();
    }

    duckdb::string GetName() const override { return "LogicalJoinFilter"; }
    duckdb::InsertionOrderPreservingMap<duckdb::string> ParamsToString()
        const override {
        duckdb::InsertionOrderPreservingMap<duckdb::string> map;

        map["filter_ids"] =
            fmt::format("{}", fmt::join(this->filter_ids, ", "));
        map["filter_columns"] = "[";
        for (const auto &cols : this->filter_columns) {
            map["filter_columns"] +=
                fmt::format("[{}], ", fmt::join(cols, ", "));
        }
        map["filter_columns"] += "]";
        map["is_first_locations"] = "[";
        for (const auto &locs : this->is_first_locations) {
            map["is_first_locations"] +=
                fmt::format("[{}], ", fmt::join(locs, ", "));
        }
        map["is_first_locations"] += "]";
        map["orig_build_key_cols"] = "[";
        for (const auto &cols : this->orig_build_key_cols) {
            map["orig_build_key_cols"] +=
                fmt::format("[{}], ", fmt::join(cols, ", "));
        }
        map["orig_build_key_cols"] += "]";

        return map;
    }

    // IDs of joins creating each filter
    const std::vector<int> filter_ids;
    // Mapping columns of the join to the columns in the current table
    const std::vector<std::vector<int64_t>> filter_columns;
    // Indicating for which of the columns is it the first filtering site
    const std::vector<std::vector<bool>> is_first_locations;
    const std::vector<std::vector<int64_t>> orig_build_key_cols;

   protected:
    void ResolveTypes() override { types = children[0]->types; }
};

}  // namespace bodo

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
 * @brief Creates a new table index.
 *
 * @return duckdb::idx_t
 */
duckdb::idx_t getTableIndex();

/**
 * @brief Creates a LogicalMaterializedCTE node.
 *
 * @param duplicated - the duplicated part of the plan
 * @param uses_duplicated - the plan that uses the duplicated part
 * @param out_schema_py - the schema of data coming out
 * @param table_index - a pre-allocated table_index to match with CTE ref
 * @return duckdb::unique_ptr<duckdb::LogicalMaterializedCTE> output node
 */
duckdb::unique_ptr<duckdb::LogicalMaterializedCTE> make_cte(
    std::unique_ptr<duckdb::LogicalOperator> &duplicated,
    std::unique_ptr<duckdb::LogicalOperator> &uses_duplicated,
    PyObject *out_schema_py, duckdb::idx_t table_index);

/**
 * @brief Creates a LogicalCTERef node.
 *
 * @param out_schema_py - the schema of data coming out
 * @param table_index - a pre-allocated table_index that matches with CTE
 * @return duckdb::unique_ptr<duckdb::LogicalCTERef> output node
 */
duckdb::unique_ptr<duckdb::LogicalCTERef> make_cte_ref(
    PyObject *out_schema_py, duckdb::idx_t table_index);

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
    std::vector<std::pair<int, int>> &cond_vec, int join_id);

/**
 * @brief Creates a LogicalJoinFilter node.
 *
 * @param source - input table
 * @param filter_ids - IDs of joins creating each filter
 * @param filter_columns - Mapping columns of the join to the columns in the
 * current table
 * @param is_first_locations - Indicating for which of the columns is it the
 * first filtering site
 * @return duckdb::unique_ptr<bodo::LogicalJoinFilter> output node
 */
duckdb::unique_ptr<bodo::LogicalJoinFilter> make_join_filter(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> filter_ids,
    std::vector<std::vector<int64_t>> filter_columns,
    std::vector<std::vector<bool>> is_first_locations,
    std::vector<std::vector<int64_t>> orig_build_key_cols);

/**
 * @brief Creates a LogicalSetOperation node.
 *
 * @param lhs - left-side of the set operation
 * @param rhs - right-side of the set operation
 * @param setop - the type of set operation, e.g., "UNION"
 * @return duckdb::unique_ptr<duckdb::LogicalSetOperation> output node
 */
duckdb::unique_ptr<duckdb::LogicalSetOperation> make_set_operation(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, const std::string &setop,
    int64_t num_cols);

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
 * @brief Creates a LogicalDistinct node.
 *
 * @param source - the data source to do distinct on
 * @param select_vec - vector of column indices to be distinct
 * @param out_schema_py - the schema of data coming out of the distinct
 * @return duckdb::unique_ptr<duckdb::LogicalDistinct> output node
 */
duckdb::unique_ptr<duckdb::LogicalDistinct> make_distinct(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py);

/**
 * @brief Creates a LogicalOrder node.
 *
 * @param source - the data source to order
 * @param asc - vector of bool to say whether corresponding key is sorted
 *              ascending (true) or descending (false)
 * @param na_position - vector of bool to say whether corresponding key places
 *              na values first (true) or last (false)
 * @param cols - vector of int specifying the key column indices for sorting
 * @param schema_py - the schema of data coming into the order
 * @return duckdb::unique_ptr<duckdb::LogicalOrder> output node
 */
duckdb::unique_ptr<duckdb::LogicalOrder> make_order(
    std::unique_ptr<duckdb::LogicalOperator> &source, std::vector<bool> &asc,
    std::vector<bool> &na_position, std::vector<int> &cols,
    PyObject *schema_py);

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
 * @param out_schema_py output data type (single column for df.apply)
 * @param args arguments to the UDF
 * @param selected_columns column indices for input table columns to pass to the
 * UDF
 * @param is_cfunc Whether to compile and run func as a cfunc
 * @param has_state Whether the UDF requires separate initialization state
 * @param func_name Name of Arrow Compute function, empty string for Python
 * execution
 * @return duckdb::unique_ptr<duckdb::Expression> Expression node for UDF
 */
duckdb::unique_ptr<duckdb::Expression> make_scalar_func_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    PyObject *args, const std::vector<int> &selected_columns, bool is_cfunc,
    bool has_state, const std::string arrow_compute_func);

/**
 * @brief Create an expression for a NULL value of given type.
 *
 * @param val - the type to create the NULL value of
 * @return duckdb::unique_ptr<duckdb::Expression> - the const null expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_null(PyObject *out_schema_py,
                                                       int64_t field_idx);

/**
 * @brief Create an expression from a constant integer.
 *
 * @param val - the constant int for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const int expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int64_t val);

/**
 * @brief Create an expression from a constant float.
 *
 * @param val - the constant float for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const float expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_double_expr(double val);

/**
 * @brief Create an expression from a constant string.
 *
 * @param val - the constant string for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const string expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_string_expr(
    const std::string &val);

/**
 * @brief Create an expression from a constant bool.
 *
 * @param val - the constant bool for the expression
 * @return duckdb::unique_ptr<duckdb::Expression> - the const bool expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_bool_expr(bool val);

/**
 * @brief Create an expression from a constant timestamp with ns resolution.
 *
 * @param val - the constant timestamp for the expression in ns since epoch
 * @return duckdb::unique_ptr<duckdb::Expression> - the const timestamp expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_timestamp_ns_expr(
    int64_t val);

/**
 * @brief Create an expression from a constant date32.
 *
 * @param val - the constant date for the expression in days since epoch
 * @return duckdb::unique_ptr<duckdb::Expression> - the const date expr
 */
duckdb::unique_ptr<duckdb::Expression> make_const_date32_expr(int32_t val);

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
 * @param out_schema_py output data type, used only for reduction operators
 * @param function_name function name for matching in backend
 * @param input_column_indices argument column indices for the input source
 * @param dropna argument column indices for the input source
 * @return duckdb::unique_ptr<duckdb::Expression> new BoundAggregateExpression
 * object
 */
duckdb::unique_ptr<duckdb::Expression> make_agg_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    std::string function_name, PyObject *py_udf_args,
    std::vector<int> input_column_indices, bool dropna);

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
 * @param out_schema_py output data type
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
std::unique_ptr<duckdb::Expression> make_arithop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, std::string opstr,
    PyObject *out_schema_py);

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
 * @brief Create a cast expression.
 *
 * @param source - the source of the expression
 * @param out_schema_py output data type
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
std::unique_ptr<duckdb::Expression> make_cast_expr(
    std::unique_ptr<duckdb::Expression> &source, PyObject *out_schema_py);

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
 * @brief Create a case expression from a source.
 *
 * @param when - the condition to evaluate
 * @param then - the expression to return if the condition is true
 * @param else_ - the expression to return if the condition is false
 * @return duckdb::unique_ptr<duckdb::Expression> - the output expr
 */
duckdb::unique_ptr<duckdb::Expression> make_case_expr(
    std::unique_ptr<duckdb::Expression> &when,
    std::unique_ptr<duckdb::Expression> &then,
    std::unique_ptr<duckdb::Expression> &else_);

/**
 * @brief Create a cross product node.
 *
 * @param lhs - the left-hand side of the cross product
 * @param rhs - the right-hand side of the cross product
 * @return duckdb::unique_ptr<duckdb::LogicalCrossProduct> - the cross product
 * node
 */
duckdb::unique_ptr<duckdb::LogicalCrossProduct> make_cross_product(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs);

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
 * @param pyarrow_schema schema of the dataframe
 * @param num_rows estimated number of rows for this source
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    PyObject *parquet_path, PyObject *pyarrow_schema, PyObject *storage_options,
    int64_t num_rows);

/**
 * @brief Create a LogicalCopyToFile node for writing a Parquet dataset.
 *
 * @param source input data to write
 * @param pyarrow_schema schema of the data to write
 * @param path path to write
 * @param compression compression type to use (e.g., "snappy")
 * @param bucket_region region for the S3 bucket (if applicable)
 * @param row_group_size row group size for Parquet files
 * @return duckdb::unique_ptr<duckdb::LogicalCopyToFile> created node
 */
duckdb::unique_ptr<duckdb::LogicalCopyToFile> make_parquet_write_node(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *pyarrow_schema,
    std::string path, std::string compression, std::string bucket_region,
    int64_t row_group_size);

duckdb::unique_ptr<duckdb::LogicalCopyToFile> make_iceberg_write_node(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *pyarrow_schema,
    std::string table_loc, std::string bucket_region, int64_t max_pq_chunksize,
    std::string compression, PyObject *partition_tuples, PyObject *sort_tuples,
    std::string iceberg_schema_str, PyObject *output_pa_schema, PyObject *pyfs);

/**
 * @brief Create a LogicalCopyToFile node for writing S3 Vectors.
 * @param source input data to write
 * @param pyarrow_schema schema of the data to write
 * @param vector_bucket_name name of the S3 bucket to write vectors to
 * @param index_name name of the vector index to write to
 * @param region AWS region for the S3 bucket
 * @return duckdb::unique_ptr<duckdb::LogicalCopyToFile> created node
 */
duckdb::unique_ptr<duckdb::LogicalCopyToFile> make_s3_vectors_write_node(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *pyarrow_schema,
    std::string vector_bucket_name, std::string index_name, PyObject *region);

/**
 * @brief Create LogicalGet node for reading a dataframe sequentially
 *
 * @param df input dataframe to read
 * @param pyarrow_schema schema of the dataframe
 * @param num_rows estimated number of rows for this source
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output DuckDB node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_seq_node(
    PyObject *df, PyObject *pyarrow_schema, int64_t num_rows);

/**
 * @brief Create LogicalGet node for reading a dataframe in parallel
 *
 * @param result_id input dataframe id on workers to read
 * @param pyarrow_schema schema of the dataframe
 * @param num_rows estimated number of rows for this source
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output DuckDB node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_parallel_node(
    std::string result_id, PyObject *pyarrow_schema, int64_t num_rows);

/**
 * @brief Creates a LogicalGet node for reading an Iceberg dataset in DuckDB
 * with Bodo metadata.
 *
 * @param pyarrow_schema schema of the Iceberg dataset
 * @param table_name Identifier for the Iceberg table, includes schema and table
 * @param pyiceberg_catalog Iceberg catalog object
 * @param iceberg_filter Iceberg filter expression
 * @param iceberg_schema Iceberg schema object
 * @param snapshot_id Snapshot ID to read from
 * @param table_len_estimate Estimated number of rows in the Iceberg table
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_iceberg_get_node(
    PyObject *pyarrow_schema, std::string table_identifier,
    PyObject *pyiceberg_catalog, PyObject *iceberg_filter,
    PyObject *iceberg_schema, int64_t snapshot_id, uint64_t table_len_estimate);

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
 * @brief Convert DuckDB types to an Arrow schema
 *
 * @param types vector of DuckDB LogicalTypes
 * @param names optional column names
 * @return arrow schema
 */
std::shared_ptr<arrow::Schema> duckdb_to_arrow_schema(
    const std::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &names = {});

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
 * @brief convert a PyArrow table to a Bodo C++ table pointer.
 *
 * @param pyarrow_table input PyArrow table object
 * @return int64_t C++ table pointer cast to int64_t
 */
int64_t pyarrow_to_cpp_table(PyObject *pyarrow_table);

/**
 * @brief Convert a PyArrow array to a C++ table pointer with column names and
 * metadata set properly.
 * Uses in_cpp_table for appending Index arrays if any and pandas metadata.
 * Deletes in_cpp_table after use.
 *
 * @param arrow_array input Arrow array object
 * @param name column name
 * @param in_cpp_table C++ table pointer cast to int64_t
 * @return int64_t C++ table pointer cast to int64_t
 */
int64_t pyarrow_array_to_cpp_table(PyObject *arrow_array, std::string name,
                                   int64_t in_cpp_table);

/**
 * @brief convert a Bodo C++ table pointer to a PyArrow table object.
 *
 * @param cpp_table C++ table pointer cast to int64_t
 * @param delete_cpp_table whether to delete the C++ table after conversion
 * @return PyObject* PyArrow table object
 */
PyObject *cpp_table_to_pyarrow(int64_t cpp_table, bool delete_cpp_table = true);

/**
 * @brief Convert the first column of a Bodo C++ table to a PyArrow array
 * object. NOTE: does not delete the C++ table.
 *
 * @param cpp_table C++ table pointer cast to int64_t
 * @return PyObject* PyArrow array object
 */
PyObject *cpp_table_to_pyarrow_array(int64_t cpp_table);

/**
 * @brief Get the name of the first field in a Bodo C++ table.
 * Returns an empty string if the table has no column names.
 * NOTE: does not delete the C++ table.
 *
 * @param cpp_table C++ table pointer cast to int64_t
 * @return std::string name of the first field
 */
std::string cpp_table_get_first_field_name(int64_t cpp_table);

/**
 * @brief Delete a Bodo C++ table pointer.
 *
 * @param cpp_table C++ table pointer cast to int64_t
 */
void cpp_table_delete(int64_t cpp_table);

/**
 * @brief Sets the use cudf flag.
 *
 * @param use_cudf bool that if true enables use of cudf in backend
 */
void set_cudf_vars(bool use_cudf, std::string cache_dir, bool dump_plans,
                   bool use_async);

/**
 * @brief Gets the use cudf flag.
 *
 * @return bool that is true if use of cudf in backend is enabled
 */
bool get_use_cudf();

/**
 * @brief Gets the Bodo cache directory.
 *
 * @return std::string the Bodo cache directory
 */
std::string get_cache_dir();

/**
 * @brief Gets the dump plans flag.
 *
 * @return bool that is true if the user wants us to dump plan details.
 */
bool get_dump_plans();
