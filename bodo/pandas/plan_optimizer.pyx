"""Provides wrappers around DuckDB's nodes and optimizer for use in Python.
"""
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.memory cimport unique_ptr, make_unique, dynamic_pointer_cast
from libcpp.utility cimport move, pair
from libcpp.string cimport string as c_string
from libcpp.vector cimport vector
from libcpp cimport bool as c_bool
import operator
import datetime
from libc.stdint cimport int64_t, uint64_t, int32_t
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

import pyarrow as pa
import bodo

from cpython.ref cimport PyObject
ctypedef PyObject* PyObjectPtr
ctypedef unsigned long long idx_t
ctypedef pair[int, int] int_pair

cdef extern from "duckdb/common/types.hpp" namespace "duckdb" nogil:
    cpdef enum class CLogicalTypeId "duckdb::LogicalTypeId":
        INVALID "duckdb::LogicalTypeId::INVALID"
        SQLNULL "duckdb::LogicalTypeId::SQLNULL"
        UNKNOWN "duckdb::LogicalTypeId::UNKNOWN"
        ANY "duckdb::LogicalTypeId::ANY"
        BOOLEAN "duckdb::LogicalTypeId::BOOLEAN"
        TINYINT "duckdb::LogicalTypeId::TINYINT"
        SMALLINT "duckdb::LogicalTypeId::SMALLINT"
        INTEGER "duckdb::LogicalTypeId::INTEGER"
        BIGINT "duckdb::LogicalTypeId::BIGINT"
        DATE "duckdb::LogicalTypeId::DATE"
        TIME "duckdb::LogicalTypeId::TIME"
        TIMESTAMP_SEC "duckdb::LogicalTypeId::TIMESTAMP_SEC"
        TIMESTAMP_MS "duckdb::LogicalTypeId::TIMESTAMP_MS"
        TIMESTAMP "duckdb::LogicalTypeId::TIMESTAMP"
        TIMESTAMP_NS "duckdb::LogicalTypeId::TIMESTAMP_NS"
        DECIMAL "duckdb::LogicalTypeId::DECIMAL"
        FLOAT "duckdb::LogicalTypeId::FLOAT"
        DOUBLE "duckdb::LogicalTypeId::DOUBLE"
        CHAR "duckdb::LogicalTypeId::CHAR"
        VARCHAR "duckdb::LogicalTypeId::VARCHAR"
        BLOB "duckdb::LogicalTypeId::BLOB"
        INTERVAL "duckdb::LogicalTypeId::INTERVAL"
        UTINYINT "duckdb::LogicalTypeId::UTINYINT"
        USMALLINT "duckdb::LogicalTypeId::USMALLINT"
        UINTEGER "duckdb::LogicalTypeId::UINTEGER"
        UBIGINT "duckdb::LogicalTypeId::UBIGINT"
        TIMESTAMP_TZ "duckdb::LogicalTypeId::TIMESTAMP_TZ"
        TIME_TZ "duckdb::LogicalTypeId::TIME_TZ"
        BIT "duckdb::LogicalTypeId::BIT"
        STRING_LITERAL "duckdb::LogicalTypeId::STRING_LITERAL"
        INTEGER_LITERAL "duckdb::LogicalTypeId::INTEGER_LITERAL"
        UHUGEINT "duckdb::LogicalTypeId::UHUGEINT"
        HUGEINT "duckdb::LogicalTypeId::HUGEINT"
        POINTER "duckdb::LogicalTypeId::POINTER"
        VALIDITY "duckdb::LogicalTypeId::VALIDITY"
        UUID "duckdb::LogicalTypeId::UUID"
        STRUCT "duckdb::LogicalTypeId::STRUCT"
        LIST "duckdb::LogicalTypeId::LIST"
        MAP "duckdb::LogicalTypeId::MAP"
        TABLE "duckdb::LogicalTypeId::TABLE"
        ENUM "duckdb::LogicalTypeId::ENUM"
        AGGREGATE_STATE "duckdb::LogicalTypeId::AGGREGATE_STATE"
        LAMBDA "duckdb::LogicalTypeId::LAMBDA"
        UNION "duckdb::LogicalTypeId::UNION"
        ARRAY "duckdb::LogicalTypeId::ARRAY"

cdef extern from "duckdb/common/types.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalType "duckdb::LogicalType":
        CLogicalType(CLogicalTypeId)

cdef extern from "duckdb/common/enums/expression_type.hpp" namespace "duckdb" nogil:
    cpdef enum class CExpressionType "duckdb::ExpressionType":
        INVALID "duckdb::ExpressionType::INVALID"
        OPERATOR_CAST "duckdb::ExpressionType::OPERATOR_CAST"
        OPERATOR_NOT "duckdb::ExpressionType::OPERATOR_NOT"
        OPERATOR_IS_NULL "duckdb::ExpressionType::OPERATOR_IS_NULL"
        OPERATOR_IS_NOT_NULL "duckdb::ExpressionType::OPERATOR_IS_NOT_NULL"
        OPERATOR_UNPACK "duckdb::ExpressionType::OPERATOR_UNPACK"
        COMPARE_EQUAL "duckdb::ExpressionType::COMPARE_EQUAL"
        COMPARE_BOUNDARY_START "duckdb::ExpressionType::COMPARE_BOUNDARY_START"
        COMPARE_NOTEQUAL "duckdb::ExpressionType::COMPARE_NOTEQUAL"
        COMPARE_LESSTHAN "duckdb::ExpressionType::COMPARE_LESSTHAN"
        COMPARE_GREATERTHAN "duckdb::ExpressionType::COMPARE_GREATERTHAN"
        COMPARE_LESSTHANOREQUALTO "duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO"
        COMPARE_GREATERTHANOREQUALTO "duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO"
        COMPARE_IN "duckdb::ExpressionType::COMPARE_IN"
        COMPARE_NOT_IN "duckdb::ExpressionType::COMPARE_NOT_IN"
        COMPARE_DISTINCT_FROM "duckdb::ExpressionType::COMPARE_DISTINCT_FROM"
        COMPARE_BETWEEN "duckdb::ExpressionType::COMPARE_BETWEEN"
        COMPARE_NOT_BETWEEN "duckdb::ExpressionType::COMPARE_NOT_BETWEEN"
        COMPARE_NOT_DISTINCT_FROM "duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM"
        COMPARE_BOUNDARY_END "duckdb::ExpressionType::COMPARE_BOUNDARY_END"
        CONJUNCTION_AND "duckdb::ExpressionType::CONJUNCTION_AND"
        CONJUNCTION_OR "duckdb::ExpressionType::CONJUNCTION_OR"
        VALUE_CONSTANT "duckdb::ExpressionType::VALUE_CONSTANT"
        VALUE_PARAMETER "duckdb::ExpressionType::VALUE_PARAMETER"
        VALUE_TUPLE "duckdb::ExpressionType::VALUE_TUPLE"
        VALUE_TUPLE_ADDRESS "duckdb::ExpressionType::VALUE_TUPLE_ADDRESS"
        VALUE_NULL "duckdb::ExpressionType::VALUE_NULL"
        VALUE_VECTOR "duckdb::ExpressionType::VALUE_VECTOR"
        VALUE_SCALAR "duckdb::ExpressionType::VALUE_SCALAR"
        VALUE_DEFAULT "duckdb::ExpressionType::VALUE_DEFAULT"
        AGGREGATE "duckdb::ExpressionType::AGGREGATE"
        BOUND_AGGREGATE "duckdb::ExpressionType::BOUND_AGGREGATE"
        GROUPING_FUNCTION "duckdb::ExpressionType::GROUPING_FUNCTION"
        WINDOW_AGGREGATE "duckdb::ExpressionType::WINDOW_AGGREGATE"
        WINDOW_RANK "duckdb::ExpressionType::WINDOW_RANK"
        WINDOW_RANK_DENSE "duckdb::ExpressionType::WINDOW_RANK_DENSE"
        WINDOW_NTILE "duckdb::ExpressionType::WINDOW_NTILE"
        WINDOW_PERCENT_RANK "duckdb::ExpressionType::WINDOW_PERCENT_RANK"
        WINDOW_CUME_DIST "duckdb::ExpressionType::WINDOW_CUME_DIST"
        WINDOW_ROW_NUMBER "duckdb::ExpressionType::WINDOW_ROW_NUMBER"
        WINDOW_FIRST_VALUE "duckdb::ExpressionType::WINDOW_FIRST_VALUE"
        WINDOW_LAST_VALUE "duckdb::ExpressionType::WINDOW_LAST_VALUE"
        WINDOW_LEAD "duckdb::ExpressionType::WINDOW_LEAD"
        WINDOW_LAG "duckdb::ExpressionType::WINDOW_LAG"
        WINDOW_NTH_VALUE "duckdb::ExpressionType::WINDOW_NTH_VALUE"
        FUNCTION "duckdb::ExpressionType::FUNCTION"
        BOUND_FUNCTION "duckdb::ExpressionType::BOUND_FUNCTION"
        CASE_EXPR "duckdb::ExpressionType::CASE_EXPR"
        OPERATOR_NULLIF "duckdb::ExpressionType::OPERATOR_NULLIF"
        OPERATOR_COALESCE "duckdb::ExpressionType::OPERATOR_COALESCE"
        ARRAY_EXTRACT "duckdb::ExpressionType::ARRAY_EXTRACT"
        ARRAY_SLICE "duckdb::ExpressionType::ARRAY_SLICE"
        STRUCT_EXTRACT "duckdb::ExpressionType::STRUCT_EXTRACT"
        ARRAY_CONSTRUCTOR "duckdb::ExpressionType::ARRAY_CONSTRUCTOR"
        ARROW "duckdb::ExpressionType::ARROW"
        OPERATOR_TRY "duckdb::ExpressionType::OPERATOR_TRY"
        SUBQUERY "duckdb::ExpressionType::SUBQUERY"
        STAR "duckdb::ExpressionType::STAR"
        TABLE_STAR "duckdb::ExpressionType::TABLE_STAR"
        PLACEHOLDER "duckdb::ExpressionType::PLACEHOLDER"
        COLUMN_REF "duckdb::ExpressionType::COLUMN_REF"
        FUNCTION_REF "duckdb::ExpressionType::FUNCTION_REF"
        TABLE_REF "duckdb::ExpressionType::TABLE_REF"
        LAMBDA_REF "duckdb::ExpressionType::LAMBDA_REF"
        CAST "duckdb::ExpressionType::CAST"
        BOUND_REF "duckdb::ExpressionType::BOUND_REF"
        BOUND_COLUMN_REF "duckdb::ExpressionType::BOUND_COLUMN_REF"
        BOUND_UNNEST "duckdb::ExpressionType::BOUND_UNNEST"
        COLLATE "duckdb::ExpressionType::COLLATE"
        LAMBDA "duckdb::ExpressionType::LAMBDA"
        POSITIONAL_REFERENCE "duckdb::ExpressionType::POSITIONAL_REFERENCE"
        BOUND_LAMBDA_REF "duckdb::ExpressionType::BOUND_LAMBDA_REF"
        BOUND_EXPANDED "duckdb::ExpressionType::BOUND_EXPANDED"

def str_to_expr_type(val):
    if val is operator.eq:
        return CExpressionType.COMPARE_EQUAL
    elif val is operator.ne:
        return CExpressionType.COMPARE_NOTEQUAL
    elif val is operator.gt:
        return CExpressionType.COMPARE_GREATERTHAN
    elif val is operator.lt:
        return CExpressionType.COMPARE_LESSTHAN
    elif val is operator.ge:
        return CExpressionType.COMPARE_GREATERTHANOREQUALTO
    elif val is operator.le:
        return CExpressionType.COMPARE_LESSTHANOREQUALTO
    elif val == "__and__":
        return CExpressionType.CONJUNCTION_AND
    elif val == "__or__":
        return CExpressionType.CONJUNCTION_OR
    elif val == "__invert__":
        return CExpressionType.OPERATOR_NOT
    elif val == "notnull":
        return CExpressionType.OPERATOR_IS_NOT_NULL
    else:
        raise NotImplementedError(f"Unhandled case {str(val)} in str_to_expr_type")

cdef extern from "duckdb/common/enums/expression_type.hpp" namespace "duckdb" nogil:
    cpdef enum class CExpressionClass "duckdb::ExpressionClass":
        INVALID "duckdb::ExpressionType::INVALID"
        AGGREGATE "duckdb::ExpressionType::AGGREGATE"
        CASE "duckdb::ExpressionType::CASE"
        CAST "duckdb::ExpressionType::CAST"
        COLUMN_REF "duckdb::ExpressionType::COLUMN_REF"
        COMPARISON "duckdb::ExpressionType::COMPARISON"
        CONJUNCTION "duckdb::ExpressionType::CONJUNCTION"
        CONSTANT "duckdb::ExpressionType::CONSTANT"
        DEFAULT "duckdb::ExpressionType::DEFAULT"
        FUNCTION "duckdb::ExpressionType::FUNCTION"
        OPERATOR "duckdb::ExpressionType::OPERATOR"
        STAR "duckdb::ExpressionType::STAR"
        SUBQUERY "duckdb::ExpressionType::SUBQUERY"
        WINDOW "duckdb::ExpressionType::WINDOW"
        PARAMETER "duckdb::ExpressionType::PARAMETER"
        COLLATE "duckdb::ExpressionType::COLLATE"
        LAMBDA "duckdb::ExpressionType::LAMBDA"
        POSITIONAL_REFERENCE "duckdb::ExpressionType::POSITIONAL_REFERENCE"
        BETWEEN "duckdb::ExpressionType::BETWEEN"
        LAMBDA_REF "duckdb::ExpressionType::LAMBDA_REF"
        BOUND_AGGREGATE "duckdb::ExpressionType::BOUND_AGGREGATE"
        BOUND_CASE "duckdb::ExpressionType::BOUND_CASE"
        BOUND_CAST "duckdb::ExpressionType::BOUND_CAST"
        BOUND_COLUMN_REF "duckdb::ExpressionType::BOUND_COLUMN_REF"
        BOUND_COMPARISON "duckdb::ExpressionType::BOUND_COMPARISON"
        BOUND_CONJUNCTION "duckdb::ExpressionType::BOUND_CONJUNCTION"
        BOUND_CONSTANT "duckdb::ExpressionType::BOUND_CONSTANT"
        BOUND_DEFAULT "duckdb::ExpressionType::BOUND_DEFAULT"
        BOUND_FUNCTION "duckdb::ExpressionType::BOUND_FUNCTION"
        BOUND_OPERATOR "duckdb::ExpressionType::BOUND_OPERATOR"
        BOUND_PARAMETER "duckdb::ExpressionType::BOUND_PARAMETER"
        BOUND_REF "duckdb::ExpressionType::BOUND_REF"
        BOUND_SUBQUERY "duckdb::ExpressionType::BOUND_SUBQUERY"
        BOUND_WINDOW "duckdb::ExpressionType::BOUND_WINDOW"
        BOUND_BETWEEN "duckdb::ExpressionType::BOUND_BETWEEN"
        BOUND_UNNEST "duckdb::ExpressionType::BOUND_UNNEST"
        BOUND_LAMBDA "duckdb::ExpressionType::BOUND_LAMBDA"
        BOUND_LAMBDA_REF "duckdb::ExpressionType::BOUND_LAMBDA_REF"
        BOUND_EXPRESSION "duckdb::ExpressionType::BOUND_EXPRESSION"
        BOUND_EXPANDED "duckdb::ExpressionType::BOUND_EXPANDED"

cdef extern from "duckdb/parser/base_expression.hpp" namespace "duckdb" nogil:
    cdef cppclass CBaseExpression "duckdb::BaseExpression":
        CBaseExpression(CExpressionType type, CExpressionClass expression_class)
        CExpressionType type
        CExpressionClass expression_class

cdef extern from "duckdb/planner/expression.hpp" namespace "duckdb" nogil:
    cdef cppclass CExpression "duckdb::Expression"(CBaseExpression):
        CExpression(CExpressionType type, CExpressionClass expression_class, CLogicalType return_type)
        CLogicalType return_type

cdef extern from "duckdb/planner/bound_result_modifier.hpp" namespace "duckdb" nogil:
    cdef cppclass CBoundOrderByNode "duckdb::BoundOrderByNode":
        pass

cdef extern from "duckdb/planner/logical_operator.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalOperator "duckdb::LogicalOperator":
        vector[unique_ptr[CLogicalOperator]] children
        idx_t estimated_cardinality
        bint has_estimated_cardinality

cdef extern from "duckdb/common/enums/join_type.hpp" namespace "duckdb" nogil:
    cpdef enum class CJoinType "duckdb::JoinType":
        INVALID "duckdb::JoinType::INVALID"
        LEFT "duckdb::JoinType::LEFT"
        RIGHT "duckdb::JoinType::RIGHT"
        INNER "duckdb::JoinType::INNER"
        OUTER "duckdb::JoinType::OUTER"
        SEMI "duckdb::JoinType::SEMI"
        ANTI "duckdb::JoinType::ANTI"
        MARK "duckdb::JoinType::MARK"
        SINGLE "duckdb::JoinType::SINGLE"
        RIGHT_SEMI "duckdb::JoinType::RIGHT_SEMI"
        RIGHT_ANTI "duckdb::JoinType::RIGHT_ANTI"

cdef extern from "duckdb/planner/operator/logical_materialized_cte.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalMaterializedCTE" duckdb::LogicalMaterializedCTE"(CLogicalOperator):
        idx_t table_index

cdef extern from "duckdb/planner/operator/logical_cteref.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalCTERef" duckdb::LogicalCTERef"(CLogicalOperator):
        idx_t table_index

cdef extern from "duckdb/planner/operator/logical_comparison_join.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalComparisonJoin" duckdb::LogicalComparisonJoin"(CLogicalOperator):
        CJoinType join_type

cdef extern from "duckdb/planner/operator/logical_cross_product.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalCrossProduct" duckdb::LogicalCrossProduct"(CLogicalOperator):
       pass

cdef extern from "duckdb/planner/operator/logical_set_operation.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalSetOperation" duckdb::LogicalSetOperation"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_projection.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalProjection" duckdb::LogicalProjection"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_distinct.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalDistinct" duckdb::LogicalDistinct"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_order.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalOrder" duckdb::LogicalOrder"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_aggregate.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalAggregate" duckdb::LogicalAggregate"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_filter.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalFilter" duckdb::LogicalFilter"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_limit.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalLimit" duckdb::LogicalLimit"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_sample.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalSample" duckdb::LogicalSample"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_get.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalGet" duckdb::LogicalGet"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_copy_to_file.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalCopyToFile" duckdb::LogicalCopyToFile"(CLogicalOperator):
        pass

cdef extern from "_plan.h" nogil:
    cdef cppclass CLogicalJoinFilter" bodo::LogicalJoinFilter"(CLogicalOperator):
        pass

    cdef idx_t getTableIndex() except +
    cdef unique_ptr[CLogicalGet] make_parquet_get_node(object parquet_path, object arrow_schema, object storage_options, int64_t num_rows) except +
    cdef unique_ptr[CLogicalGet] make_dataframe_get_seq_node(object df, object arrow_schema, int64_t num_rows) except +
    cdef unique_ptr[CLogicalGet] make_dataframe_get_parallel_node(c_string res_id, object arrow_schema, int64_t num_rows) except +
    cdef unique_ptr[CLogicalGet] make_iceberg_get_node(object arrow_schema, c_string table_identifier, object pyiceberg_catalog, object iceberg_filter, object iceberg_schema, int64_t snapshot_id, uint64_t table_len_estimate) except +
    cdef unique_ptr[CLogicalMaterializedCTE] make_cte(unique_ptr[CLogicalOperator] duplicated, unique_ptr[CLogicalOperator] uses_duplicated, object out_schema, idx_t table_index) except +
    cdef unique_ptr[CLogicalCTERef] make_cte_ref(object out_schema, idx_t table_index) except +
    cdef unique_ptr[CLogicalComparisonJoin] make_comparison_join(unique_ptr[CLogicalOperator] lhs, unique_ptr[CLogicalOperator] rhs, CJoinType join_type, vector[int_pair] cond_vec, int join_id) except +
    cdef unique_ptr[CLogicalJoinFilter] make_join_filter(unique_ptr[CLogicalOperator] source, vector[int] join_filter_ids, vector[vector[int64_t]] equality_filter_columns, vector[vector[c_bool]] equality_is_first_locations, vector[vector[int64_t]] orig_build_key_cols) except +
    cdef unique_ptr[CLogicalCrossProduct] make_cross_product(unique_ptr[CLogicalOperator] lhs, unique_ptr[CLogicalOperator] rhs) except +
    cdef unique_ptr[CLogicalSetOperation] make_set_operation(unique_ptr[CLogicalOperator] lhs, unique_ptr[CLogicalOperator] rhs, c_string setop, int64_t num_cols) except +
    cdef unique_ptr[CLogicalOperator] optimize_plan(unique_ptr[CLogicalOperator]) except +
    cdef unique_ptr[CLogicalProjection] make_projection(unique_ptr[CLogicalOperator] source, vector[unique_ptr[CExpression]] expr_vec, object out_schema) except +
    cdef unique_ptr[CLogicalDistinct] make_distinct(unique_ptr[CLogicalOperator] source, vector[unique_ptr[CExpression]] expr_vec, object out_schema) except +
    cdef unique_ptr[CLogicalOrder] make_order(unique_ptr[CLogicalOperator] source, vector[c_bool] asc, vector[c_bool] na_position, vector[int] cols, object in_schema) except +
    cdef unique_ptr[CLogicalAggregate] make_aggregate(unique_ptr[CLogicalOperator] source, vector[int] key_indices, vector[unique_ptr[CExpression]] expr_vec, object out_schema) except +
    cdef unique_ptr[CExpression] make_scalar_func_expr(unique_ptr[CLogicalOperator] source, object out_schema, object args, vector[int] input_column_indices, c_bool is_cfunc, c_bool has_state, c_string arrow_compute_func) except +
    cdef unique_ptr[CExpression] make_comparison_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, CExpressionType etype) except +
    cdef unique_ptr[CExpression] make_arithop_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, c_string opstr, object out_schema) except +
    cdef unique_ptr[CExpression] make_unaryop_expr(unique_ptr[CExpression] source, c_string opstr) except +
    cdef unique_ptr[CExpression] make_cast_expr(unique_ptr[CExpression] source, object out_schema) except +
    cdef unique_ptr[CExpression] make_conjunction_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, CExpressionType etype) except +
    cdef unique_ptr[CExpression] make_unary_expr(unique_ptr[CExpression] lhs, CExpressionType etype) except +
    cdef unique_ptr[CExpression] make_case_expr(unique_ptr[CExpression] when, unique_ptr[CExpression] then, unique_ptr[CExpression] else_) except +
    cdef unique_ptr[CLogicalFilter] make_filter(unique_ptr[CLogicalOperator] source, unique_ptr[CExpression] filter_expr) except +
    cdef unique_ptr[CExpression] make_const_null(object arrow_schema, int64_t field_idx) except +
    cdef unique_ptr[CExpression] make_const_int_expr(int64_t val) except +
    cdef unique_ptr[CExpression] make_const_double_expr(double val) except +
    cdef unique_ptr[CExpression] make_const_timestamp_ns_expr(int64_t val) except +
    cdef unique_ptr[CExpression] make_const_date32_expr(int32_t val) except +
    cdef unique_ptr[CExpression] make_const_string_expr(c_string val) except +
    cdef unique_ptr[CExpression] make_const_bool_expr(c_bool val) except +
    cdef unique_ptr[CExpression] make_col_ref_expr(unique_ptr[CLogicalOperator] source, object field, int col_idx) except +
    cdef unique_ptr[CExpression] make_agg_expr(unique_ptr[CLogicalOperator] source, object out_schema, c_string function_name, object py_udf_args, vector[int] input_column_indices, c_bool dropna) except +
    cdef unique_ptr[CLogicalCopyToFile] make_parquet_write_node(unique_ptr[CLogicalOperator] source, object out_schema, c_string path, c_string compression, c_string bucket_region, int64_t row_group_size) except +
    cdef unique_ptr[CLogicalCopyToFile] make_iceberg_write_node(unique_ptr[CLogicalOperator] source, object out_schema, c_string table_loc,
        c_string bucket_region, int64_t max_pq_chunksize, c_string compression, object partition_tuples, object sort_tuples, c_string iceberg_schema_str,
        object output_pa_schema, object fs) except +
    cdef unique_ptr[CLogicalCopyToFile] make_s3_vectors_write_node(unique_ptr[CLogicalOperator] source, object out_schema, c_string vector_bucket_name,
        c_string index_name, object region) except +
    cdef unique_ptr[CLogicalLimit] make_limit(unique_ptr[CLogicalOperator] source, int n) except +
    cdef unique_ptr[CLogicalSample] make_sample(unique_ptr[CLogicalOperator] source, int n) except +
    cdef pair[int64_t, PyObjectPtr] execute_plan(unique_ptr[CLogicalOperator], object out_schema) except +
    cdef c_string plan_to_string(unique_ptr[CLogicalOperator], c_bool graphviz_format) except +
    cdef vector[int] get_projection_pushed_down_columns(unique_ptr[CLogicalOperator] proj) except +
    cdef int planCountNodes(unique_ptr[CLogicalOperator] root) except +
    cdef int64_t pyarrow_to_cpp_table(object arrow_table) except +
    cdef int64_t pyarrow_array_to_cpp_table(object arrow_array, c_string name, int64_t in_cpp_table) except +
    cdef object cpp_table_to_pyarrow_array(int64_t cpp_table) except +
    cdef c_string cpp_table_get_first_field_name(int64_t cpp_table) except +
    cdef object cpp_table_to_pyarrow(int64_t cpp_table, c_bool delete_cpp_table) except +
    cdef void cpp_table_delete(int64_t cpp_table) except +
    cdef void set_use_cudf(c_bool use_cudf, c_string cache_dir) except +


def join_type_to_string(CJoinType join_type):
    """
    Convert a CJoinType enum to a string representation for printing purposes.
    """
    if join_type == CJoinType.INVALID:
        return "INVALID"
    elif join_type == CJoinType.LEFT:
        return "LEFT"
    elif join_type == CJoinType.RIGHT:
        return "RIGHT"
    elif join_type == CJoinType.INNER:
        return "INNER"
    elif join_type == CJoinType.OUTER:
        return "OUTER"
    elif join_type == CJoinType.SEMI:
        return "SEMI"
    elif join_type == CJoinType.ANTI:
        return "ANTI"
    elif join_type == CJoinType.MARK:
        return "MARK"
    elif join_type == CJoinType.SINGLE:
        return "SINGLE"
    elif join_type == CJoinType.RIGHT_SEMI:
        return "RIGHT_SEMI"
    elif join_type == CJoinType.RIGHT_ANTI:
        return "RIGHT_ANTI"
    else:
        raise NotImplementedError("Unknown Join Type")


cdef class LogicalOperator:
    """Wrapper around DuckDB's LogicalOperator to provide access in Python.
    """
    cdef unique_ptr[CLogicalOperator] c_logical_operator
    cdef readonly out_schema
    cdef public list sources

    def __str__(self):
        return "LogicalOperator()"

    def set_estimated_cardinality(self, estimated_cardinality):
        self.c_logical_operator.get().has_estimated_cardinality = True
        self.c_logical_operator.get().estimated_cardinality = estimated_cardinality

    def toGraphviz(self):
        return plan_to_string(self.c_logical_operator, True).decode("utf-8")

    def toString(self):
        return plan_to_string(self.c_logical_operator, False).decode("utf-8")

    def getCardinality(self):
        return None

cdef class Expression:
    """Wrapper around DuckDB's Expression to provide access in Python.
    """
    cdef readonly out_schema
    cdef unique_ptr[CExpression] c_expression

    def __str__(self):
        return "Expression()"


cdef class LogicalMaterializedCTE(LogicalOperator):
    """Wrapper around DuckDB's LogicalMaterializedCTE to provide access in Python.
    """

    def __cinit__(self, out_schema, LogicalOperator duplicated, LogicalOperator uses_duplicated, idx_t table_index):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalMaterializedCTE] c_logical_cte = make_cte(duplicated.c_logical_operator, uses_duplicated.c_logical_operator, out_schema, table_index)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_cte.release())

    def __str__(self):
        return f"LogicalMaterializedCTE()"


cdef class LogicalCTERef(LogicalOperator):
    """Wrapper around DuckDB's LogicalCTERef to provide access in Python.
    """

    def __cinit__(self, out_schema, idx_t table_index):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalCTERef] c_logical_cte_ref = make_cte_ref(out_schema, table_index)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_cte_ref.release())

    def __str__(self):
        return f"LogicalCTERef()"


cdef class LogicalComparisonJoin(LogicalOperator):
    """Wrapper around DuckDB's LogicalComparisonJoin to provide access in Python.
    """

    def __cinit__(self, out_schema, LogicalOperator lhs, LogicalOperator rhs, CJoinType join_type, conditions, int join_id=-1):
        self.out_schema = out_schema
        cdef vector[int_pair] cond_vec
        for cond in conditions:
            cond_vec.push_back(int_pair(cond[0], cond[1]))

        cdef unique_ptr[CLogicalComparisonJoin] c_logical_comparison_join = make_comparison_join(lhs.c_logical_operator, rhs.c_logical_operator, join_type, cond_vec, join_id)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_comparison_join.release())

    def __str__(self):
        join_type = join_type_to_string((<CLogicalComparisonJoin*>(self.c_logical_operator.get())).join_type)
        return f"LogicalComparisonJoin({join_type})"

cdef class LogicalCrossProduct(LogicalOperator):
    """Wrapper around DuckDB's LogicalCrossProduct to provide access in Python.
    """

    def __cinit__(self, out_schema, LogicalOperator lhs, LogicalOperator rhs):
        self.out_schema = out_schema

        cdef unique_ptr[CLogicalCrossProduct] c_logical_cross_product = make_cross_product(lhs.c_logical_operator, rhs.c_logical_operator)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_cross_product.release())

    def __str__(self):
        return f"LogicalCrossProduct()"

cdef class LogicalInsertScalarSubquery(LogicalCrossProduct):
    """
    Wrapper around LogicalCrossProduct to represent an insert of a scalar subquery result.
    """
    def __str__(self):
        return f"LogicalInsertScalarSubquery()"

cdef class LogicalSetOperation(LogicalOperator):
    """Wrapper around DuckDB's LogicalSetOperation to provide access in Python.
    """

    def __cinit__(self, out_schema, LogicalOperator lhs, LogicalOperator rhs, str setop):
        """
        setop - only value supported for now is "union all".  In the future,
                "union" and "intersect" may be supported.
        """
        self.out_schema = out_schema

        cdef unique_ptr[CLogicalSetOperation] c_logical_set_operation = make_set_operation(lhs.c_logical_operator, rhs.c_logical_operator, setop.encode(), len(self.out_schema))
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_set_operation.release())

    def __str__(self):
        return f"LogicalSetOperation()"


cdef class LogicalProjection(LogicalOperator):
    """Wrapper around DuckDB's LogicalProjection to provide access in Python.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, object exprs):
        cdef vector[unique_ptr[CExpression]] expr_vec

        for expr in exprs:
            expr_vec.push_back(move((<Expression>expr).c_expression))

        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalProjection] c_logical_projection = make_projection(source.c_logical_operator, expr_vec, out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_projection.release())

    def __str__(self):
        return f"LogicalProjection({self.out_schema})"

    def getCardinality(self):
        return self.sources[0].getCardinality()


cdef class LogicalDistinct(LogicalOperator):
    """Wrapper around DuckDB's LogicalDistinct to provide access in Python.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, object exprs):
        cdef vector[unique_ptr[CExpression]] expr_vec

        for expr in exprs:
            expr_vec.push_back(move((<Expression>expr).c_expression))

        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalDistinct] c_logical_distinct = make_distinct(source.c_logical_operator, expr_vec, out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_distinct.release())

    def __str__(self):
        return f"LogicalDistinct({self.out_schema})"

    def getCardinality(self):
        return None


cdef class LogicalAggregate(LogicalOperator):
    """Wrapper around DuckDB's LogicalAggregate to provide access in Python.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, vector[int] key_indices, object exprs):
        cdef vector[unique_ptr[CExpression]] expr_vec

        for expr in exprs:
            expr_vec.push_back(move((<Expression>expr).c_expression))

        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalAggregate] c_logical_projection = make_aggregate(source.c_logical_operator, key_indices, expr_vec, out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_projection.release())

    def __str__(self):
        return f"LogicalAggregate({self.out_schema})"


cdef class LogicalOrder(LogicalOperator):
    """Wrapper around DuckDB's LogicalOrder to provide access in Python.
    """

    def __cinit__(self,
                  object out_schema,
                  LogicalOperator source,
                  vector[c_bool] asc,
                  vector[c_bool] na_position,
                  vector[int] cols,
                  object in_schema):
        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalOrder] c_logical_order = make_order(source.c_logical_operator, asc, na_position, cols, in_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_order.release())

    def __str__(self):
        return f"LogicalOrder({self.out_schema})"

    def getCardinality(self):
        return self.sources[0].getCardinality()


cpdef get_pushed_down_columns(proj):
    """Get column indices that are pushed down from projection to its source node. Used for testing.
    """
    cdef LogicalOperator wrapped_operator = proj
    cdef vector[int] pushed_down_columns = get_projection_pushed_down_columns(wrapped_operator.c_logical_operator)
    return pushed_down_columns


cdef class ColRefExpression(Expression):
    """Wrapper around DuckDB's BoundColumnRefExpression to provide access in Python.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, int col_index):
        self.out_schema = out_schema
        self.c_expression = make_col_ref_expr(source.c_logical_operator, out_schema[0], col_index)

    def __str__(self):
        return f"ColRefExpression({self.out_schema})"


cdef class NullExpression(Expression):
    """Wrapper around DuckDB's BoundConstantExpression to provide access in Python.
    """

    def __cinit__(self, object dummy_schema, int64_t field_idx):
        self.c_expression = make_const_null(dummy_schema, field_idx)

    def __str__(self):
        return f"NullExpression()"


cdef class ConstantExpression(Expression):
    """Wrapper around DuckDB's BoundConstantExpression to provide access in Python.
    """

    def __cinit__(self, object dummy_schema, object value):
        self.c_expression = make_const_expr(value)

    def __str__(self):
        return f"ConstantExpression()"


cdef class AggregateExpression(Expression):
    """Wrapper around DuckDB's AggregateExpression to provide access in Python.
    """
    cdef readonly str function_name

    def __cinit__(self, object out_schema, LogicalOperator source, str function_name, object udf_args, vector[int] input_column_indices, c_bool dropna):
        self.out_schema = out_schema
        self.function_name = function_name
        self.c_expression = make_agg_expr(source.c_logical_operator, out_schema, function_name.encode(), udf_args, input_column_indices, dropna)

    def __str__(self):
        return f"AggregateExpression({self.function_name})"

cdef class PythonScalarFuncExpression(Expression):
    """Wrapper around DuckDB's BoundFunctionExpression for running Python/Arrow functions.
    """

    def __cinit__(self,
        object out_schema,
        LogicalOperator source,
        object args,
        vector[int] input_column_indices,
        c_bool is_cfunc,
        c_bool has_state):

        self.out_schema = out_schema
        empty_str = ""
        self.c_expression = make_scalar_func_expr(
            source.c_logical_operator, out_schema, args, input_column_indices, is_cfunc, has_state, empty_str.encode())

    def __str__(self):
        return f"PythonScalarFuncExpression({self.function_name})"


cdef class ArrowScalarFuncExpression(Expression):
    """Wrapper around DuckDB's BoundFunctionExpression for running Python/Arrow functions.
    """

    def __cinit__(self,
        object out_schema,
        LogicalOperator source,
        vector[int] input_column_indices,
        str function_name,
        object args):

        self.out_schema = out_schema
        self.c_expression = make_scalar_func_expr(
            source.c_logical_operator, out_schema, args, input_column_indices, False, False, function_name.encode())

    def __str__(self):
        return f"ArrowScalarFuncExpression({self.function_name})"


cdef unique_ptr[CExpression] make_const_expr(val):
    """Convert a filter expression tree from Cython wrappers
       to duckdb.
    """
    # TODO: support other scalar types
    # See pandas scalars in pd.api.types.is_scalar
    cdef c_string val_cstr

    if isinstance(val, (int, np.int64)):
        return move(make_const_int_expr(val))
    elif isinstance(val, float):
        return move(make_const_double_expr(val))
    elif isinstance(val, str):
        val_cstr = val.encode()
        return move(make_const_string_expr(val_cstr))
    elif isinstance(val, bool):
        return move(make_const_bool_expr(val))
    elif isinstance(val, pd.Timestamp):
        # NOTE: Timestamp.value always converts to nanoseconds
        # https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/_libs/tslibs/timestamps.pyx#L242
        return move(make_const_timestamp_ns_expr(val.value))
    elif isinstance(val, (datetime.datetime, datetime.date)):
        return move(make_const_timestamp_ns_expr(pd.Timestamp(val).value))
    elif isinstance(val, pa.Date32Scalar):
        return move(make_const_date32_expr(val.value))
    elif isinstance(val, bodo.pandas.scalar.BodoScalar):
        return move(make_const_expr(val.get_value()))
    else:
        raise NotImplementedError("Unknown expr type in make_const_expr " + str(type(val)))


cdef class LogicalFilter(LogicalOperator):
    def __cinit__(self, out_schema, LogicalOperator source, Expression key):
        self.out_schema = out_schema
        self.sources = [source]
        cdef unique_ptr[CLogicalFilter] c_logical_filter = make_filter(source.c_logical_operator, key.c_expression)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_filter.release())

    def __str__(self):
        return f"LogicalFilter()"


cdef class ComparisonOpExpression(Expression):
    """Wrapper around DuckDB's BoundComparisonExpression and other binary operators to provide access in Python.
    """

    def __cinit__(self, object out_schema, lhs, rhs, binop):
        cdef unique_ptr[CExpression] lhs_expr
        cdef unique_ptr[CExpression] rhs_expr

        lhs_expr = move((<Expression>lhs).c_expression) if isinstance(lhs, Expression) else move(make_const_expr(lhs))
        rhs_expr = move((<Expression>rhs).c_expression) if isinstance(rhs, Expression) else move(make_const_expr(rhs))

        self.out_schema = out_schema
        self.c_expression = make_comparison_expr(
            lhs_expr,
            rhs_expr,
            str_to_expr_type(binop))

    def __str__(self):
        return f"ComparisonOpExpression({self.out_schema})"


def python_arith_dunder_to_duckdb(str opstr):
    """
    Convert a Python arithmetic dunder method name to duckdb catalog name.
    """
    if opstr == "__add__" or opstr == "__radd__":
        return "+"
    elif opstr == "__sub__" or opstr == "__rsub__":
        return "-"
    elif opstr == "__mul__" or opstr == "__rmul__":
        return "*"
    elif opstr == "__truediv__" or opstr == "__rtruediv__":
        return "/"
    elif opstr == "__mod__" or opstr == "__rmod__":
        return "%"
    elif opstr == "__floordiv__" or opstr == "__rfloordiv__":
        # NOTE: only used for integers since "//" is integer division in duckdb
        return "//"
    else:
        raise NotImplementedError("Unknown Python arith dunder method name")


def is_integer_expr(object expr):
    """
    Check if the expression's output schema is of integer type.
    """
    if isinstance(expr, Expression):
        return pa.types.is_integer(expr.out_schema[0].type)

    if isinstance(expr, int):
        return True

    return False


cdef class ArithOpExpression(Expression):
    """Wrapper around DuckDB's BoundComparisonExpression and other binary operators to provide access in Python.
    """

    def __cinit__(self, object out_schema, lhs, rhs, str opstr):
        cdef unique_ptr[CExpression] lhs_expr
        cdef unique_ptr[CExpression] rhs_expr

        lhs_expr = move((<Expression>lhs).c_expression) if isinstance(lhs, Expression) else move(make_const_expr(lhs))
        rhs_expr = move((<Expression>rhs).c_expression) if isinstance(rhs, Expression) else move(make_const_expr(rhs))

        self.out_schema = out_schema
        # The // operator in Python we have to implement as a truediv followed by a floor.
        # Do the semantics work here for negative divisors?
        if opstr in ["__floordiv__", "__rfloordiv__"] and not (is_integer_expr(lhs) and is_integer_expr(rhs)):
            # "//" is integer division in duckdb, so we need to handle float floor division separately
            truediv_expression = make_arithop_expr(
                lhs_expr,
                rhs_expr,
                "/".encode(),
                self.out_schema)
            unaryop_expr = make_unaryop_expr(truediv_expression, "floor".encode())
            self.c_expression = make_cast_expr(unaryop_expr, self.out_schema)
        else:
            duckdb_op = python_arith_dunder_to_duckdb(opstr)
            self.c_expression = make_arithop_expr(
                lhs_expr,
                rhs_expr,
                duckdb_op.encode(),
                self.out_schema)

    def __str__(self):
        return f"ArithOpExpression({self.out_schema})"


cdef class ConjunctionOpExpression(Expression):
    """Wrapper around DuckDB's BoundConjunctionExpression and other binary operators to provide access in Python.
    """

    def __cinit__(self, object out_schema, lhs, rhs, binop):
        cdef unique_ptr[CExpression] lhs_expr
        cdef unique_ptr[CExpression] rhs_expr

        lhs_expr = move((<Expression>lhs).c_expression) if isinstance(lhs, Expression) else move(make_const_expr(lhs))
        rhs_expr = move((<Expression>rhs).c_expression) if isinstance(rhs, Expression) else move(make_const_expr(rhs))

        self.out_schema = out_schema
        self.c_expression = make_conjunction_expr(
            lhs_expr,
            rhs_expr,
            str_to_expr_type(binop))

    def __str__(self):
        return f"ConjunctionOpExpression({self.out_schema})"


cdef class UnaryOpExpression(Expression):
    def __cinit__(self, object out_schema, source, op):
        cdef unique_ptr[CExpression] source_expr

        source_expr = move((<Expression>source).c_expression) if isinstance(source, Expression) else move(make_const_expr(source))

        self.out_schema = out_schema
        self.c_expression = make_unary_expr(
            source_expr,
            str_to_expr_type(op))

    def __str__(self):
        return f"UnaryOpExpression({self.out_schema})"

cdef class CaseExpression(Expression):
    """Wrapper around DuckDB's BoundCaseExpression to provide access in Python.
    """

    def __cinit__(self, object out_schema, when, then, else_):
        cdef unique_ptr[CExpression] when_expr
        cdef unique_ptr[CExpression] then_expr
        cdef unique_ptr[CExpression] else_expr

        when_expr = move((<Expression>when).c_expression) if isinstance(when, Expression) else move(make_const_expr(when))
        then_expr = move((<Expression>then).c_expression) if isinstance(then, Expression) else move(make_const_expr(then))
        else_expr = move((<Expression>else_).c_expression) if isinstance(else_, Expression) else move(make_const_expr(else_))

        self.out_schema = out_schema
        self.c_expression = make_case_expr(
            when_expr,
            then_expr,
            else_expr
        )

    def __str__(self):
        return f"CaseExpression({self.out_schema})"


cdef class LogicalLimit(LogicalOperator):
    cdef public int n

    def __cinit__(self, out_schema, LogicalOperator source, n):
        self.out_schema = out_schema
        self.sources = [source]
        self.n = n

        cdef unique_ptr[CLogicalLimit] c_logical_limit = make_limit(source.c_logical_operator, n)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_limit.release())

    def __str__(self):
        return f"LogicalLimit(n={self.n})"


cdef class LogicalGetParquetRead(LogicalOperator):
    """Wrapper around DuckDB's LogicalGet for reading Parquet datasets.
    """
    cdef readonly object path
    cdef readonly object storage_options
    cdef readonly int64_t nrows

    def __cinit__(self, object out_schema, object parquet_path, object storage_options):
        from bodo.ext import hdist

        self.out_schema = out_schema
        self.path = parquet_path
        self.storage_options = storage_options
        self.nrows = -1
        cdef int64_t nrows_estimate = hdist.bcast_int64_py_wrapper(self._get_nrows(exact=False) if bodo.get_rank() == 0 else 0)
        cdef unique_ptr[CLogicalGet] c_logical_get = make_parquet_get_node(parquet_path, out_schema, storage_options, nrows_estimate)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())

    def __str__(self):
        return f"LogicalGetParquetRead({self.path})"

    def getCardinality(self):
        from bodo.ext import hdist

        if self.nrows == -1:
            self.nrows = hdist.bcast_int64_py_wrapper(self._get_nrows(exact=True) if bodo.get_rank() == 0 else 0)
        return self.nrows

    def _get_nrows(self, exact : bool = True):
        """ Get the number of rows in the Parquet dataset.
        If exact is False, estimate the number of rows by sampling files.
        """
        from bodo.io.fs_io import (
            expand_path_globs,
            getfs,
            parse_fpath,
        )
        from bodo.io.parquet_pio import get_fpath_without_protocol_prefix

        fpath, parsed_url, protocol = parse_fpath(self.path)
        fs = getfs(fpath, protocol, self.storage_options, parallel=False)

        # Since we are supplying the filesystem to pq.read_table,
        # Any prefixes e.g. s3:// should be removed.
        fpath_noprefix, _ = get_fpath_without_protocol_prefix(
            fpath, protocol, parsed_url
        )

        fpath_noprefix = expand_path_globs(fpath_noprefix, protocol, fs)

        if exact:
            return pq.read_table(fpath_noprefix, filesystem=fs, columns=[]).num_rows

        if isinstance(fpath_noprefix, str):
            fpath_noprefix = [fpath_noprefix]

        # TODO: Make parquet file detection more robust.
        def is_parquet_file(info: pa.fs.FileInfo):
            return info.extension in ["parquet", "pq"]

        files = []

        for path in fpath_noprefix:
            info = fs.get_file_info(path)

            if info.type == pa.fs.FileType.File:
                if is_parquet_file(info):
                    files.append(path)

            elif info.type == pa.fs.FileType.Directory:
                selector = pa.fs.FileSelector(path, recursive=True)
                infos = fs.get_file_info(selector)
                files.extend(
                    i.path for i in infos
                    if i.type == pa.fs.FileType.File
                    and is_parquet_file(i)
                )

        n_files = len(files)
        if n_files == 0:
            return 0

        n_sampled = max(3, int(0.001 * n_files))
        sampled = files[:min(n_sampled, n_files)]

        rows = pq.read_table(sampled, filesystem=fs, columns=[]).num_rows

        return int(rows * (n_files / len(sampled)))


cdef class LogicalGetSeriesRead(LogicalOperator):
    """Represents an already materialized BodoSeries."""
    def __cinit__(self, out_schema, result_id):
        self.out_schema = out_schema
        raise NotImplementedError("LogicalGetSeriesRead not yet implemented.")


cdef class LogicalGetPandasReadSeq(LogicalOperator):
    """Represents sequential scan of a Pandas dataframe passed into from_pandas."""
    cdef readonly object df

    def __cinit__(self, object out_schema, object df):
        self.out_schema = out_schema
        self.df = df
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_seq_node(df, out_schema, self.getCardinality())
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())

    def getCardinality(self):
        return len(self.df)


cdef class LogicalGetPandasReadParallel(LogicalOperator):
    cdef int64_t nrows

    """Represents parallel scan of a Pandas dataframe passed into from_pandas."""
    def __cinit__(self, object out_schema, int64_t nrows, object result_id):
        # result_id could be a string or LazyPlanDistributedArg if we are constructing the
        # plan locally for cardinality.  If so, extract res_id from that object.
        if not isinstance(result_id, str):
            result_id = result_id.res_id
            # Set dummy result_id when not available, which is the case when we are constructing the plan just for cardinality
            # and not for execution.
            if result_id is None:
                result_id = ""
        self.out_schema = out_schema
        self.nrows = nrows
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_parallel_node(result_id.encode(), out_schema, self.getCardinality())
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())

    def getCardinality(self):
        return self.nrows

cdef class LogicalGetIcebergRead(LogicalOperator):
    """
    Wrapper around DuckDB's LogicalGet for reading Iceberg datasets.
    """
    cdef readonly str table_identifier

    def __cinit__(self, object out_schema, str table_identifier, object catalog_name, object catalog_properties, object iceberg_filter, object iceberg_schema, object snapshot_id, uint64_t table_len_estimate):
        import pyiceberg.catalog
        cdef object catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)
        self.out_schema = out_schema
        self.table_identifier = table_identifier
        cdef unique_ptr[CLogicalGet] c_logical_get = make_iceberg_get_node(out_schema, table_identifier.encode(), catalog, iceberg_filter, iceberg_schema, snapshot_id, table_len_estimate)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())

    def __str__(self):
        return f"LogicalGetIcebergRead({self.table_identifier})"


cdef class LogicalParquetWrite(LogicalOperator):
    """
    Wrapper around DuckDB's LogicalCopyToFile for writing Parquet datasets.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, str path, str compression, str bucket_region, int64_t row_group_size):
        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalCopyToFile] c_logical_copy_to_file = make_parquet_write_node(source.c_logical_operator, out_schema, path.encode(), compression.encode(), bucket_region.encode(), row_group_size)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_copy_to_file.release())

    def __str__(self):
        return f"LogicalParquetWrite()"


cdef class LogicalIcebergWrite(LogicalOperator):
    """
    Wrapper around DuckDB's LogicalCopyToFile for writing Iceberg datasets.
    """

    def __cinit__(self, object out_schema, LogicalOperator source,
            str table_loc,
            str bucket_region,
            int64_t max_pq_chunksize,
            str compression,
            object partition_tuples,
            object sort_tuples,
            str iceberg_schema_str,
            object output_pa_schema,
            object fs):
        self.out_schema = out_schema
        self.sources = [source]

        cdef unique_ptr[CLogicalCopyToFile] c_logical_copy_to_file = make_iceberg_write_node(source.c_logical_operator, out_schema, table_loc.encode(),
                bucket_region.encode(), max_pq_chunksize, compression.encode(), partition_tuples, sort_tuples, iceberg_schema_str.encode(), output_pa_schema, fs)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_copy_to_file.release())

    def __str__(self):
        return f"LogicalIcebergWrite()"


cdef class LogicalS3VectorsWrite(LogicalOperator):
    """
    Wrapper around DuckDB's LogicalCopyToFile for writing S3 Vector datasets.
    """
    cdef readonly str vector_bucket_name
    cdef readonly str index_name

    def __cinit__(self, object out_schema, LogicalOperator source,
            str vector_bucket_name,
            str index_name, object region):
        self.out_schema = out_schema
        self.sources = [source]
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name

        cdef unique_ptr[CLogicalCopyToFile] c_logical_copy_to_file = make_s3_vectors_write_node(source.c_logical_operator, out_schema, vector_bucket_name.encode(),
                index_name.encode(), region)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_copy_to_file.release())

    def __str__(self):
        return f"LogicalS3VectorsWrite({self.vector_bucket_name}, {self.index_name})"


cdef class LogicalJoinFilter(LogicalOperator):
    """
    Logical type for Join Filter operations (not part of DuckDB).
    """

    def __cinit__(self, object out_schema, LogicalOperator source,
            vector[int] join_filter_ids,
            vector[vector[int64_t]] equality_filter_columns,
            vector[vector[c_bool]] equality_is_first_locations):
        cdef vector[vector[int64_t]] orig_build_key_cols

        cdef unique_ptr[CLogicalJoinFilter] c_logical_join_filter = make_join_filter(source.c_logical_operator, join_filter_ids, equality_filter_columns, equality_is_first_locations, orig_build_key_cols)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_join_filter.release())

    def __str__(self):
        return f"LogicalJoinFilter({self.join_filter_ids}, {self.equality_filter_columns}, {self.equality_is_first_locations})"


cpdef count_nodes(object root):
    cdef LogicalOperator wrapped_operator

    if not isinstance(root, LogicalOperator):
        raise TypeError("Expected a LogicalOperator instance")

    wrapped_operator = root

    return planCountNodes(wrapped_operator.c_logical_operator)


cpdef arrow_to_cpp_table(arrow_table):
    """Convert an Arrow table to a C++ table pointer with column names and
    metadata set properly.
    """
    return pyarrow_to_cpp_table(arrow_table)


cpdef arrow_array_to_cpp_table(object arr, str name, in_cpp_table):
    """Convert an Arrow array to a C++ table pointer with column names and
    metadata set properly.
    Uses in_cpp_table for appending Index arrays if any and pandas metadata.
    Deletes in_cpp_table after use.
    """
    return pyarrow_array_to_cpp_table(arr, name.encode(), in_cpp_table)


cpdef cpp_table_to_arrow(cpp_table, delete_cpp_table=True):
    """Convert a C++ table pointer to Arrow table.
    """
    return cpp_table_to_pyarrow(cpp_table, delete_cpp_table)


cpdef cpp_table_to_arrow_array(cpp_table, delete_cpp_table=True):
    """Convert the first column of C++ table to Arrow array and column name.
    """
    out = cpp_table_to_pyarrow_array(cpp_table), cpp_table_get_first_field_name(cpp_table).decode()
    if delete_cpp_table:
        cpp_table_delete(cpp_table)
    return out


cpdef c_set_use_cudf(use_cudf, cache_dir):
    set_use_cudf(use_cudf, cache_dir.encode())


cpdef py_optimize_plan(object plan):
    """Optimize a logical plan using DuckDB's optimizer
    """
    cdef LogicalOperator wrapped_operator

    if not isinstance(plan, LogicalOperator):
        raise TypeError("Expected a LogicalOperator instance")

    wrapped_operator = plan

    optimized_plan = LogicalOperator()
    optimized_plan.c_logical_operator = optimize_plan(move(wrapped_operator.c_logical_operator))
    return optimized_plan


cpdef py_execute_plan(object plan, output_func, out_schema):
    """Execute a logical plan in the C++ backend
    """
    cdef LogicalOperator wrapped_operator
    cdef pair[int64_t, PyObjectPtr] exec_output
    cdef int64_t cpp_table

    if not isinstance(plan, LogicalOperator):
        raise TypeError("Expected a LogicalOperator instance")

    wrapped_operator = plan

    exec_output = execute_plan(move(wrapped_operator.c_logical_operator), out_schema)
    cpp_table = exec_output.first

    # Write doesn't return output data
    if cpp_table == 0:
        # Iceberg write returns file information for later commit
        if exec_output.second != NULL:
            return <object>exec_output.second
        return None

    arrow_schema = <object>exec_output.second
    if output_func is None:
        raise ValueError("output_func is None.")
    return output_func(cpp_table, out_schema)

def py_get_table_index():
    """Python-callable wrapper for getTableIndex()."""
    return getTableIndex()
