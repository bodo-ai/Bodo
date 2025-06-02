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
from libc.stdint cimport int64_t
import pandas as pd
import pyarrow.parquet as pq

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
        USER "duckdb::LogicalTypeId::USER"
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
        VARINT "duckdb::LogicalTypeId::VARINT"
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


cdef extern from "duckdb/planner/operator/logical_comparison_join.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalComparisonJoin" duckdb::LogicalComparisonJoin"(CLogicalOperator):
        CLogicalComparisonJoin(CJoinType join_type)
        CJoinType join_type

cdef extern from "duckdb/planner/operator/logical_projection.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalProjection" duckdb::LogicalProjection"(CLogicalOperator):
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


cdef extern from "_plan.h" nogil:
    cdef unique_ptr[CLogicalGet] make_parquet_get_node(object parquet_path, object arrow_schema, object storage_options) except +
    cdef unique_ptr[CLogicalGet] make_dataframe_get_seq_node(object df, object arrow_schema) except +
    cdef unique_ptr[CLogicalGet] make_dataframe_get_parallel_node(c_string res_id, object arrow_schema) except +
    cdef unique_ptr[CLogicalGet] make_iceberg_get_node(object arrow_schema, c_string table_identifier, object pyiceberg_catalog, object iceberg_filter, object iceberg_schema) except +
    cdef unique_ptr[CLogicalComparisonJoin] make_comparison_join(unique_ptr[CLogicalOperator] lhs, unique_ptr[CLogicalOperator] rhs, CJoinType join_type, vector[int_pair] cond_vec) except +
    cdef unique_ptr[CLogicalOperator] optimize_plan(unique_ptr[CLogicalOperator]) except +
    cdef unique_ptr[CLogicalProjection] make_projection(unique_ptr[CLogicalOperator] source, vector[unique_ptr[CExpression]] expr_vec, object out_schema) except +
    cdef unique_ptr[CLogicalAggregate] make_aggregate(unique_ptr[CLogicalOperator] source, vector[int] key_indices, vector[unique_ptr[CExpression]] expr_vec, object out_schema) except +
    cdef unique_ptr[CExpression] make_python_scalar_func_expr(unique_ptr[CLogicalOperator] source, object out_schema, object args, vector[int] input_column_indices) except +
    cdef unique_ptr[CExpression] make_comparison_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, CExpressionType etype) except +
    cdef unique_ptr[CExpression] make_arithop_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, c_string opstr) except +
    cdef unique_ptr[CExpression] make_unaryop_expr(unique_ptr[CExpression] source, c_string opstr) except +
    cdef unique_ptr[CExpression] make_conjunction_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, CExpressionType etype) except +
    cdef unique_ptr[CExpression] make_unary_expr(unique_ptr[CExpression] lhs, CExpressionType etype) except +
    cdef unique_ptr[CLogicalFilter] make_filter(unique_ptr[CLogicalOperator] source, unique_ptr[CExpression] filter_expr) except +
    cdef unique_ptr[CExpression] make_const_int_expr(int64_t val) except +
    cdef unique_ptr[CExpression] make_const_double_expr(double val) except +
    cdef unique_ptr[CExpression] make_const_timestamp_ns_expr(int64_t val) except +
    cdef unique_ptr[CExpression] make_const_string_expr(c_string val) except +
    cdef unique_ptr[CExpression] make_col_ref_expr(unique_ptr[CLogicalOperator] source, object field, int col_idx) except +
    cdef unique_ptr[CExpression] make_agg_expr(unique_ptr[CLogicalOperator] source, object field, c_string function_name, vector[int] input_column_indices) except +
    cdef unique_ptr[CLogicalLimit] make_limit(unique_ptr[CLogicalOperator] source, int n) except +
    cdef unique_ptr[CLogicalSample] make_sample(unique_ptr[CLogicalOperator] source, int n) except +
    cdef pair[int64_t, PyObjectPtr] execute_plan(unique_ptr[CLogicalOperator], object out_schema) except +
    cdef c_string plan_to_string(unique_ptr[CLogicalOperator], c_bool graphviz_format) except +
    cdef vector[int] get_projection_pushed_down_columns(unique_ptr[CLogicalOperator] proj) except +
    cdef int planCountNodes(unique_ptr[CLogicalOperator] root) except +
    cdef void set_table_meta_from_arrow(int64_t table_pointer, object arrow_schema) except +


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


cdef class LogicalComparisonJoin(LogicalOperator):
    """Wrapper around DuckDB's LogicalComparisonJoin to provide access in Python.
    """

    def __cinit__(self, out_schema, LogicalOperator lhs, LogicalOperator rhs, CJoinType join_type, conditions):
        self.out_schema = out_schema
        cdef vector[int_pair] cond_vec
        for cond in conditions:
            cond_vec.push_back(int_pair(cond[0], cond[1]))

        cdef unique_ptr[CLogicalComparisonJoin] c_logical_comparison_join = make_comparison_join(lhs.c_logical_operator, rhs.c_logical_operator, join_type, cond_vec)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_comparison_join.release())

    def __str__(self):
        join_type = join_type_to_string((<CLogicalComparisonJoin*>(self.c_logical_operator.get())).join_type)
        return f"LogicalComparisonJoin({join_type})"


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


cdef class LogicalColRef(LogicalOperator):
    cdef readonly vector[int] select_vec

    def __cinit__(self, object out_schema, LogicalOperator source, select_idxs):
        self.out_schema = out_schema
        self.select_vec = select_idxs
        self.sources = [source]

    def __str__(self):
        return f"LogicalColRef({self.select_vec}, {self.out_schema})"


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

    def __cinit__(self, object out_schema, LogicalOperator source, str function_name, vector[int] input_column_indices):
        self.out_schema = out_schema
        self.function_name = function_name
        self.c_expression = make_agg_expr(source.c_logical_operator, out_schema[0], function_name.encode(), input_column_indices)

    def __str__(self):
        return f"AggregateExpression({self.function_name})"



cdef class PythonScalarFuncExpression(Expression):
    """Wrapper around DuckDB's BoundFunctionExpression for running Python functions.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, object args, vector[int] input_column_indices):
        self.out_schema = out_schema
        self.c_expression = make_python_scalar_func_expr(source.c_logical_operator, out_schema, args, input_column_indices)

    def __str__(self):
        return f"PythonScalarFuncExpression({self.out_schema})"


cdef unique_ptr[CExpression] make_const_expr(val):
    """Convert a filter expression tree from Cython wrappers
       to duckdb.
    """
    # TODO: support other scalar types
    # See pandas scalars in pd.api.types.is_scalar
    cdef c_string val_cstr

    if isinstance(val, int):
        return move(make_const_int_expr(val))
    elif isinstance(val, float):
        return move(make_const_double_expr(val))
    elif isinstance(val, str):
        val_cstr = val.encode()
        return move(make_const_string_expr(val_cstr))
    elif isinstance(val, pd.Timestamp):
        # NOTE: Timestamp.value always converts to nanoseconds
        # https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/_libs/tslibs/timestamps.pyx#L242
        return move(make_const_timestamp_ns_expr(val.value))
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
    else:
        raise NotImplementedError("Unknown Python arith dunder method name")


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
        if opstr in ["__floordiv__", "__rfloordiv__"]:
            truediv_expression = make_arithop_expr(
                lhs_expr,
                rhs_expr,
                "/".encode())
            self.c_expression = make_unaryop_expr(truediv_expression, "floor".encode())
        else:
            duckdb_op = python_arith_dunder_to_duckdb(opstr)
            self.c_expression = make_arithop_expr(
                lhs_expr,
                rhs_expr,
                duckdb_op.encode())

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

    def __cinit__(self, object out_schema, object parquet_path, object storage_options):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalGet] c_logical_get = make_parquet_get_node(parquet_path, out_schema, storage_options)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
        self.path = parquet_path

    def __str__(self):
        return f"LogicalGetParquetRead({self.path})"

    def getCardinality(self):
        return pq.read_table(self.path, columns=[]).num_rows


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
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_seq_node(df, out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
        self.df = df

    def getCardinality(self):
        return len(self.df)


cdef class LogicalGetPandasReadParallel(LogicalOperator):
    cdef int nrows

    """Represents parallel scan of a Pandas dataframe passed into from_pandas."""
    def __cinit__(self, object out_schema, int nrows, object result_id):
        # result_id could be a string or LazyPlanDistributedArg if we are constructing the
        # plan locally for cardinality.  If so, extract res_id from that object.
        if not isinstance(result_id, str):
            result_id = result_id.res_id
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_parallel_node(result_id.encode(), out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
        self.nrows = nrows

    def getCardinality(self):
        return self.nrows

cdef class LogicalGetIcebergRead(LogicalOperator):
    """
    Wrapper around DuckDB's LogicalGet for reading Iceberg datasets.
    """
    cdef readonly str table_identifier

    def __cinit__(self, object out_schema, str table_identifier, object catalog_name, object catalog_properties, object iceberg_filter, object iceberg_schema):
        import pyiceberg.catalog
        cdef object catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)
        self.out_schema = out_schema
        self.table_identifier = table_identifier
        cdef unique_ptr[CLogicalGet] c_logical_get = make_iceberg_get_node(out_schema, table_identifier.encode(), catalog, iceberg_filter, iceberg_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())

    def __str__(self):
        return f"LogicalGetIcebergRead({self.table_identifier})"

cpdef count_nodes(object root):
    cdef LogicalOperator wrapped_operator

    if not isinstance(root, LogicalOperator):
        raise TypeError("Expected a LogicalOperator instance")

    wrapped_operator = root

    return planCountNodes(wrapped_operator.c_logical_operator)


cpdef set_cpp_table_meta(table_pointer, object arrow_schema):
    """Set the metadata of a C++ table from an Arrow schema.
    """
    cdef int64_t cpp_table = table_pointer
    set_table_meta_from_arrow(cpp_table, arrow_schema)


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
    arrow_schema = <object>exec_output.second
    if output_func is None:
        raise ValueError("output_func is None.")
    return output_func(cpp_table, out_schema)
