"""Provides wrappers around DuckDB's nodes and optimizer for use in Python.
"""
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.memory cimport unique_ptr, make_unique, dynamic_pointer_cast
from libcpp.utility cimport move, pair
from libcpp.string cimport string as c_string
from libcpp.vector cimport vector
import operator
from libc.stdint cimport int64_t

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
    if val is operator.gt:
        return CExpressionType.COMPARE_GREATERTHAN
    else:
        assert False

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

cdef extern from "duckdb/planner/operator/logical_filter.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalFilter" duckdb::LogicalFilter"(CLogicalOperator):
        pass

cdef extern from "duckdb/planner/operator/logical_get.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalGet" duckdb::LogicalGet"(CLogicalOperator):
        pass


cdef extern from "_plan.h" nogil:
    cdef unique_ptr[CLogicalGet] make_parquet_get_node(c_string parquet_path, object arrow_schema, object storage_options)
    cdef unique_ptr[CLogicalGet] make_dataframe_get_seq_node(object df, object arrow_schema)
    cdef unique_ptr[CLogicalGet] make_dataframe_get_parallel_node(c_string res_id, object arrow_schema)
    cdef unique_ptr[CLogicalComparisonJoin] make_comparison_join(unique_ptr[CLogicalOperator] lhs, unique_ptr[CLogicalOperator] rhs, CJoinType join_type, vector[int_pair] cond_vec)
    cdef unique_ptr[CLogicalOperator] optimize_plan(unique_ptr[CLogicalOperator])
    cdef unique_ptr[CLogicalProjection] make_projection(unique_ptr[CLogicalOperator] source, vector[int] select_vec, object out_schema)
    cdef unique_ptr[CLogicalProjection] make_projection_python_scalar_func(unique_ptr[CLogicalOperator] source, object out_schema, object args)
    cdef unique_ptr[CExpression] make_binop_expr(unique_ptr[CExpression] lhs, unique_ptr[CExpression] rhs, CExpressionType etype)
    cdef unique_ptr[CLogicalFilter] make_filter(unique_ptr[CLogicalOperator] source, unique_ptr[CExpression] filter_expr)
    cdef unique_ptr[CExpression] make_const_int_expr(int val)
    cdef unique_ptr[CExpression] make_col_ref_expr(unique_ptr[CLogicalOperator] source, object field, int col_idx)
    cdef pair[int64_t, PyObjectPtr] execute_plan(unique_ptr[CLogicalOperator], object out_schema)
    cdef c_string plan_to_string(unique_ptr[CLogicalOperator])
    cdef vector[int] get_projection_pushed_down_columns(unique_ptr[CLogicalOperator] proj)


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
        raise ValueError("Unknown Join Type")


cdef class LogicalOperator:
    """Wrapper around DuckDB's LogicalOperator to provide access in Python.
    """
    cdef unique_ptr[CLogicalOperator] c_logical_operator
    cdef readonly out_schema

    def __str__(self):
        return "LogicalOperator()"

    def set_estimated_cardinality(self, estimated_cardinality):
        self.c_logical_operator.get().has_estimated_cardinality = True
        self.c_logical_operator.get().estimated_cardinality = estimated_cardinality

    def toGraphviz(self):
        return plan_to_string(self.c_logical_operator).decode("utf-8")

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

    cdef readonly vector[int] select_vec

    def __cinit__(self, object out_schema, LogicalOperator source, select_idxs):
        self.out_schema = out_schema
        self.select_vec = select_idxs

        cdef unique_ptr[CLogicalProjection] c_logical_projection = make_projection(source.c_logical_operator, self.select_vec, self.out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_projection.release())

    def __str__(self):
        return f"LogicalProjection({self.select_vec}, {self.out_schema})"


cpdef get_pushed_down_columns(proj):
    """Get column indices that are pushed down from projection to its source node. Used for testing.
    """
    cdef LogicalOperator wrapped_operator = proj
    cdef vector[int] pushed_down_columns = get_projection_pushed_down_columns(wrapped_operator.c_logical_operator)
    return pushed_down_columns


cdef class LogicalProjectionPythonScalarFunc(LogicalOperator):
    """Wrapper around DuckDB's LogicalProjection with a ScalarFunc inside to provide access in Python.
    """

    def __cinit__(self, object out_schema, LogicalOperator source, object args):
        self.out_schema = out_schema

        cdef unique_ptr[CLogicalProjection] c_logical_projection = make_projection_python_scalar_func(source.c_logical_operator, self.out_schema, args)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_projection.release())

    def __str__(self):
        return f"LogicalProjectionPythonScalarFunc({self.out_schema})"


cdef unique_ptr[CExpression] make_expr(val):
    cdef LogicalOperator source

    if isinstance(val, int):
        return make_const_int_expr(val)
    elif isinstance(val, LogicalProjection):
        select_vec = val.select_vec
        field = val.out_schema.field(0)
        assert len(select_vec) == 1
        source = val
        return make_col_ref_expr(source.c_logical_operator, field, select_vec[0])
    else:
        assert False

def get_source(val):
    if isinstance(val, int):
        return None
    elif isinstance(val, LogicalProjection):
        return val
    else:
        assert False

cdef class LogicalFilter(LogicalOperator):
    def __cinit__(self, out_schema, LogicalOperator source, key):
       self.out_schema = out_schema

       cdef unique_ptr[CExpression] c_filter_expr
       if isinstance(key, LogicalBinaryOp):
            lhs_expr = make_expr(key.lhs)
            rhs_expr = make_expr(key.rhs)
            lhs_source = get_source(key.lhs)
            rhs_source = get_source(key.rhs)
            c_filter_expr = make_binop_expr(lhs_expr, rhs_expr, str_to_expr_type(key.binop))
            if lhs_source is not None:
                source = lhs_source
            elif rhs_source is not None:
                source = rhs_source
       else:
            assert False & "Unimplemented"

       cdef unique_ptr[CLogicalFilter] c_logical_filter = make_filter(source.c_logical_operator, c_filter_expr)
       self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_filter.release())

    def __str__(self):
        return f"LogicalFilter()"

class LogicalBinaryOp(LogicalOperator):
    def __init__(self, out_schema, lhs, rhs, binop):
        self.lhs = lhs
        self.rhs = rhs
        self.binop = binop

cdef class LogicalGetParquetRead(LogicalOperator):
    """Wrapper around DuckDB's LogicalGet for reading Parquet datasets.
    """
    cdef readonly str path

    def __cinit__(self, object out_schema, c_string parquet_path, object storage_options):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalGet] c_logical_get = make_parquet_get_node(parquet_path, out_schema, storage_options)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
        self.path = (<bytes>parquet_path).decode("utf-8")

    def __str__(self):
        return f"LogicalGetParquetRead({self.path})"


cdef class LogicalGetSeriesRead(LogicalOperator):
    """Represents an already materialized BodoSeries."""
    def __cinit__(self, out_schema, result_id):
        self.out_schema = out_schema
        assert False & "Not implemented yet."


cdef class LogicalGetDataframeRead(LogicalOperator):
    """Represents an already materialized BodoDataFrame."""
    def __cinit__(self, out_schema, result_id):
        self.out_schema = out_schema
        assert False & "Not implemented yet."


cdef class LogicalGetPandasReadSeq(LogicalOperator):
    """Represents sequential scan of a Pandas dataframe passed into from_pandas."""
    cdef readonly object df

    def __cinit__(self, object out_schema, object df):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_seq_node(df, out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
        self.df = df


cdef class LogicalGetPandasReadParallel(LogicalOperator):
    """Represents parallel scan of a Pandas dataframe passed into from_pandas."""
    def __cinit__(self, object out_schema, str result_id):
        self.out_schema = out_schema
        cdef unique_ptr[CLogicalGet] c_logical_get = make_dataframe_get_parallel_node(result_id.encode(), out_schema)
        self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())


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
    assert output_func is not None
    return output_func(cpp_table, out_schema)
