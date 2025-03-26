"""Provides wrappers around DuckDB's nodes and optimizer for use in Python.
"""
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.memory cimport unique_ptr, make_unique, dynamic_pointer_cast
from libcpp.utility cimport move
from libcpp.string cimport string as c_string


cdef extern from "duckdb/planner/logical_operator.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalOperator" duckdb::LogicalOperator":
        pass


cdef extern from "duckdb/common/enums/join_type.hpp" namespace "duckdb" nogil:
    cpdef enum class CJoinType" duckdb::JoinType":
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


cdef extern from "duckdb/planner/operator/logical_get.hpp" namespace "duckdb" nogil:
    cdef cppclass CLogicalGet" duckdb::LogicalGet"(CLogicalOperator):
        pass


cdef extern from "_bodo_plan.h" nogil:
    cdef unique_ptr[CLogicalGet] make_parquet_get_node(c_string parquet_path)
    cdef unique_ptr[CLogicalOperator] optimize_plan(unique_ptr[CLogicalOperator])


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

    def __str__(self):
        return "LogicalOperator()"


cdef class LogicalComparisonJoin(LogicalOperator):
    """Wrapper around DuckDB's LogicalComparisonJoin to provide access in Python.
    """

    def __cinit__(self, CJoinType join_type):
       cdef unique_ptr[CLogicalComparisonJoin] c_logical_comparison_join = make_unique[CLogicalComparisonJoin](join_type)
       self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalOperator*> c_logical_comparison_join.release())

    def __str__(self):
        return f"LogicalComparisonJoin({join_type_to_string(self.c_logical_comparison_join.get().join_type)})"


cdef class LogicalGetParquetRead(LogicalOperator):
    """Wrapper around DuckDB's LogicalGet for reading Parquet datasets.
    """
    cdef readonly str path

    def __cinit__(self, c_string parquet_path):
       cdef unique_ptr[CLogicalGet] c_logical_get = make_parquet_get_node(parquet_path)
       self.c_logical_operator = unique_ptr[CLogicalOperator](<CLogicalGet*> c_logical_get.release())
       self.path = (<bytes>parquet_path).decode("utf-8")

    def __str__(self):
        return f"LogicalGetParquetRead({self.path})"


cdef public py_optimize_plan(object plan):
    """Optimize a logical plan using DuckDB's optimizer
    """
    cdef LogicalOperator wrapped_operator

    if not isinstance(plan, LogicalOperator):
        raise TypeError("Expected a LogicalOperator instance")
    
    wrapped_operator = plan

    optimized_plan = LogicalOperator()
    optimized_plan.c_logical_operator = optimize_plan(move(wrapped_operator.c_logical_operator))
    return optimized_plan
