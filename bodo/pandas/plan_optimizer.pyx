# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.memory cimport unique_ptr, make_unique


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
    cdef cppclass CLogicalComparisonJoin" duckdb::LogicalComparisonJoin":
        CLogicalComparisonJoin(CJoinType join_type)


cdef class LogicalComparisonJoin:
    cdef:
        unique_ptr[CLogicalComparisonJoin] c_logical_comparison_join

    def __cinit__(self, CJoinType join_type):
       self.c_logical_comparison_join = make_unique[CLogicalComparisonJoin](join_type)
