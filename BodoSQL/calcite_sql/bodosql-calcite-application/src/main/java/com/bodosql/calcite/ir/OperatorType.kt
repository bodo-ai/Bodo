package com.bodosql.calcite.ir

/**
 * @brief All supported streaming operator types. This must match the values in bodo/libs/memory_budget.py
 */
enum class OperatorType {
    UNKNOWN, SNOWFLAKE_WRITE, SNOWFLAKE_READ, JOIN, GROUPBY, UNION, ACCUMULATE_TABLE, ENCODE_DICT
}
