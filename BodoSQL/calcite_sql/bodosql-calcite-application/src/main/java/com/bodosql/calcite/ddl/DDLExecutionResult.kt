package com.bodosql.calcite.ddl

/**
 * Represents the result of a DDL operation. The maps to a columnar DataFrame result.
 * TODO: Determine if we need a type code for each column when we have a non-string result.
 */
data class DDLExecutionResult(val columnNames: List<String>, val columnValues: List<List<Any?>>)
