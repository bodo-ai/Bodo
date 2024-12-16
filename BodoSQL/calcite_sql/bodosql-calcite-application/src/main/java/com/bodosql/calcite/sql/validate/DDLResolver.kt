package com.bodosql.calcite.sql.validate

import com.bodosql.calcite.ddl.DDLExecutionResult
import org.apache.calcite.sql.SqlNode

/**
 * Validator that is used for DDL operations. Most DDL operations
 * require very little validation, but to ensure proper execution
 * do need information like fully qualifying identifiers.
 */
interface DDLResolver {
    /**
     * Executes a DDL operation.
     */
    fun executeDDL(node: SqlNode): DDLExecutionResult
}
