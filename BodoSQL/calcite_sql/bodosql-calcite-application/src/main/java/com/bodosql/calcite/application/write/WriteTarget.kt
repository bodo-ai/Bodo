package com.bodosql.calcite.application.write

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable

/**
 * Base abstract class for any write destination. This provides a standard
 * interface for a write output target based on the destination table type
 * (e.g. SnowflakeNativeTable, IcebergTable, SnowflakeIcebergTable, Parquet).
 *
 * Ideally in most cases we can fully separate the implementation of a destination
 * from its catalog aside from a few standard interfaces (e.g. connection string).
 * However, if an implementation requires a specialized workaround that will not
 * conform with a "standard" implementation, then a new class should be extended to
 * provide the workaround.
 */
abstract class WriteTarget(
    protected val tableName: String,
    protected val schema: ImmutableList<String>,
    protected val createTableType: SqlCreateTable.CreateTableType,
    protected val ifExistsBehavior: IfExistsBehavior,
    // TODO: Standardize to have all write targets use it init and move to init.
    protected val columnNamesGlobal: Variable,
) {
    /**
     * Initialize the streaming create table state information for a given write target.
     * @param operatorID The operatorID used for tracking memory allocation.
     * @return A code generation expression for initializing the table.
     */
    abstract fun streamingCreateTableInit(operatorID: Expr.IntegerLiteral): Expr

    /**
     * Implement append to a table for a given write target.
     * @param visitor The PandasCodeGenVisitor used to lower globals. (TODO: REMOVE)
     * @param stateVar The variable for the write state.
     * @param tableVar The variable for the current table chunk we want to write.
     * @param isLastVar The variable tracking if this is the last iteration.
     * @param iterVar The variable tracking what iteration we are on.
     * @param columnPrecisions Expression containing any column precisions for create
     * table information. TODO: Move to init.
     * @param meta Expression containing the metadata information for init table information.
     * TODO: Move to init.
     * @return The write expression call.
     *
     */
    abstract fun streamingWriteAppend(
        visitor: PandasCodeGenVisitor,
        stateVar: Variable,
        tableVar: Variable,
        isLastVar: Variable,
        iterVar: Variable,
        columnPrecisions: Expr,
        meta: SnowflakeCreateTableMetadata,
    ): Expr

    /**
     * Final step to mark a create table operation as done.
     * Most writes don't need to truly "finalize" anything yet,
     * so we default to a NoOp.
     * @return An operation that includes the finalization behavior.
     */
    open fun streamingCreateTableFinalize(): Op {
        return Op.NoOp
    }

    /** Enum describing write behavior when the table already exists.  */
    enum class IfExistsBehavior {
        REPLACE,
        APPEND,
        FAIL,
        ;

        fun asToSqlKwArgument(): String {
            return when (this) {
                REPLACE -> "replace"
                FAIL -> "fail"
                APPEND -> "append"
            }
            throw RuntimeException("Reached Unreachable code in toToSqlKwArgument")
        }
    }

    enum class WriteTargetEnum {
        PARQUET,
        ICEBERG,
        ;

        companion object {
            @JvmStatic
            fun fromString(strVal: String): WriteTargetEnum {
                return when (strVal) {
                    "parquet" -> {
                        PARQUET
                    }

                    "iceberg" -> {
                        ICEBERG
                    }

                    else -> {
                        throw java.lang.RuntimeException("Unsupported Write Target Enum")
                    }
                }
            }
        }
    }
}
