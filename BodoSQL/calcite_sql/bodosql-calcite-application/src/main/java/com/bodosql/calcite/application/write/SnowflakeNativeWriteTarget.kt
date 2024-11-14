package com.bodosql.calcite.application.write

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.OperatorID
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType

/**
 * A WriteTarget implementation for writing to a Snowflake
 * Native Table.
 */
class SnowflakeNativeWriteTarget(
    tableName: String,
    schema: ImmutableList<String>,
    ifExistsBehavior: IfExistsBehavior,
    columnNamesGlobal: Variable,
    private val connectionString: Expr,
) : WriteTarget(tableName, schema, ifExistsBehavior, columnNamesGlobal) {
    /**
     * Initialize the streaming create table state information for a Snowflake Native Table.
     * @param visitor The PandasCodeGenVisitor used to lower globals.
     * @param operatorID The operatorID used for tracking memory allocation.
     * @param createTableType The type of the create table operation. This is unused by parquet.
     * @param meta Expression containing the metadata information for init table information.
     * @return A code generation expression for initializing the table.
     */
    override fun streamingCreateTableInit(
        visitor: BodoCodeGenVisitor,
        operatorID: OperatorID,
        createTableType: CreateTableType,
        meta: CreateTableMetadata,
    ): Expr =
        Expr.Call(
            "bodo.io.snowflake_write.snowflake_writer_init",
            operatorID.toExpr(),
            connectionString,
            Expr.StringLiteral(tableName),
            // TODO: Can we remove? This is already in the connection string.
            Expr.StringLiteral(schema[1]),
            Expr.StringLiteral(ifExistsBehavior.asToSqlKwArgument()),
            Expr.StringLiteral(createTableType.asStringKeyword()),
        )

    /**
     * Initialize the streaming insert into state information for a given write target.
     * @param visitor The PandasCodeGenVisitor used to lower globals.
     * @param operatorID The operatorID used for tracking memory allocation.
     * @return A code generation expression for initializing the insert into.
     */
    override fun streamingInsertIntoInit(
        visitor: BodoCodeGenVisitor,
        operatorID: OperatorID,
    ): Expr = streamingCreateTableInit(visitor, operatorID, CreateTableType.DEFAULT, CreateTableMetadata())

    /**
     * Implement append for a Snowflake native table.
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
    override fun streamingWriteAppend(
        visitor: BodoCodeGenVisitor,
        stateVar: Variable,
        tableVar: Variable,
        isLastVar: Variable,
        iterVar: Variable,
        columnPrecisions: Expr,
        meta: CreateTableMetadata,
    ): Expr {
        val ctasMetaCall = meta.emitCtasExpr()
        val ctasMetaGlobal: Expr = visitor.lowerAsGlobal(ctasMetaCall)
        val args = listOf(stateVar, tableVar, columnNamesGlobal, isLastVar, iterVar, columnPrecisions, ctasMetaGlobal)
        return Expr.Call("bodo.io.snowflake_write.snowflake_writer_append_table", args)
    }
}
