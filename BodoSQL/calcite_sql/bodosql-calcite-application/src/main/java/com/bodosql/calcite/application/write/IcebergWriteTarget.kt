package com.bodosql.calcite.application.write

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.OperatorID
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType

/**
 * A WriteTarget implementation for writing to an Iceberg table.
 */
open class IcebergWriteTarget(
    tableName: String,
    schema: ImmutableList<String>,
    ifExistsBehavior: IfExistsBehavior,
    columnNamesGlobal: Variable,
    protected val icebergPath: Expr,
) : WriteTarget(tableName, schema, ifExistsBehavior, columnNamesGlobal) {
    protected val icebergConnectionString = pathToIcebergConnectionString(icebergPath)

    private fun pathToIcebergConnectionString(path: Expr): Expr =
        if (path is Expr.StringLiteral) Expr.StringLiteral("iceberg+" + path.arg) else path

    open fun allowsThetaSketches(): Boolean = true

    /**
     * Initialize the streaming create table state information for an Iceberg Table.
     * @param visitor The PandasCodeGenVisitor used to lower globals.
     * @param operatorID The operatorID used for tracking memory allocation.
     * @param createTableType The type of the create table operation. This is unused by Iceberg.
     * @param meta Expression containing the metadata information for init table information.
     * @return A code generation expression for initializing the table.
     */
    override fun streamingCreateTableInit(
        visitor: BodoCodeGenVisitor,
        operatorID: OperatorID,
        createTableType: CreateTableType,
        meta: CreateTableMetadata,
    ): Expr {
        val ctasMetaCall = meta.emitCtasExpr()
        val ctasMetaGlobal: Expr = visitor.lowerAsGlobal(ctasMetaCall)
        var args =
            listOf(
                operatorID.toExpr(),
                icebergConnectionString,
                Expr.StringLiteral(
                    if (schema.size == 0) tableName else schema.joinToString(separator = ".") + "." + tableName,
                ),
                columnNamesGlobal,
                Expr.StringLiteral(ifExistsBehavior.asToSqlKwArgument()),
                ctasMetaGlobal,
            )
        var kwargs =
            listOf(
                "allow_theta_sketches" to Expr.BooleanLiteral(allowsThetaSketches()),
            )
        return Expr.Call(
            "bodo.io.iceberg.stream_iceberg_write.iceberg_writer_init",
            args,
            kwargs,
        )
    }

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
     * Implement append to an Iceberg table.
     * @param visitor The PandasCodeGenVisitor used to lower globals. This is unused
     * by this implementation. (TODO: REMOVE)
     * @param stateVar The variable for the write state.
     * @param tableVar The variable for the current table chunk we want to write.
     * @param isLastVar The variable tracking if this is the last iteration.
     * @param iterVar The variable tracking what iteration we are on.
     * @param columnPrecisions Expression containing any column precisions for create
     * table information. This is unused by this implementation. TODO: Move to init.
     * @param meta Expression containing the metadata information for init table information.
     * This is unused by this implementation. TODO: Move to init.
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
        val args = listOf(stateVar, tableVar, columnNamesGlobal, isLastVar, iterVar)
        return Expr.Call("bodo.io.iceberg.stream_iceberg_write.iceberg_writer_append_table", args)
    }
}
