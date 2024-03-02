package com.bodosql.calcite.application.write

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable

class ParquetWriteTarget(
    tableName: String,
    schema: ImmutableList<String>,
    createTableType: SqlCreateTable.CreateTableType,
    ifExistsBehavior: IfExistsBehavior,
    columnNamesGlobal: Variable,
    // Note: This should be the full path, including the table name.
    private val parquetPath: String,
) : WriteTarget(tableName, schema, createTableType, ifExistsBehavior, columnNamesGlobal) {
    /**
     * Initialize the streaming create table state information for a Parquet Table.
     * @param operatorID The operatorID used for tracking memory allocation.
     * @return A code generation expression for initializing the table.
     */
    override fun streamingCreateTableInit(operatorID: Expr.IntegerLiteral): Expr {
        return Expr.Call(
            "bodo.io.stream_parquet_write.parquet_writer_init",
            operatorID,
            Expr.StringLiteral(parquetPath),
            // We don't enable users to specify compression yet.
            Expr.StringLiteral("snappy"),
            // We don't enable user to specify row group size yet.
            Expr.IntegerLiteral(-1),
            Expr.StringLiteral("part-"),
            // We don't pass timezone information yet.
            Expr.StringLiteral(""),
        )
    }

    /**
     * Implement append to a ParquetTable.
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
        visitor: PandasCodeGenVisitor,
        stateVar: Variable,
        tableVar: Variable,
        isLastVar: Variable,
        iterVar: Variable,
        columnPrecisions: Expr,
        meta: SnowflakeCreateTableMetadata,
    ): Expr {
        val args = listOf(stateVar, tableVar, columnNamesGlobal, isLastVar, iterVar)
        return Expr.Call("bodo.io.stream_parquet_write.parquet_writer_append_table", args)
    }
}
