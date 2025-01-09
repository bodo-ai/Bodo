package com.bodosql.calcite.application.write

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Variable
import com.google.common.collect.ImmutableList

/**
 * A WriteTarget implementation for writing to a Snowflake
 * Iceberg Table. This is largely reusing the IcebergWriteTarget,
 * but since Snowflake doesn't support native write, we must undergo
 * additional steps to convert the Iceberg table to a Snowflake
 * managed table, which deviates from standard Iceberg support.
 */
class SnowflakeIcebergWriteTarget(
    tableName: String,
    schema: ImmutableList<String>,
    ifExistsBehavior: IfExistsBehavior,
    columnNamesGlobal: Variable,
    icebergPath: Expr,
    private val icebergVolume: String,
    private val snowflakeConnectionString: Expr,
    uuid: String,
) : IcebergWriteTarget(
        tableName,
        // Note: Conceptually for Iceberg we add extra levels of indirection to ensure we don't conflict with
        // the other files in the Iceberg volume. This is hardcoded during codegen testing.
        ImmutableList.of("bodo_write_temp", schema[0], schema[1], tableName + "_" + uuid),
        ifExistsBehavior,
        columnNamesGlobal,
        icebergPath,
    ) {
    /**
     * Final step to mark a create table operation as done.
     * Since Snowflake doesn't support native write we need to convert
     * the table to managed Iceberg table.
     * @return A stmt that contains a call to convert the table.
     */
    override fun streamingCreateTableFinalize(): Op.Stmt {
        val expr =
            Expr.Call(
                "bodo.io.iceberg.stream_iceberg_write.convert_to_snowflake_iceberg_table",
                snowflakeConnectionString,
                icebergConnectionString,
                Expr.StringLiteral(schema.joinToString(separator = "/")),
                // Note: This is the name of the volume, not the path or connection string.
                Expr.StringLiteral(icebergVolume),
                Expr.StringLiteral(tableName),
                Expr.BooleanLiteral(ifExistsBehavior == IfExistsBehavior.REPLACE),
            )
        return Op.Stmt(expr)
    }

    override fun allowsThetaSketches(): Boolean = false
}
